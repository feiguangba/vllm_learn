# 第 4 章 · ModelRunner 与 CUDA 加速（ch04）参考答案

> 参考答案，建议先自己动手再看。
>
> 对应笔记本路径（7 本，均在 `exercises/ch04/`）：
> - `21_metadata_assembly.ipynb`、`22_modelrunner_dataflow.ipynb`、`23_kv_allocator.ipynb`、`24_cuda_graph_principle.ipynb`、`25_cudagraph_capture.ipynb`、`26_torch_compile.ipynb`、`27_profiling_metrics.ipynb`
>
> 下列代码复用各 notebook 已定义的真实符号（组装函数与 block_size/seq_lens/block_table/slot_mapping、MiniGPT、trace_forward、BlockAllocator、OpCounter、launch 开销 t_l 等）。涉及 GPU/编译/CUDA profiler 的练习本机（无 GPU/MSVC）只给骨架与思路，在有 GPU/Linux/CUDA 的环境运行验证。

## 第 21 课 · 元数据组装：一维流 + 恢复现场（metadata_assembly）

### 练习 1 · 把 block_size 从 4 改成 16，重跑组装，观察 slot_mapping 与 block_table 怎么变

一句话思路：block_size 越大，每个逻辑块能容纳的 slot 越多，块表越短（块数变少），而 slot = 块号×block_size + 块内偏移的映射公式不变。

```python
for bs in [4, 16]:
    block_size = bs
    input_ids, positions, slot_mapping, block_table, seq_lens = assemble(reqs)
    print(f"block_size={bs}: block_table={block_table.tolist()}")
    print(f"                 slot_mapping={slot_mapping.tolist()}")
    # slot 校验: slot == block_table[req, pos//bs]*bs + pos%bs
```

预期：block_size=16 时每请求块数减为 block_size=4 的 1/4，block_table 行变短；slot_mapping 数值随块号/偏移变化但始终满足 `slot = 块号×block_size + 偏移` 这一硬性公式。

### 练习 2 · 加入一条 0 长度的空序列，验证组装函数是否还能正确处理（边界情况）

一句话思路：空序列长度为 0、不占任何 slot，组装函数应跳过它或只登记长度 0 且 slot_mapping 不为其写任何位置，不导致索引错乱。

```python
reqs2 = reqs + [{"prompt_len": 0, "max_new": 8}]   # 追加一条空序列
input_ids, positions, slot_mapping, block_table, seq_lens = assemble(reqs2)
print("seq_lens:", seq_lens)
print("slot_mapping 长度:", len(slot_mapping))
```

预期：组装不报错；seq_lens 中出现 0，空序列不占用任何 token 流位置，其后的 slot_mapping 仍按真实长度正确对齐（不因空序列错位）。

### 练习 3 · 把 slot_mapping 换成「跨层」版本（每层一个槽位偏移），模拟多层的真实场景

一句话思路：真实模型每层有独立的 KV 缓存，物理 slot 需按层分块：`slot_layer = layer*blocks_per_layer*block_size + slot`，把 slot_mapping 从「单层平铺」扩展成「每层加一个层偏移」。

```python
num_layers, block_size = 4, 4
for layer in range(num_layers):
    layer_slot = layer * (num_layers * block_size) + slot_mapping
    # 示例: 逐层整体加一个常量层偏移, 对应物理缓存按 [层][块][偏移] 排布
    print(f"layer {layer}: slot 最小/最大 = {layer_slot.min()} / {layer_slot.max()}")
```

预期：跨层 slot 在单层基础上叠加 `layer × 每层容量` 的偏移，各层互不重叠；保证多层的 K/V 写入落在各自层的物理区间。

### 练习 4 · 用第 5 节公式，画 padding 浪费 vs 序列数（1~256）的曲线，找到浪费的拐点

一句话思路：padding 浪费 = `1 - T/(S×max_len)`，序列数越多、总 token 与最大长度拉的差距越大，浪费越高并在某处出现拐点。

```python
import numpy as np, matplotlib.pyplot as plt
def pad_waste(n, S=32, max_len=1024):
    T = n * S
    return 1 - T / (n * max_len)          # 未对齐部分的占比
ns = np.arange(1, 257)
ws = np.array([pad_waste(n) for n in ns])
plt.plot(ns, ws); plt.xlabel("序列数 n"); plt.ylabel("padding 浪费率"); plt.show()
print(f"拐点附近(n<16)浪费快速上升, n>128 后趋于 {pad_waste(256):.2f}")
```

预期：序列数越少，max_len 占位越大、padding 浪费越高；随 n 增大浪费下降并趋饱和（真实场景 batch 不大时 padding 浪费显著，是连续批/桶对齐要解决的问题）。

---

## 第 22 课 · ModelRunner 数据流：九段前向（modelrunner_dataflow）

### 练习 1 · 把 MiniGPT 改成 4 层，重跑第 3/5 节，观察中间张量与显存账本如何翻倍

一句话思路：每层都含一套 Attn+FFN，把 layers 从 2 改 4 后逐层中间张量个数/参数量与显存账本近似翻倍，shape 契约不变。

```python
m4 = MiniGPT(num_layers=4, vocab=..., hidden=..., ...)
# 第3节: 统计参数量
n_params = sum(p.numel() for p in m4.parameters())
print(f"4层 MiniGPT 参数量 = {n_params/1e6:.2f} M")   # 与 2 层 version 对比约翻倍
# 第5节: prefill B×S 与 decode T=B 的显存账本各元素数翻倍
```

预期：参数量、trace_forward 打印的中间张量数、[T,4H] FFN 中间量等显存账本都随层数翻倍（约 2×），第 5 节 prefill/decode 的耗时/内存区分被放大。

### 练习 2 · 在 trace_forward 里插入 dropout，确认推理时被关闭（与训练前向的差异）

一句话思路：dropout 只应在训练开（`self.training=True`）、推理关（`eval()` 时 training=False）；插入 `self.drop(p=0.1)` 后在 eval 模式应恒等输出。

```python
m = MiniGPT(...).eval()          # 推理模式
x = torch.randn(1, 8, m.hidden)
y1 = m.forward_with(insert_dropout=False, x)   # 常规
y2 = m.forward_with(insert_dropout=True,  x)   # 插入 dropout
print("eval 下带 dropout 输出与不带一致?", torch.allclose(y1, y2))  # True
```

预期：`eval()` 下 dropout 层不激活（identity），加入与否输出完全一致；训练模式（training=True）下才随机置零，体现「推理必须关闭 dropout」这一差异。

### 练习 3 · 给 decode 加上 KV cache（参考 ch02/07 课），把 T=B 的 decode 前向改成「只算新 token」

一句话思路：复用 ch02 的 AttentionWithCache 思想，decode 时只对当前 B 个新 token 做 QKV 投影并追加写入缓存，注意力只和缓存中新旧 token 做，省去重复计算历史。

```python
# 伪代码: 把 trace_forward 的 decode 分支换成
def forward_decode_with_kvcache(self, x, k_cache, v_cache, cache_len):
    q = self.wq(x); k = self.wk(x); v = self.wv(x)
    B, T, _ = x.shape
    # 只把新增 T=B 个 token 写进缓存
    k_cache[:, :, cache_len:cache_len+T] = k
    v_cache[:, :, cache_len:cache_len+T] = v
    # 注意力只读缓存全部 token, 计算量只随 T 增长
    return attention(q, k_cache[:, :, :cache_len+T], v_cache[:, :, :cache_len+T])
```

预期：decode 只前向 O(T=B) 的新 token，历史 K/V 从缓存读取，不再每步重算整段 prompt；这是 prefill/decode 分离与 KV Cache 复用进入 ModelRunner 的第一步。

### 练习 4 · 用 torch.profiler 对 MiniGPT 做一次剖析，找出耗时前三的算子（第 27 课预告）

一句话思路：用 `torch.profiler.profile` 包住一次前向，按 `cpu_time_total` 排序取前 3，通常命中最重的矩阵乘（Linear / bmm / softmax）。

```python
import torch, torch.profiler as prof
x = torch.randn(1, 64, m.hidden)
with prof.profile(activities=[prof.ProfilerActivity.CUDA, prof.ProfilerActivity.CPU]) as p:
    m(x)
filtered = [e for e in p.key_averages() if e.self_cpu_time_total > 0]
top3 = sorted(filtered, key=lambda e: e.cpu_time_total, reverse=True)[:3]
for e in top3:
    print(f"{e.key}: {e.cpu_time_total:.2f} ms")
```

预期：耗时前三常为与 hidden 维相关的矩阵乘算子（Linear/mm/bmm），量化哪个算子吃掉最多时间，为后续 cuda graph/torch.compile 的融合候选提供依据。

---

## 第 23 课 · KV Allocator 与显存账本（kv_allocator）

### 练习 1 · 给 BlockAllocator 加「空闲块数不足时先压缩空洞」的逻辑，对比碎片率

一句话思路：在分配失败或碎片率高时，把所有存活块搬移到头端连续拼接（compact），消除外部碎片后再分配；代价是一次批量搬移。

```python
class CompactBlockAlloc(BlockAllocator):
    def compact(self):
        # 把 alive 块按顺序重排到连续编号, 更新所有引用者的块表
        order = sorted(self.alive)              # 按当前物理块号
        remap = {}
        for new_pb, old_pb in enumerate(order):
            remap[old_pb] = new_pb
            self.blocks[new_pb] = self.blocks.pop(old_pb)
        for owner in self.refs:
            self.refs[owner] = [remap.get(b, b) for b in self.refs[owner]]
        return len(order)                        # 紧凑后占用的物理块数

# 对比: 不 compact 时碎片率高, 每次先 compact 后碎片率归零
print("compact 后外部碎片:", round(alloc.ext_frag(), 3), "(≈0)")
```

预期：compact 后 external fragmentation 归零、后续大请求可被满足；代价是每次整理要移动所有存活块并更新引用者的块表（O(存活请求数)）。碎片率低于不 compact 版本。

### 练习 2 · 实现共享的释放语义：两个序列共享一块，一个释放后 refcount 从 2 变 1，块不回收

一句话思路：release 时只把该块 refcount 减一，减到 0 才回 free list；两个序列共享一块时一个释放后 ref 从 2→1，块仍被另一序列持有。

```python
alloc = NewBlockAllocator(...)
b = alloc.share_block(seqA, seqB)      # 两序列共享同一物理块 b
print("共享后 refcount:", alloc.refcount[b])      # 2
alloc.release(seqA)                    # A 释放, 只减计数
print("A 释放后 refcount:", alloc.refcount[b])    # 1 (块未回收)
print("块 b 仍在用?", b in alloc.alive)            # True
```

预期：A 释放后 refcount 从 2 变 1，物理块仍存活（B 继续占用）；只有第二个序列也释放（ref 归 0）块才回收进 free list。这就是共享前缀块「一个人退租不等于下架」的语义。

### 练习 3 · 把压力测试的分配概率从 0.6 改成 0.9，观察碎片率是否恶化（高负载更碎）

一句话思路：分配概率越高（0.9），请求更频繁地分配/释放，空洞被挖得更多更碎；对比 internal/external 碎片在 0.6 vs 0.9 下的差异。

```python
for p_alloc in [0.6, 0.9]:
    frag, internal, rejected = stress_test(alloc, prob_alloc=p_alloc, steps=1000)
    print(f"分配概率 {p_alloc}: external={frag:.3f} internal={internal:.3f} 拒绝={rejected}")
```

预期：分配概率 0.9 时 external fragmentation（碎片率）明显高于 0.6，拒绝次数可能上升；internal（块内浪费）两种概率相同（由 block_size 决定），说明高负载主要恶化的是外部碎片与体验。

### 练习 4 · 用第 8 节的 per_tok 公式，计算 block_size=16/32/64 时「一块」的显存，画对比表

一句话思路：每块显存 = `per_tok × block_size`，其中 per_tok = 2×L×H_kv×D×dtype_bytes；三档 block_size 各自乘 block_size 得单块字节并对比。

```python
L, H_kv, D, dt = 80, 8, 128, 2          # 例: fp16, GQA-8
per_tok = 2 * L * H_kv * D * dt
print(f"每 token = {per_tok} B")
for bs in [16, 32, 64]:
    one_block = per_tok * bs
    print(f"block_size={bs:2d}: 一块 = {one_block/2**10:7.1f} KiB = {one_block/2**20:.3f} MiB")
```

预期：per_tok 固定（由 L、H_kv、D、dtype 决定），单块显存随 block_size 线性增长：block_size=16/32/64 分别约为 miB/kib 递增（例中每块 16→32→64 KiB 量级）。这个「一块」的显存正是 num_gpu_blocks 的分母。

---

## 第 24 课 · CUDA Graph 原理：launch 开销与捕获（cuda_graph_principle）

### 练习 1 · 把 GPU 微基准的 N 从 200 改成 50/500/2000，重测 launch 占比，验证「占比与 n 无关」

一句话思路：launch 占比 = `t_l/(t_l+t_e)`，只由单 kernel 大小决定；重复 N 次只改变能省下的总手续费 (n-1)×t_l，占比基本不变。

```python
def launch_share(N, size=64):
    t_l, t_e = measure_launch(N, size)      # 返回单次 launch/fixed 与执行时间
    share = t_l / (t_l + t_e)
    saved = (N - 1) * t_l
    return share, saved
for N in [50, 500, 2000]:
    share, saved = launch_share(N)
    print(f"N={N:5d}: launch 占比={share:.3f}  (省手续费 {saved:.3f} ms)")
```

预期：三档 N 的 launch 占比基本一致（≈同一常数，只由日起 kernel 大小决定），但省下的总手续费随 N 线性增长——「占比与 n 无关、省量随 n 线性」。

### 练习 2 · 把单个小 matmul 尺寸从 64 放大到 512，看 t_e 变大后占比是否下降

一句话思路：kernel 越大 t_e 越大、t_l 不变，`t_l/(t_l+t_e)` 分母变大，占比下降——大 kernel 的启动手续费被摊薄。

```python
for size in [64, 128, 256, 512]:
    t_l, t_e = measure_launch(N=200, matmul_size=size)
    print(f"matmul {size:3d}: t_l={t_l:.2f}ms t_e={t_e:.2f}ms 占比={t_l/(t_l+t_e):.3f}")
```

预期：随 matmul 尺寸从 64→512 增大，t_e 明显上升而 t_l 基本不变，launch 占比从高值（如 0.5+）下降到低值——说明小 kernel 手续费过半、大 kernel 手续费被摊薄。这解释了为什么 decode 的小 kernel 栈特别适合 CUDA Graph。

### 练习 3 · 用 CPU 类比换不同工作负载（如 dict 操作），确认「调用次数才是关键」这一普适性

一句话思路：CPU 上一次函数调用也有固定开销；把同样的加法换成 dict 操作，只要保持「多次小调用 vs 合并一次」的对比，加速比仍主要取决于调用次数差。

```python
import timeit
def repeat_calls(n, op):
    for _ in range(n): op()          # N 次小调用
def single_call(op_hat):
    op_hat()                         # 一次合并调用
d = {}
t_rep = timeit.timeit(lambda: repeat_calls(1000, lambda: d.setdefault(1, 0)), number=100)
t_merge = timeit.timeit(lambda: single_call(lambda: d.update({1: 0})), number=100)
print(f"1000 次小调用 {t_rep:.3f}s vs 合并 {t_merge:.3f}s, 加速 {t_rep/t_merge:.1f}×")
```

预期：即使换成交互式 dict 操作，多次小调用仍显著慢于合并的一次调用（Python 函数调用/解释开销就是固定手续费），验证「调用次数才是关键」对任意工作负载都成立。

### 练习 4 · 读一遍 CUDA Programming Guide 的 CUDA Graphs 章节，找出 stream capture 的三种模式

一句话思路：CUDA stream capture 把一段流中的 kernel 录进图，有三种捕获模式由 `cudaStreamCaptureMode` 控制。

答案要点：三种模式为 `cudaStreamCaptureModeGlobal`（全局捕获，所有在捕获线程中合法开启的流都会被捕获）、`cudaStreamCaptureModeThreadLocal`（线程局部捕获，只捕获当前线程的捕获流）、`cudaStreamCaptureModeRelaxed`（放宽模式，不强制跨线程侵入，允许捕获流之间更宽松的同步）。日常用 ThreadLocal/Relaxed 居多，Global 更严格。

---

## 第 25 课 · CUDA Graph 捕获（cudagraph_capture）

### 练习 1 · 把 OpCounter 换成记录每个算子名，打印前 10 个热点算子

一句话思路：OpCounter 用 TorchDispatchMode 拦截算子调用，额外记录 `.op_type` 并按出现次数排序，取前 10 个即看到小算子栈的高频成员。

```python
class OpCounterNames(OpCounter):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func.__name__)          # 记录算子名
        return func(*args, **(kwargs or {}))
counter = OpCounterNames()
with counter:
    out = model(inputs)
from collections import Counter
hot = Counter(counter.ops).most_common(10)
for name, cnt in hot: print(f"{name}: {cnt}")
```

预期：打印出 MiniGPT 一次前向里调用次数最多的 10 个算子（如 addmm / bmm / softmax / layer_norm 等），可量化的看到哪些算子反复被调用，构成后续 CUDA Graph 可以合并的候选。

### 练习 2 · 把 recorded_forward 改成真正的 TorchDynamo 图导出，对比算子数

一句话思路：ToryDynamo 把 eager Python 解释器按算子一层层跑压缩成一张 FX 图，导出后整体是一次（或极少量）图执行，算子「数」与实际触发的底层 kernel 不同。

```python
import torch
xfx = torch.compile(model, mode="trace")        # 交给 dynamo 追踪/导出图
with torch.no_grad():
    out = xfx(inputs)
print(torch._dynamo.export(model)(inputs))
```

预期：Dynamo 把 eager 里几十个 Python 级算子调用编译成一张图，图导出后的算子集合更少、更融合；与 OpCounter 数的 eager 算子数对比，能看出「编译把多次算子折叠成少量大操作」的量级差距。

### 练习 3 · 用 torch.cuda.graph 在 Linux/有 GPU 的机器上实测 B=1/8/32 三档的加速比

一句话思路：按「侧流预热 + with torch.cuda.graph 捕获 + replay」的标准流程，对 B=1/8/32 三档各做 eager vs cuda graph 的计时，求加速比。

```python
# 需 GPU/Linux 环境。骨架:
def bench_graph(B):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):                # 预热
        out = model(inputs)
    s.synchronize()
    with torch.cuda.graph(g):
        out = model(inputs)                   # 捕获
    t0 = perf_counter()
    g.replay()                                # 重放
    return (perf_counter() - t0) * 1e3
for B in [1, 8, 32]:
    eager_ms, graph_ms = bench_eager(B), bench_graph(B)
    print(f"B={B}: eager {eager_ms:.2f}ms vs graph {graph_ms:.2f}ms, 加速 {eager_ms/graph_ms:.2f}×")
```

预期：B 越小（kernel 越小）加速比越大（省下的 t_l 占大头）；B 越大 kernel 本身耗时越长，加速比下降——验证 CUDA Graph 收益集中在解码阶段的小 kernel 栈（如 B=1 时加速 1.3~2×，B=32 时接近 1×）。

### 练习 4 · 把 capture sizes 改成等距 [1,2,...,128]，比较最坏浪费与预录成本

一句话思路：vLLM 按桶预录多张图，实际 B 向上取整到最近桶；等距 [1..128] 比幂次稀疏桶覆盖更密，最坏 pad 浪费更小，但预录的图张数更多、成本更高。

```python
sizes = list(range(1, 129))                    # 等距 1..128
n_cap = len(sizes)
# 某实际 batch B 向上取整到最近桶的浪费
def worst_pad(sizes, B, step=128):
    lo = sizes[0]
    while lo < B and lo+step < B: lo += step
    hi = min(lo + step, 128)
    return (hi - B) / B                        # 相对的 pad 浪费 (大致)
worst = max(worst_pad(sizes, b) for b in range(1, 129))
print(f"等距[1..128]共 {n_cap} 张图, 最坏浪费 ≈ {worst:.2f}")
print("对比 vLLM 幂次桶(稀疏): 图张数少但最坏 pad 浪费更大")
```

预期：等距 [1..128] 把覆盖撑满、最坏 padding 浪费封顶更小（每次实际 B 就近取整），但需预录 128 张图、捕获成本高；vLLM 用稀疏幂次桶换取少录图、接受略高的最坏 pad——「最坏浪费 vs 预录成本」的权衡。

---

## 第 26 课 · torch.compile：融合与图编译（torch_compile）

### 练习 1 · 在 Linux/GPU 机器上跑 Inductor，对比 compiled vs eager 的真实加速比

一句话思路：用 `torch.compile(model, backend="inductor")` 编译后与 eager 各跑一次 warmup + 计时，加速比即 eager_ms/compiled_ms（论文报推理约 2.27×）。

```python
# 需 GPU/Linux (本机 Windows 无 MSVC, Inductor 失败, 用 aot_eager 可过)。骨架:
model_comp = torch.compile(model, backend="inductor")
for _ in range(3): model_comp(inputs)          # warm up
t_comp = timeit(lambda: model_comp(inputs), number=50) / 50
t_eager = timeit(lambda: model(inputs), number=50) / 50
print(f"eager {t_eager*1e3:.3f} ms vs inductor {t_comp*1e3:.3f} ms, 加速 {t_eager/t_comp:.2f}×")
```

预期：Inductor 把 eager 的逐算子启动冻成融合 Triton/C++ kernel，减少每次启动手续费与显存往返，编译后单步更快（2 层 MiniGPT 约 1.2~2×，取决于 kernel 与 batch）。aot_eager 只交图不生成代码，无加速但仍可验证图能导出。

### 练习 2 · 用 torch._dynamo.config.trace 打开日志，观察 Dynamo 的 graph break 位置

一句话思路：Dynamo 追踪时遇到无法编译的 Python 操作就在那里 graph break（拆成多张图）；开 trace 日志会把每个 break 的位置/原因打出来。

```python
import logging
torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.verbose = True
torch._dynamo.config.trace = True             # 打开 trace
with torch.no_grad():
    torch.compile(model)(inputs)              # 触发追踪
```

预期：日志标注出整个前向被分成了几张图，以及 graph break 发生在哪个算子（通常因控制流/数据依赖/不支持的操作），说明 torch.compile 无法把全部前向融合成一张图，只能逐段编译。

### 练习 3 · 把 MiniGPT 换成 4 层，重新数算子并更新融合收益估算

一句话思路：用 OpCounter 数 4 层 MiniGPT 一次前向的算子数，与 2 层对比；算出的「少开的火 × 手续费 + 省显存往返」融合收益随算子数翻倍而放大。

```python
for name, model in [("2层", m2), ("4层", m4)]:
    c = OpCounter(); 
    with c: out = model(inputs)
    n_ops = len(c.ops)
    est = (n_ops - 1) * t_l                    # 少开的火 × 手续费估算
    print(f"{name}: 算子 {n_ops} 个, 估算融合收益 {est:.3f} ms/步")
```

预期：4 层前向的算子数近似 2 层的 2 倍，融合收益的估算也随算子数放大（每步省下的启动手续费更多）——说明模型越大、算子越多的前向，torch.compile/CUDA Graph 的收益占比越大。

### 练习 4 · 阅读 vLLM compilation.py，找出 cudagraph_mode 与 cudagraph_capture_sizes 的关系

一句话思路：vLLM 的 CompilationMode 决定是否用 CUDA Graph 双层；cudagraph_capture_sizes 决定预录哪几档 batch 桶，`cudagraph_mode` 决定用什么捕获策略去把桶里的 forward 录成图。

答案要点：`cudagraph_mode`（如 lazy/atmost1 等）控制 CUDA Graph 的捕获时机与策略，决定「什么时候、以什么方式为某个 batch 桶建图」；`cudagraph_capture_sizes` 提供一批离散的 batch 尺寸（如 [1,2,4,8,...] 桶），运行时的实际 batch 向上取整到最近桶，用该桶预录好的图来重放。模式决定「怎么录」，capture_sizes 决定「录哪些档」。vLLM 默认在 INDUCTOR_CUDA_GRAPHS 模式下「先 Inductor 融合、再整段捕获成 CUDA Graph」。

---

## 第 27 课 · 性能剖析与 Serving 指标（profiling_metrics）

### 练习 1 · 把 B 从 4 改成 32，重跑第 4/5 节，观察 TTFT/TPOT 如何随并发变化

一句话思路：B 越大 prefill 那一批要处理的 token 越多，TTFT 上升；decode 每步仍是 T=B，TPOT 因批内共享前向上升略为明显，整体并发越高排队越久。

```python
for B in [4, 32]:
    metrics = run_serving(B=B)                 # 复用 4/5 节 serving 模拟
    print(f"B={B:2d}: TTFT={metrics['ttft']:.1f}ms  "
          f"TPOT={metrics['tpot']:.1f}ms  TPS={metrics['tps']:.1f}")
```

预期：B 从 4→32，prefill 批变大使首 token time（TTFT）明显上升；TPOT 略微上升（批内并行的 decode 变重）；TPS（吞吐）随并发上升。三者随 B 的移动反映了「并发高→排队与批重→TTFT/TPOT 上升、吞吐提升」的量级权衡。

### 练习 2 · 在 decode 循环里人为插入一次长 prefill（模拟 Sarathi 场景），观察 ITL 的尖峰

一句话思路：把一些 prefill 与 decode 混排（Sarathi 的分块/迭代调度），在 decode 序列里插入一次耗时长的 prefill，ITL（生成间隔）会在那一帧出现明显尖峰并抬高高位 P95。

```python
itls = []
for step in range(N):
    if step == mid: itls.append(long_prefill_ms)   # 人为插入一次长 prefill
    else: itls.append(step_decode_ms)
p95 = np.percentile(itls, 95)
print(f"ITL 平均 {np.mean(itls):.1f}ms, P95 {p95:.1f}ms (尖峰把 P95 抬高)")
```

预期：长 prefill 那一帧的 ITL 数倍于普通 decode，成为序列里的尖峰；P95 被这个尖峰显著抬高，说明在混批场景下以平均 ITL 判断会漏掉最差帧的体验。

### 练习 3 · 用 torch.profiler 的 ProfilerActivity.CUDA 在 GPU 上剖析，对比 kernel 时间

一句话思路：profiler 同时开 CUDA/CPU activity，比较 cpu_time_total（含调度）与 kernel 层的 cuda_time_total，区分「调度开销」与「GPU 计算」各占多少。

```python
import torch, torch.profiler as prof
with prof.profile(activities=[prof.ProfilerActivity.CUDA, prof.ProfilerActivity.CPU]) as p:
    out = model(inputs)
evs = [e for e in p.key_averages() if e.device_type == 1]   # CUDA kernel
top = sorted(evs, key=lambda e: e.cuda_time_total, reverse=True)[:5]
for e in top: print(f"{e.key}: {e.cuda_time_total:.2f} ms")
```

预期：在 GPU 上能直接看到各 kernel 的 cuda_time_total，cpu_total 往往远大于 cuda_total（大量时间花在 Python/调度层），说明 decode 场景 CPU launch 开销与 kernel 时间并重，为 CUDA Graph/compile 提供量化依据。

### 练习 4 · 用 Etalon 的 fluidity-index 思路，给本课的 decode 时间序列算一个「流畅度」分数

一句话思路：Etalon 流畅度 = `(1 + CoV)^(-1)`，其中 CoV = 标准差/均值；ITL 序列越平稳（CoV 越小）分数越接近 1，越抖动分数越低。

```python
def fluidity(itls):
    itls = np.array(itls)
    cov = itls.std() / itls.mean()
    return 1.0 / (1.0 + cov)                 # CoV 越小越流畅, 分数→1
smooth = fluidity([5,5,5,5,5])               # 平稳 => CoV=0 => 分数 1.0
jittery = fluidity([2,1,8,1,2,20])           # 抖动 => 分数 < 0.5
print(f"平稳序列流畅度 {smooth:.2f}, 抖动序列流畅度 {jittery:.2f}")
```

预期：平稳的 decode 时间序列流畅度约 1.0，混入长 prefill 尖峰后 CoV 拉大致分数显著下降（如 0.4 以下）——把「生成是否卡顿」用一个 0~1 的分数量化，可直接对比不同调度在流畅度维度的差异。