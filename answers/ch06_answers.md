# 第 6 章 · 分布式并行 — 参考答案（ch06，第 35–41 课）

> **参考答案 · 建议先自己动手再看。** 本文件与仓库 `exercises/ch06/` 下 35–41 号 notebook 一一对应。
> 代码假设与你 notebook 中**已运行过的定义一致**才可独立运行（会标注依赖）；术语保留英文，答案用中文。

---

## 第 35 课 · 分布式并行总览

> 对应 `exercises/ch06/35_parallel_overview.ipynb`。依赖：`estimate_weights_gb` / `estimate_kv_gb`（作者 notebook 定义的估算函数）。

### 练习 1：ctx_len 从 4096 改成 128k，重算 KV 显存
**思路**：KV 显存与 `ctx_len` 成正比，128k 把每 token 的 KV 占用放大 31.25 倍。

```python
def kv_gb(L, h_kv, d, ctx, batch=1, b=2):
    return 2*L*h_kv*d*b*ctx*batch / 1024**3
for ctx in [4096, 128000]:
    g = kv_gb(32, 8, 128, ctx)                     # Llama-3-8B 类配置
    print(f"ctx={ctx:>7d}: KV = {g:.2f} GB (fp16, 8B 模型)")
```
**预期结果**：4k 时不到 1 GB，128k 时约放大３１倍、可达数十 GB——长上下文把 KV cache 变成显存第一大户，直观体现长上下文压力。

### 练习 2：给 pyecharts 架构图加"延迟影响"属性，比较四种并行
**思路**：给并行维度补一个"每单位通信/延迟开销"维度，相对大小供条形图比较。

```python
# 延迟影响(相对,新增一个单位的相对成本):TP 每层 allreduce 最重,CP 交换 context 次之,
# PP 只加微批气泡(对单请求延迟小),DP 副本独立几乎不增加单请求延迟。
lat = {"TP": 1.0, "PP": 0.6, "CP": 0.8, "DP": 0.2}
for k, v in lat.items():
    print(f"{k}: 延迟影响(相对) = {v}  (TP 通信最密集,DP 最轻)")
# 用 pyecharts Bar 的 name 字段带上这个属性即可
```
**结论**：TP/CP 引入逐层/逐 token 通信，延迟影响最大；PP 的代价是管道气泡（单位微批粒度）；DP 完全不加单请求延迟。这是选择并行策略的顺序依据。

### 练习 3：70B 模型 int8 时，最少几张 16GB 卡只放权重
**思路**：int8 每参数 1 字节 → 70B 权重 = 70 GB；不把 KV/激活算进去的最小卡数。

```python
weights_gb = 70 * 1            # 70B 参数 × 1 B/参数(int8)
per_card = 16
print(f"需卡数 = ceil(70/16) = {(-(-weights_gb//per_card))} 张(不计 KV/激活)")
```
**预期结果**：`ceil(70/16) = 5` 张（每卡约 14 GB，凑得下）。实际部署还要给激活和 KV cache 留余量，通常需更多卡。

### 练习 4：查 vLLM 文档，vLLM 何时自动启用 TP
**思路**：查 `vllm/serving` 与 `docs 分布式推理` 章节。
**结论**：vLLM 在**单张 GPU 放不下模型权重时自动启用 TP**——加载权重时检测 `--tensor-parallel-size` 下每卡分片能否装下，放不下则自动把 TP 提升到能装下的最小整数（也可用 `tensor_parallel_size="auto"` 让框架动态选择）。单卡能装下时 TP 默认取你指定值；`--gpu-memory-utilization` 与 `--max-model-len` 也参与"是否触发自动扩张"的判断。

---

## 第 36 课 · AllReduce 与 NCCL：ring / tree 算法

> 对应 `exercises/ch06/36_allreduce_nccl.ipynb`。依赖：`ring_allreduce` / `tree_allreduce` / `make_chunks` / `ring_rounds` / `tree_rounds`。

### 练习 1：卡数改成 8，验证正确性并打印 record
**思路**：`ring_allreduce` 对任意 N 成立；`record` 参数记录每一步 `(阶段, 步, 发送卡, 接收卡, 块号)`。

```python
rng = np.random.default_rng(0); n = 8
chunks = make_chunks(rng, n, 1000)
answer = np.stack(chunks).sum(axis=0)
rec = []
res = ring_allreduce([c.copy() for c in chunks], record=rec)
ok = all(np.allclose(res[r], answer, atol=1e-4, rtol=1e-4) for r in range(n))
print("8 卡 ring-allreduce:", "Pass" if ok else "Fail", f"轮数={len(rec)}")
print("前 10 步 record:", *rec[:10], sep="\n  ")
```
**预期结果**：8 卡仍全部 Pass；`len(rec) = 2(N-1) = 14`（reduce-scatter 7 步 + allgather 7 步），record 显示每步每个 rank 与前一家 `(r-1)%8` 配对。

### 练习 2：实现 ring-reduce（只 reduce-scatter，不做 allgather），统计发送量
**思路**：删掉 `ring_allreduce` 的 allgather 循环，只保留 reduce-scatter，每卡最终只剩"自己负责的那一块的全局和"。

```python
def ring_reduce_scatter(chunks):
    n = len(chunks)
    for step in range(n-1):
        snap = [c.copy() for c in chunks]
        for r in range(n):
            src, blk = (r-1) % n, (r-step) % n
            chunks[r][blk] = snap[r][blk] + snap[src][blk]
    return chunks

n = 8; data_mb = 1.0
sent_per_step = data_mb / n          # 每步每卡发 1 块
print(f"ring-reduce 每卡发送量 = {n-1} 步 × {sent_per_step:.2f}MB = {(n-1)*sent_per_step:.2f}MB")
print(f"(full allreduce 发送量 = 2(N-1)/N×data = {2*(n-1)/n*data_mb:.2f}MB, 是其一半)")
```
**预期结果**：发送量 = `(N-1)·(data/N)`，恰好是 full allreduce（含 allgather）的一半。reduce-scatter 只把数据"减"到位，未做广播。

### 练习 3：用 numpy 验证 tree 的 halving 结束后 0 号卡持全量总和
**思路**：halving 阶段就是"距离 2^k 的伙伴两两求和"，log2(N) 步后拍平到 rank 0。

```python
data = [rng.normal(size=500).astype(np.float32) for _ in range(4)]
answer = np.stack(data).sum(axis=0)
d = [x.copy() for x in data]; dist = 1
while dist < len(d):
    for r in range(len(d)):
        if (r // dist) % 2 == 1:
            d[r - dist] = d[r - dist] + d[r]
    dist *= 2
print("halt halving 后 rank0 持全量总和:", np.allclose(d[0], answer, atol=1e-4, rtol=1e-4))
```
**预期结果**：`True`——halving 阶段结束，`d[0] == Σ所有卡`。后续 doubling 阶段只是把这个总和二分广播回各卡。

### 练习 4：推导并验证 N=2 时 ring 和 tree 轮数都为 2
**思路**：直接代入各自公式 `ring_rounds=2(N-1)`、`tree_rounds=2·⌈log2 N⌉`。

```python
print(f"ring_rounds(2) = 2*(2-1) = {ring_rounds(2)}")
print(f"tree_rounds(2) = 2*ceil(log2 2) = {tree_rounds(2)}")
```
**预期结果**：两者都等于 **2**。N=2 时 ring 只需 1 步 reduce-scatter + 1 步 allgather，tree 只需 1 步 halving + 1 步 doubling——两种算法在双卡情形退化为相同的最短轮数。

---

## 第 37 课 · 张量并行：列切与行切

> 对应 `exercises/ch06/37_tensor_parallel.ipynb`。依赖：`column_parallel_linear` / `row_parallel_linear` / `gelu` / `mlp_column_row_shard` / `bench`，及 `rng` / `X` / `W` / `A`。

### 练习 1：把 TP_LIN 的 tp 改成 4，验证组合仍一致
**思路**：列切（沿 out 维 `axis=1`）+ 行切（沿 in 维 `axis=0` 部分和相加）对任意正整数 tp 都保持 `xW` 的数学等价。

```python
def col(x, Wfull, tp): return [x@w for w in np.split(Wfull, tp, axis=1)]
def row(xs, Afull, tp):
    return sum(x@a for x, a in zip(xs, np.split(Afull, tp, axis=0)))
tp = 4
Y_tp = row(col(X, W, tp), A, tp)
print("TP4 列切+行切 vs 稠密:", "一致" if np.allclose(Y_tp, (X@W)@A, atol=1e-5) else "不一致")
```
**预期结果**：`一致`。切法由 `axis` 决定、与 tp 无关，故任意卡数都成立。

### 练习 2：把 GELU 挪到 down_proj 之后，观察是否还一致，并解释 why
**思路**：GELU 是非线性、不可与 allreduce 的求和交换，必须紧跟在列切 `up_proj` 之后的**每卡本地**执行。

```python
xs = np.split(X, 4, axis=1)                       # 等份输入(配对前一步)
up  = np.split(W_up, 4, axis=1); down = np.split(W_down, 4, axis=0)
# 错误：先 down 后 gelu —— 非线性作用在"部分和"上
wrong = gelu(sum(x@u@dl for x,u,dl in zip(xs, up, down)))
# 正确：gelu 在 up 后、allreduce 前
right = sum(gelu(x@u)@dl for x,u,dl in zip(xs, up, down))
ref = gelu(X@W_up) @ W_down
print("错误(先down后gelu)一致?", np.allclose(wrong, ref, atol=1e-4))
print("正确(gelu在up后)  一致?", np.allclose(right, ref, atol=1e-4))
```
**预期结果**：错误写法 **不一致**，正确写法 **一致**。因为 `gelu(a+b) ≠ gelu(a)+gelu(b)`，非线性不能跨过 allreduce 的求和。这也解释了 Megatron 里激活函数必须夹在 up_proj 列切和 down_proj 行切之间。

### 练习 3：实现 4 卡版 column→row 配对，画每卡内存占用条形图
**思路**：每卡持有自己那份 up 权重列 + down 权重行，内存近似均分。

```python
import matplotlib.pyplot as plt
tp = 4
IN, HID, OUT = 64, 128, 64
mem_per_card = []                  # 每卡: up列切片行数 + down行切片行数
for i in range(tp):
    mem = (IN//tp)*HID + (HID//tp)*OUT           # 列切部分 + 行切部分(元素数)
    mem_per_card.append(mem)
plt.bar(range(tp), mem_per_card, color="steelblue")
plt.xlabel("rank"); plt.ylabel("权重元素数/卡"); plt.title("4 卡 TP 每卡显存"); plt.show()
print("每卡权重元素数:", mem_per_card)
```
**预期结果**：4 根柱子高度几乎相同——列切+行切后各卡权重量与通信量都均衡，无热点副本。

### 练习 4：查 Megatron 图 3，对照本课 up/down 切法画示意
**思路**：Megatron-LM Fig.3 展示一个 Transformer 层的两种并行分割：Attention 的 QKV 列切 + output 层行切，以及 FFN 的 up 列切 + down 行切（与本课 `mlp_column_row_shard` 完全同构）。
**示意结论**（open-ended，可手绘或用 mermaid/ matplotlib）：
- 左半（column）：`x @ W_up_i` 各卡本地算 `h_i`（无需通信）；
- 中间：本地 `gelu(h_i)`；
- 右半（row）：`h_i @ W_down_i` 得部分和，一次 AllReduce 相加=完整输出。
**要点**：权重被切成 `1/tp` 份降低单卡显存；每层仅 1 次 AllReduce（关键通信点），因此 TP 的可扩展性受通信带宽限制。

---

## 第 38 课 · 流水线并行：切层、微批与气泡

> 对应 `exercises/ch06/38_pipeline_parallel.ipynb`。依赖：`_run_pipeline` / `schedule_gpipe` / `schedule_1f1b` / `stage_peak_inflight` / `gantt_stats`。

### 练习 1：f=2、b=1 重算气泡，验证是否还等于 (p-1)/(m+p-1)
**思路**：气泡公式的推导假设各舞台 f、b 相同但**不要求 f=b**——前向+反向的一个"完整槽位"时长 `f+b` 会在分子分母同时约掉。

```python
for f, b in [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0)]:
    st = gantt_stats(schedule_1f1b(4, 8, f=f, b=b), 4)
    formula = (4-1)/(8+4-1)
    print(f"f={f},b={b}: 模拟气泡={st['bubble']:.4f}  公式={formula:.4f}  匹配={abs(st['bubble']-formula)<1e-9}")
```
**预期结果**：三组 f/b 全部匹配公式。因为流水线每个微批通过每个舞台耗时恒为 `f+b`，总时长 `(m+p-1)(f+b)`、气泡 `(p-1)(f+b)`，比值与 f/b 无关（前提各舞台 f、b 相同）。

### 练习 2：输出 stage_peak_inflight，对比 GPipe 与 1F1B
**思路**：`stage_peak_inflight` 已定义，直接对两种调度取各舞台最大驻留激活数。

```python
for name, f in [("GPipe", schedule_gpipe), ("1F1B", schedule_1f1b)]:
    ev = f(4, 8)
    peaks = [stage_peak_inflight(ev, s) for s in range(4)]
    print(f"{name}: 各舞台峰值驻留 = {peaks}, 全局最大 = {max(peaks)}")
```
**预期结果**：GPipe 峰值驻留 ≈ `m=8`（所有微批激活几乎同时驻留）；1F1B ≈ `p=4` 附近。**1F1B 把激活显存从 m 压到 ≈p**，这是它省显存的关键。

### 练习 3：实现"交错式"(interleaved) 调度并比较气泡
**思路**：把 p 个物理 stage 再细分 v 个 virtual stage，每个微批以更高粒度交错，气泡从 `(p-1)/(m+p-1)` 降到约 `(p-1)/(v·m+p-1)`。

```python
# 简化模型:把有效"微批槽 v·m"代入气泡公式
def interleaved_bubble(p, m, v):
    return (p-1) / (v*m + p - 1)
for v in [1, 2, 3, 4]:
    print(f"v={v}: 交错气泡 ≈ {interleaved_bubble(4, 8, v):.3f}  (原GPipe/1F1B v=1)")
```
**预期结果**：v 越大气泡越低（micro-batch 交错粒度越细、管道填充/排空占比越小）。你可将 `schedule_1f1b` 的微批列表换成 v 份更细队列并用 `gantt_stats` 画出甘特图作对比。代价是实现与通信更复杂。

### 练习 4：p 固定 8、m 从 2 到 64，画气泡下降曲线找"性价比"区间
**思路**：气泡 `=(p-1)/(m+p-1)` 随 m 单调递减但斜率减小的凸函数。

```python
import numpy as np
p = 8
for m in [2, 4, 8, 16, 32, 64]:
    print(f"m={m:>3}: 气泡 = {(p-1)/(m+p-1):.3f}")
# 边际收益 = 气泡下降量
last = 1.0
for m in [2, 4, 8, 16, 32, 64]:
    b = (p-1)/(m+p-1); print(f"m {last:.0f}->{m}: 气泡下降 {last-b:.3f}"); last = b
```
**预期结果**：m 从 2→8 气泡从 0.78 猛降到 0.47（性价比最高）；8→16 降到 0.33；此后每加倍只降几个点（收益递减）。**"性价比最高"区间约在 m≈p(8) 到 m≈4p(32) 之间**；m 超过 ~32 后加大微批收益甚微，却线性增加激活显存。

---

## 第 39 课 · 数据并行：推理负载均衡与训练梯度同步

> 对应 `exercises/ch06/39_data_parallel_dp.ipynb`。依赖：`make_stream` / `dispatch_dp` / `dp_stats` / `linear_grad`，及 `Req` / 数据 `X_all` / `y_all` / `full_grad`。

### 练习 1：改 wmin/wmax，看哪种策略占优
**思路**：请求耗时跨度越大（wmax/wmin 比大）越不均 → `least_loaded` 优势越明显；越均匀 → `round_robin` 也几乎最优。

```python
for lo, hi in [(2, 30), (5, 80), (20, 24)]:
    stream = make_stream(24, np.random.default_rng(3), lo, hi)
    for pol in ["round_robin", "least_loaded"]:
        lanes = dispatch_dp([Req(r.rid, r.arrive, r.work) for r in stream], 4, pol)
        st = dp_stats(lanes)
        print(f"w∈[{lo},{hi}] {pol:<13}: p95={st['p95_lat']:6.2f}  imbalance={st['imbalance']:.3f}")
```
**预期结果**：`(2,30)` 跨度和 `(5,80)` 下 `least_loaded` 的 p95 与 imbalance 明显更小；`(20,24)` 均匀请求时两策略接近（round_robin 已够用）。负载越不均，动态决策越值得。

### 练习 2：加一种"按预估耗时最长的副本优先避让"策略
**思路**：`dispatch_dp` 的 `least_loaded` 本质就是把请求派给"预估剩余负载最小"（`loads[k]=max(loads,r.arrive)+r.work`）的副本——即自动避让已堆积最多工作的副本。

```python
def dispatch_avoid_heaviest(reqs, n_replica):
    loads = [0.0]*n_replica; lanes = [[] for _ in range(n_replica)]
    for r in reqs:
        k = int(np.argmin(loads))          # 挑"预计最早空闲"=剩余负载最小
        lanes[k].append(r)
        loads[k] = max(loads[k], r.arrive) + r.work      # 预估其完成时刻
    return lanes
st_avoid = dp_stats(dispatch_avoid_heaviest(reqs, 4))
print("避让最重副本策略:", st_avoid)
```
**预期结果**：该策略与 `least_loaded` 行为一致（都以 `max(load,r.arrive)+r.work` 为预计完成时刻），对"哪个副本最忙就少派给它"给出了显式表达；相比 `round_robin` 在长耗时请求下 p95 更低。

### 练习 3：把 DP_GRAD 的副本数 R 改成 6，重跑梯度同步
**思路**：梯度对样本是**线性的**，局部梯度平均=全量梯度与副本数 R 无关。

```python
R = 6
shards = np.array_split(np.arange(N), R)
local = [linear_grad(X_all[idx], y_all[idx], w) for idx in shards]
avg_grad = np.mean(local, axis=0)
print("R=6 AllReduce平均梯度 vs 全量梯度:",
      "一致" if np.allclose(avg_grad, full_grad, atol=1e-6) else "不一致")
```
**预期结果**：`一致`。因为 `avg 局部梯度(各样本子集)` = `全量梯度`，R 取任何值都成立——这是分布训练"梯度一致"的数学保证。

### 练习 4：pyecharts 画"副本数 vs 总完成时间"，找收益递减点
**思路**：推理 DP 只分散排队、不降低总计算量，副本多了仅改善排队的边际收益。

```python
import numpy as np
def total_time(replicas):
    lanes = dispatch_dp([Req(r.rid, r.arrive, r.work) for r in stream], replicas, "least_loaded")
    return dp_stats(lanes)["makespan"]
for r in [1, 2, 4, 8, 16]:
    print(f"副本数={r:>2}: 总完成时间={total_time(r):.2f}")
```
**预期结果**：makespan 随副本数先快速下降、后趋于平台（请求总数 24 有限，副本超过一定程度后每条 lane 不再排队，收益归零）。**收益递减点约在副本数≈每条请求之间的竞争拐点处**；继续加副本纯粹浪费显存。

---

## 第 40 课 · vLLM 的 CustomAllreduce：共享内存替掉 NCCL

> 对应 `exercises/ch06/40_custom_allreduce.ipynb`。依赖：`custom_ar_latency` / `nccl_ar_latency` / `ar_latency`。

### 练习 1：调整 alpha_c/alpha_n，观察 crossover 移动方向
**思路**：crossover 是"custom 与 nccl 延迟相等"的消息大小；固定开销越小，相交点越靠左（越小消息就越值得用）。

```python
def crossover(ac, an):
    lo, hi = 1e-4, 1e4
    for _ in range(200):                    # 二分求 custom==nccl 的 msg_mb
        mid = (lo+hi)/2
        if custom_ar_latency(mid, 8, alpha_s=ac) < nccl_ar_latency(mid, 8, alpha_s=an):
            lo = mid
        else:
            hi = mid
    return lo
for (ac, an) in [(2e-6, 25e-6), (10e-6, 25e-6), (2e-6, 50e-6)]:
    print(f"alpha_c={ac:.0e} alpha_n={an:.0e}: crossover ≈ {crossover(ac,an):.2f} MB")
```
**预期结果**：增大 `alpha_c`（custom 固定开销变高）→ crossover **右移**（要更大的消息才值得用 custom）；增大 `alpha_n`（nccl 开销变高）→ crossover **左移**（custom 更容易赢）。cross 点决定了小消息选 custom、大消息才考虑别的权衡。

### 练习 2：写 fixed_share 占比公式，手算 0.01MB 时固定开销占比
**思路**：`固定占比 = alpha / (alpha + 传输项)`，小消息下传输项 ≈ 0，因此占比接近 100%。

```python
def fixed_share(mb, n, alpha_s=2e-6, bw=80e9):
    msg = mb * 1e6; ring = 2*(n-1)/n
    total = (alpha_s + ring*msg/bw)         # 秒
    return alpha_s / total
for mb in [0.01, 1.0, 100.0]:
    print(f"msg={mb:6.2f}MB: 固定开销占比 = {fixed_share(mb,8):.1%}")
```
**预期结果**：`0.01MB`（≈10KB）时占比 ≈90%+（传输项 `ring·10KB/80GB/s≈0.2µs`，几乎全是 2µs 固定开销）；增大到 `1MB` 略降、`100MB` 时占比可忽略。公式：`alpha / (alpha + 2(N-1)/N · msg_bytes/bw)`。

### 练习 3：为什么 CustomAllreduce 不适用于跨机
**思路**：CustomAllreduce 用**单机内共享内存（shm/mmap）**做进程间直接读写，避免 PCIe/NVLink 的拷贝开销。
**结论**：共享内存只在**单机**内有效；跨机没有共享内存，必须走网络（IB/以太），也就享受不到"共享内存直写"的收益，所以跨机自动回退到 NCCL。这也是它只适用于单机多卡、不跨节点的原因。

### 练习 4：查源码 custom_all_reduce.py，找"回退到 NCCL"的触发条件
**思路**：在 `vllm/distributed/device_communicators/custom_all_reduce.py` 里 grep `fallback` / `return False` / `should_custom_ar`。
**预期（open-ended，以源码为准）**：大约在以下条件之一时对外表现为"不用 CustomAR/回退 NCCL"：
- 要求的卡数超出支持的 world size（CustomAR 一般只对较少的卡数生效）；
- 共享内存分配失败或 `cudaIpcMemHandle` 获取失败；
- 检查 NVLink/SHM 拓扑不满足（如没有 NVLink、PCIe-only）；
- `should_custom_ar` 判定当前 rank 数/张量桶对不上。
**建议**：在源码里搜 `"CustomAllreduce not on NVIDIA platform"` 或 `"unicast address"` 等日志串，可精确还原回退判定。

---

## 第 41 课 · 并行组合：TP × PP × DP

> 对应 `exercises/ch06/41_parallel_combination.ipynb`。依赖：`estimate_combo` / `all_combos`，模型 70B 配置。

### 练习 1：batch 4→64，看 KV 显存暴涨、哪些组合装不下
**思路**：KV ∝ batch，从 4 到 64 放大约 16 倍；权重 + KV 超单卡的组合被排除。

```python
def show(batch):
    print(f"--- batch={batch} ---")
    for tp, pp, dp in all_combos(8):
        e = estimate_combo(70, 80, 8192, gpus=8, tp=tp, pp=pp, dp=dp, batch=batch)
        tot = e["weights_gb"] + e["kv_gb"]
        print(f"TP{tp}*PP{pp}*DP{dp:}: 权重{e['weights_gb']:6.1f}+KV{e['kv_gb']:6.1f}={tot:6.1f} GB")
show(4); show(64)     # 对比两档 batch
```
**预期结果**：batch=64 时 KV 行全部放大约 16 倍，DP 高（权重多副本）的组合总显存轻松突破单卡（如 TP1×DP8 权重 140GB 本就放不下），少数高 TP 低 DP 组合勉强可装；batch 越大、能合法装下的组合越少。

### 练习 2：给定单卡显存，自动筛选能放下的合法组合
**思路**：遍历 `all_combos`，保留 `权重+KV ≤ gpu_gb` 的组合。

```python
def combos_fit(gpu_gb, batch=8):
    res = []
    for tp, pp, dp in all_combos(8):
        e = estimate_combo(70, 80, 8192, tp=tp, pp=pp, dp=dp, batch=batch)
        tot = e["weights_gb"] + e["kv_gb"]
        if tot <= gpu_gb:
            res.append((f"TP{tp}*PP{pp}*DP{dp}", round(tot, 1)))
    return res
print("单卡 24GB 能放下:", combos_fit(24))
print("单卡 48GB 能放下:", combos_fit(48))
```
**预期结果**：单卡 24GB 时只剩少数高 TP 组合；48GB 能容纳更多。该筛选函数正是容量规划/选型的第一步。

### 练习 3：405B 重跑 all_combos(8)，看是否 8 卡装不下及需多少卡
**思路**：405B fp16 权重 810 GB；8 卡里最极端 TP8 也每卡 101 GB，仍超常规单卡。

```python
for tp, pp, dp in all_combos(8):
    wg = 405 * 2 / tp / pp
    print(f"TP{tp}*PP{pp}*DP{dp}: 每卡权重 = {wg:.0f} GB")
# 需要卡数:最小 TP8 时每卡 405*2/tp ≥ 目标单卡容量
for target in [40, 80, 100]:
    tp = 8; cards = max(8, int(np.ceil(405 * 2 / (tp*target))))
    print(f"目标单卡 {target}GB: 至少需要 TP{tp} 时 {cards} 张卡(dy pp/DP 分摊)")
```
**预期结果**：8 卡内无组合能放下（TP8 仍 101 GB/卡）。要让 405B 权重装下，需更多卡（如 `TP8×PP2` 共 16 卡则每卡 ≈50 GB）或改 int8/更高量化。405B 也因此是"必须走并行甚至多机"的典型。

### 练习 4：pyecharts 图加第三个指标"每卡总显存"做三维取舍
**思路**：把"每卡总显存 = 权重 + KV"作为第三维，用散点/3D 展示 TP/PP/DP 的折中。

```python
from pyecharts.charts import Scatter3D
pts = []
for tp, pp, dp in all_combos(8):
    e = estimate_combo(70, 80, 8192, tp=tp, pp=pp, dp=dp, batch=8)
    pts.append([round(e["weights_gb"],1), round(e["kv_gb"],1),
                round(e["weights_gb"]+e["kv_gb"],1)])   # 第三维=每卡总显存
sc = (Scatter3D().add("TP×PP×DP", pts)
      .set_global_opts(title_opts=..., xaxis3d_opts=..., yaxis3d_opts=..., zaxis3d_opts=...))
sc.render_notebook()
```
**预期结果**：三维图上每个合法组合一个点，直观看到"加 DP → 权重多头显存量增、KV 分流；加 TP → 权重与 KV 都摊薄但通信升；加 PP → 权重降但气泡升"。第 3 维"每卡总显存"帮助锁定哪些点不越界。

---