# 第 10 章 · AI 编译器 · 参考答案

> 参考答案，建议先自己动手再看。对应 notebook：`exercises/ch10/61–70`，10 课。
> 代码假定已 `import torch, torch.nn.functional as F`，并导入 `torch.fx` / `torch._inductor` 等按各课需要。涉及 GPU/后端编译的代码在本机无 CUDA 时以 `torch.compile` 的 fallback 或概念示意为主。

---

## 第 61 课 · AI 编译器全景：为什么需要、怎么工作

### 练习 1：给 compute 函数再加一个 dropout 或 LayerNorm，重新 symbolic_trace，观察图节点如何变多

```python
def compute(x, w, b):
    h = torch.relu(x @ w + b)
    h = F.layer_norm(h, (w.size(1),))
    return h + 0.5 * h                       # 或加 F.dropout(x, p=0.5)

gm = torch.fx.symbolic_trace(compute)        # fx 对 random/受控分支要改写，dropout 需自行随机化
gm.graph.print_tabular()
print(len(gm.graph.nodes), "个节点")
```
- 思路：LayerNorm 会展开成 mean/var/sub/div/mul/add 一串节点；dropout 因含随机分支，fx 静态 trace 会报错，需把随机逻辑抽成独立函数再 trace。节点数随逐元素算子增加而线性变多。

### 练习 2：在生态矩阵里新增『cuDNN/cuBLAS（算子库）』，标上 0/1，说说它覆盖哪几层

- 思路：cuDNN/cuBLAS 属**算子库**而非编译器——它提供高度优化的算子实现，但**顶层不推导、不重排、不做全局图优化**。覆盖“编译流水线”中的 kernel 生成/选择层（提供即用 kernel），不覆盖图优化/调度/代码生成全局层。生态矩阵相应标“中间 kernel 层=1，全局优化层=0”。

### 练习 3：把第 3 节的全景结构图改造成水平四段式，并标注每个 pass 的具体名字

- 思路：改画为 前端/图IR/优化/后端 四段水平流程；在“优化”段标 `常量折叠(constant folding)`、`算子融合(fusion)`、`死代码消除(DCE)`、`循环变换(loop distribution/fusion)` 等 pass 名。每段用箭头串联，箭头上写对应 OKL–TVM 的工具。

### 练习 4：查 TVM 的 Ansor 与 Halide 的 autoscheduler 关系，各用一句话概括定位差异

- 思路：Ansor 是 TVM 的自动调度器，用“模板采样+进化搜索”在高层计算表达上自动找优化 schedule；Halide 的 autoscheduler 是 Halide 框架内用 beam search 自动选 schedule 的组件。差异：Ansor 面向 TVM 的算子/子图并对跨 pass 做联合搜索，Halide autoscheduler 面向 Halide 的算法/调度解耦模型。

---

## 第 62 课 · 计算图优化：让图更小、更简单、更快

### 练习 1：给 constant_fold 增加“所有输入都是常量”的标量算子处理，测一测折叠能省几个节点

```python
def constant_fold(gm):
    cst = lambda n: n.op == "get_attr" or (n.op == "call_function" and all(p in ["literal"] for p in n.args))
    for n in gm.graph.nodes:
        if n.op == "call_function" and all(a.op == "get_attr" for a in n.args):
            with gm.graph.inserting_before(n):
                new = gm.graph.get_attr("fold_"+n.name,  evaluate_const(*[a.target for a in n.args]))
            n.replace_all_uses_with(new); gm.graph.erase_node(n)
    gm.graph.lint(); return gm
```
- 思路：遍历图中入参全为常量的算子（如 `add(sub)`、`mul`），在 Python 侧算出结果替换为一个常量节点并删原节点。统计删掉节点数，常量先验越多的图折叠收益越大。

### 练习 2：给 simplify 增加 relu(relu(x)) → relu(x) 的规则，统计一整个 block 能被化简掉多少

```python
def simplify(gm):
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is torch.nn.functional.relu:
            src = n.args[0]
            if src.op == "call_function" and src.target is torch.nn.functional.relu:
                n.replace_all_uses_with(src); gm.graph.erase_node(n)   # relu 幂等：合并两层
    gm.recompile(); return gm
```
- 思路：利用 `relu` 幂等性（`relu(relu(x))==relu(x)`），把连续两层缩成一层。逐节点重复直到不动点，统计减少节点数。因 relu 保序单调，此规则对任意嵌套层数安全。

### 练习 3：把链长加到 10 再实测，画出『链长 → 加速比』曲线，看收益是否边际递减

```python
L = list(range(2, 11)); speedups = []
for n in L:
    x = torch.randn(1024, 1024)
    t_eager = time(lambda: f(x, n))          # eager 逐算子
    t_opt   = time(lambda: f_optimized(x, n))  # 折叠/融合后
    speedups.append(t_eager / t_opt)
plt.plot(L, speedups)
```
- 思路：每加一层 fusion 至少省一次 kernel 启动与一次中间读写；但新增算子的边际收益逐步走平（融合到一定程度已把可消除读写消除完），曲线呈边际递减并趋于拐点。

### 练习 4：思考：为什么 matmul 不适合和相邻逐元素算子无脑融合

- 思路：matmul 是**计算受限**算子，已高度优化、访存占比低；硬把它与前后逐元素算子融进一个 kernel，会在张量核流水线里插入额外工作、抬高寄存器压力并占用优化机会，常常拖慢而不是加快。只有“融合不改某算子主导计算量且显著省读写”时才值。

---

## 第 63 课 · 内存规划与生命周期：把显存用得抠门

### 练习 1：给 ops 增加一个“被后续两个算子共用”的张量（如残差连接），看生命周期与复用怎么变

- 思路：残差张量 `x` 被两条路径使用，它的最后使用点（liveness 的 last-use）会延后到两条路径都结束，导致这块显存存活期变长、可被复用的窗口变小。峰值内存可能上升；Liveness 分析需在“最后一处使用”才释放，这正是内存规划算法要正确处理的点。

### 练习 2：把 best-fit 改成 first-fit（第一个够用的槽），对比峰值内存是否变差

```python
def first_fit(alloc, size):       # 取第一个空闲且容量>=size 的槽
    for i, slot in enumerate(free_slots):
        if slot >= size: return i
    return new_slot(size)
```
- 思路：first-fit 取第一个够用的槽，会把大块快速切碎导致碎片化，可能使峰值内存高于 best-fit。best-fit 选最接近 size 的槽碎片更小。实测一般 best-fit 峰值 ≤ first-fit。

### 练习 3：在 torch 里对 8 层循环分别用 out-of-place 与 in-place 测峰值显存，画『链长 → 峰值』曲线

```python
def chain_peak(n, inplace):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(n):
        x = x + torch.randn_like(x) if not inplace else x.add_(torch.randn_like(x))
    return torch.cuda.max_memory_allocated()/1e6
L = range(1, 9)
plt.plot(L, [chain_peak(n, False) for n in L], label="out-of-place")
plt.plot(L, [chain_peak(n, True)  for n in L], label="in-place")
```
- 预期：out-of-place 每层分配新张量、峰值随 n 增长；in-place 复用同一块，峰值近常量（只随单张量尺寸）。缺口随链长拉大，直观体现“别乱分配”。

### 练习 4：思考：为什么 LLM 的 KV Cache 需要 paged 分配而不是简单复用

- 思路：KV Cache 大小**随请求动态增长**（每生成一个 token 就多一格），不同序列长度差异巨大，若按最大长度一次性预分配会严重浪费、简单“复用”又无法容纳变长请求。paged 固定块粒度分配：块可动态分配/释放/按序列只占需要的块数，且允许跨请求共享，把碎片化控制在块粒度、降低峰值，也是连续批处理的前提。

---

## 第 64 课 · MLIR 与中间表示：一座『方言之城』

### 练习 1：把 TTGIR 完整打印出来，找出 ttg.blocked 里的 sizePerThread / warpsPerCTA 等调度属性

```python
compiled = vec_kernel.warmup(a_ptr, b_ptr, N, 256, grid=(1,))
ttgir = compiled.asm["ttgir"]
for line in ttgir.splitlines():
    if "sizePerThread" in line or "warpsPerCTA" in line or "threadsPerWarp" in line:
        print(line.strip())
```
- 预期：`ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1/4], order = [0]}>` 等布局元数据，直接反映 num_warps 与 tile 铺开策略。

### 练习 2：自己写一个更复杂的 Triton kernel（逐元素 + 归约），看它的 TTGIR 结构如何变化

```python
@triton.jit
def mix_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(x_ptr+offs)
    b = tl.sum(a, 0)                      # 归约 → TTGIR 会出现 reduce/linear_layout 化
    tl.store(out_ptr, b * a)              # 广播逐元素
```
- 思路：加 `tl.sum` 归约后，TTGIR 会插入 `tt.reduce` 与对应布局转换（layout convert / broadcast），不再只是纯逐元素布局，会出现 `tt.reduce`、`tt.broadcast` 与跨布局 `ordered_broadcast` 节点。

### 练习 3：把『抽象金字塔』图扩展成 5 层，标注每层之间由哪个 pass 完成降级

- 思路：由高到低 5 层：领域专用 IR(TTIR) → tile/并行 IR(TTGIR) → LLVM IR → PTX → SASS/机器码；层间降级分别由 Triton→MLIR 前端、TTGIR pass(布局/流水线/分配到布局)、MLIR LLVM 方言 lower、LLVM → PTX、PTX→SASS(CUDA 最后一步) 完成。每层标记 pass 名。

### 练习 4：查 torch-mlir 如何把 PyTorch 模型导入 MLIR 的 tensor/linalg 方言，画一条完整链路

- 思路：`torch-mlir` 链路：PyTorch 模型 (`torch.fx`/pt2 前端) → `torch` 方言（保留 torch 语义）→ 符号/类型推断 → 降级到 `tensor`/`linalg`（线代表达）→ 再经 linalg→loops→LLVM 等降到可执行硬件目标。链路：PT→torch dialect→linalg/tensor→底层 dialect→LLVM/TPU/GPU target。

---

## 第 65 课 · 自动调优：让编译器自己找最快的菜谱

### 练习 1：把 GEMM 的 tile 配置再扩大一档（如 256 或加 num_warps），补进热力图看性能地形

- 思路：在候选表里加 `(BM=BN=256, BLOCK_K=64)` 与 `num_warps=8/16`，重扫热力图。大 tile 配合多 warps 才能在打满张量核的同时不因 SMEM 超限失败，性能地形会出现新的“高原”区域，最佳点可能右移。

### 练习 2：给演化算法加『变异率』参数，画不同变异率下的收敛曲线

```python
for mr in (0.05, 0.2, 0.5):
    best_seq = run_evolution(mutation_rate=mr)   # 变异率=每代随机扰动某旋钮的概率
    plt.plot(best_seq, label=f"mr={mr}")
```
- 预期：变异率过小收敛慢、易陷局部最优；过大则震荡、难收敛；中间值最快收敛。曲线用“代数 vs 最佳吞吐”表示。退火思路（黄金变异率或衰减学习率）可兼顾。

### 练习 3：在更大规模 GEMM（4096³）上重跑 max-autotune，看它是否更划算

```python
@torch.compile(mode="max-autotune")        # 或 triton.autotune + 更大 M=N=K=4096
def f(a, b): return a @ b
f(x, y)
```
- 预期：大矩阵时编译开销分摊到海量计算上更值，`max-autotune` 的收益更容易超过 `default`/eager。但要记录：编译时间明显变长（数秒级 vs 毫秒级），这就是“编译 vs 运行”的规模拐点。

### 练习 4：思考：为什么成本模型『猜错方向』会陷进局部最优？加『偶尔随机跳』的退火如何改善

- 思路：贪心/爬山按成本模型给出的局部梯度走，若模型在某区域系统性误判“哪个方向更快”，会卡在次级解。退火（Metropolis/simulated annealing）按 `exp(Δ/T)` 概率接受更差解、温度随时间降低，允许初期“随机跳”逃出局部最优，后期收敛到好解。成本模型只需大致对、配合退火即能达到鲁棒优化。

---

## 第 66 课 · 从 PyTorch 到后端代码：模型如何变成 Triton kernel

### 练习 1：给 fwd 加一个 LayerNorm，重跑 torch.fx + get_triton_code，看 kernel 名里的融合算子怎么变

```python
def fwd(x, w, b):
    return F.layer_norm(x, (x.size(-1),)) @ w + b
fx_g = torch.fx.symbolic_trace(fwd)
from torch._inductor.codegen.triton import ...   # inductor 生成的 triton 源码
print(get_triton_code(fx_g))
```
- 预期：kernel 名会包含被融合的算子名（如 `layer_norm*matmul` 或 `triton_per_...`），说明 LayerNorm 的 mean/var 被并入周边 kernel。逐元素链变成 `fused` 一个 kernel，节点显著变少。

### 练习 2：把 backend 换成 inductor 的 mode='reduce-overhead'（CUDA graph），对比耗时与 kernel 数

```python
@torch.compile(mode="reduce-overhead")     # 用 CUDA graph 消除 kernel 间启动开销
def f(x): return torch.relu(x @ w + b)
```
- 思路：`reduce-overhead` 用 CUDA Graph 把一串 kernel 捕获成一次图执行、启动开销几乎归一。对比 eager：kernel 次数接近（算子没少）、但每次启动的 CPU→GPU 间隙被吞掉，小微模型上吞吐提升明显、时延抖动下降。

### 练习 3：用 torch.profiler（若本机 CUPTI 可用）对比 eager 与 inductor 的 kernel 启动序列

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    f(x)
prof.key_averages().table(sort_by="cuda_time_total")
```
- 预期：eager 的 kernel 张量与顺序零散、启动次数多；inductor 序列更短且出现 `triton_...`/`fused` kernel。无 CUPTI 时用 `torch._dynamo` 的 stats 或 kernel 计数代替。

### 练习 4：读一读 get_triton_code 返回的完整源码，找出 num_warps / num_stages 等调度参数

- 思路：生成的 triton 源码开头会有显式启动参数（`triton_heuristics` 派生），在 `@triton.jit` 外层或 `torch._inductor` 配置里找 `num_warps=`, `num_stages=`（也常以 Config/`triton.Config` 形式出现在命令行注入）。inductor 默认按启发式给每 kernel 定配置，可全局覆盖 `torch._inductor.config`。

---

## 第 67 课 · Kernel 融合实战：把一串小 kernel 并成一个

### 练习 1：把 elem_chain 换成 LayerNorm 式融合（减均值/除方差/缩放/加偏置），看融合成几个 kernel

- 思路：LayerNorm 含一阶归约（mean/var），融合后仍是**一个** kernel：先块内 sum/sumsq 归约出 mean、var，再对每元素归一化并缩放加 bias。相比分离版“逐算子各一个 kernel”，融合版读写 HBM 的次数从 O(算子数) 降到 2 次（一次读、一次写）。

### 练习 2：在更大规模（8192²）上重测 softmax，看融合收益是否随规模变化

- 思路：softmax 是访存受限，融合（一行 block 内减 max/归一）省下的中间张量读写随规模放大。8192² 时中间 S 张量本身巨大，融合省的上百 MB HBM 往返比小规模更显著，加速比随规模上升（在带宽平台上走平）。

### 练习 3：思考：为什么大矩阵乘一般不和其他算子硬融

- 思路：大 GEMM 是 compute-bound，张量核已几乎打满，加进逐元素算子只会在核心计算流里插额外工作、占用寄存器与流水线周期，纯粹拖慢。而且它本身就“很优”，融合并不能像对 memory-bound 的逐元素链那样靠省读写明显获益。

### 练习 4：写一个 try/except 包装的 profiler 函数，本机自动回退到编译产物计数

```python
def safe_profile(fn, *args):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(*args)
        return p.key_averages()
    except Exception:
        print("profiler 不可用，回退到 kernel 计数")
        return count_kernels(fn, *args)   # 用 torch._dynamo/torch._inductor.CompiledFxGraph 统计
```
- 思路：把 CUDA profiler 封装进 try/except，出错时回退到统计编译产物 kernel 数。这样代码在本机无 CUPTI/无 GPU 环境仍能给出可比结果，保证练习“可复现、不崩”。

---

## 第 68 课 · 代码生成与调度：决定循环怎么走

### 练习 1：修改调度生成器的 T 与开关，生成 4 种不同调度的伪代码，对比循环层数

- 思路：调节 `T`（splits/分块粒度）与开关（是否 tiling/是否 vectorize/是否并行）生成 4 套循环伪代码；打印每套的循环嵌套层数。如：不 vectorize 则每元素一重循环、vectorize 则展开为 SIMD 宽度循环，tiling 则多出 tile 外层循环，层数与访存相邻性随调度不同。

### 练习 2：把『tiling 甜点』曲线改成不同缓存大小参数，看甜点位置如何移动

- 思路：tile 尺寸的最优点 ≈ 缓存/SMEM 能装下的块大小。把缓存参数学 L1/L2 或 SMEM 尺寸调大，甜点（最优 BLOCK）会向右移；调小则左移。本质是“块大则复用多但装不进缓存，块小装得下但复用少”的平衡点随容量滑动。

### 练习 3：用 torch 对一个大矩阵分别做逐元素循环与向量化加法，画『规模 → 加速比』曲线

```python
def loop_add(a, b):
    out = torch.empty_like(a)
    for i in range(a.numel()): out.flatten()[i] = a.flatten()[i] + b.flatten()[i]  # 逐元素(慢)
    return out
N = np.logspace(4, 7, 8).astype(int)
speedup = [time(vec_add(n)) / time(loop_add(n)) for n in ...]   # 向量化 = a+b 直写
plt.semilogx(N, speedup)
```
- 预期：向量化比逐元素循环快（bandwidth 打满 vs 标量回归），加速比随规模增大趋于某常值（向量宽度决定的理论上限），直观展示“向量化调度”的收益量级。

### 练习 4：思考：为什么自动调优（65 课）要在调度空间里搜？把两者串起来写一段 100 字总结

- 思路：自动调优在为“同一计算选最优调度”而搜——因为不同规模/硬件下最优 tile、向量位宽、loop order 都不同，手写一种调度无法通吃。代码生成负责把调度变成可执行代码，自动调优负责在调度空间里搜最优。二者配合：生成器产出候选，调优器用实测/成本模型打分选优，落地为高性能 kernel。

---

## 第 69 课 · 算子库 vs 编译器：专业厨师与万能厨师

### 练习 1：把矩阵乘的 Triton tile 换成第 65 课的自动调优最优值，看能否反超 cuBLAS

- 思路：用第 65 课 `triton.autotune` 找出的最优 BLOCK_M/N/K + num_warps/stages 配成 custom GEMM，与 cuBLAS（`torch.matmul`）对拍。预期在中等矩阵（如 4096³）能追到 cuBLAS 的 90%–120%；小矩阵因启动/调度开销仍落后。能否反超取决于规模与是否用了 swizzle 等进阶技巧。

### 练习 2：给逐元素链再叠几层算子，对比『逐算子库』与『编译器融合』的差距随层数如何变化

- 思路：编译器融合一整条逐元素链为 1 个 kernel，读写次数不随算子层数增长；算子库逐层执行则每层一次读写的 HBM 往返线性增长。随层数变多，二者差距**拉开**（融合加速比逐渐上升）。本质是 memory-bound 链上“省读写次数”收益随链长累加。

### 练习 3：思考：如果厂商只提供 cuBLAS 而没有编译器，自定义融合算子怎么处理

- 思路：没有编译器时，只能**手写 kernel**：把 cuBLAS 那一步拿到手（理解其融合/布局理念）后，用 CUDA/Triton 手写同等功能的融合 kernel，替代掉那条由多个库调用拼成的链。这正说明编译器是对手写 kernel 生产力的自动化封装，缺编译器就得用“徒手写优化 kernel”兜底。

### 练习 4：查一查 CUTLASS 的『epilogue 融合』，说说它和编译器自动融合的异同

- 思路：CUTLASS epilogue 融合：在 GEMM 主循环之后的 epilogue 阶段把偏置/ReLU/缩放等附加算子融进输出写回，是库内置的“手工模板融合”。与编译器自动融合（如 inductor 对任意链自动分析融合）相比：CUTLASS 只在它预设好的 epilogue 结构里融合有限算子、快而可控；编译器融合更通用、任意组合，但生成的调度未必能像手工模板那样贴近硬件。二者目标一致（省读写），路径不同（模板预置 vs 自动推导）。

---

## 第 70 课 · AI 编译器发展趋势：站在浪潮之巅

### 练习 1：把第 61-70 课的标题画成一张知识地图，标注每课之间的依赖关系

- 思路：主线建议 `61→64→66→67→69`（全景→IR→PT到后端→融合→库与编译器对比），侧枝 `62`(图优化)与 `67` 依赖、`63`(内存)与 `67`/`68` 相关、`65`(自动调优)与 `68`(调度) 串起“搜索调度”线、`70` 汇总。画有向图，箭头标注“打基础→被应用”。

### 练习 2：选一篇论文（Halide / TVM / Ansor）通读，写 500 字笔记

- 思路：示范 Ansor：它解决“TVM/TVM 自动调优效率”问题，把算子拆成 skc（sketch-candidate）模板 + 进化搜索，避免穷举整个优化空间；针对 70 课“编译开销 vs 收益”的核心权衡给出方案——先用低成本 skc 剪枝、再对 top 计划精细化 bench，实现了“快且优的调度搜索”。

### 练习 3：用 torch.compile 的 max-autotune 在更大模型上跑一次，记录编译时间与加速比

```python
import time
import torch
t0 = time.time()
f = torch.compile(lambda x: torch.relu(x @ w + b), mode="max-autotune")
t_compile = time.time() - t0
t0 = time.time()
for _ in range(50): f(x)
torch.cuda.synchronize(); t_run = time.time() - t0
print("编译", t_compile, "s | 运行", t_run, "s")
```
- 预期：编译秒级、运行加速明显；小模型“编译 >> 运行”不划算，大模型运算量大时编译开销被摊薄。记录二者权衡即是本章核心论断的实证。

### 练习 4：为『LLM 推理专用编译』写一段展望

- 思路（要点）：未来 3 年编译器在四类环节继续突破——① 推理专用构图（paged attention/KV cache 作为一等民），② 动态 shape 与 CUDA/昇腾图复用（reduce launch overhead），③ 自动量化/稀疏与算子融合的联合调度，④ 跨硬件（NVIDIA/昇腾/OpenAI 后端）可移植 kernel 自动生成。核心仍是“把硬件细节交给编译器、把性能通过结构分析而非手写解放出来”。