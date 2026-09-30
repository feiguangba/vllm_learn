# 第 9 章 · Triton GPU 编程 · 参考答案

> 参考答案，建议先自己动手再看。对应 notebook：`exercises/ch09/51–60`，10 课。
> 代码假定已按各 notebook 首段设置 `KMP_DUPLICATE_LIB_OK`、`torch.manual_seed(0)` 并导入 `triton` / `triton.language as tl` / `torch`。缩略的 kernel 依赖 notebook 中已定义的变量时会在答案中标明。

---

## 第 51 课 · Triton 是什么

### 练习 1：把 grid_demo 的 BLOCK 改成 8、2，观察每个 program 处理的元素个数如何变化

把 `BLOCK` 从 4 改 8 / 2 重跑，`grid = (cdiv(n, BLOCK),)` 自动缩放。

```python
for BLOCK in (8, 2):                 # n=16, BLOCK=8 → 2 个 program；BLOCK=2 → 8 个 program
    out = torch.zeros(16, device="cuda")
    grid_demo[(triton.cdiv(16, BLOCK),)](out, 16, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()
    print(f"BLOCK={BLOCK}, grid={grid}, 整数部分集合={set(out.int().tolist())}")
```

- 预期：同一 program 主控的元素数 = BLOCK；总数的整数部分集合大小 = grid 大小（BLOCK=8 → `{0,1}`，BLOCK=2 → `{0,...,7}`）。grid 个数 = n/BLOCK。

### 练习 2：给 grid_demo 加一行，把 pid*lane 也存下来，验证块内通道编号

BLOCK=4 时 `pid*lane` 在 program 0 为 `[0,0,0,0]`、program 1 为 `[1,2,3,4]`……

```python
@triton.jit
def grid_demo2(out_ptr, pl_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, pid * 1.0 + tl.arange(0, BLOCK) * 1e-3, mask=mask)
    tl.store(pl_ptr   + offs, pid * tl.arange(0, BLOCK), mask=mask)   # pid*lane
```

- 预期：`pid*lane` 数组整数部分 = pid×块内序号，证明 `lane` 就是块内 0..BLOCK-1 的通道号。

### 练习 3：用 triton.cdiv 解释，当 n 不能被 BLOCK 整除时为什么要向上取整

`cdiv(n, BLOCK) = (n + BLOCK - 1) // BLOCK`，即向上取整。

- 思路：若向下取整，最后的 `n%BLOCK` 个元素就没 program 负责，会漏写；向上取整让最后一个 program 拖动 `offs < n` 掩码兜底，恰好覆盖全部元素。例 `n=17, BLOCK=4`：`cdiv=5`，第 5 个 program 只写下标 16。

### 练习 4：查 triton 的磁盘缓存目录，看编译产物存在哪里

```python
import os
print(os.environ.get("TRITON_CACHE_DIR", "未设置，用系统默认临时目录"))
```
子进程并行编译时多进程可能权限冲突；直接设 `os.environ["TRITON_CACHE_DIR"]=r"D:\tmp\triton_cache"` 指到本机可写目录。

- 预期：产物是 `*.ttir` / `*.ttgir` / `*.llir` / `*.ptx` / `*.cubin` / `.json` 的哈希目录，按 kernel+参数+架构当作 cache key 复用。

---

## 第 52 课 · 第一个 Triton kernel：向量加法

### 练习 1：把 BLOCK 从 1024 改成 333（非 2 的幂），观察是否报错

- 思路：`tl.arange(0, BLOCK)` 要求 BLOCK 是 2 的幂，否则编译时报 `arange` 的上界不是 2 的幂错误。改动前无需重写 kernel，仅把入参 `BLOCK: tl.constexpr = 333` 试一下即可。
- 预期：`TL_ASSERT 2 ** power number` 之类报错。若想支持任意 N，应保持 BLOCK 为 2 的幂，用 `mask = offsets < N` 兜底。

### 练习 2：给 add_kernel 增加第三个向量 z，实现三向量相加 y = a + b + c

```python
@triton.jit
def add3_kernel(a, b, c, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    av, bv, cv = tl.load(a+offs, mask=mask), tl.load(b+offs, mask=mask), tl.load(c+offs, mask=mask)
    tl.store(y+offs, av + bv + cv, mask=mask)
```
调用时多传一个 `c` 张量。对拍 `y_ref = a+b+c`，`assert_allclose` 误差在 `1e-5` 量级内通过。

### 练习 3：用 torch.cuda.synchronize 前后计时对比，体会不同步导致的计时虚低

```python
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100): add_kernel[grid](a, b, y, N, BLOCK=1024)
t_no_sync = time.perf_counter() - t0
torch.cuda.synchronize(); t1 = time.perf_counter()   # 仅最后一次同步
```
- 预期：因为 GPU 异步队列堆积，不同步会测得明显偏低的“假快”。正确姿势：每个循环内 `torch.cuda.synchronize()`，或对整批只计时且末尾同步一次、并同量级对比。

### 练习 4：把 BLOCK 扫到 8192，看带宽是否回落，找出最佳点

```python
for BLOCK in (512, 1024, 2048, 4096, 8192):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100): add_kernel[grid](a, b, y, N, BLOCK=BLOCK)
    torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/100
    gbs = 2 * N * 4 / dt / 1e9
    print(f"BLOCK={BLOCK}: {gbs:.0f} GB/s")
```
- 预期：带宽先在中等 BLOCK 上进入平台期（接近 HBM 峰值），BLOCK 过大后寄存器/占用率下降、调度变差，带宽轻微回落。最佳点需在本机实测，常见 1k–4k。

---

## 第 53 课 · tile 编程模型

### 练习 1：把 tile2d 的 BM、BN 改成非整除值（如 3、6），观察 grid 与掩码是否需要处理边界

- 思路：M=6,N=8 时 `BM=3,BN=6`，`grid=(M//BM, N//BN)=(2,2)` 恰好整除不必兜底；若改成 `BM=4,BN=5` 这类不整除的，`pid_m*BM+rows` 会越过边界，必须加 `mask = (rows < M) & (cols < N)`，否则读垃圾值或越界。
- 预期：不整除时去掉掩码会得到错误/随机值，补掩码后与参考一致。

### 练习 2：修改 tile2d 编码为 pid_m*1000 + pid_n*100 + offs_m*10 + offs_n，重画棋盘

```python
val = pid_m * 1000 + pid_n * 100 + tl.arange(0, BM)[:, None] * 10 + tl.arange(0, BN)[None, :]
tl.store(out_ptr + (pid_m*BM + tl.arange(0, BM))[:, None]*N
         + (pid_n*BN + tl.arange(0, BN))[None, :], val)
```
- 预期：Heatmap 中数值的千位=行 program、百位=列 program、十位=块内行、个位=块内列，棋盘边界清晰可分。

### 练习 3：用 make_block_ptr 改写 tile2d，体会与“指针+下标”写法的区别

```python
A = tl.make_block_ptr(base=out_ptr, shape=(M, N), strides=(N, 1),
                      offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
tl.store(A, val)   # block 指针自动按 shape/strides 铺开，无需手写二维下标与掩码
```
- 思路：block_ptr 一次性声明“整块数据怎么读”，由编译器决定布局；省去手动二维切片，是 GEMM/FlashAttention 的标配写法。

### 练习 4：思考：M 不能被 BM 整除时，程序编号和边界 tile 该如何处理

- 思路：先按 `tm = tl.cdiv(M, BM)` 向上取整定 grid，多出的最后一个 program 用 `other=0`（如 `tl.load(..., mask, other=0.0)`）给越界元素填 0。这样既保证正确性又不浪费已分派线程。

---

## 第 54 课 · Triton GEMM

### 练习 1：把 BLOCK_K 从 32 改成 64/128，观察吞吐与 num_stages 的配合

```python
for BK, ns in [(32,2),(32,4),(64,3),(128,4),(128,5)]:
    ms = bench_gemm(M,N,K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=BK, num_warps=4, num_stages=ns)
```
- 思路：BLOCK_K 影响 tl.dot 内部 K 方向累计宽度与访存粒度，需要与 num_stages 联动。预期中等 BLOCK_K（64）配合适度 stages（3–4）最稳，过大过小都会因占用率或流水线不饱和而掉吞吐。

### 练习 2：给 tl.dot 加 input_precision='ieee'，对比 fp32 精确模式的精度与吞吐

```python
acc = tl.dot(a, b, acc, input_precision="ieee")   # 精确 fp32
acc = tl.dot(a, b, acc, input_precision="tf32")   # 默认 tf32，快但略伤精度
```
- 思路：`input_precision` 三个档 `tf32 / tf32x3 / ieee`。ieee 全精度、吞吐最低；tf32 用截断的张量核、吞吐高、误差 ~1e-3。做精度敏感训练用 ieee，推理尽量 tf32。误差用 `torch.allclose(atol=1e-3)` 评估。

### 练习 3：试 num_stages=4/5，观察流水线加深后吞吐与显存占用的变化

- 思路：num_stages 增加会预取更多 BLOCK_K 分片进 SMEM，流水线更深、吞吐先升后趋稳，但 SMEM 占用随 stages 线性增长。寄存器/SMEM 超过上限时会编译失败或占用率下降。实测找本机 K 下的甜点（通常 3–5）。

### 练习 4：把矩阵改成非方阵（如 4096×1024×2048），验证 mask 边界处理是否正确

```python
M, N, K = 4096, 1024, 2048
M % BLOCK_M 表示尾块：用 `off_m = pid_m*BM + tl.arange(0,BM)`，列 `off_n = pid_n*BN + arange`
mask_m = off_m[:, None] < M；mask_n = off_n[None, :] < N
a = tl.load(a_ptr + off_m[:,None]*K + offk[None,:], mask=(off_m[:,None]<M)&(offk[None,:]<K), other=0.0)
```
- 思路：对 K 和 M/N 三方向都要保 mask，K 方向越界用 `other=0` 让 `tl.dot(a,b,acc)` 不污染累加。与 `(a@b)` 对拍，误差 1e-3 量级通过即正确。

---

## 第 55 课 · Triton FlashAttention

### 练习 1：给 fa_kernel 加因果掩码（下三角），实现 GPT 式 FlashAttention

```python
# 在 q@k 得分上加的是负无穷而不是删行，因为 online softmax 需要完整行归约
scores = tl.dot(q, k.T) * SM_SCALE
mask_causal = col_offs[None, :] <= row_offs[:, None]          # 下三角
scores = tl.where(mask_causal, scores, float("-inf"))
# 后续 running max/sum 归一化逻辑不变；-inf 会导致 -inf - m = nan，
# 更好做法：m_new = tl.maximum(m, tl.max(scores, 0)) 前先把 -inf 换成极小值
```
- 思路：在下三角外的位置放 `-inf`（或用 `mask` + `other`），`-inf` 经 subtract-max 后的 nan 需用 `tl.where(mask_causal, x, 0.0)` 在归一化处再兜底。正确性对拍误差 ~1e-4。

### 练习 2：给 fa_kernel 增加 batch/head 维度，处理 (B, H, N, d) 输入

- 思路：把 B*H 当成一个扁平维度：`bh = tl.program_id(0)`，再 `b = bh // H, h = bh % H`（或直接把 `(B,H)` 压平，grid=(B*H,)），对每个 (b,h) 复用原有 N×N 分块逻辑；指针用 `base + b*H*N*d + h*N*d` 定位。vLLM 的 flash backend 正是这种扁平化处理。

### 练习 3：扫描不同 BLOCK_M/BLOCK_N，找当前 N 下吞吐最高的配置

```python
best = max(((bm, bn) for bm in (32,64,128) for bn in (32,64,128)),
           key=lambda bb: bench_fa(N, *bb))
print("最优 BLOCK_M×BLOCK_N =", best)
```
- 思路：固定 N，暴力网格扫描 BLOCK_M/BLOCK_N，画吞吐热力图找甜点。小 N 用小块更省、大 N 用大块更好；同时观察 num_warps 是否需同步调。

### 练习 4：把 LOG2E 去掉改用 tl.exp，对比正确性与吞吐

```python
# 原: exp2(s * LOG2E) 等价 exp(s)
tl.math.exp(s - m_new)                    # 直接用 exp
tl.exp2(s * 1.4426950408889634 - m_new)   # LOG2E 版本
```
- 思路：`tl.exp2` 是硬件指令、快；`tl.exp` 会编译器展开成 math 调用系列、略慢。两者数值差在 1e-6 量级几乎无差别。体会“乘 LOG2E 用 exp2”是精度与速度的平衡技巧。

---

## 第 56 课 · Triton 编译器原理

### 练习 1：把 vector add 的 BLOCK 从 256 改成 1024 重新编译，对比 TTGIR 里 sizePerThread 的变化

```python
for BLOCK in (256, 1024):
    compiled = vec_kernel.warmup(a_ptr, b_ptr, N, BLOCK, grid=(1,))
    ttgir = compiled.asm["ttgir"]
    print(BLOCK, "sizePerThread=",  # 在 ttg.blocked 布局行解析
          [l for l in ttgir.splitlines() if "sizePerThread" in l])
```
- 思路：sizePerThread 描述每个线程负责的元素数。BLOCK 变大、线程数（num_warps×32）不变时，每个线程要处理的元素线性增多，sizePerThread 随之增大，体现在 TTGIR 的 `ttg.blocked` 布局元数据里。

### 练习 2：给 vector add 加一层 mask（offs < N），再看 TTIR 里多了哪些节点

- 思路：加 `mask` 后 TTIR 图会在 store 前多出 `tt.broadcast`/`tt.masked_load`/谓词扫描等节点。用 `compiled.asm["ttir"]` 打印并 diff，能直观看到条件加载如何影响数据流。预期：多出比较（`arith.cmpi`）、扩展位宽与 predicated op。

### 练习 3：数一数 matmul kernel 的 PTX 行数，并找到 mma / ldmatrix 指令（张量核证据）

```python
ptx = matmul_kernel.warmup(...).asm["ptx"]
print("PTX 行数", len(ptx.splitlines()))
print([l for l in ptx.splitlines() if "mma.sync" in l or "ldmatrix" in l])
```
- 预期：PTX 数百行；`mma.sync.aligned.m16n8k8` 系列的 `mma` 指令存在即证明走 Tensor Core，`ldmatrix` 证明共享内存按矩阵布局读取。

### 练习 4：用 triton.compiler.compile 直接编译一段手写 TTIR 字符串，体验 MLIR 层面

```python
from triton.compiler import compile as tc
try:
    asm = tc(ttir_src, target=f"triton:sm90")          # ttir_src 为 Triton TTIR 方言文本
    print(asm.asm["ttgir"][:2000])
except Exception as e:
    print("TTIR 直接编译对语法 host 约定较高，多用于 Triton 源码、示例见官方 compiler 目录", e)
```
- 思路：这是进阶体验，重点体会“MLIR 方言文本 + 编译 target + options”的输入形态。生产配套用 `triton.compile` + `CompilationUnit`，一般业务只需 `@triton.jit`。

---

## 第 57 课 · Triton 性能调优

### 练习 1：把热力图扩展到 BLOCK_K ∈ {32, 64}，观察 K 方向分块的影响

- 思路：在第 6 节二维热力图的循环里把 `BLOCK_K` 加入候选集合（如 `(16,32),(32,32),(64,64),(32,64)`），同样扫 BLOCK_M/BLOCK_N 画 K× (M,N) 的吞吐面。预期：K 分块过小张量核打不满、过大 SMEM 爆，存在与 M/N 无关的最佳 K。

### 练习 2：在 num_warps=4 时给 num_stages 从 1 扫到 6，画出 stages-time 曲线找拐点

```python
for ns in range(1, 7):
    t = bench_gemm(M, N, K, BLOCK_K=64, num_warps=4, num_stages=ns)
    plt.plot(ns, t, "o")
```
- 预期：曲线先陡降（流水线从不饱和到充分预取）后走平或回升（SMEM 超限→占用率降），拐点通常在 stages=3–5。拐点即该规模下最优 stages。

### 练习 3：给 matmul_auto 加一个 BLOCK=256×256 的候选，观察它是否被选中/是否编译失败

```python
@triton.autotune(configs=[
    triton.Config({"BLOCK_M":128,"BLOCK_N":128,"BLOCK_K":32}, num_warps=4),
    triton.Config({"BLOCK_M":256,"BLOCK_N":256,"BLOCK_K":64}, num_warps=8),  # 大块
], key=["M","N","K"])
```
- 思路：256×256 大块在中等规模可能因 SMEM 超限无法编译，autotune 会跳过失败配置并记录；若编译通过但占用率低，表现在 bench 中较慢就不会被选中，编译器自动完成“能选则选、不选最优淘汰”。

### 练习 4：试试 M=512 时各配置的相对排名是否变化——调参结论依赖规模

- 思路：把同组配置分别跑 M=512 与 M=2048/4096，画横轴=规模、纵轴=每配置吞吐的折线。预期：小规模时启动开销/小块占比高，小 BLOCK 配置占优；大规模时大 BLOCK+深流水线占优，排名发生倒挂。结论：调参必须绑定具体问题规模。

---

## 第 58 课 · Triton 与 vLLM

### 练习 1：把 mini kernel 的 head_size 改成 128，重新对拍

- 思路：head_size 从 d=64 改 d=128 时，让 `BLOCK_D` 取 128（仍是 2 的幂），K 方向 tile 与 softmax 归一化段自动对齐。其余代码不变。对拍（torch 手写 attn）相对误差仍 ~1e-8，即通过。

### 练习 2：给 block_table 加一点“乱序”（块号打散），验证对拍仍成立，体会分页意义

```python
import random
random.seed(0)
blocks = list(range(num_blocks))
random.shuffle(blocks)
bt = torch.tensor([blocks], device="cuda").int()   # 页面在物理块上乱序摆放
```
- 思路：只要 block table 把“逻辑序号→物理块号”映射关系写对，乱序铺放不影响结果。对拍误差仍 ~1e-8，证明 attention 只依赖逻辑顺序与 block table 间接取数据，物理分页与结果解耦。

### 练习 3：在 kernel 里把 tl.exp 换回 tl.exp2，观察误差如何变大，理解第 6 节注释

- 思路：`exp2(s*LOG2E)` 数学上等于 `exp(s)`。若只把 `exp` 换成 `exp2` 而**不再乘** `LOG2E`，相当于指数底从 e 变 2，等于 softmax 温度被缩放，结果明显错误（误差 ~O(1) 而非 1e-8）。这正说明固定写法里 `LOG2E` 是精度环节的一部分。

### 练习 4：读 vendor/vllm 里 triton_reshape_and_cache_flash 的签名，说出 slot_mapping 的用途

- 思路：`reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)` 把每请求新算的增量 K/V 按 `slot_mapping`（逻辑 slot 编号）写入物理 KV cache 的对应位置。`slot_mapping` 是“本 step 该写进物理存储的哪个槽”的映射，是分页写入动作的索引。

---

## 第 59 课 · 调试与验证

### 练习 1：给 safe_kernel 加上“读满两个块”的版本（先算 num_blocks 再循环），验证不越界

```python
@triton.jit
def multi_block_sum(x, out, n, BLOCK: tl.constexpr):
    num_blocks = tl.program_id(1)
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(num_blocks):                        # 每个 program 处理 num_blocks 块
        offs = (tl.program_id(0)*num_blocks + i)*BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x + offs, mask=offs < n, other=0.0)
    tl.store(out + tl.program_id(0)*BLOCK + tl.arange(0, BLOCK), acc, mask=...)
```
- 思路：外层用 `for i in range(num_blocks)` 循环累加多个块，每个块都带 `mask=offs<n`，因此即使 n 不能整除也绝不越界。用越界工具（如开启边界检查）或对拍确认 0 错误读。

### 练习 2：把第 7 节曲线改画成“不归一化、直接画绝对误差”，观察绝对误差随 n 的线性增长

- 思路：把归一化误差改成 `abs_err = torch.abs(out - ref)` 直接画。fp16 单次加法绝对误差随元素数变多而近似线性累积（每次舍入 ~ULP），因此横轴 n 纵轴绝对误差近似直线。这直观展示了“规模越大误差越大”的 fp16 局限。

### 练习 3：写一个 triton 求和 kernel，故意用 fp16 累加，跑第 7 节对比，确认误差数量级

```python
@triton.jit
def sum_fp16(x, out, BLOCK: tl.constexpr):
    offs = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(x+offs).to(tl.float16)          # fp16 累加器 → 大量舍入
    s = tl.sum(acc, 0).to(tl.float32)
    tl.store(out + tl.program_id(0), s)
```
- 思路：fp16 指数位少、尾数舍入大，累加 N 个时误差随 N 累积，最终相对误差比 fp32 高约 2 个数量级（1e-2 vs 1e-4）。这正是第 59 课“累加器一律 fp32”的实践佐证。

### 练习 4：把第 6 节的 assert_allclose 加到第 58 课 paged attention 上，给每个 (seq, head) 打对拍报告

```python
for s in range(seqs):
    for h in range(heads):
        ref = ref_attn[s, h]; got = out[s, h]
        ok  = torch.allclose(got, ref, atol=1e-6, rtol=1e-6)
        maxe = (got-ref).abs().max().item()
        print(f"seq={s} head={h} max_err={maxe:.2e} allclose={ok}")
```
- 预期：绝大多数 (seq,head) allclose=True；若有 False，用误差热力图定位到底是哪个 page/slot 的块表取错了。

---

## 第 60 课 · 用 Triton 写 mini 算子库

### 练习 1：给 OPS 注册表加一个 fused_ln_gelu，并验证它与 run_op('gelu', 'layernorm') 数值一致

```python
# 融合 kernel：读入一行 → LayerNorm → GELU → 一次写回（省掉两趟 HBM 往返）
def fused_ln_gelu(x, eps=1e-5):
    ...: 行内求 mean/var，归一化后 `tl.math.erf` 套 GELU 近似再写回
OPS["fused_ln_gelu"] = fused_ln_gelu
ref = OPS["gelu"](OPS["layernorm"](x))
assert torch.allclose(OPS["fused_ln_gelu"](x), ref, atol=1e-5)
```
- 预期：数值一致（误差 ~1e-5），且融合版只读写各一次，规模越大（HBM 往返占比越高）加速越明显，实测 1.3x–4x。

### 练习 2：把 fused_ln_gelu 的 BLOCK_N 换成 2 的幂 + mask 兜底，支持任意 N

```python
@triton.jit
def fused_ln_gelu(x_ptr, y_ptr, n, eps, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK); mask = offs < n
    x = tl.load(x_ptr+offs, mask=mask, other=0.0)
    mean = tl.sum(x, 0)/BLOCK;  var = tl.sum(x*x, 0)/BLOCK - mean*mean
    xn = (x-mean) * tl.rsqrt(var+eps);  y = xn * (1 + tl.math.erf(xn*0.707)) * 0.5
    tl.store(y_ptr+offs, y, mask=mask)
```
- 思路：BLOCK 取 2 的幂满足 `arange` 约束；把 `n` 补到 BLOCK 的尾部用 `mask`（other=0）在求 mean/var 时不带偏地处理（对 `x< n` 部分统计）。支持任意 N 非 2 的幂。

### 练习 3：用第 57 课的 matmul 思路，把 Softmax 推广成“多行多 program”版本，对比单 program 性能

```python
@triton.jit
def softmax_multi(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    rows = pm*BM + tl.arange(0, BM);  cols = tl.arange(0, BN)
    offs = rows[:, None]*N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr+offs, mask=mask, other=float("-inf"))
    x = x - tl.max(x, 1)[:, None];  e = tl.exp(x);  e = tl.where(mask, e, 0.0)
    tl.store(out_ptr+offs, e / tl.sum(e, 1)[:, None], mask=mask)
```
- 预期：多 program 版把 (M×N) 切成 (M/BM)×(N/BN) 个块并行，吞吐远高于单 program 串行；且删去了单 program 版的 for 循环，靠网格自然并行。

### 练习 4：调研真实 vLLM triton kernel（triton_reshape_and_cache_flash.py），说出输入/输出签名

- 思路：签名为 `(key, value, key_cache, value_cache, slot_mapping, ...)`：输入随前一个 token 的增量 K/V `key,value`，输出写入已管理的分页物理缓存 `key_cache,value_cache`，位置由 `slot_mapping` 指明。它把不同请求的增量 K/V 按槽号 dump 进共享 KV cache——是 vLLM PagedAttention 写侧的关键 kernel。