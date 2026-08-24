# -*- coding: utf-8 -*-
"""生成 57_triton_tuning.ipynb 与 app_57_triton_tuning.py"""
from helpers import D, CH09, TRITON_HEADER, BENCH, GEMM, new_nb
from pathlib import Path
from nb_builder import chapter_cover, wrapup

APP_57 = D('''
# -*- coding: utf-8 -*-
# app_57_triton_tuning.py — GEMM 调参扫描 🎛️
import os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_PTXAS = r"D:\\CUDA\\v13.3\\bin\\ptxas.exe"
if os.path.exists(_PTXAS):
    os.environ.setdefault("TRITON_PTXAS_PATH", _PTXAS)
import streamlit as st
import plotly.graph_objects as go
import torch
import triton
import triton.language as tl

st.set_page_config(page_title="Triton 性能调优 🎛️", layout="wide")
st.title("🎛️ 第 57 课 · Triton 性能调优:把 GEMM 调到最快")

st.markdown("""
`num_warps / num_stages / BLOCK` 是 triton 的三大调参旋钮。本页用**真实的 triton 矩阵乘 kernel**
扫描参数:拖动滑杆实测单个配置,点击“扫描全网格”画出 `num_warps × num_stages` 的吞吐热力图,
并与 torch 的 cuBLAS 对比。
""")

# ---------------------------------------------------------------- 与 notebook 一致的 GEMM kernel
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def mm_triton(a, b, BM, BN, BK, nw, ns):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                        num_warps=nw, num_stages=ns)
    return c

def bench_ms(fn, *args, warmup=3, iters=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

# ---------------------------------------------------------------- 侧边栏参数
with st.sidebar:
    st.header("🎛️ 参数")
    M = st.slider("矩阵规模 M=N=K", 128, 512, 256, 64)
    BLOCK_M = st.selectbox("BLOCK_M", [64, 128, 256])
    BLOCK_N = st.selectbox("BLOCK_N", [64, 128, 256])
    BLOCK_K = st.selectbox("BLOCK_K", [32, 64])
    num_warps = st.slider("num_warps", 1, 8, 4, 1)
    num_stages = st.slider("num_stages", 1, 5, 3, 1)
    st.caption("num_warps=线程束数;num_stages=K 方向软件流水线深度;BLOCK=tile 尺寸。")
    scan = st.button("🔍 扫描全网格(num_warps × num_stages)")
    meas = st.button("⏱️ 实测当前配置")

# ---------------------------------------------------------------- 数据准备
a = torch.randn(M, M, device="cuda", dtype=torch.float16)
b = torch.randn(M, M, device="cuda", dtype=torch.float16)

# ---------------------------------------------------------------- 实测当前配置
if meas or scan:
    t_tri = bench_ms(lambda: mm_triton(a, b, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages))
    t_tor = bench_ms(lambda: torch.matmul(a, b))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("triton", f"{t_tri:.3f} ms")
    c2.metric("torch/cuBLAS", f"{t_tor:.3f} ms")
    c3.metric("加速比(torch/triton)", f"{t_tor / max(t_tri, 1e-9):.2f}x")
    c4.metric("triton 吞吐", f"{2 * M ** 3 / (t_tri * 1e6):.1f} GFLOPS")
    bar = go.Figure(go.Bar(x=["triton", "torch/cuBLAS"], y=[t_tri, t_tor],
                           marker_color=["#4C78A8", "#E45756"],
                           text=[f"{t_tri:.3f}", f"{t_tor:.3f}"], textposition="outside"))
    bar.update_layout(title=f"{M}×{M} 矩阵乘:BLOCK={BLOCK_M}×{BLOCK_N}×{BLOCK_K}, "
                            f"warps={num_warps}, stages={num_stages}",
                      yaxis_title="耗时(ms)", height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(bar, use_container_width=True)
    st.caption("🎯 注意:我们的简化 kernel 一般难超 cuBLAS;调参的目标是“尽量接近”,并理解每个旋钮的作用。")

# ---------------------------------------------------------------- 扫描全网格
if scan:
    warps = [1, 2, 4, 8]
    stages = [1, 2, 3, 4]
    z = []
    with st.spinner("正在编译并实测 16 个配置……"):
        for ns in stages:
            row = []
            for nw in warps:
                t = bench_ms(lambda: mm_triton(a, b, BLOCK_M, BLOCK_N, BLOCK_K, nw, ns),
                             warmup=1, iters=5)
                row.append(2 * M ** 3 / (t * 1e6))
            z.append(row)
    fig = go.Figure(go.Heatmap(
        x=[f"warps={w}" for w in warps], y=[f"stages={s}" for s in stages], z=z,
        colorscale="YlOrRd", text=[[f"{v:.0f}" for v in r] for r in z], texttemplate="%{text}",
        colorbar=dict(title="GFLOPS")))
    fig.update_layout(title=f"num_warps × num_stages 吞吐热力图(BLOCK={BLOCK_M}×{BLOCK_N}×{BLOCK_K})",
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ 观察:warps 与 stages 太多/太少都会变慢;最佳点通常在“块内数据量≈线程数”附近。")

st.markdown("""
> 💡 **调参心法**:先固定 BLOCK 扫 warps×stages 找到“调度甜点”,再固定调度扫 BLOCK 找“tile 甜点”,
> 最后用 `triton.autotune` 把整组候选丢给编译器自动选择。
""")
''')

NB = new_nb("第 57 课 · Triton 性能调优",
            subtitle="num_warps / num_stages / BLOCK:用一次真实扫描,把 GEMM 从“能跑”调到“跑得快”",
            emoji="🎛️")

chapter_cover(NB,
    objectives=[
        "掌握 triton 四大性能旋钮:num_warps、num_stages、BLOCK 尺寸、向量化",
        "用 warmup + torch.cuda.synchronize 科学地在 GPU 上计时",
        "对 GEMM 做参数网格扫描,画热力图与曲线找出最优配置",
        "理解 triton.autotune 如何自动替你在候选配置里选最优",
    ],
    toc=[
        ("直觉:厨房里的三个旋钮", "厨师数、上菜流水线、每盘菜的分量"),
        ("四大旋钮逐个拆解", "每个旋钮调的是什么硬件资源"),
        ("先写好“温度计”:科学计时", "GPU 异步执行,不 synchronize 的计时全是幻觉"),
        ("基准 GEMM kernel", "一个 40 行的矩阵乘 + 正确性对拍"),
        ("扫描 num_warps × num_stages", "固定 BLOCK,看调度参数的曲线"),
        ("扫描 BLOCK 尺寸:热力图", "固定调度,看 tile 尺寸的甜点"),
        ("最优配置挑战 cuBLAS", "诚实结论:能接近,但难超越"),
        ("triton.autotune:自动调参", "把选择权交给编译器"),
        ("配套 Streamlit 演示", "app_57_triton_tuning.py:滑杆实时扫描吞吐"),
    ],
    links=[
        ("Triton 官方教程:矩阵乘优化", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
        ("torch.utils.benchmark 文档", "https://pytorch.org/tutorials/recipes/recipes/benchmark.html"),
        ("Roofline 模型(算力 vs 访存的天花板)", "https://en.wikipedia.org/wiki/Roofline_model"),
    ])

NB.md("## 1️⃣ 直觉:厨房里的三个旋钮 🍳",
D('''
把一次 kernel 执行想象成一家餐厅同时给几百桌客人上菜:

- **num_warps(线程束数)= 厨师人数**:人越多出菜越快,但厨房(寄存器/共享内存)就那么大,
  人太多反而互相挤;
- **num_stages(流水线深度)= 上菜流水线的层数**:备菜、炒菜、装盘三站重叠进行,谁都不干等——
  对应 K 方向的软件流水线,让“搬下一块数据”和“算当前块”同时发生;
- **BLOCK(tile 尺寸)= 每盘菜的分量**:一份菜越多,备菜时的数据复用率越高,但盘子(共享内存)
  装不下就溢出。

还有第四个旋钮——**向量化**:一次从内存搬 4 个 fp32(128 bit),好比一辆车一次拉 4 份货而不是 1 份。
这四件事互相耦合,所以“调参”本质上是一场小实验:遍历组合,找出最优。
'''))

NB.code(TRITON_HEADER, "✅ 环境自检:与上一课相同,先设好 KMP 与 ptxas,确认 triton 可编译。")

NB.md("## 2️⃣ 四大旋钮逐个拆解 🔧",
D('''
| 旋钮 | 调的是什么 | 太小的后果 | 太大的后果 | 一般经验 |
|------|-----------|------------|------------|----------|
| `num_warps` | 一个 CTA 的线程束数 | 并行度不足,SM 喂不饱 | 寄存器溢出、同步开销 | GEMM 常用 4~8 |
| `num_stages` | K 循环流水线深度 | 访存与计算串行 | 寄存器/共享内存被占满 | GEMM 常用 2~4 |
| `BLOCK_M/N` | 输出 tile 尺寸 | 数据复用率低 | 共享内存不够,launch 失败 | 64~128 |
| `BLOCK_K` | K 方向分块宽度 | 粒度细、循环多 | 中间 tile 太大 | 32~64 |
| 向量化 | 单条访存指令搬运的字节 | 访存指令多 | (128 bit 已到上限) | fp32 用 4 元素 |

注意这些参数在编译期(`tl.constexpr`)就固定了,所以**换一个参数就要重新编译一次**——
这也是为什么下面要做“扫描”而不是一个个手改。
'''))

NB.md("## 3️⃣ 先写好“温度计”:科学计时 ⏱️",
D('''
GPU 的 kernel 是**异步**的:调用返回时内核可能还没执行完,甚至还没启动。如果直接用
`time.perf_counter` 计时,测的往往是“排队时间”。科学的做法是:

1. **warmup**:先跑几遍,让 JIT 编译、缓存、驱动预热完成;
2. **同步**:每轮结束后 `torch.cuda.synchronize()` 等内核真正结束;
3. **多次取平均**:单次抖动大,至少 10 次取均值。

下面就是本课通用的计时函数(和后续几课共用):
'''))

NB.code(BENCH, "🔍 注意 `torch.cuda.synchronize()` 前后的位置:它保证计时区间覆盖“所有已提交的内核”。")

NB.md("## 4️⃣ 基准 GEMM kernel 📐",
D('''
先写一个最简的 GEMM(矩阵乘)。它把输出 `C` 切成 `BLOCK_M × BLOCK_N` 的 tile,每个 tile 交给一个
program;K 方向再用 `BLOCK_K` 分块循环,`tl.dot` 做块级矩阵乘,`acc` 用 fp32 累加避免精度漂移。
这是后续所有调参实验的“试验台”。
'''))

NB.code(GEMM, "🎯 这个 kernel 是教材版:没有 swizzle、没有 group 排序,但足够用来观察四个旋钮的作用。")

NB.code(D('''
# 正确性对拍:与 torch.matmul(fp32) 比较
M, N, K = 256, 256, 256
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = matmul(a, b)
ref = (a.float() @ b.float())
print(f"max err = {(c - ref).abs().max().item():.3e}  (fp16 输入,误差 ~1e-2 量级正常)")
'''), "✅ fp16 输入 + fp32 累加,与 torch 相比误差应在 1e-2 以内(这是 fp16 本身的精度上限)。")

NB.md("## 5️⃣ 扫描 num_warps × num_stages 📈",
D('''
固定 `BLOCK_M=BLOCK_N=64, BLOCK_K=32`,遍历 `num_warps ∈ {1,2,4,8}` 与
`num_stages ∈ {1,2,3,4}` 共 **16 个配置**,每个配置跑 10 次取平均。每换一个配置都会触发一次
编译(前几课见过的 ~0.1s 量级),所以这一节是扫描的主力。

为了不让单 cell 超过 20 秒,我们把矩阵固定在 **256×256**,并复用 `bench` 的默认 10 次迭代。
'''))

NB.code(D('''
M = 256
a = torch.randn(M, M, device="cuda", dtype=torch.float16)
b = torch.randn(M, M, device="cuda", dtype=torch.float16)

configs = [(nw, ns) for nw in (1, 2, 4, 8) for ns in (1, 2, 3, 4)]
times = []
for nw, ns in configs:
    t = bench(lambda: matmul(a, b, 64, 64, 32, nw, ns))
    times.append(t)
    print(f"num_warps={nw}  num_stages={ns} : {t:7.3f} ms  {2*M**3/(t*1e6):6.1f} GFLOPS")
'''), "⏱️ 你会看到 num_warps 太少(1)明显偏慢,num_warps 太大(8)在 BLOCK=64 时也开始吃亏——因为块内线程数超过了数据量。")

NB.code(D('''
# num_warps × time 曲线,颜色区分 num_stages
import numpy as np
wlist = [1, 2, 4, 8]
tmap = np.array(times).reshape(4, 4)   # 行=stages, 列=warps
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for i, ns in enumerate((1, 2, 3, 4)):
    ax.plot(wlist, tmap[i], "o-", label=f"num_stages={ns}")
ax.set_xticks(wlist)
ax.set_xlabel("num_warps")
ax.set_ylabel("耗时(ms)")
ax.set_title("num_warps × num_stages 扫描曲线(BLOCK=64×64×32, M=256)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
'''), "🎨 注意观察:对同一个 num_stages,曲线存在“谷值”——这就是这个 BLOCK 尺寸下的 warps 甜点。")

NB.md("## 6️⃣ 扫描 BLOCK 尺寸:热力图 🌡️",
D('''
第二个维度是 **tile 尺寸**。固定 `num_warps=4, num_stages=3`,遍历
`BLOCK_M × BLOCK_N ∈ {64, 128} × {64, 128}`,并把结果画成吞吐热力图。

> 提示:tile 越大,一个 program 覆盖的输出越大,K 方向循环里对 A/B 的**复用率**越高,
> 但共享内存占用也越大;太大(如 256×256)在保守 GPU 上可能直接编译失败。
'''))

NB.code(D('''
M = 256
a = torch.randn(M, M, device="cuda", dtype=torch.float16)
b = torch.randn(M, M, device="cuda", dtype=torch.float16)

bm_list = [64, 128]
bn_list = [64, 128]
gflops = np.zeros((len(bm_list), len(bn_list)))
for i, bm in enumerate(bm_list):
    for j, bn in enumerate(bn_list):
        t = bench(lambda: matmul(a, b, bm, bn, 32, 4, 3))
        gflops[i, j] = 2 * M ** 3 / (t * 1e6)
        print(f"BLOCK_M={bm:3d} BLOCK_N={bn:3d} : {t:7.3f} ms  {gflops[i, j]:6.1f} GFLOPS")

fig, ax = plt.subplots(figsize=(6.5, 5))
im = sns.heatmap(gflops, annot=True, fmt=".0f", cmap="YlOrRd",
                 xticklabels=[f"BN={n}" for n in bn_list],
                 yticklabels=[f"BM={m}" for m in bm_list],
                 cbar_kws=dict(label="GFLOPS"), ax=ax)
ax.set_title("BLOCK_M × BLOCK_N 吞吐热力图(warps=4, stages=3)")
ax.set_xlabel("BLOCK_N"); ax.set_ylabel("BLOCK_M")
plt.tight_layout()
'''), "📊 热力图让你一眼看到甜点区域:通常 128×128 或 64×128 会比 64×64 明显更快(复用率上去了)。")

NB.md("## 7️⃣ 最优配置挑战 cuBLAS 🥊",
D('''
把所有测过的配置放在一起找最优,再和 torch 的 `matmul`(底层是 cuBLAS)正面比一次。
cuBLAS 是 NVIDIA 手写、调了十几年的库,还做了 tile swizzle、双缓冲、特化指令调度。
我们的 40 行教材版 kernel 通常落在 cuBLAS 的 60%~120% 之间;本次矩阵只有 256,启动开销占比高,结果偏慢,属正常现象。
'''))

NB.code(D('''
M = 256
a = torch.randn(M, M, device="cuda", dtype=torch.float16)
b = torch.randn(M, M, device="cuda", dtype=torch.float16)

# 在所有 16+4 个配置里找最优
best_t, best_cfg = 1e9, None
for nw, ns in configs:
    t = bench(lambda: matmul(a, b, 64, 64, 32, nw, ns))
    if t < best_t:
        best_t, best_cfg = t, (nw, ns)
for bm in (64, 128):
    for bn in (64, 128):
        t = bench(lambda: matmul(a, b, bm, bn, 32, 4, 3))
        if t < best_t:
            best_t, best_cfg = t, (f"BM={bm}", f"BN={bn}")

t_torch = bench(lambda: torch.matmul(a, b))
print(f"triton 最佳: {best_t:7.3f} ms  (配置 {best_cfg})")
print(f"cuBLAS   : {t_torch:7.3f} ms")
print(f"triton / cuBLAS = {best_t / t_torch * 100:.0f}%")

names = ["triton 最佳", "cuBLAS"]
vals = [best_t, t_torch]
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(names, vals, color=["#4C78A8", "#E45756"], width=0.5)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f} ms", ha="center", va="bottom")
ax.set_ylabel("耗时(ms)")
ax.set_title(f"{M}×{M} fp16 矩阵乘:教材版 triton vs cuBLAS")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
'''), "🎯 诚实的结论:40 行教材版能到 cuBLAS 的 60%~100%,取决于机器;要超越它需要 swizzle、persistent kernel 等进阶技巧。")

NB.md("## 8️⃣ triton.autotune:让编译器自己调 🤖",
D('''
手扫太累?triton 内置 `@triton.autotune`:你给出一组候选 `Config`,它在**第一次调用**时逐个试跑、
记住最优配置,后续调用直接使用。它的 `key=["n"]` 指定“按什么维度去区分场景”(比如矩阵规模)。
'''))

NB.code(D('''
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["M", "N"],
)
@triton.jit
def matmul_auto(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)
'''), "🎛️ 候选里只放了 3 个配置(演示),实际项目会放十几二十个。")

NB.code(D('''
c_auto = torch.empty_like(a).float()
grid = (triton.cdiv(M, 64) * triton.cdiv(M, 64),)
matmul_auto[grid](a, b, c_auto, M, M, M,
                  a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                  c_auto.stride(0), c_auto.stride(1), BLOCK_K=32)
torch.cuda.synchronize()
print("autotune 选中的最优配置:", matmul_auto.best_config)
print(f"max err = {(c_auto - (a.float() @ b.float())).abs().max().item():.3e}")
'''), "✅ 第一次调用会自动试跑 3 个候选、选出最优;之后每次调用直接用最优配置。")

NB.md("## 9️⃣ 配套 Streamlit 演示:滑杆实时扫描吞吐 🎛️",
D('''
运行同目录下的 `app_57_triton_tuning.py`,**拖动 num_warps / num_stages / BLOCK 滑杆**实测
单个配置,或点“扫描全网格”画出 `num_warps × num_stages` 的吞吐热力图:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_57_triton_tuning.py
```

浏览器打开 **http://localhost:8501**。建议先固定 BLOCK,扫一次全网格看热力图的“甜点”在哪,
再把滑杆移过去验证。完整源码如下(与同目录 `app_57_triton_tuning.py` 一字不差):
'''))

NB.code("%%writefile app_57_triton_tuning.py\n" + APP_57, "📜 app_57_triton_tuning.py 完整源码:notebook 与 app 共用同一个 `matmul_kernel` 与计时函数。")

wrapup(NB,
    summary=[
        "四个旋钮:num_warps(线程束数)、num_stages(流水线深度)、BLOCK(tile 尺寸)、向量化(访存宽度)",
        "GPU 计时三件套:warmup → torch.cuda.synchronize() → 多次平均,缺一不可",
        "调参就是网格扫描:固定一个维度、扫另一个维度,画曲线/热力图找甜点",
        "教材版 kernel 通常落在 cuBLAS 的 60%~120%(矩阵越小启动开销占比越高,越吃亏);想超越需要 swizzle / persistent / 特化指令等进阶技巧",
        "triton.autotune 用一组候选 Config 在首次调用时自动选优,把选择权交给编译器",
    ],
    practice=[
        "把第 6 节热力图扩展到 BLOCK_K ∈ {32, 64},观察 K 方向分块的影响",
        "在 num_warps=4 时给 num_stages 从 1 扫到 6,画出 stages-time 曲线,找拐点",
        "给 matmul_auto 加一个 BLOCK=256×256 的候选,观察它会不会被选中/是否编译失败",
        "试试 M=512 时各配置的相对排名是否变化——调参结论是依赖规模的",
    ],
    links=[
        ("Triton 矩阵乘官方教程(进阶版含 swizzle)", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
        ("torch.utils.benchmark 教程", "https://pytorch.org/tutorials/recipes/recipes/benchmark.html"),
        ("CUTLASS(NVIDIA 手写 GEMM 的标杆)", "https://github.com/NVIDIA/cutlass"),
    ])

NB.save(str(Path(CH09) / "57_triton_tuning.ipynb"))

app_path = Path(CH09) / "app_57_triton_tuning.py"
app_path.write_text(APP_57 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
