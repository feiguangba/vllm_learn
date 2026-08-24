# -*- coding: utf-8 -*-
"""生成 54_triton_gemm.ipynb 与 app_54_triton_gemm.py"""
from pathlib import Path
from triton_helpers import D, new_nb, chapter_cover, wrapup, CH09
from triton_helpers import ENV_HEADER, PLT_STYLE, BENCH, GEMM_KERNEL

APP_FILE = "app_54_triton_gemm.py"

APP_54 = D('''
# -*- coding: utf-8 -*-
# app_54_triton_gemm.py — Triton GEMM:块尺寸 / num_warps 实时看吞吐 📐
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch, triton, triton.language as tl

st.set_page_config(page_title="📐 54 · Triton GEMM", layout="wide")
st.title("📐 第 54 课 · Triton GEMM:从 naive 到 tile,用 tl.dot 打满张量核")

st.markdown("""
矩阵乘法 `C = A@B` 是深度学习的心脏。Triton 把它拆成一块块 **BM×BN 的输出 tile**,每个 program
沿 K 方向逐步 `tl.dot` 累加。**BLOCK_M/N/K** 决定 tile 形状,**num_warps** 决定块内并行度,
**num_stages** 决定流水线深度——这些旋钮直接决定吞吐。下方拖动这些参数,在 RTX 5060 上实时测
Triton GEMM 的 TFLOPS,并与 cuBLAS(torch.matmul)对比。
""")

@triton.jit
def matmul_kernel(A, B, C, M, N, K, sm, ak, bk, bn, cm, cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_am[:, None] * sm + offs_k[None, :] * ak
    b_ptrs = B + offs_k[:, None] * bk + offs_bn[None, :] * bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * ak; b_ptrs += BLOCK_K * bk
    c_ptrs = C + offs_am[:, None] * cm + offs_bn[None, :] * cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def bench(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

@st.cache_data(show_spinner=False)
def gemm_cfg(size, bm, bn, bk, nw, ns):
    M = N = K = size
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    def tfn():
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        matmul_kernel[grid](A, B, C, M, N, K, K, 1, N, 1, N, 1,
                            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                            num_warps=nw, num_stages=ns)
    try:
        t0 = time.perf_counter(); tfn(); torch.cuda.synchronize()
        t = bench(tfn, warmup=3, iters=15)
    except Exception:
        return None
    def cfn(): torch.matmul(A, B, out=C)
    tc = bench(cfn, warmup=3, iters=15)
    flop = 2 * M * N * K
    return t, tc, flop / (t / 1000) / 1e12, flop / (tc / 1000) / 1e12

with st.sidebar:
    st.header("🎛️ 参数")
    size = st.select_slider("矩阵大小 (M=N=K)", options=[1024, 2048, 4096], value=2048)
    bm = st.select_slider("BLOCK_M", options=[32, 64, 128, 256], value=128)
    bn = st.select_slider("BLOCK_N", options=[32, 64, 128, 256], value=128)
    bk = st.select_slider("BLOCK_K", options=[16, 32, 64], value=32)
    nw = st.select_slider("num_warps", options=[1, 2, 4, 8], value=4)
    ns = st.select_slider("num_stages", options=[1, 2, 3, 4], value=2)
    st.caption("num_warps=块内 warp 数;num_stages=循环流水线深度。")

res = gemm_cfg(size, bm, bn, bk, nw, ns)
if res is None:
    st.error("该配置编译失败(可能寄存器/块过大),请换一组参数。")
    st.stop()
t, tc, tf, cf = res
c1, c2, c3, c4 = st.columns(4)
c1.metric("Triton 吞吐", f"{tf:.1f} TFLOPS")
c2.metric("cuBLAS 吞吐", f"{cf:.1f} TFLOPS")
c3.metric("Triton 耗时", f"{t:.3f} ms")
c4.metric("cuBLAS 耗时", f"{tc:.3f} ms")
st.caption(f"M=N=K={size} | BLOCK=({bm},{bn},{bk}) | warps={nw} | stages={ns}")

# ---------- 对比柱状图 ----------
fig = go.Figure(go.Bar(x=["Triton", "cuBLAS"], y=[tf, cf],
                       marker_color=["#4C78A8", "#E45756"],
                       text=[f"{tf:.1f}", f"{cf:.1f}"], textposition="outside"))
fig.update_layout(title=f"GEMM {size}³ 吞吐(TFLOPS)", yaxis_title="TFLOPS",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
> 💡 **结论**:块越大,单块做矩阵乘的算术强度越高、越容易打满张量核;但块太大会挤占寄存器、
> 降低占用率。`num_stages` 通过流水线预取隐藏访存延迟,`num_warps` 平衡块内并行与资源占用。
> 用 Triton 写好 tile GEMM,完全能达到与 cuBLAS 相当的吞吐——这就是“写 kernel 不写 CUDA”的底气。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 9 章 · 第 54 课配套演示")
''')

NB = new_nb("第 54 课 · Triton GEMM",
            subtitle="用 tl.dot 写矩阵乘法,从 naive 到 tile 版,并在 RTX 5060 上与 cuBLAS 同台竞技",
            emoji="📐")

chapter_cover(NB,
    objectives=[
        "理解矩阵乘法的 tile 分解:输出按 BM×BN 切块,沿 K 方向归约",
        "掌握 tl.dot 的用法与累加器语义(acc = tl.dot(a, b, acc))",
        "会用 num_warps / num_stages 两个旋钮调优 Triton GEMM",
        "在 GPU 上实测 Triton GEMM 与 cuBLAS(torch.matmul)的吞吐对比",
        "画出性能随 block 大小变化的曲线,找到甜点区间",
    ],
    toc=[
        ("直觉:矩阵乘是最大的算术强度", "为什么 GEMM 是深度学习的心脏"),
        ("从 naive 到 tile 分解", "三重循环 → 输出 tile × K 归约"),
        ("tl.dot:块级矩阵乘", "kernel 源码逐行解读 + 正确性验证"),
        ("num_warps / num_stages 调参", "块内并行度与流水线深度的权衡"),
        ("与 cuBLAS 同台竞技", "RTX 5060 上 Triton vs torch.matmul 实测"),
        ("性能随 block 大小变化曲线", "matplotlib 曲线 + plotly 交互"),
        ("配套 Streamlit 演示", "app_54_triton_gemm.py:拖块尺寸/num_warps 看吞吐"),
    ],
    links=[
        ("Triton 03-matrix-multiplication 教程", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
        ("cuBLAS 文档", "https://docs.nvidia.com/cuda/cublas/"),
        ("FlashAttention-3(GEMM 的极致形态)", "https://arxiv.org/abs/2407.08608"),
    ])

NB.md("## 1️⃣ 直觉:矩阵乘是最大的算术强度 🏭",
D('''
`C = A@B`,其中 A 是 M×K、B 是 K×N、C 是 M×N。它做 **2·M·N·K** 次浮点运算——神经网络里
几乎全部计算量都来自这种矩阵乘(线性层、attention 的 QK^T 和 PV 都是 GEMM)。

和向量加法那种“访存受限”不同,GEMM 是 **计算受限(compute-bound)** 的:只要数据在寄存器里,
一次乘加能复用多次,算术强度极高。所以打满 GPU 的**张量核(Tensor Core)**是关键。

Triton 的思路:把输出 C 切成一块块 **BM×BN** 的 tile,每个 program 负责算一块。对每一块,沿着
K 方向,反复读 A 的一个 **BM×BK** 子块、B 的一个 **BK×BN** 子块,用 `tl.dot` 做块级乘加并累加。
'''))

NB.code(ENV_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed、确认 triton / torch / CUDA 就绪。")

NB.md("## 2️⃣ 从 naive 到 tile 分解 📐",
D('''
最朴素的三重循环(naive)长这样(概念示意):

```
for i in range(M):        # 每一行
  for j in range(N):      # 每一列
    for k in range(K):    # 累加
      C[i,j] += A[i,k]*B[k,j]
```

它每次只算 C 的一个元素,访存效率极低。**tile 分解**把它重排成“块”的层次:

```
for i in range(0, M, BM):     # 输出行块
  for j in range(0, N, BN):   # 输出列块
    acc[BM,BN] = 0
    for k in range(0, K, BK): # K 归约
      acc += A[i:i+BM, k:k+BK] @ B[k:k+BK, j:j+BN]
    C[i:i+BM, j:j+BN] = acc
```

现在最内层是 **BM×BK 乘 BK×BN → BM×BN** 的块级矩阵乘,一次把一大块数据放进寄存器反复复用,
这正是张量核擅长的形态。外层两个循环就是 `tl.program_id(0)` 和 `program_id(1)`。
'''))

NB.code(PLT_STYLE, "🎨 先套用统一样式头(浅色网格、大图、高 DPI),下面的示意图都用它。")

NB.code(D('''
# 画 tile GEMM 的示意图:A 切成 BM×BK 块,B 切成 BK×BN 块,累加出 BM×BN
fig, ax = plt.subplots(figsize=(8.5, 5))
BM, BK, BN = 4, 4, 4
# A
ax.add_patch(plt.Rectangle((0, 1), 3.5, 3.5, fc="#DCE6F2", ec="#4C78A8", lw=1.5))
ax.add_patch(plt.Rectangle((0, 1), 3.5, 3.5, fc="none", ec="orange", lw=2.5))
ax.text(1.75, 2.75, "A 的\\nBM×BK 块", ha="center", va="center", fontsize=9)
ax.text(1.75, 5.0, "A (M×K)", ha="center", fontsize=10)
# B
ax.add_patch(plt.Rectangle((6.2, 1), 3.5, 3.5, fc="#E7E2F6", ec="#7A6FB0", lw=1.5))
ax.add_patch(plt.Rectangle((6.2, 1), 3.5, 3.5, fc="none", ec="orange", lw=2.5))
ax.text(7.95, 2.75, "B 的\\nBK×BN 块", ha="center", va="center", fontsize=9)
ax.text(7.95, 5.0, "B (K×N)", ha="center", fontsize=10)
# C
ax.add_patch(plt.Rectangle((3.2, -5.2), 3.5, 3.5, fc="#F6E3E3", ec="#E45756", lw=1.5))
ax.text(4.95, -3.45, "C 输出\\nBM×BN 累加器", ha="center", va="center", fontsize=9)
ax.text(4.95, -1.2, "C (M×N)", ha="center", fontsize=10)
# 箭头
ax.annotate("", xy=(4.0, 0.9), xytext=(3.5, 2.0), arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate("", xy=(4.0, -4.5), xytext=(5.0, -0.9), arrowprops=dict(arrowstyle="->", color="gray"))
ax.text(4.4, -0.4, "tl.dot 累加", fontsize=9, color="gray")
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-6, 5.8); ax.axis("off")
ax.set_title("tile GEMM:BM×BK 与 BK×BN 的块乘加 → BM×BN 累加器", fontsize=12)
plt.tight_layout(); plt.show()
'''), "🎨 每个 program 负责一块 BM×BN 输出;沿 K 方向反复 tl.dot,这就是 GEMM 的 tile 分解。")

NB.md("## 3️⃣ tl.dot:块级矩阵乘 🧱",
D('''
下面是最简 tile GEMM kernel。核心一行是 `acc = tl.dot(a, b, acc)`——第三个参数 `acc` 是累加器,
`tl.dot` 会返回 `acc + a@b`,自动完成跨 K 的累加。`BLOCK_M/N/K` 都是 `tl.constexpr`,编译器据此
把 `tl.dot` 调度到张量核。
'''))

NB.code(GEMM_KERNEL, "**逐行读**:program_id(0/1) 定位输出 tile;`tl.arange` 生成块内下标;K 循环里读 A、B 子块,`tl.dot` 累加;最后写回 C。")

NB.code(D('''
M = N = K = 2048
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.empty(M, N, device="cuda", dtype=torch.float16)

grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
matmul_kernel[grid](A, B, C, M, N, K, K, 1, N, 1, N, 1,
                    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8)
torch.cuda.synchronize()

ref = (A.float() @ B.float()).half()
err = (C.float() - ref.float()).abs().max().item()
rel = err / ref.float().abs().max().item()
print(f"M=N=K=2048, fp16: max abs err = {err:.3e}  rel = {rel:.2e}")
'''), "✅ 与 `A.float() @ B.float()` 对照。fp16 张量核的误差在 1e-3 量级,属于预期精度(若想要精确 fp32,可给 tl.dot 传 `input_precision='ieee'`)。")

NB.md("## 4️⃣ num_warps / num_stages 调参 🎛️",
D('''
`tl.dot` 之外,还有两个影响吞吐的旋钮:

- **num_warps**:每个 program 用多少个 warp(32 线程)执行块内计算。块内并行度低则喂不饱张量核,
  太高则寄存器不够、占用率下降;
- **num_stages**:K 循环的**流水线深度**。编译器把后续块的数据提前预取到共享内存/寄存器,
  隐藏访存延迟;stages 越多越能隐藏延迟,但占更多显存。

它们是“块内并行 × 流水线 × 寄存器占用”三者之间的平衡。下面实测不同配置的吞吐。
'''))

NB.code(BENCH, "复用 warmup + synchronize 的计时函数,确保编译开销不计入结果。")

NB.md("## 5️⃣ 与 cuBLAS 同台竞技 ⚔️",
D('''
在 RTX 5060 上对比 **cuBLAS(torch.matmul)** 与我们手写的 Triton GEMM,都用 fp16。衡量指标是
**TFLOPS = 2·M·N·K / 耗时**。
'''))

NB.code(D('''
M = N = K = 2048
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.empty(M, N, device="cuda", dtype=torch.float16)
flop = 2 * M * N * K

def cublas(): torch.matmul(A, B, out=C)
tc = bench(cublas, warmup=5, iters=20)
print(f"cuBLAS: {tc:.3f} ms = {flop/(tc/1000)/1e12:.1f} TFLOPS")

def mk(bm, bn, bk, w, s):
    def f():
        matmul_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](A, B, C, M, N, K, K, 1, N, 1, N, 1,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s)
    return f

cfgs = [(64, 64, 32, 4, 2), (128, 128, 32, 4, 2), (128, 128, 32, 8, 2),
        (128, 128, 64, 8, 3), (256, 128, 64, 8, 3), (64, 128, 64, 4, 3)]
names = ["64x64x32 w4 s2", "128x128x32 w4 s2", "128x128x32 w8 s2",
         "128x128x64 w8 s3", "256x128x64 w8 s3", "64x128x64 w4 s3"]
for nm, (bm, bn, bk, w, s) in zip(names, cfgs):
    t = bench(mk(bm, bn, bk, w, s), warmup=3, iters=15)
    print(f"Triton {nm:16s}: {t:.3f} ms = {flop/(t/1000)/1e12:6.1f} TFLOPS")
'''), "🚀 记录多种 tile/num_warps/num_stages 组合的吞吐,下面画成对比图。")

NB.md("## 6️⃣ 性能随 block 大小变化曲线 📈",
D('''
固定 num_warps/num_stages,把 **BLOCK_M = BLOCK_N 从 32 扫到 256**,看吞吐如何变化:块太小时
单块算术强度不足、块数太多调度开销大;块足够大后吞吐进入平台,但过大又会挤爆寄存器回落。
'''))

NB.code(D('''
sizes = [32, 64, 128, 256]
tf_list, name_list = [], []
for bm in sizes:
    t = bench(mk(bm, bm, 32, 4, 2), warmup=3, iters=15)
    tf_list.append(flop / (t / 1000) / 1e12)
    name_list.append(str(bm))
    print(f"BLOCK_M=N={bm:3d}: {t:.3f} ms = {tf_list[-1]:6.1f} TFLOPS")
'''), "🔍 记录吞吐曲线数据,下面绘制。")

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(sizes, tf_list, "o-", color="#4C78A8", lw=2.5, label="Triton")
ax.axhline(flop / (tc / 1000) / 1e12, color="#E45756", ls="--", lw=2, label="cuBLAS 参考")
ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel("BLOCK_M = BLOCK_N(tile 大小)"); ax.set_ylabel("TFLOPS")
ax.set_title("Triton GEMM 吞吐随 tile 大小变化(RTX 5060, 2048³, fp16)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
'''), "📊 曲线先升后平(或略回落):找到“既够大、又不撑爆寄存器”的甜点区间,正是调 tile 的意义。")

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
fig.add_trace(go.Bar(x=names, y=[flop/(bench(mk(*c), warmup=2, iters=10)/1000)/1e12 for c in cfgs],
                     marker_color="#4C78A8", name="Triton"))
fig.add_trace(go.Bar(x=names, y=[flop/(tc/1000)/1e12]*len(names),
                     marker_color="#E45756", name="cuBLAS"))
fig.update_layout(barmode="group", title="不同 Triton 配置 vs cuBLAS 的吞吐(TFLOPS)",
                  yaxis_title="TFLOPS", xaxis_tickangle=30, height=400,
                  margin=dict(l=10, r=10, t=50, b=10))
fig
'''), "📊 plotly 交互柱状图:并排对比各配置与 cuBLAS,悬停看具体数值。")

NB.md("## 7️⃣ 配套 Streamlit 演示:拖块尺寸看吞吐 🎛️",
D('''
运行同目录下的 `app_54_triton_gemm.py`,拖动 **矩阵大小、BLOCK_M/N/K、num_warps、num_stages**,
在 RTX 5060 上实时跑 Triton GEMM、显示 TFLOPS,并与 cuBLAS 对比:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_54_triton_gemm.py
```

浏览器打开 **http://localhost:8501**。建议先固定 block 大小,只动 num_warps 看吞吐变化,再反过来
固定 num_warps 只动 block 大小。完整源码如下(与同目录 `app_54_triton_gemm.py` 一字不差):
'''))

NB.code(f"%%writefile {APP_FILE}\n" + APP_54, "📜 这就是 app_54_triton_gemm.py 的完整源码,用 st.cache_data 缓存每种配置的基准,拖拽实时响应。")

wrapup(NB,
    summary=[
        "GEMM 是计算受限的:关键是打满张量核,而不是省访存",
        "tile 分解:输出按 BM×BN 切块,每个 program 沿 K 方向 tl.dot 累加",
        "acc = tl.dot(a, b, acc) 一行完成块级矩阵乘与跨 K 累加",
        "num_warps 调块内并行度,num_stages 调流水线深度,两者平衡寄存器占用",
        "精心调的 Triton tile GEMM 能达到与 cuBLAS 相当的吞吐",
    ],
    practice=[
        "把 BLOCK_K 从 32 改成 64/128,观察吞吐与 num_stages 的配合",
        "给 tl.dot 加 input_precision='ieee',对比 fp32 精确模式的精度与吞吐",
        "试 num_stages=4/5,观察流水线加深后吞吐与显存占用的变化",
        "把矩阵改成非方阵(如 4096×1024×2048),验证 mask 边界处理是否正确",
    ],
    links=[
        ("Triton 03-matrix-multiplication 教程", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
        ("cuBLAS 文档", "https://docs.nvidia.com/cuda/cublas/"),
    ])

NB.save(str(Path(CH09) / "54_triton_gemm.ipynb"))

app_path = Path(CH09) / APP_FILE
app_path.write_text(APP_54 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
