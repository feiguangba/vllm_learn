# -*- coding: utf-8 -*-
"""生成 55_triton_flashattn.ipynb 与 app_55_triton_fa.py"""
from pathlib import Path
from triton_helpers import D, new_nb, chapter_cover, wrapup, CH09
from triton_helpers import ENV_HEADER, PLT_STYLE, BENCH, FA_KERNEL

APP_FILE = "app_55_triton_fa.py"

APP_55 = D('''
# -*- coding: utf-8 -*-
# app_55_triton_fa.py — Triton FlashAttention:块大小/序列长度实时对比 ⚡
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch, triton, triton.language as tl

st.set_page_config(page_title="⚡ 55 · Triton FlashAttention", layout="wide")
st.title("⚡ 第 55 课 · Triton FlashAttention:分块 + 在线 softmax")

LOG2E = 1.4426950408889634

@triton.jit
def fa_kernel(Q, K, V, O, sm_scale, LOG2E, M, N, D,
              sqm, sqk, skn, skk, svn, svk, som, sok,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=(offs_m[:, None] < M), other=0.0)
    q = (q * sm_scale).to(tl.float16)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for start_n in range(0, tl.cdiv(N, BLOCK_N)):
        offs_nn = start_n * BLOCK_N + offs_n
        mask_k = (offs_d[:, None] < D) & (offs_nn[None, :] < N)
        mask_v = (offs_nn[:, None] < N) & (offs_d[None, :] < D)
        mask_qk = (offs_m[:, None] < M) & (offs_nn[None, :] < N)
        k = tl.load(K + offs_nn[None, :] * skn + offs_d[:, None] * skk, mask=mask_k, other=0.0)
        v = tl.load(V + offs_nn[:, None] * svn + offs_d[None, :] * svk, mask=mask_v, other=0.0)
        qk = tl.dot(q, k)
        qk = tl.where(mask_qk, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2((qk - m_ij[:, None]) * LOG2E)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2((m_i - m_ij) * LOG2E)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * som + offs_d[None, :] * sok,
             acc.to(O.dtype.element_ty), mask=(offs_m[:, None] < M))

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
def fa_bench(N, bm, bn, nw):
    D = 64
    torch.manual_seed(0)
    q = torch.randn(N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(N, D, device="cuda", dtype=torch.float16)
    o = torch.zeros(N, D, device="cuda", dtype=torch.float16)
    def tfn():
        fa_kernel[(triton.cdiv(N, bm),)](q, k, v, o, 1.0 / (D ** 0.5), LOG2E, N, N, D,
            D, 1, D, 1, D, 1, D, 1, BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=D, num_warps=nw)
    try:
        t0 = time.perf_counter(); tfn(); torch.cuda.synchronize()
        t = bench(tfn, warmup=3, iters=15)
    except Exception:
        return None
    qb, kb, vb = q[None, None], k[None, None], v[None, None]
    ts = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qb, kb, vb),
               warmup=3, iters=15)
    flop = 4 * N * N * D
    return t, ts, flop / (t / 1000) / 1e12, flop / (ts / 1000) / 1e12

with st.sidebar:
    st.header("🎛️ 参数")
    N = st.select_slider("序列长度 N", options=[1024, 2048, 4096, 8192], value=2048)
    bm = st.select_slider("BLOCK_M(Q 块)", options=[64, 128], value=128)
    bn = st.select_slider("BLOCK_N(K/V 块)", options=[64, 128], value=64)
    nw = st.select_slider("num_warps", options=[4, 8], value=8)
    st.caption("BLOCK_M 管 Q 的行块,BLOCK_N 管 K/V 的列块;序列越长,分块扫描的轮数越多。")

res = fa_bench(N, bm, bn, nw)
if res is None:
    st.error("该配置编译失败,请换一组参数。")
    st.stop()
t, ts, tf, sf = res
# 内存对比
naive_mb = (2 * N * N * 2) / 1e6      # S 与 P 各 N×N fp16
flash_mb = (4 * N * 64 * 2) / 1e6     # Q/K/V/O
c1, c2, c3, c4 = st.columns(4)
c1.metric("Triton FA 吞吐", f"{tf:.0f} TFLOPS")
c2.metric("torch SDPA 吞吐", f"{sf:.0f} TFLOPS")
c3.metric("Triton 耗时", f"{t:.3f} ms")
c4.metric("naive 峰值显存(S+P)", f"{naive_mb:.0f} MB")

# ---------- 吞吐对比 ----------
fig = go.Figure(go.Bar(x=["Triton FA", "torch SDPA"], y=[tf, sf],
                       marker_color=["#4C78A8", "#E45756"],
                       text=[f"{tf:.0f}", f"{sf:.0f}"], textposition="outside"))
fig.update_layout(title=f"FlashAttention 吞吐(TFLOPS), N={N}, d=64, fp16",
                  yaxis_title="TFLOPS", height=340, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------- 内存节省曲线 ----------
ns = [1024, 2048, 4096, 8192, 16384]
naive = [2 * n * n * 2 / 1e6 for n in ns]
flash = [4 * n * 64 * 2 / 1e6 for n in ns]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=naive, mode="lines+markers", name="naive(S+P 落盘)", line=dict(color="#E45756", width=3)))
fig2.add_trace(go.Scatter(x=ns, y=flash, mode="lines+markers", name="flash(不落 S/P)", line=dict(color="#4C78A8", width=3)))
fig2.update_layout(xaxis=dict(type="log", title="序列长度 N"),
                   yaxis=dict(type="log", title="显存 MB"),
                   height=360, margin=dict(l=10, r=10, t=40, b=10),
                   title="峰值显存:naive O(N²) vs flash O(N)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:FlashAttention 用 **分块(tiling)+ 在线 softmax(online softmax)** 让 S/P 这两个
> N×N 大矩阵永不落盘,把峰值显存从 O(N²) 压到 O(N),并靠 `exp2 + LOG2E` 在 Triton 里精确复现
> 标准 softmax。序列越长,节省越夸张(N=8192 时省几十倍)。这就是 vLLM 在长序列场景提速的关键。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 9 章 · 第 55 课配套演示")
''')

NB = new_nb("第 55 课 · Triton FlashAttention",
            subtitle="用 Triton 手写分块 + 在线 softmax 的注意力,把 O(N²) 显存压回 O(N),并实测与 torch 原生 attention 对比",
            emoji="⚡")

chapter_cover(NB,
    objectives=[
        "复述 attention 三步与 S/P 两个 O(N²) 大张量是访存/显存瓶颈",
        "理解分块(tiling)+ 在线 softmax(online softmax)如何避免 S/P 落盘",
        "掌握在 tl 中做 block-wise softmax:running max m 与 running sum l 的更新",
        "写一个正确的 Triton FlashAttention kernel 并在 GPU 上验证",
        "实测 Triton FA 与 torch 原生 attention 的吞吐,并量化内存节省",
    ],
    toc=[
        ("直觉:图书馆不搬整本书", "attention 慢在访存,不在算力"),
        ("分块 + 在线 softmax 的钥匙", "running max m / running sum l 边走边校正"),
        ("Triton 中的 block-wise softmax", "fa_kernel 逐行解读 + exp2/LOG2E 技巧"),
        ("正确性验证", "与 torch 原生 attention 对照误差"),
        ("与 torch SDPA 实测对比", "RTX 5060 上 Triton FA vs 原生 attention"),
        ("内存节省分析", "O(N²) → O(N) 的公式与数值曲线"),
        ("配套 Streamlit 演示", "app_55_triton_fa.py:拖块大小/序列长度看吞吐"),
    ],
    links=[
        ("FlashAttention 论文", "https://arxiv.org/abs/2205.14135"),
        ("Triton 官方文档", "https://triton-lang.org"),
        ("Triton 融合注意力教程", "https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html"),
    ])

NB.md("## 1️⃣ 直觉:图书馆不搬整本书 📚",
D('''
标准 attention 对输入 $Q, K, V \\in \\mathbb{R}^{N \\times d}$ 做三步:

$$S = \\frac{QK^\\top}{\\sqrt{d}}, \\qquad P = \\text{softmax}(S), \\qquad O = PV$$

每一步都要**把 N×N 的中间矩阵 S、P 整个写进显存(HBM)、再读出来**。N 翻倍,S/P 涨到 4 倍——
这是**平方级**的开销。想象图书馆里每查一本书都要把整本搬上桌(即使只看一页):attention 的瓶颈
从来不是算了多少(算力),而是**搬了多少(访存)**。

**FlashAttention 的钥匙**就两件事,让 S/P 永不落盘:

1. **分块(tiling)**:把 K/V 沿 N 方向切成小块,一次只读一小块进片上(SRAM),算完即弃;
2. **在线 softmax(online softmax)**:softmax 的分母要“看完全部块”才知道,于是用**运行时最大值
   m** 与**运行时求和 l** 边走边校正,保证数学上等价。
'''))

NB.code(ENV_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed、确认 triton / torch / CUDA 就绪。")

NB.md("## 2️⃣ 分块 + 在线 softmax 的钥匙 🔑",
D('''
在线 softmax 维护两个标量状态:

- **m**(running max):目前已见分数的最大值;
- **l**(running sum):目前已见块的加权分母和。

对每个块 $S_j$:

$$
m_{\\text{new}} = \\max(m,\\, \\max(S_j)), \\qquad
l = l \\cdot e^{m - m_{\\text{new}}} + \\sum e^{S_j - m_{\\text{new}}}, \\qquad
O = O \\cdot e^{m - m_{\\text{new}}} + P_j V_j
$$

关键洞察:**不必等 softmax 求完再算 O,可以边扫边累积输出**,每块都用“旧 m 与当前块 max”
校正已累积的 l 与 O。全程 m、l、O 都住在寄存器/片上,不落 HBM。下面先用 torch 在 CPU 上
把这种“分块手写版”跑通,建立直觉(第 42/43 课有更细的推导)。
'''))

NB.code(D('''
def flash_chunked_torch(Q, K, V, block_M=64):
    """torch 手写分块 FlashAttention(CPU 演示):running max m + running sum l。"""
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)
    m = torch.full((B, H, N, 1), float("-inf"), device=Q.device)
    l = torch.zeros((B, H, N, 1), device=Q.device)
    scale = 1.0 / math.sqrt(d)
    for j in range(0, N, block_M):
        Kj = K[:, :, j:j + block_M, :]
        Vj = V[:, :, j:j + block_M, :]
        S = torch.einsum("bhnd,bhmd->bhnm", Q, Kj) * scale
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)
        P = torch.exp(S - m_new)
        l_new = l * torch.exp(m - m_new) + P.sum(dim=-1, keepdim=True)
        O = O * torch.exp(m - m_new) + torch.einsum("bhnm,bhmd->bhnd", P, Vj)
        m, l = m_new, l_new
    return O / l

B, H, N, d = 1, 4, 256, 64
q = torch.randn(B, H, N, d); k = torch.randn(B, H, N, d); v = torch.randn(B, H, N, d)
S = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(d)
P = torch.softmax(S, dim=-1)
ref = torch.einsum("bhnm,bhmd->bhnd", P, v)
out = flash_chunked_torch(q, k, v, block_M=64)
print("分块版与标准版最大误差:", (out - ref).abs().max().item())
'''), "✅ 分块版在 CPU 上先验证:与标准 attention 结果一致,这就是 online softmax 正确的证据。")

NB.md("## 3️⃣ Triton 中的 block-wise softmax 🧱",
D('''
现在把同样的逻辑写进 Triton kernel。与 torch 手写版的差别是:**Q 也被切块**(每个 program 管
BM 行),K/V 被切块(每轮 BLOCK_N 列),全部状态 m、l、acc 都在寄存器里。

一个小技巧:**`tl.math.exp2` 是 2 的幂,比 `exp` 快**;但 softmax 是自然底数 e。所以把指数乘上
`LOG2E = 1/ln2 ≈ 1.4427`,`exp2(x·LOG2E) = e^x`,既快又精确。下面 kernel 的 `qk`、`p`、`alpha`
都用这个技巧。
'''))

NB.code(FA_KERNEL, "**逐行读**:外层 `for start_n` 沿 K/V 分块;`m_ij = max(m_i, max(qk,1))` 更新 running max;`p` 以新 max 归一;`alpha` 校正旧累加;`l_i`、`acc` 边走边更新。")

NB.md("## 4️⃣ 正确性验证 ✅",
D('''
在 GPU 上运行我们的 Triton FA kernel,与 **torch 原生 attention**(`F.scaled_dot_product_attention`,
内部就是 FlashAttention 或 memory-efficient 实现)对照误差。fp16 下相对误差在 1e-4 量级即视为正确。
'''))

NB.code(D('''
N, D = 1024, 64
q = torch.randn(N, D, device="cuda", dtype=torch.float16)
k = torch.randn(N, D, device="cuda", dtype=torch.float16)
v = torch.randn(N, D, device="cuda", dtype=torch.float16)
o = torch.zeros(N, D, device="cuda", dtype=torch.float16)

BM, BN = 128, 64
fa_kernel[(triton.cdiv(N, BM),)](q, k, v, o, 1.0 / (D ** 0.5), LOG2E, N, N, D,
    D, 1, D, 1, D, 1, D, 1, BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=D, num_warps=8)
torch.cuda.synchronize()

ref = torch.nn.functional.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None])[0, 0]
err = (o.float() - ref.float()).abs().max().item()
rel = err / ref.float().abs().max().item()
print(f"N=1024, d=64, fp16: max abs err = {err:.3e}  rel = {rel:.2e}")
'''), "🎯 相对误差 1e-4 量级 = 与 torch 原生 attention 等价,我们的 Triton FA 写对了。")

NB.md("## 5️⃣ 与 torch SDPA 实测对比 ⚔️",
D('''
在 RTX 5060 上,把我们的 Triton FA 与 torch 原生 `scaled_dot_product_attention` 做吞吐对比。
衡量指标是 **TFLOPS = 4·N²·d / 耗时**(attention 主要 FLOPs 在 QK^T 与 PV 两处矩阵乘)。
'''))

NB.code(BENCH, "复用 warmup + synchronize 的计时函数,编译开销不计入结果。")

NB.code(D('''
N, D = 2048, 64
q = torch.randn(N, D, device="cuda", dtype=torch.float16)
k = torch.randn(N, D, device="cuda", dtype=torch.float16)
v = torch.randn(N, D, device="cuda", dtype=torch.float16)
o = torch.zeros(N, D, device="cuda", dtype=torch.float16)
flop = 4 * N * N * D

def triton_fa(bm, bn, w):
    def f():
        fa_kernel[(triton.cdiv(N, bm),)](q, k, v, o, 1.0 / (D ** 0.5), LOG2E, N, N, D,
            D, 1, D, 1, D, 1, D, 1, BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=D, num_warps=w)
    return f

def torch_sdpa():
    torch.nn.functional.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None])

ts = bench(torch_sdpa, warmup=3, iters=15)
print(f"torch SDPA: {ts:.3f} ms = {flop/(ts/1000)/1e12:.0f} TFLOPS")
for nm, (bm, bn, w) in [("BM=128 BN=64 w=8", (128, 64, 8)), ("BM=128 BN=128 w=8", (128, 128, 8)),
                        ("BM=64 BN=128 w=8", (64, 128, 8)), ("BM=128 BN=64 w=4", (128, 64, 4))]:
    t = bench(triton_fa(bm, bn, w), warmup=3, iters=15)
    print(f"Triton FA {nm:18s}: {t:.3f} ms = {flop/(t/1000)/1e12:.0f} TFLOPS")
'''), "🚀 记录 torch 与各 Triton 配置的吞吐,下面画对比图。")

NB.md("## 6️⃣ 内存节省分析 O(N²) → O(N) 🧮",
D('''
除了吞吐,FlashAttention 还省**显存**:naive 需要把 S 与 P 两个 N×N 矩阵整体落盘,而 flash 从不
写它们,只需 Q/K/V/O。

- **naive 峰值显存** ≈ $2 N^2 \\cdot \\text{bytes}$(S 与 P 各一个 N×N);
- **flash 峰值显存** ≈ $4 N d \\cdot \\text{bytes}$(Q/K/V/O)。

N 越大,naive 越离谱。下面算数值并画对数轴曲线。
'''))

NB.code(PLT_STYLE, "🎨 先套用统一样式头(浅色网格、大图、高 DPI),下面的曲线都用它。")

NB.code(D('''
def mem_naive(N, d=64, b=2): return (2 * N * N + 4 * N * d) * b / 1e6   # S、P + QKV O
def mem_flash(N, d=64, b=2): return (4 * N * d) * b / 1e6               # Q/K/V/O

for N in [1024, 2048, 4096, 8192]:
    mf, mn = mem_flash(N), mem_naive(N)
    print(f"N={N:5d}: flash={mf:8.2f}MB  naive={mn:9.1f}MB  节省 {mn/mf:6.1f}×")
'''), "🔍 序列翻倍,naive 显存涨 4 倍,flash 只涨 2 倍——差距越拉越大。")

NB.code(D('''
ns = [512, 1024, 2048, 4096, 8192, 16384]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(ns, [mem_naive(n) for n in ns], "o-", color="#E45756", lw=2.5, label="naive(S+P 落盘)")
ax.plot(ns, [mem_flash(n) for n in ns], "o-", color="#4C78A8", lw=2.5, label="flash(不落 S/P)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("序列长度 N"); ax.set_ylabel("峰值显存 MB")
ax.set_title("naive O(N²) vs flash O(N) 的峰值显存")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
'''), "📊 对数轴上,naive 是斜率 2 的陡线,flash 是斜率 1 的缓线——这正是 FlashAttention 的显存红利。")

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=["torch SDPA", "Triton FA\\n(128,64,w8)", "Triton FA\\n(128,128,w8)",
                        "Triton FA\\n(64,128,w8)", "Triton FA\\n(128,64,w4)"],
                     y=[flop/(bench(torch_sdpa, warmup=2, iters=10)/1000)/1e12,
                        flop/(bench(triton_fa(128,64,8), warmup=2, iters=10)/1000)/1e12,
                        flop/(bench(triton_fa(128,128,8), warmup=2, iters=10)/1000)/1e12,
                        flop/(bench(triton_fa(64,128,8), warmup=2, iters=10)/1000)/1e12,
                        flop/(bench(triton_fa(128,64,4), warmup=2, iters=10)/1000)/1e12],
                     marker_color="#4C78A8"))
fig.update_layout(title="FlashAttention 吞吐对比(TFLOPS, N=2048, d=64)",
                  yaxis_title="TFLOPS", xaxis_tickangle=20, height=400,
                  margin=dict(l=10, r=10, t=50, b=10))
fig
'''), "📊 plotly 交互柱状图:并排对比 torch SDPA 与我们不同配置的 Triton FA,悬停看数值。")

NB.md("## 7️⃣ 配套 Streamlit 演示:拖块大小/序列长度看吞吐 🎛️",
D('''
运行同目录下的 `app_55_triton_fa.py`,拖动 **序列长度 N**、**BLOCK_M/N**、**num_warps**,在
RTX 5060 上实时测 Triton FA 吞吐,与 torch SDPA 对比,并展示 naive vs flash 的显存曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_55_triton_fa.py
```

浏览器打开 **http://localhost:8501**。建议把 N 从 1024 拖到 8192,看显存曲线如何“一个陡一个缓”。
完整源码如下(与同目录 `app_55_triton_fa.py` 一字不差):
'''))

NB.code(f"%%writefile {APP_FILE}\n" + APP_55, "📜 这就是 app_55_triton_fa.py 的完整源码,用 st.cache_data 缓存各块大小/序列长度的基准,拖拽实时响应。")

wrapup(NB,
    summary=[
        "attention 慢在访存:naive 要把 S、P 两个 N×N 矩阵整体读写 HBM",
        "分块(tiling)+ 在线 softmax(online softmax)让 S/P 永不落盘,是 FlashAttention 的钥匙",
        "Triton 里用 running max m 与 running sum l 做 block-wise softmax;exp2+LOG2E 兼顾速度与精度",
        "正确性验证:相对误差 1e-4 量级,与 torch 原生 attention 等价",
        "峰值显存从 O(N²) 压到 O(N),序列越长节省越夸张,是长序列推理提速的关键",
    ],
    practice=[
        "把 fa_kernel 加因果掩码(下三角),实现 GPT 式的因果 FlashAttention",
        "给 fa_kernel 增加 batch/head 维度,处理 (B, H, N, d) 输入",
        "扫描不同 BLOCK_M/BLOCK_N,找当前 N 下吞吐最高的配置",
        "把 LOG2E 去掉改用 tl.exp,对比正确性与吞吐(体会 exp2 的收益)",
    ],
    links=[
        ("FlashAttention 论文", "https://arxiv.org/abs/2205.14135"),
        ("Triton 官方文档", "https://triton-lang.org"),
        ("Triton 06-fused-attention 教程", "https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html"),
    ])

NB.save(str(Path(CH09) / "55_triton_flashattn.ipynb"))

app_path = Path(CH09) / APP_FILE
app_path.write_text(APP_55 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
