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
st.caption("《minivllm: 图解 vLLM 推理引擎》第 9 章 · 第 55 课配套演示")
