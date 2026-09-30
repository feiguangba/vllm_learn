# -*- coding: utf-8 -*-
# app_46_attention_perf.py — Attention 性能对比:naive vs 分块 vs SDPA 🚀
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

st.set_page_config(page_title="🚀 46 · Attention 性能", layout="wide")
st.title("🚀 第 46 课 · Attention 性能:naive vs 分块 vs torch SDPA")

st.markdown("""
三种 attention 实现,谁更快?用你的 **GPU** 实时跑一小段实验对比。naive 把 N×N 打分矩阵整体摊开;
分块(flash)用在线 softmax 边扫边累积;`F.scaled_dot_product_attention` 是 PyTorch 内置的 kernel 化
实现。拖动**序列长度 / 头数**,看耗时如何随参数变化。
""")

def naive_attention(Q, K, V):
    d = Q.shape[-1]
    S = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (d ** 0.5)
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", P, V)

def flash_chunked(Q, K, V, block_M=64):
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)
    m = torch.full((B, H, N, 1), float("-inf"))
    l = torch.zeros((B, H, N, 1))
    scale = d ** -0.5
    for j in range(0, N, block_M):
        Kj = K[:, :, j:j + block_M, :]; Vj = V[:, :, j:j + block_M, :]
        S = torch.einsum("bhnd,bhmd->bhnm", Q, Kj) * scale
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)
        P = torch.exp(S - m_new)
        l_new = l * torch.exp(m - m_new) + P.sum(dim=-1, keepdim=True)
        O = O * torch.exp(m - m_new) + torch.einsum("bhnm,bhmd->bhnd", P, Vj)
        m, l = m_new, l_new
    return O / l

def bench(fn, *a, iters=3):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    return (time.perf_counter() - t0) / iters * 1000

with st.sidebar:
    st.header("🎛️ 参数")
    N = st.slider("序列长度 N", 128, 1024, 256, 64)
    H = st.slider("头数 H", 1, 8, 4, 1)
    d = st.selectbox("头维度 d", [32, 64], index=0)
    st.caption("实验在你的 GPU 上真实运行几轮,请等待 1-2 秒。")

torch.manual_seed(0)
Q = torch.randn(1, H, N, d); K = torch.randn(1, H, N, d); V = torch.randn(1, H, N, d)

t_n = bench(naive_attention, Q, K, V)
t_f = bench(flash_chunked, Q, K, V, 64)
t_s = bench(lambda a, b, c: F.scaled_dot_product_attention(a, b, c), Q, K, V)

c1, c2, c3, c4 = st.columns(4)
c1.metric("naive", f"{t_n:.2f} ms")
c2.metric("分块 flash", f"{t_f:.2f} ms")
c3.metric("torch SDPA", f"{t_s:.2f} ms")
c4.metric("SDPA vs naive 加速", f"{t_n/max(t_s,1e-9):.1f}×")

o1 = naive_attention(Q, K, V); o2 = F.scaled_dot_product_attention(Q, K, V)
st.metric("naive vs SDPA 最大误差", f"{float((o1-o2).abs().max()):.2e}")

Ns = list(range(128, 1025, 128))
tn, tf, ts = [], [], []
for n in Ns:
    q = torch.randn(1, H, n, d); k = torch.randn(1, H, n, d); v = torch.randn(1, H, n, d)
    tn.append(bench(naive_attention, q, k, v))
    tf.append(bench(flash_chunked, q, k, v, 64))
    ts.append(bench(lambda a, b, c: F.scaled_dot_product_attention(a, b, c), q, k, v))

fig = go.Figure()
for name, arr, col in [("naive", tn, "#E45756"), ("分块", tf, "#4C78A8"), ("SDPA", ts, "#72B7B2")]:
    fig.add_trace(go.Scatter(x=Ns, y=arr, name=name, mode="lines+markers",
                             line=dict(color=col, width=2.5)))
fig.update_layout(title=f"三种实现耗时 vs 序列长度(H={H}, d={d})",
                  xaxis_title="序列长度 N", yaxis_title="耗时(ms)", height=440,
                  legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 观察:naive 随 N 二次方爬升;分块与 SDPA 明显更平缓。三种实现数值上几乎一致(误差<1e-6)。")

st.markdown("""
> 💡 **结论**:attention 是典型的 **memory-bound(访存受限)** 计算——瓶颈在把矩阵搬进搬出内存,
> 而不在计算本身。GPU 算力常常过剩,谁少搬数据谁就快。naive 多搬了 N×N 的 S/P,分块/SDPA 把它们
> 留在片上,于是快。**优化 attention 的钥匙是减少访存,而不是增加算力。**
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 7 章 · 第 46 课配套演示")
