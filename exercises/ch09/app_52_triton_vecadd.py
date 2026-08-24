# -*- coding: utf-8 -*-
# app_52_triton_vecadd.py — 向量加法:tile 大小滑杆看性能 ➕
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch, triton, triton.language as tl

st.set_page_config(page_title="➕ 52 · Triton 向量加法", layout="wide")
st.title("➕ 第 52 课 · 向量加法:grid / block / pid 与性能")

st.markdown("""
向量加法 `y = x + a` 是最简单的 Triton kernel,却足以讲清 **grid(几个 program)/ block(一块多大)/
pid(我是第几块)** 三个核心概念。GPU kernel 是**访存受限**的:瓶颈在读写显存的带宽,而不是算力。
下方拖动 **tile 大小(BLOCK)** 与 **向量长度**,实时测 Triton 内核的带宽,并与 torch 原生 GPU 加法对比。
""")

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def bench(fn, *args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

@st.cache_data(show_spinner=False)
def bench_block(n, block):
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    def tfn():
        add_kernel[(triton.cdiv(n, block),)](x, y, out, n, BLOCK_SIZE=block)
    t0 = time.perf_counter(); tfn(); torch.cuda.synchronize()  # 触发编译
    t = bench(tfn, warmup=5, iters=30)
    def gfn(): torch.add(x, y, out=out)
    tg = bench(gfn, warmup=5, iters=30)
    bd = n * 4 * 2 / (t / 1000) / 1e9
    bdg = n * 4 * 2 / (tg / 1000) / 1e9
    return t, tg, bd, bdg

with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("向量长度 n(元素数)", 1_000_000, 64_000_000, 16_000_000, 1_000_000)
    block = st.select_slider("tile 大小 BLOCK", options=[32, 64, 128, 256, 512, 1024, 2048, 4096], value=1024)
    iters = st.slider("计时迭代次数", 10, 100, 30, 10)
    st.caption("BLOCK 越大,每个 program 一次处理的元素越多;向量长度需能被 BLOCK 近似整除才有意义。")

t_tri, t_torch, bd_tri, bd_torch = bench_block(n, block)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Triton 带宽", f"{bd_tri:.0f} GB/s")
c2.metric("torch GPU 带宽", f"{bd_torch:.0f} GB/s")
c3.metric("Triton 耗时", f"{t_tri:.3f} ms")
c4.metric("torch 耗时", f"{t_torch:.3f} ms")
st.caption(f"n={n:,} 元素 | BLOCK={block} | 理论需读写 {n*8/1e6:.0f} MB(读 x,y + 写 out)")

# ---------- 性能随 BLOCK 变化曲线 ----------
blocks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
rows = []
for b in blocks:
    tt, tg, bd, bdg = bench_block(n, b)
    rows.append((b, bd, bdg))
bs = [r[0] for r in rows]; bds = [r[1] for r in rows]; bdgs = [r[2] for r in rows]
fig = go.Figure()
fig.add_trace(go.Scatter(x=bs, y=bds, mode="lines+markers", name="Triton", line=dict(color="#4C78A8", width=3)))
fig.add_trace(go.Scatter(x=bs, y=bdgs, mode="lines+markers", name="torch GPU", line=dict(color="#E45756", width=3, dash="dot")))
fig.add_vline(x=block, line_dash="dash", line_color="gray")
fig.update_layout(xaxis=dict(type="log", title="BLOCK(tile 大小)"),
                  yaxis_title="带宽 GB/s", height=380, margin=dict(l=10, r=10, t=40, b=10),
                  title="带宽随 BLOCK 变化(虚线=当前选择)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:GPU 加法是访存受限的,带宽趋近于显存的峰值(HBM)。BLOCK 太小时,块数过多、
> 启动/调度开销占比大,带宽上不去;BLOCK 足够大后,曲线进入平台期——再大也不会有明显提升。
> 这解释了为什么“调 tile 大小”是 Triton 里最常见的性能旋钮之一。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 9 章 · 第 52 课配套演示")
