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
st.caption("《minivllm: 图解 vLLM 推理引擎》第 9 章 · 第 54 课配套演示")
