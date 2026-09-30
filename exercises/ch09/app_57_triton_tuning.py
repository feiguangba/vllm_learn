# -*- coding: utf-8 -*-
# app_57_triton_tuning.py — GEMM 调参扫描 🎛️
import os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import shutil
_PTXAS = os.environ.get("TRITON_PTXAS_PATH") or shutil.which("ptxas")   # 系统 CUDA 自带 ptxas
if _PTXAS:                                                     # 未命中则回退 PATH 查找
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
