# -*- coding: utf-8 -*-
# app_60_triton_oplib.py — 算子选择器对比 📦
import os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_PTXAS = r"D:\CUDA\v13.3\bin\ptxas.exe"
if os.path.exists(_PTXAS):
    os.environ.setdefault("TRITON_PTXAS_PATH", _PTXAS)
import streamlit as st
import plotly.graph_objects as go
import torch
import triton
import triton.language as tl

st.set_page_config(page_title="Triton mini 算子库 📦", layout="wide")
st.title("📦 第 60 课 · 用 Triton 写 mini 算子库:选算子,看对比")

st.markdown("""
本页把第 60 课实现的三个 triton 算子(**LayerNorm / GELU / Softmax**)做成交互对比:
选择算子、调整规模,点“重新测量”,即可看到 triton vs torch 的耗时、吞吐与对拍误差。
""")

# ---------------------------------------------------------------- 三个 triton 算子(与 notebook 一致)
@triton.jit
def layer_norm_fwd(x_ptr, y_ptr, w_ptr, b_ptr, eps, M, N, row_stride,
                   BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * row_stride + offs, (xc * rstd) * w + b, mask=mask)


def layer_norm_triton(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    layer_norm_fwd[(M,)](x, y, w, b, eps, M, N, N, BLOCK_N=triton.next_power_of_2(N))
    return y


@triton.jit
def gelu_fwd(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    o = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(o_ptr + offs, o, mask=mask)


def gelu_triton(x, BLOCK=1024):
    n = x.numel()
    o = torch.empty_like(x)
    gelu_fwd[(triton.cdiv(n, BLOCK),)](x, o, n, BLOCK=BLOCK)
    return o


@triton.jit
def softmax_fwd(x_ptr, o_ptr, M, N, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    p = tl.exp(x - m)
    s = tl.sum(p, axis=0)
    tl.store(o_ptr + row * row_stride + offs, p / s, mask=mask)


def softmax_triton(x, BLOCK_N=128):
    M, N = x.shape
    o = torch.empty_like(x)
    softmax_fwd[(M,)](x, o, M, N, N, BLOCK_N=triton.next_power_of_2(N))
    return o


def bench_ms(fn, *args, warmup=3, iters=20):
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
    op = st.selectbox("算子", ["LayerNorm", "GELU", "Softmax"])
    N = st.slider("每行元素数 N", 256, 4096, 1024, 128)
    rows = st.slider("行数(批次)", 64, 512, 128, 32)
    backend = st.radio("测量对象", ["Triton", "torch", "两者对比"])
    st.caption("这些算子都是 memory-bound,规模越大越能看出带宽的差异。")
    meas = st.button("⏱️ 重新测量")

# ---------------------------------------------------------------- 测量
if meas:
    x = torch.randn(rows, N, device="cuda")
    w = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    ops = {
        "LayerNorm": (lambda: layer_norm_triton(x, w, b),
                      lambda: torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5),
                      lambda: torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5)),
        "GELU":     (lambda: gelu_triton(x),
                     lambda: torch.nn.functional.gelu(x),
                     lambda: torch.nn.functional.gelu(x)),
        "Softmax":  (lambda: softmax_triton(x),
                     lambda: torch.softmax(x, dim=-1),
                     lambda: torch.softmax(x, dim=-1)),
    }
    fn_tri, fn_tor, fn_ref = ops[op]
    t_tri = bench_ms(fn_tri)
    t_tor = bench_ms(fn_tor)
    y_tri, y_ref = fn_tri(), fn_ref()
    torch.cuda.synchronize()
    err = (y_tri - y_ref).abs().max().item()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("triton", f"{t_tri:.4f} ms")
    c2.metric("torch", f"{t_tor:.4f} ms")
    c3.metric("torch/triton", f"{t_tor / max(t_tri, 1e-9):.2f}x")
    c4.metric("对拍 max 误差", f"{err:.1e}")

    if backend == "两者对比" or backend == "Triton":
        vals = [t_tri]
        names = ["triton"]
    if backend == "两者对比" or backend == "torch":
        vals = vals + [t_tor]
        names = names + ["torch"]
    fig = go.Figure(go.Bar(x=names, y=vals,
                           marker_color=["#4C78A8", "#E45756"],
                           text=[f"{v:.4f}" for v in vals], textposition="outside"))
    fig.update_layout(title=f"{op} · {rows}×{N}:triton vs torch",
                      yaxis_title="耗时(ms)", height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 扫规模曲线
    with st.spinner("正在扫描规模曲线……"):
        ns = [256, 1024, 4096]
        tt, ts = [], []
        for n in ns:
            xx = torch.randn(rows, n, device="cuda")
            ww = torch.randn(n, device="cuda"); bb = torch.randn(n, device="cuda")
            f1 = {"LayerNorm": (lambda: layer_norm_triton(xx, ww, bb)),
                  "GELU": (lambda: gelu_triton(xx)),
                  "Softmax": (lambda: softmax_triton(xx))}[op]
            f2 = {"LayerNorm": (lambda: torch.nn.functional.layer_norm(xx, (n,), ww, bb, 1e-5)),
                  "GELU": (lambda: torch.nn.functional.gelu(xx)),
                  "Softmax": (lambda: torch.softmax(xx, dim=-1))}[op]
            tt.append(bench_ms(f1, warmup=1, iters=10))
            ts.append(bench_ms(f2, warmup=1, iters=10))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[str(n) for n in ns], y=tt, mode="lines+markers",
                              name="triton", line=dict(color="#4C78A8", width=3)))
    fig2.add_trace(go.Scatter(x=[str(n) for n in ns], y=ts, mode="lines+markers",
                              name="torch", line=dict(color="#E45756", width=3)))
    fig2.update_layout(title=f"{op} 耗时随规模变化",
                       xaxis_title="N(每行元素)", yaxis_title="耗时(ms)",
                       height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("⭐ 这些简单算子都是 memory-bound,triton 与 torch 通常在同一量级;"
               "triton 的价值在融合与定制(下一节),而不是这些元素级算子本身。")

st.markdown("""
> 💡 **结论**:对 LayerNorm/GELU/Softmax 这类访存受限算子,triton 与 torch 差距不大;
> 算子库真正的价值是“一个接口、多后端可替换”,以及给融合算子(如 LayerNorm+GELU)留下舞台。
""")
