# -*- coding: utf-8 -*-
# app_56_triton_compiler.py — 编译流程概念图 + 真实编译产物查看器 ⚙️
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

st.set_page_config(page_title="Triton 编译器原理 ⚙️", layout="wide")
st.title("⚙️ 第 56 课 · Triton 编译器原理:从 Python 到 PTX")

st.markdown("""
Triton 不是解释执行你的 kernel,而是把它**逐级翻译、逐级优化**成 GPU 机器码。
下方可以调整 `BLOCK / num_warps / num_stages`,每动一次都会**真实地重新编译**一次
triton 矩阵乘 kernel,并展示各中间产物的规模与片段 —— 编译器流水线一目了然。
""")

# ---------------------------------------------------------------- 编译用的 kernel(与 notebook 一致)
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

STAGES = [
    ("Python 源码", "你写的 @triton.jit 函数\n(Python + tl.* 操作)"),
    ("AST", "语法树\n函数被拆成节点"),
    ("TTIR", "Triton IR\n循环拍平,只剩基本运算"),
    ("TTGIR", "Triton GPU IR\ntile 布局 + 共享内存 + 调度"),
    ("LLVM IR", "寄存器分配\n指令选择"),
    ("PTX", "NVIDIA 虚拟指令集\n经 ptxas → SASS"),
]

# ---------------------------------------------------------------- 编译缓存
def get_compiled(BLOCK, nw, ns):
    key = (BLOCK, nw, ns)
    if key in st.session_state:
        return st.session_state[key]
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = torch.empty(256, 256, device="cuda", dtype=torch.float32)
    t0 = time.perf_counter()
    kern = matmul_kernel.warmup(a, b, c, 256, 256, 256,
                                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1),
                                BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=32,
                                num_warps=nw, num_stages=ns, grid=(1,))
    dt = (time.perf_counter() - t0) * 1000
    st.session_state[key] = (kern, dt)
    return st.session_state[key]

# ---------------------------------------------------------------- 侧边栏参数
with st.sidebar:
    st.header("🎛️ 编译参数")
    BLOCK = st.slider("BLOCK(输出 tile 边长)", 32, 128, 64, 16)
    num_warps = st.slider("num_warps(线程束数)", 1, 8, 4, 1)
    num_stages = st.slider("num_stages(流水线深度)", 1, 6, 3, 1)
    stage = st.selectbox("📄 查看哪一级产物", ["TTIR", "TTGIR", "LLVM IR", "PTX"])
    st.caption("滑杆变化即触发重新编译;TTGIR 会显示 num-warps / shared memory 等调度信息。")

kern, dt = get_compiled(BLOCK, num_warps, num_stages)
asm = kern.asm
stage_keys = {"TTIR": "ttir", "TTGIR": "ttgir", "LLVM IR": "llir", "PTX": "ptx"}
sizes = {s: len(asm[k]) for s, k in stage_keys.items()}
ptx_lines = asm["ptx"].count("\n")

# ---------------------------------------------------------------- 指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("编译耗时", f"{dt:.0f} ms")
c2.metric("TTIR", f"{sizes['TTIR']} 字符")
c3.metric("TTGIR", f"{sizes['TTGIR']} 字符")
c4.metric("PTX", f"{ptx_lines} 行")
c5.metric("num_warps", num_warps)

# ---------------------------------------------------------------- 编译流水线概念图(plotly)
st.subheader("🧭 编译流水线(概念图)")
labels = [s[0] for s in STAGES]
x_pos = list(range(len(STAGES)))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_pos, y=[1] * len(STAGES), mode="markers+text", text=labels,
    textposition="bottom center", textfont=dict(size=13),
    marker=dict(size=34, color=["#72B7B2", "#4C78A8", "#F28E2B", "#E45756", "#76B7B2", "#59A14F"])))
for i in range(len(STAGES) - 1):
    fig.add_annotation(x=(x_pos[i] + x_pos[i + 1]) / 2, y=1.0, text="➜", showarrow=False,
                       font=dict(size=20))
for i, (t, d) in enumerate(STAGES):
    fig.add_annotation(x=x_pos[i], y=0.72, text=d, showarrow=False, font=dict(size=12))
fig.update_yaxes(range=[0.4, 1.4], visible=False)
fig.update_xaxes(visible=False)
fig.update_layout(height=260, title="Python 源码 一路“翻译 + 优化”到 PTX / SASS",
                  margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("当前这个 kernel 已经编译到最后一站;你可以在侧边栏切换查看每一站的产物文本。")

# ---------------------------------------------------------------- 各阶段规模柱状图(plotly)
fig2 = go.Figure(go.Bar(x=list(sizes.keys()), y=list(sizes.values()),
                        marker_color="#4C78A8",
                        text=[f"{v // 1000:.1f} KB" for v in sizes.values()],
                        textposition="outside"))
fig2.update_layout(title="各编译阶段的产物规模(TTIR 最精简,PTX 最庞大)",
                   xaxis_title="编译阶段", yaxis_title="字符数", height=360,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- 产物片段
st.subheader(f"📄 {stage} 片段(前 30 行)")
lines = asm[stage_keys[stage]].splitlines()
st.code("\n".join(lines[:30]), language="text")
if stage == "TTGIR":
    hit = [ln for ln in lines if "num-warps" in ln or "local_alloc" in ln][:4]
    if hit:
        st.markdown("**🔍 关键调度信息**")
        st.code("\n".join(hit), language="text")
        st.caption("num-warps:该 kernel 编译成了几个线程束;local_alloc:在共享内存里分配了 tile 缓冲。")

st.markdown("""
> 💡 **结论**:同一个 Python 函数,编译器在不同参数下会编出**不同的 TTGIR 调度与 PTX**。
> tile 编程之所以高效,是因为编译器能基于 tile 做寄存器分配、共享内存复用与软件流水线。
""")
