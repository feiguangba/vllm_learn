# -*- coding: utf-8 -*-
# app_51_triton_intro.py — Triton 是什么:与 CUDA 的概念对照 🐍
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🐍 51 · Triton 入门", layout="wide")
st.title("🐍 第 51 课 · Triton:GPU 编程新范式")

st.markdown("""
Triton 是一种**面向 GPU 的编程语言**。与 CUDA 需要你关心“线程、共享内存、同步”这些底层细节不同,
Triton 让你只关心**数据分块(tile)**:你说“把矩阵切成 128×128 的块,每块怎么算”,编译器帮你
把块里的并行、访存、同步全部搞定。下方用三个控件,直观对比 Triton 与 CUDA 的概念。
""")

concepts = {
    "tile 编程模型 (tl 块)": {
        "triton": "@triton.jit\ndef k(x):\n    offs = tl.arange(0, BLOCK)\n    a = tl.load(x + offs)",
        "cuda": "int tid = blockIdx.x*blockDim.x + threadIdx.x;\nfloat a = x[tid];",
        "说明": "Triton 以“块(block/tile)”为最小思考单位,写代码像在处理一个小矩阵;CUDA 则面向单个线程 (thread)。Triton 由编译器自动铺开线程。",
    },
    "program_id(块编号)": {
        "triton": "pid = tl.program_id(axis=0)",
        "cuda": "int pid = blockIdx.x;   // 每个 block",
        "说明": "告诉“这个块是第几个”。Triton 里每个 program 相当于 CUDA 的一个 block;块与块之间互相独立、可并行。",
    },
    "tl.arange(连续下标)": {
        "triton": "offs = tl.arange(0, BLOCK)",
        "cuda": "for(int i=tid; i<N; i+=stride) ...   // 手动算下标",
        "说明": "tl.arange(0, BLOCK) 一次性生成 0..BLOCK-1 的向量,是 Triton 最常用的“下标生成器”,把一维循环改写成向量化。",
    },
    "constexpr 编译期常量": {
        "triton": "def k(..., BLOCK: tl.constexpr):",
        "cuda": "#define BLOCK 128   // 模板参数",
        "说明": "标成 constexpr 的参数必须在编译期已知(如块大小),编译器据此展开、特化并优化;运行时参数则保持为变量。",
    },
    "num_warps(线程束数)": {
        "triton": "k[grid](..., num_warps=4)",
        "cuda": "blockDim.x = 128;  // 1 warp = 32 线程",
        "说明": "每个 program 内部用多少个 warp(32 线程为一束)去执行块内计算。调大通常提高块内并行度,但也增加寄存器压力。",
    },
    "tl.dot(块级矩阵乘)": {
        "triton": "acc = tl.dot(a, b, acc)",
        "cuda": "调用 tensor core / 手写分块矩阵乘",
        "说明": "Triton 把矩阵乘抽象成 tl.dot,自动调度张量核(Tensor Core);CUDA 需要手写或调 cuBLAS/mma 指令。",
    },
    "@triton.jit 装饰器": {
        "triton": "@triton.jit\ndef k(...): ...",
        "cuda": "nvvcc 编译 .cu 源码 → cubin",
        "说明": "用 @triton.jit 装饰 Python 函数,它会被即时编译(JIT)成 GPU 机器码;第一次调用时编译并缓存,之后直接复用。",
    },
    "block 指针 (make_block_ptr)": {
        "triton": "p = tl.make_block_ptr(base, shape, strides, ...)",
        "cuda": "T* p = base + row*ld + col;   // 指针运算",
        "说明": "Triton 用“块指针”描述 HBM 上一块二维数据及其步长,配合 tl.load/store 一次性搬运整块;CUDA 用裸指针手动算偏移。",
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    sel = st.selectbox("选择一个 Triton 概念", list(concepts.keys()))
    grid_n = st.slider("grid 大小(program 个数)", 1, 32, 8, 1)
    model = st.radio("编程范式", ["SPMD(数据并行)", "线程中心(thread-centric)"])
    st.caption("grid = 一次启动的 program 总数,每个 program 独立处理自己那块数据。")

c = concepts[sel]
st.subheader("🔀 对照:" + sel)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**🐍 Triton**")
    st.code(c["triton"], language="python")
with col2:
    st.markdown("**🎯 CUDA**")
    st.code(c["cuda"], language="cpp")
st.markdown("**💡 解读**:" + c["说明"])

# ---------- grid / program 可视化 ----------
st.subheader("🧮 grid 上的 program 分布")
rows = int(grid_n ** 0.5)
cols = (grid_n + rows - 1) // rows
xs, ys, labels = [], [], []
for i in range(grid_n):
    xs.append(i % cols + 0.5)
    ys.append(i // cols + 0.5)
    labels.append(str(i))
fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers+text",
                           text=labels, textposition="middle center",
                           marker=dict(size=40, color="#4C78A8", line=dict(width=2, color="white"))))
fig.update_layout(xaxis=dict(range=[0, cols], showgrid=False, title="program 列"),
                  yaxis=dict(range=[0, rows], showgrid=False, title="program 行", autorange="reversed"),
                  height=300, margin=dict(l=10, r=10, t=30, b=10),
                  title=f"grid = {grid_n} 个 program(≈ CUDA 的 {grid_n} 个 block,块间并行)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:Triton 的核心心智模型是 **SPMD(单程序多数据)**——同一份 kernel 源码,被并行地在
> 很多个 program(block)上执行,每个 program 用 `tl.program_id` 知道自己管哪一块数据。
> 它把 CUDA 里“线程 ↔ 数据”的手动映射,升级成了“块 ↔ 数据”的声明式描述,复杂度交给编译器。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 9 章 · 第 51 课配套演示")
