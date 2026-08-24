# -*- coding: utf-8 -*-
# app_61_compiler_overview.py — AI 编译器全景:编译流水线交互浏览 🗺️
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="AI 编译器全景 🗺️", layout="wide")
st.title("🗺️ 第 61 课 · AI 编译器全景:编译流水线交互浏览")

st.markdown("""
AI 编译器就像一位**总厨**:先把整份"菜谱"(计算图)读完,再统一调度切配与掌勺,而不是
每道工序各请一位厨师各开一灶。它的核心是一条 **流水线:前端 → IR → 优化 → 后端**。
在下方挑选一个框架,逐步点开流水线的每一站,看看这一站到底在做什么、输入输出是什么。
""")

# ---------------- 数据:各框架的流水线要素 ----------------
PIPELINE = {
    "TVM": {
        "前端": "Relay / TE(Tensor Expression)表达式,描述算子组合",
        "IR": "Relay IR(高层)→ TensorIR(中层,带循环结构)",
        "优化": "算子融合、layout 转换、AutoTVM/Ansor 自动调优",
        "后端": "代码生成 → CUDA/OpenCL/LLVM/ARM CPU 等",
        "一句话": "把 Python 式算子组合降到可调度的 tensor 程序,再按后端生成代码",
    },
    "MLIR": {
        "前端": "多级 Dialect(方言)统一表示,如 torch-mlir / StableHLO 进入",
        "IR": "linalg / tensor / scf / arith 等多级 Dialect 逐级降低",
        "优化": "canonicalize、CSE、循环分块、向量化等 pass 管线",
        "后端": "LLVM Dialect → 机器码;可对接任意硬件后端",
        "一句话": "一座「通用编译器基础设施」,用「方言 + pass 管线」描述任意抽象层次",
    },
    "XLA": {
        "前端": "HLO(High Level Optimizer)计算图,来自 JAX/TF",
        "IR": "HLO → LHLO → LLVM IR 逐级降低",
        "优化": "算子融合、layout 分配、内存规划、buffer 复用",
        "后端": "CUDA(经 LLVM)/TPU/CPU",
        "一句话": "Google 的深度学习编译器,以「整图融合」闻名",
    },
    "torch.compile(Inductor)": {
        "前端": "TorchDynamo 追踪 Python,得到 FX Graph(aten 算子级)",
        "IR": "FX/ATen 图 → Inductor 的 Scheduler IR(含 loop 结构)",
        "优化": "算子融合、dead code 消除、layout 选择、tiling",
        "后端": "GPU:生成 Triton kernel;CPU:生成 C++(需 MSVC)",
        "一句话": "PyTorch 自带的 JIT 编译器,把 eager 的小算子并成大 kernel",
    },
}

FRAMES = ["前端", "IR", "优化", "后端"]

st.sidebar.header("🎛️ 参数")
framework = st.sidebar.selectbox("选择一个编译器框架", list(PIPELINE.keys()))
stage = st.sidebar.radio("当前浏览的流水线阶段", FRAMES, index=0)
show_ir_levels = st.sidebar.checkbox("展开多级 IR 示意图", value=True)
show_table = st.sidebar.checkbox("显示全框架对比表", value=True)
st.sidebar.caption("AI 编译器的核心思想:把「能优化」的抽象(IR)和「能跑」的抽象分开,中间用 pass 连接。")

info = PIPELINE[framework]
st.markdown(f"### 当前框架: **{framework}**")
st.markdown(f"**💡 {info['一句话']}**")

# ---------------- 当前阶段讲解 ----------------
st.info(f"**{stage}** 阶段:{info[stage]}")

# ---------------- 流水线结构图(plotly)----------------
st.subheader("🔧 编译流水线(可切换阶段)")
st.markdown("选中阶段会高亮;流水线从左到右:源代码 → 前端 → IR → 优化 → 后端 → 可执行 kernel。")
stages_x = ["源码/模型", "前端", "IR(中间表示)", "优化 pass", "后端", "kernel"]
y0 = 0.5
fig = go.Figure()
for i, stg in enumerate(stages_x):
    hl = (stg == stage) or (stg == "IR(中间表示)" and stage == "IR")
    fig.add_shape(type="rect", x0=i - 0.35, x1=i + 0.35, y0=y0 - 0.25, y1=y0 + 0.25,
                  line=dict(color="#2c3e50", width=1.5),
                  fillcolor="#e74c3c" if hl else "#d5dbdb")
    fig.add_annotation(x=i, y=y0, text=stg, showarrow=False, font=dict(size=11))
    if i < len(stages_x) - 1:
        fig.add_annotation(x=i + 0.5, y=y0, text="→", showarrow=False, font=dict(size=16))
fig.update_xaxes(range=[-0.7, len(stages_x) - 0.3], showticklabels=False)
fig.update_yaxes(range=[0, 1], showticklabels=False)
fig.update_layout(title=f"{framework} 的编译流水线", height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------- 多级 IR 示意图 ----------------
if show_ir_levels:
    st.subheader("🏗️ 多级 IR:从「看得懂」降到「跑得快」")
    st.markdown("IR 往往不止一级:高层 IR 贴近语义(好优化),低层 IR 贴近硬件(好执行),中间用 pass 逐级降低。")
    levels = ["高层 IR\n(Tensor / 算子级)", "中层 IR\n(循环 / Tile 级)", "低层 IR\n(LLVM / 指令级)"]
    abstr = ["抽象度高、易做融合", "可调度、可 tiling", "贴近硬件、可执行"]
    fig2 = go.Figure(go.Bar(x=[1, 2, 3], y=[5, 3, 1],
                            marker_color=["#1f77b4", "#2ca02c", "#d62728"],
                            text=[f"{l}<br>{a}" for l, a in zip(levels, abstr)],
                            textposition="outside"))
    fig2.update_layout(title="多级 IR 的抽象层级(柱高示意抽象度)", height=360,
                       xaxis=dict(tickvals=[1, 2, 3], ticktext=["高层", "中层", "低层"]),
                       yaxis=dict(showticklabels=False))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- 全框架对比表 ----------------
if show_table:
    st.subheader("📋 全框架对比")
    rows = [{"框架": k, **{f: PIPELINE[k][f] for f in FRAMES}} for k in PIPELINE]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.metric("已讲解框架数", len(PIPELINE))
st.caption("💡 建议:反复切换框架与阶段,体会「同一套流水线思想、不同实现侧重」这一主线。")
