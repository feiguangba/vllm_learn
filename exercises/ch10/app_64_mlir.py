# -*- coding: utf-8 -*-
# app_64_mlir.py — MLIR 与中间表示:Dialect / Pass 概念浏览 🏗️
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="MLIR 与中间表示 🏗️", layout="wide")
st.title("🏗️ 第 64 课 · MLIR 与中间表示")

st.markdown("""
MLIR(Multi-Level Intermediate Representation)是 Google 推出的一套**编译器基础设施**。它的核心
思想是:**没有一种 IR 能同时『好优化』和『好执行』** —— 所以它造了一堆能相互转换的 **Dialect(方言)**,
每级方言贴近不同的抽象层次,用 **Pass 管线**逐级降低。下方挑选一个方言看它负责什么,再选几个 pass
拼一条管线,看看"降级"是怎么一步步发生的。
""")

DIALECTS = {
    "tensor": {
        "层次": "高层(语义)",
        "作用": "描述『整个张量』级别的操作,如 tensor.collapse_shape / tensor.extract_slice,不做循环展开",
        "类比": "菜谱里写的『把土豆切丁』 —— 只说做什么,不说怎么切",
        "颜色": "#1f77b4",
    },
    "linalg": {
        "层次": "中高层(结构化)",
        "作用": "以结构化形式描述『逐元素 / 归约 / 矩阵乘』等算子,保留循环结构供后续优化(tiling/fusion)",
        "类比": "菜谱写的『用刀把每个土豆切成 1cm 见方』 —— 已经暗示了循环结构",
        "颜色": "#2ca02c",
    },
    "scf": {
        "层次": "中层(控制流)",
        "作用": "结构化控制流:scf.for / scf.if,负责循环与分支的显式表达",
        "类比": "『重复 10 次:切一个土豆』 —— 循环写出来了",
        "颜色": "#d62728",
    },
    "arith": {
        "层次": "中低层(算术)",
        "作用": "标量算术:arith.addf / arith.mulf,单元素运算",
        "类比": "『把这一小块加那一小块』 —— 从张量降到标量",
        "颜色": "#9467bd",
    },
    "vector": {
        "层次": "低层(向量)",
        "作用": "向量/SIMD 级操作,vector.contract / vector.transfer,贴近 CPU 向量化与 GPU 并行",
        "类比": "『一次处理 8 个土豆』 —— 开始为硬件并行做准备",
        "颜色": "#ff7f0e",
    },
    "llvm": {
        "层次": "最低层(机器)",
        "作用": "LLVM IR Dialect,交给 LLVM 做寄存器分配与指令选择,最终生成机器码",
        "类比": "『用机器指令切土豆』 —— 完全贴近硬件",
        "颜色": "#8c564b",
    },
}

PASSES = {
    "canonicalize": "把算子化简成规范形式(x+0→x、合并嵌套算子),减小后续工作",
    "CSE(公共子表达式消除)": "删掉重复计算:同一个子表达式只算一次",
    "loop-fusion": "合并相邻循环,减少循环开销与中间缓冲",
    "tiling": "把大循环切成小块(tile),提高缓存/寄存器局部性",
    "vectorize": "把标量循环向量化(一次处理多个元素),利用 SIMD 或 GPU 并行",
    "lower-to-llvm": "把高层次方言『降级(lower)』成 LLVM Dialect",
}

st.sidebar.header("🎛️ 参数")
dialect = st.sidebar.selectbox("选择一个 Dialect(方言)", list(DIALECTS.keys()))
show_all_passes = st.sidebar.checkbox("显示完整 pass 管线", value=True)
st.sidebar.caption("MLIR 的优雅之处:每条 pass 只做一件小事,由框架按顺序串成管线。")

info = DIALECTS[dialect]
st.markdown(f"### 当前 Dialect:**{dialect}**  ({info['层次']})")
c1, c2 = st.columns([1, 1])
c1.info(f"**作用**:{info['作用']}")
c2.success(f"**类比**:{info['类比']}")

# ---------------- 抽象层级图 ----------------
st.subheader("⛰️ 多级 IR 的抽象金字塔")
levels = list(DIALECTS.keys())
yvals = list(range(len(levels), 0, -1))[::-1]
fig = go.Figure(go.Bar(x=[1] * len(levels), y=yvals,
                       orientation="v",
                       marker_color=[DIALECTS[k]["颜色"] for k in levels],
                       text=levels, textposition="inside",
                       hovertemplate="%{text}<extra></extra>"))
fig.update_layout(title="从高层语义(down)到低层机器:每级方言一个抽象层次",
                  xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                  height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# 标注当前方言在金字塔的位置
idx = levels.index(dialect)
st.markdown(f"**当前选中**:{dialect}(第 {len(levels) - idx} 层,从底往上是 {idx + 1} 层)")

# ---------------- pass 管线 ----------------
st.subheader("🔧 Pass 管线")
if show_all_passes:
    st.markdown("一条典型的『高到低』pass 管线(可理解为从左到右逐步降级):")
    plist = ["tensor", "canonicalize", "linalg", "tiling", "vectorize", "lower-to-llvm", "llvm"]
    fig2 = go.Figure()
    for i, p in enumerate(plist):
        fig2.add_annotation(x=i, y=0, text=p, showarrow=False,
                            font=dict(size=11, color="white"),
                            bgcolor="#c0392b" if p in PASSES else "#2c3e50",
                            borderpad=6)
        if i < len(plist) - 1:
            fig2.add_annotation(x=i + 0.5, y=0, text="→", showarrow=False, font=dict(size=16))
    fig2.update_xaxes(range=[-0.5, len(plist) - 0.5], showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    fig2.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📚 pass 词条")
for name, desc in PASSES.items():
    st.markdown(f"- **{name}**:{desc}")

st.caption("💡 记不住方言没关系,抓住一条主线即可:『每级方言降一点抽象,每条 pass 做一件小事,串起来就是编译』。")
