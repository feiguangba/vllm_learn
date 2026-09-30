# -*- coding: utf-8 -*-
# app_53_triton_tile.py — tile 编程模型:矩阵分块可视化 🧩
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧩 53 · Triton tile 模型", layout="wide")
st.title("🧩 第 53 课 · tile 编程模型:把矩阵切成砖块")

st.markdown("""
Triton 的思考单位是 **tile(数据块)**。对矩阵运算,你会把它沿两个方向切成一块块 **BM×BN 的小瓷砖**,
每个 tile 交给一个 program(用 `tl.program_id(0/1)` 定位行列),块内再用 `tl.arange` 铺开成线程。
下方选择**矩阵大小**与**tile 形状**,实时看矩阵被分成几行几列的块、每块长什么样。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    M = st.slider("矩阵行数 M", 8, 1024, 128, 8)
    N = st.slider("矩阵列数 N", 8, 1024, 256, 8)
    BM = st.select_slider("tile 行数 BM", options=[16, 32, 64, 128], value=64)
    BN = st.select_slider("tile 列数 BN", options=[16, 32, 64, 128], value=64)
    show_id = st.checkbox("在每个 tile 上标注 program_id", value=True)
    st.caption("BM×BN 决定每块 tile 的大小;程序编号按行优先排列。")

gm = (M + BM - 1) // BM   # 行方向 program 数
gn = (N + BN - 1) // BN   # 列方向 program 数
grid = gm * gn

c1, c2, c3 = st.columns(3)
c1.metric("行方向 program 数", gm)
c2.metric("列方向 program 数", gn)
c3.metric("grid 总 program 数", grid)

# ---------- tile 布局矩阵 ----------
Z = np.zeros((M, N), dtype=int)
for i in range(gm):
    for j in range(gn):
        pid = i * gn + j                       # 行优先编号
        r0, r1 = i * BM, min((i + 1) * BM, M)
        c0, c1 = j * BN, min((j + 1) * BN, N)
        Z[r0:r1, c0:c1] = pid

fig = go.Figure(go.Heatmap(
    z=Z, x=[f"{j}" for j in range(N)], y=[f"{i}" for i in range(M)],
    colorscale="Blues", showscale=False,
    text=Z if show_id else np.zeros_like(Z),
    texttemplate="%{text}", textfont=dict(size=9),
    hovertemplate="program %{z}<extra>%{y} 行, %{x} 列</extra>"))
fig.update_layout(title=f"矩阵 {M}×{N} 按 tile {BM}×{BN} 切分 → grid {gm}×{gn} 个 program",
                  height=max(420, M * 3), margin=dict(l=10, r=10, t=50, b=10),
                  xaxis=dict(title="列 j", scaleanchor="y"), yaxis=dict(title="行 i", autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:每个 program 通过 `pid = tl.program_id(0) * gn + tl.program_id(1)`(或直接两个维度)找到
> 自己负责的那块 tile,再用 `offs = pid * BLOCK + tl.arange(0, BLOCK)` 生成块内下标。**tile 的大小
> (BM×BN) 和数量(grid)是一对跷跷板**:tile 越大、块越少;tile 越小、块越多、每块并行度越低。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 9 章 · 第 53 课配套演示")
