# -*- coding: utf-8 -*-
# app_74_ascend_c.py — Ascend C 算子数据流交互 ⚙️
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Ascend C 算子数据流 ⚙️", layout="wide")
st.title("⚙️ 第 74 课 · Ascend C 算子编程:数据流交互")

st.markdown("""
Ascend C 是昇腾的算子开发语言。写算子时你不必关心"每个线程干什么",而要安排**数据流水**:
把全局内存(HBM)的数据**分块搬入**片内 L1/L0 → **矢量单元**计算 → 把结果**搬回**全局内存。
下方拖动**向量长度**与**分块大小**,观察流水被切成了几段、每段多大、总访存量是多少。
""")

n = st.sidebar.slider("向量长度(元素数)", 1_000_000, 50_000_000, 10_000_000, 1_000_000)
block = st.sidebar.slider("分块大小(每块元素数)", 1_000, 500_000, 100_000, 1_000)
show_pipeline = st.sidebar.checkbox("展示流水阶段图", value=True)
st.sidebar.caption("分块越大,流水段越少但片上 L1/L0 压力越大;分块越小,循环调度开销越高。")

el_bytes = 4                       # fp32
n_blocks = int(np.ceil(n / block))
block_bytes = block * el_bytes
total_traffic = n * el_bytes * 3   # 读 A + 读 B + 写 C

c1, c2, c3, c4 = st.columns(4)
c1.metric("流水段数(循环次数)", n_blocks)
c2.metric("每段字节数", f"{block_bytes/1024:.0f} KB")
c3.metric("总访存量", f"{total_traffic/1e6:.1f} MB")
c4.metric("调度开销(相对)", f"{n_blocks:.0f}")

if show_pipeline:
    st.subheader("🔁 单段流水:搬入 → 矢量计算 → 搬出")
    stages = ["HBM 读 A", "L1/L0 缓存", "Vector 计算", "写回 HBM"]
    fig = go.Figure()
    for i, s in enumerate(stages):
        fig.add_shape(type="rect", x0=i - 0.35, x1=i + 0.35, y0=0, y1=1,
                      line=dict(color="#2c3e50", width=1.5), fillcolor="#aed6f1")
        fig.add_annotation(x=i, y=0.5, text=s, showarrow=False, font=dict(size=11))
        if i < len(stages) - 1:
            fig.add_annotation(x=i + 0.5, y=0.5, text="→", showarrow=False, font=dict(size=16))
    fig.update_xaxes(showticklabels=False, range=[-0.7, 3.7])
    fig.update_yaxes(showticklabels=False, range=[0, 1.3])
    fig.update_layout(title=f"流水线重复 {n_blocks} 次", height=250,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 流水段数随分块大小的变化")
blocks_range = np.geomspace(1_000, 500_000, 40).astype(int)
nseg = [int(np.ceil(n / b)) for b in blocks_range]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=blocks_range, y=nseg, mode="lines+markers",
                          line=dict(color="#E45756", width=3), name="流水段数"))
fig2.add_vline(x=block, line_dash="dash", line_color="#4C78A8",
               annotation_text=f"当前分块 {block:,}", annotation_position="top right")
fig2.update_layout(xaxis_type="log", title="分块越大,流水段越少(调度开销越低)",
                   xaxis_title="分块大小(对数)", yaxis_title="流水段数", height=360)
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 工程上要在『段数少(省调度)』与『片上放得下(不爆 L1/L0)』之间折中。")

st.markdown("""
> 💡 **结论**:Ascend C 的核心是把『数据搬移 + 计算』编排成流水。段数(N=len/block)越少,
> 越省循环与同步开销;但块太大,片内 L0/L1 放不下就会『溢出搬回』,得不偿失。
""")
