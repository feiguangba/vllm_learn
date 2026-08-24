# -*- coding: utf-8 -*-
# app_100_summary.py — 全书总结:学习路线交互浏览 🏁
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="VLLM_learn 全书路线图 🏁", layout="wide")
st.title("🏁 第 100 课 · 全书总结:vLLM 推理引擎全景与学习路线")

st.markdown("""
从第 1 课『认识 Token』到第 100 课(就是本页),我们走完了 **LLM 推理的完整地图**:
**推理基础 → KV Cache → 批处理调度 → 执行加速 → 量化 → 分布式 → Attention →
部署监控 → Triton → AI 编译器 → 华为昇腾生态**。下方可任选章节 / 拖动课号范围 /
切换主题主线,把全书"地图"翻来覆去看。
""")

CHS = [
    (1, "LLM 推理基础", 1, 6), (2, "KV Cache 与 PagedAttention", 7, 13),
    (3, "Continuous Batching 与调度", 14, 20), (4, "模型执行与 CUDA Graph", 21, 27),
    (5, "量化", 28, 34), (6, "分布式并行", 35, 41),
    (7, "Attention Kernel 实战", 42, 47), (8, "端到端 vLLM 部署", 48, 50),
    (9, "Triton 编程", 51, 60), (10, "AI 编译器原理", 61, 70),
    (11, "华为昇腾与 MindSpore 生态", 71, 100),
]
LINES = {
    "KV Cache 主线": [7, 8, 9, 10, 11, 12, 13, 45, 85, 96],
    "批处理与调度主线": [14, 15, 16, 17, 18, 19, 20],
    "量化主线": [28, 29, 30, 31, 32, 33, 34, 87, 97],
    "编译器主线": [26, 51, 56, 61, 64, 66, 67, 70, 79, 95],
    "并行与通信主线": [35, 36, 37, 38, 39, 40, 41, 80, 98],
    "部署与监控主线": [24, 25, 27, 48, 49, 50, 81, 99],
}

st.sidebar.header("🎛️ 参数")
sel_chs = st.sidebar.multiselect("选择章节", [f"第{c[0]}章 {c[1]}" for c in CHS],
                                 default=["第11章 华为昇腾与 MindSpore 生态"])
rng = st.sidebar.slider("课号范围", 1, 100, (1, 100))
line = st.sidebar.radio("主题主线", list(LINES.keys()))
show_scatter = st.sidebar.checkbox("显示全书课程分布散点", value=True)
st.sidebar.caption("每一条主线都是『自洽的进阶路径』 —— 跳着学也能成体系。")

sel_idx = {f"第{c[0]}章 {c[1]}": c[0] for c in CHS}
covered = [c for c in CHS if f"第{c[0]}章 {c[1]}" in sel_chs]
total_lessons = sum(c[3] - c[2] + 1 for c in covered)

c1, c2, c3, c4 = st.columns(4)
c1.metric("全书课数", 100)
c2.metric("选中章节覆盖课数", total_lessons)
c3.metric("主题主线数", len(LINES))
c4.metric("当前主线课数", len(LINES[line]))

st.subheader("🏔️ 全书地图:课程分布")
if show_scatter:
    pts_x, pts_y, pts_c = [], [], []
    for ch, name, lo, hi in CHS:
        for n in range(lo, hi + 1):
            if ch in sel_idx.values() or f"第{ch}章 {name}" in sel_chs:
                if rng[0] <= n <= rng[1]:
                    pts_x.append(n); pts_y.append(ch)
    fig = go.Figure(go.Scatter(x=pts_x, y=pts_y, mode="markers",
                               marker=dict(size=11, color=pts_y, colorscale="Viridis",
                                           showscale=True, colorbar=dict(title="章节")),
                               text=[f"第{n}课" for n in pts_x], hovertemplate="%{text}<br>章节 %{y}<extra></extra>"))
    for ch, name, lo, hi in CHS:
        fig.add_vline(x=lo - 0.5, line_dash="dot", line_color="#ccc")
        fig.add_annotation(x=(lo + hi) / 2, y=11.6, text=f"第{ch}章", showarrow=False,
                           font=dict(size=9, color="#888"))
    fig.update_layout(title="全书 100 课按章节分布(横轴=课号,纵轴=章节)",
                      xaxis=dict(range=[0, 101]), yaxis=dict(range=[0.5, 11.5], tickvals=list(range(1, 12))),
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧵 当前主线:各课连起来看")
l_lessons = [n for n in LINES[line] if rng[0] <= n <= rng[1]]
fig2 = go.Figure(go.Scatter(x=l_lessons, y=[0.5] * len(l_lessons), mode="markers+text",
                            marker=dict(size=15, color="#c0392b"),
                            text=[f"L{n}" for n in l_lessons], textposition="bottom center"))
fig2.add_trace(go.Scatter(x=l_lessons, y=[0.5] * len(l_lessons), mode="lines",
                          line=dict(color="#c0392b", width=2), showlegend=False))
fig2.update_layout(title=f"主线『{line}』的进阶路径(可再叠加范围筛选)", yaxis=dict(range=[0, 1], showticklabels=False),
                   xaxis=dict(range=[0, 101], title="课号"), height=260,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 各章课数统计")
fig3 = go.Figure(go.Bar(x=[f"第{c[0]}章" for c in CHS], y=[c[3] - c[2] + 1 for c in CHS],
                        text=[c[3] - c[2] + 1 for c in CHS], textposition="outside",
                        marker_color="#2e86c1"))
fig3.update_layout(title="每章课数:1-6 章打基础,9-11 章深入工具链", yaxis_title="课数",
                   height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
> 💡 **给毕业生的三句话**:① 会用 vLLM 只是起点,懂它**为什么快**(内存/调度/编译)才是
> 面试与调优的分水岭;② 手写 Triton 让你从『调包』变『造轮子』;③ 多硬件视野(GPU + 昇腾)
> 是 2026 年的加分项 —— 你已经在路上了。
""")
