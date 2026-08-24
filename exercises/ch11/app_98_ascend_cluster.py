# -*- coding: utf-8 -*-
# app_98_ascend_cluster.py — 昇腾集群通信:HCCL AllReduce 交互 🌐
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾集群通信 HCCL 🌐", layout="wide")
st.title("🌐 第 98 课 · 昇腾集群通信:HCCL 与 AllReduce 交互")

st.markdown("""
多卡推理(张量并行)需要把每张卡的梯度/中间结果**求和汇总**,这就是集合通信。
HCCL 是昇腾的集合通信库(对标 NCCL),最经典的是 **Ring AllReduce**:把数据切成 N 片,
沿环走 **2(N-1) 步**,每步每卡只搬 1/N 的数据。拖一拖卡数,看步数与通信量怎么变。
""")

st.sidebar.header("🎛️ 参数")
n_rank = st.sidebar.slider("卡数(rank)", 2, 32, 8, 1)
algo = st.sidebar.radio("通信算法", ["Ring AllReduce", "Tree(Recursive Halving)"])
data = st.sidebar.slider("每卡数据量(MB)", 1, 512, 128, 1)
lat = st.sidebar.slider("单步延迟(us)", 5, 50, 20, 5)
st.sidebar.caption("Ring 步骤多但每步搬得少;Tree 步骤少但每步搬得多 —— 大数据量 Ring 胜,小数据量 Tree 胜。")

steps = 2 * (n_rank - 1) if algo == "Ring AllReduce" else 2 * int(np.ceil(np.log2(n_rank)))
per_step = data / n_rank if algo == "Ring AllReduce" else data / 2
total_t = steps * (lat + per_step * 12)          # 时间 = 步数 × (延迟 + 每步数据 × 带宽系数)
t_other = (2 * int(np.ceil(np.log2(n_rank))) * (lat + data / 2 * 12)) if algo == "Ring AllReduce"     else (2 * (n_rank - 1) * (lat + data / n_rank * 12))

c1, c2, c3, c4 = st.columns(4)
c1.metric("通信步数", steps)
c2.metric("每步数据量", f"{per_step:.1f} MB")
c3.metric("估算总耗时", f"{total_t/1000:.2f} ms")
c4.metric("另一算法耗时", f"{t_other/1000:.2f} ms")

st.subheader("📈 卡数 → 通信步数与数据量")
ns = np.arange(2, 33)
if algo == "Ring AllReduce":
    s = 2 * (ns - 1); d = data / ns
else:
    s = 2 * np.ceil(np.log2(ns)); d = data / 2
fig = go.Figure()
fig.add_trace(go.Scatter(x=ns, y=s, mode="lines", name="步数",
                         line=dict(color="#2e86c1", width=3)))
fig.add_trace(go.Scatter(x=ns, y=d, mode="lines", name="每步数据量(MB)",
                         line=dict(color="#e67e22", width=3), yaxis="y2"))
fig.update_layout(title=f"{algo}:卡数增多时,步数与每步数据量的取舍",
                  xaxis_title="rank 数", yaxis=dict(title="通信步数"),
                  yaxis2=dict(title="每步数据量(MB)", overlaying="y", side="right"),
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🛰️ 环形拓扑示意图")
fig2 = go.Figure()
theta = np.linspace(0, 2*np.pi, n_rank, endpoint=False)
for i in range(n_rank):
    x0, y0 = np.cos(theta[i]), np.sin(theta[i])
    x1, y1 = np.cos(theta[(i+1) % n_rank]), np.sin(theta[(i+1) % n_rank])
    fig2.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                              line=dict(color="#bbb", width=2), showlegend=False))
    fig2.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers+text",
                              marker=dict(size=22, color="#2e86c1"), text=[f"R{i}"],
                              textposition="top center", showlegend=False))
fig2.update_layout(title=f"{n_rank} 卡环形拓扑(数据沿环流动)",
                   height=340, xaxis=dict(showticklabels=False, range=[-1.4, 1.4]),
                   yaxis=dict(showticklabels=False, range=[-1.4, 1.4]),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:Ring 的通信量(每卡)是 `2S(N-1)/N`,随卡数增多趋近 `2S`;Tree 的步数
> 只随 `log2(N)` 增长。工程上:**大张量、卡数多 → Ring;小张量、延迟敏感 → Tree/Hierarchical**。
> vLLM-Ascend 的张量并行与数据并行正是靠 HCCL 这些原语跑起来的。
""")
