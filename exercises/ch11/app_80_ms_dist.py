# -*- coding: utf-8 -*-
# app_80_ms_dist.py — MindSpore 分布式训练:并行策略选择 🌐
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="MindSpore 分布式 🌐", layout="wide")
st.title("🌐 第 80 课 · MindSpore 分布式训练:并行策略选择")

st.markdown("""
MindSpore 支持**数据并行(DP)、模型并行(MP)、流水线并行(PP)与自动并行**。
下方选择模型规模与并行策略,对比**每卡显存、AllReduce 通信时间与理论吞吐**,
体会『切数据 vs 切模型』的成本差异。
""")

model = st.sidebar.selectbox("模型规模", ["7B", "70B", "405B"])
strategy = st.sidebar.radio("并行策略", ["数据并行 DP", "模型并行 MP(切权重)", "流水线并行 PP(切层)", "自动并行"])
gpus = st.sidebar.slider("卡数", 1, 64, 8, 1)
batch = st.sidebar.slider("batch(每卡)", 1, 64, 16, 1)
st.sidebar.caption("数据并行切『数据』、模型并行切『权重』、流水线切『层』 —— 三把剪刀切三处。")

params_b = {"7B": 7, "70B": 70, "405B": 405}[model]
bytes_el = 2                       # fp16
weights_total = params_b * bytes_el
bandwidth_gbps = 100               # 示意:昇腾 HCCS/网络带宽

if strategy == "数据并行 DP":
    w_per = weights_total
    kv_per = 0
    grad_comm = weights_total * 2 * (gpus - 1) / gpus    # AllReduce 通信量(每卡)
    comm_ms = grad_comm / (bandwidth_gbps / 8)
    speed = min(gpus, 16)          # 吞吐近线性但通信占用
elif strategy == "模型并行 MP(切权重)":
    w_per = weights_total / gpus
    kv_per = 0
    grad_comm = 0                  # 切权重的卡间通信为 allreduce 每算子
    comm_ms = weights_total * 2 / (bandwidth_gbps / 8)
    speed = gpus
elif strategy == "流水线并行 PP(切层)":
    w_per = weights_total / gpus
    kv_per = 0
    comm_ms = weights_total / gpus / (bandwidth_gbps / 8) * 0.5
    speed = gpus * 0.7             # 有气泡
else:
    w_per = weights_total / max(gpus, 2)
    kv_per = 0
    comm_ms = weights_total * 1.5 / (bandwidth_gbps / 8)
    speed = gpus * 0.9

c1, c2, c3, c4 = st.columns(4)
c1.metric("模型权重(总量)", f"{weights_total:.0f} GB")
c2.metric("每卡权重", f"{w_per:.1f} GB")
c3.metric("通信时间(示意)", f"{comm_ms:.1f} ms")
c4.metric("相对吞吐", f"{speed:.0f}")

st.subheader("📊 每卡显存 vs 通信时间")
strategies = ["数据并行 DP", "模型并行 MP", "流水线并行 PP"]
wps = [weights_total, weights_total / gpus, weights_total / gpus]
comms = [weights_total * 2 * (gpus - 1) / gpus, weights_total * 2, weights_total / gpus * 0.5]
fig = go.Figure()
fig.add_trace(go.Bar(x=strategies, y=wps, name="每卡权重(GB)", marker_color="#4C78A8",
                     text=[f"{v:.0f}" for v in wps], textposition="outside"))
fig.add_trace(go.Bar(x=strategies, y=comms, name="通信量(GB,示意)", marker_color="#E45756",
                     text=[f"{v:.0f}" for v in comms], textposition="outside"))
fig.update_layout(barmode="group", title=f"{model} 在 {gpus} 卡上:三策略的显存与通信",
                  yaxis_title="GB", height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 观察:DP 每卡都要装整份权重(显存最大、通信跟卡数走);MP/PP 摊薄权重但引入通信。")

st.subheader("📈 AllReduce 通信量随卡数变化")
cards = list(range(2, 65))
comm_dp = [weights_total * 2 * (g - 1) / g for g in cards]
fig2 = go.Figure(go.Scatter(x=cards, y=comm_dp, mode="lines+markers",
                            line=dict(color="#B279A2", width=3), name="DP AllReduce/卡"))
fig2.add_hline(y=weights_total * 2, line_dash="dash", line_color="#4C78A8",
               annotation_text="MP 每次权重同步量(固定)", annotation_position="top right")
fig2.update_layout(title=f"{model}:DP 通信量随卡数趋近于『两倍权重』,MP 则固定",
                   xaxis_title="卡数", yaxis_title="通信量(GB)", height=360)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Ring-AllReduce 下每卡通信量 = 2(N-1)/N × 数据量,卡越多单卡通信反而越小,但总通信线性涨。")

st.markdown("""
> 💡 **结论**:MindSpore 的自动并行帮你决定『三把剪刀怎么下』;推理场景(vLLM 的 TP/PP)
> 用的是同一套思想 —— 切权重省显存,切层降通信,数据并行提吞吐。
""")
