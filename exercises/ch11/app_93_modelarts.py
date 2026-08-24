# -*- coding: utf-8 -*-
# app_93_modelarts.py — ModelArts 与昇腾云:云上 AI 流程交互 ☁️
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ModelArts 与昇腾云 ☁️", layout="wide")
st.title("☁️ 第 93 课 · ModelArts 与昇腾云:云服务流程交互")

st.markdown("""
ModelArts 是华为云的 AI 一站式平台:像把整条"AI 生产线"外包给云厂商 ——
**要训练就开训练作业、要推理就发推理服务、不想写代码就交给自动学习**。
下面选一种服务、拖一下卡数与时长,实时算出费用并看流程时间线。
""")

st.sidebar.header("🎛️ 参数")
svc = st.sidebar.selectbox("服务类型", ["训练作业", "在线推理服务", "自动学习"])
pool = st.sidebar.radio("资源池", ["公共资源池", "专属资源池", "裸金属服务器"])
cards = st.sidebar.slider("昇腾 910 卡数", 1, 16, 8, 1)
hours = st.sidebar.slider("使用时长(小时)", 0.5, 24.0, 6.0, 0.5)
st.sidebar.caption("公共池按量计费最贵但零门槛;专属池包年便宜但要规划;裸金属完全独占最灵活。")

# 单价:公共池按卡时(¥/卡时),专属池按包年折算,裸金属按整机(¥/时)
if pool == "公共资源池":
    unit, label = 42.0, "¥/卡时(按量)"
elif pool == "专属资源池":
    unit, label = 26.0, "¥/卡时(包年折算)"
else:
    unit, label = 260.0, "¥/整机时"

if svc == "训练作业":
    total = unit * cards * hours
    steps = ["数据准备", "作业排队", "训练运行", "产物归档"]
    times = [0.5, 0.3, hours, 0.2]
else:
    total = unit * cards * hours * 1.6
    steps = ["模型发布", "服务拉起", "持续推理", "监控扩缩"]
    times = [0.3, 0.4, hours, 0.3]

c1, c2, c3, c4 = st.columns(4)
c1.metric("单价", f"{unit:.1f} {label}")
c2.metric("估算费用", f"¥{total:,.0f}")
c3.metric("昇腾 910 卡数", cards)
c4.metric("服务类型", svc)

st.subheader("🛠️ 作业流程时间线")
fig = go.Figure(go.Bar(x=steps, y=times, marker_color="#2e86c1",
                       text=[f"{t:.1f} h" for t in times], textposition="outside"))
fig.update_layout(title=f"{svc}({pool})的流程时间分布", yaxis_title="时长(小时)",
                  height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 费用 vs 使用时长(三种资源池)")
hs = np.arange(1, 25)
costs = {
    "公共资源池": 42.0 * cards * hs,
    "专属资源池": 26.0 * cards * hs,
    "裸金属服务器": 260.0 * hs,
}
fig2 = go.Figure()
colors = {"公共资源池": "#e74c3c", "专属资源池": "#27ae60", "裸金属服务器": "#2e86c1"}
for name, c in costs.items():
    fig2.add_trace(go.Scatter(x=hs, y=c, mode="lines", name=name,
                              line=dict(color=colors[name], width=3)))
fig2.add_vline(x=hours, line_dash="dash", line_color="#888")
fig2.update_layout(title="三种资源池的费用随使用时长增长(昇腾 910)",
                   xaxis_title="使用时长(小时)", yaxis_title="估算费用(¥)",
                   height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:云上的核心优势是**弹性** —— 想用 1000 卡就开 1000 卡,用完即还;
> 本地部署则胜在**可控与数据不出域**。选公共池跑实验、专属池跑生产、裸金属跑特殊负载,是常见分工。
""")
