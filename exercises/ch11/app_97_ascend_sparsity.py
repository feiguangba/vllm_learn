# -*- coding: utf-8 -*-
# app_97_ascend_sparsity.py — 昇腾量化与稀疏化:参数交互实验 🔢
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾量化与稀疏化 🔢", layout="wide")
st.title("🔢 第 97 课 · 昇腾量化与稀疏化:瘦身提速交互实验")

st.markdown("""
**量化** = 用更少的位存权重(W8A8 / FP8),显存减半、计算变快;**稀疏化** = 只保留重要的
权重(2:4 结构化稀疏 / 剪枝),跳过零运算。两者都是"用一点精度换一大截性能"。
拖一拖参数,观察精度损失与加速比的此消彼长。
""")

st.sidebar.header("🎛️ 参数")
bits = st.sidebar.slider("量化位数", 4, 16, 8, 1)
sparse_ratio = st.sidebar.slider("稀疏比例(0-90%)", 0, 90, 50, 5)
mode = st.sidebar.radio("稀疏模式", ["非结构化剪枝", "2:4 结构化稀疏"])
weight_gb = st.sidebar.slider("权重规模(GB)", 1, 64, 14, 1)
st.sidebar.caption("量化位数越低越省,但误差越大;稀疏比例越高越快,但精度掉得越快 —— 鱼与熊掌。")

# ---- 量化误差模拟:误差 ~ 2^(-bits+1) ----
qerr = 2 ** -(bits - 1) * 0.02
mem = weight_gb * (bits / 16)                     # 位宽减半 → 权重减半
# ---- 稀疏加速模拟 ----
if mode == "2:4 结构化稀疏":
    eff = min(50, sparse_ratio) * 0.9             # 2:4 的硬件收益更稳定
else:
    eff = min(50, sparse_ratio) * 0.7             # 非结构化收益打折
speedup = 1 + eff / (100 - eff + 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("量化位数", f"{bits} bit")
c2.metric("相对精度损失", f"{qerr*100:.2f}%")
c3.metric("权重占用", f"{mem:.1f} GB(省 {100*(1-bits/16):.0f}%)")
c4.metric("稀疏加速(示意)", f"{speedup:.2f}×")

st.subheader("📉 量化误差 vs 位数")
b = np.arange(4, 17)
errs = 2 ** -(b - 1) * 0.02 * 100
fig = go.Figure()
fig.add_trace(go.Scatter(x=b, y=errs, mode="lines+markers",
                         line=dict(color="#c0392b", width=3), name="相对误差 %"))
fig.add_vline(x=bits, line_dash="dash", line_color="#888")
fig.update_layout(title="量化位数越低,误差指数级上升(对数感)", xaxis_title="位数(bits)",
                  yaxis_title="相对误差(%)", height=320,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧊 稀疏矩阵可视化(模拟)")
n = 16
rng = np.random.default_rng(0)
w = rng.normal(size=(n, n))
w[abs(w) < np.quantile(abs(w), 1 - sparse_ratio / 100)] = 0   # 剪掉最小的一批
fig2 = go.Figure(go.Heatmap(z=w, colorscale="RdBu", zmid=0,
                            showscale=False,
                            hovertemplate="值: %{z:.3f}<extra></extra>"))
fig2.update_layout(title=f"稀疏权重矩阵({mode},稀疏比例 {sparse_ratio}%)",
                   height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:量化管「存得少、算得快」,稀疏管「跳着算」。生产实践通常**先量化后稀疏**,
> 且用「精度回退」指标监控 —— 一旦准确率掉出红线,就回调一档。昇腾上 MindIE 已内置
> W8A8 与 KV 量化,稀疏则多见于剪枝后的推理模型。
""")
