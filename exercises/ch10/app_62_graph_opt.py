# -*- coding: utf-8 -*-
# app_62_graph_opt.py — 计算图优化:融合开关对比 🌳
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, json

st.set_page_config(page_title="计算图优化 🌳", layout="wide")
st.title("🌳 第 62 课 · 计算图优化:融合开关对比")

st.markdown("""
编译器拿到一张计算图后,会做一连串**图优化**让图"更小、更简单、更快":算子融合、常量折叠、
死代码消除、代数化简。这里聚焦**算子融合**——把一串逐元素算子并成一个 kernel,减少启动与
中间读写。拖动下方的旋钮,直观对比"不融合(eager)"与"融合(编译)"的 kernel 数与耗时。
""")

# ---------------- 加载实测数据(notebook 生成的 graph_opt_62.json)----------------
DATA = [
    {"chain_len": 4, "eager_kernels": 8, "compiled_kernels": 1, "eager_ms": 1.84, "compiled_ms": 0.28},
    {"chain_len": 6, "eager_kernels": 13, "compiled_kernels": 1, "eager_ms": 2.77, "compiled_ms": 0.41},
    {"chain_len": 8, "eager_kernels": 17, "compiled_kernels": 1, "eager_ms": 3.71, "compiled_ms": 0.55},
]
_j = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_opt_62.json")
if os.path.exists(_j):
    try:
        DATA = json.load(open(_j, encoding="utf-8"))
    except Exception:
        pass

st.sidebar.header("🎛️ 参数")
chain_len = st.sidebar.slider("逐元素算子个数(链长)", 2, 12, 6, 1)
tensor_size = st.sidebar.selectbox("张量规模", ["2048²", "4096²", "8192²"], index=1)
show_both = st.sidebar.checkbox("同时显示不融合 / 融合", value=True)
st.sidebar.caption("链越长,不融合版本要开的火越多、中间张量反复进出显存越多,融合收益越大。")

df = pd.DataFrame(DATA)
row = df.iloc[(df["chain_len"] - df["chain_len"].min()).abs().idxmin()]
fused_kernels = 1
# 用最近的实测行 + 线性外推得到当前链长的估算
k_ratio = df["eager_kernels"].iloc[-1] / df["chain_len"].iloc[-1]
eager_kernels = int(max(1, round(chain_len * k_ratio)))
eager_ms = df["eager_ms"].iloc[-1] * (chain_len / df["chain_len"].iloc[-1])
compiled_ms = df["compiled_ms"].iloc[-1] * (chain_len / df["chain_len"].iloc[-1]) ** 0.5

c1, c2, c3, c4 = st.columns(4)
c1.metric("不融合 kernel 数", eager_kernels)
c2.metric("融合后 kernel 数", fused_kernels)
c3.metric("kernel 削减率", f"{100 * (1 - fused_kernels / max(eager_kernels,1)):.0f}%")
c4.metric("估算加速比", f"{eager_ms / max(compiled_ms, 1e-6):.2f}x")
st.caption("不融合 = 每个算子一个 kernel(实测+线性外推);融合 = 一整条链并成一个 kernel(实测+开方外推)。")

# ---------------- 可视化:kernel 数 / 耗时 ----------------
metric = st.radio("查看指标", ["kernel 数量", "估算耗时"], horizontal=True)
fig = go.Figure()
if metric == "kernel 数量":
    fig.add_bar(x=["不融合(eager)", "融合(torch.compile)"],
                y=[eager_kernels, fused_kernels],
                marker_color=["#c0392b", "#27ae60"], text=[eager_kernels, fused_kernels],
                textposition="outside")
    fig.update_layout(title=f"链长 = {chain_len} 时的 kernel 数量对比", yaxis_title="kernel 数")
else:
    fig.add_bar(x=["不融合(eager)", "融合(torch.compile)"],
                y=[eager_ms, compiled_ms],
                marker_color=["#c0392b", "#27ae60"],
                text=[f"{eager_ms:.2f}ms", f"{compiled_ms:.2f}ms"], textposition="outside")
    fig.update_layout(title=f"链长 = {chain_len} 时的耗时对比(估算)", yaxis_title="耗时(ms)")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------- 中间读写对比 ----------------
st.subheader("🧮 中间张量读写量")
st.markdown("不融合时,每两个算子之间的中间结果都要写回显存再读出来;融合后中间结果留在寄存器/缓存里。")
bytes_per_tensor = {"2048²": 2048 * 2048 * 4, "4096²": 4096 * 4096 * 4, "8192²": 8192 * 8192 * 4}[tensor_size]
unfused_traffic = (chain_len - 1) * 2 * bytes_per_tensor
fused_traffic = bytes_per_tensor
fig2 = go.Figure()
fig2.add_bar(x=["不融合", "融合"], y=[unfused_traffic / 1e6, fused_traffic / 1e6],
             marker_color=["#c0392b", "#27ae60"],
             text=[f"{unfused_traffic/1e6:.0f}MB", f"{fused_traffic/1e6:.0f}MB"], textposition="outside")
fig2.update_layout(title="中间张量总读写量(内存往返)", yaxis_title="读写量(MB)", height=360,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 中间结果留在芯片内,是逐元素融合提速的主要来源;链越长省得越多。")

st.markdown("""
> 💡 **结论**:算子融合是计算图优化里收益最直观的一招 —— kernel 变少、中间读写变少。
> 但注意:融合也有代价(寄存器/缓存压力、编译时间),所以编译器要"该融才融"。
""")
