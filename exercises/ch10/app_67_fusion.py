# -*- coding: utf-8 -*-
# app_67_fusion.py — Kernel 融合实战:融合开关对比 🧬
import streamlit as st
import plotly.graph_objects as go
import os, json

st.set_page_config(page_title="Kernel 融合实战 🧬", layout="wide")
st.title("🧬 第 67 课 · Kernel 融合实战:融合开关对比")

st.markdown("""
**逐元素 / 广播**算子最值得融合:它们形状一致、访存密集,融合后中间结果留在芯片内、kernel 启动
次数骤减。下方选一个计算场景、拨张量规模,对比『不融合(eager)』与『融合(torch.compile)』的
kernel 数与耗时。
""")

# ---------------- 实测数据(notebook 生成的 fusion_67.json)----------------
DATA = [
    {"scenario": "elementwise 链(6算子)", "eager_kernels": 13, "compiled_kernels": 1, "eager_ms": 2.77, "compiled_ms": 0.41},
    {"scenario": "broadcast 乘加", "eager_kernels": 3, "compiled_kernels": 1, "eager_ms": 0.15, "compiled_ms": 0.04},
    {"scenario": "softmax(归约)", "eager_kernels": 5, "compiled_kernels": 1, "eager_ms": 0.31, "compiled_ms": 0.09},
]
_j = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion_67.json")
if os.path.exists(_j):
    try:
        DATA = json.load(open(_j, encoding="utf-8"))
    except Exception:
        pass

st.sidebar.header("🎛️ 参数")
scenario = st.sidebar.selectbox("计算场景", [d["scenario"] for d in DATA])
size = st.sidebar.selectbox("张量规模", [1024, 2048, 4096, 8192], index=2, format_func=lambda v: f"{v}²")
metric = st.sidebar.radio("查看指标", ["kernel 数", "耗时(ms)"], horizontal=True)
show_traffic = st.sidebar.checkbox("显示中间读写量", value=True)
st.sidebar.caption("融合省下的是『开火手续费 + 中间读写』;算子越多、规模越大,省得越多。")

row = next(d for d in DATA if d["scenario"] == scenario)
scale = size / 4096
eager_k = max(1, int(row["eager_kernels"])); comp_k = row["compiled_kernels"]
eager_ms = row["eager_ms"] * scale
comp_ms = row["compiled_ms"] * scale

c1, c2, c3, c4 = st.columns(4)
c1.metric("不融合 kernel 数", eager_k)
c2.metric("融合后 kernel 数", comp_k)
c3.metric("kernel 削减率", f"{100 * (1 - comp_k / eager_k):.0f}%")
c4.metric("加速比", f"{eager_ms / max(comp_ms, 1e-6):.2f}x")

# ---------------- 指标图 ----------------
fig = go.Figure()
if metric == "kernel 数":
    fig.add_bar(x=["不融合(eager)", "融合(compile)"], y=[eager_k, comp_k],
                marker_color=["#c0392b", "#27ae60"], text=[eager_k, comp_k], textposition="outside")
    fig.update_layout(title=f"{scenario}:kernel 数对比", yaxis_title="kernel 数")
else:
    fig.add_bar(x=["不融合(eager)", "融合(compile)"], y=[eager_ms, comp_ms],
                marker_color=["#c0392b", "#27ae60"],
                text=[f"{eager_ms:.2f}ms", f"{comp_ms:.2f}ms"], textposition="outside")
    fig.update_layout(title=f"{scenario}:耗时对比(按规模外推)", yaxis_title="耗时(ms)")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

if show_traffic:
    st.subheader("🧮 中间读写量")
    bytes_t = size * size * 4
    unfused = (eager_k - 1) * 2 * bytes_t
    fused = bytes_t
    fig2 = go.Figure(go.Bar(x=["不融合", "融合"], y=[unfused / 1e6, fused / 1e6],
                            marker_color=["#c0392b", "#27ae60"],
                            text=[f"{unfused/1e6:.0f}MB", f"{fused/1e6:.0f}MB"], textposition="outside"))
    fig2.update_layout(title="中间张量读写量(内存往返)", yaxis_title="MB", height=340,
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("⭐ 中间结果留在寄存器/缓存,是融合提速的核心来源。")

st.markdown("""
> 💡 **结论**:逐元素/广播/轻归约算子融合收益最直接 —— kernel 变少、中间读写变少。
> 重算子(如大矩阵乘)本身已很优,一般保持独立,这正是编译器『该融才融』的分寸。
""")
