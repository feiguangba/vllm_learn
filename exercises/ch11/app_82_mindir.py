# -*- coding: utf-8 -*-
# app_82_mindir.py — MindIR 中间表示浏览器:计算图 DAG 一览 📦
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📦 82 · MindIR 中间表示", layout="wide")
st.title("📦 第 82 课 · MindIR 中间表示:像“话单”一样的模型文件")

st.markdown("""
**MindIR(MindSpore IR)** 是把训练好的模型固化成的一个**计算图文件**(`.mindir`):
里面存着算子列表(nodes)、张量流向(edges)与权重(parameters),不含任何 Python 代码。
它就像餐厅之间传递的**菜谱**——后厨(MindSpore)写完菜谱,任何会看菜谱的厨师
(云侧 Runtime / Lite / MindIE)都能照着做出同一道菜。
下方选择一个示例计算图,浏览它的节点 / 边,并在网络图中交互拖拽。
""")

GRAPHS = {
    "线性网络(3 算子)": {
        "nodes": [("input", "Parameter"), ("matmul", "MatMul"), ("add_bias", "Add"),
                  ("relu", "ReLU"), ("output", "Output")],
        "edges": [("input", "matmul"), ("matmul", "add_bias"), ("add_bias", "relu"), ("relu", "output")],
        "desc": "MindIR 里最常见的形状:一条笔直的数据流水线。",
    },
    "残差块(分支 + 汇合)": {
        "nodes": [("input", "Parameter"), ("conv", "Conv2D"), ("bn", "BatchNorm"),
                  ("skip", "Add(直连)"), ("relu", "ReLU"), ("output", "Output")],
        "edges": [("input", "conv"), ("conv", "bn"), ("bn", "skip"), ("input", "skip"), ("skip", "relu"), ("relu", "output")],
        "desc": "ResNet 残差块:主路卷积,旁路“抄近道”,最后汇合——IR 里就是一个分叉再合并的 DAG。",
    },
    "注意力算子(fused)": {
        "nodes": [("query", "Parameter"), ("key", "Parameter"), ("value", "Parameter"),
                  ("qk", "MatMul"), ("scale", "Mul"), ("softmax", "Softmax"), ("attn", "MatMul"), ("output", "Output")],
        "edges": [("query", "qk"), ("key", "qk"), ("qk", "scale"), ("scale", "softmax"),
                  ("softmax", "attn"), ("value", "attn"), ("attn", "output")],
        "desc": "昇腾上常被“融合”成一个算子的注意力模式:MatMul→Scale→Softmax→MatMul。",
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    gname = st.selectbox("选择示例计算图", list(GRAPHS.keys()))
    show_weights = st.checkbox("展示 Parameter 权重形状", value=True)
    node_size = st.slider("节点半径", 12, 30, 18, 1)
    st.caption("MindIR 的“节点”就是算子,“边”就是张量依赖。")

G = GRAPHS[gname]
nodes, edges = G["nodes"], G["edges"]
names = [n[0] for n in nodes]
kinds = [n[1] for n in nodes]

# 简单分层布局:按拓扑排序(先输入的在前)
pos = {n: i for i, n in enumerate(names)}
x = [pos[n] for n in names]
y = [0.5 - (0.6 if k in ("Parameter", "Output") else 0.0) for n, k in zip(names, kinds)]

c1, c2, c3 = st.columns(3)
c1.metric("节点数", len(nodes))
c2.metric("边数", len(edges))
c3.metric("是否含环", "否(DAG)")
st.caption(G["desc"])

# ---------------- 节点 / 边表格 ----------------
st.subheader("🗂️ 节点与边清单")
st.dataframe(pd.DataFrame({"节点": names, "类型": kinds}), width="stretch")
st.dataframe(pd.DataFrame({"源节点": [e[0] for e in edges], "目标节点": [e[1] for e in edges]}),
             width="stretch")
if show_weights:
    st.caption("Parameter 节点还携带权重张量(形状 / 类型),例如 MatMul 的 weight 形状 (16, 16)。")

# ---------------- 网络图 ----------------
fig = go.Figure()
for s, t in edges:
    fig.add_trace(go.Scatter(x=[x[pos[s]], x[pos[t]]], y=[y[pos[s]], y[pos[t]]],
                             mode="lines", line=dict(color="#B8C4D8", width=2),
                             hoverinfo="none", showlegend=False))
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers+text", text=names, textposition="top center",
    marker=dict(size=node_size * 2, color=["#E45756" if k == "Output" else "#4C78A8"
                                          for n, k in zip(names, kinds)], line=dict(width=1, color="white")),
    hovertext=[f"{n} ({k})" for n, k in zip(names, kinds)], showlegend=False))
fig.update_layout(title=f"MindIR 计算图: {gname}",
                  xaxis=dict(showticklabels=False, title="拓扑顺序"), yaxis=dict(showticklabels=False),
                  height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 鼠标悬停看算子类型,拖拽缩放看整条依赖链。残差块里的“分叉再汇合”就是 DAG 的标志。")

st.markdown("""
> 💡 **结论**:MindIR = 计算图(DAG)+ 权重 + 元信息,一次导出、多端复用。
> 它和 ONNX 同属“图 IR”,和 TorchScript 同属“图 + 代码”的过渡形态。
> 任何会“读图”的运行时都能执行它——这就是中间表示的威力。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 82 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
