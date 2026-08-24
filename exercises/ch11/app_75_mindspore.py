# -*- coding: utf-8 -*-
# app_75_mindspore.py — MindSpore 张量运算交互 🧮
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch

st.set_page_config(page_title="MindSpore 张量 🧮", layout="wide")
st.title("🧮 第 75 课 · MindSpore 基础:张量运算交互")

st.markdown("""
MindSpore(昇思)的编程模型与 PyTorch 高度神似:**张量(Tensor)+ 算子(ops)+ 网络(nn)+
自动微分(autograd)**。本演示用 **torch 模拟 MindSpore 的 API 设计理念**:
调整张量的形状、精度与运算,实时看结果统计、内存占用与自动微分示例。
""")

shape_r = st.sidebar.slider("行数", 1, 16, 4, 1)
shape_c = st.sidebar.slider("列数", 1, 16, 6, 1)
dtype = st.sidebar.selectbox("dtype", ["fp32", "fp16", "bf16"])
op = st.sidebar.selectbox("运算", ["square 平方", "exp 指数", "relu", "matmul × 转置"])
show_grad = st.sidebar.checkbox("展示自动微分示例", value=True)
st.sidebar.caption("mindspore.Tensor 与 torch.tensor 概念一一对应,API 可查官方映射表。")

shape = (shape_r, shape_c)
torch.manual_seed(0)
x = torch.randn(*shape)
bytes_el = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
xt = {"fp32": x.float(), "fp16": x.half(), "bf16": x.bfloat16()}[dtype]

out = {"square 平方": xt ** 2, "exp 指数": torch.exp(xt.float()),
       "relu": torch.relu(xt), "matmul × 转置": xt @ xt.t()}[op]

c1, c2, c3 = st.columns(3)
c1.metric("shape", f"{list(shape)}")
c2.metric("dtype", dtype)
c3.metric("内存占用", f"{shape_r * shape_c * bytes_el / 1024:.2f} KB")

st.subheader("📊 输入张量视图(热力图)")
data = xt.float().cpu().numpy()
fig = go.Figure(data=go.Heatmap(z=data, colorscale="YlGnBu",
                                text=np.round(data, 2), texttemplate="%{text}",
                                colorbar_title="数值"))
fig.update_layout(title=f"输入张量 {shape} · {dtype}", height=300,
                  yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"🔧 运算结果: {op}")
ov = out.float().cpu().numpy().ravel()
c1, c2, c3, c4 = st.columns(4)
c1.metric("min", f"{ov.min():.4f}")
c2.metric("max", f"{ov.max():.4f}")
c3.metric("mean", f"{ov.mean():.4f}")
c4.metric("std", f"{ov.std():.4f}")

fig2 = go.Figure(go.Bar(x=[f"r{i}" for i in range(shape_r)],
                        y=out.float().abs().sum(dim=1).cpu().numpy(),
                        marker_color="#4C78A8", textposition="outside"))
fig2.update_layout(title="每行 L1 范数(绝对值求和)", height=300,
                   xaxis_title="行", yaxis_title="Σ|值|", margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 观察:改变 shape / dtype / 运算,热力图与行范数实时变化 —— 张量是这一切的公共语言。")

if show_grad:
    st.subheader("🧮 自动微分示例(模拟 mindspore.GradOperation)")
    xg = torch.tensor([[2.0, 3.0]], requires_grad=True)
    y = (xg ** 2).sum() + 3 * xg.sum() + 1
    y.backward()
    st.markdown(f"对 **f(x)=x²+3x+1** 在 x=[2,3] 处求梯度:dy/dx = **{xg.grad.tolist()}**(解析式 2x+3 的精确值)")
    st.markdown("MindSpore 里写法是 `mindspore.GradOperation()(net, inputs)` → 返回同样的梯度向量。")
    st.caption("grad 是逐个元素求导:∂f/∂x = 2x+3,在 x=2 时为 7,在 x=3 时为 9。")

st.markdown("""
> 💡 **一句话**:MindSpore 四大件(Tensor / ops / nn / GradOperation)与 PyTorch 同构,
> 区别主要在『图模式编译』与『昇腾下沉』(第 76-77 课)。会用 torch,就能很快上手 MindSpore。
""")
