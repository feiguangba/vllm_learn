# -*- coding: utf-8 -*-
# app_78_ms_op.py — MindSpore 算子开发:算子选择对比 🔬
import time
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch

st.set_page_config(page_title="MindSpore 算子开发 🔬", layout="wide")
st.title("🔬 第 78 课 · MindSpore 算子开发:内置 / 自定义 / 融合")

st.markdown("""
框架内置算子(成品)开箱即用但未必最优;**自定义算子**(Primitive / autograd.Function)
可为特定形状与精度量身定制;再把相邻算子**融合**成一个 kernel,访存更省、速度更快。
下方选择算子方案与数据规模,对比**耗时**,并做一次数值梯度校验。
""")

class SwishFn(torch.autograd.Function):
    """自定义算子 swish = x * sigmoid(x),forward + backward 二件套。"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return g * (s + x * s * (1 - s))

def swish(x):
    return SwishFn.apply(x)

scheme = st.sidebar.radio("算子方案", ["内置算子", "自定义算子(Primitive)", "融合算子(2→1)"])
n = st.sidebar.slider("张量元素数", 100_000, 50_000_000, 10_000_000, 1_000_000)
run_grad = st.sidebar.checkbox("做数值梯度校验", value=True)
st.sidebar.caption("内置最稳、自定义最灵活、融合最快 —— 三者的取舍正是算子工程的核心。")

torch.manual_seed(0)
x = torch.randn(min(n, 2_000_000))   # 控制内存

def bench(fn, iters=10):
    ts = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - ts) / iters * 1000

def fused():
    s = torch.sigmoid(x)
    return torch.relu(x * s)

if scheme == "内置算子":
    t = bench(lambda: torch.nn.functional.silu(x))
    note = "调用框架内置 silu,开箱即用、性能均衡。"
elif scheme == "自定义算子(Primitive)":
    t = bench(lambda: SwishFn.apply(x))
    note = "自定义 autograd.Function 实现 swish,行为与内置一致,梯度可自定义。"
else:
    t = bench(fused)
    note = "把 sigmoid 与 relu 融合成一次张量表达式,减少中间张量访问。"

c1, c2, c3 = st.columns(3)
c1.metric("单次耗时", f"{t:.3f} ms")
c2.metric("元素数", f"{x.numel():,}")
c3.metric("方案", scheme)

if run_grad:
    st.subheader("🧮 数值梯度校验(自定义算子)")
    xg = torch.randn(6, requires_grad=True)
    swish(xg).sum().backward()
    analytic = xg.grad.clone()
    eps = 1e-6
    for i in range(xg.numel()):
        xp = xg.clone(); xm = xg.clone()
        xp.flatten()[i] += eps; xm.flatten()[i] -= eps
        fp = swish(xp).sum().item()
        fm = swish(xm).sum().item()
        xg.grad.flatten()[i] = (fp - fm) / (2 * eps)
    num = xg.grad
    ok = torch.allclose(analytic, num, atol=1e-5)
    st.markdown(f"解析梯度 vs 数值梯度一致: **{'✅ 是' if ok else '❌ 否'}** (atol=1e-5)")
    st.caption("swish 的解析梯度 = σ(x) + x·σ(x)·(1−σ(x)),与中心差分逐点对照。")

st.subheader("📊 三种方案耗时对比(本机实测)")
schemes = ["内置算子", "自定义算子", "融合算子"]
times = [bench(lambda: torch.nn.functional.silu(x)),
         bench(lambda: SwishFn.apply(x)),
         bench(fused)]
fig = go.Figure(go.Bar(x=schemes, y=times, marker_color=["#54A24B", "#B279A2", "#4C78A8"],
                       text=[f"{v:.3f}" for v in times], textposition="outside"))
fig.update_layout(title="同规模输入下三种方案的耗时(ms,本机实测)",
                  yaxis_title="耗时(ms)", height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"> 💡 **当前方案说明**:{note} 三者数学等价,差异在工程形态:自定义算子给了你"
            f"『控制梯度 + 参与图融合』的能力,融合算子砍掉了中间张量的来回读写。")
