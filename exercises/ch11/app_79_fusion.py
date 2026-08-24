# -*- coding: utf-8 -*-
# app_79_fusion.py — 图算融合(Graph Kernel Fusion)开关对比 🧬
import time
import streamlit as st
import plotly.graph_objects as go
import torch

st.set_page_config(page_title="图算融合 🧬", layout="wide")
st.title("🧬 第 79 课 · 图算融合:融合开关对比")

st.markdown("""
昇腾 CANN 的**图引擎 GE** 会在编译期做**图算融合(Graph Kernel Fusion)**:把多个相邻
算子合并成一个大 kernel,省掉中间张量的『写出去再读回来』。下方**打开 / 关闭融合**、
调整**链式算子个数**与**张量大小**,实时对比耗时与访存量。
""")

fuse = st.sidebar.checkbox("启用算子融合", value=True)
n_ops = st.sidebar.slider("链式算子个数", 2, 8, 4, 1)
size = st.sidebar.slider("张量元素数", 100_000, 10_000_000, 2_000_000, 100_000)
show_dag = st.sidebar.checkbox("显示融合前后 DAG", value=True)
st.sidebar.caption("融合的核心收益:中间张量不再落回全局内存(省 2×访存/算子)。")

torch.manual_seed(0)
x = torch.randn(size)

def chain(x, n):
    y = x
    for i in range(n):
        y = y * 1.0001 + 0.5        # 逐元素算子链(加法、乘法)
    return y

def bench(fn, iters=10):
    ts = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - ts) / iters * 1000

if fuse:
    t = bench(lambda: chain(x, n_ops))
else:
    def chain_sep(x, n):
        y = x
        for i in range(n):
            y = y * 1.0001
            y = y + 0.5              # 每个算子单独一轮(模拟不融合,中间落盘)
        return y
    t = bench(lambda: chain_sep(x, n_ops))

traffic = size * 4 * 2 if fuse else size * 4 * 2 * n_ops   # 每算子读+写 8B

c1, c2, c3, c4 = st.columns(4)
c1.metric("单次耗时", f"{t:.3f} ms")
c2.metric("算子数", f"{2 * n_ops} 个基础算子")
c3.metric("访存量", f"{traffic/1e6:.0f} MB")
c4.metric("融合状态", "开" if fuse else "关")

if show_dag:
    st.subheader("🌳 融合前后 DAG")
    fig = go.Figure()
    if fuse:
        fig.add_trace(go.Scatter(x=[0.2], y=[0.5], mode="markers+text", marker=dict(size=60, color="#4C78A8"),
                                 text=[f"Fused kernel\n(n_ops×2 算子合并)"], textposition="middle center", textfont=dict(size=12)))
        fig.add_annotation(x=0.2, y=-0.15, text="输入 x", showarrow=False)
        fig.add_annotation(x=0.2, y=1.15, text="输出 y", showarrow=False)
    else:
        xs = [i / (n_ops * 2 + 1) for i in range(1, n_ops * 2 + 1)]
        fig.add_trace(go.Scatter(x=xs, y=[0.5] * len(xs), mode="markers",
                                 marker=dict(size=26, color="#E45756"),
                                 text=[f"op{i}" for i in range(1, len(xs) + 1)],
                                 textposition="top center"))
        fig.add_annotation(x=0.0, y=0.5, text="输入", showarrow=False)
        fig.add_annotation(x=1.0, y=0.5, text="输出", showarrow=False)
    fig.update_xaxes(showticklabels=False, range=[-0.1, 1.1])
    fig.update_yaxes(showticklabels=False, range=[-0.3, 1.4])
    fig.update_layout(title="算子链:红点=未融合的独立算子,蓝块=融合后的单 kernel",
                      height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("融合后中间张量不再落盘,访存从 2n 次降到 2 次。")

st.subheader("📊 不同张量大小下的耗时")
sizes = [100_000, 500_000, 1_000_000, 5_000_000]
times_on = [bench(lambda: chain(torch.randn(s), n_ops)) for s in sizes]
def chain_sep(x, n):
    y = x
    for i in range(n):
        y = y * 1.0001
        y = y + 0.5
    return y
times_off = [bench(lambda: chain_sep(torch.randn(s), n_ops)) for s in sizes]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sizes, y=times_off, mode="lines+markers", name="不融合(逐算子)",
                          line=dict(color="#E45756", width=3)))
fig2.add_trace(go.Scatter(x=sizes, y=times_on, mode="lines+markers", name="融合(单 kernel)",
                          line=dict(color="#4C78A8", width=3)))
fig2.update_layout(xaxis_type="log", title="融合 vs 不融合:张量越大访存节省越明显",
                   xaxis_title="张量元素数(对数)", yaxis_title="耗时(ms)", height=380)
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 观察:算子数越多、张量越大,融合收益越大 —— 大模型推理的逐元素链正是融合的富矿。")

st.markdown("""
> 💡 **一句话**:GE 图算融合 ≈ torch.compile 的融合 —— 都是把『小算子链』压成
> 『一个大 kernel』。昇腾把它放在编译期自动完成,开发者无感但收益常在。
""")
