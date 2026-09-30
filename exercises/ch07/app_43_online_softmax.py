# -*- coding: utf-8 -*-
# app_43_online_softmax.py — 在线 softmax:逐块状态可视化 🧮
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧮 43 · 在线 Softmax", layout="wide")
st.title("🧮 第 43 课 · 在线 Softmax:边扫边修正")

st.markdown("""
softmax 的分母(所有 $e^{x_i}$ 之和)需要 **“看完一整行”** 才算得出来。可 FlashAttention 偏偏要
**分块扫描**、边算边丢。怎么办?用**在线 softmax**:维护一个**运行时最大值 $m$** 和**运行时总和 $l$**,
每看到一个块,就用「旧 $m$ 与当前块最大值」去**校正**已经累积的 $l$。
下方拖动**块大小**与**序列长度**,逐块观察 $m$、$l$ 如何一步步逼近全局值。
""")

def full_softmax(x):
    # 整体 softmax:减 max 再 exp 归一(数值稳定,但要读两遍)
    e = np.exp(x - x.max())
    return e / e.sum()

def online_softmax_trace(x, block):
    # 逐块记录 (m, l) 快照,用于可视化;m:running max, l:running sum
    m = -np.inf
    l = 0.0
    trace = []
    n = len(x)
    for i in range(0, n, block):
        xb = x[i:i + block]                    # 读一个块
        m_new = max(m, xb.max())               # 更新 running max
        l = l * np.exp(m - m_new) + np.exp(xb - m_new).sum()   # 用旧 m 与块内值校正再累加
        m = m_new
        trace.append((m, l, i // block + 1))
    return trace

def online_softmax(x, block):
    # 一趟扫描得到 m/l;最终再读一遍 x 做归一
    m = -np.inf
    l = 0.0
    n = len(x)
    for i in range(0, n, block):
        xb = x[i:i + block]
        m_new = max(m, xb.max())
        l = l * np.exp(m - m_new) + np.exp(xb - m_new).sum()
        m = m_new
    return np.exp(x - m) / l

with st.sidebar:
    st.header("🎛️ 参数")
    block = st.slider("块大小 Bm", 1, 16, 4, 1)
    n = st.slider("序列长度 N", 8, 64, 24, 1)
    mode = st.radio("数值分布", ["正态", "右偏(有尖峰)"])
    st.caption("块越小,在线 softmax 的 m/l 越“勤快”地更新;尖峰分布更能看出校正的过程。")

rng = np.random.default_rng(42)
if mode == "正态":
    x = rng.normal(0, 1, n)
else:
    x = rng.normal(0, 1, n)
    x[n // 2] = 20.0
    x[n // 2 + 1] = 15.0

if block > n:
    block = n

p_full = full_softmax(x)
p_on = online_softmax(x, block)
err = float(np.max(np.abs(p_full - p_on)))
trace = online_softmax_trace(x, block)
nblocks = len(trace)

c1, c2, c3 = st.columns(3)
c1.metric("块数", f"{nblocks}")
c2.metric("最大绝对误差 vs 整体", f"{err:.2e}")
c3.metric("最终 m(全局 max)", f"{trace[-1][0]:.3f}")

bs = [t[2] for t in trace]; ms = [t[0] for t in trace]; ls = [t[1] for t in trace]
fig = go.Figure()
fig.add_trace(go.Bar(x=bs, y=ms, name="running max m", marker_color="#4C78A8"))
fig.add_trace(go.Bar(x=bs, y=ls, name="running sum l", marker_color="#F2C14E", yaxis="y2"))
fig.add_hline(y=max(ms), line_dash="dot", line_color="#4C78A8",
              annotation_text=f"全局 max={max(ms):.3f}")
fig.update_layout(title=f"每处理一个块后,running m / l 的状态(块大小 {block})",
                  xaxis_title="已处理的块序号", yaxis_title="m", height=420,
                  yaxis2=dict(title="l", overlaying="y", side="right"),
                  legend=dict(orientation="h", y=1.12), barmode="overlay",
                  margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(range(n)), y=p_full, mode="lines+markers",
                          name="整体 softmax", line=dict(color="#4C78A8", width=2)))
fig2.add_trace(go.Scatter(x=list(range(n)), y=p_on, mode="markers",
                          name="在线 softmax", marker=dict(color="#E45756", size=5)))
fig2.update_layout(title="整体 vs 在线 softmax 的概率分布(应几乎重合)",
                   xaxis_title="位置 i", yaxis_title="概率", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.caption("💡 观察:m 单调不减(每次取 max),l 会先被校正(乘上旧最大值的衰减系数)再累加当前块;"
           "块越小,中间状态越多、但最终误差同样极小——在线 softmax 与整体 softmax 数值等价。")

st.markdown(r"""
> 💡 **结论**:在线 softmax 通过 $m \leftarrow \max(m, \text{块内max})$、
> $l \leftarrow l \cdot e^{m-m_{\text{new}}} + \sum e^{x-m_{\text{new}}}$ 的两行更新,
> 在不重新读回整行的情况下精确算出分母。这就是 FlashAttention 能分块扫描的理论地基。
> 来源:[Milakov & Gimelshein, 2018 (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 7 章 · 第 43 课配套演示")
