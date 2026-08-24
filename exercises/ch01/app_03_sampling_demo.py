# -*- coding: utf-8 -*-
# app_03_sampling_demo.py — 采样策略演示 🎲
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="采样策略 🎲", layout="wide")
st.title("🎲 第 03 课 · 采样策略:greedy / temperature / top-k / top-p")

st.markdown("""
模型每次生成前都会给候选 token 打分(logits)。**采样策略**决定怎么把 logits 变成
真正吐出来的那个字:温度控制分布的“陡峭度”,top-k / top-p 决定候选被压缩到多小。
下方拖动滑杆,实时观察概率分布如何变化,以及采样最终抽中的 token。
""")

VOCAB = ["猫", "狗", "鱼", "鸟", "马", "牛"]
LOGITS = np.array([3.2, 2.1, 1.5, 0.6, 0.2, -0.5], dtype=np.float64)

def softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()

def top_k_mask(logits, k):
    mask = np.zeros_like(logits, dtype=bool)
    if k <= 0 or k >= len(logits):
        mask[:] = True
    else:
        mask[np.argsort(logits)[-k:]] = True
    return mask

def top_p_mask(logits, p):
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = cum - probs[order] <= p
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    return mask

with st.sidebar:
    st.header("🎛️ 参数")
    temperature = st.slider("温度 temperature", 0.1, 3.0, 1.0, 0.1)
    top_k = st.slider("top-k(0=关闭)", 0, 6, 6, 1)
    top_p = st.slider("top-p(1.0=关闭)", 0.1, 1.0, 1.0, 0.05)
    n_sim = st.slider("采样次数(点图)", 10, 500, 200, 10)
    st.caption("温度>1 更随机,<1 更确定;top-k / top-p 压缩候选范围。")

probs = softmax(LOGITS, temperature)
mask = np.ones_like(probs, dtype=bool)
if top_k < 6:
    mask &= top_k_mask(LOGITS, top_k)
if top_p < 1.0:
    mask &= top_p_mask(LOGITS, top_p)
final = probs * mask
final = final / final.sum()

fig = go.Figure()
fig.add_trace(go.Bar(x=VOCAB, y=probs, name="温度后原始概率",
                     marker_color="#C7CDD9", opacity=0.8,
                     text=[f"{p:.3f}" for p in probs], textposition="outside"))
fig.add_trace(go.Bar(x=VOCAB, y=final, name="top-k/top-p 后最终概率",
                     marker_color="#E45756",
                     text=[f"{p:.3f}" for p in final], textposition="outside"))
fig.update_layout(title="候选概率分布:灰色=温度后,红色=再经 top-k/top-p 过滤",
                  yaxis_title="概率", barmode="overlay", height=380,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

rng = np.random.default_rng()
counts = np.zeros(len(VOCAB))
for _ in range(n_sim):
    counts[rng.choice(len(final), p=final)] += 1

pick = rng.choice(len(final), p=final)
st.subheader("🎯 采样结果")
c1, c2, c3 = st.columns(3)
c1.metric("抽中的 token", f"“{VOCAB[pick]}”")
c2.metric("单次最高概率候选", VOCAB[int(np.argmax(final))])
c3.metric("有效候选数", int(mask.sum()))

fig2 = go.Figure(go.Bar(x=VOCAB, y=counts, marker_color="#4C78A8",
                        text=counts, textposition="outside"))
fig2.update_layout(title=f"采样 {n_sim} 次的命中频次(近似于红色概率)",
                   yaxis_title="次数", height=300,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 抽中频次大致正比于最终概率:概率高的候选被抽中的次数更多。")
