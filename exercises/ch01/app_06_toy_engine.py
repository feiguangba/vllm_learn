# -*- coding: utf-8 -*-
# app_06_toy_engine.py — 玩具推理引擎演示 🛠️
import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from collections import defaultdict, Counter

st.set_page_config(page_title="玩具推理引擎 🛠️", layout="wide")
st.title("🛠️ 第 06 课 · 组装一个玩具推理引擎")

st.markdown("""
把第 02 课的 **n-gram 语言模型** 与第 03 课的 **采样策略** 拼起来,就是一个能**批量生成**
多条文本、并统计吞吐(tokens/s)的**玩具推理引擎**。下方调整批大小、温度、top-p 等参数,
观察批量生成的吞吐如何变化。
""")

CORPUS = ["我喜欢学习机器学习和深度学习", "机器学习很热门", "深度学习让机器更聪明",
          "我每天写代码和学算法", "我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉"]

def build_ngram(corpus, n=2):
    model = defaultdict(Counter)
    for text in corpus:
        toks = list(text)
        for i in range(len(toks) - n + 1):
            model[tuple(toks[i:i + n - 1])][toks[i + n - 1]] += 1
    return model

def next_candidates(model, context, n):
    ctx = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)
    counts = model.get(ctx, Counter())
    total = sum(counts.values())
    if total == 0:
        return [], []
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    toks = [t for t, _ in items]
    probs = [c / total for _, c in items]
    return toks, probs

def softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()

def sample_from(logits, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    probs = softmax(logits, temperature)
    idx = rng.choice(len(probs), p=probs)
    return idx

class ToyEngine:
    def __init__(self, model, n=2, temperature=1.0, seed=0):
        self.model = model
        self.n = n
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.W = np.random.randn(32, 32) * 0.05

    def shared_forward(self):
        return np.random.randn(32, 32) @ self.W

    def generate_batch(self, prompts, max_new):
        seqs = [list(p) for p in prompts]
        tokens = 0
        for _ in range(max_new):
            self.shared_forward()
            for seq in seqs:
                toks, probs = next_candidates(self.model, seq, self.n)
                if not toks:
                    continue
                logits = np.log(np.array(probs) + 1e-12)
                seq.append(toks[sample_from(logits, self.temperature, self.rng)])
                tokens += 1
        return seqs, tokens

model = build_ngram(CORPUS, 2)

with st.sidebar:
    st.header("🎛️ 参数")
    batch = st.slider("批大小 batch", 1, 16, 4, 1)
    max_new = st.slider("每序列生成步数", 5, 40, 15, 1)
    temperature = st.slider("温度", 0.3, 2.0, 1.0, 0.1)
    prompts_input = st.text_area("多条提示词(每行一条)", "我喜欢\n机器\n学习")
    st.caption("批大小越大,单次前向处理的序列越多,固定开销被摊薄。")

prompts = [p for p in prompts_input.splitlines() if p.strip()] or ["我喜欢"]
engine = ToyEngine(model, 2, temperature, seed=0)
seqs, tokens = engine.generate_batch(prompts[:batch], max_new)

st.subheader("🖊️ 批量生成结果")
for i, s in enumerate(seqs):
    st.write(f"**序列 {i + 1}**: " + "".join(s))

c1, c2, c3 = st.columns(3)
c1.metric("批大小", len(seqs))
c2.metric("生成总 token 数", tokens)
c3.metric("吞吐", f"{tokens / max(max_new / 60, 1e-6):.0f} tokens/s")

st.subheader("📈 吞吐 vs 批大小(实测)")
bs = [1, 2, 4, 8, 16]
tps = []
for b in bs:
    e2 = ToyEngine(model, 2, temperature, seed=0)
    t0 = time.perf_counter()
    _, tok = e2.generate_batch(["我喜欢"] * b, 30)
    tps.append(tok / (time.perf_counter() - t0))
fig = go.Figure(go.Scatter(x=bs, y=tps, mode="lines+markers",
                           line=dict(width=3, color="#4C78A8"),
                           marker=dict(size=9)))
fig.update_layout(title="批量生成吞吐(批大小越大吞吐越高,后趋饱和)",
                  xaxis_title="批大小 batch", yaxis_title="tokens/s",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 批处理让一次前向同时算多个序列,固定开销被摊薄,吞吐上升;"
           "当固定开销占比变小时,曲线趋于饱和。")
