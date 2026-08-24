# -*- coding: utf-8 -*-
# app_02_generate_demo.py — 自回归生成演示 🔁
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict, Counter

st.set_page_config(page_title="自回归生成 🔁", layout="wide")
st.title("🔁 第 02 课 · 自回归生成:一字接一字")

st.markdown("""
大模型生成文本就像**接龙游戏**:它一次只预测**下一个** token,再把新 token 拼回上下文,
如此循环 L 次就得到 L 个新 token(需要 **L 次前向**)。下面用一个 n-gram 玩具模型
在选定的中文语料上,逐步预测下一个字,并展示每一步的候选概率。
""")

CORPORA = {
    "动物与生活": ["我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉", "小狗喜欢奔跑和玩耍", "小猫喜欢睡觉"],
    "机器学习": ["我喜欢学习机器学习和深度学习", "机器学习很热门", "深度学习让机器更聪明", "我每天写代码和学算法"],
    "美食": ["我喜欢吃火锅和饺子", "饺子很好吃", "火锅很辣很香", "我喜欢喝豆浆和牛奶"],
}

def build_ngram(corpus, n):
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

with st.sidebar:
    st.header("🎛️ 参数")
    corpus_name = st.selectbox("语料主题", list(CORPORA))
    n = st.select_slider("n-gram 阶数", options=[2, 3],
                         format_func=lambda v: "bigram" if v == 2 else "trigram")
    start = st.text_input("起始文本", "我喜欢")
    steps = st.slider("生成步数", 1, 15, 6, 1)
    st.caption("阶数越高,条件越具体,预测越“有依据”。")

model = build_ngram(CORPORA[corpus_name], n)
path = list(start)
cand_hist = []
for _ in range(steps):
    toks, probs = next_candidates(model, path, n)
    if not toks:
        break
    path.append(toks[0])
    cand_hist.append((toks, probs))

generated = list(start) + path[len(start):]
st.subheader("🖊️ 生成结果")
st.write(" " + " ".join(generated))
c1, c2, c3 = st.columns(3)
c1.metric("提示词长度", len(start))
c2.metric("生成 token 数", len(generated) - len(start))
c3.metric("所需前向次数", len(generated) - len(start))

st.subheader("🧪 每一步的候选概率(greedy 取最高)")
for step, (toks, probs) in enumerate(cand_hist):
    ctx_show = "".join(generated[:len(start) + step])
    fig = go.Figure(go.Bar(x=toks, y=probs,
                           text=[f"{p:.2f}" for p in probs],
                           textposition="outside",
                           marker_color="#4C78A8"))
    fig.update_layout(title=f"第 {step + 1} 步 · 上下文“{ctx_show}”",
                      yaxis_title="概率", height=230,
                      margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 观察:每一步的候选概率都来自“上下文里最近 n-1 个字”的条件分布;"
           "生成 L 个 token 就要做 L 次这样的预测,无法并行。")
