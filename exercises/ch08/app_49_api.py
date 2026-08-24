# -*- coding: utf-8 -*-
# app_49_api.py — OpenAI 客户端实战:采样参数、finish_reason 与流式响应
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="49 · OpenAI 客户端", layout="wide")
st.title("第 49 课 · OpenAI 兼容接口:采样参数、finish_reason 与 SSE 流式")

st.markdown("""
客户端(网页、App、脚本)通过 **OpenAI 兼容 API** 与服务对话。一次 `chat/completions` 请求里,
`temperature` / `top_p` / `max_tokens` / `stream` 决定了模型**怎么生成**;响应里的 `finish_reason`
则告诉你**为什么结束**(`stop` 自然结束 / `length` 被 max_tokens 截断)。
本页用**本地真实采样**(不依赖网络)当场生成,并实时画出采样分布、给出真实 finish_reason。
""")

VOCAB = ["<eos>", "你好", "世界", "vLLM", "部署", "GPU", "推理", "token", "OpenAI", "性能"]
EOS_ID = 0

def next_token_probs(rng, temperature=1.0, top_p=1.0):
    logits = rng.normal(0, 2.0, size=len(VOCAB))
    logits = logits / max(temperature, 1e-3)
    p = np.exp(logits - logits.max()); p = p / p.sum()
    order = np.argsort(-p); cum = np.cumsum(p[order])
    keep = cum <= top_p
    if not keep.any(): keep[0] = True
    keep[np.argmax(cum > top_p)] = True
    mask = np.zeros_like(p, dtype=bool); mask[order[keep]] = True
    p2 = p * mask
    return p2 / p2.sum() if p2.sum() > 0 else p

def generate(prompt, temperature=0.7, top_p=1.0, max_tokens=8, seed=0):
    rng = np.random.default_rng(seed)
    tokens = []
    for _ in range(max_tokens):
        p = next_token_probs(rng, temperature, top_p)
        tok = int(rng.choice(len(VOCAB), p=p))
        tokens.append(tok)
        if tok == EOS_ID:
            body = [i for i in tokens if i != EOS_ID]
            return "".join(VOCAB[i] for i in body), tokens, "stop"
    return "".join(VOCAB[i] for i in tokens), tokens, "length"

with st.sidebar:
    st.header("采样参数")
    prompt = st.text_input("用户消息 (user)", "请介绍一下 vLLM")
    temperature = st.slider("temperature", 0.0, 2.0, 0.7, 0.1,
                            help="越大越随机,越小越确定;0 时退化为贪心(总是取最高概率)")
    top_p = st.slider("top_p (核采样)", 0.1, 1.0, 0.9, 0.05,
                      help="只从累计概率不超过 top_p 的候选里采样")
    max_tokens = st.slider("max_tokens", 1, 16, 8, 1)
    seed = st.number_input("随机种子", 0, 9999, 42, 1)
    st.caption("temperature/top_p 是采样参数;max_tokens 是硬性上限。finish_reason 由真实生成决定。")

result, tokens, finish = generate(prompt, temperature=temperature, top_p=top_p,
                                  max_tokens=max_tokens, seed=seed)
prompt_tokens = max(len(prompt), 1)
st.subheader("chat/completions 响应")
finish_cn = "stop(自然结束,采到 <eos>)" if finish == "stop" else "length(被 max_tokens 截断)"
st.markdown(f"**model**: mock-chat  ·  **finish_reason**: `{finish}`({finish_cn})")
st.info(f"回复: {result}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("prompt_tokens", prompt_tokens)
c2.metric("completion_tokens", len(tokens))
c3.metric("total_tokens", prompt_tokens + len(tokens))
c4.metric("finish_reason", finish)

rng = np.random.default_rng(seed)
logits = rng.normal(0, 2.0, size=len(VOCAB))
def probs(temp, p_top):
    l = logits / max(temp, 1e-3)
    p = np.exp(l - l.max()); p = p / p.sum()
    order = np.argsort(-p); cum = np.cumsum(p[order])
    keep = cum <= p_top
    if not keep.any(): keep[0] = True
    keep[np.argmax(cum > p_top)] = True
    mask = np.zeros_like(p, dtype=bool); mask[order[keep]] = True
    p2 = p * mask
    return p2 / p2.sum() if p2.sum() > 0 else p

p_lo = probs(0.3, top_p)
p_mid = probs(temperature, top_p)
p_hi = probs(1.5, top_p)
x = list(range(len(VOCAB)))
fig = go.Figure()
for lbl, p, col in [("T=0.3", p_lo, "#4C78A8"), (f"T={temperature:.1f}", p_mid, "#72B7B2"), ("T=1.5", p_hi, "#E45756")]:
    fig.add_trace(go.Bar(x=x, y=p, name=lbl, marker_color=col))
fig.update_layout(title="softmax(logits/T) 概率分布:温度越高分布越平",
                  xaxis=dict(tickvals=x, ticktext=VOCAB), yaxis_title="采样概率",
                  barmode="group", height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("观察:T=0.3 时最高概率 token 一骑绝尘(接近贪心);T=1.5 时分布被抹平。"
           "top_p 会裁掉累计概率之外的候选。finish_reason 与这些采样参数的真实交互共同决定输出。")

st.markdown("""
> **真实客户端**:若装了 `openai` 库,可 `from openai import OpenAI;
> client = OpenAI(base_url="http://localhost:8000/v1")`,再
> `client.chat.completions.create(model=..., messages=..., stream=True)`。
> 本机未安装 openai 库,故用 `requests` 直连或本页纯采样模拟,协议与真机一致。
> 见 [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server)
> 与 [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 8 章 · 第 49 课配套演示")
