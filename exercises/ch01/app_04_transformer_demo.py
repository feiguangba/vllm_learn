# -*- coding: utf-8 -*-
# app_04_transformer_demo.py — 手写单层 Transformer 演示 🧠
import streamlit as st
import torch
import torch.nn as nn
import math
import plotly.graph_objects as go

st.set_page_config(page_title="单层 Transformer 🧠", layout="wide")
st.title("🧠 第 04 课 · 手写单层 Transformer 块")

st.markdown("""
Transformer 是现代大模型的核心积木。这一课用手写一个**单层 transformer block**:
embedding → QKV → 缩放点积注意力 → FFN → LayerNorm → 残差,并在 CPU 上跑前向。
下方调整**序列长度**与**注意力头数**,实时看 attention 权重热图——每个 token 在关注谁。
""")

HIDDEN = 32

class ToyBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        assert hidden % n_heads == 0, f"hidden({hidden}) 必须能被 n_heads({n_heads}) 整除"
        self.head_dim = hidden // n_heads
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, hidden, bias=False)
        self.wv = nn.Linear(hidden, hidden, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, verbose=False):
        B, T, H = x.shape
        hd = self.head_dim
        if verbose: print(f"[1] embedding        {tuple(x.shape)}")
        q = self.wq(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        if verbose: print(f"[2] Q/K/V 多头拆分    {tuple(q.shape)}")
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        if verbose: print(f"[3] 缩放点积 scores   {tuple(scores.shape)}")
        weights = torch.softmax(scores, dim=-1)
        if verbose: print(f"[4] attention 权重    {tuple(weights.shape)}")
        attn = weights @ v
        attn = attn.transpose(1, 2).reshape(B, T, H)
        out_attn = self.wo(attn)
        if verbose: print(f"[5] 注意力输出       {tuple(out_attn.shape)}")
        h = x + out_attn
        if verbose: print(f"[6] + 残差           {tuple(h.shape)}")
        h = self.norm1(h)
        if verbose: print(f"[7] LayerNorm        {tuple(h.shape)}")
        h = h + self.ffn(h)
        if verbose: print(f"[8] FFN + 残差        {tuple(h.shape)}")
        h = self.norm2(h)
        if verbose: print(f"[9] LayerNorm        {tuple(h.shape)}")
        return h, weights

with st.sidebar:
    st.header("🎛️ 参数")
    seq_len = st.slider("序列长度 T", 4, 16, 8, 1)
    # 下拉只给 hidden=32 的因数,保证 head_dim = hidden/n_heads 能整除,view 不报错
    n_heads = st.select_slider("注意力头数(须整除 hidden=32)", options=[1, 2, 4, 8, 16, 32], value=4)
    head_idx = st.slider("查看第几个头", 0, n_heads - 1, 0, 1)
    st.caption(f"head_dim = hidden / n_heads = {HIDDEN}/{n_heads} = {HIDDEN // n_heads};"
               f"下拉选项均为 32 的因数,保证可整除。")

torch.manual_seed(0)
block = ToyBlock(HIDDEN, n_heads)
x = torch.randn(1, seq_len, HIDDEN)
out, weights = block(x)

w = weights[0, head_idx].detach().numpy()
st.subheader(f"🔍 Attention 权重热图(head {head_idx}, T={seq_len})")
fig = go.Figure(go.Heatmap(
    z=w, x=[f"t{i}" for i in range(seq_len)],
    y=[f"t{i}" for i in range(seq_len)],
    colorscale="Blues", zmin=0, zmax=w.max(),
    text=[[f"{v:.2f}" for v in row] for row in w], texttemplate="%{text}"))
fig.update_layout(title="第 j 行 = token t_j 对前面所有 token 的关注权重(行和为 1)",
                  xaxis_title="被关注的 key token", yaxis_title="查询 query token",
                  height=460, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 每一行是 softmax 的结果,加起来为 1:它表示“当前 token 把注意力花在谁身上”。"
           "头数越多,模型能从不同视角关注不同位置。")
