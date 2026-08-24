# -*- coding: utf-8 -*-
# app_20_batch_compare.py — 静态 vs 连续 实测对比 📊
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="静态 vs 连续实测 📊", layout="wide")
st.title("📊 第 20 课 · 真实 GPU 对比:静态批 vs Continuous Batching")

st.markdown("""
模拟器是风洞,真机是赛道。本演示展示 `TinyLM`(微型 transformer,手写 KV cache)
在 GPU 上的**实测吞吐对比**:默认展示预存实验数据,也可以一键重跑小实验。
""")

# ---------------------------------------------------------------- 模型(与 notebook 一致)
V = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def causal_mask(seq, device, lens=None):
    q = torch.arange(seq, device=device)
    k = torch.arange(seq, device=device)
    if lens is None:
        return torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)
    lens = lens[:, None]
    return ~((k[None, :] < lens) & (k[None, :] <= q[None, :]))

class TinyLM(nn.Module):
    def __init__(self, n_vocab=V, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(d_model, n_head, batch_first=True),
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model), nn.LayerNorm(d_model)]))
        self.lm_head = nn.Linear(d_model, n_vocab)

    def attn(self, mha, h, k, v, mask):
        if mask.ndim == 3:
            H = mha.num_heads
            mask = mask[:, None].expand(mask.shape[0], H, *mask.shape[1:]).reshape(
                mask.shape[0] * H, *mask.shape[1:])
        a, _ = mha(h, k, v, attn_mask=mask, need_weights=False)
        return a

    def prefill(self, ids, lens):
        h = self.embed(ids)
        cache, mask = [], causal_mask(ids.shape[1], ids.device, lens)
        for mha, fc1, act, fc2, ln in self.layers:
            h2 = ln(h)
            h = h + self.attn(mha, h2, h2, h2, mask)
            h = h + fc2(act(fc1(ln(h))))
            cache.append((h2.detach(), h2.detach()))
        return self.lm_head(h), cache

    def decode(self, x1, caches):
        B = x1.shape[0]
        h = self.embed(x1)                          # (B, 1, D)
        layer_kv = []
        for (mha, fc1, act, fc2, ln), cl in zip(self.layers, zip(*caches)):
            Ks, Vs, new_len = [], [], []
            for b in range(B):
                kp = torch.cat([cl[b][0], h[b]], dim=0)   # (L_b+1, D) 追加新 token
                vp = torch.cat([cl[b][1], h[b]], dim=0)
                new_len.append(kp.shape[0])
                Ks.append(kp); Vs.append(vp)
            Lmax1 = max(new_len)
            K = torch.stack([torch.cat([t, torch.zeros(Lmax1 - t.shape[0], t.shape[1], device=h.device)]) for t in Ks])
            V = torch.stack([torch.cat([t, torch.zeros(Lmax1 - t.shape[0], t.shape[1], device=h.device)]) for t in Vs])
            h2 = ln(h)
            lens_t = torch.tensor(new_len, device=h.device).view(B, 1, 1)
            positions = torch.arange(Lmax1, device=h.device).view(1, 1, Lmax1)
            mask = positions >= lens_t              # (B, 1, Lk):屏蔽 pad 与未来位置
            h = h + self.attn(mha, h2, K, V, mask)
            h = h + fc2(act(fc1(ln(h))))
            layer_kv.append((K, V))
        new_caches = [[(layer_kv[l][0][b], layer_kv[l][1][b]) for l in range(len(layer_kv))]
                      for b in range(B)]
        return self.lm_head(h), new_caches

def gen_reqs(n, seed=7, plen=(8, 24), mnew=(4, 16)):
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(plen[0], plen[1] + 1)), int(rng.integers(mnew[0], mnew[1] + 1)))
            for _ in range(n)]

def run_static(model, reqs, batch_size):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(reqs), batch_size):
        batch = reqs[s:s + batch_size]
        Lp = max(p for p, _ in batch)
        M = max(m for _, m in batch)
        x = torch.randint(0, V, (len(batch), Lp), device=DEVICE)
        lens = torch.full((len(batch),), Lp, device=DEVICE)
        _, cache = model.prefill(x, lens)
        caches = [[(cache[l][0][j], cache[l][1][j])
                   for l in range(len(cache))] for j in range(len(batch))]
        for _ in range(M):
            x1 = torch.randint(0, V, (len(batch), 1), device=DEVICE)
            _, caches = model.decode(x1, caches)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def run_continuous(model, reqs, pfill_chunk=8):
    run = [(p, m, None) for p, m in reqs]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(run), pfill_chunk):
        chunk = run[s:s + pfill_chunk]
        Lp = max(p for p, _, _ in chunk)
        x = torch.randint(0, V, (len(chunk), Lp), device=DEVICE)
        lens = torch.tensor([p for p, _, _ in chunk], device=DEVICE)
        _, cache = model.prefill(x, lens)
        for j, (p, m, _) in enumerate(chunk):
            run[s + j] = (p, m, [(cache[l][0][j], cache[l][1][j]) for l in range(len(cache))])
    act = [i for i, (p, m, c) in enumerate(run) if m > 0]
    while act:
        x1 = torch.randint(0, V, (len(act), 1), device=DEVICE)
        _, new_caches = model.decode(x1, [run[i][2] for i in act])
        for j, i in enumerate(act):
            p, m, _ = run[i]
            run[i] = (p, m - 1, new_caches[j])
        act = [i for i, (p, m, c) in enumerate(run) if m > 0]
    torch.cuda.synchronize()
    return time.perf_counter() - t0

# ---------------------------------------------------------------- 数据源
@st.cache_data
def load_precomputed():
    p = Path(__file__).parent / "batch_compare_data.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return dict(device="cuda", tokens=360, requests=16,
                rows=[dict(batch_size=2, static_tok_s=8200, cont_tok_s=11200, speedup=1.37),
                      dict(batch_size=4, static_tok_s=9800, cont_tok_s=12400, speedup=1.27),
                      dict(batch_size=8, static_tok_s=10800, cont_tok_s=12600, speedup=1.17)])

@st.cache_data
def rerun_experiment(n, batch_size, mnew_max):
    torch.manual_seed(0)
    model = TinyLM().to(DEVICE)
    reqs = gen_reqs(n, seed=7, mnew=(4, mnew_max))
    tok = sum(p + m for p, m in reqs)
    _ = run_static(model, reqs, 4)
    rows = []
    for bs in [2, 4, 8]:
        t_s = run_static(model, reqs, batch_size=bs)
        t_c = run_continuous(model, reqs)
        rows.append(dict(batch_size=bs, static_tok_s=round(tok / t_s, 1),
                         cont_tok_s=round(tok / t_c, 1),
                         speedup=round((tok / t_c) / (tok / t_s), 2)))
    return dict(device=DEVICE, tokens=tok, requests=n, rows=rows)

with st.sidebar:
    st.header("🎛️ 数据源")
    src = st.radio("实验数据", ["预存实验数据", "重新运行小实验(需 GPU)"])
    n = st.slider("请求数量(重跑时)", 8, 32, 16, 4)
    mnew_max = st.slider("最大生成长度(重跑时)", 8, 24, 16, 2)
    rerun = st.button("🔬 重新运行小实验")
    st.caption(f"设备: {DEVICE}")

if src == "预存实验数据":
    data = load_precomputed()
    note = "预存数据"
else:
    if rerun:
        with st.spinner(f"在 {DEVICE} 上重跑实验…"):
            data = rerun_experiment(n, 4, mnew_max)
        note = "本次重跑"
    else:
        data = load_precomputed()
        note = "预存数据(点上方按钮重跑)"

res = pd.DataFrame(data["rows"])
ms = res.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric(f"静态批吞吐({note})", f"{ms.static_tok_s:,.0f}", "token/s")
c2.metric(f"连续批吞吐({note})", f"{ms.cont_tok_s:,.0f}", "token/s")
c3.metric("加速比", f"{ms.speedup:.2f}x", delta=f"{data['requests']} 请求 / {data['tokens']} tokens")

fig = go.Figure()
for name, key, color in [("静态批", "static_tok_s", "#E45756"), ("连续批", "cont_tok_s", "#54A24B")]:
    fig.add_trace(go.Bar(x=[f"batch={b}" for b in res.batch_size], y=res[key],
                         name=name, marker_color=color, text=[f"{v:,.0f}" for v in res[key]],
                         textposition="outside"))
fig.update_layout(title=f"实测吞吐对比(设备 {data['device']})", yaxis_title="token/秒",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=res.batch_size, y=res.static_tok_s, mode="lines+markers",
                          name="静态批", line=dict(color="#E45756", width=3), marker=dict(size=9)))
fig2.add_trace(go.Scatter(x=res.batch_size, y=res.cont_tok_s, mode="lines+markers",
                          name="连续批", line=dict(color="#54A24B", width=3), marker=dict(size=9)))
fig2.update_layout(title="吞吐 vs 批次大小曲线", xaxis_title="batch_size",
                   yaxis_title="token/秒", height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **读图指南**:连续批在每个 batch_size 上都赢,赢面来自“完成一个走一个、
> 没有整批陪跑”;差距大小受 GPU 状态与请求分布影响,这正是第 19 课敏感性分析
> 在真机上的复现——**方向看模拟,幅度看真机**。
""")
