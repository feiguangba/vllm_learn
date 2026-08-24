# -*- coding: utf-8 -*-
# app_15_continuous_batch.py — Continuous Batching 逐步演示 🍳
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="Continuous Batching 🍳", layout="wide")
st.title("🍳 第 15 课 · Continuous Batching:随做随上,不凑桌")

st.markdown("""
静态批像**凑满一车才发车的大巴**,continuous batching 则像**随做随上的自助餐**:
每完成一道菜(一个请求)就立刻翻台,GPU 永远在干活。
本演示用**步进控制**逐迭代观察 batch 成员的变化,并和静态批做同屏对比。
""")

# ---------------------------------------------------------------- 模拟器(与 notebook 一致)
@dataclass
class Req:
    rid: int
    arrive: float
    prompt_len: int
    max_new: int
    state: str = "WAITING"
    start: float = None
    end: float = None
    generated: int = 0
    prefilled: int = 0

def make_reqs(n, mode, rate, seed, plen=(5, 30), mnew=(5, 20)):
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        if mode == "poisson":
            t += rng.exponential(1.0 / rate)
        a = 0.0 if mode == "burst" else t
        reqs.append(Req(i, a, int(rng.integers(plen[0], plen[1] + 1)),
                        int(rng.integers(mnew[0], mnew[1] + 1))))
    return reqs

def simulate_continuous(reqs, token_budget=8):
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    steps = []
    while done < total:
        for r in list(waiting):
            if r.arrive <= t:
                running.append(r); waiting.remove(r)
        budget = token_budget
        for r in running:
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in running:
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
                running.remove(r); done += 1
        steps.append(dict(step=int(t), running=[r.rid for r in running],
                          waiting=[r.rid for r in waiting], finished=done))
        t += 1.0
    return reqs, steps

def simulate_static(reqs, batch_size):
    queue = sorted(reqs, key=lambda r: r.arrive)
    t, batches = 0.0, []
    while queue:
        batch = []
        for r in list(queue):
            if r.arrive <= t and len(batch) < batch_size:
                batch.append(r)
                queue.remove(r)
        if not batch:
            t += 1.0
            continue
        for r in batch:
            r.state, r.start = "RUNNING", t
        bt = max(r.prompt_len + r.max_new for r in batch)
        for r in batch:
            r.end, r.state = t + bt, "FINISHED"
        batches.append(dict(start=t, end=t + bt, members=[r.rid for r in batch]))
        t += bt
    return reqs, batches

# ---------------------------------------------------------------- 参数
with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("请求数量", 8, 24, 12, 1)
    token_budget = st.slider("token 预算(每步)", 2, 16, 8, 1)
    mode = st.radio("到达模式", ["burst(同时到达)", "poisson(泊松流)"])
    rate = st.slider("到达率 λ(请求/步)", 0.1, 1.5, 0.5, 0.1) if mode == "poisson(泊松流)" else None
    seed = st.slider("随机种子", 0, 99, 7, 1)
    st.caption("💡 token 预算 ≈ vLLM 的 max_num_scheduled_tokens")

# ---------------------------------------------------------------- 模拟
reqs_c, steps = simulate_continuous(make_reqs(n, mode, rate, seed), token_budget)
total_steps = len(steps)
if "step" not in st.session_state or st.session_state.get("n") != n:
    st.session_state.step = 0
    st.session_state.n = n

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    st.markdown("#### ⏯️ 步进控制")
    if st.button("⏮ 回到第 0 步"):
        st.session_state.step = 0
    if st.button("⏭ 下一步"):
        st.session_state.step = min(st.session_state.step + 1, total_steps - 1)
    if st.button("🏁 跳到结束"):
        st.session_state.step = total_steps - 1
with c2:
    k = st.slider("当前迭代(步)", 0, total_steps - 1, st.session_state.step)
    st.session_state.step = k
with c3:
    snap = steps[k]
    st.metric("当前步", k)
    st.metric("运行中请求", len(snap["running"]))
    st.metric("已完成", snap["finished"])

st.markdown(f"#### 第 {k} 步快照:running = `{snap['running']}`,waiting = `{snap['waiting']}`")
snap_rows = [dict(迭代=k, running=str(s["running"]), waiting=str(s["waiting"]), 已完成=s["finished"])
             for s in steps[max(0, k - 5): k + 1]]
st.dataframe(pd.DataFrame(snap_rows), use_container_width=True)
st.caption("📋 上面列出最近几步的 batch 成员:每次迭代 batch 都在变化——这正是 continuous batching 的名字由来。")

# ---------------------------------------------------------------- 甘特图(截至当前步)
df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                        work=r.prompt_len + r.max_new) for r in reqs_c])
df["latency"] = df["end"] - df["arrive"]
df["wait"] = df["start"] - df["arrive"]
cmap = {f"Req{i}": c for i, c in enumerate(["#4C78A8", "#72B7B2", "#E45756", "#F2C14E",
        "#54A24B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#605B8C"] * 4)}

fig = go.Figure()
for _, row in df.iterrows():
    show_end = min(row.end, k)
    if row.arrive < row.start:
        fig.add_trace(go.Bar(x=[row.start - row.arrive], y=[f"Req {row.rid}"],
                             base=[row.arrive], orientation="h",
                             marker_color="#E0E0E0", showlegend=False, width=0.6))
    if show_end > row.start:
        fig.add_trace(go.Bar(x=[show_end - row.start], y=[f"Req {row.rid}"],
                             base=[row.start], orientation="h",
                             marker_color=cmap[f"Req{row.rid}"], showlegend=False, width=0.6))
fig.update_layout(title=f"Continuous Batching 甘特图(截至第 {k} 步):灰=等待,彩=运行/完成",
                  xaxis_title="时间(步)", yaxis_title="请求", height=60 + 28 * len(df),
                  margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 对比第 14 课:这里的彩色段**长短不一、各自进退**,完成一个走一个,没有“整批陪跑”。")

# ---------------------------------------------------------------- 静态 vs 连续对比
st.subheader("⚖️ 同屏对比:静态批 vs Continuous Batching")
reqs_s, batches = simulate_static(make_reqs(n, mode, rate, seed), 4)
dfc, dfs = reqs_c, pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                                      work=r.prompt_len + r.max_new) for r in reqs_s])
dfc["latency"], dfs["latency"] = dfc["end"] - dfc["arrive"], dfs["end"] - dfs["arrive"]
ms_c, ms_s = float(dfc.end.max()), float(dfs.end.max())
c1, c2, c3 = st.columns(3)
c1.metric("墙钟时间(步)", f"{ms_c:.0f}", delta=f"{ms_s - ms_c:.0f} 较静态")
c2.metric("平均延迟(步)", f"{dfc['latency'].mean():.1f}", delta=f"{dfs['latency'].mean() - dfc['latency'].mean():.1f} 较静态")
c3.metric("吞吐(token/步)", f"{dfc['work'].sum() / ms_c:.1f}", delta=f"{dfc['work'].sum() / ms_c - dfs['work'].sum() / ms_s:.1f} 较静态")

def cum(df, ms):
    xs, ys, acc = [], [], 0
    for t in range(0, int(ms) + 1):
        acc = int((df.end <= t).sum())
        xs.append(t); ys.append(acc)
    return xs, ys

fig2 = go.Figure()
xs, ys = cum(dfc, ms_c)
fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="continuous", line=dict(color="#54A24B", width=3)))
xs, ys = cum(dfs, ms_s)
fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="static", line=dict(color="#E45756", width=3, dash="dash")))
fig2.update_layout(title="累积完成请求数 vs 时间:continuous 曲线始终在 static 上方",
                   xaxis_title="时间(步)", yaxis_title="已完成的请求数",
                   margin=dict(l=10, r=10, t=40, b=10), height=360)
st.plotly_chart(fig2, use_container_width=True)
