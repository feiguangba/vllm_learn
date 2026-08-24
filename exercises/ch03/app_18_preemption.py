# -*- coding: utf-8 -*-
# app_18_preemption.py — 抢占策略演示 ⚔️
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="抢占:recompute vs swap ⚔️", layout="wide")
st.title("⚔️ 第 18 课 · 抢占:显存不够时,recompute 还是 swap?")

st.markdown("""
显存有限时,新请求会把运行中的请求**赶下车**(LIFO 抢占)。被赶的请求怎么办?
- 🔄 **recompute**:KV 作废,恢复后从头重新 prefill——省显存,但重算浪费;
- 💾 **swap**:KV 搬到 CPU 内存,恢复时搬回来(有搬运耗时)——保留进度,但吃带宽。
本演示用**显存压力 / 策略 / 恢复开销**三个旋钮,展示抢占事件与延迟惩罚。
""")

# ---------------------------------------------------------------- 模拟器(与 notebook 一致)
@dataclass
class Req:
    rid: int; arrive: float; prompt_len: int; max_new: int
    state: str = "WAITING"; start: float = None; end: float = None
    generated: int = 0; prefilled: int = 0
    preempted: int = 0; resume_left: int = 0

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

def simulate_preempt(reqs, token_budget=8, max_running=4, policy="recompute", swap_cost=3.0):
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    events = []   # (类型, 时间, 请求, 对方请求)
    while done < total:
        new = [r for r in waiting if r.arrive <= t]
        for r in new:
            waiting.remove(r)
        for r in list(new):
            if r.state == "PREEMPTED" and len(running) < max_running:
                running.append(r); new.remove(r)
        for r in list(new):
            if len(running) < max_running:
                running.append(r); new.remove(r)
            else:
                victim = running.pop()
                victim.state, victim.preempted = "PREEMPTED", victim.preempted + 1
                if policy == "recompute":
                    victim.generated, victim.prefilled = 0, 0
                    victim.state, victim.resume_left = "WAITING", 0
                else:
                    victim.resume_left = swap_cost
                waiting.insert(0, victim)
                events.append(("抢占", t, victim.rid, r.rid))
                running.append(r); new.remove(r)
        for r in running:
            if r.resume_left > 0:
                r.resume_left -= 1
                if r.resume_left == 0 and r.state == "PREEMPTED":
                    r.state = "RUNNING"
        budget = token_budget
        for r in running:
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use; r.prefilled += use
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
        t += 1.0
    return reqs, events

def summarize(reqs):
    df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                            work=r.prompt_len + r.max_new, preempted=r.preempted) for r in reqs])
    df["latency"] = df["end"] - df["arrive"]
    return df

# ---------------------------------------------------------------- 参数
with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("请求数量", 8, 40, 20, 1)
    max_running = st.slider("显存并发上限(压力)", 1, 6, 3, 1)
    policy = st.radio("抢占策略", ["recompute(重算)", "swap(搬移)"])
    swap_cost = st.slider("swap 恢复耗时(步)", 1, 8, 3, 1) if policy == "swap(搬移)" else 3
    token_budget = st.slider("token 预算", 2, 16, 8, 1)
    mode = st.radio("到达模式", ["burst(同时到达)", "poisson(泊松流)"])
    rate = st.slider("到达率 λ", 0.1, 1.5, 0.5, 0.1) if mode == "poisson(泊松流)" else None
    seed = st.slider("随机种子", 0, 99, 3, 1)
    st.caption("⚠️ 上限越小压力越大;burst 模式下抢占最惨烈")

# ---------------------------------------------------------------- 运行
reqs, events = simulate_preempt(make_reqs(n, mode, rate, seed),
                                token_budget, max_running, "recompute" if policy.startswith("re") else "swap",
                                swap_cost)
df = summarize(reqs)
n_pre = sum(1 for e in events if e[0] == "抢占")
c1, c2, c3, c4 = st.columns(4)
c1.metric("抢占次数", n_pre)
c2.metric("平均延迟(步)", f"{df.latency.mean():.1f}")
c3.metric("最大延迟(步)", f"{df.latency.max():.0f}")
c4.metric("吞吐(请求/步)", f"{n / df.end.max():.3f}")

# ---------------------------------------------------------------- 抢占事件表
if events:
    ev_df = pd.DataFrame([dict(时间=e[1], 被抢占=f"R{e[2]}", 抢占者=f"R{e[3]}") for e in events])
    st.subheader("⚔️ 抢占事件")
    st.dataframe(ev_df, use_container_width=True, height=180)
else:
    st.info("当前参数下没有发生抢占——显存压力不够大,调低并发上限试试。")

# ---------------------------------------------------------------- 甘特图 + 抢占标记
st.subheader("📈 甘特图(红 ✕ = 被抢占时刻)")
fig = go.Figure()
for _, row in df.iterrows():
    fig.add_trace(go.Bar(x=[row.end - row.start], base=[row.start], y=[f"R{row.rid}"],
                         orientation="h", marker_color="#54A24B", showlegend=False, width=0.6))
fig.update_layout(title="执行区间(绿)与抢占时刻(红 ✕)",
                  xaxis_title="时间(步)", yaxis_title="请求", height=60 + 30 * n,
                  margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
pre_x = [e[1] for e in events]
pre_y = [f"R{e[2]}" for e in events]
fig.add_trace(go.Scatter(x=pre_x, y=pre_y, mode="markers",
                         marker=dict(symbol="x", size=12, color="#E45756", line=dict(width=2)),
                         name="抢占时刻", hovertemplate="被抢占于 %{x:.0f} 步<extra></extra>"))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- recompute vs swap 对比
st.subheader("⚖️ recompute vs swap 同压力对比")
reqs2, ev2 = simulate_preempt(make_reqs(n, mode, rate, seed), token_budget, max_running, "recompute", swap_cost)
df_r, ev_r = summarize(reqs2), ev2
reqs3, ev3 = simulate_preempt(make_reqs(n, mode, rate, seed), token_budget, max_running, "swap", swap_cost)
df_s, ev_s = summarize(reqs3), ev3
n_r = sum(1 for e in ev_r if e[0] == "抢占")
n_s = sum(1 for e in ev_s if e[0] == "抢占")

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=["recompute", "swap"], y=[df_r.latency.mean(), df_s.latency.mean()],
                      marker_color=["#F2C14E", "#72B7B2"], name="平均延迟", text=[f"{df_r.latency.mean():.1f}", f"{df_s.latency.mean():.1f}"],
                      textposition="outside"))
fig2.update_layout(title=f"平均延迟对比(抢占 {n_r} vs {n_s} 次;swap 恢复耗时 {swap_cost} 步)",
                   yaxis_title="平均延迟(步)", height=340, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("💡 recompute 的浪费 = 重算的 token 数;swap 的浪费 = 搬移步数 × 次数。"
           "prompt 很长时 recompute 更痛,swap 次数多时 swap 更痛。")
