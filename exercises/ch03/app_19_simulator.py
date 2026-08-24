# -*- coding: utf-8 -*-
# app_19_simulator.py — 完整调度模拟器 🛰️
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="完整调度模拟器 🛰️", layout="wide")
st.title("🛰️ 第 19 课 · 完整调度模拟器:参数敏感性实验")

st.markdown("""
把前面所有旋钮拧到一个 **Simulator** 里:到达率 × token 预算 × 调度策略 × 显存上限 × 抢占。
像风洞实验一样**一次只动一个参数**,观察吞吐与延迟如何响应——
这就是调参前必须做的“敏感性分析”。
""")

# ---------------------------------------------------------------- 模拟器(与 notebook 一致)
@dataclass
class Req:
    rid: int; arrive: float; prompt_len: int; max_new: int
    priority: float = 0.0
    state: str = "WAITING"; start: float = None; end: float = None
    generated: int = 0; prefilled: int = 0
    preempted: int = 0; resume_left: int = 0

class Simulator:
    """完整调度模拟器:一次 run() 返回全部指标与事件流"""
    def __init__(self, rate=0.5, n=20, token_budget=8, policy="FCFS",
                 max_running=None, preempt="recompute", swap_cost=3.0, seed=1,
                 plen=(5, 30), mnew=(5, 20)):
        self.rate, self.n = rate, n
        self.token_budget, self.policy = token_budget, policy
        self.max_running, self.preempt = max_running, preempt
        self.swap_cost, self.seed = swap_cost, seed
        self.plen, self.mnew = plen, mnew

    def make_requests(self):
        rng = np.random.default_rng(self.seed)
        reqs, t = [], 0.0
        for i in range(self.n):
            t += rng.exponential(1.0 / self.rate)
            reqs.append(Req(i, t, int(rng.integers(self.plen[0], self.plen[1] + 1)),
                            int(rng.integers(self.mnew[0], self.mnew[1] + 1)),
                            priority=float(rng.random())))
        return reqs

    def pick_next(self, waiting):
        if self.policy == "FCFS":
            return sorted(waiting, key=lambda r: r.arrive)
        if self.policy == "SJF":
            return sorted(waiting, key=lambda r: r.prompt_len + r.max_new)
        if self.policy == "Priority":
            return sorted(waiting, key=lambda r: -r.priority)
        return waiting

    def run(self):
        waiting = sorted(self.make_requests(), key=lambda r: r.arrive)
        running, t, done, total = [], 0.0, 0, len(waiting)
        events = []
        while done < total:
            waiting = self.pick_next(waiting)
            new = [r for r in waiting if r.arrive <= t]
            for r in new:
                waiting.remove(r)
            for r in list(new):                          # 抢占逻辑
                if self.max_running is None or len(running) < self.max_running:
                    running.append(r); new.remove(r)
                else:
                    victim = running.pop()
                    victim.state, victim.preempted = "PREEMPTED", victim.preempted + 1
                    if self.preempt == "recompute":
                        victim.generated, victim.prefilled, victim.resume_left = 0, 0, 0
                        victim.state = "WAITING"
                    else:
                        victim.resume_left = self.swap_cost
                    waiting.insert(0, victim)
                    events.append(("抢占", t, victim.rid))
                    running.append(r); new.remove(r)
            for r in running:
                if r.resume_left > 0:
                    r.resume_left -= 1
                    if r.resume_left == 0 and r.state == "PREEMPTED":
                        r.state = "RUNNING"
            budget = self.token_budget
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
        df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                                work=r.prompt_len + r.max_new, preempted=r.preempted) for r in waiting])
        df["latency"] = df["end"] - df["arrive"]
        df["wait"] = df["start"] - df["arrive"]
        ms = float(df.end.max())
        stats = dict(makespan=ms,
                     req_per_step=float(self.n / ms),
                     token_per_step=float(df.work.sum() / ms),
                     avg_latency=float(df.latency.mean()),
                     p95_latency=float(np.percentile(df.latency, 95)),
                     preemptions=len(events))
        return stats, df, events

# ---------------------------------------------------------------- 参数
with st.sidebar:
    st.header("🎛️ 参数")
    rate = st.slider("到达率 λ(请求/步)", 0.1, 1.5, 0.5, 0.05)
    token_budget = st.slider("token 预算", 2, 24, 8, 1)
    policy = st.radio("调度策略", ["FCFS", "SJF", "Priority"])
    n = st.slider("请求数量", 10, 60, 30, 5)
    enable_preempt = st.checkbox("启用抢占", value=False)
    max_running = st.slider("显存并发上限", 1, 6, 3, 1) if enable_preempt else None
    seed = st.slider("随机种子", 0, 99, 1, 1)
    st.caption("🛰️ 一次只动一个参数,其余保持默认,是敏感性分析的正确姿势")

# ---------------------------------------------------------------- 单次运行
sim = Simulator(rate=rate, n=n, token_budget=token_budget, policy=policy,
                max_running=max_running, seed=seed)
stats, df, events = sim.run()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("吞吐(请求/步)", f"{stats['req_per_step']:.3f}")
c2.metric("吞吐(token/步)", f"{stats['token_per_step']:.1f}")
c3.metric("平均延迟", f"{stats['avg_latency']:.1f} 步")
c4.metric("P95 延迟", f"{stats['p95_latency']:.1f} 步")
c5.metric("抢占次数", stats["preemptions"])

st.subheader("📈 甘特图")
fig = go.Figure()
for _, row in df.iterrows():
    fig.add_trace(go.Bar(x=[row.wait], y=[f"R{row.rid}"], base=[row.arrive],
                         orientation="h", marker_color="#BAB0AC", showlegend=False, width=0.6))
    fig.add_trace(go.Bar(x=[row.latency - row.wait], y=[f"R{row.rid}"], base=[row.start],
                         orientation="h", marker_color="#54A24B", showlegend=False, width=0.6))
for e in events:
    fig.add_trace(go.Scatter(x=[e[1]], y=[f"R{e[2]}"], mode="markers", showlegend=False,
                             marker=dict(symbol="x", size=12, color="#E45756")))
fig.update_layout(title="灰=等待,绿=执行,红 ✕=抢占", xaxis_title="时间(步)",
                  yaxis_title="请求", height=60 + 28 * n,
                  margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 敏感性扫描
st.subheader("🧪 参数敏感性实验:一次只动一个参数")
scan_var = st.radio("扫描哪个参数?", ["到达率 λ", "token 预算"], horizontal=True)
compare_policy = st.radio("对比哪种策略?", ["FCFS", "SJF", "Priority"], horizontal=True)
if scan_var == "到达率 λ":
    xs = np.round(np.arange(0.15, 1.65, 0.15), 2)
    tgt = "rate"
else:
    xs = np.arange(2, 25, 2)
    tgt = "token_budget"

rows = []
for x in xs:
    kw = dict(rate=x) if tgt == "rate" else dict(token_budget=int(x))
    for p in [policy, compare_policy]:
        s = Simulator(**kw, n=n, policy=p, max_running=max_running, seed=seed)
        st_, _, _ = s.run()
        rows.append(dict(x=x, 策略=p, 吞吐=st_["req_per_step"], 平均延迟=st_["avg_latency"]))
scan = pd.DataFrame(rows)
if compare_policy != policy:
    fig2 = go.Figure()
    for p in [policy, compare_policy]:
        sub = scan[scan.策略 == p]
        fig2.add_trace(go.Scatter(x=sub.x, y=sub.吞吐, name=f"{p} 吞吐",
                                  line=dict(width=3), yaxis="y1"))
        fig2.add_trace(go.Scatter(x=sub.x, y=sub.平均延迟, name=f"{p} 平均延迟",
                                  line=dict(width=2, dash="dot"), yaxis="y2"))
    fig2.update_layout(title=f"敏感性扫描:{scan_var}(实线=吞吐,虚线=平均延迟)",
                       xaxis_title=scan_var, yaxis=dict(title="吞吐(请求/步)"),
                       yaxis2=dict(title="平均延迟(步)", overlaying="y", side="right"),
                       height=400, margin=dict(l=10, r=10, t=50, b=10), hovermode="x")
    st.plotly_chart(fig2, use_container_width=True)
else:
    fig2 = go.Figure()
    sub = scan[scan.策略 == policy]
    fig2.add_trace(go.Scatter(x=sub.x, y=sub.吞吐, name="吞吐", line=dict(width=3), yaxis="y1"))
    fig2.add_trace(go.Scatter(x=sub.x, y=sub.平均延迟, name="平均延迟",
                              line=dict(width=2, dash="dot"), yaxis="y2"))
    fig2.update_layout(title=f"敏感性扫描:{scan_var}(实线=吞吐,虚线=平均延迟)",
                       xaxis_title=scan_var, yaxis=dict(title="吞吐(请求/步)"),
                       yaxis2=dict(title="平均延迟(步)", overlaying="y", side="right"),
                       height=400, margin=dict(l=10, r=10, t=50, b=10), hovermode="x")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- 策略 × 显存热力图
st.subheader("🔥 策略 × 显存并发上限 → 平均延迟")
caps = [2, 3, 4, 6, 8]
policies = ["FCFS", "SJF", "Priority"]
hm = np.zeros((len(policies), len(caps)))
for i, p in enumerate(policies):
    for j, cap in enumerate(caps):
        s = Simulator(rate=rate, n=n, token_budget=token_budget, policy=p,
                      max_running=cap, seed=seed)
        st_, _, _ = s.run()
        hm[i, j] = st_["avg_latency"]
fig3 = go.Figure(go.Heatmap(z=hm, x=[str(c) for c in caps], y=policies,
                            colorscale="Viridis", text=np.round(hm, 1), texttemplate="%{text}",
                            colorbar=dict(title="平均延迟")))
fig3.update_layout(title="平均延迟热力图(浅=快,深=慢)", height=320,
                   margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig3, use_container_width=True)
