# -*- coding: utf-8 -*-
# app_17_scheduler_demo.py — 调度策略演示 🎛️
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="调度策略演示 🎛️", layout="wide")
st.title("🎛️ 第 17 课 · Scheduler 设计:token 预算 × 调度策略")

st.markdown("""
调度器 = **预算(每步能算多少 token)+ 策略(先服务谁)**。
本演示可切换 `FCFS / SJF / Priority` 三种策略、调节 token 预算与请求分布,
实时查看**每一步的调度决策**、延迟分布与 FCFS 基准对比。
""")

# ---------------------------------------------------------------- 模拟器(与 notebook 一致)
@dataclass
class Req:
    rid: int; arrive: float; prompt_len: int; max_new: int
    priority: float = 0.0
    state: str = "WAITING"; start: float = None; end: float = None
    generated: int = 0; prefilled: int = 0

def make_reqs(n, dist, seed, plen=(5, 30), mnew=(5, 20)):
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.exponential(1.0 / 0.6)
        p = int(rng.integers(plen[0], plen[1] + 1))
        m = int(rng.integers(mnew[0], mnew[1] + 1))
        if dist == "长尾(20% 超长请求)":
            if rng.random() < 0.2:
                p = int(p * 2.5); m = int(m * 2.5)
        reqs.append(Req(i, t, p, m, priority=float(rng.random())))
    return reqs

def pick_next(waiting, policy):
    if policy == "FCFS":
        return sorted(waiting, key=lambda r: r.arrive)
    if policy == "SJF":
        return sorted(waiting, key=lambda r: r.prompt_len + r.max_new)
    if policy == "Priority":
        return sorted(waiting, key=lambda r: -r.priority)
    raise ValueError(policy)

class SimpleScheduler:
    """手写简化调度器:一次 schedule() = 一个迭代的完整调度决策"""
    def __init__(self, token_budget=8, policy="FCFS"):
        self.token_budget, self.policy = token_budget, policy
        self.waiting, self.running, self.finished = [], [], []
        self.t, self.done, self.total = 0.0, 0, 0
        self.log = []

    def admit(self, reqs):
        self.total = len(reqs)
        self.waiting.extend(reqs)

    def schedule(self):
        self.waiting = pick_next(self.waiting, self.policy)
        for r in list(self.waiting):
            if r.arrive <= self.t:
                self.running.append(r); self.waiting.remove(r)
        budget = self.token_budget
        plan = []
        for r in self.running:
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use; r.prefilled += use
                plan.append((r.rid, use, "prefill"))
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = self.t
        for r in self.running:
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
                plan.append((r.rid, 1, "decode"))
        for r in list(self.running):
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", self.t
                self.running.remove(r); self.finished.append(r); self.done += 1
        self.log.append(dict(step=self.t, plan=plan, running=[x.rid for x in self.running]))
        self.t += 1.0

    def run_all(self):
        while self.done < self.total:
            self.schedule()
        return self.waiting + self.running + self.finished, self.log

# ---------------------------------------------------------------- 参数
with st.sidebar:
    st.header("🎛️ 参数")
    policy = st.radio("调度策略", ["FCFS", "SJF", "Priority"])
    token_budget = st.slider("token 预算(每步)", 2, 24, 8, 1)
    n = st.slider("请求数量", 8, 40, 20, 1)
    dist = st.radio("请求长度分布", ["均匀", "长尾(20% 超长请求)"])
    seed = st.slider("随机种子", 0, 99, 11, 1)
    compare = st.checkbox("与 FCFS 基准对比", value=True)
    st.caption("💡 预算 ≈ vLLM max_num_scheduled_tokens;策略决定 waiting 队列的取人顺序")

# ---------------------------------------------------------------- 运行
s = SimpleScheduler(token_budget, policy)
s.admit(make_reqs(n, dist, seed))
reqs, log = s.run_all()
df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                        work=r.prompt_len + r.max_new, prio=round(r.priority, 2)) for r in reqs])
df["latency"] = df["end"] - df["arrive"]
df["wait"] = df["start"] - df["arrive"]
ms = float(df.end.max())

if compare:
    s0 = SimpleScheduler(token_budget, "FCFS")
    s0.admit(make_reqs(n, dist, seed))
    _, _ = s0.run_all()
    d0 = pd.DataFrame([dict(latency=r.end - r.arrive, wait=r.start - r.arrive) for r in s0.waiting + s0.running])

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"平均延迟({policy})", f"{df.latency.mean():.1f} 步",
          delta=f"{d0.latency.mean() - df.latency.mean():+.1f} vs FCFS" if compare else None)
c2.metric("P95 延迟", f"{np.percentile(df.latency, 95):.1f} 步")
c3.metric("平均等待", f"{df.wait.mean():.1f} 步")
c4.metric("吞吐(请求/步)", f"{n / ms:.3f}")

# ---------------------------------------------------------------- 每步调度表
st.subheader("📋 每步调度决策(最近 12 步)")
rows = []
for entry in log[-12:]:
    plan_txt = ", ".join(f"R{rid}+{tk}{'P' if k == 'prefill' else 'D'}" for rid, tk, k in entry["plan"][:8])
    rows.append(dict(步骤=entry["step"], 调度=plan_txt or "空", running=str(entry["running"])))
st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.caption("P = prefill token,D = decode token;预算扣完即停——看 budget 如何卡住每一步。")

# ---------------------------------------------------------------- 延迟分布
st.subheader("📊 延迟分布")
fig = make_subplots(rows=1, cols=2, subplot_titles=("延迟直方图", "延迟箱线图"))
fig.add_trace(go.Histogram(x=df.latency, nbinsx=14, marker_color="#4C78A8",
                           name=policy, opacity=0.85), row=1, col=1)
fig.add_trace(go.Box(y=df.latency, name=policy, marker_color="#4C78A8", boxmean=True), row=1, col=2)
if compare:
    fig.add_trace(go.Histogram(x=d0.latency, nbinsx=14, marker_color="#E45756",
                               name="FCFS", opacity=0.5), row=1, col=1)
    fig.add_trace(go.Box(y=d0.latency, name="FCFS", marker_color="#E45756", boxmean=True), row=1, col=2)
fig.update_layout(title=f"{policy} 策略下的延迟分布(与 FCFS 对比)", height=380,
                  margin=dict(l=10, r=10, t=50, b=10), showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 调度时间线
fig2 = go.Figure()
for _, row in df.iterrows():
    fig2.add_trace(go.Bar(x=[row.wait], y=[f"R{row.rid}"], base=[row.arrive],
                          orientation="h", marker_color="#BAB0AC", showlegend=False, width=0.6))
    fig2.add_trace(go.Bar(x=[row.latency - row.wait], y=[f"R{row.rid}"], base=[row.start],
                          orientation="h", marker_color="#54A24B", showlegend=False, width=0.6))
fig2.update_layout(title=f"{policy} 调度甘特图:灰=等待,绿=执行(等待短 = 调度快)",
                   xaxis_title="时间(步)", yaxis_title="请求", height=60 + 28 * n,
                   margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
st.plotly_chart(fig2, use_container_width=True)
