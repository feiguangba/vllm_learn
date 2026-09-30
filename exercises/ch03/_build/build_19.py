# -*- coding: utf-8 -*-
"""生成 19_iterative_scheduler_sim.ipynb 与 app_19_simulator.py"""
from helpers import (D, chapter_cover, wrapup, new_nb, app_cell, CH03)
from pathlib import Path

APP_19 = D('''
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
''')

NB = new_nb("第 19 课 · 完整调度模拟器与参数敏感性实验",
            subtitle="把 到达率 × 预算 × 策略 × 显存 × 抢占 拧进一个 Simulator,像风洞实验一样一次只动一个参数",
            emoji="🛰️")

chapter_cover(NB,
    objectives=[
        "把前五课的所有旋钮整合成一个完整 Simulator 类,run() 一次输出全部指标",
        "掌握参数敏感性实验的方法论:一次只动一个变量,其余锁定",
        "扫描到达率与 token 预算,绘制吞吐/延迟双轴曲线并解释拐点",
        "用热力图做策略 × 显存的双因素分析,建立“调参前先扫描”的习惯",
    ],
    toc=[
        ("直觉:风洞实验", "一次只动一个参数,其余全锁死,才能知道谁在起作用"),
        ("Simulator:把所有旋钮拧到一起", "一个类、一个 run(),包揽五课的全部机制"),
        ("基线运行:指标面板", "吞吐 / 延迟 / 抢占,一行代码全拿到"),
        ("敏感性 I:到达率 λ 扫描", "吞吐先升后饱和,延迟单调上涨——负载的脾气"),
        ("敏感性 II:token 预算扫描", "预算不是越大越好:拐点在哪?"),
        ("敏感性 III:策略 × 显存热力图", "pyecharts HeatMap 双因素分析"),
        ("结论:调度器的调参地图", "把三张图变成决策规则"),
        ("配套 Streamlit 演示", "app_19_simulator.py:全参数交互 + 扫描面板"),
    ],
    links=[
        ("vLLM Scheduler 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("vLLM 官方文档:引擎配置", "https://docs.vllm.ai/en/latest/features/engine_args.html"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
    ])

NB.md("## 1️⃣ 直觉:风洞实验 🌪️",
D('''
飞机设计师在风洞里调机翼,一次**只改一个参数**——改攻角就不动风速,改风速就不动攻角。
如果两个参数一起动,实验做完你根本说不清是哪个在起作用。

调度器也一样。前五课我们拆开了到达流、预算、策略、抢占,现在把它们**全部装进
一个 Simulator 类**,然后用风洞的方法论做**参数敏感性实验**:

1. 锁定其余所有参数;
  2. 只扫描目标参数 $x \\in \\{x_1, x_2, \\dots, x_k\\}$;
3. 记录每个 $x_i$ 下的吞吐与延迟,画曲线;
4. 解读拐点:为什么到这一步,再加参数就不灵了?

这套方法不仅适用于本课的玩具模拟器——**任何系统(数据库、缓存、推理引擎)
调优的第一步都是敏感性分析**。
'''))

NB.md("## 2️⃣ Simulator:把所有旋钮拧到一起 🛠️",
D('''
把 `make_reqs`(到达流)、`pick_next`(策略)、预算分配、抢占逻辑整合成一个类。
注意类的设计意图:所有参数在 `__init__` 里暴露,`run()` 一次跑完并返回
`(stats, df, events)` 三件套——指标、请求明细、抢占事件,一个不少。
'''))

NB.code(D('''
from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class Req:
    rid: int; arrive: float; prompt_len: int; max_new: int
    priority: float = 0.0
    state: str = "WAITING"; start: float = None; end: float = None
    generated: int = 0; prefilled: int = 0
    preempted: int = 0; resume_left: int = 0

class Simulator:
    """完整调度模拟器:一次 run() 返回 指标 + 请求明细 + 事件流"""
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
        running, finished, t, done, total = [], [], 0.0, 0, len(waiting)
        events = []
        while done < total:
            waiting = self.pick_next(waiting)
            new = [r for r in waiting if r.arrive <= t]      # ① 新到达
            for r in new:
                waiting.remove(r)
            for r in list(new):                              # ② 抢占或补位
                if self.max_running is None or len(running) < self.max_running:
                    running.append(r); new.remove(r)
                else:
                    victim = running.pop()                   # LIFO
                    victim.state, victim.preempted = "PREEMPTED", victim.preempted + 1
                    if self.preempt == "recompute":
                        victim.generated, victim.prefilled, victim.resume_left = 0, 0, 0
                        victim.state = "WAITING"
                    else:
                        victim.resume_left = self.swap_cost
                    waiting.insert(0, victim)
                    events.append(("抢占", t, victim.rid))
                    running.append(r); new.remove(r)
            for r in running:                                # ③ swap 恢复
                if r.resume_left > 0:
                    r.resume_left -= 1
                    if r.resume_left == 0 and r.state == "PREEMPTED":
                        r.state = "RUNNING"
            budget = self.token_budget
            for r in running:                                # ④ decode 优先(running 优先)
                if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                    r.generated += 1; budget -= 1
            for r in running:                                # ⑤ 剩余预算分块 prefill
                if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                    use = min(budget, r.prompt_len - r.prefilled)
                    budget -= use; r.prefilled += use
                    if r.prefilled >= r.prompt_len:
                        r.state = "RUNNING"
                        if r.start is None:
                            r.start = t
            for r in list(running):                          # ⑥ 完成
                if r.generated >= r.max_new:
                    r.state, r.end = "FINISHED", t
                    running.remove(r); finished.append(r); done += 1
            t += 1.0
        all_reqs = waiting + running + finished
        df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                                work=r.prompt_len + r.max_new, preempted=r.preempted) for r in all_reqs])
        df["latency"] = df["end"] - df["arrive"]
        df["wait"] = df["start"] - df["arrive"]
        ms = float(df.end.max())
        stats = dict(makespan=ms, req_per_step=float(self.n / ms),
                     token_per_step=float(df.work.sum() / ms),
                     avg_latency=float(df.latency.mean()),
                     p95_latency=float(np.percentile(df.latency, 95)),
                     preemptions=len(events))
        return stats, df, events
'''), "🛠️ 注意 `run()` 的六个步骤与前面各课的对应关系——这就是“组装”的艺术:先拆解,再整合。")

NB.md("## 3️⃣ 基线运行:一行代码出全部指标 📊",
D('''
先用默认参数跑一次基线,拿到“参考点”,后面所有扫描都跟它比:
'''))

NB.code(D('''
sim = Simulator(seed=1)                    # 全部默认:FCFS / 预算 8 / 无抢占
stats, df, events = sim.run()
for k, v in stats.items():
    print(f"{k:<14} = {v:.3f}")
'''), "✅ 这就是模拟器的“面板”:吞吐、延迟、抢占,一行代码全拿到。")

NB.md("## 4️⃣ 敏感性 I:到达率 λ 扫描(负载的脾气) 📈",
D('''
把 λ 从 0.2 一路加到 1.6(请求越来越密),其余全锁死。画**双轴曲线**:
左轴吞吐(请求/步),右轴平均延迟(步)。

预期形状:

- **吞吐**:先线性上升(负载不满),然后**饱和**(预算/算力见顶),再往后只涨排队不涨吞吐;
- **平均延迟**:前期平缓,过了饱和点后**陡峭上涨**(排队论里这叫 "utilization approaching 1")。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

lambdas = np.round(np.arange(0.2, 1.7, 0.15), 2)
rows = []
for lam in lambdas:
    s, d, _ = Simulator(rate=lam, n=40, seed=1).run()
    rows.append(dict(lam=lam, tput=s["req_per_step"], lat=s["avg_latency"]))
scan = pd.DataFrame(rows)
print(scan.round(3).to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Scatter(x=scan.lam, y=scan.tput, name="吞吐(请求/步)", yaxis="y1",
                         line=dict(color="#54A24B", width=3)))
fig.add_trace(go.Scatter(x=scan.lam, y=scan.lat, name="平均延迟(步)", yaxis="y2",
                         line=dict(color="#E45756", width=2, dash="dot")))
fig.update_layout(title="敏感性:到达率 λ(实线=吞吐,虚线=延迟)",
                  xaxis_title="到达率 λ", yaxis=dict(title="吞吐(请求/步)"),
                  yaxis2=dict(title="平均延迟(步)", overlaying="y", side="right"),
                  height=420, margin=dict(l=10, r=10, t=50, b=10), hovermode="x")
fig.show()
'''), "📊 找到吞吐的“膝盖”(饱和点):它告诉你这台“GPU”该配多大的请求压力——超过它,延迟暴涨,吞吐纹丝不动。")

NB.md("## 5️⃣ 敏感性 II:token 预算扫描(预算不是越大越好) 🪙",
D('''
扫描 `token_budget ∈ [2, 4, ..., 24]`。预期:

- 预算小:每步能干的活少,吞吐低、排队久;
- 预算大:每步 batch 更满,吞吐上升——但到一定程度后**边际递减**:
  GPU 单步算力已到顶,再多预算也只是“一步干不完,留到下一步”;
- 延迟在预算变大后**先降后升**:预算太小抢不到算力,预算太大则 prefill
  一次性吃下整块预算,把 decode 请求挤到下一个迭代(抢占与延迟的交锋)。
'''))

NB.code(D('''
budgets = np.arange(2, 26, 2)
rows = []
for b in budgets:
    s, d, _ = Simulator(rate=0.8, n=40, token_budget=int(b), seed=1).run()
    rows.append(dict(budget=int(b), tput=s["req_per_step"], lat=s["avg_latency"]))
scan2 = pd.DataFrame(rows)
print(scan2.round(3).to_string(index=False))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=scan2.budget, y=scan2.tput, name="吞吐", yaxis="y1",
                          line=dict(color="#4C78A8", width=3)))
fig2.add_trace(go.Scatter(x=scan2.budget, y=scan2.lat, name="平均延迟", yaxis="y2",
                          line=dict(color="#F2C14E", width=2, dash="dot")))
fig2.update_layout(title="敏感性:token 预算(注意拐点位置)", xaxis_title="token 预算",
                   yaxis=dict(title="吞吐"), yaxis2=dict(title="平均延迟", overlaying="y", side="right"),
                   height=420, margin=dict(l=10, r=10, t=50, b=10), hovermode="x")
fig2.show()
'''), "🔍 找到吞吐的饱和点后,把预算定在“饱和点的 1.2 倍左右”,既不浪费也不吃亏——这就是敏感性分析直接产出的调参建议。")

NB.md("## 6️⃣ 敏感性 III:策略 × 显存热力图 🔥",
D('''
双因素分析:策略 × 显存并发上限,输出**平均延迟热力图**。热力图能一眼看出
"哪个因素起主导作用":

- 如果**列与列之间差别大** → 显存上限是主导变量;
- 如果**行与行之间差别大** → 策略是主导变量。
'''))

NB.code(D('''
from pyecharts.charts import HeatMap
from pyecharts import options as opts

caps = [2, 3, 4, 6, 8, 12]
policies = ["FCFS", "SJF", "Priority"]
data, maxv = [], 0.0
for i, p in enumerate(policies):
    for j, cap in enumerate(caps):
        s, d, _ = Simulator(rate=0.8, n=40, policy=p, max_running=cap, seed=1).run()
        v = round(float(s["avg_latency"]), 1)
        data.append([j, i, v])
        maxv = max(maxv, v)

hm = (HeatMap()
      .add_xaxis([f"上限={c}" for c in caps])
      .add_yaxis("策略", policies, data)
      .set_global_opts(title_opts=opts.TitleOpts(title="平均延迟热力图:策略 × 显存上限"),
                       visualmap_opts=opts.VisualMapOpts(max_=maxv, is_calculable=True,
                                                         range_color=["#edf8b1", "#2c7fb8", "#d95f0e"])))
hm.render_notebook()
'''), "📊 用鼠标在热力图上划过,每个格子都带出精确数值;颜色梯度本身就是一张“调参地图”。")

NB.md("## 7️⃣ 真实 GPU 参考墙:把「步」换算成 token/s ⏱️",
D('''
前面的敏感性实验全在「步」这个抽象单位里——步数不等于秒。给模拟器配一堵**真实参考墙**:
用 `bench_throughput_curve` 在真实 GPU 上量出「一步(迭代)实际多少毫秒」,把模拟器的
`token/步` 乘上一个真实换算因子,就得到可对外的 token/s,让你心里有数:这套配置在真机上能跑多快。
'''))

NB.code(D('''
import sys, os
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

# 1) 真实参考墙:固定 batch 在 GPU 上的单步耗时(ms/迭代)
b_list, tps, mps = bench_throughput_curve(batch=(4, 8, 16, 32), token_len=16, reps=5)
print("设备:", cuda_info())
print("真实 batch 吞吐曲线:")
for b, ms, tp in zip(b_list, mps, tps):
    print(f"  batch={b:2d}  单步 {ms:6.3f} ms  →  {tp/1e3:6.1f} k token/s")

# 2) 把模拟器基线的"token/步"乘以真实单步耗时 → 估计真实 token/s
sim = Simulator(seed=1)
stats, _, _ = sim.run()
step_ms = float(np.median(mps))                     # 参考墙:取 batch≈16 量级的单步耗时
est_tps = stats["token_per_step"] * (1000.0 / step_ms)
print(f"\\n模拟器基线: 吞吐 = {stats['token_per_step']:.2f} token/步 · makespan = {stats['makespan']:.0f} 步")
print(f"叠加真实单步 {step_ms:.3f} ms → 这台配置约 {est_tps/1e3:.1f} k token/s (参考墙)")
'''), "🚀 真实数字做参照物。模拟器内部仍以「步」为逻辑单位(与调度无关),真实曲线只负责把步换算成秒——这就是第 20 课真机实验的“热身”。")

NB.md("## 8️⃣ 结论:调参的地图 🗺️",
D('''
把三张图合起来,得到一条可复用的决策链:

1. **负载侧**:λ 扫描找到吞吐饱和点 → 决定这台机器能扛多少并发;
2. **算力侧**:token 预算扫描找到边际递减拐点 → 决定每步排多少活;
3. **资源侧**:策略 × 显存热力图 → 决定“先服务谁”与“显存给多大”。

> 🏷️ 这也正是 vLLM 生产调优的核心动作:看 `max_num_scheduled_tokens` 与
> `max_num_seqs` 对 `generation_tokens_per_second` 的敏感性曲线。
> 我们五课以来搭的玩具模拟器,和真机之间只差“真实 kernel 耗时”,**方法论完全一致**。
'''))

NB.md("## 9️⃣ 配套 Streamlit 演示:全参数风洞 🛰️",
D('''
运行 `app_19_simulator.py`:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_19_simulator.py
```

左边一排旋钮是“风洞设置”,右边实时刷新指标面板、甘特图、敏感性扫描双轴曲线
与策略 × 显存热力图。完整源码如下(与 `app_19_simulator.py` 一致):
'''))

app_cell(NB, APP_19, "app_19_simulator.py",
         "📜 app_19_simulator.py 完整源码(守卫包裹):全参数交互模拟器 + 扫描面板 + 热力图。")

wrapup(NB,
    summary=[
        "Simulator 类把 到达流/预算/策略/显存/抢占 整合成一次 run(),输出 stats+df+events",
        "敏感性实验方法论:一次只动一个参数,其余锁定,找拐点",
        "λ 扫描:吞吐先升后饱和,延迟在饱和点后陡涨(排队论效应)",
        "token 预算扫描:边际递减,预算定在饱和点 1.2 倍最划算",
        "热力图适合双因素分析,一眼看出哪个变量主导",
    ],
    practice=[
        "给 Simulator 增加 plen/mnew 的“长尾”选项(20% 请求 ×2.5),重跑 λ 扫描,观察饱和点是否左移",
        "扫描 swap_cost ∈ [1..10](抢占模式下),画“swap_cost vs 平均延迟”,与第 18 课的公式对照",
        "把热力图的指标从平均延迟换成 P95,重新解读一次",
        "给 Simulator 加 triton 式的“每秒迭代数”换算(假设每步 5ms),输出真实的 tokens/sec",
    ],
    links=[
        ("vLLM 引擎参数文档", "https://docs.vllm.ai/en/latest/features/engine_args.html"),
        ("vLLM Scheduler 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
    ])

NB.save(str(Path(CH03) / "19_iterative_scheduler_sim.ipynb"))

app_path = Path(CH03) / "app_19_simulator.py"
app_path.write_text(APP_19 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")