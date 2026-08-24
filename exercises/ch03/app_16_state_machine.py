# -*- coding: utf-8 -*-
# app_16_state_machine.py — 请求状态机演示 🚦
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="请求状态机 🚦", layout="wide")
st.title("🚦 第 16 课 · 请求状态机:WAITING → RUNNING → FINISHED")

st.markdown("""
调度器眼里的每个请求,和物流包裹一样**有明确的状态**:
`WAITING`(排队)→ `RUNNING`(正在算)→ `FINISHED`(完成);
显存不够时还会被 `PREEMPTED`(抢占,回队重排)。
本演示让你**输入请求事件流**(调节参数即改变到达 / 抢占事件),实时观察
状态迁移时间线与迁移计数。
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

def simulate_fsm(reqs, token_budget=8, max_running=None):
    """continuous batching + 状态迁移日志;max_running 设定显存并发上限(可抢占)。"""
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    trans = []   # (时间, 请求, 旧状态, 新状态, 原因)
    def mv(r, to, reason):
        trans.append(dict(t=t, rid=r.rid, frm=r.state, to=to, reason=reason))
        r.state = to
    while done < total:
        new = [r for r in waiting if r.arrive <= t]        # ① 本步新到达
        for r in new:
            waiting.remove(r)
        if max_running is not None:                        # ② 抢占:running 满则抢 LIFO
            for r in list(new):
                if len(running) < max_running:
                    running.append(r)
                else:
                    victim = running.pop()
                    mv(victim, "PREEMPTED", "显存不足,被抢占")
                    victim.generated, victim.prefilled = 0, 0   # KV 作废,回队重排
                    waiting.insert(0, victim)
                    running.append(r)
        else:
            for r in new:
                running.append(r)
        budget = token_budget
        for r in running:                                  # ③ 分块 prefill(含重新 prefill)
            if r.state in ("WAITING", "PREEMPTED") and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    mv(r, "RUNNING", "prefill 完成")
                    if r.start is None:
                        r.start = t
        for r in running:                                  # ④ decode
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):                            # ⑤ 完成
            if r.generated >= r.max_new:
                mv(r, "FINISHED", "生成完毕")
                running.remove(r); done += 1
        t += 1.0
    return reqs, trans

def segments_from_trans(reqs, trans):
    """把迁移日志转换成 (rid, start, end, state) 分段,供状态时间线使用"""
    evs = {r.rid: [dict(t=r.arrive, state="WAITING")] for r in reqs}
    for e in trans:
        evs[e["rid"]].append(dict(t=e["t"], state=e["to"]))
    for r in reqs:
        evs[r.rid].append(dict(t=r.end, state="FINISHED"))
    segs = []
    for rid, ev in evs.items():
        ev.sort(key=lambda x: x["t"])
        for a, b in zip(ev[:-1], ev[1:]):
            if b["t"] > a["t"]:
                segs.append(dict(rid=f"R{rid}", start=a["t"], end=b["t"], state=a["state"]))
    return pd.DataFrame(segs)

# ---------------------------------------------------------------- 参数
with st.sidebar:
    st.header("🎛️ 参数(请求事件流)")
    n = st.slider("请求数量", 6, 24, 12, 1)
    token_budget = st.slider("token 预算(每步)", 2, 16, 8, 1)
    enable_preempt = st.checkbox("启用抢占(显存受限)", value=True)
    max_running = st.slider("显存并发上限 max_running", 1, 6, 3, 1) if enable_preempt else None
    mode = st.radio("到达模式", ["burst(同时到达)", "poisson(泊松流)"])
    rate = st.slider("到达率 λ", 0.1, 1.5, 0.5, 0.1) if mode == "poisson(泊松流)" else None
    seed = st.slider("随机种子", 0, 99, 5, 1)
    st.caption("⚠️ max_running 越小,显存压力越大,抢占事件越多")

# ---------------------------------------------------------------- 模拟与指标
reqs, trans = simulate_fsm(make_reqs(n, mode, rate, seed), token_budget, max_running)
df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end) for r in reqs])
df["latency"] = df["end"] - df["arrive"]
n_pre = sum(1 for e in trans if e["to"] == "PREEMPTED")
c1, c2, c3, c4 = st.columns(4)
c1.metric("状态迁移总数", len(trans))
c2.metric("抢占次数", n_pre)
c3.metric("平均延迟(步)", f"{df.latency.mean():.1f}")
c4.metric("吞吐(请求/步)", f"{n / df.end.max():.2f}")

# ---------------------------------------------------------------- 状态时间线
cmap = {"WAITING": "#BAB0AC", "RUNNING": "#54A24B", "PREEMPTED": "#E45756", "FINISHED": "#4C78A8"}
segs = segments_from_trans(reqs, trans)
fig = go.Figure()
for _, row in segs.iterrows():
    fig.add_trace(go.Bar(x=[row.end - row.start], y=[row.rid], base=[row.start],
                         orientation="h", marker_color=cmap[row.state],
                         hovertemplate=f"{row.rid}<br>{row.state}: %{{x|.0f}}~%{{x|.0f}}<extra></extra>",
                         showlegend=False, width=0.6))
for state, color in cmap.items():
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color=color, size=10), name=state))
fig.update_layout(title="请求状态时间线(灰=等待,绿=运行,红=被抢占,蓝=完成)",
                  xaxis_title="时间(步)", yaxis_title="请求", height=60 + 30 * n,
                  margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 红色段落 = 被抢占:请求回到 WAITING 重新 prefill;抢占越多,红线越长、延迟越痛。")

# ---------------------------------------------------------------- 事件流表格
st.subheader("📜 请求事件流(状态迁移日志)")
if trans:
    ev_df = pd.DataFrame(trans)
    ev_df.columns = ["时间(步)", "请求", "旧状态", "新状态", "原因"]
    ev_df["请求"] = ev_df["请求"].map(lambda x: f"R{x}")
    st.dataframe(ev_df, use_container_width=True, height=260)
else:
    st.info("当前参数下没有迁移事件——请求一次到位,全是 WAITING → RUNNING → FINISHED。")

# ---------------------------------------------------------------- 迁移 Sankey 图
st.subheader("🔁 状态迁移计数(Sankey)")
pairs = pd.DataFrame(trans).groupby(["frm", "to"]).size().reset_index(name="cnt")
all_states = ["WAITING", "RUNNING", "PREEMPTED", "FINISHED"]
idx = {s: i for i, s in enumerate(all_states)}
fig2 = go.Figure(go.Sankey(
    node=dict(label=all_states, color=[cmap[s] for s in all_states], pad=20, thickness=24),
    link=dict(source=[idx[r.frm] for r in pairs.itertuples()],
              target=[idx[r.to] for r in pairs.itertuples()],
              value=list(pairs.cnt),
              hovertemplate="%{source.label} → %{target.label}: %{value} 次<extra></extra>")))
fig2.update_layout(title="状态迁移流向图:谁去哪、去了多少次",
                   margin=dict(l=10, r=10, t=40, b=10), height=320)
st.plotly_chart(fig2, use_container_width=True)
