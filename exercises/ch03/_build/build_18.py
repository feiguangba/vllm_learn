# -*- coding: utf-8 -*-
"""生成 18_preemption.ipynb 与 app_18_preemption.py"""
from helpers import (D, chapter_cover, wrapup, new_nb, app_cell, CH03)
from pathlib import Path

APP_18 = D('''
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
''')

NB = new_nb("第 18 课 · 抢占:显存不够,recompute 还是 swap?",
            subtitle="被赶下车的请求怎么办?—— 重算一切的 recompute,与保留进度的 swap,谁更划算",
            emoji="⚔️")

chapter_cover(NB,
    objectives=[
        "理解抢占的触发条件:显存不够时,新请求会把 running 请求赶下车(LIFO)",
        "区分 recompute(重算)与 swap(搬移)两条恢复路线,并量化解法成本",
        "手写带抢占的模拟器,输出抢占事件与延迟惩罚",
        "在 vLLM 源码里找到抢占的实现:request.num_preemptions 与 recompute 默认策略",
    ],
    toc=[
        ("直觉:餐厅翻台与外卖重做", "桌子不够时,倒掉重做 vs 端回后厨保温,各有代价"),
        ("vLLM 的真实设计", "为什么 vLLM 默认 recompute?swap 在什么时候用?"),
        ("显存压力模型与抢占模拟器", "max_running 并发上限 + LIFO 抢占规则,翻译成代码"),
        ("实验:压力越大,代价越大", "显存压力扫描,抢占次数与平均延迟的曲线"),
        ("甘特图:被抢占的伤疤", "plotly 甘特图 + 抢占时刻标记"),
        ("recompute vs swap 正面交锋", "pyecharts 对比图:两种策略的延迟惩罚"),
        ("数学:两种代价公式", "重算代价 vs 搬移代价,什么时候该用谁"),
        ("与 vLLM 对接:num_preemptions 指标", "把抢占变成可观测的指标"),
        ("配套 Streamlit 演示", "app_18_preemption.py:压力/策略/恢复开销三旋钮"),
    ],
    links=[
        ("vLLM Scheduler _preempt_request 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
    ])

NB.md("## 1️⃣ 直觉:餐厅翻台与外卖重做 🍳",
D('''
餐厅座位满了,来了新客人,怎么办?两种做法:

- 🔄 **倒掉重做(recompute)**:把吃到一半的菜倒掉,客人重新点菜、后厨重做。
  厨师(GPU)多干一份活,但**不占多余的地方**(不需要冰箱);
- 💾 **端回后厨保温(swap)**:菜端到后厨保温箱(CPU 内存)放着,客人复桌后再端回来。
  菜没浪费,但**端来端去要时间**,而且保温箱(CPU 带宽)也不是无限的。

映射到推理引擎:**KV cache** 就是那盘“吃到一半的菜”。显存不够时,
新请求(prefill 需要 KV 空间)必须挤掉某个 running 请求——这就是**抢占(preemption)**。
被挤掉的请求只有两条路:丢掉 KV 重新算(recompute),或者把 KV 搬去 CPU 再搬回来(swap)。
'''))

NB.md("## 2️⃣ vLLM 的真实设计:默认 recompute 🏛️",
D('''
翻开 `vendor/vllm/vllm/v1/core/sched/scheduler.py`,抢占的入口是 `_preempt_request`:

```python
# 第 1349 行附近
assert request.status == RequestStatus.RUNNING   # 只有 running 才能被抢占
request.status = RequestStatus.PREEMPTED
request.num_preemptions += 1                     # 记下被抢次数
```

vLLM v1 调度器的默认策略是 **recompute**:被抢占请求的 KV block 直接释放,
重新进入 waiting 队列,恢复时从头 prefill。为什么不用 swap?因为 vLLM 的 KV
管理基于 PagedAttention 的 block 表,recompute 的实现最干净——**重算只是把
GPU 时间表里再排一次**,而 swap 需要管理 CPU 显存池、异步搬移与更复杂的一致性。

swap 依然存在于代码路径中(配合 CPU offload / 多卡场景),但主流推理里
**显存是瓶颈、算力不是**:宁可多算,也不愿卡在搬移带宽上。本课我们用模拟器
把两种策略的账算清楚,结论会比“默认值”更有说服力。
'''))

NB.md("## 3️⃣ 显存压力模型与抢占模拟器 🛠️",
D('''
把显存抽象成一个数字:`max_running` = 能同时运行的请求数(KV cache 放得下几个)。
规则:

1. 新请求到达,若 running 未满,直接进入;
2. 若 running 已满,按 **LIFO**(后进先出)抢占最后一个 running 请求;
   (vLLM 的默认 victim 选择也是“最晚进来的优先被抢”)
3. 被抢占的请求回到 waiting 队首:
   - `recompute`:清零 prefilled/generated,恢复后重新 prefill;
   - `swap`:置 `resume_left = swap_cost`,恢复期间占用座位但不产出 token(在搬 KV);
4. 其余规则与第 15 课完全一致(token 预算 + 分块 prefill)。
'''))

NB.code(D('''
from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class Req:
    rid: int; arrive: float; prompt_len: int; max_new: int
    state: str = "WAITING"; start: float = None; end: float = None
    generated: int = 0; prefilled: int = 0
    preempted: int = 0; resume_left: int = 0

def make_reqs(n=20, mode="burst", rate=0.5, seed=3, plen=(5, 30), mnew=(5, 20)):
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
    """带抢占的 continuous batching 模拟器(LIFO 抢占)"""
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    events = []   # (类型, 时间, 请求, 对方请求)
    while done < total:
        new = [r for r in waiting if r.arrive <= t]       # ① 本步新到达
        for r in new:
            waiting.remove(r)
        for r in list(new):                               # ② 被抢占请求优先复座
            if r.state == "PREEMPTED" and len(running) < max_running:
                running.append(r); new.remove(r)
        for r in list(new):                               # ③ 新请求进入,running 满则抢位
            if len(running) < max_running:
                running.append(r); new.remove(r)
            else:
                victim = running.pop()                    # LIFO:抢最后进来的
                victim.state, victim.preempted = "PREEMPTED", victim.preempted + 1
                if policy == "recompute":
                    victim.generated, victim.prefilled = 0, 0   # KV 作废
                    victim.state, victim.resume_left = "WAITING", 0
                else:
                    victim.resume_left = swap_cost              # KV 已搬去 CPU
                waiting.insert(0, victim)
                events.append(("抢占", t, victim.rid, r.rid))
                running.append(r); new.remove(r)
        for r in running:                                 # ④ swap:搬 KV 期间不产出
            if r.resume_left > 0:
                r.resume_left -= 1
                if r.resume_left == 0 and r.state == "PREEMPTED":
                    r.state = "RUNNING"                    # KV 搬回,继续 decode
        budget = token_budget
        for r in running:                                 # ⑤ decode 优先(避免被 prefill 饿死)
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in running:                                 # ⑥ 剩余预算分块 prefill
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use; r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in list(running):                           # ⑦ 完成离场
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
'''), "🛠️ 与第 15 课的 continuous 模拟器只差“抢占三连”:满员判断、LIFO 弹栈、回队重排。")

NB.md("## 4️⃣ 实验:显存压力扫描,代价有多大 📉",
D('''
固定策略(recompute),把 `max_running` 从 8 一路压到 2,看抢占次数与平均延迟
如何膨胀:
'''))

NB.code(D('''
rows = []
for cap in [8, 6, 4, 3, 2]:
    reqs, events = simulate_preempt(make_reqs(seed=3), max_running=cap)
    df = summarize(reqs)
    rows.append(dict(max_running=cap,
                     抢占次数=sum(1 for e in events if e[0] == "抢占"),
                     平均延迟=round(float(df.latency.mean()), 1),
                     最大延迟=int(df.latency.max()),
                     被抢请求人均次数=round(float(df.preempted.mean()), 2)))
print(pd.DataFrame(rows).to_string(index=False))
'''), "✅ 看表格:并发上限从 8 压到 2,平均延迟膨胀了多少?抢占次数每+1,账上就要多付一次重算。")

NB.md("## 5️⃣ 甘特图:被抢占的伤疤 📈",
D('''
画甘特图时,把**抢占时刻**用红色 ✕ 标记在对应请求的时间线上——
每次 ✕ 都是一次“重做”(recompute)或一次“搬移”(swap):
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

reqs, events = simulate_preempt(make_reqs(seed=3), max_running=3)
df = summarize(reqs)
fig = go.Figure()
for _, row in df.iterrows():
    fig.add_trace(go.Bar(x=[row.end - row.start], y=[f"R{row.rid}"], base=[row.start],
                         orientation="h", marker_color="#54A24B", showlegend=False, width=0.6))
pre = [e for e in events if e[0] == "抢占"]
fig.add_trace(go.Scatter(x=[e[1] for e in pre], y=[f"R{e[2]}" for e in pre], mode="markers",
                         marker=dict(symbol="x", size=13, color="#E45756", line=dict(width=2)),
                         name="抢占时刻"))
fig.update_layout(title="抢占甘特图:红 ✕ 每出现一次,请求就白干一段(recompute)或搬一次家(swap)",
                  xaxis_title="时间(步)", yaxis_title="请求",
                  height=60 + 30 * len(reqs), margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
fig.show()
'''), "🎨 同一个请求身上出现多个 ✕ = 反复被抢 = 反复重算——这种情况真实系统里要尽量避免(通常用配额限制)。")

NB.md("## 6️⃣ recompute vs swap:正面交锋 ⚔️",
D('''
现在让两种策略在同一显存压力、同一请求流下对决。核心变量:
prompt 长度(决定重算代价)与 `swap_cost`(决定搬移代价)。
'''))

NB.code(D('''
def compare_policies(n=20, seed=3, max_running=3, token_budget=8, swap_cost=3.0):
    out = {}
    for policy in ["recompute", "swap"]:
        reqs, events = simulate_preempt(make_reqs(n, seed=seed), token_budget,
                                        max_running, policy, swap_cost)
        df = summarize(reqs)
        out[policy] = dict(抢占=sum(1 for e in events if e[0] == "抢占"),
                           平均延迟=float(df.latency.mean()),
                           最大延迟=float(df.latency.max()),
                           墙钟=float(df.end.max()))
    return out

r = compare_policies()
print(pd.DataFrame(r).T.round(1).to_string())
print(f"\\nswap 相对 recompute:平均延迟 {'-%.1f' % (r['recompute']['平均延迟'] - r['swap']['平均延迟'])} 步")
'''), "🔍 同一套请求,recompute 与 swap 各有胜负——胜负手就是 prompt 长度与 swap_cost 之比。")

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

bar = (Bar()
       .add_xaxis(["平均延迟(步)", "最大延迟(步)", "墙钟时间(步)"])
       .add_yaxis("recompute", [r["recompute"][k] for k in ("平均延迟", "最大延迟", "墙钟")],
                  color="#F2C14E")
       .add_yaxis("swap", [r["swap"][k] for k in ("平均延迟", "最大延迟", "墙钟")],
                  color="#72B7B2")
       .set_global_opts(title_opts=opts.TitleOpts(title="recompute vs swap:延迟惩罚对比"),
                        yaxis_opts=opts.AxisOpts(name="步数"),
                        legend_opts=opts.LegendOpts(pos_top="6%")))
bar.render_notebook()
'''), "📊 三个指标、两组柱子:谁低谁赢。把 swap_cost 调成 8 再跑一次,胜利天平就倒向 recompute 了。")

NB.md("## 7️⃣ 数学:两种代价公式 🧮",
D('''
设请求 $r$ 被抢占时已完成的 prefill+decode token 数为 $p_r$(recompute 下全部作废),
`swap_cost` 为搬一次家的固定步数 $C$。

- **recompute 的浪费**($P_r$ = 总提示词长度 + 已生成数):

$$W_r^{re} = P_r \\times \\mathbb{1}[r \\text{ 被抢占}]$$

- **swap 的浪费**(与进度无关,只看次数):

$$W_r^{sw} = C \\times \\text{preemptions}(r)$$

所以:

- prompt 很长、生成很长(大 $P_r$)→ recompute 的浪费大,**swap 更划算**;
- 抢占频繁(次数多)→ swap 的浪费线性累积,**recompute 更划算**;
- 决策法则:$\\mathbb{E}[P_r] > C \\times \\frac{1}{\\text{抢到后能活多久}}$ 时倾向 swap,
  否则 recompute。vLLM 默认选 recompute,正是因为在线场景抢占不算频繁,
  而 swap 的搬移带宽在 decode 密集时更稀缺。
'''))

NB.md("## 8️⃣ 真实 GPU:recompute vs swap 的真实代价账 ⏱️",
D('''
模拟器里 recompute 用「作废 token 数」、swap 用「固定步数」计价——现在把两条路的花费
真正放到 GPU 上量一量:**recompute = 把被抢占请求未完成的 prompt 重新 prefill 一遍**;
**swap = 把同量 token 的 KV cache 从 GPU 搬到 CPU(内存)再搬回来**。前者是「重算」,
后者是「搬运」。量的是真毫秒。

关键对照:两者都随 token 数增长,但增长的理由不同——recompute 吃的是**计算+权重带宽**,
swap 吃的是**主存传输带宽**。对 KV 较短的请求,谁的固定开销更小往往就决定胜负(这正是第 7 节的公式)。
'''))

NB.code(D('''
import sys, os
import torch, numpy as np, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
from vllm_real import bench_prefill_decode, cuda_info

print("设备:", cuda_info())
d_model, layers = 256, 8                       # 与 bench 一致的 TinyGPT 尺寸
kv_bytes_per_token = 2 * d_model * layers * torch.float32.itemsize   # K+V 每 token 字节

print("\\nL(token) | recompute=重算prefill(ms) | swap=KV搬到CPU(ms) | 本轮胜负")
res = []
for L in (64, 128, 256, 512):
    b = bench_prefill_decode(d=d_model, layers=layers, L=L, steps=64, reps=5)
    recompute_ms = b["prefill_ms"]
    # swap:把同样 token 数的 KV 从 GPU 拷到 CPU(近似 H2D/D2H 的传输成本)
    kv = torch.randn(L, 2*d_model*layers, dtype=torch.float32, device="cuda")
    t0 = time.perf_counter()
    kv.cpu().numpy()                            # GPU→CPU 拷贝(模拟 swap 一次搬出)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    swap_ms = (time.perf_counter() - t0) * 1e3
    res.append((L, recompute_ms, swap_ms))
    ver = "recompute 赢" if recompute_ms <= swap_ms else "swap 赢"
    print(f"{L:6d} | {recompute_ms:8.3f}            | {swap_ms:11.3f}   | {ver}")
'''), "🚀 **真实 GPU 数字**。注意：swap 这步只是 GPU→CPU 的一次搬运；真实 vLLM 的 swap 还要算上分配/异步/再搬回的往返，所以表中 swap 是偏乐观的下界——即便如此，在小 L 下 recompute 通常也不吃亏（kernel 启动开销小）。")

NB.md("本地这套小模型 KV 很小，差距可能不直观。把两路的**带宽账**算清楚——recompute 要重新读一遍权重做矩阵乘，"
"swap 要搬往返两份 KV——两者谁更划算取决于「重算的单位成本」与「搬运字节」之比，正是第 7 节公式的实测版。"
"真实 decode 密集场景里交换带宽(PCIe/HBM 到主存)远比算力稀缺，所以 vLLM 默认 recompute——不是它算得快，"
"而是 swap 的带宽代价更容易成为第二个瓶颈。"
"")

NB.md("## 9️⃣ 与 vLLM 对接:把抢占变成指标 🔗",
D('''
真实 vLLM 里“被抢占”不是一个黑盒事件,而是一组可观测指标:

- `request.num_preemptions`:单个请求被抢次数(我们的 `preempted` 字段就是它的影子);
- 调度器输出 `preempted_req_ids`:本轮被抢的请求列表(见 `scheduler.py` 的
  `SchedulerOutput`),日志与监控都靠它;
- 抢占后的重算发生在同一批请求被重新调度时——正如我们的模拟器:
  **抢走的是“这次迭代的显存权利”,不是“请求本身”**。

> 💡 实战建议:如果你在日志里看到大量 `PREEMPTED`,先检查 `max_num_seqs`
> 是否过大、KV cache 分配是否过激——抢占是结果,不是原因。
'''))

NB.md("## 1️⃣0️⃣ 配套 Streamlit 演示:压力旋钮在手,代价看得见 🎛️",
D('''
运行 `app_18_preemption.py`:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_18_preemption.py
```

调节**显存并发上限**(压力)、**抢占策略**与 **swap 恢复耗时**三个旋钮,
抢占事件表、甘特图 ✕ 标记与 recompute/swap 对比柱状图实时刷新。
完整源码如下(与 `app_18_preemption.py` 一致):
'''))

app_cell(NB, APP_18, "app_18_preemption.py",
         "📜 app_18_preemption.py 完整源码(守卫包裹):抢占事件 + 甘特标记 + 策略对比,一个文件全包含。")

wrapup(NB,
    summary=[
        "抢占的触发:显存不够时新请求挤掉 running 请求,vLLM 按 LIFO 选 victim",
        "recompute:KV 作废从头重算,代价正比于请求的 token 总量;vLLM 默认策略",
        "swap:KV 搬到 CPU 再搬回,代价正比于次数 × 搬运耗时,但保留进度",
        "代价公式 W_re ≈ 总 token 数,W_sw ≈ C × 次数;prompt 长选 swap、抢占频繁选 recompute",
        "vLLM 把抢占做成可观测指标:num_preemptions + preempted_req_ids",
    ],
    practice=[
        "扫描 max_running ∈ [2..8],画“抢占次数 vs 平均延迟”双轴图,找临界点",
        "把 recompute 与 swap 的墙钟时间之差画成 swap_cost 的函数,找交叉点",
        "给模拟器加“同一请求最多被抢 2 次,否则直接丢弃(FINISHED_ABORTED)”的规则,观察对平均延迟的影响",
        "阅读 scheduler.py 里 _preempt_request 的完整实现,列出它与模拟器的 3 处差异",
    ],
    links=[
        ("vLLM Scheduler _preempt_request", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
    ])

NB.save(str(Path(CH03) / "18_preemption.ipynb"))

app_path = Path(CH03) / "app_18_preemption.py"
app_path.write_text(APP_18 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")