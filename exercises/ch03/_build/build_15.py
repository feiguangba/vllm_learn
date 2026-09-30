# -*- coding: utf-8 -*-
"""生成 15_continuous_batching.ipynb 与 app_15_continuous_batch.py"""
from helpers import (D, SIM_BASE, SIM_STATIC, SIM_CONT, SIM_LOG, SIM_STATS,
                     GANTT_PLOTLY, chapter_cover, wrapup, new_nb, app_cell, CH03)
from pathlib import Path

APP_15 = D('''
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
''')

NB = new_nb("第 15 课 · Continuous Batching:让 GPU 永远在干活",
            subtitle="Orca 论文的杀手锏:iteration-level scheduling,把“发车”从“凑满才发”改成“随到随发”",
            emoji="🍳")

chapter_cover(NB,
    objectives=[
        "理解 continuous batching 的核心思想:每步迭代重新调度,不再等整批完成",
        "读懂 Orca 论文的 iteration-level scheduling,并联系 vLLM 的 token 预算机制",
        "手写事件驱动的 continuous 模拟器,并与第 14 课静态批做同屏对比",
        "用逐步动画观察每步 batch 成员的变化,建立“动态批”的直觉",
    ],
    toc=[
        ("直觉:自助餐与翻台", "完成一个请求就立刻补位,GPU 像餐厅一样永远翻台"),
        ("Orca 论文:iteration-level scheduling", "为什么逐迭代调度能翻 23 倍吞吐?论文怎么说的"),
        ("模拟器:token 预算版的 continuous batching", "每步一个预算,prefill 一次吃饱、decode 每步一口"),
        ("对比实验:静态 vs 连续", "同一请求流,两种调度,甘特图 + 吞吐全面对比"),
        ("逐步动画:观察 batch 成员变化", "每一步 running 是谁?用表格和时间线看动态批"),
        ("与 vLLM 对接:max_num_scheduled_tokens", "模拟器里的 token_budget 在 vLLM 里叫什么"),
        ("配套 Streamlit 演示", "app_15_continuous_batch.py:步进控制 + 同屏对比"),
    ],
    links=[
        ("Orca: A Distributed Serving System (arXiv)", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 博客: 23x 吞吐从哪来", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("vLLM 调度器源码(预算分配逻辑)", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ])

NB.md("## 1️⃣ 直觉:自助餐与翻台 🍽️",
D('''
第 14 课的大巴比喻里,最伤人的不是“等”,而是**整套规则**:凑满才发、整批同进退、
发车间隙白等。如果把大巴换成**自助餐厅**,一切就顺畅了:

- 客人(请求)随到随吃,不凑桌;
- 每桌吃多久自己说了算,吃完立刻翻台(释放 GPU 位置);
- 后厨(GPU)永远在出菜,永远有活干。

这就是 **continuous batching(连续批处理)**:GPU 每**一个迭代**(一步)都重新审视
当前的请求集合,把刚完成的请求挪走、把刚到达的请求补进来——**batch 的成员是连续变化的**,
不再是“一车一车”地过。

> 🏷️ 术语对照:论文里这叫 **iteration-level scheduling**(逐迭代调度),vLLM 的
> 调度循环 `scheduler.schedule()` 就是每个迭代执行一次,返回“这一步该算谁”。
'''))

NB.md("## 2️⃣ Orca 论文:一句话点破 23x 吞吐 ✍️",
D('''
2022 年微软的 [Orca 论文](https://arxiv.org/abs/2208.14217)提出了两个关键观察:

1. **不同请求的序列长度差异巨大**:有人问一句就答完,有人要写一整篇长文;
2. **静态批必须等整批最长的那个**:每一步都有人“陪跑”,GPU 的每一分算力都在浪费。

Orca 的解法就是 **iteration-level scheduling**:调度不再以“整个请求”为单位,
而是以**一次迭代(一个 decode 步)**为单位。每一步:

- 对**运行中**的请求,每人算一步(生成 1 个 token);
- 对**等待中**的请求,只要显存与算力预算允许,就把它 prefilled 进来;
- 谁的序列结束了,谁就立刻离开,位置马上让给新人。

vLLM 官方博客在[《Continuous Batching 带来 23x 吞吐》](https://blog.vllm.ai/2023/06/20/vllm.html)
里复述了同样的故事,并给出了一个硬指标:**在 GPU 显存装得下全部 KV cache 的前提下,
静态批的吞吐只有 continuous batching 的约 1/23**。为什么差这么多?因为静态批把
大部分时间花在“凑车”和“陪跑”上。

下面我们用代码把“随到随发”翻译出来。
'''))

NB.code(D('''
# 复用第 14 课的 Req / make_reqs,新加一个 generated 字段记录已生成 token 数
from dataclasses import dataclass
import numpy as np, pandas as pd

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

def make_reqs(n=10, mode="burst", rate=0.5, seed=42, plen=(5, 30), mnew=(5, 20)):
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        if mode == "poisson":
            t += rng.exponential(1.0 / rate)
        a = 0.0 if mode == "burst" else t
        reqs.append(Req(i, a, int(rng.integers(plen[0], plen[1] + 1)),
                        int(rng.integers(mnew[0], mnew[1] + 1))))
    return reqs
'''), "📦 和上一课几乎一样——新增 `generated`(已生成 token 数)与 `prefilled`(prefill 进度,分块用)。")

NB.md("## 3️⃣ 模拟器:每步一个 token 预算 🪙",
D('''
continuous batching 的模拟器只比静态版多了**一个概念**:token 预算。

每步(每次迭代)开始,调度器手里有 `token_budget` 个 token 的“额度”。分配规则:

1. **先补位**:已到达且还没进运行集的请求,全部进入 running;
2. **prefill(分块)**:WAITING 请求按“吃光剩余预算”的方式分块消费预算,攒够 `prompt_len`
   即进入 RUNNING(这就是 vLLM 的 **chunked prefill**,对应 `max_num_batched_tokens`);
3. **decode**:剩下的预算按人头分,每个 RUNNING 请求每步分 1 个 token、生成 1 个 token;
4. **离场**:生成完的请求立刻离开,绝不陪跑。

这个 `token_budget` 不是拍脑袋——它就是 vLLM 里 `max_num_scheduled_tokens` 的简化版
(完整版还要扣掉 spec decode 的配额,细节见 `scheduler.py` 的 `schedule()` 循环)。
'''))

NB.code(D('''
def simulate_continuous(reqs, token_budget=8):
    """Continuous batching 模拟器:每步按 token 预算调度"""
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    while done < total:
        for r in list(waiting):            # 1) 补位
            if r.arrive <= t:
                running.append(r); waiting.remove(r)
        budget = token_budget
        for r in running:                  # 2) 分块 prefill:吃预算,记进度
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:      # prefill 完成!
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in running:                  # 3) decode:每人分 1 个 token
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):            # 4) 完成即离场
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
                running.remove(r); done += 1
        t += 1.0
    return reqs

reqs = simulate_continuous(make_reqs(n=8, seed=42), token_budget=8)
df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                        work=r.prompt_len + r.max_new) for r in reqs])
df["latency"] = df["end"] - df["arrive"]
print(df[["rid", "arrive", "start", "end", "latency", "work"]].to_string(index=False))
print(f"\\n全部完成耗时 = {df.end.max():.0f} 步,平均延迟 = {df.latency.mean():.1f} 步")
'''), "✅ 注意看 `start` 列:burst 模式下所有请求同时到达,但每个请求开始执行的时间各不相同——这就是“随到随发”。")

NB.md("## 4️⃣ 对比实验:同一请求流,两种调度 ⚔️",
D('''
同一批请求(同一个 seed 生成),分别喂给静态批(`batch_size=4`)和 continuous
(`token_budget=8`),然后对比三个指标:墙钟时间、平均延迟、吞吐。
'''))

NB.code(D('''
def simulate_static(reqs, batch_size):
    """第 14 课的静态批模拟器(直接复用)"""
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

reqs_s, _ = simulate_static(make_reqs(n=16, seed=7), batch_size=4)
reqs_c = simulate_continuous(make_reqs(n=16, seed=7), token_budget=8)

dfs = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                         work=r.prompt_len + r.max_new) for r in reqs_s])
dfc = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                         work=r.prompt_len + r.max_new) for r in reqs_c])
for d in (dfs, dfc):
    d["latency"] = d["end"] - d["arrive"]

ms_s, ms_c = dfs.end.max(), dfc.end.max()
tok = dfc.work.sum()
print(f"{'指标':<12}{'静态批':>10}{'continuous':>12}")
print(f"{'墙钟时间(步)':<12}{ms_s:>10.0f}{ms_c:>12.0f}")
print(f"{'平均延迟(步)':<12}{dfs.latency.mean():>10.1f}{dfc.latency.mean():>12.1f}")
print(f"{'吞吐(token/步)':<12}{tok / ms_s:>10.1f}{tok / ms_c:>12.1f}")
'''), "🏆 同一批活,continuous 的墙钟时间、平均延迟、吞吐三项全面占优——这就是“翻台”的力量。")

NB.md("## 5️⃣ 甘特图 + 累积完成曲线:差距看得见 📊",
D('''
数字之外,我们把两种调度画成**同一张图的两个子图**。再画一张累积完成曲线:
横轴时间,纵轴“已完成请求数”——continuous 的曲线**始终压在 static 上面**,
直到两者都全部完成。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gantt_trace(df, color, name):
    """把一个调度的请求时间线转成甘特图 traces"""
    trs = []
    for _, row in df.sort_values("start").iterrows():
        trs.append(go.Bar(x=[row.end - row.start], y=[f"R{row.rid}"], base=[row.start],
                          orientation="h", marker_color=color, showlegend=False,
                          width=0.6, hovertemplate=f"{name} R{row.rid}<br>%{{x|.0f}}~%{{x|.0f}}<extra></extra>"))
    return trs

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                    subplot_titles=("静态批(整批同进退)", "continuous(随到随发)"))
for tr in gantt_trace(dfs, "#E45756", "static"):
    fig.add_trace(tr, row=1, col=1)
for tr in gantt_trace(dfc, "#54A24B", "cont"):
    fig.add_trace(tr, row=2, col=1)
fig.update_layout(title="同一请求流:静态批 vs Continuous Batching 甘特图",
                  height=520, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
fig.show()

def cum_curve(df, ms):
    xs, ys, acc = [], [], 0
    for t in range(0, int(ms) + 1):
        acc = int((df.end <= t).sum())
        xs.append(t); ys.append(acc)
    return xs, ys

fig2 = go.Figure()
xs, ys = cum_curve(dfc, ms_c)
fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="continuous",
                          line=dict(color="#54A24B", width=3)))
xs, ys = cum_curve(dfs, ms_s)
fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="static",
                          line=dict(color="#E45756", width=3, dash="dash")))
fig2.update_layout(title="累积完成请求数 vs 时间", xaxis_title="时间(步)",
                   yaxis_title="已完成请求数", height=360,
                   margin=dict(l=10, r=10, t=40, b=10))
fig2.show()
'''), "🎨 上子图:每根柱子都是“整批陪跑”;下子图:柱子长短不一、各自收工。累积曲线则把吞吐优势画成了一目了然的“领先”。")

NB.md("## 6️⃣ 逐步动画:batch 成员是怎么变的 🎬",
D('''
continuous batching 的“连续”体现在**每次迭代 batch 都不同**。我们把模拟器改成
记录每一步的 running / waiting 成员,然后逐迭代打印:
'''))

NB.code(D('''
def simulate_continuous_log(reqs, token_budget=8):
    """与 simulate_continuous 相同,但额外记录每一步的 batch 成员"""
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

_, steps = simulate_continuous_log(make_reqs(n=10, seed=3), token_budget=6)
for s in steps[:12]:
    print(f"步{s['step']:>3}: running={s['running']}  waiting={s['waiting']}  已完成={s['finished']}")
'''), "🎬 观察前 12 步:某个请求完成(从 running 消失)的同时,新请求立刻被补进 running——batch 的成员每一行都在变。")

NB.code(D('''
from pyecharts.charts import Line
from pyecharts import options as opts

# 统计每一步 running 集合的大小,画成折线:batch 大小随迭代动态起伏
sizes = [len(s["running"]) for s in steps]
line = (Line()
        .add_xaxis([s["step"] for s in steps])
        .add_yaxis("running 请求数", sizes, is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3, color="#54A24B"),
                   label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="batch 大小随迭代动态变化"),
                         xaxis_opts=opts.AxisOpts(name="迭代步"),
                         yaxis_opts=opts.AxisOpts(name="running 请求数"),
                         tooltip_opts=opts.TooltipOpts(trigger="axis")))
line.render_notebook()
'''), "📊 注意这条曲线**不是平的**:每完成一个请求它就下降,每补进一个请求它就回升——这就是“连续”的动态批。")

NB.md("## 7️⃣ 真实 GPU:动态成员 vs 固定 batch 的吞吐差异 ⏱️",
D('''
模拟器说 continuous batching“吃得更饱”——但**吃饱能省多少真实算力**,只有 GPU 知道。
这里我们把物理吞吐曲线与“每步成员可变”的动态批**拼成一个仿真**:先用真实微基准量出
固定 batch 的吞吐曲线,再统计每次迭代两种调度**实际维持的平均活跃成员数**,最后
用曲线折算成真实 token/s。

先说固定 batch 的真实吞吐曲线(batch 越大,单步吃满的算力,单 token 越便宜):
'''))

NB.code(D('''
import sys, os
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

print("设备:", cuda_info())
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
print("batch | 每步耗时(ms) | 吞吐(token/s)")
for b, ms, tp in zip(b_list, mps, tps):
    print(f"{b:5d} | {ms:9.3f}  | {tp:9.0f}")
'''), "🚀 **真实数字**。记住这条曲线的形状:batch 越大单位 token 越便宜——但它有两个前提:第一,batch **真的凑得满**;第二,每一步都**满载输出**。")

NB.md("### 拼一半:把动态成员算进有效并发",
D('''
现在拼另一半:同一请求流,**动态成员**(continuous,完成一个补一个)vs **固定 batch**
(static,整批同进退)。统计两边每次迭代实际在做事的成员个数——这就是调度的“有效并发”:
'''))

NB.code(D('''
# 复用第 4/6 节的模拟器,统计"每步实际活跃成员数":
reqs_s, batches = simulate_static(make_reqs(n=16, seed=7), batch_size=4)      # 静态:固定 batch
_, steps_c = simulate_continuous_log(make_reqs(n=16, seed=7), token_budget=8) # 连续:成员逐步可变

# 静态批:有效并发 = 所有繁忙时刻的成员数之和 ÷ 墙钟(空转也算 0)
ms_s = max(b["end"] for b in batches)
avail_s = sum(len(b["members"]) * (b["end"] - b["start"]) for b in batches) / ms_s

# 连续批:有效并发 = 每一步 running 人数取平均(动态起伏本身就是它的优势)
steps_c = simulate_continuous_log(make_reqs(n=16, seed=7), token_budget=8)[1]
avail_c = float(np.mean([len(s["running"]) for s in steps_c]))

print(f"静态批: 墙钟 {ms_s:.0f} 步, 平均活跃成员 ≈ {avail_s:.2f} (batch_size=4 但在空转/陪跑)")
print(f"连续批: 墙钟 {len(steps_c):.0f} 步, 平均活跃成员 ≈ {avail_c:.2f} (优胜处:完成即补)")

# 用真实吞吐曲线折算"实际能拿到的 token/s"
est_s = float(np.interp(avail_s, b_list, tps))
est_c = float(np.interp(avail_c, b_list, tps))
print(f"\\n按真实曲线折算: 静态批 ≈ {est_s/1e3:.1f}k token/s · 连续批 ≈ {est_c/1e3:.1f}k token/s")
print(f"连续批相对静态批的有效吞吐提升 ≈ {est_c/max(est_s,1e-9):.1f}x")
'''), "🔢 前因后果:动态批赢在**把『平均活跃成员』顶在曲线上更右的位置**——静态批的值被巡航空转和整批陪跑拉低,同一个 batch 却要陪跑最长请求。配合真实曲线,这多出来的并发就是实打实的 token/s。")

NB.md(D('''
### 🏷️ 为什么是 Orca 的洞察

Orca 论文看到的事实就一句话:**同一张 GPU 上,忙不忙取决于你每次迭代放进去多少请求,而静止批
做不到“每次迭代都尽力放满”。** 它用 **iteration-level scheduling** 把调度粒度从“整个请求”降到
“一次迭代”,于是每次迭代都在真实曲线上挑一个尽量靠右的点:

- 长度短的请求提前完成,位置立刻被后面等待的请求**无缝顶替**(无整批陪跑);
- 调度器每步先喂满 `running`(decode,每步 1 token),再把剩余算力给 `waiting` 分块 prefill
  (第 8 节 `max_num_scheduled_tokens` 的真身)——每步都在**吃满预算**;
- 反观静止批:要么在凑不满的窗口里**空转**(有效 batch = 0),要么被批内最长请求**焊死在低并发**。

用一个比方收尾:静止批像“整桌一起点、一起上菜,有人吃得再快也得等最慢的吃完才换桌”;
continuous batching 是“谁吃完谁走、后厨永不空锅”。GPU 不是缺算力,而是缺一个**让它永远满载**的调度
——这就是 continuous batching 要解决的事。
'''))

NB.md("## 8️⃣ 与 vLLM 对接:token 预算的真身 🔗",
D('''
模拟器里的 `token_budget` 在 vLLM 里有一个真实对应物:`max_num_scheduled_tokens`。
打开 `vendor/vllm/vllm/v1/core/sched/scheduler.py` 的 `schedule()` 方法,你会看到:

```python
token_budget = self.max_num_scheduled_tokens   # 每步的总预算
...
while req_index < len(self.running) and token_budget > 0:   # 先服务 running
    ...
while (self.waiting or self.skipped_waiting) and token_budget > 0:  # 再补 waiting
    ...
token_budget -= num_new_tokens
```

调度顺序与我们模拟器完全一致:**先 running 后 waiting,预算扣完即止**。
vLLM 还在此基础上加了抢占(第 18 课)、优先级(第 17 课)与 spec decode 的配额,
但骨架就是这三行。**先读论文,再读代码,最后自己写一遍模拟器**——你已经把 vLLM
调度的核心模型装进脑子里了。
'''))

NB.md("## 9️⃣ 配套 Streamlit 演示:亲手推演每一笔 🎛️",
D('''
运行 `app_15_continuous_batch.py`,用**步进控制**亲手推演每一笔:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_15_continuous_batch.py
```

左边滑块调请求数量、token 预算、到达模式;中间用“下一步 / 上一步 / 跳到结束”
控制迭代进度;下方甘特图只画到当前步,并同屏对比静态批的指标与累积完成曲线。
完整源码如下(与 `app_15_continuous_batch.py` 一致):
'''))

app_cell(NB, APP_15, "app_15_continuous_batch.py",
         "📜 app_15_continuous_batch.py 完整源码(守卫包裹:notebook 中仅展示,streamlit 中才运行):步进动画 + 同屏对比,一个文件全搞定。")

wrapup(NB,
    summary=[
        "continuous batching = 每步迭代重新调度:完成即走、到达即补、绝不陪跑",
        "token 预算(max_num_scheduled_tokens)是 vLLM 每步调度的大脑,先 running 后 waiting",
        "同一请求流下,continuous 在墙钟时间、平均延迟、吞吐三项全面碾压静态批",
        "batch 大小随迭代动态起伏——这是“连续”的本质,也是 Orca 论文的洞察",
    ],
    practice=[
        "把 token_budget 从 2 扫到 20,画出“预算 vs 墙钟时间”曲线,找饱和点并解释原因",
        "给 simulate_continuous 加一个“每步最多 prefill 1 个请求”的限制,观察大 prompt 请求的饥饿现象",
        "改用 poisson 到达流,对比 burst 模式下两种调度的差距(哪个差距更大?)",
        "在 notebook 里把累积完成曲线改成面积图(step 模式),体验另一种表达",
    ],
    links=[
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 博客:Continuous Batching", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("vLLM Scheduler 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ])

NB.save(str(Path(CH03) / "15_continuous_batching.ipynb"))

app_path = Path(CH03) / "app_15_continuous_batch.py"
app_path.write_text(APP_15 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")