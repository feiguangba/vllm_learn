# -*- coding: utf-8 -*-
"""生成 16_request_state_machine.ipynb 与 app_16_state_machine.py"""
from helpers import (D, chapter_cover, wrapup, new_nb, app_cell, CH03)
from pathlib import Path

APP_16 = D('''
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
''')

NB = new_nb("第 16 课 · 请求状态机:调度器眼中的快递包裹",
            subtitle="WAITING → RUNNING → PREEMPTED → FINISHED:一个请求从进来到出去,中间经历了什么",
            emoji="🚦")

chapter_cover(NB,
    objectives=[
        "掌握 vLLM 请求的四个核心状态:WAITING / RUNNING / PREEMPTED / FINISHED",
        "理解每条状态迁移边的触发事件(到达、调度、抢占、完成)",
        "手写带迁移日志的模拟器,把每个请求的一生录成事件流",
        "用 pyecharts 状态图 + matplotlib 手绘 + plotly 时间线三种方式画状态机",
    ],
    toc=[
        ("直觉:物流包裹的四格仓库", "从“已下单/运输中/已签收”理解状态的必要性"),
        ("四个状态与五条迁移边", "对照 vLLM 源码里的 RequestStatus 枚举"),
        ("状态迁移图:两种画法", "pyecharts Graph 交互图 + matplotlib 手绘示意图"),
        ("事件流模拟器:记录每一笔迁移", "手写 simulate_fsm,输出迁移日志"),
        ("状态时间线:plotly 分段甘特", "每个请求一生的颜色变化一目了然"),
        ("迁移计数与 Sankey 图", "谁去哪、去了多少次,用数字和流向图回答"),
        ("与 vLLM 真实现对照", "engine core 的调度循环如何驱动状态流转"),
        ("配套 Streamlit 演示", "app_16_state_machine.py:输入事件流,看迁移时间线"),
    ],
    links=[
        ("vLLM RequestStatus 枚举", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/enum.py"),
        ("vLLM 调度器源码(状态流转)", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.md("## 1️⃣ 直觉:物流包裹的四格仓库 📦",
D('''
你在淘宝下单一个包裹,物流系统里它的状态是明确的:`已下单 → 运输中 → 已签收`,
每一笔状态变化都有**时间戳和原因**("15:30 到达杭州转运中心")。

推理引擎的调度器也是一样。一个请求从进来到出去,不是一团乱麻,
而是**状态机**:每个请求在任何时刻都处于且仅处于一个状态,状态之间由
**事件**驱动迁移:

- 🧾 **WAITING(等待)**:请求已到达,但 GPU 还在忙,排队中;
- ⚙️ **RUNNING(运行)**:正在被调度器执行(prefill 或 decode);
- 🚫 **PREEMPTED(被抢占)**:显存不够,被新请求赶下车,KV 作废回队重排;
- ✅ **FINISHED(完成)**:生成完毕,交付结果,退出系统。

为什么要把状态搞得这么细?因为**调度器必须知道每个请求在哪一步**:
`running` 集合决定“下一步谁该算",`waiting` 集合决定”新请求能不能进",
被抢占的请求必须记录“重算到哪了”。没有状态机,调度器就是一团乱麻。
'''))

NB.md("## 2️⃣ 五个状态、五条迁移边(vLLM 视角) 🏷️",
D('''
翻开 `vendor/vllm/vllm/v1/engine/enum.py`,你能看到真实的状态枚举:

```python
class RequestStatus(enum.Enum):
    WAITING = "waiting"        # 在队列里排队
    RUNNING = "running"        # 正在执行
    PREEMPTED = "preempted"    # 被抢占,回到队列
    FINISHED_STOPPED = "finished_stopped"    # 正常结束
    FINISHED_LENGTH_CAPPED = "finished_length_capped"  # 到达长度上限
    FINISHED_ABORTED = "finished_aborted"    # 被用户中止
    FINISHED_IGNORED = "finished_ignored"    # 因系统过载被忽略
```

为了教学,我们把所有 FINISHED_* 合并成一个 `FINISHED`,得到五条迁移边:

| 迁移 | 触发事件 | 原因 |
|------|---------|------|
| WAITING → RUNNING | **调度** | 调度器把请求选入运行集(第 17 课讲选谁) |
| RUNNING → PREEMPTED | **抢占** | 显存不够,被新请求赶下车(第 18 课讲怎么抢) |
| PREEMPTED → WAITING | **回队重排** | KV 已作废,回到等待队列重新 prefill |
| PREEMPTED → RUNNING | **重新调度** | 队列里被再次选中,从头重算 |
| RUNNING → FINISHED | **完成** | 生成到 max_new,交付结果 |

> 💡 注意:PREEMPTED → RUNNING 是“重新调度”而不是“恢复”——vLLM 默认采用
> **recompute 策略**,被抢占的请求重新执行时,它的 KV cache 已经丢了,一切从头算。
'''))

NB.md("## 3️⃣ 状态迁移图:两种画法 🎨",
D('''
先画“教科书版”——matplotlib 手绘的状态图,节点是状态、箭头是事件:
'''))

NB.code(D('''
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9, 5.2))
pos = {"WAITING": (0.06, 0.55), "RUNNING": (0.5, 0.8), "PREEMPTED": (0.5, 0.2), "FINISHED": (0.92, 0.55)}
colors = {"WAITING": "#BAB0AC", "RUNNING": "#54A24B", "PREEMPTED": "#E45756", "FINISHED": "#4C78A8"}
for name, (x, y) in pos.items():
    box = mpatches.FancyBboxPatch((x - 0.11, y - 0.09), 0.22, 0.18, boxstyle="round,pad=0.01",
                                  fc=colors[name], ec="#333", lw=1.5)
    ax.add_patch(box)
    ax.text(x, y, name, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
edges = [("WAITING", "RUNNING", "调度"), ("RUNNING", "PREEMPTED", "抢占"),
         ("PREEMPTED", "WAITING", "回队重排"), ("PREEMPTED", "RUNNING", "重新调度"),
         ("RUNNING", "FINISHED", "完成")]
for a, b, label in edges:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.8,
                                connectionstyle="arc3,rad=0.18" if a != "WAITING" else "arc3,rad=-0.15"))
    ax.text((x1 + x2) / 2 + 0.02, (y1 + y2) / 2 + 0.05, label, fontsize=10, color="#222",
            ha="center", bbox=dict(fc="white", ec="#aaa", boxstyle="round,pad=0.15"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("请求状态机:五个状态、五条迁移边", fontsize=13)
plt.tight_layout()
plt.show()
'''), "🎨 这是“图解”系列的标准姿势:先手绘示意,再看代码,最后看真机。")

NB.code(D('''
from pyecharts.charts import Graph
from pyecharts import options as opts

nodes = [opts.GraphNode(name="WAITING", value=0, symbol_size=42,
                        itemstyle_opts=opts.ItemStyleOpts(color="#BAB0AC")),
         opts.GraphNode(name="RUNNING", value=0, symbol_size=46,
                        itemstyle_opts=opts.ItemStyleOpts(color="#54A24B")),
         opts.GraphNode(name="PREEMPTED", value=0, symbol_size=46,
                        itemstyle_opts=opts.ItemStyleOpts(color="#E45756")),
         opts.GraphNode(name="FINISHED", value=0, symbol_size=46,
                        itemstyle_opts=opts.ItemStyleOpts(color="#4C78A8"))]
links = [opts.GraphLink(source="WAITING", target="RUNNING", value="调度", linestyle_opts=opts.LineStyleOpts(width=3)),
         opts.GraphLink(source="RUNNING", target="PREEMPTED", value="抢占", linestyle_opts=opts.LineStyleOpts(width=3)),
         opts.GraphLink(source="PREEMPTED", target="WAITING", value="回队", linestyle_opts=opts.LineStyleOpts(width=2)),
         opts.GraphLink(source="PREEMPTED", target="RUNNING", value="重调度", linestyle_opts=opts.LineStyleOpts(width=2)),
         opts.GraphLink(source="RUNNING", target="FINISHED", value="完成", linestyle_opts=opts.LineStyleOpts(width=3))]
g = (Graph()
     .add("", nodes, links, repulsion=4000, layout="force", edge_label=opts.LabelOpts(is_show=True, formatter="{c}"))
     .set_global_opts(title_opts=opts.TitleOpts(title="状态迁移图(交互版:可拖动节点)")))
g.render_notebook()
'''), "🖱️ pyecharts 版可以拖动节点、悬停查看——适合放进交互文档;matplotlib 版适合打印。两版并存的习惯建议保留。")

NB.md("## 4️⃣ 事件流模拟器:记录每一笔迁移 🧾",
D('''
现在让模拟器开口说话:给 continuous batching 模拟器加一个**迁移日志**,
每次状态变化都记一笔 `(时间, 请求, 旧状态, 新状态, 原因)`。
这样我们就把“模拟运行”升级成了“事件流”,这是第 19 课完整模拟器的基础。
'''))

NB.code(D('''
# 轻量请求模型(只带状态机需要的字段)
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class Req:
    rid: int; arrive: float; prompt_len: int; max_new: int
    state: str = "WAITING"; start: float = None; end: float = None
    generated: int = 0; prefilled: int = 0

def make_reqs(n=12, seed=5, rate=0.5, plen=(5, 30), mnew=(5, 20)):
    """泊松到达流:请求不是同时涌到,而是随时间稀疏到达——避免太多请求在同一步争抢预算。"""
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.exponential(1.0 / rate)
        reqs.append(Req(i, t, int(rng.integers(plen[0], plen[1] + 1)),
                        int(rng.integers(mnew[0], mnew[1] + 1))))
    return reqs

def simulate_fsm(reqs, token_budget=8, max_running=None):
    """continuous batching + 状态迁移日志;max_running 设定显存并发上限(可抢占)。
    预算分配遵循 vLLM 的真实顺序:先给已 prefilled 的 RUNNING 请求 decode(每步 1 token),
    剩余预算才给 WAITING/PREEMPTED 做分块 prefill——被抢占的请求恢复后 token 不作废。
    """
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
        for r in running:                                  # ③ decode 优先(running 逐 token)
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in running:                                  # ④ 剩余预算分块 prefill
            if r.state in ("WAITING", "PREEMPTED") and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    mv(r, "RUNNING", "prefill 完成")
                    if r.start is None:
                        r.start = t
        for r in list(running):                            # ⑤ 完成
            if r.generated >= r.max_new:
                mv(r, "FINISHED", "生成完毕")
                r.end = t                                  # 记录完成时刻(供状态分布/甘特图用)
                running.remove(r); done += 1
        t += 1.0
    return reqs, trans

reqs, trans = simulate_fsm(make_reqs(seed=5), token_budget=8, max_running=4)
ev = pd.DataFrame(trans)
ev.columns = ["时间", "请求", "旧状态", "新状态", "原因"]
ev["请求"] = ev["请求"].map(lambda x: f"R{x}")
ev.head(14)
'''), "✅ 看这张表:每个请求的一生被拆成一笔笔带原因的事件——这就是“事件流”的字面意思。")

NB.md("## 5️⃣ 状态时间线:plotly 分段甘特图 📈",
D('''
迁移日志转成**分段甘特图**:每个请求一行,按时间切成 WAITING(灰)/ RUNNING(绿)/
PREEMPTED(红)/ FINISHED(蓝)的色段。红色段越多,说明抢占越频繁。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

def segments_from_trans(reqs, trans):
    """把迁移日志转换成 (rid, start, end, state) 分段"""
    evs = {r.rid: [dict(t=r.arrive, state="WAITING")] for r in reqs}
    for e in trans:
        evs[e["rid"]].append(dict(t=e["t"], state=e["to"]))
    segs = []
    for rid, ev_ in evs.items():
        ev_.sort(key=lambda x: x["t"])
        r = reqs[rid]
        ev_.append(dict(t=r.end, state="FINISHED"))
        for a, b in zip(ev_[:-1], ev_[1:]):
            if b["t"] > a["t"]:
                segs.append(dict(rid=f"R{rid}", start=a["t"], end=b["t"], state=a["state"]))
    return pd.DataFrame(segs)

cmap = {"WAITING": "#BAB0AC", "RUNNING": "#54A24B", "PREEMPTED": "#E45756", "FINISHED": "#4C78A8"}
segs = segments_from_trans(reqs, trans)
fig = go.Figure()
for _, row in segs.iterrows():
    fig.add_trace(go.Bar(x=[row.end - row.start], y=[row.rid], base=[row.start],
                         orientation="h", marker_color=cmap[row.state],
                         hovertemplate=f"{row.rid}: {row.state} %{{x|.0f}}~%{{x|.0f}}<extra></extra>",
                         showlegend=False, width=0.6))
for state, color in cmap.items():
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color=color, size=10), name=state))
fig.update_layout(title="请求状态时间线(抢占开启时,你能看到红色段落)",
                  xaxis_title="时间(步)", yaxis_title="请求",
                  height=60 + 30 * len(reqs), margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
fig.show()
'''), "🎨 把 `max_running` 改成 None 再跑一次,红线全部消失——抢占确实来自显存压力。")

NB.md("## 6️⃣ 迁移计数:谁去哪、去了多少次 🔁",
D('''
把迁移日志按 (旧状态 → 新状态) 分组计数,回答“这套请求流到底经历了什么”:

- WAITING → RUNNING 的次数 = **调度次数**(大于请求数,因为被抢占的请求会再次被调度);
- RUNNING → PREEMPTED 的次数 = **抢占次数**(显存压力的直接度量);
- RUNNING → FINISHED 的次数 = **完成请求数**。
'''))

NB.code(D('''
pairs = pd.DataFrame(trans).groupby(["frm", "to"]).size().reset_index(name="cnt")
print(pairs.to_string(index=False))

import plotly.graph_objects as go
all_states = ["WAITING", "RUNNING", "PREEMPTED", "FINISHED"]
idx = {s: i for i, s in enumerate(all_states)}
fig2 = go.Figure(go.Sankey(
    node=dict(label=all_states, color=[cmap[s] for s in all_states], pad=20, thickness=24),
    link=dict(source=[idx[r.frm] for r in pairs.itertuples()],
              target=[idx[r.to] for r in pairs.itertuples()],
              value=list(pairs.cnt))))
fig2.update_layout(title="状态迁移计数 Sankey 图", margin=dict(l=10, r=10, t=40, b=10), height=340)
fig2.show()
'''), "📊 Sankey 图把“有多少次 WAITING→RUNNING、多少次被抢占”画成了粗粗细细的流。")

NB.md("## 7️⃣ 同一时刻的状态分布:把系统看成流动的池子 🌊",
D('''
前面 5 节的甘特图是"每个请求的一生";换个横截面看——**站在某个时刻 t,系统里有多少个
请求分别处于 WAITING / RUNNING / FINISHED?** 这正是监控面板上最常看的图:它告诉你 GPU 队列排多长、
有多少请求正挤在 running 池里、已经交付了多少。

我们用第 4 节的模拟日志做横截面统计:对每个整数时刻 t,把每个请求的当前状态查出来、按桶计数。
'''))

NB.code(D('''
def state_distribution(reqs, trans, T_end):
    """按整数时刻统计 RUNNING / WAITING / FINISHED / PREEMPTED 的存活人数。
    每个请求:arrive→WAITING;trans 里每次迁移改状态;end→FINISHED。"""
    evs = {r.rid: [("WAITING", r.arrive)] for r in reqs}
    for e in trans:
        evs[e["rid"]].append((e["to"], e["t"]))
    for r in reqs:
        evs[r.rid].append(("FINISHED", r.end))
    rows = []
    for t in range(T_end + 1):
        cnt = {"RUNNING": 0, "WAITING": 0, "PREEMPTED": 0, "FINISHED": 0}
        for rid, ev in evs.items():
            cur = [s for s, tt in ev if tt <= t]
            cnt[cur[-1] if cur else "WAITING"] += 1
        rows.append(dict(t=t, **cnt))
    return pd.DataFrame(rows)

reqs, trans = simulate_fsm(make_reqs(seed=5), token_budget=8, max_running=4)
dist = state_distribution(reqs, trans, int(max(r.end for r in reqs)))
print(dist.head(15).to_string(index=False))
'''), "📋 这张表就是“同一时刻的状态截面”:某一步 running 有 3 个、waiting 有 4 个、finished 已有 5 个——系统里每一个请求都落在且只落在一个桶里。")

NB.md(D('''
把横截面连成一张 **堆叠面积图**,就能看到一组流动的水池:**waiting 是进水的池子,
running 是工作的池子,FINISHED 是积水的池子**——三条叠起来恒等于请求总数,一眼看出
系统在哪一段卡住、在哪一段繁忙:
'''))

NB.code(D('''
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 4.6))
order = ["WAITING", "PREEMPTED", "RUNNING", "FINISHED"]
colors = {"WAITING": "#BAB0AC", "PREEMPTED": "#E45756", "RUNNING": "#54A24B", "FINISHED": "#4C78A8"}
t = dist["t"]
ax.stackplot(t, [dist[s] for s in order],
             labels=order, colors=[colors[s] for s in order], alpha=0.85)
ax.set_xlabel("时间(步)"); ax.set_ylabel("处于该状态的请求数")
ax.set_title("同一时刻的请求状态分布(waiting / preempted / running / finished)")
ax.legend(loc="upper left", ncol=4)
ax.set_xlim(0, dist["t"].max())
plt.tight_layout(); plt.show()

# "一段简单计时":每一步快照只打印当时占比最高的两个桶
for _, row in dist.head(12).iterrows():
    top = row[["RUNNING", "WAITING", "FINISHED"]].sort_values(ascending=False)
    print(f"步{int(row['t']):>3}: running={int(row.RUNNING)} waiting={int(row.WAITING)} "
          f"finished={int(row.FINISHED)}  → 最忙足是 {top.index[0]}")
'''), "🎨 实线堆叠面积图中三条泳道永远相加等于请求总数。如果 waiting 长期居高不下,说明 GPU 是瓶颈(喂不饱);如果 running 总是很满,说明并发上限(max_running)在起作用——这张图把调度瓶颈直接顶出来。")

NB.md("## 8️⃣ 真实 GPU:给状态机的“每步”一个时间刻度 ⏱️",
D('''
上面的状态机用「步」作时间单位,很抽象。真实世界里每一步(一次调度迭代)**不是免费的**:
它至少包含一次 decode(每 running 请求 1 token,memory-bound)或一次 prefill(prefill 请求,
compute-bound)。我们在真机上给这两类“成本”标个刻度,好把抽象的“步”换算成毫秒:
'''))

NB.code(D('''
import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_prefill_decode, cuda_info

b = bench_prefill_decode(d=256, layers=8, L=256, steps=64, reps=7)
print("设备:", cuda_info())
print(f"一次 prefill(并行 {b['L']} token)      : {b['prefill_ms']:.3f} ms")
print(f"一次 decode 单步(1 token,memory-bound): {b['decode_step_ms']:.3f} ms")
print(f"跑满 1000 步(仅 decode)              : {b['decode_step_ms']:.0f} × 1000 ≈ {b['decode_step_ms']:.0f} ms")
print("→ 状态机里的每 1 个 RUNNING 时间片,在真实硬件上都要花这么多真实毫秒。")
'''), "📡 把「步」换成毫秒后,第 4 节甘特图里横轴的长度就有了真实的物理意义——mils 级 latency 就是这么来的。")

NB.md("## 9️⃣ 与 vLLM 真实现对照:状态由谁驱动 🔗",
D('''
在 `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 里,状态流转是这么发生的:

```python
# 第 1135 行附近:被调度的请求置为 RUNNING
request.status = RequestStatus.RUNNING
# 第 1355 行附近:抢占时置为 PREEMPTED
request.status = RequestStatus.PREEMPTED
# 第 1879 行附近:完成时置为 FINISHED_STOPPED
request.status = RequestStatus.FINISHED_STOPPED
```

注意细节:vLLM 把 PREEMPTED 的请求**留在等待队列里**(`waiting` 队列),
与我们的 `waiting.insert(0, victim)` 行为一致;同时给请求挂上 `num_preemptions`
计数器,把“被抢了几次”变成可观测的指标。**状态机不是纸面模型,它就是调度器
代码本身的分支结构**——把这张图印在脑子里,读调度器源码就像读注释一样轻松。
'''))

NB.md("## 🔟 配套 Streamlit 演示:自己制造事件流 🎛️",
D('''
运行 `app_16_state_machine.py`:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_16_state_machine.py
```

调节“显存并发上限”滑块就是在**改变事件流**:上限越小,抢占事件越多,
红色段越长。右侧的迁移日志表 + Sankey 图实时刷新,把每次状态变化摊开给你看。
完整源码如下(与 `app_16_state_machine.py` 一致):
'''))

app_cell(NB, APP_16, "app_16_state_machine.py",
         "📜 app_16_state_machine.py 完整源码(守卫包裹):事件流输入 + 状态时间线 + 迁移 Sankey,一屏看全。")

wrapup(NB,
    summary=[
        "请求状态机四个状态:WAITING / RUNNING / PREEMPTED / FINISHED,五条迁移边",
        "每个迁移都有触发事件:调度、抢占、回队、重调度、完成",
        "事件流 = 带时间戳和原因的迁移日志,是理解调度的第一手资料",
        "vLLM 源码里状态流转就是调度器分支逻辑的镜子:schedule() 里改状态、计数、排队",
    ],
    practice=[
        "把 max_running 从 1 扫到 8,统计 PREEMPTED→RUNNING 次数,画出“上限 vs 抢占次数”曲线",
        "给 Req 加一个 deadline 字段,在模拟器里把超时请求置为 ABORTED(新增一个状态!),并画迁移图",
        "用 pyecharts Graph 给边加上线宽=迁移次数,做成带权状态图",
        "阅读 scheduler.py 中 finish_requests 的实现,数一数 FINISHED 有几种子状态",
    ],
    links=[
        ("vLLM RequestStatus 枚举", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/enum.py"),
        ("vLLM Scheduler 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH03) / "16_request_state_machine.ipynb"))

app_path = Path(CH03) / "app_16_state_machine.py"
app_path.write_text(APP_16 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")