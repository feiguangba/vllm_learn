# -*- coding: utf-8 -*-
"""生成 17_scheduler_design.ipynb 与 app_17_scheduler_demo.py"""
from helpers import (D, chapter_cover, wrapup, new_nb, app_cell, CH03)
from pathlib import Path

APP_17 = D('''
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
''')

NB = new_nb("第 17 课 · Scheduler 设计:token 预算与调度策略",
            subtitle="调度的两个旋钮:每步能算多少 token(预算)、先服务谁(策略)——手写一个简化 Scheduler",
            emoji="🎛️")

chapter_cover(NB,
    objectives=[
        "理解 token 预算的本质:它是把“并发上限”从请求数换成算力的关键抽象",
        "手写一个简化 Scheduler 类,封装 预算分配 + 队列策略 两个旋钮",
        "对比 FCFS / SJF / Priority 三种策略的平均延迟、P95 与公平性",
        "在 vLLM scheduler.py 里找到预算与策略的真实落点",
    ],
    toc=[
        ("直觉:银行柜台与 VIP 通道", "叫号策略决定了“谁先办”,预算决定了“一天办多少”"),
        ("token 预算的本质", "max_num_scheduled_tokens:把并发从“个数”换成“算力”"),
        ("手写 Scheduler 类", "一次 schedule() = 一个迭代的完整决策,200 行内全搞定"),
        ("三种策略,同一请求流", "FCFS / SJF / Priority 各跑一遍,数字说话"),
        ("延迟分布:直方图 + 箱线图", "plotly 双图看分布形状,平均延迟会撒谎"),
        ("策略对比:pyecharts 柱状图", "平均延迟 vs P95,谁更稳、谁更快、谁更狠"),
        ("公平性:Priority 的饥饿问题", "短作业为什么能饿死长作业?用等待时间证明"),
        ("与 vLLM 对接", "vLLM 的策略优先级与预算细节"),
        ("配套 Streamlit 演示", "app_17_scheduler_demo.py:三策略实时对比"),
    ],
    links=[
        ("vLLM Scheduler schedule() 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("vLLM 官方博客:Continuous Batching", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Orca 论文(iteration-level scheduling)", "https://arxiv.org/abs/2208.14217"),
    ])

NB.md("## 1️⃣ 直觉:银行柜台与 VIP 通道 🏦",
D('''
取号机前的**叫号策略**决定了“谁先办”——这是调度策略;柜员一天能办多少笔,
取决于“营业时长 × 每笔耗时”——这是预算。两个旋钮拧在一起,就是调度器的全部。

回到 GPU 推理场景:

- 🎛️ **策略**:等待队列里有 20 个请求,GPU 一次只能服务一部分,先服务谁?
  - `FCFS`(先来先服务):排队的天然顺序,最公平、最无脑;
  - `SJF`(短作业优先):总耗时最短的先上,平均延迟最优;
  - `Priority`(优先级):priority 大的先上,但低优先级可能永远轮不到(饥饿)。
- 🪙 **预算**:每个迭代 GPU 能处理的 token 总数上限。

两者的关系很像水龙头:**策略**决定先放哪杯水,**预算**决定每秒钟放多少。
'''))

NB.md("## 2️⃣ token 预算的本质:从“个数”到“算力” 🪙",
D('''
第 15 课我们引入了 `token_budget`。为什么 vLLM 不用“每步最多 N 个请求”这种直观写法?
因为**请求之间太不一样了**:

- 一个 1000 token 的 prompt 做 prefill,一个迭代就要吃掉上千 token 的算力;
- 一个 decode 请求,每步只需要 1 个 token 的算力。

用“个数”做预算,一个 1000 token 的 prefill 就会把整个 GPU 卡住一个世纪;
用 **token 数**做预算,GPU 的每单位算力都物尽其用。vLLM 的真实参数
`max_num_scheduled_tokens` 就是这个思路,加上 `max_num_seqs`(请求数上限,防止
单个请求独占)与 `max_num_batched_tokens`(单次前向的最大 token 数,对应分块 prefill),
三件套一起约束每个迭代的工作量。

我们模拟器里只用最核心的 `token_budget`,其余两个作为练习留给你补全。
'''))

NB.md("## 3️⃣ 手写 Scheduler 类:一次 schedule() = 一个迭代 🛠️",
D('''
把第 15 课的函数式模拟器升级成**类**——这样更接近 vLLM 的真实结构
(`Scheduler.schedule()` 就是引擎每个迭代调用的入口)。类的成员:

- `waiting / running`:两个队列(真实 vLLM 里是 `Deque[Request]`);
- `token_budget / policy`:两个旋钮;
- `schedule()`:一步之内完成 排序 → 补位 → 分预算 → 结算 的完整决策;
- `log`:记录每一步的分发计划,供可视化。
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

def make_reqs(n=20, seed=11, plen=(5, 30), mnew=(5, 20)):
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.exponential(1.0 / 0.6)      # 泊松到达
        reqs.append(Req(i, t, int(rng.integers(plen[0], plen[1] + 1)),
                        int(rng.integers(mnew[0], mnew[1] + 1)),
                        priority=float(rng.random())))
    return reqs

def pick_next(waiting, policy):
    """策略:决定 waiting 队列的取人顺序"""
    if policy == "FCFS":
        return sorted(waiting, key=lambda r: r.arrive)
    if policy == "SJF":
        return sorted(waiting, key=lambda r: r.prompt_len + r.max_new)
    if policy == "Priority":
        return sorted(waiting, key=lambda r: -r.priority)
    raise ValueError(f"未知策略: {policy}")

class SimpleScheduler:
    """手写简化调度器:预算 + 策略,一次 schedule() = 一个迭代的完整决策"""
    def __init__(self, token_budget=8, policy="FCFS"):
        self.token_budget, self.policy = token_budget, policy
        self.waiting, self.running, self.finished = [], [], []
        self.t, self.done, self.total = 0.0, 0, 0
        self.log = []

    def admit(self, reqs):
        self.total = len(reqs)
        self.waiting.extend(reqs)

    def schedule(self):
        """一步调度:排序 → 补位 → 分预算 → 结算,返回这一步的 token 分发计划"""
        self.waiting = pick_next(self.waiting, self.policy)   # ① 策略排序
        for r in list(self.waiting):                          # ② 已到达请求补位
            if r.arrive <= self.t:
                self.running.append(r); self.waiting.remove(r)
        budget = self.token_budget
        plan = []
        for r in self.running:                                # ③ prefill:分块吃预算
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use; r.prefilled += use
                plan.append((r.rid, use, "prefill"))
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = self.t
        for r in self.running:                                # ④ decode:每人 1 token
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
                plan.append((r.rid, 1, "decode"))
        for r in list(self.running):                          # ⑤ 完成离场
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", self.t
                self.running.remove(r); self.finished.append(r); self.done += 1
        self.log.append(dict(step=self.t, plan=plan, running=[x.rid for x in self.running]))
        self.t += 1.0

    def run_all(self):
        while self.done < self.total:
            self.schedule()
        return self.waiting + self.running + self.finished, self.log
'''), "🛠️ 注意 `schedule()` 的五个步骤——它就是我们第 15 课模拟器的类化版本,而它正是 vLLM `Scheduler.schedule()` 的骨架。")

NB.md("## 4️⃣ 三种策略,同一请求流:数字说话 🔢",
D('''
同一个请求流(同一 seed),分别交给三种策略的 Scheduler,然后对比延迟指标。
注意:模拟器里“同一份请求”必须**每次重建**(请求对象会被模拟器改状态),
所以每次 `make_reqs(n, seed)` 生成三份一模一样的副本。
'''))

NB.code(D('''
def run_policy(policy, n=20, token_budget=8, seed=11):
    s = SimpleScheduler(token_budget, policy)
    s.admit(make_reqs(n, seed=seed))
    reqs, log = s.run_all()
    df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                            work=r.prompt_len + r.max_new, prio=round(r.priority, 2)) for r in reqs])
    df["latency"] = df["end"] - df["arrive"]
    df["wait"] = df["start"] - df["arrive"]
    return df, log

results = {}
for policy in ["FCFS", "SJF", "Priority"]:
    df, _ = run_policy(policy)
    results[policy] = df
    print(f"{policy:<9} 平均延迟={df.latency.mean():6.1f}  P95={np.percentile(df.latency, 95):6.1f}  "
          f"最大等待={df.wait.max():6.1f}  墙钟={df.end.max():5.0f}")
'''), "✅ 注意 SJF 的平均延迟通常最低,但 Priority 的分布可能非常不均——平均延迟会撒谎,下一节拆开看。")

NB.md("## 5️⃣ 延迟分布:平均延迟会撒谎 📊",
D('''
平均延迟只有 30 步,但可能一半请求 5 步就完成、另一半要等 80 步。
**分布的形状才反映真实体验**。画直方图 + 箱线图(plotly 双图):
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=("延迟直方图(重叠)", "延迟箱线图"))
colors = {"FCFS": "#E45756", "SJF": "#54A24B", "Priority": "#4C78A8"}
for policy, df in results.items():
    fig.add_trace(go.Histogram(x=df.latency, nbinsx=14, name=policy, opacity=0.55,
                               marker_color=colors[policy]), row=1, col=1)
    fig.add_trace(go.Box(y=df.latency, name=policy, marker_color=colors[policy],
                         boxmean=True), row=1, col=2)
fig.update_layout(title="三种策略的延迟分布对比", height=420, barmode="overlay",
                  margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "📊 看箱线图里的“胡须”和离群点:Priority 的平均延迟可能不高,但长尾更极端——某个低优先级请求可能被饿到天荒地老。")

NB.md("## 6️⃣ 策略对比:pyecharts 双指标柱状图 📊",
D('''
把三种策略的平均延迟与 P95 延迟并排画成柱状图,一眼看出“谁快、谁稳”:
'''))

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

policies = list(results)
avg = [round(float(results[p].latency.mean()), 1) for p in policies]
p95 = [round(float(np.percentile(results[p].latency, 95)), 1) for p in policies]
wait = [round(float(results[p].wait.max()), 1) for p in policies]

bar = (Bar()
       .add_xaxis(policies)
       .add_yaxis("平均延迟(步)", avg, color="#4C78A8")
       .add_yaxis("P95 延迟(步)", p95, color="#E45756")
       .add_yaxis("最大等待(步)", wait, color="#F2C14E")
       .set_global_opts(title_opts=opts.TitleOpts(title="FCFS vs SJF vs Priority"),
                        yaxis_opts=opts.AxisOpts(name="步数"),
                        legend_opts=opts.LegendOpts(pos_top="6%")))
bar.render_notebook()
'''), "🎛️ 三个策略三个性格:FCFS 稳定但平均偏慢,SJF 平均最优但可能饿死大请求,Priority 需要设计者自己保证低优先级不被饿死。")

NB.md("## 7️⃣ 公平性:Priority 的饥饿问题 🍽️",
D('''
把 Priority 策略下的请求按 `priority` 排序,看每个优先级区间的平均等待时间——
低优先级请求被高优先级不断插队,等待时间会爆炸:
'''))

NB.code(D('''
df_p = results["Priority"]
df_p["优先级档"] = pd.cut(df_p.prio, bins=[0, 0.33, 0.66, 1.0], labels=["低", "中", "高"])
print(df_p.groupby("优先级档", observed=True)[["wait"]].agg(["mean", "max"]).round(1).to_string())

from pyecharts.charts import Pie
pie = (Pie()
       .add("", [(f"{g}优先级", float(v)) for g, v in
                 df_p.groupby("优先级档", observed=True)["wait"].mean().items()],
            radius=["35%", "65%"])
       .set_global_opts(title_opts=opts.TitleOpts(title="不同优先级请求的总等待时间占比"))
       .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c:.0f} 步 ({d}%)")))
pie.render_notebook()
'''), "⚠️ 这张饼图如果被“低优先级”占掉一大半,说明策略在系统性饿死低优先级请求——真实系统里这叫 starvation,通常用**权重轮询(WRR)**或**配额**缓解。")

NB.md("## 8️⃣ 调度器为什么必须存在于 token 预算与抢占 🔗",
D('''
策略讲完,回到一个更根本的问题:**为什么调度器必须存在?** 答案是两个硬约束——
「算力不共享会饿死别人」和「显存装不下会把人挤下车」:

**① 为什么需要 token 预算?** 请求之间的「一次性成本」差异太大:
一个 1000 token 的 prompt 做 prefill 的一步,就要吃上千 token 的算力;一个 decode 请求每步只吃 1 token。
如果并发上限用「请求个数」来定(比如 max_num_seqs=4),那么 4 个 1000-token 的 prefill 就会把 GPU 卡住
一个世纪——因为单个请求就能独占整块预算。所以 vLLM 用 `max_num_scheduled_tokens`(每步 token 上限)
把「并发」从个数换算成算力:**每个请求按需申报 token 数,调度器照单排队、超预算的推下一步**。

**② 为什么需要抢占?** 算力管住了,还有显存。KV cache 是逐 token 增长的,prefill/decode 过程中
显存会被慢慢填满。一旦新请求的 KV 放不下,就必须把某个 running 请求「赶下车」(PREEMPTED,第 18 课讲细节)
来腾地方——否则新请求会直接 OOM。于是「谁能进 running、谁被挤出」成了一道必须实时裁决的题。

**③ token 预算与抢占如何“对话”?** 预算控制的是每个迭代的**算力供给,FCFS/SJF/Priority 控制的是 wait queue 的
取人顺序;显存上限(max_running)则是一道物理墙。三者拧在一起,就成了调度器的全部动机——
*每步都在回答:这一步把算力分给谁?新请求往哪放?显存不够先牺牲谁?* 下面把这个"为什么"落到数字上。
'''))

NB.code(D('''
# 预算 × 策略 双扫描:同一请求流,看"为什么预算必须存在"(预算太小大家抢、太大边际递减)
rows = []
for budget in [2, 4, 8, 16, 32]:
    for policy in ["FCFS", "SJF", "Priority"]:
        df, _ = run_policy(policy, token_budget=budget, seed=11)
        rows.append(dict(budget=budget, 策略=policy,
                         平均延迟=round(float(df.latency.mean()), 1),
                         P95=round(float(np.percentile(df.latency, 95)), 1),
                         平均等待=round(float(df.wait.mean()), 1),
                         墙钟=int(df.end.max())))
scan = pd.DataFrame(rows)
print(scan.round(1).to_string(index=False))
print("\\n观察:预算那么小时,连队都排不开(平均等待≈墙钟);预算很大时每个策略都收敛到相近延迟——"
      "说明瓶颈不再是预算而是算力,这正是第 19 课的调查地图。")
'''), "🔢 前因后果落到数字:预算小→ everyone 抢同一块算力(等待=墙钟);预算大→ 每步吃满但边际递减。调度策略的差别只在预算中等时最明显——这就是调参时要找的区间。")

NB.md("## 9️⃣ 真实 GPU:给 token 预算一个物理刻度 ⏱️",
D('''
上面的预算/策略都是「步」的单位。落到真实硬件,**一个 token 预算 = 真实的一次 decode
前向里多塞进来的一个 token 位**,它受 memory-bound 约束(见 ch01/05)。我们用真机测一下
「batch 增大 → 吞吐怎么变」,你就能理解为什么调度器要**尽量把预算喂满**而不是空着:
'''))

NB.code(D('''
import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

b_list, tps, mps = bench_throughput_curve(batch=(1, 4, 16, 64, 128), token_len=32, reps=5)
print("设备:", cuda_info())
print("batch(token 预算近似) | 每步耗时(ms) | 吞吐(tokens/s)")
for b, ms, tp in zip(b_list, mps, tps):
    print(f"        {int(b):>4}            |  {ms:7.3f}   |  {tp:10.0f}")
'''), "📡 同一段 GPU 时间片,批次越大吞吐越高——所以调度器的职责之一就是**别让预算空转**:每步尽量把 batch 装满(token 预算顶格用),这正是连续 batching 的物理收益。")

NB.md("## 1️⃣0️⃣ 与 vLLM 对接:真实调度器的旋钮 🔗",
D('''
打开 `vendor/vllm/vllm/v1/core/sched/scheduler.py`,你会看到:

- **预算旋钮**:`max_num_scheduled_tokens`、`max_num_seqs`、`max_num_batched_tokens`
  三个上限在 `schedule()` 开头统一成 `token_budget` 与 `input_budget`;
- **策略旋钮**:waiting 队列的顺序由 `policy.py` 中的 `RequestSchedulingPolicy`
  (`FCFS` / `Priority`)决定,默认 `FCFS`;`priority` 字段则来自请求自身的
  `priority(priority float, default=0)` 参数;
- **调度顺序**:`while req_index < len(self.running) and token_budget > 0` 先服务
  running 再服务 waiting——和我们 `schedule()` 的第 ③④ 步一致。

我们的模拟器与真机的差距只剩三点:真实的 KV 显存约束(第 18 课)、多队列细节
(`skipped_waiting` 等)以及 spec decode 配额。**骨架已经一模一样了。**
'''))

NB.md("## 1️⃣1️⃣ 配套 Streamlit 演示:三策略同屏 PK 🎛️",
D('''
运行 `app_17_scheduler_demo.py`:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_17_scheduler_demo.py
```

切换策略、拖动预算、换成“长尾分布”请求流,延迟直方图 / 箱线图 / 甘特图实时刷新,
并默认与 FCFS 基准同屏对比。完整源码如下(与 `app_17_scheduler_demo.py` 一致):
'''))

app_cell(NB, APP_17, "app_17_scheduler_demo.py",
         "📜 app_17_scheduler_demo.py 完整源码(守卫包裹):策略/预算/分布三旋钮 + 实时调度表。")

wrapup(NB,
    summary=[
        "调度器 = 预算旋钮(每步 token 上限)+ 策略旋钮(waiting 队列取人顺序)",
        "token 预算比“请求个数”更本质:1000 token 的 prefill 与 1 token 的 decode 不能同日而语",
        "FCFS 公平稳定、SJF 平均最优、Priority 灵活但会饿死低优先级",
        "平均延迟会撒谎:必须同时看 P95 与最大等待;vLLM 默认 FCFS,优先级可自定义",
    ],
    practice=[
        "给 SimpleScheduler 加 max_num_seqs 上限,复现“大 prompt 独占 GPU”的阻塞现象",
        "实现权重轮询(WRR)策略:每步按 priority 加权轮转,验证是否消除饥饿",
        "把 run_policy 改成扫描 token_budget ∈ [2, 4, 8, 16, 32],画“预算 vs 平均延迟”曲线并解释拐点",
        "阅读 vLLM policy.py 的 FCFSPolicy / PriorityPolicy 实现,说出与模拟器的两处差异",
    ],
    links=[
        ("vLLM 调度策略源码(policy.py)", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/policy.py"),
        ("vLLM Scheduler schedule()", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
    ])

NB.save(str(Path(CH03) / "17_scheduler_design.ipynb"))

app_path = Path(CH03) / "app_17_scheduler_demo.py"
app_path.write_text(APP_17 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")