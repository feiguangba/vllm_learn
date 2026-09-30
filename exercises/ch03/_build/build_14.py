# -*- coding: utf-8 -*-
"""生成 14_static_batching_problem.ipynb 与 app_14_static_batch.py"""
from helpers import (D, SIM_BASE, SIM_STATIC, SIM_STATS, GANTT_PLOTLY,
                     chapter_cover, wrapup, new_nb, app_cell, CH03)
from pathlib import Path

APP_14 = D('''
# -*- coding: utf-8 -*-
# app_14_static_batch.py — 静态批处理缺陷交互演示 🐌
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="静态批处理缺陷 🐌", layout="wide")
st.title("🐌 第 14 课 · 静态批处理的缺陷:队头阻塞与尾部浪费")

st.markdown("""
静态批处理 = **凑满一车才发车**。GPU 一次只服务一个固定大小的批次,批内所有请求
同时开始、同时结束(以最慢的为准),批与批之间还可能空转。
下方可自由调节请求流与批次大小,用**交互甘特图**观察两大缺陷:
- 🚦 **队头阻塞**:批内短请求被长请求拖住,提前完成也只能干等;
- 🗑️ **尾部浪费**:最后一批凑不满,GPU 座位空着。
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
        batch_time = max(r.prompt_len + r.max_new for r in batch)
        for r in batch:
            r.end, r.state = t + batch_time, "FINISHED"
        batches.append(dict(start=t, end=t + batch_time, members=[r.rid for r in batch]))
        t += batch_time
    return reqs, batches

# ---------------------------------------------------------------- 侧边栏参数
with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("请求数量", 8, 40, 16, 1)
    batch_size = st.slider("批次大小 batch_size", 1, 10, 4, 1)
    mode = st.radio("到达模式", ["burst(同时到达)", "poisson(泊松流)"])
    rate = st.slider("到达率 λ(请求/步)", 0.1, 1.5, 0.5, 0.1) if mode == "poisson(泊松流)" else None
    seed = st.slider("随机种子", 0, 99, 42, 1)
    st.caption("模拟单位:1 步 = 1 次迭代;prompt 与生成均按 token 计")

# ---------------------------------------------------------------- 运行模拟
reqs, batches = simulate_static(make_reqs(n, mode, rate, seed), batch_size)
df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                        work=r.prompt_len + r.max_new) for r in reqs])
df["latency"] = df["end"] - df["arrive"]
df["wait"] = df["start"] - df["arrive"]
makespan = float(df["end"].max())
util = float(df["work"].sum() / makespan)
idle = makespan - sum(b["end"] - b["start"] for b in batches)
tail = batches[-1]["members"] if batches else []
tail_waste = 1.0 - len(tail) / batch_size

c1, c2, c3, c4 = st.columns(4)
c1.metric("平均完成时间(步)", f"{df['latency'].mean():.1f}")
c2.metric("GPU 利用率", f"{util * 100:.1f}%")
c3.metric("GPU 空转(步)", f"{idle:.0f}")
c4.metric("最后一批不满比例", f"{tail_waste * 100:.0f}%")
st.caption(f"共 {len(batches)} 批;最后一批 {len(tail)} 个请求(设计容量 {batch_size})")

# ---------------------------------------------------------------- 甘特图
rows = []
for i, b in enumerate(batches):
    for rid in b["members"]:
        r = df[df.rid == rid].iloc[0]
        rows.append(dict(rid=f"Req {rid}", start=r.arrive, end=r.start, kind="等待", batch=f"批 {i + 1}"))
        rows.append(dict(rid=f"Req {rid}", start=r.start, end=r.end, kind=f"批 {i + 1}", batch=f"批 {i + 1}"))
gdf = pd.DataFrame(rows)
palette = ["#4C78A8", "#72B7B2", "#E45756", "#F2C14E", "#54A24B", "#B279A2",
           "#FF9DA6", "#9D755D", "#BAB0AC", "#605B8C"]
cmap = {"等待": "#E0E0E0"}
for i, b in enumerate(batches):
    cmap[f"批 {i + 1}"] = palette[i % len(palette)]

fig = go.Figure()
for _, row in gdf.iterrows():
    fig.add_trace(go.Bar(
        x=[row.end - row.start], y=[row.rid], base=[row.start], orientation="h",
        marker_color=cmap[row.kind], text=row.kind if row.kind != "等待" else "",
        hoverinfo="x+y", showlegend=False, width=0.6))
fig.update_layout(title="静态批处理甘特图:灰色=排队等待,彩色=该批执行(同批同色)",
                  xaxis_title="时间(步)", yaxis_title="请求",
                  height=60 + 26 * len(gdf), margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 观察:批内颜色相同的请求**同时开始、同时结束**;短请求被迫等待长请求 = 队头阻塞;"
           "灰色等待段很长 = 凑不满批时 GPU 在空转。")

# ---------------------------------------------------------------- 每批统计表
batch_rows = []
for i, b in enumerate(batches):
    m = df[df.rid.isin(b["members"])]
    batch_rows.append(dict(批次=i + 1, 成员数=len(b["members"]),
                           批内最长耗时=int(b["end"] - b["start"]),
                           平均单请求耗时=round(float(m.work.mean()), 1),
                           队头阻塞浪费=round(float(b["end"] - b["start"] - m.work.mean()), 1),
                           开始步=int(b["start"])))
st.subheader("📋 每批明细:浪费一目了然")
st.dataframe(pd.DataFrame(batch_rows), use_container_width=True)
st.markdown("""
> 💡 **结论**:批内“最长耗时 − 平均耗时”就是队头阻塞的浪费;批次越大浪费越重,
> 但批次太小又会让 GPU 频繁空转。静态批的这两个矛盾,正是下一课 continuous
> batching 要解决的问题。
""")
''')

NB = new_nb("第 14 课 · 静态批处理的缺陷:队头阻塞与尾部浪费",
            subtitle="为什么“凑满一车才发车”会让 GPU 又慢又闲?—— 从餐馆翻台到大巴发车的直觉出发,手写一个静态批处理模拟器",
            emoji="🐌")

chapter_cover(NB,
    objectives=[
        "用生活比喻理解静态批处理的两大缺陷:🚦 队头阻塞 与 🗑️ 尾部浪费",
        "手写一个 40 行以内的静态批处理模拟器,输出甘特图与统计指标",
        "用 pyecharts + plotly 双图量化“批次大小 × 延迟 × 利用率”三者关系",
        "为第 15 课 continuous batching 做好对比基准",
    ],
    toc=[
        ("直觉:固定座位的大巴", "凑满一车才发车,车上有人快有人慢,谁都得等最慢的"),
        ("抽象:静态批处理的三个规则", "满批发车 / 整批同进退 / 空转等待,全部翻译成代码"),
        ("模拟器:20 行规则,50 行代码", "make_reqs 造请求流,simulate_static 模拟调度"),
        ("量化两大缺陷", "队头阻塞与尾部浪费到底浪费了多少时间?用数字说话"),
        ("甘特图:一眼看穿问题", "plotly 交互甘特图,灰色排队段 vs 彩色执行段"),
        ("参数扫描:批次大小的影响", "pyecharts 柱状图展示 batch_size 扫描结果"),
        ("数学视角:三个公式", "把浪费写成公式,推导静态批的最优批次"),
        ("配套 Streamlit 演示", "app_14_static_batch.py:交互甘特图 + 实时指标"),
    ],
    links=[
        ("Orca 论文(continuous batching 出处)", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 官方博客《Continuous Batching》", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("vLLM 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ])

NB.md("## 1️⃣ 直觉:固定座位的大巴 🚌",
D('''
想象你在景区门口等**景区大巴**:每辆车有固定的座位数(比如 4 个),**凑满一车才发车**,
车上的人一起出发、一起到站——哪怕有人只想逛 10 分钟,也得陪着要逛 3 小时的游客坐完全程,
到站后大巴才能回去接下一批人。

把“景区大巴”换成 GPU、把“游客”换成推理请求,这就是**静态批处理(static batching)**:

- 🚌 **座位 = batch_size**:GPU 一次并行处理多少个请求;
- ⏳ **凑满才发车 = 满批发车**:请求先到先排队,不满一批不开算;
- 🐢 **一起到站 = 整批同进退**:批内耗时取**最长**请求(木桶原理);
- 🕳️ **回站等人 = GPU 空转**:发车间隙和最后凑不满的一批,计算资源全闲着。

这套规则在深度学习框架的早期(以及 `torch.no_grad` 时代很多朴素推理服务)非常常见:
把请求攒成固定大小的 batch,一次 `forward` 算一批。它简单、整齐、好实现,
但有两个致命缺陷,正是本课要量化的东西。
'''))

NB.code(D('''
# 1) 制造一批“游客”:每个请求有到达时间、提示词长度、最大生成长度
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class Req:
    rid: int                 # 请求编号
    arrive: float            # 到达时间(单位:步)
    prompt_len: int          # prefill 阶段要处理的 token 数
    max_new: int             # decode 阶段最多生成的 token 数
    state: str = "WAITING"   # WAITING / RUNNING / FINISHED
    start: float = None      # 首次开始执行的时间
    end: float = None        # 完成时间

def make_reqs(n=10, mode="burst", rate=0.5, seed=42, plen=(5, 30), mnew=(5, 20)):
    """制造请求流:mode="burst" 同时到达;mode="poisson" 按泊松过程到达"""
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        if mode == "poisson":
            t += rng.exponential(1.0 / rate)
        a = 0.0 if mode == "burst" else t
        reqs.append(Req(i, a, int(rng.integers(plen[0], plen[1] + 1)),
                        int(rng.integers(mnew[0], mnew[1] + 1))))
    return reqs

reqs = make_reqs(n=8, seed=42)
for r in reqs:
    print(f"Req{r.rid}: 到达={r.arrive:.0f} 提示词={r.prompt_len} 最大生成={r.max_new}")
'''), "**模拟单位说明**:1 步 = 1 次迭代。prefill 与 decode 都按 token 计——这是为了和第 15 课的 token 预算接轨。")

NB.md("## 2️⃣ 模拟器:把大巴规则翻译成代码 🐢",
D('''
规则只有三条,代码也只有三块:

1. **满批发车**:从队列里按到达顺序捞人,捞满 `batch_size` 个就发车;
2. **整批同进退**:本批耗时 = 批内所有请求耗时(prompt_len + max_new)的**最大值**;
3. **空转等待**:一个能发车的都没有,时间照走、GPU 白等。

下面这个 `simulate_static` 就是完整的大巴调度器,顺便把每批的起止时间记录成 `batches`,
供画甘特图用:
'''))

NB.code(D('''
def simulate_static(reqs, batch_size):
    """静态批处理模拟器:GPU 一次服务一批,批内全部完成才换下一批"""
    queue = sorted(reqs, key=lambda r: r.arrive)   # 待发车队列
    t, batches = 0.0, []
    while queue:
        batch = []
        for r in list(queue):
            if r.arrive <= t and len(batch) < batch_size:
                batch.append(r)
                queue.remove(r)
        if not batch:              # 一个能发车的都没有 → GPU 空转等待
            t += 1.0
            continue
        for r in batch:            # 发车:整批同时开始
            r.state, r.start = "RUNNING", t
        batch_time = max(r.prompt_len + r.max_new for r in batch)  # 木桶原理
        for r in batch:
            r.end, r.state = t + batch_time, "FINISHED"
        batches.append(dict(start=t, end=t + batch_time, members=[r.rid for r in batch]))
        t += batch_time
    return reqs, batches

reqs, batches = simulate_static(make_reqs(n=8, seed=42), batch_size=3)
for i, b in enumerate(batches, 1):
    print(f"批{i}: 步 {b['start']:.0f} → {b['end']:.0f},成员 {b['members']},耗时 {b['end']-b['start']:.0f}")
'''), "🚀 跑完上面这段,你就能看到“批1 等到了步 0 才发车、批2 里谁在拖后腿”这类信息。")

NB.md("## 3️⃣ 量化两大缺陷:让数字自己说话 📏",
D('''
大巴开完了,现在算账。我们关心三个指标:

- **平均完成时间**(平均 latency)= 每个请求 `end − arrive` 的平均值;
- **GPU 利用率** = 有效计算量(所有请求的 token 总数)÷ 墙钟时间(makespan);
- **GPU 空转步数** = makespan − 各批实际耗时之和。

**队头阻塞**体现在批内:短请求明明 10 步能干完,却要等批里最长的 30 步——
白白多等 20 步。**尾部浪费**体现在批尾:最后一批只有 2 个人,大巴 4 个座位空了 2 个。
'''))

NB.code(D('''
def reqs_df(reqs):
    """把请求对象汇总成 DataFrame,附带延迟指标"""
    df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                            work=r.prompt_len + r.max_new) for r in reqs])
    df["latency"] = df["end"] - df["arrive"]
    df["wait"] = df["start"] - df["arrive"]
    return df

def throughput(df):
    makespan = float(df["end"].max())
    return dict(req_per_step=float(len(df) / makespan),
                token_per_step=float(df["work"].sum() / makespan),
                makespan=makespan, avg_latency=float(df["latency"].mean()))

reqs, batches = simulate_static(make_reqs(n=16, seed=7), batch_size=4)
df = reqs_df(reqs)
tput = throughput(df)
idle = tput["makespan"] - sum(b["end"] - b["start"] for b in batches)
print(f"全部完成耗时(makespan)    = {tput['makespan']:.0f} 步")
print(f"平均完成时间               = {tput['avg_latency']:.1f} 步")
print(f"GPU 利用率                 = {df['work'].sum() / tput['makespan'] * 100:.1f}%")
print(f"GPU 空转                   = {idle:.0f} 步({idle / tput['makespan'] * 100:.0f}%)")
print(f"队头阻塞示例:同一批内,最短请求最多要多等 "
      f"{max((df[df.rid.isin(b['members'])].work.max() - df[df.rid.isin(b['members'])].work.min()) for b in batches):.0f} 步")
print(f"尾部浪费:最后一批 {len(batches[-1]['members'])} 人,容量 4,空座率 "
      f"{(4 - len(batches[-1]['members'])) / 4 * 100:.0f}%")
'''), "✅ 跑出来的数字就是证据:同一批请求,静态批处理把大量时间花在了“等人”上。")

NB.md("## 4️⃣ 甘特图:一眼看穿问题 👀",
D('''
数字有了,但“形象”还不够。我们画一张 **plotly 交互甘特图**:横轴是时间,
每个请求一行,灰色段 = 排队等待,彩色段 = 执行(同一批同一种颜色)。

你会看到两件非常直观的事:

1. **队头阻塞**:批内彩色段**同时开始、同时结束**,长短不一的请求被强行对齐成同一个长度;
2. **尾部浪费**:最后一行的灰色等待段特别长——凑不满一批,GPU 只能干等。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

def gantt_plotly(df, batches, title="静态批处理甘特图"):
    rows = []
    palette = ["#4C78A8", "#72B7B2", "#E45756", "#F2C14E", "#54A24B",
               "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#605B8C"]
    cmap = {"等待": "#E0E0E0"}
    for i, b in enumerate(batches):
        cmap[f"批{i + 1}"] = palette[i % len(palette)]
        for rid in b["members"]:
            r = df[df.rid == rid].iloc[0]
            rows.append(dict(rid=f"Req {rid}", start=r.arrive, end=r.start, kind="等待"))
            rows.append(dict(rid=f"Req {rid}", start=r.start, end=r.end, kind=f"批{i + 1}"))
    fig = go.Figure()
    for _, row in pd.DataFrame(rows).iterrows():
        fig.add_trace(go.Bar(x=[row.end - row.start], y=[row.rid], base=[row.start],
                             orientation="h", marker_color=cmap[row.kind],
                             showlegend=False, width=0.6))
    fig.update_layout(title=title, xaxis_title="时间(步)", yaxis_title="请求",
                      height=60 + 26 * len(pd.DataFrame(rows)),
                      margin=dict(l=10, r=10, t=40, b=10), bargap=0.2)
    return fig

reqs, batches = simulate_static(make_reqs(n=16, seed=7), batch_size=4)
gantt_plotly(reqs_df(reqs), batches).show()
'''), "🎨 把鼠标悬停在柱子上可以看到起止时间;灰色等待段越长,浪费越明显。")

NB.md("## 5️⃣ 参数扫描:批次大小是把双刃剑 ⚔️",
D('''
直觉告诉我们:批次太小 → 批次多、发车间隙多 → GPU 空转;批次太大 → 批内“木桶”越长 → 队头阻塞更重。
到底多大合适?扫一遍 `batch_size ∈ [2, 4, 6, 8, 10]`,画一张 **pyecharts 双柱状图**:
蓝色 = 平均完成时间,橙色 = GPU 利用率。
'''))

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

sizes = [2, 4, 6, 8, 10]
avg_lat, utils = [], []
for bs in sizes:
    r, b = simulate_static(make_reqs(n=30, seed=7), bs)
    d = reqs_df(r)
    avg_lat.append(round(float(d["latency"].mean()), 1))
    utils.append(round(float(d["work"].sum() / d["end"].max() * 100), 1))

bar = (Bar()
       .add_xaxis([f"batch={s}" for s in sizes])
       .add_yaxis("平均完成时间(步)", avg_lat, color="#4C78A8")
       .add_yaxis("GPU 利用率(%)", utils, color="#F2C14E")
       .set_global_opts(title_opts=opts.TitleOpts(title="批次大小的双刃剑"),
                        yaxis_opts=opts.AxisOpts(name="数值"),
                        legend_opts=opts.LegendOpts(pos_top="8%")))
bar.render_notebook()
'''), "📊 注意观察:利用率随 batch 增大先升后降,平均延迟几乎单调上升——两个目标互相打架。")

NB.md("## 6️⃣ 数学视角:三个公式定乾坤 🧮",
D('''
把观察写成公式。设第 $i$ 批有 $B$ 个请求,第 $i$ 批的耗时为:

$$T_i^{batch} = \\max_{r \\in batch_i} (\\text{prompt}_r + \\text{max\\_new}_r)$$

**队头阻塞浪费**(批内):

$$W_i^{head} = T_i^{batch} - \\frac{1}{B}\\sum_{r \\in batch_i} (\\text{prompt}_r + \\text{max\\_new}_r)$$

**尾部浪费**:设共有 $K$ 批,最后一批 $B_{last} < B$,则 GPU 实际利用的座位比例:

$$\\eta_{tail} = \\frac{B_{last}}{B}$$

**总墙钟时间** = 空转时间 + 各批耗时之和:

$$T_{total} = T_{idle} + \\sum_{i=1}^{K} T_i^{batch}$$

静态批处理的本质矛盾:批次 $B$ 越大,$T_i^{batch}$ 越大(木桶越长)、队头阻塞越重;
批次 $B$ 越小,批数 $K$ 越多、$T_{idle}$ 越大(凑不满的车次变多)。
**没有任何一个 $B$ 能同时赢下两个目标**——这就是为什么我们需要第 15 课的 continuous batching:
让“发车”这个动作从“凑满才发”变成“随时可发”。
'''))

NB.md("## 7️⃣ 真实 GPU:静止 batch 在长尾请求下的浪费 ⏱️",
D('''
前面 1~6 节全程在“步”这个抽象时间单位里谈浪费。现在把**真实算力**搬进来,回答一个
更硬的问题:**静止 batch 在长尾请求下到底浪费了多少真实的 token/s?**

两个观察必须分开:同一时刻 GPU **吃满**能有多快(真实吞吐曲线),和静止 batch
**吃不满**时实际有多慢(调度空转)。差距就是浪费。

首先,在真实 GPU 上用 `bench_throughput_curve` 测一批“固定 batch”的吞吐——
它告诉我们 **batch 越大,单步吃下的算力越满、每 token 越便宜**(这正是第 15 课
continuous batching 把 batch 拼大动机的来源):
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

print("设备:", cuda_info())
# 固定 batch:每步把 batch 个请求各算 1 个 token(连续批的雏形),测真实吞吐
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
print("batch | 每步耗时(ms)   | 吃满时的吞吐(token/s)")
for b, ms, tp in zip(b_list, mps, tps):
    print(f"{b:5d} | {ms:9.3f}   | {tp:10.0f}")
'''), "🚀 **真实数字**。注意 batch 从 1 到 32,吞吐并不是线性翻倍而是**非线性上涨**——GPU 的固定开销(launch、搬运权重)被更多 token 摊薄。这是 continuous batching 想赚的第一笔钱。")

NB.md("### 吃不满:真实的浪费发生在哪",
D('''
上面“吃满时能多快”给的是**物理上限**;现在看静止 batch **实际**在怎么走。同样一批长尾请求,
喂给静止 batch(凑满 8 个才开工),把调度空转和“吃满曲线”放到一起看:
'''))

NB.code(D('''
# 造一个"长尾到达"的请求流:前 16 个在 t=0 同时涌到,后 6 个拖到很晚才零星到达。
arrives = [0.0]*16 + [30, 40, 46, 52, 57, 60]
reqs = make_reqs(n=len(arrives), seed=42)
for r, a in zip(reqs, arrives):
    r.arrive = a                      # 覆盖到达时间,模拟真实流量脉冲+长尾

# 静止 batch:batch_size=8,凑不满就不开工(复用第 2 节模拟器)
reqs_s, batches = simulate_static(reqs, batch_size=8)
df = reqs_df(reqs_s)
makespan = float(df["end"].max())
idle = makespan - sum(b["end"]-b["start"] for b in batches)
util = float(df["work"].sum() / makespan)
total_tok = int(df["work"].sum())

print(f"墙钟 makespan          = {makespan:.0f} 步")
print(f"其中 GPU 空转          = {idle:.0f} 步 ({idle/makespan*100:.0f}%)")
print(f"有效算力利用率         = {util*100:.0f}%  (真正用来算 token 的时间占比)")
print(f"总计产生              = {total_tok} token")
# 真实"参考墙":把 GPU 吃满(取上述曲线 batch=16 的吞吐)能拿到多少
real_tps = dict(zip(b_list, tps))[16]
ideal_ms = total_tok / real_tps * 1e3
actual_ms = makespan * 9.0    # 模拟器 1 步 ≈ 真实单步耗时(ms,基线量级)
print(f"\\n• 若 GPU 全程吃满(batch=16 实测 {real_tps/1e3:.0f}k token/s):  "
      f"{total_tok} token 只需 ≈ {ideal_ms:.0f} ms")
print(f"• 静止 batch 实际墙钟(按每步9ms折算)      : ≈ {actual_ms:.0f} ms")
print(f"• 浪费占比 ≈ {(1 - ideal_ms/actual_ms)*100:.0f}%  —— 长尾期 GPU 在 空等凑批 + 短请求陪跑")
'''), "🔢 前因后果:浪费 = (吃满的速度 − 实际的速度) ÷ 吃满的速度。模拟器的空转步、队头陪跑,都被真实吞吐曲线放大成了毫秒级的‘白烧电’。")

NB.md("### 🤔 为什么 naive batching 必然浪费?一段前因后果\n\n"
"naive batching(朴素的请求级批处理,深框架时代最常见)只有**两个动作**:攒满一批、整批跑完。\n"
"这两个动作对 GPU 的真实吞吐曲线极不友好:\n\n"
"- **吃满才有边际收益**:吞吐曲线告诉我们,batch 越大单 token 越便宜。可 naive batching 为了"
"‘凑满’一个 8 的 batch,在长尾时段要**干等很久**——虚线(吃满)越来越陡,batch 却没动;\n"
"- **action 越短越吃亏**:每一小批都要重新发车(kernel launch + 权重搬运),batch 小的时候这批"
"固定开销占比极高,正是上面曲线低 batch 端那个‘贵’的区间;\n"
"- **同步终点放大了波动**:请求天然长短不一(长尾),静止 batch 被最长的那个请求拖住整批,"
"而长尾的空窗又让下一批迟迟开不了工——**波动被放大,批量被焊死**。\n\n"
"> 🏷️ 结论:naive batching 把‘物理的 GPU’(非线性吞吐、固定开销)和‘统计的流量’(长尾到达、长短不一)"
"> 各自的问题都吃了一遍。第 15 课 continuous batching 正是在这两处松绑。")

NB.md("## 8️⃣ 配套 Streamlit 演示:拖一拖,看得更清楚 🎛️",
D('''
光看静态图不过瘾?运行同目录下的 `app_14_static_batch.py`,可以**拖动滑块实时改变**
请求数量、批次大小、到达模式与随机种子,甘特图和所有指标随之刷新:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_14_static_batch.py
```

浏览器打开 **http://localhost:8501**。建议把 batch_size 从 2 一路拖到 10,观察:
平均延迟稳步上升、利用率先升后降、最后一批的空座率怎么变。完整源码如下(与同目录
`app_14_static_batch.py` 一字不差,可直接复制运行):
'''))

app_cell(NB, APP_14, "app_14_static_batch.py",
         "📜 这就是 app_14_static_batch.py 的完整源码(守卫包裹)。notebook 与 app 共享同一套模拟器,保证演示与讲解完全一致。")

wrapup(NB,
    summary=[
        "静态批处理 = 满批发车 + 整批同进退 + 空转等待,简单但低效",
        "🚦 队头阻塞:批内最长请求决定整批耗时,短请求被迫陪跑",
        "🗑️ 尾部浪费:凑不满一批就发车不了,最后一批的 GPU 座位大量闲置",
        "批次大小是把双刃剑:没有单一 batch_size 能同时最优利用率和延迟",
        "甘特图 + 三个公式把缺陷量化,为 continuous batching 立好对比基准",
    ],
    practice=[
        "把 make_reqs 的 mnew 改为固定值(所有请求生成长度相同),队头阻塞是不是消失了?为什么?",
        "在 simulate_static 里加一个 max_wait 上限:请求排队超过它就直接丢弃,观察平均延迟变化",
        "扫描 batch_size ∈ [1..16],画出“利用率 − 平均延迟”的 Pareto 曲线,找出拐点",
        "把模拟单位从“步”改成“毫秒”(假设 prefill 每 token 0.5ms、decode 每 token 2ms),重新计算墙钟时间",
    ],
    links=[
        ("Orca: 论文原文", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 博客: Continuous Batching 带来的 23x 吞吐提升", "https://blog.vllm.ai/2023/06/20/vllm.html"),
    ])

NB.save(str(Path(CH03) / "14_static_batching_problem.ipynb"))

app_path = Path(CH03) / "app_14_static_batch.py"
app_path.write_text(APP_14 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")