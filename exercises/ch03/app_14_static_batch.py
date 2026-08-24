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
