# -*- coding: utf-8 -*-
# app_39_data_parallel.py — 推理数据并行的负载均衡 🧑‍🤝‍🧑
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="🧑‍🤝‍🧑 39 · 数据并行", layout="wide")
st.title("🧑‍🤝‍🧑 第 39 课 · 推理数据并行:多个完整模型,一起接客")

st.markdown("""
**数据并行(DP)** 在推理时最简单:**每张卡各放一份完整的模型副本**,把进来的请求
「分发」给不同副本并行处理。既然每卡都能独立出答案,关键在于**怎么分请求才最均衡**——
有的请求难(耗时久),分配不均会导致某些副本排队、某些副本空闲。本页对比三种分发策略。
""")

@dataclass
class Req:
    rid: int
    arrive: float
    work: float

def make_stream(n, rng, wmin, wmax):
    t = 0.0
    reqs = []
    for i in range(n):
        t += rng.exponential(0.8)
        reqs.append(Req(i, t, float(rng.uniform(wmin, wmax))))
    return reqs

def dispatch_dp(reqs, n_replica, policy):
    loads = [0.0] * n_replica
    lanes = [[] for _ in range(n_replica)]
    rr = 0
    for r in reqs:
        if policy == "轮询 round_robin":
            k = rr % n_replica; rr += 1
        elif policy == "最短队列 least_loaded":
            k = int(np.argmin(loads))
        else:
            k = int(np.random.randint(n_replica))
        lanes[k].append(r)
        loads[k] = max(loads[k], r.arrive) + r.work
    return lanes

def dp_stats(lanes):
    fin, lats = [], []
    for lane in lanes:
        t = 0.0
        for r in lane:
            start = max(t, r.arrive)
            fin.append(start + r.work); lats.append(start + r.work - r.arrive)
            t = start + r.work
    total_work = [sum(r.work for r in lane) for lane in lanes]
    return dict(makespan=max(fin), avg_lat=float(np.mean(lats)),
                imbalance=float(np.std(total_work) / np.mean(total_work)))

with st.sidebar:
    st.header("🎛️ 参数")
    n_req = st.slider("请求数", 8, 80, 30, 1)
    n_rep = st.slider("DP 副本数", 1, 8, 4, 1)
    policy = st.radio("分发策略", ["轮询 round_robin", "最短队列 least_loaded", "随机 random"])
    seed = st.slider("随机种子", 0, 20, 3, 1)
    st.caption("请求耗时随机:有的快有的慢,正是「不均匀」考验负载均衡。")

rng = np.random.default_rng(seed)
stream = make_stream(n_req, rng, 2, 30)
lanes = dispatch_dp(stream, n_rep, policy)
stt = dp_stats(lanes)

c1, c2, c3, c4 = st.columns(4)
c1.metric("副本数", n_rep)
c2.metric("总完成时间", f"{stt['makespan']:.1f}")
c3.metric("平均延迟", f"{stt['avg_lat']:.1f}")
c4.metric("负载不均衡度", f"{stt['imbalance']:.2f}")

st.subheader("⚖️ 各副本承接的工作量")
loads = [sum(r.work for r in lane) for lane in lanes]
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"副本{i+1}" for i in range(n_rep)], y=loads,
                     marker_color="#4C78A8", text=[f"{v:.0f}" for v in loads],
                     textposition="outside"))
fig.update_layout(title=f"每个 DP 副本被分配的请求总耗时(越均衡越好)",
                  yaxis_title="工作量", height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("⏱️ 各请求延迟分布")
lats = []
for lane in lanes:
    t = 0.0
    for r in lane:
        start = max(t, r.arrive)
        lats.append(start + r.work - r.arrive); t = start + r.work
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=lats, nbinsx=20, marker_color="#72B7B2"))
fig2.update_layout(title="请求延迟直方图(左移 = 更快)",
                   xaxis_title="延迟", yaxis_title="请求数", height=340,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **直觉**:「轮询」最公平但不管难易,「随机」最随意、常把多批难请求塞给同一个副本,
> 「最短队列」总是派给最闲的副本 → 负载最均衡、总完成时间最短。
> **推理 DP 之间几乎不通信**(各自独立出答案),所以它主要用来扩吞吐、做负载均衡。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 39 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
