# -*- coding: utf-8 -*-
# app_38_pipeline_parallel.py — 流水线并行的 GPipe vs 1F1B 甘特图 🏭
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🏭 38 · 流水线并行", layout="wide")
st.title("🏭 第 38 课 · 流水线并行:切层 + 微批,看懂气泡")

st.markdown("""
**流水线并行(PP)** 把模型按层切成若干「舞台(stage)」,像工厂流水线一样接力处理一批样本。
为了不让上游舞台干等,把大 batch 拆成 **micro-batch** 流水推进。
但流水线会留下 **气泡(bubble)**——某些舞台在某些时刻在空等。对比两种调度:
- **GPipe**:每舞台先做完全部前向,再做反向(U 形);
- **1F1B(PipeDream-Flush)**:稳态期一个前向接一个反向,气泡更小。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    p = st.slider("舞台数(层分组)", 2, 8, 4, 1)
    m = st.slider("micro-batch 数", 2, 24, 8, 1)
    f = st.slider("前向耗时 f", 0.5, 3.0, 1.0, 0.1)
    b = st.slider("反向耗时 b", 1.0, 5.0, 2.0, 0.1)
    st.caption("m 越大气泡越小,但激活驻留越高(1F1B 缓解);f/b 为每个 micro-batch 的计算耗时。")

def run_pipeline(order_per_stage, p, f, b):
    # 事件驱动流水线模拟器
    t0 = [0.0] * p
    fin = {}
    ptr = [0] * p
    events = []
    remaining = sum(len(q) for q in order_per_stage)
    while remaining:
        progressed = False
        for s in range(p):
            if ptr[s] >= len(order_per_stage[s]):
                continue
            mb, typ = order_per_stage[s][ptr[s]]
            if typ == "F":
                dep = fin.get((s - 1, mb, "F"), 0.0) if s > 0 else 0.0
            else:
                if s < p - 1:
                    if (s + 1, mb, "B") not in fin:
                        continue         # 下游同 mb 的 B 还没排到,本轮到下一圈再处理
                    dep = fin[(s + 1, mb, "B")]
                else:
                    dep = 0.0
            cost = f if typ == "F" else b
            start = max(t0[s], dep)
            events.append((s, mb, typ, start, start + cost))
            fin[(s, mb, typ)] = start + cost
            t0[s] = start + cost
            ptr[s] += 1
            remaining -= 1
            progressed = True
        assert progressed, "调度死锁"
    return events

def schedule_gpipe(p, m, f, b):
    return run_pipeline([[(mb, "F") for mb in range(m)] + [(mb, "B") for mb in reversed(range(m))]
                         for s in range(p)], p, f, b)

def schedule_1f1b(p, m, f, b):
    orders = []
    for s in range(p):
        w = min(p - 1 - s, m)
        q = [(mb, "F") for mb in range(w)]
        for i in range(m - w):
            q.append((w + i, "F")); q.append((i, "B"))
        for i in range(m - w, m):
            q.append((i, "B"))
        orders.append(q)
    return run_pipeline(orders, p, f, b)

def stats(events, p):
    makespan = max(e[4] for e in events)
    busy = sum(e[4] - e[3] for e in events)
    return makespan, 1 - busy / (p * makespan)

ev_g = schedule_gpipe(p, m, f, b)
ev_1 = schedule_1f1b(p, m, f, b)
ms_g, bub_g = stats(ev_g, p)
ms_1, bub_1 = stats(ev_1, p)
formula = (p - 1) / (m + p - 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("GPipe 总时长", f"{ms_g:.1f}")
c2.metric("GPipe 气泡占比", f"{bub_g:.1%}")
c3.metric("1F1B 气泡占比", f"{bub_1:.1%}")
c4.metric("公式 (p-1)/(m+p-1)", f"{formula:.1%}")

def gantt(events, title):
    fig = go.Figure()
    for s, mb, typ, stt, enn in events:
        fig.add_trace(go.Bar(x=[enn - stt], y=[f"Stage {s}"],
                             base=stt, orientation="h", name="",
                             marker_color="#72B7B2" if typ == "F" else "#E45756",
                             hovertemplate=f"mb{mb} {typ} [{stt:.1f},{enn:.1f}]<extra></extra>",
                             showlegend=False))
    fig.update_layout(title=title, xaxis_title="时间", yaxis_title="舞台",
                      barmode="stack", height=360, bargap=0.1,
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

left, right = st.columns(2)
with left:
    st.subheader("GPipe")
    st.plotly_chart(gantt(ev_g, "GPipe(先全部 F 再全部 B)"), use_container_width=True)
with right:
    st.subheader("1F1B")
    st.plotly_chart(gantt(ev_1, "1F1B(1 前向接 1 反向)"), use_container_width=True)
st.caption("🟦 前向 / 🟥 反向。1F1B 的反向更早开始、气泡更小,且激活驻留更低。")

st.markdown(f"""
> 💡 **直觉**:舞台越多气泡越大,微批越多气泡越小。公式 **(p-1)/(m+p-1)** 在理想假设下精确成立
> (与 f、b 无关)。当前:公式 {formula:.1%},1F1B 实测 {bub_1:.1%}。1F1B 相比 GPipe 把
> 「同时驻留的激活数」从 m 降到约 p,更省显存,所以现代框架(含 vLLM 的 PP)普遍用它。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 38 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
