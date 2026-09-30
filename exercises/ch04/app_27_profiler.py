# -*- coding: utf-8 -*-
"""📈 app_27_profiler.py — 性能剖析仪表盘(minivllm 第 4 章 · 第 27 课)

运行: streamlit run app_27_profiler.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📈 27 · 性能剖析仪表盘", layout="wide")
st.title("📈 第 27 课配套 App · 性能剖析仪表盘")

PROFILE_FILE = Path(__file__).parent / "profile_summary_27.json"
if not PROFILE_FILE.exists():
    st.error(f"缺少数据文件 {PROFILE_FILE.name} —— 请先运行第 27 课 notebook 生成剖析数据。")
    st.stop()
data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
ops = data["ops"]
metrics = data["metrics"]

st.markdown(
    "基于 **torch.profiler** 对迷你 GPT 前向的剖析结果。仪表盘展示三组关键性能指标:"
    "**吞吐**(词元/秒)、**延迟**分布与 **SM 利用率**估计,以及开销最高的 top-N 算子。"
)

st.sidebar.header("🎛️ 仪表盘参数")
top_n = st.sidebar.slider("Top-N 算子", 5, min(30, len(ops)), 10)
metric_col = st.sidebar.radio("算子排序指标", ["cpu_total", "cpu_self", "calls"])
chart_type = st.sidebar.radio("图表类型", ["柱状图", "表格"])
show_gauge = st.sidebar.checkbox("显示 SM 利用率仪表", value=True)

m = metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("吞吐(词元/秒)", f"{m['throughput_tok_s']:.0f}")
c2.metric("平均单步延迟(ms)", f"{m['latency_mean_ms']:.3f}")
c3.metric("P99 延迟(ms)", f"{m['latency_p99_ms']:.3f}")
c4.metric("计算忙占比(≈利用率)", f"{m['sm_busy_pct']:.1f} %",
          help="CPU 剖析中前向计算时间占墙钟时间的比例,作为利用率代理指标(GPU 场景对应 SM 利用率)")

if show_gauge:
    st.markdown("### 🎯 性能仪表")
    st.caption("计算忙占比越接近 100%,说明调度/启动开销越少、流水线越满;这是 CUDA Graph 优化瞄准的目标。")
    g1, g2, g3 = st.columns(3)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=m["sm_busy_pct"],
        title={"text": "计算忙占比 %"}, gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": "#16a085"},
            "steps": [{"range": [0, 50], "color": "#f8d7da"}, {"range": [50, 80], "color": "#fff3cd"},
                      {"range": [80, 100], "color": "#d4edda"}],
            "threshold": {"line": {"color": "red", "width": 3}, "value": m["sm_busy_pct"]}}))
    gauge.update_layout(height=280)
    g1.plotly_chart(gauge, width="stretch")
    g2.plotly_chart(go.Figure(go.Indicator(
        mode="gauge+number", value=m["throughput_tok_s"], title={"text": "吞吐 tok/s"},
        gauge={"axis": {"range": [0, m["throughput_tok_s"] * 1.5]},
               "bar": {"color": "#2980b9"}})).update_layout(height=280), width="stretch")
    g3.plotly_chart(go.Figure(go.Indicator(
        mode="gauge+number", value=m["latency_mean_ms"], title={"text": "平均延迟 ms"},
        gauge={"axis": {"range": [0, m["latency_mean_ms"] * 2.5]},
               "bar": {"color": "#8e44ad"}})).update_layout(height=280), width="stretch")

st.markdown(f"### 🧭 Top-{top_n} 算子(按 {metric_col} 排序)")
st.caption("『cpu_self』指算子自身(不含子算子)的 CPU 耗时 —— 越大越说明它值得优化或融合。")

df = pd.DataFrame(ops).sort_values(metric_col, ascending=False).head(top_n).copy()
if chart_type == "柱状图":
    fig = px.bar(df, x="name", y=metric_col, color="name",
                 text=df[metric_col].round(3))
    fig.update_layout(height=420, xaxis_tickangle=-40, showlegend=False,
                      yaxis_title=metric_col)
    st.plotly_chart(fig, width="stretch")
else:
    st.dataframe(df[["name", "calls", "cpu_self", "cpu_total"]].round(4),
                 width="stretch", hide_index=True)

st.markdown("### 📊 算子耗时分布(累计占比)")
df_sorted = pd.DataFrame(ops).sort_values("cpu_total", ascending=False)
cum = df_sorted["cpu_total"].cumsum() / max(df_sorted["cpu_total"].sum(), 1e-9)
df_sorted["累计占比"] = cum
fig2 = px.line(df_sorted, x=np.arange(len(df_sorted)), y="累计占比", markers=True)
fig2.add_hline(y=0.8, line_dash="dash", line_color="red",
               annotation_text="80% 线:少数算子吃掉大部分时间")
fig2.update_layout(height=320, xaxis_title="算子序号(按耗时降序)", yaxis_title="累计占比")
st.plotly_chart(fig2, width="stretch")

st.markdown("---")
st.markdown(
    "💡 **直觉**:剖析器像给推理引擎装了一台**心电图仪 📈**。先看大方向(吞吐/延迟/SM 利用率),"
    "再放大到具体算子,找到「少数吃掉多数时间」的热点,最后用 kernel 融合 / CUDA Graph / 并行策略去优化。"
    "本数据来自第 27 课 notebook 的 torch.profiler CPU 剖析与 perf_counter 计时。"
)
st.caption("《minivllm: 图解 vLLM 推理引擎》第 4 章 · 第 27 课配套演示")

# ============ PyCharm / 直接运行入口 ============
# 说明: 在 PyCharm 里直接 Run 本文件,即可启动 Streamlit 服务(浏览器打开 http://localhost:8501)。
# 与命令行 `streamlit run app_XX.py` 完全等价(用子进程方式, 避免与顶层 st 调用冲突)。
if __name__ == "__main__":
    # 如果已在 Streamlit Runtime 中(AppTest/嵌入时加载),不再启动子进程。
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
