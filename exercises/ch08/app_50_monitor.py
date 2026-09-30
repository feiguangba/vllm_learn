# -*- coding: utf-8 -*-
# app_50_monitor.py — 性能指标与监控仪表盘
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="50 · 指标监控", layout="wide")
st.title("第 50 课 · 性能指标与监控:TTFT / TPOT / 吞吐 / Prometheus")

st.markdown("""
衡量一个 LLM 推理服务「快不快」,主要看三类指标:**首字延迟(TTFT)**、**每字延迟(TPOT)** 与
**吞吐(throughput)**。下方用**并发滑杆**模拟不同负载,实时计算这些指标并画成仪表盘。
这对应 vLLM 通过 Prometheus 暴露给运维监控的核心指标(概念一致)。
""")

MODELS = {
    "小模型 (1.5B)":  dict(ttft=0.08, tpot=0.012),
    "中模型 (7B)":    dict(ttft=0.25, tpot=0.035),
    "大模型 (14B)":   dict(ttft=0.50, tpot=0.070),
}

with st.sidebar:
    st.header("负载与模型")
    model = st.selectbox("模型规模", list(MODELS.keys()))
    concurrency = st.slider("并发请求数 (concurrency)", 1, 64, 8, 1,
                            help="同一时刻有多少个请求在服务端并行处理")
    out_tokens = st.slider("平均输出 token 数/请求", 16, 1024, 256, 16)
    alpha = st.slider("争抢系数(并发越高越慢)", 0.0, 0.5, 0.20, 0.01)
    st.caption("并发越高,共享 GPU 与 KV cache 的争抢越严重,单请求延迟越高。")

cfg = MODELS[model]

def simulate(c, ttft_base, tpot_base, out_tokens, alpha):
    """给定并发 c,模拟出 TTFT / TPOT / E2E / 吞吐 四项指标。"""
    ttft = ttft_base * (1 + alpha * (c - 1))        # TTFT 随争抢线性上升
    tpot = tpot_base * (1 + alpha * (c - 1) * 0.5)  # TPOT 上升更缓(系数减半)
    e2e = ttft + out_tokens * tpot                  # 端到端 = TTFT + 输出数×TPOT
    throughput = c * out_tokens / e2e               # 系统吞吐 tokens/s
    return ttft, tpot, e2e, throughput

ttft, tpot, e2e, throughput = simulate(concurrency, cfg["ttft"], cfg["tpot"], out_tokens, alpha)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("TTFT(首字延迟)", f"{ttft*1000:.0f} ms", help="Time To First Token:首个 token 前的时间")
c2.metric("TPOT(每字延迟)", f"{tpot*1000:.1f} ms", help="Time Per Output Token:每输出一个 token 的时间")
c3.metric("E2E(端到端)", f"{e2e:.2f} s", help="从发出请求到收完整段回答")
c4.metric("吞吐", f"{throughput:.0f} tokens/s", help="单位时间整个服务生成的 token 数")
c5.metric("并发", f"{concurrency} req", help="当前模拟负载")
st.caption(f"{model} · 平均输出 {out_tokens} token/请求 · 争抢系数 α={alpha:.2f}")

st.subheader("吞吐与延迟随并发变化")
cs = list(range(1, 65))
ts = [simulate(c, cfg["ttft"], cfg["tpot"], out_tokens, alpha) for c in cs]
e2es = [t[2] for t in ts]
thrs = [t[3] for t in ts]
fig = go.Figure()
fig.add_trace(go.Scatter(x=cs, y=thrs, mode="lines+markers", name="吞吐 (tokens/s)",
                         line=dict(width=3, color="#4C78A8"), yaxis="y"))
fig.add_trace(go.Scatter(x=cs, y=e2es, mode="lines", name="E2E 延迟 (s)",
                         line=dict(width=2, dash="dot", color="#E45756"), yaxis="y2"))
fig.add_vline(x=concurrency, line_dash="dash", line_color="#72B7B2",
              annotation_text=f"当前并发 {concurrency}", annotation_position="top")
fig.update_layout(title="负载扫描:吞吐 vs 端到端延迟(注意吞吐的饱和点)",
                  xaxis_title="并发请求数", height=440,
                  yaxis=dict(title="吞吐 (tokens/s)"),
                  yaxis2=dict(title="E2E 延迟 (s)", overlaying="y", side="right"),
                  legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("观察:并发低时吞吐随并发近线性上涨;超过拐点后延迟暴涨、吞吐趋于饱和——这就是运维要找的「甜蜜点」。")

st.subheader("E2E 延迟组成:TTFT + TPOT×输出数")
t_list = []
for c in cs:
    a, b, e, _ = simulate(c, cfg["ttft"], cfg["tpot"], out_tokens, alpha)
    t_list.append((a, b * out_tokens, e))
ttft_comp = [t[0] for t in t_list]
decode_comp = [t[1] for t in t_list]
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=cs, y=ttft_comp, name="TTFT(预填充)", marker_color="#72B7B2"))
fig2.add_trace(go.Bar(x=cs, y=decode_comp, name="解码(TPOT×输出)", marker_color="#4C78A8"))
fig2.update_layout(barmode="stack", title="E2E 延迟组成随并发变化",
                   xaxis_title="并发请求数", yaxis_title="秒", height=400,
                   legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("长回答时解码部分(TPOT×输出数)占 E2E 大头;短回答时 TTFT 占比上升。所以「快不快」要分场景看。")

st.markdown("""
> **vLLM 的监控**:真实 vLLM 服务通过 **Prometheus** 在 `/metrics` 暴露指标(如 `time_to_first_token_seconds`、
> `num_requests_running`),配合 Grafana 画成仪表盘。
> 注意:真实 vLLM 指标名前缀是 `vllm:`(带冒号,这是 vLLM 的一个已知特例);而**我们自己写的模拟指标**
> 遵循 Prometheus 最佳实践用下划线(`vllm_time_to_first_token_seconds`),因为冒号被保留给 recording rule。
> 见 [vLLM Metrics](https://docs.vllm.ai/en/stable/design/metrics) 与
> [Prometheus Naming](https://prometheus.io/docs/practices/naming)。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 8 章 · 第 50 课配套演示")
