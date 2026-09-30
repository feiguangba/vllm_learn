# -*- coding: utf-8 -*-
# app_41_combination.py — TP × PP × DP 组合估算 🧩
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🧩 41 · 并行组合", layout="wide")
st.title("🧩 第 41 课 · 并行组合:TP × PP × DP,一手算清通信/显存/气泡")

st.markdown("""
现实里三种并行**组合使用**:**TP 摊矩阵、PP 摊层、DP 摊数据**。本页给你一套估算公式,
实时算出指定组合下的**每卡权重显存、KV 显存、TP 通信量、PP 气泡占比**,并列出
固定总卡数下所有合法组合的对比表,帮你理解「为什么这样配」。
""")

def estimate_combo(params_b, layers, hidden, gpus, tp, pp, dp, ctx=4096, batch=8,
                   dtype_bytes=2, m=8, kv_heads=8, head_dim=128):
    weights_gb = params_b * dtype_bytes / tp / pp
    kv_gb = 2 * layers * kv_heads * head_dim * dtype_bytes * ctx * batch / 1024 ** 3 / tp
    ar_mb = 2 * (layers / pp) * 2 * (tp - 1) / tp * batch * ctx * hidden * dtype_bytes / 1e6
    bubble = (pp - 1) / (m + pp - 1)
    return weights_gb, kv_gb, ar_mb, bubble

with st.sidebar:
    st.header("🎛️ 参数")
    params_b = st.slider("模型参数量(B)", 7, 405, 70, 1)
    layers = st.slider("总层数 L", 32, 96, 80, 1)
    hidden = st.slider("隐藏维度 hidden", 4096, 8192, 8192, 256)
    gpus = st.slider("总卡数", 2, 32, 8, 1)
    tp = st.slider("TP", 1, 8, 2, 1)
    pp = st.slider("PP", 1, 8, 2, 1)
    m = st.slider("micro-batch m", 4, 32, 8, 1)
    st.caption("DP 自动算为 gpus/(TP×PP)。TP 需能整除总卡数;TP/PP 越大越省显存但通信/气泡更高。")

dp = gpus // (tp * pp)
w_gb, k_gb, ar_mb, bubble = estimate_combo(params_b, layers, hidden, gpus, tp, pp, dp, m=m)
valid = (tp * pp * dp == gpus)

c1, c2, c3, c4 = st.columns(4)
c1.metric("组合", f"TP{tp}×PP{pp}×DP{dp}" + ("" if valid else " (无效)"))
c2.metric("每卡权重(GB)", f"{w_gb:.1f}")
c3.metric("每卡 KV(GB)", f"{k_gb:.1f}")
c4.metric("PP 气泡占比", f"{bubble:.1%}")

st.subheader("🧮 每卡负担分解")
fig = go.Figure()
labels = ["每卡权重", "每卡 KV"]
fig.add_trace(go.Bar(x=labels, y=[w_gb, k_gb], marker_color=["#4C78A8", "#72B7B2"],
                     text=[f"{w_gb:.1f}", f"{k_gb:.1f}"], textposition="outside"))
fig.update_layout(title=f"TP{tp}×PP{pp}×DP{dp} 下每卡显存(GB)", yaxis_title="GB",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 所有合法组合对比(纯计算)")
rows = []
for tt in [1, 2, 4, 8]:
    for pp_ in [1, 2, 4, 8]:
        dd = gpus // (tt * pp_)
        if tt * pp_ * dd == gpus:
            w, k, a, bub = estimate_combo(params_b, layers, hidden, gpus, tt, pp_, dd, m=m)
            rows.append(dict(组合=f"TP{tt}×PP{pp_}×DP{dd}", TP=tt, PP=pp_, DP=dd,
                             权重GB=round(w, 1), KV_GB=round(k, 1),
                             TP通信MB=round(a, 1), 气泡=round(bub, 3)))
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

st.markdown("""
> 💡 **选型直觉**:显存紧张→加大 TP/PP;吞吐不足→加大 DP;气泡敏感→少用 PP 或加大 m。
> 实际配置还要看单卡显存容量、总线拓扑与负载特征,本页是「方向正确」的估算,非精确仿真。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 41 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
