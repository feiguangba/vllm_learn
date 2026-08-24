# -*- coding: utf-8 -*-
# app_87_ascend_quant.py — 华为量化方案:位宽、误差与体积交互 🔢
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🔢 87 · 华为量化方案", layout="wide")
st.title("🔢 第 87 课 · 华为量化方案:用位宽换体积,用校准保精度")

st.markdown("""
昇腾侧的量化(MindSpore PTQ/QAT、MindIE 低比特)与 vLLM 侧(GPTQ/AWQ/FP8)遵循同一套底层逻辑:
**用更低的位宽表示权重/激活,用校准数据保住精度**。下方拖动**位宽**、选择**校准策略**与
**数值分布**,观察 **量化误差 vs 体积** 的权衡,以及 FP8 的 E4M3/E5M2 两种尾数布局。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    bits = st.slider("量化位宽(bit)", 1, 8, 4, 1)
    scheme = st.radio("校准策略", ["全局 scale", "逐层 scale", "逐通道 scale"])
    dist = st.select_slider("权重分布尺度(σ)", options=[0.05, 0.1, 0.3, 0.5, 1.0], value=0.3)
    n_weights = st.slider("权重数量(万)", 10, 500, 100, 10)
    st.caption("量化误差 = |q(x) - x| 的均值;scale 越细(逐通道)、分布越集中,误差越小。")

rng = np.random.default_rng(0)
x = rng.normal(0, dist, int(n_weights * 1e4))

def quant_error(x, bits, scheme):
    if scheme == "全局 scale":
        s = x.abs().max() / (2 ** (bits - 1) - 1)
        q = np.clip(np.round(x / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        return np.abs(q * s - x).mean(), 1.0 / s
    if scheme == "逐层 scale":
        s = np.quantile(np.abs(x), 0.99) / (2 ** (bits - 1) - 1)
        q = np.clip(np.round(x / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        return np.abs(q * s - x).mean(), 1.0 / s
    chunks = np.array_split(x, 32)                     # 逐通道
    errs, scales = [], []
    for c in chunks:
        s = c.abs().max() / (2 ** (bits - 1) - 1)
        q = np.clip(np.round(c / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        errs.append(np.abs(q * s - c).mean()); scales.append(1.0 / s)
    return np.mean(errs), np.mean(scales)

err, _ = quant_error(x, bits, scheme)
vol_ratio = bits / 8.0 * 100
c1, c2, c3 = st.columns(3)
c1.metric("量化误差(均值)", f"{err:.4f}")
c2.metric("体积占比", f"{vol_ratio:.0f} %", f"省 {(1-vol_ratio):.0f}%")
c3.metric("数据点", f"{len(x)/1e4:.0f} 万")

st.subheader("📉 误差 vs 位宽曲线")
bits_range = np.arange(1, 9)
fig = go.Figure()
for scheme in ["全局 scale", "逐层 scale", "逐通道 scale"]:
    ys = [quant_error(x, b, scheme)[0] for b in bits_range]
    fig.add_trace(go.Scatter(x=bits_range, y=ys, mode="lines+markers",
                             name=scheme, line=dict(width=3)))
fig.add_vline(x=bits, line_dash="dash", line_color="#E45756")
fig.update_layout(title="量化误差随位宽指数下降;校准越细,曲线越低",
                  xaxis_title="位宽(bit)", yaxis_title="平均绝对误差",
                  height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 4 bit 是“性价比甜点”,8 bit(FP8/INT8)几乎无损——所以 GPTQ/AWQ 用 4bit,昇腾 FP8 用 8bit。")

st.subheader("🔬 FP8 的两种尾数布局")
if st.checkbox("对比 E4M3 与 E5M2", value=True):
    # 示意:E4M3 精度高范围小;E5M2 范围大精度低
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[0.0625, 0.125, 0.25, 0.5, 1.0], mode="lines+markers",
                              name="E4M3(精度优先)", line=dict(color="#4C78A8", width=3)))
    fig2.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[0.5, 1.0, 2.0, 4.0, 8.0], mode="lines+markers",
                              name="E5M2(范围优先)", line=dict(color="#E45756", width=3)))
    fig2.update_layout(title="E4M3:小步长高精度;E5M2:大步长宽范围(示意图)",
                       xaxis_title="指数位步进", yaxis_title="可表示步长", height=380,
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("⭐ 权重常用 E4M3(精度重要),梯度/激活常用 E5M2(防溢出)——FP8 训练与推理的通行做法。")

st.markdown("""
> 💡 **结论**:昇腾与 vLLM 的量化只是“外壳”不同——PTQ/QAT 与 GPTQ/AWQ 的差异在于
> **校准方法与误差补偿**,底层都是把权重压进更低位宽。误差随位宽指数下降、
> 随校准粒度(逐通道>逐层>全局)变细而下降,这两条规律放之四海皆准。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 87 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
