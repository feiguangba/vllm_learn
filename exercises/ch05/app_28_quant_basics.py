# -*- coding: utf-8 -*-
# app_28_quant_basics.py — 量化基础:位宽与量化误差交互演示 📉
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📉 28 · 量化基础", layout="wide")
st.title("📉 第 28 课 · 量化基础:位宽越小,格子越粗,误差越大")

st.markdown("""
把一段光滑的波形塞进 **$2^b$ 个离散格子**里,就是量化。格子越少(位宽 $b$ 越小),
每个格子越宽(scale 越大),舍入误差越大。下方拖一拖滑块,直观感受「精度」是怎么被位宽吃掉的:
- 🧮 **对称量化**:正负各一半,zero-point 固定为 0;
- ⚖️ **非对称量化**:给分布加上零点偏移,适合全为正的激活。
""")

def quantize(x, bits, mode):
    # 统一量化函数:按模式返回 (量化整数, 重建值, scale, 级别数)
    x = np.asarray(x, dtype=np.float64)
    if mode == "对称(symmetric)":
        qmax = 2 ** (bits - 1) - 1          # int 最大正值:2^(b-1)-1
        scale = max(np.max(np.abs(x)) / qmax, 1e-12)   # scale = max|x| / Qmax
        q = np.round(x / scale).clip(-qmax, qmax)      # 量化:round(x/scale)
        x_hat = q * scale                              # 反量化:q * scale
        levels = 2 * qmax + 1                          # 对称可表示级别数
    else:
        qmin, qmax = 0, 2 ** bits - 1        # 非对称用 uint 范围 [0, 2^b-1]
        rmin, rmax = x.min(), x.max()        # 数据的真实范围
        scale = max((rmax - rmin) / (qmax - qmin), 1e-12)  # scale 由范围决定
        zp = np.round(qmin - rmin / scale).clip(qmin, qmax) # zero-point:浮点0 落在哪个整数
        q = np.round(x / scale + zp).clip(qmin, qmax)      # 量化:round(x/scale + zp)
        x_hat = (q - zp) * scale                          # 反量化:(q - zp) * scale
        levels = 2 ** bits                                # 非对称级别数 = 2^b
    return q, x_hat, scale, levels

with st.sidebar:
    st.header("🎛️ 参数")
    bits = st.slider("量化位宽 b(bit)", 2, 16, 8, 1)
    n = st.slider("采样点数量", 32, 512, 128, 8)
    mode = st.radio("量化方式", ["对称(symmetric)", "非对称(asymmetric)"])
    outlier = st.slider("离群值幅度(越大分布越尖)", 0.0, 5.0, 1.0, 0.1,
                        help="给波形叠加一个尖峰,观察离群值如何拉大 scale、恶化整体精度")
    st.caption("离群值是量化的大敌:一个极端值会撑大 scale,让所有格子一起变粗。")

x = np.linspace(-1, 1, n)                          # 均匀采样点作为横轴
signal = np.cos(2 * np.pi * x) + outlier * np.exp(-((x - 0.5) ** 2) / 0.002)  # 波形信号
q, x_hat, scale, levels = quantize(signal, bits, mode)   # 量化整段信号

err = x_hat - signal                             # 量化误差
mse = float(np.mean(err ** 2))                   # 均方误差
maxerr = float(np.max(np.abs(err)))              # 最大绝对误差

c1, c2, c3, c4 = st.columns(4)
c1.metric("量化步长 scale", f"{scale:.5f}")
c2.metric("可表示级别数", f"{levels}")
c3.metric("均方误差 MSE", f"{mse:.3e}")
c4.metric("最大绝对误差", f"{maxerr:.4f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=signal, mode="lines", name="原始浮点信号",
                         line=dict(width=2, color="#4C78A8")))
fig.add_trace(go.Scatter(x=x, y=x_hat, mode="markers", name="量化重建信号",
                         marker=dict(size=5, color="#F2C14E", symbol="diamond")))
fig.add_trace(go.Scatter(x=x, y=err, mode="lines", name="量化误差",
                         line=dict(width=1, dash="dot", color="#E45756"),
                         yaxis="y2"))
fig.update_layout(
    title=f"位宽 {bits} bit · 量化前后对比(误差放大在右轴)",
    xaxis_title="x", height=460,
    yaxis=dict(title="信号值"), yaxis2=dict(title="误差", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("💡 观察:把位宽从 8 拖到 2,黄色重建点明显变稀疏、误差曲线剧烈抖动;"
           "把离群值调大,即使位宽不变,误差也会整体放大——因为一个尖峰撑大了 scale。")

st.subheader("📊 位宽扫描:MSE 随位宽指数下降")
sweep = []
for b in range(2, 17):                              # 扫描位宽
    _, xh, s, lv = quantize(signal, b, mode)        # 重新量化
    sweep.append(dict(bits=b, mse=float(np.mean((xh - signal) ** 2)), levels=lv, scale=s))
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=[s["bits"] for s in sweep], y=[s["mse"] for s in sweep],
                      marker_color="#72B7B2", name="MSE"))
fig2.update_layout(title="位宽 b vs 均方误差 MSE(注意纵轴对数)",
                   xaxis_title="位宽 b(bit)", yaxis_title="MSE", height=360,
                   yaxis_type="log", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **直觉**:量化就像用一把有刻度的尺子去量长度——刻度越粗,量得越粗。
> 对称量化把 0 放在中间(正负各一半刻度),非对称量化则给零点加个偏移,
> 让有限的格子尽可能盖住数据的真实分布范围。
> 参考:[A White Paper on Neural Network Quantization (arXiv:2106.08295)](https://arxiv.org/abs/2106.08295)。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 28 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
