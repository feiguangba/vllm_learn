# -*- coding: utf-8 -*-
# app_29_sym_asym.py — 对称 vs 非对称量化:分布直方图实时对比 ⚖️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="⚖️ 29 · 对称 vs 非对称量化", layout="wide")
st.title("⚖️ 第 29 课 · 对称 vs 非对称量化:谁的格子用得更值?")

st.markdown("""
同一批数据、同样的位宽,**对称量化**把 0 钉在正中间(正负各半),
**非对称量化**给零点加一个 zero-point 偏移,让有限格子盖住数据的真实范围。
选择哪个,取决于分布长什么样。下面拖动参数,实时看量化前后的直方图对比:
- 📊 均值远离 0(比如 ReLU 激活全为正)→ 非对称明显占优;
- ⚖️ 分布关于 0 对称 → 两者几乎打平,对称更简单(不用存 zero-point)。
""")

def quantize(x, bits, mode):
    x = np.asarray(x, dtype=np.float64)
    if mode == "对称(symmetric)":
        qmax = 2 ** (bits - 1) - 1
        scale = max(np.max(np.abs(x)) / qmax, 1e-12)
        q = np.round(x / scale).clip(-qmax, qmax)
        x_hat = q * scale
        zp = 0
    else:
        qmax = 2 ** bits - 1
        rmin, rmax = x.min(), x.max()
        scale = max((rmax - rmin) / qmax, 1e-12)
        zp = int(np.clip(np.round(-rmin / scale), 0, qmax))
        q = np.round(x / scale + zp).clip(0, qmax)
        x_hat = (q - zp) * scale
    return x_hat, scale, zp

with st.sidebar:
    st.header("🎛️ 分布与量化参数")
    dist = st.selectbox("分布类型", ["正态(对称)", "全正 ReLU 型", "偏斜 LogNormal", "均匀(对称)"])
    mu = st.slider("分布均值 μ(位置)", -4.0, 4.0, 0.0, 0.1)
    sigma = st.slider("分布标准差 σ(散布)", 0.2, 3.0, 1.0, 0.1)
    n = st.slider("采样点数", 512, 20000, 8000, 512)
    bits = st.slider("量化位宽 b(bit)", 2, 10, 4, 1)
    st.caption("💡 提示:把 μ 拉到 +3(全正分布),再切换两种模式,看 MSE 差多少倍。")

rng = np.random.default_rng(42)
if dist == "正态(对称)":
    x = rng.normal(mu, sigma, n)                         # 零对称正态
elif dist == "全正 ReLU 型":
    x = np.maximum(rng.normal(mu, sigma, n), 0.0)       # ReLU 截断
elif dist == "偏斜 LogNormal":
    x = rng.lognormal(max(mu, 0.01), sigma, n)          # 右偏对数正态
else:
    x = rng.uniform(mu - sigma, mu + sigma, n)          # 均匀

results = {}
for mode in ["对称(symmetric)", "非对称(asymmetric)"]:
    xh, s, zp = quantize(x, bits, mode)
    results[mode] = dict(xh=xh, s=s, zp=zp, mse=float(np.mean((xh - x) ** 2)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("对称 MSE", f"{results['对称(symmetric)']['mse']:.3e}")
c2.metric("非对称 MSE", f"{results['非对称(asymmetric)']['mse']:.3e}")
ratio = results["对称(symmetric)"]["mse"] / max(results["非对称(asymmetric)"]["mse"], 1e-30)
c3.metric("对称/非对称 误差比", f"{ratio:.2f}x")
c4.metric("非对称 zero-point", f"{results['非对称(asymmetric)']['zp']}")

fig = go.Figure()
fig.add_trace(go.Histogram(x=x, name="原始分布", opacity=0.55,
                           marker_color="#4C78A8", nbinsx=80))
for mode, color in [("对称(symmetric)", "#E45756"), ("非对称(asymmetric)", "#F2C14E")]:
    fig.add_trace(go.Histogram(x=results[mode]["xh"], name="量化后·" + mode,
                               opacity=0.55, marker_color=color, nbinsx=80))
fig.update_layout(barmode="overlay", title=f"{dist} · μ={mu:.1f}, σ={sigma:.1f} · {bits} bit 量化前后直方图",
                  xaxis_title="数值", yaxis_title="频数(计数)", height=460,
                  legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 位宽扫描:两种量化方式的 MSE 对比")
sweep = []
for b in range(2, 11):
    for mode in ["对称(symmetric)", "非对称(asymmetric)"]:
        xh, _, _ = quantize(x, b, mode)
        sweep.append((b, mode, float(np.mean((xh - x) ** 2))))
fig2 = go.Figure()
for mode, color in [("对称(symmetric)", "#E45756"), ("非对称(asymmetric)", "#F2C14E")]:
    ys = [v for (b, m, v) in sweep if m == mode]
    bs = [b for (b, m, v) in sweep if m == mode]
    fig2.add_trace(go.Scatter(x=bs, y=ys, mode="lines+markers", name=mode,
                              line=dict(color=color, width=2)))
fig2.update_layout(title="位宽 b vs MSE(对数轴):分布越偏,非对称优势越大",
                   xaxis_title="位宽 b(bit)", yaxis_title="MSE", yaxis_type="log",
                   height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **经验法则**:权重(大致零对称)用对称量化;激活(ReLU 后全为正)用非对称量化。
> 这也是 PyTorch / vLLM 里最常见的 QConfig 组合:权重 per-channel 对称 + 激活 per-tensor 非对称。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 5 章 · 第 29 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
