# -*- coding: utf-8 -*-
# app_30_quant_error.py — 量化误差分析:MSE/SNR 与校准方法交互演示 📏
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📏 30 · 量化误差分析", layout="wide")
st.title("📏 第 30 课 · 量化误差分析:离群值、重尾与校准的攻防战")

st.markdown("""
量化误差不只取决于位宽,更取决于**数据分布的形状**:
- 🐭 正态分布「温柔均匀」,量化轻松;
- 🦎 重尾分布(t 分布、离群值)会让 max 校准的 scale 被极端值撑爆;
- ✂️ **钳位(clipping)+ 分位数校准**:主动舍弃最极端的尾巴,换取全体更细的格子。

下方选择分布与校准方法,实时观察误差指标与 MSE-分位曲线:
""")

def quantize_clip(x, bits, clip_pct):
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    thr = np.percentile(np.abs(x), clip_pct) if clip_pct < 100 else np.abs(x).max()
    thr = max(thr, 1e-12)
    scale = thr / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale, scale

def metrics(x, xh):
    err = xh - x
    mse = float(np.mean(err ** 2))
    sig = float(np.mean(x ** 2))
    snr = 10.0 * np.log10(sig / max(mse, 1e-30))
    return mse, snr, float(np.max(np.abs(err)))

DISTS = {
    "正态 N(0,1)":        lambda rng, n, k: rng.normal(0, 1, n),
    "拉普拉斯(尖峰)":     lambda rng, n, k: rng.laplace(0, 1, n),
    "t 分布 df=3(重尾)":  lambda rng, n, k: rng.standard_t(3, n),
    "正态 + 离群值":       lambda rng, n, k: np.where(rng.random(n) < 0.01,
                                               rng.normal(0, 1, n) * (1.0 + k * 9.0),
                                               rng.normal(0, 1, n)),
}

with st.sidebar:
    st.header("🎛️ 实验参数")
    dist_name = st.selectbox("分布类型", list(DISTS.keys()))
    k = st.slider("离群值强度(仅'正态+离群值'生效)", 0.0, 3.0, 1.0, 0.1)
    bits = st.slider("量化位宽 b(bit)", 2, 10, 4, 1)
    clip = st.slider("校准分位数(%)", 90.0, 100.0, 99.9, 0.1,
                     help="100 = max 校准(不钳位);99.9 = 把最极端 0.1% 的值钳掉")
    n = st.slider("采样点数", 2000, 50000, 20000, 2000)

rng = np.random.default_rng(0)
x = DISTS[dist_name](rng, n, k)                      # 生成数据

xh_max, s_max = quantize_clip(x, bits, 100.0)       # max 校准
xh_pct, s_pct = quantize_clip(x, bits, clip)        # 分位校准

mse_max, snr_max, mae_max = metrics(x, xh_max)
mse_pct, snr_pct, mae_pct = metrics(x, xh_pct)

c1, c2, c3 = st.columns(3)
c1.metric("max 校准 MSE", f"{mse_max:.3e}", f"SNR {snr_max:.1f} dB")
c2.metric(f"{clip}% 校准 MSE", f"{mse_pct:.3e}", f"SNR {snr_pct:.1f} dB")
c3.metric("误差倍数(max/钳位)", f"{mse_max / max(mse_pct, 1e-30):.2f}x")

fig = go.Figure()
fig.add_trace(go.Histogram(x=x, name="原始分布", opacity=0.5, marker_color="#4C78A8", nbinsx=120))
fig.add_trace(go.Histogram(x=xh_max, name="max 校准重建", opacity=0.5, marker_color="#E45756", nbinsx=120))
fig.add_trace(go.Histogram(x=xh_pct, name=f"{clip}% 钳位重建", opacity=0.5, marker_color="#F2C14E", nbinsx=120))
fig.update_layout(barmode="overlay", title=f"{dist_name} · {bits} bit · 量化前后分布对比",
                  xaxis_title="数值", yaxis_title="频数", height=420,
                  legend=dict(orientation="h", y=1.14), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 校准分位数扫描:MSE 的 U 型曲线")
pcts = np.arange(80.0, 100.01, 0.5)
mses = []
for p in pcts:
    xh, _ = quantize_clip(x, bits, float(p))
    mses.append(metrics(x, xh)[0])
best = pcts[int(np.argmin(mses))]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=pcts, y=mses, mode="lines", name="MSE",
                          line=dict(color="#72B7B2", width=2)))
fig2.add_vline(x=best, line_dash="dot", line_color="#E45756",
               annotation_text=f"最优 ≈ {best:.1f}%")
fig2.add_vline(x=100.0, line_dash="dot", line_color="#999",
               annotation_text="max 校准")
fig2.update_layout(title="钳位分位 vs MSE:左边格子太粗(钳太狠),右边离群值撑爆 scale(不钳)",
                   xaxis_title="校准分位数(%)", yaxis_title="MSE", height=380,
                   yaxis_type="log", margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **校准(calibration)的本质**:找一个 clip 阈值 $c$,让「钳位误差 + 舍入误差」总和最小。
> 重尾越重,最优 $c$ 离 max 越远——这正是 GPTQ/AWQ 用几百条真实样本做校准集的原因:
> 用经验分布替你回答「哪些值可以牺牲」。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 30 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
