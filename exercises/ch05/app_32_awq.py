# -*- coding: utf-8 -*-
# app_32_awq.py — AWQ 思想:激活感知的通道保护交互演示 🛡️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🛡️ 32 · AWQ 激活感知量化", layout="wide")
st.title("🛡️ 第 32 课 · AWQ 思想:保护那 1% 的重要权重")

st.markdown("""
AWQ(Activation-aware Weight Quantization)的洞察:
**不看激活值大小,看激活通道的幅度分布**——幅度大的通道上,权重的量化误差会被放大,
所以给这些通道的权重乘上缩放 $s$ 再量化(推理时激活同除 $s$),误差就被压下去了。

数学上完全等价:$y = x W^T = (x/s)(s⊙W)^T$,但量化误差的分布完全不同。
下方调节激活分布偏斜度与保护强度,实时观察输出误差:
""")

def quantize_rtn(W, bits):
    # per-row RTN 量化(无保护基线)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=1, keepdims=True) / qmax   # 每输出行一把尺子
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(W / scale), -qmax, qmax)
    return q * scale

def awq_quantize(W, act_scale, bits, alpha):
    # numpy 版简化 AWQ:显著通道权重预缩放保护
    s = (act_scale / act_scale.mean()) ** alpha    # 保护缩放(幂函数)
    s = s / s.min()                                # 归一化
    qmax = 2 ** (bits - 1) - 1
    Ws = W * s                                     # 预缩放
    scale = np.abs(Ws).max(axis=1, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(Ws / scale), -qmax, qmax)
    return (q * scale) / s                         # 反量化并除回缩放

with st.sidebar:
    st.header("🎛️ 实验参数")
    skew = st.slider("激活分布偏斜度 σ(LogNormal)", 0.0, 3.0, 1.5, 0.1,
                     help="越大 → 少数通道的激活幅度越突出,这些通道越需要保护")
    alpha = st.slider("保护强度 α(0=不保护)", 0.0, 1.0, 0.5, 0.05,
                      help="AWQ 论文网格搜索 α∈[0,1],通常 0.5 附近最优")
    bits = st.slider("量化位宽 b(bit)", 2, 8, 3, 1)
    d = st.slider("通道数(输入维度)", 64, 512, 256, 32)
    st.caption("💡 先把 α 拖到 0 看无保护误差,再拖到 0.5,对比输出 MSE。")

rng = np.random.default_rng(0)
act_scale = rng.lognormal(0.0, skew, d)          # 每个输入通道的平均激活幅度
W = rng.standard_normal((128, d)) / np.sqrt(d)   # 权重 (128, d)
X = rng.standard_normal((1000, d)) * act_scale   # 激活与通道幅度挂钩

W_rtn = quantize_rtn(W, bits)                    # 无保护 RTN
W_awq = awq_quantize(W, act_scale, bits, alpha)  # AWQ 保护

out = X @ W.T
e_rtn = float(np.mean((X @ W_rtn.T - out) ** 2))
e_awq = float(np.mean((X @ W_awq.T - out) ** 2))

c1, c2, c3, c4 = st.columns(4)
c1.metric("无保护 RTN 输出 MSE", f"{e_rtn:.3e}")
c2.metric(f"AWQ(α={alpha:.2f}) 输出 MSE", f"{e_awq:.3e}")
c3.metric("误差降低倍数", f"{e_rtn / max(e_awq, 1e-30):.2f}x")
top1pct = int(np.ceil(d * 0.01))
c4.metric("Top 1% 通道激活占比", f"{float(np.sort(act_scale)[-top1pct:].sum() / act_scale.sum()):.1%}")

s_view = (act_scale / act_scale.mean()) ** alpha
s_view = s_view / s_view.min()
fig = go.Figure()
idx = np.argsort(act_scale)
fig.add_trace(go.Scatter(x=np.arange(d), y=act_scale[idx] / act_scale.mean(),
                         name="通道激活幅度(归一)", yaxis="y",
                         line=dict(color="#4C78A8", width=2)))
fig.add_trace(go.Scatter(x=np.arange(d), y=s_view[idx], name=f"保护缩放 s(α={alpha:.2f})",
                         yaxis="y2", line=dict(color="#E45756", width=2)))
fig.update_layout(title="按激活幅度排序的通道:红线是 AWQ 给每个通道权重的保护缩放",
                  xaxis_title="通道(按激活幅度升序)", height=420,
                  yaxis=dict(title="激活幅度(相对均值)"),
                  yaxis2=dict(title="保护缩放 s", overlaying="y", side="right", type="log"),
                  legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 保护强度 α 扫描:不是越大越好")
alphas = np.arange(0.0, 1.01, 0.05)
errs = []
for a in alphas:
    Wa = awq_quantize(W, act_scale, bits, float(a))
    errs.append(float(np.mean((X @ Wa.T - out) ** 2)))
best = alphas[int(np.argmin(errs))]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=alphas, y=errs, mode="lines+markers", name="输出 MSE",
                          line=dict(color="#72B7B2", width=2)))
fig2.add_vline(x=best, line_dash="dot", line_color="#E45756",
               annotation_text=f"最优 α ≈ {best:.2f}")
fig2.update_layout(title="α=0 无保护误差大;α 太大又伤非显著通道——U 型最优在中间",
                   xaxis_title="保护强度 α", yaxis_title="输出 MSE", height=380,
                   margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **AWQ vs GPTQ**:GPTQ 用反向式误差补偿(需要 Hessian 逆),AWQ 只用前向激活统计
> (一次推理即可),更快更稳、不易过拟合校准集。论文:*AWQ: Activation-aware Weight
> Quantization for LLM Compression and Acceleration* (arXiv:2306.00978)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 32 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
