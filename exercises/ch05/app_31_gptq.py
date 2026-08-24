# -*- coding: utf-8 -*-
# app_31_gptq.py — GPTQ 思想:误差补偿 vs 朴素逐列量化交互演示 🧮
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧮 31 · GPTQ 逐列量化与误差补偿", layout="wide")
st.title("🧮 第 31 课 · GPTQ 思想:逐列量化 + 误差补偿")

st.markdown("""
GPTQ 的核心动作只有两步:
1. **逐列量化**:按列把权重舍入到最近的量化格子上(朴素 RTN);
2. **误差补偿**:第 $i$ 列的量化残差,按 Hessian 逆指示的方向**摊到还没量化的列**上,
   让后续列「顺手」修正前面丢掉的精度。

补偿的收益取决于输入列之间的**相关性**——相关性越强,补偿越有的放矢。
下方调节矩阵大小、位宽与相关性,对比两种策略的累计输出误差:
""")

def gptq(W, X, bits, compensate):
    # numpy 版简化 GPTQ(与 notebook 相同:H = 2·X^T·X + 阻尼,Cholesky 补偿)
    d_out, d_in = W.shape
    H = 2.0 * X.T @ X                          # Hessian 近似 (d_in, d_in)
    H += 0.01 * np.diag(H).mean() * np.eye(d_in)   # 阻尼,防止奇异
    L_factor = np.linalg.cholesky(np.linalg.inv(H)).T   # H^{-1} 的上三角 Cholesky 因子
    W = W.copy()
    Q = np.zeros_like(W)
    qmax = 2 ** (bits - 1) - 1
    trail = []
    for i in range(d_in):                      # 逐列量化
        w = W[:, i]                            # 当前列
        s = max(abs(w).max() / qmax, 1e-12)    # 该列 scale
        q = np.clip(np.round(w / s), -qmax, qmax)  # 量化到 int
        Q[:, i] = q * s                        # 反量化存下
        if compensate and i + 1 < d_in:        # 误差补偿
            err = (w - Q[:, i]) / L_factor[i, i]   # 残差按对角归一
            W[:, i + 1:] -= np.outer(err, L_factor[i, i + 1:])  # 摊到未量化列
        if (i + 1) % max(d_in // 20, 1) == 0:
            trail.append(float(np.mean((X @ Q.T - X @ W0.T) ** 2)))
    return Q, trail

def output_mse(W, Q, X):
    # 输出 MSE:量化前后 XW^T 的差异
    return float(np.mean((X @ W.T - X @ Q.T) ** 2))

with st.sidebar:
    st.header("🎛️ 实验参数")
    d_out = st.slider("输出维度 d_out", 8, 128, 32, 8)
    d_in = st.slider("输入维度 d_in(量化列数)", 8, 128, 48, 8)
    corr = st.slider("输入列相关性 ρ(0=独立,1=强相关)", 0.0, 0.99, 0.8, 0.01,
                     help="相关性越高,Hessian 非对角越强,补偿能搬的'救兵'越多")
    bits = st.slider("量化位宽 b(bit)", 2, 8, 3, 1)
    st.caption("💡 把 ρ 从 0 拖到 0.9,看补偿增益如何从几乎为零涨到好几倍。")

rng = np.random.default_rng(7)
rho = corr
A = rng.standard_normal((2000, d_in))          # 独立成分
common = rng.standard_normal((2000, 1))        # 公共因子
X = np.sqrt(1 - rho) * A + np.sqrt(rho) * common   # 列间相关性与 ρ 挂钩

W0 = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)   # 权重

Q_naive, _ = gptq(W0, X, bits, compensate=False)  # 朴素 RTN
Q_gptq, trail = gptq(W0, X, bits, compensate=True) # GPTQ 补偿

e_naive = output_mse(W0, Q_naive, X)
e_gptq = output_mse(W0, Q_gptq, X)

c1, c2, c3 = st.columns(3)
c1.metric("朴素逐列(RTN) 输出 MSE", f"{e_naive:.3e}")
c2.metric("GPTQ 补偿 输出 MSE", f"{e_gptq:.3e}")
c3.metric("误差降低倍数", f"{e_naive / max(e_gptq, 1e-30):.1f}x")

st.subheader("📊 逐列量化的累计输出误差(补偿 vs 朴素)")
steps = np.arange(1, len(trail) + 1) * max(d_in // len(trail), 1)
_, trail_n = gptq(W0, X, bits, compensate=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=steps, y=trail, mode="lines+markers", name="GPTQ(带补偿)",
                         line=dict(color="#72B7B2", width=2)))
fig.add_trace(go.Scatter(x=steps, y=trail_n, mode="lines+markers", name="朴素 RTN(无补偿)",
                         line=dict(color="#E45756", width=2, dash="dot")))
fig.update_layout(title="已量化列数 vs 累计输出 MSE:补偿让误差增长更慢甚至回落",
                  xaxis_title="已量化的列数", yaxis_title="输出 MSE", height=400,
                  yaxis_type="log", margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **一句话总结 GPTQ**:量化第 $i$ 列时把残差「记账」,按照 Hessian 逆给的汇率
> 换算成对未量化列的修正量——用后面的列,还前面欠的债。
> 论文:*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (arXiv:2210.17323)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 31 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
