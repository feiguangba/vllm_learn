# -*- coding: utf-8 -*-
# app_65_autotune.py — 自动调优:搜索策略交互 🎯
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="自动调优 🎯", layout="wide")
st.title("🎯 第 65 课 · 自动调优:搜索策略交互")

st.markdown("""
同一段计算(比如矩阵乘)有**海量实现方式**:分块大小、线程数、流水线级数不同,性能天差地别。
**自动调优(auto-tuning)** 就是让编译器自己搜出最优配置:要么**穷举/随机**试,要么用**演化算法**,
要么靠**成本模型**猜一个方向再微调。下方选一种策略、拨预算,看『最好的那版』随搜索次数如何收敛。
""")

rng = np.random.default_rng(0)
# ---------------- 合成"性能地形"(更小的 tile 通常更快,但有个最优谷) ----------------
BM = np.arange(16, 145, 8)
BK = np.arange(8, 81, 8)
BMM, BKK = np.meshgrid(BM, BK)
# 地形:中心偏左下有个最优;远离则变差(带噪声)
perf = -((BMM - 80) ** 2 / 4000 + (BKK - 40) ** 2 / 900) - 0.5 + rng.normal(0, 0.08, BMM.shape)
flat = perf.ravel()
idx = np.arange(len(flat))

# ---------------- 三种搜索策略 ----------------
def random_search(budget, rng):
    picks = rng.choice(idx, budget, replace=False)
    best = max(flat[p] for p in picks)
    curve = np.maximum.accumulate([flat[p] for p in picks])
    return curve, best

def evolutionary(budget, rng, pop=20):
    best_curve, best = [], -np.inf
    pop_idx = rng.choice(idx, pop, replace=False)
    fit = np.array([flat[p] for p in pop_idx])
    for step in range(budget):
        b = max(fit); best = max(best, b); best_curve.append(best)
        keep = pop_idx[np.argsort(fit)[-pop // 2:]]
        children = []
        for k in keep:
            child = int(np.clip(k + rng.normal(0, 6), 0, len(idx) - 1))
            children.append(child)
        pop_idx = np.concatenate([keep, children])
        fit = np.array([flat[p] for p in pop_idx])
    return np.array(best_curve), best

def cost_model(budget, rng):
    best_curve, best = [], -np.inf
    # 从性能地形中心偏优的位置起步,每次沿梯度方向微调(用局部爬山近似"成本模型引导")
    cur = int(np.argmax(flat))
    for step in range(budget):
        b = flat[cur]; best = max(best, b); best_curve.append(best)
        nbrs = [cur + 1, cur - 1, cur + len(BM), cur - len(BM)]
        nbrs = [n for n in nbrs if 0 <= n < len(idx)]
        nxt = max(nbrs, key=lambda n: flat[n])
        if flat[nxt] <= b:      # 局部最优,随机跳一下(模拟退火式)
            cur = int(rng.integers(0, len(idx)))
        else:
            cur = nxt
    return np.array(best_curve), best

st.sidebar.header("🎛️ 参数")
strategy = st.sidebar.selectbox("搜索策略", ["成本模型引导", "演化算法", "随机搜索"])
budget = st.sidebar.slider("搜索预算(尝试次数)", 10, 200, 80, 10)
show_landscape = st.sidebar.checkbox("显示性能地形", value=True)
st.sidebar.caption("成本模型像『有地图的人』,演化像『一代代变异择优』,随机像『盲人摸象』。")

if strategy == "随机搜索":
    curve, best = random_search(budget, rng)
elif strategy == "演化算法":
    curve, best = evolutionary(budget, rng)
else:
    curve, best = cost_model(budget, rng)

c1, c2, c3 = st.columns(3)
c1.metric("找到的最优值", f"{best:.2f}")
c2.metric("搜索预算", budget)
c3.metric("相对全局最优", f"{100 * (best / flat.max()):.1f}%")
st.caption(f"当前策略:{strategy}。理论上限(全局最优)= {flat.max():.2f}")

# ---------------- 收敛曲线 ----------------
st.subheader("📈 收敛曲线:最好的解随尝试次数提升")
fig = go.Figure()
fig.add_hline(y=flat.max(), line_dash="dash", line_color="#c0392b",
              annotation_text="全局最优", annotation_position="top left")
fig.add_trace(go.Scatter(x=list(range(1, len(curve) + 1)), y=curve, mode="lines",
                         name=strategy, line=dict(width=3, color="#1f77b4"),
                         fill="tozeroy", fillcolor="rgba(31,119,180,0.15)"))
fig.update_layout(title=f"{strategy} 的 best-so-far 曲线", xaxis_title="尝试次数", yaxis_title="当前最好性能",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 曲线越早接近全局最优,策略越高效。成本模型通常起步就接近最优,但可能停在局部最优。")

# ---------------- 性能地形 ----------------
if show_landscape:
    st.subheader("🗺️ 性能地形(BM × BK,亮=快)")
    fig2 = go.Figure(go.Heatmap(z=perf, x=BM, y=BK, colorscale="Viridis",
                                hovertemplate="BM=%{x}, BK=%{y}<br>性能=%{z:.2f}<extra></extra>"))
    fig2.update_layout(title="tile 配置的性能地形(合成数据)", xaxis_title="BM(行块)", yaxis_title="BK(K 块)",
                       height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:自动调优就是在"性能地形"上爬山。穷举最稳但慢,随机省事但笨,演化会变异择优,
> 成本模型有先验方向但可能陷进局部最优 —— 真实编译器(TVM Ansor / Triton autotune)常混合多种策略。
""")
