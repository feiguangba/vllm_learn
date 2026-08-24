# -*- coding: utf-8 -*-
# app_76_mindspore_graph.py — 动态图 vs 静态图对比 🌳
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="MindSpore 图模式 🌳", layout="wide")
st.title("🌳 第 76 课 · MindSpore 计算图:动态 vs 静态对比")

st.markdown("""
MindSpore 有两种执行模式: **PyNative(动态图)** —— 逐算子解释执行、灵活好调试;
**Graph(静态图)** —— 先编译整图、再整体下沉执行、性能高。二者是『灵活 vs 性能』的取舍。
下方**拖动算子数与迭代次数**,看两种模式的累积耗时如何变化;再观察编译一次的成本
怎么被摊薄。
""")

mode = st.sidebar.radio("执行模式", ["PyNative 动态图", "Graph 静态图"])
ops = st.sidebar.slider("图中算子个数", 5, 100, 30, 5)
iters = st.sidebar.slider("推理/训练轮数", 10, 500, 100, 10)
show_cost = st.sidebar.checkbox("显示编译成本明细", value=True)
st.sidebar.caption("动态图每轮都要逐算子解释;静态图只编译一次,之后整体执行。")

eager_per_iter = ops * 8.0        # 示意:动态图每轮耗时(μs/算子 → 累加)
static_per_iter = 15.0            # 示意:静态图每轮整体执行
compile_ms = 50.0                 # 示意:静态图一次编译成本(ms)

t_eager_total = eager_per_iter * iters / 1000     # ms
t_static_total = compile_ms + static_per_iter * iters / 1000

c1, c2, c3 = st.columns(3)
c1.metric("动态图总耗时", f"{t_eager_total:.1f} ms")
c2.metric("静态图总耗时", f"{t_static_total:.1f} ms")
c3.metric("静态图省时", f"{(t_eager_total - t_static_total) / max(t_eager_total, 1e-9) * 100:.0f}%")

rng = np.arange(1, iters + 1)
eager_curve = eager_per_iter * rng / 1000
static_curve = compile_ms + static_per_iter * rng / 1000

fig = go.Figure()
fig.add_trace(go.Scatter(x=rng, y=eager_curve, mode="lines",
                         line=dict(color="#E45756", width=3), name="PyNative 动态图"))
fig.add_trace(go.Scatter(x=rng, y=static_curve, mode="lines",
                         line=dict(color="#4C78A8", width=3), name="Graph 静态图"))
cross = int(compile_ms / ((eager_per_iter - static_per_iter) / 1000))
if cross > 0:
    fig.add_vline(x=min(cross, iters), line_dash="dash", line_color="#2F3B52",
                  annotation_text=f"交叉点 ≈ {cross} 轮", annotation_position="top right")
fig.update_layout(title=f"累计耗时:{ops} 个算子 × {iters} 轮(示意)",
                  xaxis_title="轮数", yaxis_title="累计耗时(ms)", height=400,
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 静态图早期被编译成本『拖累』,轮数越多反超越明显 —— 所以线上推理/长期训练偏爱静态图。")

if show_cost:
    st.subheader("📋 成本明细")
    st.markdown(f"""
| 项目 | 动态图 | 静态图 |
|---|---|---|
| 每轮耗时(示意) | {eager_per_iter/1000:.3f} ms(逐算子解释) | {static_per_iter/1000:.3f} ms(整体执行) |
| 编译成本 | 0 | {compile_ms} ms(一次性) |
| 调参/调试 | 灵活,可打印中间值 | 需重编译 |
""")

st.markdown("""
> 💡 **一句话**:MindSpore 的 PyNative 适合探索与调试,Graph 模式适合部署与长期运行。
> 企业落地 LLM 推理时几乎总是静态图 —— 这也是昇腾整图下沉的价值所在。
""")
