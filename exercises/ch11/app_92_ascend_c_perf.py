# -*- coding: utf-8 -*-
# app_92_ascend_c_perf.py — Ascend C 高性能算子:优化项开关对比 ⚡
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Ascend C 算子优化 ⚡", layout="wide")
st.title("⚡ 第 92 课 · Ascend C 高性能算子:优化项开关对比")

st.markdown("""
昇腾 AI Core 里跑一个算子,像一条**流水线厨房**:先"搬料"(DDR → 统一缓冲 UB),
再"切配/掌勺"(Cube/Vector 计算),最后"端菜"(UB → DDR)。优化就是让流水线
**不空等**:能并行就并行、能提前搬就提前搬、能不搬就不搬。
下面用开关逐一"打开"优化项,看耗时和加速比怎么变。
""")

st.sidebar.header("🎛️ 参数")
opts = st.sidebar.multiselect(
    "开启的优化项",
    ["矢量并行(Vectorize)", "双缓冲(Double Buffer)", "内存复用(Buffer Reuse)", "循环展开(Unroll)"],
    default=["矢量并行(Vectorize)", "双缓冲(Double Buffer)"])
n = st.sidebar.slider("数据量(元素数,log10)", 5.0, 8.0, 7.0, 0.25)
mode = st.sidebar.radio("算子类型", ["Vector 加", "Elementwise 乘加"])
st.sidebar.caption("矢量并行消除逐元素循环;双缓冲让搬移与计算重叠;内存复用减少 DDR 往返。")

N = int(10 ** n)
base = 100.0                                          # 基线耗时(相对单位)
# 每项优化带来的"理论节省"比例(示意)
GAIN = {
    "矢量并行(Vectorize)": 0.55,
    "双缓冲(Double Buffer)": 0.30,
    "内存复用(Buffer Reuse)": 0.18,
    "循环展开(Unroll)": 0.12,
}
scale = 1.0
for o in opts:
    scale *= (1 - GAIN[o])
t = base * scale * (N / 10 ** 7) ** 0.9               # 数据量越大耗时越高
speedup = base / max(t, 1e-6)
bandwidth = N * 4 * 2 / (t * 1e-3) / 1e9              # 等效带宽 GB/s(示意)

c1, c2, c3, c4 = st.columns(4)
c1.metric("数据量", f"{N:,} 元素")
c2.metric("估算耗时(相对)", f"{t:.2f}")
c3.metric("相对加速比", f"{speedup:.2f}×")
c4.metric("等效带宽(示意)", f"{bandwidth:.2f} GB/s")

st.subheader("📊 优化项逐个加,耗时瀑布下降")
labels = ["基线"]
vals = [base]
for o in ["矢量并行(Vectorize)", "双缓冲(Double Buffer)", "内存复用(Buffer Reuse)", "循环展开(Unroll)"]:
    on = o in opts
    if vals[-1] > 0:
        nv = vals[-1] * (1 - GAIN[o]) if on else vals[-1]
        vals.append(nv)
    labels.append(("✓ " if on else "✗ ") + o)
fig = go.Figure(go.Waterfall(
    x=labels, y=[base] + [vals[i] - vals[i-1] for i in range(1, len(vals))],
    measure=["absolute"] + ["relative"] * (len(vals) - 1),
    connector=dict(line=dict(color="#888")),
    increasing=dict(marker_color="#27ae60"), decreasing=dict(marker_color="#e74c3c")))
fig.update_layout(title="优化瀑布图:每开一项,耗时下降一截(绿=下降)",
                  yaxis_title="相对耗时", height=380,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🛠️ 流水线视角:计算 vs 搬移重叠")
st.markdown(f"模式:{mode} | 数据量:{N:,} 元素")
if "双缓冲(Double Buffer)" in opts:
    st.markdown("✅ **双缓冲开启**:搬下一块料与算当前块同时进行,流水线几乎不打嗝。")
    overlap = 0.82
else:
    st.markdown("❌ **双缓冲关闭**:算完一块才能搬下一块,搬移时间白白等。")
    overlap = 0.45
stage = go.Figure(go.Bar(
    x=["搬移(DDR↔UB)", "计算(Vector/Cube)", "空等(未优化损失)"],
    y=[30 * (1 - overlap * 0.5), 40, 30 * (1 - overlap)],
    marker_color=["#3498db", "#e67e22", "#e74c3c"]))
stage.update_layout(title="算子耗时组成(示意)", height=300,
                    yaxis_title="耗时占比(示意)", margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(stage, use_container_width=True)

st.markdown("""
> 💡 **结论**:优化不是"开一个开关",而是**组合拳** —— 矢量并行解决"算得慢",
> 双缓冲解决"搬得等",内存复用解决"搬得多"。真实工程里先用 profiler 找到瓶颈再对症下药。
""")
