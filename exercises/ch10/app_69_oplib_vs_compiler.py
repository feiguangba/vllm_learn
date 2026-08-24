# -*- coding: utf-8 -*-
# app_69_oplib_vs_compiler.py — 算子库 vs 编译器:选择对比 ⚖️
import streamlit as st
import plotly.graph_objects as go
import os, json

st.set_page_config(page_title="算子库 vs 编译器 ⚖️", layout="wide")
st.title("⚖️ 第 69 课 · 算子库 vs 编译器:选择对比")

st.markdown("""
写高性能 kernel 有两条路:**算子库**(cuBLAS / cuDNN / CUTLASS,人工精调、覆盖常见算子)和
**JIT 编译器**(Triton / Inductor,自动生成、灵活覆盖任意组合)。下方选一个运算,对比两条路的
延迟与各自优劣,并看何时该选谁。
""")

# ---------------- 实测数据(notebook 生成的 oplib_69.json)----------------
DATA = [
    {"op": "矩阵乘", "lib_ms": 1.220, "triton_ms": 1.167, "inductor_ms": 1.203},
    {"op": "逐元素链", "lib_ms": 2.770, "triton_ms": 0.408, "inductor_ms": 0.410},
    {"op": "softmax", "lib_ms": 1.036, "triton_ms": 0.455, "inductor_ms": 0.455},
]
_j = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oplib_69.json")
if os.path.exists(_j):
    try:
        DATA = json.load(open(_j, encoding="utf-8"))
    except Exception:
        pass

st.sidebar.header("🎛️ 参数")
op = st.sidebar.selectbox("运算", [d["op"] for d in DATA])
metric = st.sidebar.radio("查看指标", ["延迟(ms)", "相对算子库加速比"], horizontal=True)
show_advice = st.sidebar.checkbox("显示选择建议", value=True)
st.sidebar.caption("矩阵乘 cuBLAS 极强(编译器难超越);逐元素/融合类编译器更灵活,往往更快。")

row = next(d for d in DATA if d["op"] == op)
lib, tri, ind = row["lib_ms"], row["triton_ms"], row["inductor_ms"]

c1, c2, c3 = st.columns(3)
c1.metric("算子库(cuBLAS)", f"{lib:.3f} ms")
c2.metric("手写 Triton", f"{tri:.3f} ms")
c3.metric("编译器(Inductor)", f"{ind:.3f} ms")

fig = go.Figure()
if metric == "延迟(ms)":
    fig.add_bar(x=["算子库(cuBLAS)", "手写 Triton", "编译器(Inductor)"],
                y=[lib, tri, ind], marker_color=["#c0392b", "#2ca02c", "#1f77b4"],
                text=[f"{lib:.3f}", f"{tri:.3f}", f"{ind:.3f}"], textposition="outside")
    fig.update_layout(title=f"{op}:三条路的延迟对比", yaxis_title="ms")
else:
    base = min(lib, tri, ind)
    fig.add_bar(x=["算子库(cuBLAS)", "手写 Triton", "编译器(Inductor)"],
                y=[lib / base, tri / base, ind / base], marker_color=["#c0392b", "#2ca02c", "#1f77b4"],
                text=[f"{lib/base:.2f}x", f"{tri/base:.2f}x", f"{ind/base:.2f}x"], textposition="outside")
    fig.update_layout(title=f"{op}:相对最快方案的耗时倍数(越小越快)", yaxis_title="倍数")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

if show_advice:
    st.subheader("🧭 何时选谁")
    if op == "矩阵乘":
        st.success("**矩阵乘 → 优先算子库(cuBLAS)**。它被精调多年、逼近硬件峰值,编译器(经 Triton)通常只能打平或略慢。")
        st.info("若算子形状固定且极端,可手写 Triton/CUTLASS 超越 cuBLAS(见本课实测,tile 调好也能反超)。")
    else:
        st.success("**逐元素 / 融合类 → 优先编译器(Triton/Inductor)**。算子库只覆盖常见算子,自定义组合时编译器自动融合更快。")
        st.info("算子库对『单个常见算子』很稳,但对『你的特殊组合』无能为力 —— 这正是编译器的用武之地。")

st.markdown("""
> 💡 **结论**:算子库=『专业厨师』,常见菜又快又好;编译器=『万能厨师』,啥菜都能做、还能帮你
> 把几道菜并成一锅。实际工程常两者混用:常见算子用库,特殊融合用编译器。
""")
