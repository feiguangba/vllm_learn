# -*- coding: utf-8 -*-
# app_66_torch2backend.py — 从 PyTorch 到后端代码:编译选项交互 🔄
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, json

st.set_page_config(page_title="从 PyTorch 到后端代码 🔄", layout="wide")
st.title("🔄 第 66 课 · 从 PyTorch 到后端代码")

st.markdown("""
一条完整的链路:**PyTorch 模型 → torch.fx 追踪成图(FX Graph)→ Inductor 优化 → 生成 Triton
kernel → 在 GPU 上运行**。下面选择编译后端,看『图 / kernel / 耗时』三个视角怎么随选择变化,
并可以直接查看 Inductor 生成的一段真实 Triton kernel 源码。
""")

# ---------------- 实测数据(notebook 生成的 torch2backend_66.json)----------------
DATA = {
    "backends": ["eager", "aot_eager", "inductor_default", "inductor_maxautotune"],
    "graph_nodes": 9, "fused_kernels": 2, "eager_kernels": 9,
    "eager_ms": 0.12, "default_ms": 0.084, "autotune_ms": 0.085,
    "triton_code": "def triton_poi_fused_add_relu_0(...)\n    # 实际为 Inductor 生成的 Triton 源码,见 notebook\n",
}
_j = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch2backend_66.json")
if os.path.exists(_j):
    try:
        DATA = json.load(open(_j, encoding="utf-8"))
    except Exception:
        pass

st.sidebar.header("🎛️ 参数")
backend = st.sidebar.selectbox("编译后端", DATA["backends"])
show_graph = st.sidebar.checkbox("展示 FX 计算图", value=True)
show_code = st.sidebar.checkbox("展示生成代码", value=False)
show_kernels = st.sidebar.checkbox("展示 kernel 数对比", value=True)
st.sidebar.caption("eager 不编译;aot_eager 只追踪不生成代码;inductor 才真正生成 Triton kernel。")

backend_ms = {"eager": DATA["eager_ms"], "aot_eager": DATA["eager_ms"],
              "inductor_default": DATA["default_ms"], "inductor_maxautotune": DATA["autotune_ms"]}
is_compiled = backend.startswith("inductor")

c1, c2, c3 = st.columns(3)
c1.metric("图节点数", DATA["graph_nodes"])
c2.metric("运行耗时", f"{backend_ms[backend] * 1000:.1f} us")
c3.metric("Triton kernel 数", DATA["fused_kernels"] if is_compiled else DATA["eager_kernels"])
st.caption(f"当前后端:{backend}。{'已生成 Triton kernel(融合后)' if is_compiled else '未做代码生成(eager 直跑或仅追踪)'}")

if show_graph:
    st.subheader("🕸️ FX 计算图(前端产物)")
    g = ("%x : placeholder\n"
         "%w : get_attr\n"
         "%matmul : call_function[torch.matmul](%x, %w)\n"
         "%relu : call_function[torch.relu](%matmul)\n"
         "%add : call_function[operator.add](%relu, 0.1)\n"
         "return add")
    st.code(g, language="text")
    st.caption("Dynamo 把 Python 代码捕获成这张图 —— 优化与代码生成的起点。")

if show_kernels:
    st.subheader("🧬 kernel 数对比")
    fig = go.Figure(go.Bar(x=["eager", "aot_eager", "inductor"], y=[9, 9, 2],
                           marker_color=["#c0392b", "#e67e22", "#27ae60"],
                           text=[9, 9, 2], textposition="outside"))
    fig.update_layout(title="eager 每个算子一个 kernel vs inductor 融合后 2 个", yaxis_title="kernel 数",
                      height=340, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

if show_code:
    st.subheader("📜 Inductor 生成的 Triton kernel 源码(节选)")
    st.code(DATA["triton_code"], language="python")
    st.caption("这正是『后端』的产物:一段可在 GPU 上运行的 Triton 代码 —— 编译器把图翻译成了可执行程序。")

st.markdown("""
> 💡 **结论**:从模型到可运行代码,中间是『追踪成图 → 优化 → 代码生成』三段。backend 选项决定了
> 你停在哪一段:eager 不停(直跑)、aot_eager 停在图、inductor 走完全程生成 Triton kernel。
""")
