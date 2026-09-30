# -*- coding: utf-8 -*-
# app_89_ms_autoparallel.py — MindSpore 自动并行:切分策略与通信量 🧩
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧩 89 · MindSpore 自动并行", layout="wide")
st.title("🧩 第 89 课 · MindSpore 自动并行:切分策略与通信成本")

st.markdown("""
MindSpore 的 **auto_parallel** 会把“切不切、怎么切”交给**策略搜索**:
对矩阵乘等算子的输入/输出按 **行/列/无** 切到多张卡,用 **cost model** 估算通信量,
挑出最优的切分组合。下方选择**切分方式**与**张量大小**,实时看
**每卡算量、通信量与加速比**,直观体会“切得越碎、通信越贵”的权衡。
""")

def gemm_plan(M, N, K, devices, mode):
    if mode == "数据并行(不切)":
        per = M * N * K; comm = 0.0
    elif mode == "输出行切(TP)":
        per = M * (N // devices) * K; comm = N   # 沿 N 切,结果无需通信(每卡持有部分列)
    elif mode == "输出列切(TP2)":
        per = M * N * (K // devices); comm = M * N * 2   # 沿 K 切,需要 AllReduce 合并
    elif mode == "流水线(PP)":
        per = M * N * K / devices; comm = M * N
    speedup = (M * N * K) / max(per, 1) / (1 + comm / max(M * N * K, 1) * 0.3)
    return per, comm, speedup

with st.sidebar:
    st.header("🎛️ 参数")
    M = st.slider("行数 M", 512, 8192, 2048, 512)
    K = st.slider("内维 K", 512, 8192, 4096, 512)
    N = st.slider("列数 N", 512, 8192, 2048, 512)
    devices = st.slider("卡数", 1, 16, 4, 1)
    mode = st.selectbox("切分方式", ["数据并行(不切)", "输出行切(TP)", "输出列切(TP2)", "流水线(PP)"])
    st.caption("速度只算了“通信惩罚”的一阶近似,用于理解趋势,非真实基准。")

per, comm, speedup = gemm_plan(M, N, K, devices, mode)
c1, c2, c3 = st.columns(3)
c1.metric("每卡算量(FLOPs×1e9)", f"{per / 1e9:.1f}")
c2.metric("通信量(元素×1e6)", f"{comm / 1e6:.1f}")
c3.metric("估算加速比", f"{speedup:.2f}×")

st.subheader("⚖️ 通信量 vs 切分方式")
modes = ["数据并行", "输出行切", "输出列切", "流水线"]
comms = [0, N, M * N * 2, M * N]
fig = go.Figure(go.Bar(x=modes, y=[c / 1e6 for c in comms],
                       marker_color=["#4C78A8", "#F58518", "#E45756", "#6B4FA1"],
                       text=[f"{c/1e6:.1f}" for c in comms], textposition="outside"))
fig.update_layout(title="不同切分方式的通信量(元素数,百万)", yaxis_title="通信量×1e6",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 切分方式决定通信量:有的几乎零通信(行切),有的要全量 AllReduce(列切)。")

st.subheader("📈 加速比 vs 卡数")
cs = np.arange(1, devices + 1)
fig2 = go.Figure()
for m in ["数据并行", "输出行切", "输出列切"]:
    ys = [gemm_plan(M, N, K, c, m)[2] for c in cs]
    fig2.add_trace(go.Scatter(x=cs, y=ys, mode="lines+markers", name=m, line=dict(width=3)))
fig2.add_hline(y=cs[-1], line_dash="dash", line_color="#E45756", annotation_text="理想线性")
fig2.update_layout(title="加速比随卡数:切分越好越接近线性,但通信会让曲线弯下来",
                   xaxis_title="卡数", yaxis_title="加速比(×)", height=400,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:MindSpore 的自动并行=让机器替你选切分。它把每个算子的
> 切分方式(**策略**)枚举出来,用 cost model 评估通信,再做全局协调——
> 这比人手写 TP/PP 更不容易出错,也更接近最优。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 89 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
