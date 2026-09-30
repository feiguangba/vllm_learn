# -*- coding: utf-8 -*-
# app_86_cann_oplib.py — CANN 算子库浏览器:类别、融合与对照 🗃️
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🗃️ 86 · CANN 算子库", layout="wide")
st.title("🗃️ 第 86 课 · CANN 算子库与融合:昇腾的“cuDNN/cuBLAS”")

st.markdown("""
**CANN(Compute Architecture for Neural Networks)** 是昇腾的异构计算平台,
地位相当于 CUDA。它内置一套高性能**算子库**(GEMM、卷积、归一化、FlashAttention、通信……),
并支持把多个小算子**融合**成一个大算子,减少 kernel 启动与中间数据搬运。
下方选择**算子类别**,浏览其与 CUDA 生态的**对照关系**,并拖动算子数量观察**融合的收益曲线**。
""")

DATA = {
    "矩阵乘 GEMM": {
        "ops": ["GEMM", "Strided Batch GEMM", "MatMul-Add 融合", "MatMul-GELU 融合"],
        "cuda": ["cuBLAS gemm", "cuBLAS strided_batched_gemm", "Fusion(cuBLASLt)", "Fusion(cuDNN)"],
        "comm": "对标 cuBLAS:矩阵乘最核心的高性能算子",
    },
    "神经网络 NN": {
        "ops": ["Conv2D", "BatchNorm", "LayerNorm", "Softmax", "FlashAttention"],
        "cuda": ["cuDNN conv", "cuDNN batchnorm", "ATen layernorm", "cuDNN softmax", "FlashAttention kernel"],
        "comm": "对标 cuDNN:卷积 / 归一化 / 注意力一族",
    },
    "通信 HCCL": {
        "ops": ["AllReduce", "AllGather", "ReduceScatter", "P2P Send/Recv"],
        "cuda": ["NCCL AllReduce", "NCCL AllGather", "NCCL ReduceScatter", "NCCL P2P"],
        "comm": "对标 NCCL:多卡集合通信(张量并行 / 专家并行的命脉)",
    },
    "运行时 AscendCL": {
        "ops": ["aclrtMalloc", "aclrtLaunch", "aclrtMemcpyAsync", "aclmdlExecute"],
        "cuda": ["cudaMalloc", "cudaLaunchKernel", "cudaMemcpyAsync", "cudaGraphLaunch"],
        "comm": "对标 CUDA Runtime:显存管理、kernel 启动、模型执行",
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    cat = st.selectbox("算子类别", list(DATA.keys()))
    n_ops = st.slider("融合前算子数量", 2, 12, 6, 1)
    data_gb = st.slider("中间数据规模(GB)", 0.5, 16.0, 4.0, 0.5)
    st.caption("融合的核心收益:少启动 kernel + 中间结果不落显存。")

D = DATA[cat]
c1, c2 = st.columns(2)
c1.metric("算子数", len(D["ops"]))
c2.metric("CUDA 对照", D["cuda"][0].split(" ")[0])
st.caption(D["comm"])

st.subheader("🗂️ CANN 算子 ↔ CUDA 生态对照")
st.dataframe(pd.DataFrame({"昇腾(CANN)": D["ops"], "CUDA 生态": D["cuda"]}), width="stretch")

# ---------------- 融合收益曲线 ----------------
launch_ms = 0.05 * n_ops                    # kernel 启动开销
traffic_gb = n_ops * data_gb * 0.4          # 中间搬运(融合后几乎为 0)
fig = go.Figure()
fig.add_trace(go.Bar(x=["非融合\n(逐个算子)", "融合\n(单算子)"],
                     y=[launch_ms + traffic_gb * 2, 0.05 + 0.05 * data_gb],
                     marker_color=["#E45756", "#4C78A8"],
                     text=["高", "低"], textposition="outside"))
fig.update_layout(title=f"融合前后:kernel 启动 + 中间数据搬运(示意)", yaxis_title="相对开销",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 融合省掉的“中间数据落显存再读回”往往是最大的开销——这就是 FlashAttention 快的原因。")

# ---------------- 带宽收益曲线 ----------------
ns = list(range(2, 13))
fig2 = go.Figure(go.Scatter(x=ns, y=[n * data_gb * 0.4 for n in ns], mode="lines+markers",
                            line=dict(color="#F58518", width=3),
                            name="非融合搬运量"))
fig2.add_hline(y=0.05 * data_gb, line_dash="dash", line_color="#4C78A8",
               annotation_text="融合后搬运量")
fig2.update_layout(title="中间数据搬运量随算子数增长(融合后趋近于 0)", xaxis_title="算子数",
                   yaxis_title="搬运量(GB·次)", height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:CANN 不是“昇腾版的 CUDA”这么简单——它同时提供**运行时、算子库、通信库、
> 编译工具链**。理解它最划算的方式,就是与 CUDA 生态**逐个对照**:
> `AscendCL ↔ CUDA Runtime`、`GEMM/NN 算子库 ↔ cuBLAS/cuDNN`、`HCCL ↔ NCCL`、`Ascend C ↔ CUDA C`。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 86 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
