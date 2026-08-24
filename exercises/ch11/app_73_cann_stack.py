# -*- coding: utf-8 -*-
# app_73_cann_stack.py — CANN 软件栈层次交互浏览 📚
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="CANN 软件栈 📚", layout="wide")
st.title("📚 第 73 课 · CANN 软件栈:层次交互浏览")

st.markdown("""
CANN(昇腾异构计算架构)是昇腾的**软件中枢**,把上层框架(PyTorch / MindSpore)翻译成
昇腾芯片能执行的任务。它分成好几层,每层各司其职。下方**选择一层**,看它的职责、
输入输出与对应用例;再打开对照表,与 CUDA 软件栈逐层对照。
""")

LAYERS = {
    "ACL 应用开发层(昇腾计算语言)": {
        "职责": "面向应用/推理场景的 C/Python API:设备管理、模型加载、推理执行",
        "输入": "离线模型 + 输入数据", "输出": "推理结果",
        "对应用例": "aclInit → aclrtSetDevice → aclmdlLoadFromFile → aclmdlExecute",
        "类比 CUDA": "应用 API + cudaSetDevice / cudaMemcpy",
        "作用": 5,
    },
    "图引擎 GE": {
        "职责": "把计算图做优化与编译:算子融合、内存规划、整图下沉到设备",
        "输入": "框架计算图(MindIR / ONNX 等)", "输出": "可执行图 / 融合后算子",
        "对应用例": "GE 在模型加载阶段自动完成图优化与整图下沉",
        "类比 CUDA": "CUDA Graph 捕获(减少 kernel 启动)",
        "作用": 5,
    },
    "算子层(算子库 + Ascend C)": {
        "职责": "预置 NN 算子库 + Ascend C 算子开发语言,可自研高性能算子",
        "输入": "算子的数学定义 + shape 信息", "输出": "可执行算子实现",
        "对应用例": "调用 conv/relu/softmax 等内置算子,或用 Ascend C 写自定义算子",
        "类比 CUDA": "cuDNN / cuBLAS + 手写 CUDA kernel",
        "作用": 4,
    },
    "运行时 Runtime": {
        "职责": "设备管理、内存管理、任务调度与 Stream 流管理",
        "输入": "可执行算子 / 图任务", "输出": "设备上按流调度的执行",
        "对应用例": "aclrtCreateStream / aclrtSynchronizeStream",
        "类比 CUDA": "CUDA Runtime(cudaStream 等)",
        "作用": 4,
    },
    "驱动 + HCCL": {
        "职责": "驱动对接芯片;HCCL 提供多卡集合通信(AllReduce 等)",
        "输入": "执行指令 / 梯度数据", "输出": "芯片执行 + 多卡协同",
        "对应用例": "分布式训练时的 AllReduce 梯度同步",
        "类比 CUDA": "NVIDIA 驱动 + NCCL",
        "作用": 3,
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    layer = st.selectbox("选择软件栈层次", list(LAYERS.keys()))
    show_bar = st.checkbox("显示各层作用评分图", value=True)
    show_compare = st.checkbox("显示 CANN vs CUDA 对照表", value=True)
    st.caption("CANN 每一层都能在 CUDA 生态里找到影子,但实现与调度方式不同。")

info = LAYERS[layer]
st.subheader(f"🔍 {layer}")
c1, c2 = st.columns(2)
c1.markdown(f"**职责**:{info['职责']}\n\n**输入**:{info['输入']}\n\n**输出**:{info['输出']}")
c2.markdown(f"**对应用例**:`{info['对应用例']}`\n\n**类比 CUDA**:{info['类比 CUDA']}")
st.metric("该层作用(示意 1-5)", info["作用"])

if show_bar:
    st.subheader("📊 各层作用评分(示意)")
    df = pd.DataFrame({"层": list(LAYERS.keys()), "作用": [v["作用"] for v in LAYERS.values()]})
    fig = go.Figure(go.Bar(x=df["作用"], y=df["层"], orientation="h",
                           marker_color="#4C78A8", text=df["作用"], textposition="outside"))
    fig.update_layout(title="CANN 各层次作用评分(示意)", height=380,
                      xaxis_title="作用(1-5)", yaxis_title="", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ 提示:ACL 与 GE 是『开发者接触最多』的两层,算子层是性能调优主战场。")

if show_compare:
    st.subheader("📋 CANN vs CUDA 对照表")
    rows = [{"层次": k, "CANN 对应组件": LAYERS[k]["对应用例"],
             "CUDA 类比": LAYERS[k]["类比 CUDA"]} for k in LAYERS]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("对照是『思想同源、实现各异』:CUDA 是逐 kernel 启动,昇腾更强调整图下沉。")

st.markdown("""
> 💡 **一句话**:CANN 之于昇腾,就像 CUDA 之于 NVIDIA —— 是硬件与框架之间的『总翻译官』。
> 理解 CANN 的层次,就理解了昇腾上所有推理引擎(vLLM / MindIE)的落地路径。
""")
