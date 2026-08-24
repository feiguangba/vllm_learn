# -*- coding: utf-8 -*-
# app_91_ascend_llama.py — 昇腾上跑 LLaMA:迁移步骤交互助手 🦙
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾上跑 LLaMA 🦙", layout="wide")
st.title("🦙 第 91 课 · 昇腾上跑 LLaMA:模型迁移交互助手")

st.markdown("""
把一个 PyTorch 模型"搬"到昇腾 NPU,就像**搬家**:锅碗瓢盆(算子)得看看新灶台
(昇腾 AI Core)能不能用、布局(内存)得重新规划、火候(精度)得重新校准。
下面按"迁移五步"逐步点开,并拖动精度滑杆,直观感受每一步在做什么。
""")

ROUTES = {
    "torch_npu 适配器": "PyTorch 代码几乎不改,靠 torch_npu 把算子映射到 CANN,最快但算子覆盖受限",
    "MindSpore 重写": "用 MindSpore 重写网络与训练/推理脚本,最彻底,配合图模式性能最佳",
    "vLLM-Ascend 插件": "社区维护的 vLLM 硬件插件,开箱跑 LLM 推理(PagedAttention/连续批处理开箱即用)",
}
STEPS = ["环境搭建", "代码迁移", "权重转换", "算子适配", "精度对齐"]

st.sidebar.header("🎛️ 参数")
route = st.sidebar.selectbox("迁移路线", list(ROUTES.keys()))
step = st.sidebar.radio("当前阶段", STEPS, index=0)
bits = st.sidebar.slider("目标精度(FP16→INT8)", 8, 32, 16, 1)
show_map = st.sidebar.checkbox("展示权重映射表示例", value=True)
st.sidebar.caption("精度越低,余弦相似度损失越大 —— 这就是「精度对齐」要盯的东西。")

# ---------------- 迁移流水线结构图 ----------------
st.subheader("🧭 迁移五步流水线")
st.markdown(f"**当前路线: {route}** —— {ROUTES[route]}")
boxes = STEPS
fig = go.Figure()
for i, name in enumerate(boxes):
    hl = (name == step)
    fig.add_shape(type="rect", x0=i-0.38, x1=i+0.38, y0=0.4, y1=1.0,
                  line=dict(color="#2c3e50", width=1.5),
                  fillcolor="#e74c3c" if hl else "#d5dbdb")
    fig.add_annotation(x=i, y=0.7, text=name, showarrow=False, font=dict(size=11))
    if i < len(boxes)-1:
        fig.add_annotation(x=i+0.5, y=0.7, text="→", showarrow=False, font=dict(size=15))
fig.update_xaxes(range=[-0.6, len(boxes)-0.4], showticklabels=False)
fig.update_yaxes(range=[0, 1.25], showticklabels=False)
fig.update_layout(title="PyTorch 模型 → 昇腾 NPU 的迁移旅程", height=280,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------- 当前阶段讲解 ----------------
STEPS_INFO = {
    "环境搭建": "安装 CANN + torch_npu / MindSpore,配置 npu-smi 可见 NPU;类比安装 CUDA + cuDNN。",
    "代码迁移": "替换张量/设备/算子 API:model.to('npu')、cuda()→npu();MindSpore 则重写 Cell。",
    "权重转换": "把 .pth/.safetensors 权重按名称映射表转成 .ckpt 或 MindIE 能读的格式。",
    "算子适配": "查算子支持矩阵:不支持的算子用 Ascend C 手写或改写为等价组合。",
    "精度对齐": "对比迁移前后逐层输出:余弦相似度 + 最大绝对误差,逼近 1e-3 量级才算对齐。",
}
st.info(f"**{step}**: {STEPS_INFO[step]}")

# ---------------- 精度对齐模拟 ----------------
sim = np.clip(1.0 - (32 - bits) * 0.018 - 0.002, 0.3, 1.0)      # bits 越低相似度越低
err = 10 ** (-(sim * 7 + 0.5))                                   # 误差随相似度指数下降
c1, c2, c3, c4 = st.columns(4)
c1.metric("迁移目标精度", f"FP{bits}" if bits == 32 else f"INT{bits}")
c2.metric("余弦相似度", f"{sim:.4f}")
c3.metric("最大绝对误差(示意)", f"{err:.2e}")
c4.metric("待手写算子数(示意)", int(max(0, 8 - (bits - 8) * 1.5)))

st.subheader("📈 精度 vs 位宽:对齐曲线的直觉")
b = np.arange(8, 33)
sims = np.clip(1.0 - (32 - b) * 0.018 - 0.002, 0.3, 1.0)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=b, y=sims, mode="lines+markers",
                          line=dict(color="#c0392b", width=3), name="余弦相似度"))
fig2.add_vline(x=bits, line_dash="dash", line_color="#888")
fig2.update_layout(title="量化位数 → 输出相似度(模拟曲线,真实值因模型/数据而异)",
                   xaxis_title="位宽(bits)", yaxis_title="余弦相似度",
                   height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

if show_map:
    st.subheader("🔤 权重映射表示例(PyTorch → MindSpore)")
    rows = [
        ["transformer.wte.weight", "backbone.embedding.weight"],
        ["transformer.h.0.attn.c_attn.weight", "backbone.layers.0.attention.qkv.weight"],
        ["transformer.h.0.mlp.c_fc.weight", "backbone.layers.0.mlp.dense1.weight"],
        ["lm_head.weight", "head.weight"],
    ]
    st.dataframe({"PyTorch 名": [r[0] for r in rows],
                  "MindSpore 名": [r[1] for r in rows]},
                 use_container_width=True)
    st.caption("权重转换的本质:按映射表把 state_dict 的 key 改名 + 重新分块,张量数值原样搬移。")

st.markdown("""
> 💡 **结论**:迁移不是"一行代码的事",而是一条五步流水线。用 torch_npu 最快、
> 用 MindSpore 最彻底、用 vLLM-Ascend 最省心 —— 生产推理优先考虑第三条路。
""")
