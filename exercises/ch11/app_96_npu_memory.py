# -*- coding: utf-8 -*-
# app_96_npu_memory.py — 昇腾 NPU 内存与 KV Cache 计算器 💾
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="NPU 内存与 KV Cache 💾", layout="wide")
st.title("💾 第 96 课 · 昇腾 NPU 内存与 KV Cache:显存计算器")

st.markdown("""
KV Cache 就像推理时的「**半成品库存**」:每个请求的键值对都要存下来,供后续生成使用。
它的大小 = `2 × 层数 × kv_heads × head_dim × batch × seq × dtype字节`。
拖一拖下面的参数,实时算出 KV Cache 要吃掉多少显存、在目标设备上占多大比例。
""")

st.sidebar.header("🎛️ 参数")
n_layers = st.sidebar.slider("层数", 8, 96, 32, 1)
n_heads = st.sidebar.slider("KV 头数(kv_heads, GQA)", 1, 32, 8, 1)
head_dim = st.sidebar.slider("head_dim", 32, 256, 128, 16)
batch = st.sidebar.slider("batch(并发请求数)", 1, 128, 16, 1)
seq = st.sidebar.slider("序列长度", 512, 32768, 8192, 512)
dtype = st.sidebar.radio("KV 存储精度", ["fp16/bf16(2B)", "fp8(1B)", "int8(1B)"])
dev = st.sidebar.selectbox("目标设备", ["昇腾 910B(64GB)", "RTX 5060(8GB)", "昇腾 310P(8GB)"])
st.sidebar.caption("KV 精度从 fp16 降到 fp8/int8,显存直接减半 —— 这是推理优化的头号杠杆。")

dbytes = 2 if "fp16" in dtype else 1
kv_gb = 2 * n_layers * n_heads * head_dim * batch * seq * dbytes / 1e9
mem_map = {"昇腾 910B(64GB)": 64, "RTX 5060(8GB)": 8, "昇腾 310P(8GB)": 8}
mem = mem_map[dev]
frac = kv_gb / mem * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("KV Cache 占用", f"{kv_gb:.2f} GB")
c2.metric("设备显存", f"{mem} GB")
c3.metric("显存占比", f"{frac:.1f}%")
c4.metric("每 token KV 量", f"{kv_gb / max(seq*batch, 1):.4f} GB")

st.subheader("📈 KV 占用 vs 序列长度(三种并发)")
seqs = np.arange(512, 32769, 512)
fig = go.Figure()
for b in [1, 16, 64]:
    y = 2 * n_layers * n_heads * head_dim * b * seqs * dbytes / 1e9
    fig.add_trace(go.Scatter(x=seqs, y=y, mode="lines", name=f"batch={b}",
                             line=dict(width=3)))
fig.add_hline(y=mem, line_dash="dash", line_color="#c0392b")
fig.update_layout(title=f"KV Cache 随序列长度线性增长({dev},红线=显存上限)",
                  xaxis_title="序列长度", yaxis_title="KV 占用(GB)",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🥧 推理显存账单(示意分配)")
pieces = ["KV Cache", "模型权重", "计算图/工作区", "碎片与预留"]
vals = [kv_gb, mem * 0.35, mem * 0.1, max(mem * 0.05, 0.1)]
vals[0] = min(vals[0], mem * 0.9)
fig2 = go.Figure(go.Pie(labels=pieces, values=vals, hole=0.45,
                        marker_colors=["#e74c3c", "#2e86c1", "#27ae60", "#95a5a6"]))
fig2.update_layout(title="推理显存构成(示意)", height=340,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:KV Cache 是「线性增长 + 按并发翻倍」的显存大头。三个杠杆最有效:
> ① 精度降到 fp8/int8(省一半);② GQA 减 KV 头数;③ PagedAttention 动态按需分配(不预占)。
""")
