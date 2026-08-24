# -*- coding: utf-8 -*-
# app_58_triton_vllm.py — 后端选择 + block table 交互示意 🔗
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Triton 与 vLLM 🔗", layout="wide")
st.title("🔗 第 58 课 · Triton 与 vLLM:后端选择与 block table")

st.markdown("""
vLLM 的注意力不是一个“固定实现”,而是一组可以**插拔的后端(backend)**。选不同的后端,
推理时就会调用不同的 kernel。本页模拟 vLLM 的 KV cache 内存布局:切换后端、拖动
**物理块数 / 块大小 / 批大小**,实时观察 KV cache 的形状、显存占用与 block table 的映射。
""")

BACKENDS = {
    "TRITON_ATTN": {
        "kernel": "unified_attention + triton_reshape_and_cache_flash",
        "lang": "Triton(Python 编译到 GPU)",
        "note": "vLLM 自己维护的 triton 后端,支持 FP8 KV cache、任意 16 倍数块大小、滑动窗口与 alibi。",
    },
    "FLASH_ATTN": {
        "kernel": "flash_attn_varlen_func(C++/CUDA)",
        "lang": "flash-attn 扩展库",
        "note": "NVIDIA 手写优化的 FlashAttention,峰值性能高,但功能定制需动 C++。",
    },
    "FLASHINFER": {
        "kernel": "flashinfer 的 paged attention",
        "lang": "C++/CUDA + 自研 JIT",
        "note": "高性能第三方库,专为服务场景优化,支持丰富的 page 模式。",
    },
    "MLA": {
        "kernel": "MLA 专属 kernel(MLA 后端)",
        "lang": "混合(C++/CUDA + Triton)",
        "note": "针对 DeepSeek 系 MLA 架构的专用后端,压缩 KV cache 并复用吸收矩阵。",
    },
}

# ---------------------------------------------------------------- 侧边栏参数
with st.sidebar:
    st.header("🎛️ 参数")
    backend = st.selectbox("Attention 后端", list(BACKENDS.keys()))
    num_blocks = st.slider("物理块总数(num_blocks)", 32, 512, 128, 16)
    block_size = st.selectbox("块大小(block_size, 16 的倍数)", [16, 32, 64])
    num_kv_heads = st.slider("KV 头数(num_kv_heads)", 1, 8, 4, 1)
    head_size = st.selectbox("头维度(head_size)", [64, 128])
    batch = st.slider("并发序列数(batch)", 1, 16, 8, 1)
    st.caption("TRITON_ATTN 要求 block_size 是 16 的倍数;块数决定显存里能存多少 token。")

# ---------------------------------------------------------------- KV cache 形状与显存
# TRITON_ATTN: K/V 打包在最后一维,shape = (num_blocks, num_kv_heads, block_size, 2*head_size)
per_block_tokens = block_size * num_kv_heads
kv_bytes = num_blocks * num_kv_heads * block_size * 2 * head_size * 2   # fp16 = 2 字节
max_tokens = num_blocks * block_size

# 模拟:为 batch 条序列分配物理块(顺序分配,便于可视化)
blocks_per_seq = max(1, num_blocks // max(batch, 1))
table = []
for s in range(batch):
    table.append(list(range(s * blocks_per_seq, min((s + 1) * blocks_per_seq, num_blocks))))

c1, c2, c3, c4 = st.columns(4)
c1.metric("KV cache 张量形状", f"({num_blocks}, {num_kv_heads}, {block_size}, {2 * head_size})")
c2.metric("KV cache 显存(fp16)", f"{kv_bytes / 2 ** 20:.1f} MB")
c3.metric("可缓存 token 总量", max_tokens)
c4.metric("每条序列分到块数", blocks_per_seq)

st.subheader(f"🧠 后端:{backend} —— {BACKENDS[backend]['kernel']}")
st.info(BACKENDS[backend]["note"])

# ---------------------------------------------------------------- KV cache 布局热力图(plotly)
st.subheader("🗺️ KV cache 布局:每个格子是一个 (块, 槽位) 槽")
z = np.zeros((num_blocks, block_size))
colors = ["#4C78A8", "#E45756", "#F2C14E", "#72B7B2", "#76B7B2", "#8C5B9E", "#59A14F", "#B6992D"]
for s in range(batch):
    for b in table[s]:
        z[b, :] = s + 1
fig = go.Figure(go.Heatmap(
    z=z, y=[f"块 {i}" for i in range(num_blocks)], x=[f"槽 {i}" for i in range(block_size)],
    colorscale="Viridis", showscale=False,
    hoverongaps=False))
fig.update_layout(title="x 轴=块内槽位(block_size 个 token),y 轴=物理块;同色 = 同一条序列",
                  height=max(300, num_blocks * 6), margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 观察:每一条序列只占据“整数块”的内存,序列间的空隙不会超过一个块——这就是 paged 思路对抗碎片化的方法。")

# ---------------------------------------------------------------- block table 表格
st.subheader("📇 block table(物理块号映射)")
st.markdown("每条序列一行,第 j 列 = 它逻辑上第 j 页对应的**物理块号**:")
st.write(pd.DataFrame(table, columns=[f"页{j}" for j in range(blocks_per_seq)],
                      index=[f"seq{s}" for s in range(batch)]))

st.markdown("""
> 💡 **读图方法**:vLLM 的 decode 路径正是用这张表做“按页收集”——
> kernel 里 `tl.load(block_table + seq_id * max_blocks + bi)` 逐页读物理块,拼出完整 KV 序列。
> 后端换到 FLASH_ATTN / FLASHINFER 时,只是换了 kernel 和页表格式,思路不变。
""")
