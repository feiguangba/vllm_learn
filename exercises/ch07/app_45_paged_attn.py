# -*- coding: utf-8 -*-
# app_45_paged_attn.py — PagedAttention:block table 驱动的非连续 KV 访问 📖
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📖 45 · PagedAttention", layout="wide")
st.title("📖 第 45 课 · PagedAttention:block table 驱动的 KV 访问")

st.markdown("""
LLM 生成时,KV Cache 会长长短短,如果给每个请求**连续分配**一段内存,会产生大量碎片和浪费。
**PagedAttention** 借鉴操作系统的**分页**:把 KV 切成固定大小的**物理块**,每个请求维护一张
**block table**(逻辑块 → 物理块),于是内存可以按需、不连续地分配。
下方拖动**请求数 / 块大小**,看 block table 长什么样、以及如何按它收集 KV 再做 attention。
""")

def run(num_req, block_size, total_blocks, d=8):
    rng = np.random.default_rng(7)
    req_blocks = rng.integers(1, 4, num_req)          # 每个请求占用的块数
    slots = np.arange(total_blocks)
    rng.shuffle(slots)                                # 打乱物理块,模拟碎片
    block_table = []
    cursor = 0
    for nb in req_blocks:
        block_table.append(slots[cursor:cursor + nb]) # 每个请求的逻辑块→物理块
        cursor += nb
    max_len = int(req_blocks.max() * block_size)
    return req_blocks, max_len, block_table

with st.sidebar:
    st.header("🎛️ 参数")
    num_req = st.slider("请求数(并发序列)", 1, 8, 4, 1)
    block_size = st.slider("块大小(每块 token 数)", 2, 8, 4, 1)
    total_blocks = st.slider("物理块总数", 8, 32, 16, 1)
    st.caption("请求的 KV 按需占用若干物理块,物理块可能不连续——这就是分页的威力。")

req_blocks, max_len, block_table = run(num_req, block_size, total_blocks)

st.subheader("🧱 Block Table(逻辑块 → 物理块)")
st.write("每个请求占用" + ", ".join(f"Req{i}: {int(nb)} 块" for i, nb in enumerate(req_blocks)))
st.dataframe(
    np.array([[int(v) for v in row] + [-1] * (max_len // block_size - len(row))
              for row in block_table]),
    use_container_width=True)

d = 8
rng = np.random.default_rng(7)
phys_k = rng.normal(0, 1, (total_blocks, block_size, d)).astype(np.float32)  # 物理 KV 池
phys_v = rng.normal(0, 1, (total_blocks, block_size, d)).astype(np.float32)
cont_k = rng.normal(0, 1, (int(req_blocks.sum() * block_size), d)).astype(np.float32)

results = []
for i, blocks in enumerate(block_table):
    nb = int(req_blocks[i])
    K = phys_k[blocks].reshape(nb * block_size, d)     # 按表收集:非连续块拼成连续 KV
    V = phys_v[blocks].reshape(nb * block_size, d)
    q = rng.normal(0, 1, (1, d)).astype(np.float32)
    S = K @ q / (d ** 0.5)
    p = np.exp(S - S.max()); p = p / p.sum()
    o = p @ V
    results.append((int(nb * block_size), o.reshape(-1)))

st.subheader("🔬 按 block table 收集后的 attention 结果")
st.write("每个请求:块数 → token 数 → 输出(前 3 维)")
for i, (ntok, o) in enumerate(results):
    st.write(f"Req{i}: {ntok} tokens → " + "[" + ", ".join(f"{v:.3f}" for v in o[:3]) + ", ...]")

cont_bytes = int(req_blocks.sum() * block_size) * d * 4
page_bytes = sum(nb * block_size for nb in req_blocks) * d * 4
fig = go.Figure()
fig.add_trace(go.Bar(x=["连续分配", "分页(仅用到的)"], y=[cont_bytes, page_bytes],
                     marker_color=["#E45756", "#4C78A8"], text=[cont_bytes, page_bytes],
                     textposition="outside"))
fig.update_layout(title="KV 访存字节对比:连续分配 vs 分页按需", yaxis_title="字节",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 连续分配要为整片预留(可能含空洞),分页只按实际用到的块读写——内存利用率更高。")

st.markdown("""
> 💡 **结论**:PagedAttention 的核心是 **block table**:一张「逻辑块 → 物理块」的映射表,
> 让不连续的物理内存也能被当作连续的逻辑序列访问。torch 用 advanced indexing(`tensor[blocks]`)
> 就能实现“按表收集”,这正是 vLLM 在 `paged_attn.py` 里用专门 kernel 做的事——只是 GPU kernel
> 把收集与 attention 融合在一起,省掉来回搬运。来源:
> [Kwon et al., SOSP 2023 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 7 章 · 第 45 课配套演示")
