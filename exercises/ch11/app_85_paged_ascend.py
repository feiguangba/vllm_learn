# -*- coding: utf-8 -*-
# app_85_paged_ascend.py — PagedAttention 在昇腾:块管理与碎片率 📖
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📖 85 · PagedAttention 在昇腾", layout="wide")
st.title("📖 第 85 课 · PagedAttention 在昇腾:块大小与碎片率的博弈")

st.markdown("""
昇腾上的 KV Cache 一样用**分页**管理:把 KV 切成固定大小的**块(block)**,每个请求按需占用若干块,
用 **block table** 记录「逻辑块 → 物理块」。块越大,表越短、块内空间越浪费;
块越小,浪费越少、表越长、管理开销越大。下方拖动**序列长度分布 / 块大小**,实时观察
**碎片率、块表大小与内存利用率**——体会昇腾/GPU 都在做的这个平衡。
""")

def run(seq_lens, block_size, total_gb=64.0):
    seq_lens = np.asarray(seq_lens, dtype=np.float64)
    req_blocks = np.ceil(seq_lens / block_size).astype(int)
    wasted = (req_blocks * block_size - seq_lens).sum()
    frag = wasted / (req_blocks * block_size).sum() * 100
    kv_gb = (req_blocks * block_size).sum() * 2.0 * 4096 * 2.0 / 1e9 * 0.5
    return req_blocks, frag, kv_gb

with st.sidebar:
    st.header("🎛️ 参数")
    block_size = st.slider("块大小(token/块)", 1, 32, 8, 1)
    n_req = st.slider("请求数", 10, 500, 100, 10)
    max_len = st.slider("最大序列长度(token)", 128, 4096, 1024, 128)
    skew = st.radio("长度分布", ["均匀", "偏长尾(多数短、少数长)"])
    st.caption("碎片率 = 块内没用到的那部分占已分配块总容量的比例。")

rng = np.random.default_rng(7)
if skew == "均匀":
    seqs = rng.integers(16, max_len, n_req)
else:
    seqs = np.clip(np.random.default_rng(7).exponential(max_len / 3.0, n_req).astype(int), 16, max_len)

req_blocks, frag, kv_gb = run(seqs, block_size)
c1, c2, c3 = st.columns(3)
c1.metric("平均每请求块数", f"{req_blocks.mean():.1f}")
c2.metric("碎片率", f"{frag:.1f} %")
c3.metric("估算 KV 占用", f"{kv_gb:.2f} GB")

st.subheader("🧱 Block Table(逻辑块 → 物理块)")
rows = [req_blocks[i].tolist() for i in range(min(8, n_req))]
max_b = max(len(r) for r in rows)
tbl = np.array([[int(b) for b in r] + [-1] * (max_b - len(r)) for r in rows])
st.dataframe(tbl, width="stretch")
st.caption("每格 = 该请求第 j 个逻辑块对应的物理块编号;-1 = 未分配。物理块来自全局内存池,不保证连续。")

st.subheader("📉 碎片率 vs 块大小")
bs = np.arange(1, 33)
frags = [run(seqs, b)[1] for b in bs]
fig = go.Figure(go.Scatter(x=bs, y=frags, mode="lines+markers", line=dict(color="#E45756", width=3)))
fig.add_vline(x=block_size, line_dash="dash", line_color="#4C78A8")
fig.update_layout(title="块大小越大 → 碎片率越高(曲线单调上升)", xaxis_title="块大小(token)",
                  yaxis_title="碎片率(%)", height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 但块太大不只是坏处:块表更短、算子访存更连续。工业界(昇腾/GPU)默认块大小多为 16~32。")

st.markdown("""
> 💡 **结论**:昇腾的 PagedAttention 与 CUDA 版共享同一套**分页心法**:按需分配、块表寻址、
> 非连续物理块也能当连续序列用。块大小是一个“碎片率 ↔ 管理开销”的工程旋钮——
> 选小了浪费空间,选大了浪费内存。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 85 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
