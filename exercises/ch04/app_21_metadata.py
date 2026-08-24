# -*- coding: utf-8 -*-
"""🧩 app_21_metadata.py — 变长序列组批量张量组装演示(VLLM_learn 第 4 章 · 第 21 课)

运行: streamlit run app_21_metadata.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧩 21 · 变长序列组批组装", layout="wide")
st.title("🧩 第 21 课配套 App · 变长序列组批量张量组装")
st.markdown(
    "vLLM 的一个 **调度批次(调度批)** 里,每条序列的长度各不相同。推理前,`ModelRunner` 必须把"
    "这些序列'打包'成一组 GPU 张量:`input_ids`(拼接的词元)、`positions`(绝对位置)、"
    "`slot_mapping`(词元在 KV Cache 中的槽位)、`block_table`(每个序列的 KV 块表)与 `seq_lens`(各序列长度)。"
    "本 App 让你亲手调参,观察这些张量是如何组装出来的。"
)

st.sidebar.header("🎛️ 批次参数")
n_seq = st.sidebar.slider("序列数量", 1, 8, 4, help="一个调度批里同时服务的序列条数")
len_min = st.sidebar.slider("最短序列长度(词元)", 2, 16, 4)
len_max = st.sidebar.slider("最长序列长度(词元)", 4, 48, 16)
block_size = st.sidebar.slider("KV 块大小 block_size", 1, 16, 8, help="vLLM 里每块 KV 页容纳的词元数")
vocab = st.sidebar.number_input("词表大小(仅影响 token 数值范围)", 50, 10000, 5000, step=50)
if len_min > len_max:
    st.sidebar.warning("最短长度超过了最长长度,已自动交换")
    len_min, len_max = len_max, len_min
view = st.sidebar.radio("🔍 要细看的张量", ["input_ids", "positions", "slot_mapping", "block_table"])

rng = np.random.default_rng(2026)
seq_lens = rng.integers(len_min, len_max + 1, size=n_seq).astype(int)
total_tokens = int(seq_lens.sum())
max_len = int(seq_lens.max())

token_ids = [rng.integers(0, vocab, size=s) for s in seq_lens]
offsets = np.concatenate([[0], np.cumsum(seq_lens)])[:-1]
blocks_needed = [int(np.ceil(s / block_size)) for s in seq_lens]
block_table = []
free_id = 0
slot_mapping = np.full(total_tokens, -1, dtype=np.int64)
for s in range(n_seq):
    blks = []
    for b in range(blocks_needed[s]):
        blks.append(free_id)
        free_id += 1
    block_table.append(blks)
    base = np.array(blks) * block_size
    for i in range(int(seq_lens[s])):
        slot_mapping[offsets[s] + i] = base[i // block_size] + (i % block_size)

padded = {}
for name, rows in [
    ("input_ids", token_ids),
    ("positions", [np.arange(s) for s in seq_lens]),
    ("slot_mapping", [slot_mapping[offsets[s]:offsets[s] + seq_lens[s]] for s in range(n_seq)]),
]:
    mat = np.full((n_seq, max_len), -1, dtype=np.int64)
    for r, row in enumerate(rows):
        mat[r, : len(row)] = row
    padded[name] = mat
bt_mat = np.full((n_seq, max(blocks_needed)), -1, dtype=np.int64)
for r, row in enumerate(block_table):
    bt_mat[r, : len(row)] = row

c1, c2, c3, c4 = st.columns(4)
c1.metric("总词元数 total_tokens", f"{total_tokens}")
c2.metric("最长序列(决定 padding 宽度)", f"{max_len}")
c3.metric("需要的 KV 块数", f"{free_id}")
waste = 1.0 - total_tokens / (n_seq * max_len) if n_seq * max_len > 0 else 0.0
c4.metric("若按定长 padding 的浪费", f"{waste * 100:.1f}%", help="vLLM 采用紧凑拼接,没有这种浪费")

st.markdown("### 📊 序列长度分布")
st.caption("一个批次里各序列长短不一 —— 这是'变长批'的根源。vLLM 用紧凑拼接,而不是 pad 到等长。")
fig_bar = go.Figure(go.Bar(
    x=[f"序列 {i}" for i in range(n_seq)], y=seq_lens,
    text=seq_lens, textposition="outside",
    marker_color=px.colors.qualitative.Set2,
))
fig_bar.update_layout(height=320, yaxis_title="词元数", xaxis_title="")
st.plotly_chart(fig_bar, width="stretch")

st.markdown(f"### 🔍 张量 `{view}` 组装结果")
st.caption("行 = 序列,列 = 序列内词元位置;值 -1 表示该序列在此位置没有词元(padding 占位,实际推理中不出现)。")
fig_heat = go.Figure(go.Heatmap(
    z=padded[view], colorscale="Viridis", showscale=True,
    x=[f"pos {i}" for i in range(max_len)],
    y=[f"seq {i} (len={seq_lens[i]})" for i in range(n_seq)],
    hovertemplate="序列 %{y}<br>位置 %{x}<br>值 %{z}<extra></extra>",
))
fig_heat.update_layout(height=360)
st.plotly_chart(fig_heat, width="stretch")

if view != "block_table":
    st.markdown("### 📋 分序列明细")
    st.caption("每个序列在拼接张量中的 [start, end) 区间与各自的槽位。start 由 seq_lens 的累加和决定。")
    rows = []
    for s in range(n_seq):
        start, end = offsets[s], offsets[s] + seq_lens[s]
        rows.append({
            "序列": f"seq {s}", "长度": int(seq_lens[s]), "拼接区间": f"[{start}, {end})",
            f"{view}": str(padded[view][s, : seq_lens[s]].tolist()),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")
else:
    st.markdown("### 📋 每序列的 KV 块表")
    st.caption("block_table[seq] 的第 k 项 = 该序列第 k 个 KV 块的物理块号;块内第 i 个词元的槽位 = 块号 × block_size + i。")
    rows = []
    for s in range(n_seq):
        rows.append({
            "序列": f"seq {s}", "长度": int(seq_lens[s]), "块数": blocks_needed[s],
            "block_table": str(block_table[s]),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")

st.markdown("---")
st.markdown(
    "💡 **直觉**:序列像长短不一的纸条,`seq_lens` 记录每张纸条的长度,`input_ids` 是拼起来的词元流,"
    "`positions` 记住每个词元在自己纸条上的位置,`slot_mapping` 则是把词元'寄放'进 KV Cache 的寄存柜号,"
    "`block_table` 是每个序列的柜子号清单。五个张量合在一起,GPU 就能高效地并行处理整批变长序列。"
)
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 4 章 · 第 21 课配套演示")

# ============ PyCharm / 直接运行入口 ============
# 说明: 在 PyCharm 里直接 Run 本文件,即可启动 Streamlit 服务(浏览器打开 http://localhost:8501)。
# 与命令行 `streamlit run app_XX.py` 完全等价(用子进程方式, 避免与顶层 st 调用冲突)。
if __name__ == "__main__":
    # 如果已在 Streamlit Runtime 中(AppTest/嵌入时加载),不再启动子进程。
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
