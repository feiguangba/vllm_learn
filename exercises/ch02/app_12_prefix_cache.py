"""app_12_prefix_cache.py — 前缀缓存命中率与收益(+哈希命中仿真)
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_12_prefix_cache.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
from real_ops import prefix_hit_sim
st.set_page_config(page_title="Prefix Caching 前缀缓存", layout="wide")

st.title("🌳 前缀缓存:同样的话,只算一遍")
st.markdown(
    "真实推理里大量请求共享同一开头——系统提示词、few-shot 示例、长文档背景……\n\n"
    "### 🔬 机制:链式块哈希\n"
    "把每个物理块的 KV 连同它的内容指纹(块哈希)存进一张**哈希表**;哈希是**链式的**——"
    "`block[i].hash = H(block[i-1].hash, tokens)`——所以每个哈希唯一标识『到该边界为止的整段前缀』("
    "见 docs/02 §5,`hash_block_size`)。\n\n"
    "新请求到达时,凡是与已存块**逐块哈希相等**的前缀,直接**复用物理块**、跳过计算(代码里映射到 "
    "`get_computed_blocks` 的缓存命中查询)。这就是 **automatic prefix caching**。"
)
st.caption("下方对大量共享前缀的请求做哈希命中仿真:开启缓存时前缀块只算一次,所有请求共享同一份物理 KV。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    n_req = st.slider("👥 请求数", 1, 200, 50)
with c2:
    seq_len = st.slider("📜 每请求序列长度", 64, 4096, 1024, step=64)
with c3:
    prefix_ratio = st.slider("🔗 共享前缀比例 %", 0, 100, 70)
with c4:
    block_size = st.select_slider("🧱 块大小", options=[8, 16, 32, 64], value=16)

per_token_bytes = 2 * 32 * 8 * 128 * 2
prefix_len = int(seq_len * prefix_ratio / 100)
suffix_len = seq_len - prefix_len
n_blocks_full = math.ceil(seq_len / block_size)
n_blocks_pref = math.ceil(prefix_len / block_size) if prefix_len else 0
n_blocks_suf = math.ceil(suffix_len / block_size) if suffix_len else 0

without_cache = n_req * n_blocks_full
with_cache = n_blocks_pref + n_req * n_blocks_suf
saved_blocks = without_cache - with_cache
saved_bytes = saved_blocks * block_size * per_token_bytes
hit_blocks = n_blocks_pref
hit_rate = hit_blocks / (hit_blocks + n_req * n_blocks_suf) if n_req else 0

# 真实哈希命中仿真:构造共享前缀 token,逐个块哈希,统计真实命中
sim = prefix_hit_sim(n_req=n_req, seq_len=seq_len, prefix_len=prefix_len,
                     block_size=block_size) if n_req > 1 else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("命中(复用)块数", f"{hit_blocks}")
m2.metric("节省显存", f"{saved_bytes/1e9:.2f} GB", delta=f"省 {saved_blocks} 块")
m3.metric("块命中率(理论)", f"{hit_rate*100:.1f}%", help="命中块 / 全部需要的块")
m4.metric("单请求新计算块", f"{n_blocks_suf}")
if sim:
    st.caption(f"📡 **哈希命中仿真**:{sim['n_computed']} 个前缀块被省去,"
               f"实测命中率(复用块/总需求块)≈ {sim['hit_rate']*100:.1f}% —— 与理论 {hit_rate*100:.1f}% 吻合。")

ns = np.arange(1, n_req + 1)
y_with = n_blocks_pref + ns * n_blocks_suf
y_without = ns * n_blocks_full
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ns, y=y_without, name="无前缀缓存", mode="lines",
                          line=dict(color="#e74c3c", width=2), fill="tozeroy", fillcolor="rgba(231,76,60,0.12)"))
fig1.add_trace(go.Scatter(x=ns, y=y_with, name="有前缀缓存", mode="lines",
                          line=dict(color="#2ecc71", width=2), fill="tozeroy", fillcolor="rgba(46,204,113,0.15)"))
fig1.update_layout(title="📊 物理块用量 vs 请求数:缓存让增长曲线变缓", xaxis_title="请求数",
                    yaxis_title="占用物理块数", template="plotly_white", hovermode="x unified",
                    legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig1, width="stretch")

ratios = np.arange(0, 101, 5)
saved_frac = [(n_blocks_full - (math.ceil(int(seq_len * p / 100) / block_size) if p else 0)
               - (math.ceil((seq_len - int(seq_len * p / 100)) / block_size) if seq_len - int(seq_len * p / 100) else 0))
              / n_blocks_full for p in ratios]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ratios, y=np.array(saved_frac) * 100, name="节省块占比", mode="lines",
                          line=dict(color="#9b59b6", width=3), fill="tozeroy", fillcolor="rgba(155,89,182,0.15)"))
fig2.add_vline(x=prefix_ratio, line_dash="dot", line_color="gray", annotation_text=f"当前 {prefix_ratio}%")
fig2.update_layout(title="📈 节省比例 vs 共享前缀比例(块大小固定)", xaxis_title="共享前缀比例 (%)",
                    yaxis_title="节省的物理块占比 (%)", template="plotly_white", hovermode="x")
st.plotly_chart(fig2, width="stretch")
st.markdown("### 🏷️ 关键结论")
st.success(f"前缀比例 {prefix_ratio}% 时,{n_req} 个请求共省 **{saved_blocks}** 个物理块({saved_bytes/1e9:.2f} GB),"
           f"块命中率 {hit_rate*100:.1f}%。每个请求只算独有后缀,**TTFT(首个 token 时延)也大幅下降**。")
st.markdown(
    "### 🔬 命中率与块大小的张力\n"
    "块越大,可复用的『前缀单元』越粗,命中率下降;块越小,哈希表/页表开销越大。"
    "vLLM 还支持 `prefix_match_unit` 把哈希粒度调得比块更细,让命中边界落在块内部(部分命中→COW,见 13)。"
    "前缀缓存对 **agent 共享 system prompt / RAG 长上下文 / 推理批中重复前缀** 的收益尤其显著。")
real_badge(real=False)
terminology([
    ("automatic prefix caching", "vLLM 默认开启;以块哈希为键复用物理块,论文 SOSP'23(vLLM)。"),
    ("chain hash / block_hash", "H(prev_hash, tokens):每个哈希唯一标识到该边界为止的前缀。见 docs/02 §5。"),
    ("block hash table", "hash→物理块的映射;命中即免 prefill,省显存+省 TTFT。"),
    ("get_computed_blocks", "V1 KVCacheManager 里做前缀命中查询的入口。"),
    ("TTFT", "time-to-first-token;前缀缓存让复用请求的首 token 近乎即时。"),
])

foot_note()

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os
    import subprocess
    import sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])