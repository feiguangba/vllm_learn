"""app_10_paged_demo.py — 虚拟块 → 物理块映射可视化
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_10_paged_demo.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
from real_ops import paged_map
st.set_page_config(page_title="PagedAttention 分页映射", layout="wide")

st.title("📖 PagedAttention 核心:像操作系统的虚拟内存一样管理 KV")
st.markdown(
    "连续内存被切成**固定大小的小块(block)**,默认 `block_size=16`,每块装 16 个 token 的 KV。\n\n"
    "### 🔬 双重地址空间\n"
    "- **逻辑地址(虚拟块)**:模型眼里序列是连续的 L0, L1, L2 …;\n"
    "- **物理地址(物理块)**:真实 KV 散落在任意物理块 P0, P5, P13 …;\n"
    "- **block table(页表)**:一张把逻辑块 → 物理块的映射表,等价于 OS 的页表/页框。\n\n"
    "注意力 kernel 不看逻辑连续性,只按**(块号, 块内偏移)**定位real缓存——就像 CPU 的 MMU 做虚拟→物理地址翻译。"
    "好处:KV 无需连续大块;最后一块可只填一半,内部碎片 ≤ block_size。"
)
st.caption("下方页表展示映射,热图展示物理块被哪些逻辑块占用;任选 token 可看它的 slot 落点。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    seq_len = st.slider("📜 序列长度(token 数)", 1, 512, 100)
with c2:
    block_size = st.select_slider("🧱 块大小 block_size", options=[4, 8, 16, 32, 64], value=16)
with c3:
    n_phys = st.slider("🗃️ 物理块总数", 1, 64, 16)
with c4:
    view_token = st.slider("🔍 查看 token 位置", 1, max(seq_len, 1), min(seq_len, 1), 1)

data = paged_map(seq=seq_len, block_size=block_size, n_phys=n_phys)
mapping = data["mapping"]
n_logical = data["n_logical"]
miss = data["miss"]

rows = []
for lb, pb in mapping.items():
    start = lb * block_size + 1
    end = min((lb + 1) * block_size, seq_len)
    rows.append({"逻辑块": f"L{lb}", "物理块": f"P{pb}",
                 "token 区间": f"{start} ~ {end}", "槽位数": end - start + 1})
df = pd.DataFrame(rows)

m1, m2, m3, m4 = st.columns(4)
m1.metric("逻辑块数", f"{n_logical}", help="需 n_logical×block_size 槽位")
m2.metric("物理块数", f"{n_phys}")
m3.metric("已映射物理块", f"{len(mapping)}", delta=f"空闲 {n_phys - len(mapping)}")
m4.metric("内部碎片", f"{len(mapping)*block_size - seq_len} 槽位", help="最后一块未用槽位 ≤ block_size-1")

tab1, tab2 = st.tabs(["🗺️ 页表映射", "🔎 Slot 落点"])
with tab1:
    fig1 = go.Figure(go.Table(
        header=dict(values=["逻辑块", "物理块", "token 区间", "槽位数"],
                    fill_color="#2c3e50", font=dict(color="white"), align="center"),
        cells=dict(values=[df["逻辑块"], df["物理块"], df["token 区间"], df["槽位数"]],
                   fill_color=[["#eaf2f8"] * len(df)], align="center")))
    fig1.update_layout(title="🗺️ Block Table(页表):逻辑块 → 物理块", height=80 + 30 * len(df),
                       template="plotly_white", margin=dict(t=60))
    st.plotly_chart(fig1, width="stretch")
    if miss:
        st.error(f"⚠️ 物理块不足!{len(miss)} 个逻辑块无法分配:{[f'L{m}' for m in miss]}。增大物理块总数或减小序列长度。")

with tab2:
    tok = view_token - 1
    lb, off = divmod(tok, block_size)
    pb = mapping.get(lb)
    if pb is None:
        st.error("该 token 所在逻辑块未映射到物理块。")
    else:
        slot = pb * block_size + off
        st.markdown(f"**address translation**: token **{view_token}** → 逻辑块 **L{lb}** → 物理块 **P{pb}** "
                    f"→ 块内偏移 **{off}** → `slot = P×block_size + off = {pb}×{block_size}+{off} = {slot}`")
        z = [[mapping.get(l, -1) for l in range(n_logical)]]
        fig2 = go.Figure(go.Heatmap(
            z=z, x=[f"L{l}" for l in range(n_logical)], y=["物理块占用(值=物理块号)"],
            colorscale=[[0.0, "#f5f5f5"], [0.03, "#2c3e50"], [1.0, "#27ae60"]],
            zmin=-1, zmax=n_phys, showscale=False,
            hovertemplate="逻辑块 %{x}: 物理块 %{z}<extra></extra>"))
        fig2.update_layout(title="🧩 逻辑块 → 物理块 热图", height=200, template="plotly_white",
                           xaxis=dict(side="top"), margin=dict(t=60))
        st.plotly_chart(fig2, width="stretch")
        st.caption(f"共 {len(mapping)}/{n_logical} 个逻辑块已映射;缺失(浅灰)表示分配失败。")
st.markdown("### 🏷️ 关键结论")
st.success(f"序列 {seq_len} token 切成 {n_logical} 个逻辑块,散落于 {len(mapping)} 个物理块;"
           f"只要页表在手,任意 token 都能 **O(1)** 经 slot = P×B+off 定位——这就是 PagedAttention 的地址翻译。")
real_badge(real=False)
terminology([
    ("virtual vs physical block", "逻辑块连续描述序列,物理块是真实显存,二者经页表解耦——与操作系统分页同构。"),
    ("block table / page table", "逻辑→物理映射;vLLM V1 里对应每个请求的 req_to_blocks[rid]。见 repowiki/02。"),
    ("slot / slot_mapping", "把 (物理块号,块内偏移) 线性化为显存单一下标,是 attention kernel 读写的真正地址。"),
    ("default block_size=16", "CacheConfig.DEFAULT_BLOCK_SIZE;与 half-warp(半个线程束)内处理 token 数对齐,利于 kernel 负载均衡。"),
    ("MMU 类比", "CPU 的地址翻译由 MMU+T LB 完成;vLLM 里由 block table + 指针算术完成。"),
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