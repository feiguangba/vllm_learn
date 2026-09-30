"""app_11_block_table.py — 多请求并发:block table 与 slot 映射
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_11_block_table.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math
import random
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
st.set_page_config(page_title="Block Table 与 Slot 映射", layout="wide")

st.title("🔗 多请求并发:vLLM 的 Block Table 与 Slot 映射")
st.markdown(
    "vLLM 中每个请求持有自己的 **block table**(逻辑块 → 物理块),由调度器统一分配物理块。\n\n"
    "### 🔬 分配粒度与增长\n"
    "每次 decode 新增 1 个 token,就在最后一块里多占一个 **slot**;块满了再向调度器申请新物理块——"
    "这正是 `BlockTable.num_full_slots` 与 `slot_mapping` 做的事:\n"
    "$$ \\text{slot} = \\text{物理块号} \\times \\text{block\\_size} + \\text{块内偏移} $$\n\n"
    "多个请求各自一张表,物理块池被**并发共享**,碎片极小;请求之间互不干扰。"
)
st.caption("热图 = 物理块被哪个请求占用(灰=空闲);选中请求可查它的 block table 与逐 token slot 映射。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    n_req = st.slider("👥 请求数", 1, 8, 4)
with c2:
    seq_len = st.slider("📜 每请求序列长度", 16, 512, 128, step=16)
with c3:
    block_size = st.select_slider("🧱 块大小 block_size", options=[4, 8, 16, 32], value=16)
with c4:
    n_phys = st.slider("🗃️ 物理块总数", 4, 128, 64)

n_need = math.ceil(seq_len / block_size)
rng = random.Random(7)
free = list(range(n_phys))
tables = []
failed = []
for r in range(n_req):
    if len(free) < n_need:
        failed.append(int(r))
        tables.append([])
        continue
    picks = [free.pop(rng.randrange(len(free))) for _ in range(n_need)]
    tables.append(picks)

used = sum(len(t) for t in tables)
z = [[-1] * n_phys]
for r, t in enumerate(tables):
    for pb in t:
        z[0][pb] = r + 1
fig = go.Figure(go.Heatmap(
    z=z, x=[f"P{i}" for i in range(n_phys)], y=["物理块"],
    colorscale=[[0.0, "#f5f5f5"], [0.02, "#2c3e50"], [1.0, "#2ecc71"]],
    zmin=-1, zmax=n_req + 1, showscale=False,
    hovertemplate="物理块 %{x}: 请求 %{z}<extra></extra>"))
fig.update_layout(title=f"🗃️ 物理块分配热图:共 {used}/{n_phys} 块被使用", height=220,
                  template="plotly_white", xaxis=dict(side="top"), margin=dict(t=70))
st.plotly_chart(fig, width="stretch")

m1, m2, m3, m4 = st.columns(4)
m1.metric("成功请求", f"{n_req - len(failed)}/{n_req}")
m2.metric("已用物理块", f"{used}/{n_phys}", delta=f"利用率 {used/n_phys*100:.0f}%")
m3.metric("内部碎片(全部请求)", f"{max(used*block_size - n_req*seq_len,0)} 槽位", help="每个请求最后一块的空余")
m4.metric("分配失败请求", f"{len(failed)}" if failed else "无", delta="⚠️ 物理块不足" if failed else "✅")

sel = st.selectbox("🔎 查看请求的 Block Table", [f"请求 {i}" for i in range(n_req)])
r = int(sel.split()[1])
if r in failed or not tables[r]:
    st.error(f"请求 {r} 分配失败:物理块不足。请增大物理块总数。")
else:
    tab1, tab2 = st.tabs(["🗺️ Block Table", "🔎 Slot 映射"])
    with tab1:
        bt = pd.DataFrame({
            "逻辑块": [f"L{i}" for i in range(len(tables[r]))],
            "物理块": [f"P{p}" for p in tables[r]],
            "token 区间": [f"{i*block_size+1} ~ {min((i+1)*block_size, seq_len)}" for i in range(len(tables[r]))],
        })
        fig2 = go.Figure(go.Table(
            header=dict(values=bt.columns, fill_color="#2c3e50", font=dict(color="white"), align="center"),
            cells=dict(values=[bt[c] for c in bt.columns], fill_color=[["#eaf2f8"] * len(bt)], align="center")))
        fig2.update_layout(title=f"🗺️ 请求 {r} 的 Block Table", height=100 + 30 * len(bt),
                           template="plotly_white", margin=dict(t=60))
        st.plotly_chart(fig2, width="stretch")
    with tab2:
        slot_rows = []
        for tok in range(seq_len):
            lb, off = divmod(tok, block_size)
            pb = tables[r][lb]
            slot_rows.append({"token": tok + 1, "逻辑块": f"L{lb}", "物理块": f"P{pb}",
                              "块内偏移": off, "slot": pb * block_size + off})
        sdf = pd.DataFrame(slot_rows)
        fig3 = go.Figure(go.Table(
            header=dict(values=list(sdf.columns), fill_color="#2c3e50", font=dict(color="white"), align="center"),
            cells=dict(values=[sdf[c] for c in sdf.columns], fill_color=[["#fef9e7"] * len(sdf)], align="center")))
        fig3.update_layout(title=f"🔎 请求 {r} 的逐 token slot 映射(前 {min(64, seq_len)} 行)", height=420,
                           template="plotly_white", margin=dict(t=60))
        st.plotly_chart(fig3, width="stretch")
        st.caption("slot = 物理块号 × block_size + 块内偏移 —— 即 kv_cache 里 gpu_cache[pb][off] 的真实下标。")
st.markdown("### 🏷️ 关键结论")
st.success(f"{n_req} 个请求共享 {n_phys} 个物理块,各持独立 block table;"
           f"全部请求仅浪费 {max(used*block_size - n_req*seq_len,0)} 个槽位(内部碎片),"
           f"而连续分配在同场景会因碎片拒绝请求。")
real_badge(real=False)
terminology([
    ("num_full_slots", "vLLM BlockTable 属性,一块里已填满的 slot 数,决定何时申请新块。"),
    ("slot_mapping", "逐 token 的物理线性下标,喂给 PagedAttention kernel 做 gather。"),
    ("req_to_blocks[rid]", "V1 中每请求的块表容器(见 docs/02 §3)。"),
    ("block 增长", "decode 每步 block 自动扩容:满则 alloc,不满就地写;避免预分配浪费。"),
    ("并发共享", "物理块池是全局的,多个请求可同时持有不同块,提升显存复用率。"),
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