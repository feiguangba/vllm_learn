"""app_09_fragmentation.py — 内存碎片化交互模拟(分配器仿真)
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_09_fragmentation.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
from real_ops import alloc_frag_sim
st.set_page_config(page_title="连续内存 vs 分页内存 · 碎片化模拟", layout="wide")

st.title("🧩 连续内存的碎片化问题:停车场里的空位为什么用不上?")
st.markdown(
    "把显存想成一个停车场,车辆(请求)**大小不一、随时进出**。\n\n"
    "- **连续分配(contiguous allocation)**:要求一辆车占**一整片相邻车位**。前面的车开走后留下零星空位,"
    "新来的大车塞不进去——这就是**外部碎片(external fragmentation)**。\n"
    "- **分页分配(paged allocation)**:车位被切成**固定大小的格子(block)**,一辆车可停在不相邻的多格中,"
    "只损失最后一格的**内部碎片(internal fragmentation, 至多 block_size-1)**。\n\n"
    "👇 同一随机事件流**公平对比**两种分配器:上方连续、下方分页(块大小可调)。"
)
st.caption("颜色 = 请求编号,浅灰 = 空闲。连续塞不下会**排队/拒绝**(计数),分页只按整块分配、外部碎片几乎为零。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    slots = st.slider("🎟️ 总内存槽位数", 32, 512, 128, step=16)
with c2:
    block_size = st.select_slider("🧱 块大小 block_size", options=[4, 8, 16, 32], value=8)
with c3:
    max_steps = st.slider("⏱️ 模拟步数", 5, 200, 80)
with c4:
    seed = st.selectbox("🎲 随机种子", options=list(range(5)), index=0)

sim = alloc_frag_sim(total_slots=slots, block_size=block_size, steps=max_steps, seed=seed)
stt = sim["state"]
hist = stt["hist"]

step = st.slider("🔍 回放到第几步", 1, max_steps, max_steps) - 1
occ_c = hist["cont"][step]
occ_p = hist["page"][step]

colorscale = [[0.0, "#f5f5f5"], [0.02, "#2c3e50"], [1.0, "#3498db"]]
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=[occ_c], x=list(range(slots)), colorscale=colorscale, zmin=0, zmax=step + 2,
    showscale=False, yaxis="y", name="连续分配器", hovertemplate="槽位 %{x}: 请求 %{z}<extra></extra>"))
fig.add_trace(go.Heatmap(
    z=[occ_p], x=list(range(slots)), colorscale=colorscale, zmin=0, zmax=step + 2,
    showscale=False, yaxis="y2", name="分页分配器", hovertemplate="槽位 %{x}: 请求 %{z}<extra></extra>"))
fig.update_layout(
    title=f"🧩 第 {step + 1} 步的内存布局(上:连续分配 / 下:分页分配,块={block_size})",
    height=260, template="plotly_white",
    xaxis=dict(title="槽位", domain=[0, 1]),
    yaxis=dict(tickvals=[0], ticktext=["连续分配"]),
    yaxis2=dict(tickvals=[0], ticktext=["分页分配"], anchor="x", overlaying="y", side="left"),
    margin=dict(l=90))
st.plotly_chart(fig, width="stretch")

m1, m2, m3, m4 = st.columns(4)
m1.metric("外部碎片率(连续)", f"{hist['ext'][step]*100:.1f}%", help="1 - 最大连续空闲块 / 总空闲")
m2.metric("内部碎片率(分页)", f"{hist['int'][step]*100:.1f}%", help="已分配但未使用的槽位占比(≤ block-1)")
m3.metric("排队/拒绝请求数(连续)", hist["queue"][step])
m4.metric("内存利用率", f"{hist['util'][step]*100:.1f}%")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(range(1, max_steps + 1)), y=hist["ext"], name="外部碎片(连续)",
                          mode="lines+markers", line=dict(color="#e74c3c")))
fig2.add_trace(go.Scatter(x=list(range(1, max_steps + 1)), y=hist["int"], name="内部碎片(分页)",
                          mode="lines+markers", line=dict(color="#2ecc71")))
fig2.update_layout(title="📈 两类碎片率随时间变化(同事件流公平对比)", xaxis_title="步数", yaxis_title="碎片率",
                    template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, width="stretch")
st.markdown("### 🏷️ 关键结论")
peak_ext = max(hist["ext"]) * 100
st.success(
    f"同一事件流下,连续分配器累计 **{hist['queue'][-1]}** 次请求因外部碎片而排队/拒绝,外部碎片率峰值 **{peak_ext:.0f}%**;"
    f"分页分配器外部碎片几乎为零,只付出至多 {block_size}-1 个槽位的内部碎片代价。")
st.markdown(
    "### 🔬 这对 KV Cache 意味着什么?\n"
    "真实生成中,请求长度通常数千 token、生命周期交错,**max_model_len 只能预测上界**。"
    "若按『每请求一整段连续 KV』预留,显存利用率往往只有 **20%~40%**(PagedAttention 论文实测)。"
    "分页方案把分配粒度降到 block,配合**按需分配**,把使用率拉到接近物理上限——"
    "这正是 vLLM 吞吐较 HF/FasterTransformer 提升 2~4 倍的机制之一。")
real_badge(real=False)
terminology([
    ("external fragmentation (外部碎片)", "空闲显存虽多却不连续,新请求找不到足够大的连续空洞。"),
    ("internal fragmentation (内部碎片)", "已分配给某个请求/块、却因粒度未被真正使用的槽位。"),
    ("buddy / free-list allocator", "vLLM 用空闲块双向链表(block_pool.free_block_queue)做 O(1) 取块与归还。"),
    ("max_model_len", "请求序列长度上界,连续方案被迫按它预留,是超额预留的根源。"),
    ("按需分配 (demand paging)", "只在真正需要时才从池中取物理块,避免预留下沉成本。"),
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