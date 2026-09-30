"""app_13_cow_demo.py — Copy-on-Write 写时复制交互模拟
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_13_cow_demo.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
st.set_page_config(page_title="Copy-on-Write 写时复制", layout="wide")

st.title("🖨️ Copy-on-Write:共享的快乐,写入时才付代价")
st.markdown(
    "前缀缓存让多个请求**共享同一批物理块**。但若某个请求想**修改**这块内存(比如采样与邻居出现分歧、"
    "prompt 有细微差异),就面临『这块内存到底归谁』的问题。\n\n"
    "### 🔬 引用计数 + 写时复制\n"
    "每块带 **ref_count(引用计数)**。只读时大家安心共享;某请求要**写入**时:\n"
    "1. 查 `ref_cnt`:若 `>1`,说明被共享 —— **先复制一份新块**给自己,旧块 `ref_cnt -= 1`;\n"
    "2. 若 `=1`(独享),直接就地写,零成本。\n\n"
    "这样**复制成本只在真正发生分歧写入的那一刻**才产生。这也是 OS 的经典语义,被 vLLM 的 "
    "`BlockPool` + COW 机制沿用(见 docs/02 §4)。"
)
st.caption("块编号越大越新;灰色=空闲。事件日志记录每次 COW 或直接写。")

if "state" not in st.session_state:
    st.session_state.state = None

c1, c2, c3 = st.columns(3)
with c1:
    n_req = st.slider("👥 请求数", 2, 6, 3)
with c2:
    n_pref = st.slider("🔗 共享前缀块数", 1, 6, 3)
with c3:
    n_suf = st.slider("🔀 每请求独有后缀块数", 0, 4, 1)

reset = st.button("🔄 重置模拟")

def build(n_req, n_pref, n_suf):
    blocks = []
    for p in range(n_pref):
        blocks.append({"ref": n_req, "owners": list(range(n_req)), "version": 1, "free": False})
    req_tables = {r: list(range(n_pref)) for r in range(n_req)}
    for r in range(n_req):
        for s in range(n_suf):
            blocks.append({"ref": 1, "owners": [r], "version": 1, "free": False})
            req_tables[r].append(len(blocks) - 1)
    return blocks, req_tables

cfg = (n_req, n_pref, n_suf)
if st.session_state.state is None or reset or st.session_state.state.get("cfg") != cfg:
    blocks, req_tables = build(*cfg)
    st.session_state.state = {"blocks": blocks, "tables": req_tables, "log": [], "cfg": cfg}
    st.session_state.state["log"].append(f"初始化: {n_req} 个请求共享 {n_pref} 个前缀块,各自追加 {n_suf} 个后缀块")

state = st.session_state.state
blocks, req_tables, log = state["blocks"], state["tables"], state["log"]

c1, c2 = st.columns([1, 2])
with c1:
    writer = st.selectbox("✍️ 哪个请求要写入?", [f"请求 {r}" for r in range(n_req)])
    r = int(writer.split()[1])
    own_blocks = req_tables[r]
    target = st.selectbox("🎯 写入哪个块?", [f"块 {b} (ref={blocks[b]['ref']})" for b in own_blocks])
    tb = int(target.split()[1])
    do_write = st.button("💥 执行写入")

if do_write:
    b = blocks[tb]
    if b["ref"] > 1:
        new_id = len(blocks)
        blocks.append({"ref": 1, "owners": [r], "version": b["version"] + 1, "free": False})
        b["ref"] -= 1
        b["owners"].remove(r)
        req_tables[r] = [new_id if x == tb else x for x in req_tables[r]]
        log.append(f"🖨️ COW: 请求 {r} 写共享块 {tb}(ref={b['ref']+1}) → 复制为块 {new_id}(ref=1)")
    else:
        b["version"] += 1
        log.append(f"✍️ 直接写: 请求 {r} 写独享块 {tb},版本→{b['version']}")

m1, m2, m3, m4 = st.columns(4)
naive = n_req * (n_pref + n_suf)
m1.metric("当前块总数", f"{len(blocks)}", delta=f"无共享需 {naive} 块")
m2.metric("COW 复制次数", f"{sum(1 for l in log if 'COW' in l)}")
m3.metric("节省的块", f"{max(naive - len(blocks), 0)}", help="共享使总块数远小于各请求独立复制")
m4.metric("平均引用数", f"{sum(b['ref'] for b in blocks)/len(blocks):.2f}")

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=[f"块 {i}" for i in range(len(blocks))],
    y=[b["ref"] for b in blocks],
    marker_color=["#95a5a6" if b["free"] else ("#2ecc71" if b["ref"] > 1 else "#3498db") for b in blocks],
    text=[f"v{b['version']}" for b in blocks], textposition="outside"))
fig1.update_layout(title="📊 每个块的引用计数(绿=共享,蓝=独享)", yaxis_title="ref_count",
                    template="plotly_white", height=300, margin=dict(t=60))
st.plotly_chart(fig1, width="stretch")

z = []
for rr in range(n_req):
    z.append([2 if b in req_tables[rr] else 0 for b in range(len(blocks))])
fig2 = go.Figure(go.Heatmap(
    z=z, x=[f"块 {i}" for i in range(len(blocks))], y=[f"请求 {rr}" for rr in range(n_req)],
    colorscale=[[0.0, "#f5f5f5"], [0.5, "#f5f5f5"], [0.5, "#f39c12"], [1.0, "#2c3e50"]],
    zmin=0, zmax=2, showscale=False,
    hovertemplate="%{y} ↔ 块 %{x}<extra></extra>"))
fig2.update_layout(title="🧩 块所有权矩阵(深色=该请求持有)", height=180 + 40 * n_req,
                    template="plotly_white", xaxis=dict(side="top"), margin=dict(t=60))
st.plotly_chart(fig2, width="stretch")

st.markdown("### 📜 事件日志")
st.code("\n".join(log[-12:]), language="text")
st.markdown("### 🏷️ 关键结论")
st.success(f"前缀块共享使总块数从(无共享){naive} 降到 {len(blocks)},省 {max(naive-len(blocks),0)} 块;"
           "每次写入只复制真正发生分歧的那一块——COW 是前缀缓存能安全落地的基石。")
st.markdown(
    "### 🔬 COW 与『部分命中』\n"
    "前缀命中未必整块一致:当命中边界落在块内部(`prefix_match_unit`),或某个请求要与邻居**分叉**,"
    "vLLM 就对该块做 **COW**——复制出一个 ref=1 的新块,让后续写入只落在它上。"
    "这保证:共享前缀不因一个请求的采样而被迫全量重算,代价只是一次按块的内存拷贝(`cudaMemcpy`)。\n\n"
    "思考:decode 阶段每个请求的**后缀**几乎必然不同,因此后缀块会逐渐全部复制走;"
    "但只要**前缀块永远不改写**(只读共享),它就始终 ref>1、永不复制。"
)
real_badge(real=False)
terminology([
    ("ref_cnt (ref_count)", "块被多少请求引用,COW 与释放都以它为准。见 KVCacheBlock.ref_cnt。"),
    ("Copy-on-Write", "写时才复制被共享的对象;未写前共享是零成本的。OS fork 的经典语义。"),
    ("cudaMemcpy", "COW 触发的实际拷贝动作,按块大小搬运 HKV 数据。"),
    ("null_block", "block_id=0 的占位块,is_null=True,永不缓存/释放。见 docs/02 §2。"),
    ("partial hit → COW", "前缀边界落在块内或分叉时,先复制再写,避免污染他人。"),
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