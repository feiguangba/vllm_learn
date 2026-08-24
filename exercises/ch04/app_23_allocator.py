# -*- coding: utf-8 -*-
"""🗃️ app_23_allocator.py — KV 块分配器模拟器(VLLM_learn 第 4 章 · 第 23 课)

运行: streamlit run app_23_allocator.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="🗃️ 23 · KV 块分配器模拟", layout="wide")
st.title("🗃️ 第 23 课配套 App · KV 块分配器模拟器")
st.markdown(
    "vLLM 用 **块分配器** 管理 KV Cache:空闲块组成 free list,每个块带 **refcount(引用计数)**,"
    "序列申请块、释放块,块可能因'中间空洞'产生 **外部碎片**。本 App 让你亲手分配/释放序列,"
    "实时观察块池状态与碎片率。"
)

if "alloc" not in st.session_state:
    st.session_state.alloc = None
if "history" not in st.session_state:
    st.session_state.history = []


class SimAllocator:
    def __init__(self, num_blocks, block_size, strategy="free_list"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.strategy = strategy
        self.free = list(range(num_blocks))
        self.refcount = [0] * num_blocks
        self.seqs = {}

    def _pick(self, k):
        if self.strategy == "free_list":
            return self.free[:k]
        sizes = []
        run, start = 0, None
        used = set(b for b in range(self.num_blocks) if self.refcount[b] > 0)
        for b in range(self.num_blocks):
            if b not in used:
                if run == 0:
                    start = b
                run += 1
            else:
                if run > 0:
                    sizes.append((run, start))
                run = 0
        if run > 0:
            sizes.append((run, start))
        sizes.sort()
        got = []
        for r, s in sizes:
            for b in range(s, s + min(r, k - len(got))):
                got.append(b)
            if len(got) >= k:
                break
        return got

    def allocate(self, seq_id, n_tokens):
        k = int(np.ceil(n_tokens / self.block_size))
        if k > len(self.free):
            return False
        blocks = self._pick(k)
        for b in blocks:
            self.free.remove(b)
            self.refcount[b] += 1
        self.seqs[seq_id] = blocks
        return True

    def release(self, seq_id):
        if seq_id not in self.seqs:
            return False
        for b in self.seqs.pop(seq_id):
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)
                self.free.sort()
        return True

    def stats(self):
        used = sum(1 for r in self.refcount if r > 0)
        free_blocks = self.num_blocks - used
        runs, run, start = [], 0, None
        for b in range(self.num_blocks):
            if self.refcount[b] == 0:
                if run == 0:
                    start = b
                run += 1
            else:
                if run > 0:
                    runs.append((start, run))
                run = 0
        if run > 0:
            runs.append((start, run))
        largest = max((r for _, r in runs), default=0)
        frag = 1.0 - (largest / free_blocks) if free_blocks > 0 else 0.0
        return dict(used=used, free=free_blocks, runs=runs, largest_run=largest,
                    fragmentation=frag)


with st.sidebar:
    st.header("🎛️ 池配置")
    num_blocks = st.slider("块总数", 16, 256, 48)
    block_size = st.slider("块大小 block_size(词元/块)", 4, 64, 16, step=4)
    strategy = st.radio("分配策略", ["free_list(顺序取最小块号)", "best_fit(优先填小空洞)"])
    st.header("🕹️ 操作")
    seq_tokens = st.number_input("新序列长度(词元)", 1, 4096, 64, step=8)
    if st.button("➕ 分配一个新序列", width="stretch"):
        a = SimAllocator(num_blocks, block_size, "free_list" if "free_list" in strategy else "best_fit")
        st.session_state.alloc = a
        st.session_state.history = []
        sid = f"seq_{len(st.session_state.alloc.seqs)}"
        ok = st.session_state.alloc.allocate(sid, int(seq_tokens))
        st.session_state.history.append(("allocate", sid, int(seq_tokens), ok))
    if st.button("🔄 重置并跑一个典型场景", width="stretch"):
        a = SimAllocator(num_blocks, block_size, "free_list" if "free_list" in strategy else "best_fit")
        st.session_state.alloc = a
        st.session_state.history = []
        for i, ln in enumerate([64, 96, 32, 128]):
            ok = a.allocate(f"seq_{i}", ln)
            st.session_state.history.append(("allocate", f"seq_{i}", ln, ok))
        st.session_state.alloc.release("seq_1")
        st.session_state.history.append(("release", "seq_1", 96, True))

alloc = st.session_state.alloc
if alloc is None:
    alloc = SimAllocator(num_blocks, block_size, "free_list")
    st.session_state.alloc = alloc

s = alloc.stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("已用块 / 总块", f"{s['used']} / {num_blocks}")
c2.metric("空闲块", f"{s['free']}")
c3.metric("最大连续空闲段", f"{s['largest_run']} 块")
c4.metric("外部碎片率", f"{s['fragmentation'] * 100:.1f}%",
          help="碎片率 = 1 − 最大连续空闲段 / 空闲块数,越接近 0 越好")

st.markdown("### 🗂️ 块池状态(颜色 = 引用计数)")
st.caption("每个格子是一个 KV 块。灰 = 空闲;颜色越深 = 被越多序列共享引用(refcount 越大)。")
colors = [alloc.refcount[b] for b in range(num_blocks)]
fig_grid = go.Figure(go.Heatmap(
    z=[colors], colorscale=[[0, "#e0e0e0"], [0.4, "#7fb3d5"], [1, "#1b4f72"]],
    zmin=0, zmax=max(4, max(colors)),
    x=[f"块{b}" for b in range(num_blocks)], y=["池"],
    hovertemplate="块 %{x}<br>refcount=%{z}<extra></extra>",
))
fig_grid.update_layout(height=140, xaxis_tickangle=-60, yaxis_visible=False)
st.plotly_chart(fig_grid, width="stretch")

st.markdown("### 📈 空闲段分布")
st.caption("空闲块按连续段展示:若碎片率 > 0,说明存在'可用但拼不成大段'的块,大序列可能因找不到连续段而分配失败。")
runs = s["runs"]
bar_df = pd.DataFrame({
    "空闲段": [f"段{i}(块{r[0]}~{r[0]+r[1]-1})" for i, r in enumerate(runs)] or ["(无)"],
    "块数": [r[1] for r in runs] or [0],
})
fig_runs = px.bar(bar_df, x="空闲段", y="块数", text="块数")
fig_runs.update_layout(height=300)
st.plotly_chart(fig_runs, width="stretch")

st.markdown("### 📋 当前序列的块分配")
st.caption("每序列占用的块号列表。块可以跨序列共享(refcount > 1),这是 vLLM prefix caching 的基础。")
rows = [{"序列": sid, "长度(词元)": len(b) * alloc.block_size,
         "块数": len(b), "块号": str(b)} for sid, b in alloc.seqs.items()]
if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.info("池里还没有任何序列,点左侧「➕ 分配一个新序列」试试。")

st.markdown("---")
st.markdown(
    "💡 **直觉**:KV Cache 像一间存放柜的仓库 🗃️。free list 是「空柜清单」,refcount 是「一把钥匙几个人在用」。"
    "碎片率衡量的是仓库里「空但零散」的柜子占比 —— 这正是 PagedAttention 相比整段预分配要解决的经典难题。"
)
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 4 章 · 第 23 课配套演示")

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
