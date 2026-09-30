# -*- coding: utf-8 -*-
# app_36_allreduce.py — AllReduce 的 ring / tree 通信量对比 🔄
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🔄 36 · AllReduce", layout="wide")
st.title("🔄 第 36 课 · AllReduce:让每张卡都拿到「大家之和」")

st.markdown("""
**AllReduce** 是最常用的集合通信:每张卡各持一份数据,结束后**每张卡都拿到所有卡数据的总和**。
想象 N 个同学各写了一页笔记,要把内容汇总——**每个人都必须拿到完整汇总**。
本页对比两种算法:**ring(环)** 和 **tree(树)**,看它们的通信量与延迟轮数随卡数怎么变。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("卡数 N", 2, 16, 8, 1)
    data_mb = st.slider("每卡数据量(MB)", 1, 512, 64, 1)
    st.caption("卡越多,AllReduce 需要搬运的总字节数越多;数据量越大,耗时越长。")

def ring_rounds(n):
    return 2 * (n - 1)                       # reduce-scatter N-1 步 + allgather N-1 步

def tree_rounds(n):
    return 2 * int(np.ceil(np.log2(n)))      # halving log2 N + doubling log2 N

vol_per_card = 2 * (n - 1) / n * data_mb     # 每卡通信量(MB)
total_vol = vol_per_card * n                 # 全网总通信量(MB)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ring 通信轮数", f"{ring_rounds(n)}")
c2.metric("tree 通信轮数", f"{tree_rounds(n)}")
c3.metric("每卡通信量(MB)", f"{vol_per_card:.1f}")
c4.metric("全网通信量(GB)", f"{total_vol / 1024:.2f}")

st.subheader("🔍 ring vs tree:轮数与通信量")
names = ["ring(环)", "tree(树)"]
rounds = [ring_rounds(n), tree_rounds(n)]
fig = go.Figure()
fig.add_trace(go.Bar(x=names, y=rounds, name="通信轮数",
                     marker_color=["#4C78A8", "#72B7B2"], text=rounds, textposition="outside"))
fig.update_layout(title=f"{n} 卡下两种算法的通信轮数(轮数≈延迟)",
                  xaxis_title="算法", yaxis_title="轮数", height=360,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 随卡数 N 变化:ring 线性涨, tree 对数涨")
ns = list(range(2, 17))
r_ring = [ring_rounds(nn) for nn in ns]
r_tree = [tree_rounds(nn) for nn in ns]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=r_ring, mode="lines+markers", name="ring 轮数",
                          line=dict(color="#4C78A8", width=3)))
fig2.add_trace(go.Scatter(x=ns, y=r_tree, mode="lines+markers", name="tree 轮数",
                          line=dict(color="#E45756", width=3)))
fig2.update_layout(title="卡数越多,ring 的轮数(2(N-1))涨得越快,而 tree 只有 2log2(N)",
                   xaxis_title="卡数 N", yaxis_title="通信轮数", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **直觉**:两种算法的**总字节数相同**(都是 2(N-1)/N × 每卡数据量),
> 差别在**轮数**——ring 是线性的 2(N-1) 轮,tree 是对数的 2log2(N) 轮。
> 卡少时差不多,卡一多 tree 的延迟优势就显现了;但 tree 对树高、负载均衡更敏感。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 36 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
