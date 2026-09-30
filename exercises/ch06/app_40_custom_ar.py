# -*- coding: utf-8 -*-
# app_40_custom_ar.py — vLLM CustomAllreduce 思路:NCCL vs Custom 延迟曲线 ⚡
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="⚡ 40 · CustomAllreduce", layout="wide")
st.title("⚡ 第 40 课 · vLLM 的 CustomAllreduce:小消息用共享内存替掉 NCCL")

st.markdown("""
NCCL 每次 AllReduce 都有**固定的启动/协议开销**(内核启动、握手),当消息很小时,这个固定开销
占比极高,显得「杀鸡用牛刀」。vLLM 的思路:**同一台机器上,用共享内存 + IPC 直接把结果写到
对方的内存里**,省掉那笔固定开销 → 对小消息快得多。本页用一个两段式延迟模型对比两者。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    msg_mb = st.slider("消息大小 log10(MB)", -3.0, 3.0, 0.0, 0.1,
                       help="10^-3 到 10^3 MB,即 1KB 到 1GB")
    n = st.slider("卡数 N", 2, 8, 4, 1)
    st.caption("CustomAllreduce 只适用于**单机多卡**的小消息;跨机或大消息仍走 NCCL。")

msg = 10 ** msg_mb
alpha_c, bw_c = 2e-6, 80e9          # custom: 固定开销小,共享内存带宽高
alpha_n, bw_n = 25e-6, 25e9         # nccl:   固定开销大,PCIe/NVLink 带宽
ring = 2 * (n - 1) / n              # ring 通信量系数
lat_c = (alpha_c + ring * msg * 1e6 / bw_c) * 1e6   # custom 延迟(us)
lat_n = (alpha_n + ring * msg * 1e6 / bw_n) * 1e6   # nccl 延迟(us)

c1, c2, c3, c4 = st.columns(4)
c1.metric("消息大小", f"{msg:.4g} MB")
c2.metric("Custom 延迟", f"{lat_c:.2f} us")
c3.metric("NCCL 延迟", f"{lat_n:.2f} us")
c4.metric("谁更快", "⚡ Custom" if lat_c < lat_n else "NCCL")

st.subheader("📈 消息大小 vs AllReduce 延迟(对数横轴)")
sizes = np.logspace(-3, 3, 200)
lat_c_all = [(alpha_c + ring * s * 1e6 / bw_c) * 1e6 for s in sizes]
lat_n_all = [(alpha_n + ring * s * 1e6 / bw_n) * 1e6 for s in sizes]
fig = go.Figure()
fig.add_trace(go.Scatter(x=sizes, y=lat_c_all, mode="lines", name="CustomAllreduce",
                         line=dict(color="#72B7B2", width=3)))
fig.add_trace(go.Scatter(x=sizes, y=lat_n_all, mode="lines", name="NCCL",
                         line=dict(color="#E45756", width=3)))
fig.update_layout(title="消息越小,Custom 优势越明显;消息一大两者趋同",
                  xaxis_title="消息大小(MB,对数)", yaxis_title="延迟(us)",
                  xaxis_type="log", yaxis_type="log", height=420,
                  legend=dict(orientation="h", y=1.12),
                  margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔢 卡数 N 对延迟的影响(固定 0.1MB)")
ns = list(range(2, 9))
lc = [(alpha_c + 2 * (nn - 1) / nn * 0.1 * 1e6 / bw_c) * 1e6 for nn in ns]
ln = [(alpha_n + 2 * (nn - 1) / nn * 0.1 * 1e6 / bw_n) * 1e6 for nn in ns]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=lc, mode="lines+markers", name="Custom",
                          line=dict(color="#72B7B2", width=3)))
fig2.add_trace(go.Scatter(x=ns, y=ln, mode="lines+markers", name="NCCL",
                          line=dict(color="#E45756", width=3)))
fig2.update_layout(title="0.1MB 消息下,卡数增加对两者延迟的影响",
                   xaxis_title="卡数 N", yaxis_title="延迟(us)", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:小消息(几 KB 到几 MB)下,固定开销主导延迟,CustomAllreduce 快一个量级;
> 大消息下带宽主导,两者差距缩小。vLLM 正是抓住 LLM 推理里大量「小规模张量通信」的场景,
> 用共享内存直写换来显著加速。源码参考:
> `vllm/distributed/device_communicators/custom_all_reduce.py`。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 6 章 · 第 40 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
