# -*- coding: utf-8 -*-
# app_42_flash_attn.py — FlashAttention 原理:标准 vs 分块的 HBM 访存/显存对比 ⚡
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="⚡ 42 · FlashAttention 原理", layout="wide")
st.title("⚡ 第 42 课 · FlashAttention 原理:访存才是瓶颈")

st.markdown(r"""
标准的 attention 要把 **打分矩阵 S 和概率矩阵 P(都是 $H 	imes N 	imes N$)** 整体写进显存(HBM)。
当序列 $N$ 变长时,$N^2$ 项呈平方级暴涨,显存和访存量都撑不住。
**FlashAttention** 的思路是:把 $K/V$ 切成小方块,分块读进来边算边丢,**从不整体写出 S/P**,
于是把 $O(N^2)$ 的显存降回 $O(N)$。下方拖动滑块,直观对比两者的 HBM 访存量与显存占用。
""")

def hbm_bytes(N, H, d, db=2):
    # naive:读 Q/K/V 写 O(约 5·N·H·d)+ S/P 各写读两次(4·H·N²);flash:只 Q/K/V 各读一次 + O 写一次
    naive = 5 * N * H * d * db + 4 * H * N * N * db
    flash = 4 * N * H * d * db
    return naive, flash

def mem_bytes(N, H, d, db=2):
    # 峰值显存:naive 额外持有 S 与 P 两个 N×N 张量;flash 从不落 S/P
    naive = (3 * N * H * d + 2 * H * N * N + N * H * d) * db
    flash = (3 * N * H * d + N * H * d) * db
    return naive, flash

with st.sidebar:
    st.header("🎛️ 参数")
    N = st.slider("序列长度 N", 256, 8192, 1024, 128)
    H = st.slider("注意力头数 H", 1, 16, 8, 1)
    d = st.selectbox("头维度 d", [32, 64, 128], index=1)
    db = st.radio("数据类型", ["fp16(2 字节)", "fp32(4 字节)"])
    dtype_bytes = 2 if db.startswith("fp16") else 4
    st.caption("HBM = 显存。naive 的多出部分主要是 S/P 这两个 N×N 大张量的读写。")

nv, fl = hbm_bytes(N, H, d, dtype_bytes)
mv, mf = mem_bytes(N, H, d, dtype_bytes)

c1, c2, c3, c4 = st.columns(4)
c1.metric("naive 访存量", f"{nv/1e6:.2f} MB")
c2.metric("flash 访存量", f"{fl/1e6:.2f} MB")
c3.metric("访存节省", f"{nv/fl:.1f}×")
c4.metric("显存:naive→flash", f"{mv/1e6:.0f}→{mf/1e6:.1f} MB")

Nx = np.arange(256, 8193, 256)
nv_a = [hbm_bytes(n, H, d, dtype_bytes)[0] / 1e6 for n in Nx]
fl_a = [hbm_bytes(n, H, d, dtype_bytes)[1] / 1e6 for n in Nx]
mv_a = [mem_bytes(n, H, d, dtype_bytes)[0] / 1e6 for n in Nx]
mf_a = [mem_bytes(n, H, d, dtype_bytes)[1] / 1e6 for n in Nx]

fig = go.Figure()
fig.add_trace(go.Scatter(x=Nx, y=nv_a, name="naive 访存", mode="lines",
                         line=dict(width=2.5, color="#E45756")))
fig.add_trace(go.Scatter(x=Nx, y=fl_a, name="flash 访存", mode="lines",
                         line=dict(width=2.5, color="#4C78A8")))
fig.add_trace(go.Scatter(x=Nx, y=mv_a, name="naive 显存", mode="lines", dash="dot",
                         line=dict(color="#E45756")))
fig.add_trace(go.Scatter(x=Nx, y=mf_a, name="flash 显存", mode="lines", dash="dot",
                         line=dict(color="#4C78A8")))
fig.update_layout(title=f"HBM 访存量/显存 vs 序列长度(H={H}, d={d})",
                  xaxis_title="序列长度 N", yaxis_title="MB(对数轴)",
                  yaxis_type="log", height=480,
                  legend=dict(orientation="h", y=1.12),
                  margin=dict(l=10, r=10, t=60, b=10))
fig.add_vline(x=N, line_dash="dash", line_color="#B279A2",
              annotation_text=f"N={N}")
st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 观察:naive 的曲线随 N 呈 O(N²) 上扬(斜率大),flash 只有 O(N) 的缓慢上升;"
           "序列越长,两者的差距拉得越大——这就是 FlashAttention 在大模型长上下文里不可替代的原因。")

st.markdown("""
> 💡 **结论**:标准 attention 慢在**把 S/P 大张量写进又读出 HBM**;
> FlashAttention 用分块 + 在线 softmax 把这些中间量留在片上(SRAM),HBM 访存从 $O(N^2)$ 降到 $O(N)$。
> 计算量(FLOPs)其实没变,省的是**访存**这一块。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 7 章 · 第 42 课配套演示")
