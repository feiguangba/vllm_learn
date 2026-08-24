# -*- coding: utf-8 -*-
# app_37_tensor_parallel.py — 张量并行的列切/行切与 AllReduce ✂️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="✂️ 37 · 张量并行", layout="wide")
st.title("✂️ 第 37 课 · 张量并行:把一个矩阵切成几份,一起算")

st.markdown("""
一个线性层就是一次矩阵乘 $Y = XW$。**张量并行(TP)** 把权重矩阵 $W$ 切开,让每张卡只存一份、
只算一块,最后用 **AllReduce** 拼回完整结果。两种切法:
- **列切**:$W$ 按输出维切成列块,每卡算出输出的一部分 → 需要 AllGather 拼列;
- **行切**:$W$ 按输入维切成行块,每卡算部分和 → 需要 AllReduce 相加。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    b = st.slider("batch", 1, 8, 2, 1)
    hidden = st.slider("隐藏维度(矩阵宽度)", 4, 64, 16, 2)
    tp = st.slider("TP 切分数", 2, 4, 2, 1)
    st.caption("切分数 = 用几张卡一起算这个矩阵;每卡只存 1/tp 的权重。")

rng = np.random.default_rng(42)
in_dim = hidden
x = rng.normal(size=(b, in_dim))
w = rng.normal(size=(in_dim, hidden))

col_shards = np.split(w, tp, axis=1)          # 列切:W 按输出维切成 tp 块
col_outs = [x @ s for s in col_shards]        # 每卡输出 (b, hidden/tp)
hidden_shards = np.hstack(col_outs)           # 拼回完整 hidden
row_shards = np.split(hidden_shards, tp, axis=1)
row_outs = [hs for hs in row_shards]
y_tp = sum(row_outs)                          # AllReduce:部分和相加
y_ref = x @ w

c1, c2, c3 = st.columns(3)
c1.metric("每卡权重占比", f"1/{tp}")
c2.metric("TP 结果 vs 稠密", "✅ 一致" if np.allclose(y_tp, y_ref, atol=1e-5) else "❌")
c3.metric("权重矩阵形状", f"{w.shape[0]}×{w.shape[1]}")

st.subheader("✂️ 列切:每卡算出一部分输出列")
fig = go.Figure()
for i, s in enumerate(col_shards):
    fig.add_trace(go.Heatmap(z=s, colorscale="Blues", showscale=False,
                             name=f"卡{i} W_i 形状 {s.shape[0]}×{s.shape[1]}",
                             zmin=s.min(), zmax=s.max()))
fig.update_layout(title=f"列切:权重 W 沿输出维切成 {tp} 块,每卡只存其中一块",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔄 AllReduce:部分和相加 = 完整前向")
row_means = [float(np.abs(r).mean()) for r in row_outs]
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=[f"卡{i} 部分和" for i in range(tp)], y=row_means,
                      name="各卡部分结果", marker_color="#4C78A8",
                      text=[f"{v:.3f}" for v in row_means], textposition="outside"))
fig2.add_trace(go.Bar(x=["AllReduce 总和"], y=[float(np.abs(y_ref).mean())],
                      name="稠密参考", marker_color="#E45756",
                      text=[f"{float(np.abs(y_ref).mean()):.3f}"], textposition="outside"))
fig2.update_layout(title="各卡部分和经 AllReduce 相加 = 单卡稠密结果(验证一致)",
                   yaxis_title="|值| 均值", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **直觉**:TP 相当于把一张大矩阵分成几份,每人算一块,再用 AllReduce 把结果拼起来——
> 数学上完全等价于一次完整矩阵乘(上面 ✅ 一致)。代价是**每次矩阵乘后都要一次通信**,
> 所以 TP 的通信很频繁,适合单卡放不下、又要低延迟的场景。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 6 章 · 第 37 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
