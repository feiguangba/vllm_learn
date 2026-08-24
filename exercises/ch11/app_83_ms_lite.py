# -*- coding: utf-8 -*-
# app_83_ms_lite.py — MindSpore Lite 端侧推理:量化与资源账本 📱
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📱 83 · MindSpore Lite 端侧推理", layout="wide")
st.title("📱 第 83 课 · MindSpore Lite 端侧推理:量化换小体积、换速度")

st.markdown("""
手机 / 摄像头 / 盒子上的算力只有云端的零头,所以 **MindSpore Lite** 有一套“省字诀”:
`converter_lite` 转换 → **算子裁剪**(只留用到的)→ **量化**(FP16 / INT8 减体积)。
下方选择**精度档位**、拖动**模型规模**与**端侧算力**,看端侧推理的**时延、体积与内存**账本,
并实时对比 INT8 量化带来的“精度损失 → 速度收益”。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    params = st.select_slider("模型参数量(MB, FP32)", options=[10, 50, 100, 250, 500], value=100)
    prec = st.radio("精度档位", ["FP32", "FP16", "INT8"])
    tops = st.slider("端侧算力(TOPS, INT8)", 1, 40, 8, 1)
    inputs = st.slider("单次推理输入 token 数", 16, 512, 64, 16)
    st.caption("量化用精度换体积与速度;端侧算力越高,INT8 收益越明显。")

ratio = {"FP32": 1.0, "FP16": 0.5, "INT8": 0.25}[prec]
size_mb = params * ratio
acc = {"FP32": 100.0, "FP16": 99.7, "INT8": 97.5}[prec]      # 示意精度
tops_eff = tops if prec == "INT8" else tops / (8.0 if prec == "FP16" else 16.0)
flops = 2.0 * params * 1e6 * inputs
lat_ms = flops / (tops_eff * 1e12) * 1e3 * 3.0                # 粗估
mem_mb = size_mb * 2.2

c1, c2, c3, c4 = st.columns(4)
c1.metric("模型体积", f"{size_mb:.1f} MB", f"{(1-ratio)*100:.0f}%")
c2.metric("单次推理时延", f"{lat_ms:.2f} ms")
c3.metric("峰值内存", f"{mem_mb:.1f} MB")
c4.metric("示意精度", f"{acc:.1f} %")

st.subheader("🔢 精度-体积-速度三向权衡")
fig = go.Figure()
modes = ["FP32", "FP16", "INT8"]
vol = [params * r for r in (1.0, 0.5, 0.25)]
spd = [flops / (tops / 16.0) / 1e9, flops / (tops / 8.0) / 1e9, flops / tops / 1e9]  # 相对耗时
fig.add_trace(go.Bar(x=modes, y=vol, name="体积(MB)", marker_color="#4C78A8", text=[f"{v:.0f}" for v in vol], textposition="outside"))
fig.add_trace(go.Bar(x=modes, y=spd, name="估算耗时(µs)", marker_color="#F58518", text=[f"{v:.0f}" for v in spd], textposition="outside"))
fig.update_layout(barmode="group", title=f"{params} MB 模型的量化账本", yaxis_title="数值",
                  height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ FP16 体积减半、INT8 体积减到 1/4;同时 INT8 在端侧 NPU 上往往还能用上 INT8 专用算力,速度再上一层。")

st.subheader("📉 INT8 对称量化的误差 vs 数据分布")
dist = st.select_slider("激活分布(σ 相对尺度)", options=[0.1, 0.3, 0.5, 1.0, 2.0], value=0.5)
x = np.random.default_rng(0).normal(0, dist, 20000)
q = np.clip(np.round(x / dist * 127), -127, 127)
err = np.abs(q - x).mean()
s = np.linspace(0, 5, 400)
y = np.exp(-s * s / 2)
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=x, nbinsx=60, name="原始分布", marker_color="#4C78A8", opacity=0.75))
fig2.add_trace(go.Scatter(x=s, y=np.exp(-s * s / 2) * 200, mode="lines", name="高斯参考",
                          line=dict(color="#E45756", width=2)))
fig2.update_layout(title=f"激活分布 → INT8 量化平均误差 ≈ {err:.3f}",
                   xaxis_title="数值", yaxis_title="计数", height=360,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("分布越集中(σ 越小),量化误差越小——这就是 PTQ 会先做“按层缩放”校准的原因。")

st.markdown("""
> 💡 **结论**:MindSpore Lite 的三板斧——**转换(converter_lite)+ 算子裁剪 + 量化(PTQ)**——
> 核心都是“用可接受的精度损失换体积与速度”。INT8 常能换来 4 倍体积下降,
> 而分布越集中、校准越好,误差越小。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 83 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
