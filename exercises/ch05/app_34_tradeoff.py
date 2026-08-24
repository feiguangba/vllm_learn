# -*- coding: utf-8 -*-
# app_34_tradeoff.py — 精度/速度/显存三维权衡 + 量化方法选择器 📊
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="📊 34 · 量化综合权衡", layout="wide")
st.title("📊 第 34 课 · 精度-速度-显存:量化方法的三体问题")

st.markdown("""
没有「最好」的量化方法,只有「最合适」的:
- 🎯 **精度**:量化后输出与 fp16 基线的吻合度;
- ⚡ **速度**:内核是否被硬件原生加速(GPU int8/fp8 tensor core);
- 💾 **显存**:每参数字节数,决定能装多大的模型、留多少 KV 空间。

下方选择场景与关注权重,实时看各方法的三维雷达图与对比表:
""")

METHODS = {
    "fp16 基线(16 bit)": dict(bits=16, precision=10, speed=6, memory=4,
        note="全精度基线;H100/B200 上有 fp16 tensor core 加速,显存开销大"),
    "GPTQ int4": dict(bits=4, precision=8, speed=9, memory=9,
        note="逐列量化+误差补偿;Marlin 内核极快,4 bit 显存 1/4,校准需反向信息"),
    "AWQ int4": dict(bits=4, precision=8, speed=9, memory=9,
        note="激活感知缩放保护;只需前向校准,稳定不易过拟合,与 GPTQ 精度接近"),
    "FP8 (E4M3) 权重+KV": dict(bits=8, precision=9, speed=10, memory=6,
        note="H100+ 原生 fp8 tensor core;精度几乎无损,显存减半,是训练推理一体的新宠"),
    "int8 权重+KV": dict(bits=8, precision=9, speed=8, memory=6,
        note="W8A8 方案(SmoothQuant 类);适配面广,Ampere 及以前卡的主力"),
    "int8 KV cache(权重 fp16)": dict(bits=16, precision=9, speed=7, memory=5,
        note="只压 KV 不压权重;吞吐和并发提升,权重显存不变"),
    "RTN int4(朴素)": dict(bits=4, precision=5, speed=9, memory=9,
        note="无需校准、一行代码;低位宽下精度明显掉,胜在省事"),
}

with st.sidebar:
    st.header("🎛️ 选择与权重")
    sel = st.multiselect("对比方法(勾选 2-4 个)", list(METHODS.keys()),
                         default=["fp16 基线(16 bit)", "GPTQ int4", "FP8 (E4M3) 权重+KV"])
    w_acc = st.slider("你在乎精度的程度(0-10)", 0, 10, 7)
    w_spd = st.slider("你在乎速度的程度(0-10)", 0, 10, 6)
    w_mem = st.slider("你在乎显存的程度(0-10)", 0, 10, 8)
    st.caption("💡 想想你的瓶颈:装不下模型 → 优先显存;并发低 → 优先速度;评测掉分 → 优先精度。")

if len(sel) < 1:
    st.info("请至少选择一个方法")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
avg = lambda k: float(np.mean([METHODS[m][k] for m in sel]))
c1.metric("所选方法平均精度分", f"{avg('precision'):.1f} / 10")
c2.metric("平均速度分", f"{avg('speed'):.1f} / 10")
c3.metric("平均显存分", f"{avg('memory'):.1f} / 10")

total_w = w_acc + w_spd + w_mem
scores = {m: (METHODS[m]["precision"] * w_acc + METHODS[m]["speed"] * w_spd
              + METHODS[m]["memory"] * w_mem) / max(total_w, 1) for m in sel}
best = max(scores, key=scores.get)
c4.metric("按你的权重推荐", best, f"{scores[best]:.1f} 分")

st.subheader("📡 三维雷达图:精度 / 速度 / 显存")
cats = ["精度", "速度", "显存友好"]
fig = go.Figure()
for m in sel:
    vals = [METHODS[m]["precision"], METHODS[m]["speed"], METHODS[m]["memory"]]
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1],
                                  fill="toself", name=f"{m}({METHODS[m]['bits']}b)",
                                  opacity=0.6))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], visible=True)),
                  title="每个方法的三维画像:没有全能冠军,只有场景赢家",
                  height=520, margin=dict(l=60, r=60, t=70, b=40))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 方法对比表")
rows = [dict(方法=m, 位宽=METHODS[m]["bits"], 精度分=METHODS[m]["precision"],
             速度分=METHODS[m]["speed"], 显存分=METHODS[m]["memory"],
             一句话点评=METHODS[m]["note"]) for m in sel]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("📈 帕累托视图:显存 vs 精度")
fig2 = go.Figure()
for m in sel:
    bytes_per_param = METHODS[m]["bits"] / 16
    fig2.add_trace(go.Scatter(x=[bytes_per_param], y=[METHODS[m]["precision"]],
                              mode="markers+text", name=m, text=[f"{METHODS[m]['bits']}b"],
                              textposition="top center",
                              marker=dict(size=14, opacity=0.8)))
fig2.update_layout(title="每参数字节数(相对 fp16)vs 精度分:左上角 = 帕累托最优区",
                   xaxis_title="显存占用(相对 fp16 倍数)", yaxis_title="精度分",
                   yaxis_range=[0, 11], height=400, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **选择口诀**:装不下 → GPTQ/AWQ int4;有 H100+ 且要吞吐 → FP8;
> 求稳少折腾 → int8(W8A8);只救 KV 显存 → KV FP8/int8;调试期 → fp16 基线。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 34 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
