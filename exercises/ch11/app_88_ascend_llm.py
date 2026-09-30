# -*- coding: utf-8 -*-
# app_88_ascend_llm.py — 昇腾大模型推理:显存账本与配置 🧮
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🤖 88 · 昇腾大模型推理", layout="wide")
st.title("🤖 第 88 课 · 昇腾大模型推理:显存账本、KV 与连续批处理")

st.markdown("""
昇腾(MindIE / vLLM-Ascend)跑 LLM 的账本,和 GPU 上完全同构:
**模型权重 + KV Cache + 激活/中间量** 三块抢显存。KV Cache 随
`并发 × (输入+输出)长度` 线性增长,所以**连续批处理**与**分页 KV** 缺一不可。
下方配置**模型 / 并发 / 长度 / 位宽**,看显存账本与 KV 占比,并模拟连续批处理如何省显存。
""")

def mem_budget(params, conc, in_len, out_len, bits, d=4096, layers=32):
    w_gb = params * 1e9 * bits / 8 / 1e9
    kv_gb = 2.0 * d * layers * 2.0 / 1e9 * conc * (in_len + out_len) * 0.5
    kv_contig = 2.0 * d * layers * 2.0 / 1e9 * conc * (in_len + out_len)   # 连续预留 ×2
    return w_gb, kv_gb, kv_contig

with st.sidebar:
    st.header("🎛️ 参数")
    params = st.select_slider("模型参数量(B)", options=[7, 13, 32, 70, 130], value=70)
    bits = st.select_slider("权重位宽", options=[16, 8, 4], value=8)
    conc = st.slider("并发请求数", 1, 128, 16, 1)
    in_len = st.slider("平均输入长度(token)", 128, 4096, 1024, 128)
    out_len = st.slider("平均输出长度(token)", 64, 2048, 512, 64)
    st.caption("KV 显存随 并发×(输入+输出) 线性增长;分页让“预留给”变成“按需用”。")

w_gb, kv_gb, kv_contig = mem_budget(params, conc, in_len, out_len, bits)
total = w_gb + kv_gb
c1, c2, c3, c4 = st.columns(4)
c1.metric("权重显存", f"{w_gb:.1f} GB")
c2.metric("KV Cache(分页)", f"{kv_gb:.1f} GB")
c3.metric("KV Cache(连续预留)", f"{kv_contig:.1f} GB")
c4.metric("KV 占权重比例", f"{kv_gb / w_gb * 100:.0f} %")

st.subheader("🥧 显存账本饼图")
fig = go.Figure(go.Pie(labels=["权重", "KV Cache", "激活/中间量"],
                       values=[w_gb, kv_gb, max(total * 0.06, 0.5)],
                       hole=0.42, marker=dict(colors=["#4C78A8", "#E45756", "#F58518"])))
fig.update_layout(title=f"{params}B 模型 · 并发 {conc}:显存构成", height=380,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 并发与长度越大,KV 这块“橙饼”越膨胀——这是服务引擎核心调优的对象。")

st.subheader("📈 KV 显存随长度增长")
L = np.arange(128, in_len + out_len + 1, 128)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=L, y=[2.0*4096*32*2.0/1e9*conc*l*0.5 for l in L],
                          mode="lines", name="分页按需", line=dict(color="#4C78A8", width=3)))
fig2.add_trace(go.Scatter(x=L, y=[2.0*4096*32*2.0/1e9*conc*l for l in L],
                          mode="lines", name="连续预留(×2)", line=dict(color="#E45756", width=3, dash="dot")))
fig2.update_layout(title="KV Cache 显存随序列长度线性增长(分页 vs 连续预留)",
                   xaxis_title="序列长度(token)", yaxis_title="KV 显存(GB)", height=380,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:昇腾跑 LLM 的“显存三件套”= 权重 + KV + 激活。KV 占比随并发与长度暴涨,
> 于是昇腾引擎和 vLLM 一样拥抱**分页 KV + 连续批处理**。配上 FP8/INT8 低比特权重,
> 70B 级模型才能在单机多卡上舒服地跑起来。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 88 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
