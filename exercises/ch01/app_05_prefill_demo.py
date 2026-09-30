# -*- coding: utf-8 -*-
# app_05_prefill_demo.py — Prefill vs Decode 演示 ⚡  (真实 GPU 数据版)
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
import streamlit as st, plotly.graph_objects as go
from vllm_real import bench_prefill_decode, cuda_info

st.set_page_config(page_title="Prefill vs Decode ⚡", layout="wide")
st.title("⚡ 第 05 课 · Prefill vs Decode:两阶段推理(真实 GPU 实测)")

st.markdown("""
LLM 推理分两个阶段:**prefill**(一次性处理整个提示词,高并行)与 **decode**(逐字生成,串行)。
除了理论 FLOPs,本 app **在同一张 GPU 上真实跑一个小 GPT**,给出两阶段实测的总耗时与吞吐,
让 'decode 更慢' 有真实数字支撑。
""")

def flops_total(N, L, K):
    return 2 * N * (L + K)

with st.sidebar:
    st.header("🎛️ 参数")
    n_b = st.slider("模型参数量(十亿)", 0.5, 70.0, 7.0, 0.5)
    L = st.slider("提示词长度(prefix tokens)", 64, 2048, 512, 32)
    K = st.slider("生成长度(generated tokens)", 64, 2048, 512, 32)
    do_bench = st.button("🚀 在 GPU 实测两阶段", type="primary")
    st.caption(f"检测到: **{cuda_info()}**")

N = n_b * 1e9
fp, fd = flops_total(N, L, 0), flops_total(N, 0, K)

st.subheader("📐 理论计算量账本 (FLOPs ≈ 2N×token)")
c1, c2, c3 = st.columns(3)
c1.metric("prefill FLOPs", f"{fp/1e15:.2f} PFLOPs")
c2.metric("decode FLOPs", f"{fd/1e15:.2f} PFLOPs")
c3.metric("总 FLOPs", f"{(fp+fd)/1e15:.2f} PFLOPs")

if do_bench or "bench" in st.session_state:
    if "bench" not in st.session_state:
        with st.spinner("跑小 GPT 微基准中…"):
            st.session_state["bench"] = bench_prefill_decode(
                d=256, layers=8, L=L, steps=K, reps=5)
    b = st.session_state["bench"]
    st.subheader("🛠️ 真实 GPU 实测(小 GPT 微基准)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("prefill 总耗时", f"{b['prefill_ms']:.2f} ms")
    m2.metric("decode 总耗时(L 步)", f"{b['decode_total_ms']:.2f} ms")
    m3.metric("耗时比 decode/prefill", f"{b['ratio']:.1f} ×")
    m4.metric("decode 吞吐", f"{b['decode_tok_per_s']/1e3:.1f} k tok/s")
    st.caption(f"参数 ~{b['params']/1e6:.1f}M · 设备 {b['device']}。"
               "prefill 一次并行 L token;decode 逐 token 走 L 步。")
else:
    st.info("点击左侧『🚀 在 GPU 实测两阶段』,跑真实微基准得到数字(约几秒)。再次进入本 app 会缓存结果。")
