# -*- coding: utf-8 -*-
# app_35_parallel_overview.py — DP/TP/PP/EP 分布式并行总览 🗺️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🗺️ 35 · 并行总览", layout="wide")
st.title("🗺️ 第 35 课 · 分布式并行总览:模型太大,拆开一起干")

st.markdown("""
大模型太大,单卡塞不下、算不动,于是把模型**拆开放到多张卡上**一起干。拆法有四种:
- 🧑‍🤝‍🧑 **DP 数据并行**:每卡一份完整模型,各吃各的数据(切「数据」);
- ✂️ **TP 张量并行**:每卡拿一半权重,一起算一个矩阵(切「权重/矩阵」);
- 🏭 **PP 流水线并行**:按层切开,像流水线一样接力(切「层」);
- 🧩 **EP 专家并行**:MoE 模型把不同「专家」放到不同卡(切「专家」)。
""")

def est_weights_gb(params_b, dtype_bytes):
    # 权重显存:参数量 × 每参数字节数
    return params_b * dtype_bytes

def est_kv_gb(layers, kv_heads, head_dim, ctx, batch, dtype_bytes):
    # KV cache 显存:每 token KV 字节 = 2×L×h_kv×d_head×b,再乘 token 总数
    return 2 * layers * kv_heads * head_dim * dtype_bytes * ctx * batch / 1024 ** 3

MODEL_CFG = {
    "Llama-3-8B":    dict(params=8,  layers=32, kv_heads=8,  head_dim=128),
    "Qwen2.5-32B":   dict(params=32, layers=64, kv_heads=8,  head_dim=128),
    "Llama-3-70B":   dict(params=70, layers=80, kv_heads=8,  head_dim=128),
    "Mixtral-8x7B(EP)": dict(params=47, layers=32, kv_heads=8, head_dim=128),
}

with st.sidebar:
    st.header("🎛️ 参数")
    model = st.selectbox("模型", list(MODEL_CFG.keys()), index=2)
    dtype = st.radio("权重精度", ["fp16 (2B)", "int8 (1B)", "int4 (0.5B)"], index=0)
    tp = st.slider("TP 张量并行", 1, 8, 2, 1)
    pp = st.slider("PP 流水线并行", 1, 8, 1, 1)
    batch = st.slider("并发请求数", 1, 64, 8, 1)
    st.caption("并行度 = 用多少张卡一起干;TP/PP 会把权重摊薄到每卡。")

cfg = MODEL_CFG[model]
params_b = cfg["params"]
dtype_bytes = {"fp16 (2B)": 2, "int8 (1B)": 1, "int4 (0.5B)": 0.5}[dtype]
gpus = tp * pp

weights_total = est_weights_gb(params_b, dtype_bytes)
weights_per = weights_total / gpus            # TP/PP 都摊薄权重
kv_total = est_kv_gb(cfg["layers"], cfg["kv_heads"], cfg["head_dim"], 4096, batch, dtype_bytes)
kv_per = kv_total / tp                        # KV 被 TP 摊薄(每个 DP 组各用各的)
per_card = weights_per + kv_per

c1, c2, c3, c4 = st.columns(4)
c1.metric("总权重显存(GB)", f"{weights_total:.0f}")
c2.metric("权重 / 卡(GB)", f"{weights_per:.1f}")
c3.metric("KV / 卡(GB)", f"{kv_per:.1f}")
c4.metric("每卡合计(GB)", f"{per_card:.1f}")

st.subheader("📊 显存摊薄:并行度越高,每卡负担越轻")
gpus_range = list(range(1, 17))
w_per = [weights_total / g for g in gpus_range]
kv_per_g = [kv_total / min(tp, g) if g >= tp else kv_total / g for g in gpus_range]
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"{g} 卡" for g in gpus_range], y=w_per, name="权重/卡",
                     marker_color="#4C78A8"))
fig.add_trace(go.Bar(x=[f"{g} 卡" for g in gpus_range], y=kv_per_g, name="KV/卡",
                     marker_color="#72B7B2"))
fig.update_layout(title="随卡数增加,每卡显存负担下降(权重被 TP/PP 摊薄)",
                  xaxis_title="总卡数", yaxis_title="显存(GB)", height=420,
                  barmode="stack", margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧭 四种并行,各切什么")
st.markdown("""
| 并行 | 切什么 | 通信模式 | 适用场景 |
|------|--------|----------|----------|
| 🧑‍🤝‍🧑 DP | 数据/请求 | AllReduce(梯度) | 训练、推理扩吞吐 |
| ✂️ TP | 权重矩阵 | AllReduce/AllGather(频繁) | 大模型放不下、低延迟 |
| 🏭 PP | 网络层 | 点对点(稀疏) | 超大模型、层间解耦 |
| 🧩 EP | 专家 | 路由 AllToAll | MoE 稀疏模型 |
""")
st.caption("💡 生产环境常 TP×PP×DP 组合使用(第 41 课细讲)。")
st.markdown("""
> 💡 **结论**:单卡放不下→用 TP/PP 摊权重;单卡算不动(吞吐不够)→用 DP 摊数据。
> 并行度不是越高越好,通信会成为新瓶颈。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 35 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
