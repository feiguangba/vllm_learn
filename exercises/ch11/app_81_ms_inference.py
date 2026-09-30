# -*- coding: utf-8 -*-
# app_81_ms_inference.py — MindSpore 推理部署:链路与代价估算 🚀
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚀 81 · MindSpore 推理部署", layout="wide")
st.title("🚀 第 81 课 · MindSpore 推理部署:导出 → 加载 → 推理")

st.markdown("""
**MindSpore(昇思)** 是华为的 AI 框架,推理侧标准链路是:
`训练导出(MindIR)→ 模型转换 → 加载推理`。这和我们熟悉的
`PyTorch 导出(torch.jit / safetensors)→ vLLM 加载 → serve` 是同构的。
下方拖动**模型规模 / 批量 / 输入长度**,估算一条推理链路的**时延构成**与 **KV Cache 显存**,直观体会"模型越大、批量越大、prompt 越长 → 越贵"。
""")

# ---------------- 推理代价模型(与 notebook 一致,均为量级估算) ----------------
def est_latency(params, batch, prompt_tokens, threads, d=4096, model_len=8192):
    p = float(params)                                   # 参数量(十亿)
    fwd_flops = 2.0 * p * 1e9 * batch * prompt_tokens   # prefill 一次前向
    tops = 60.0 * threads / 8.0                         # 等效 NPU 算力(模拟)
    prefill_ms = fwd_flops / (tops * 1e12) * 1e3
    per_tok_ms = (2.0 * p * 1e9 / (tops * 1e12)) * 1e3  # 单 token 解码
    decode_ms = per_tok_ms * batch
    kv_gb = 2.0 * d * 2.0 / 1e9 * prompt_tokens * batch * 0.5
    return prefill_ms, decode_ms, per_tok_ms, kv_gb

with st.sidebar:
    st.header("🎛️ 参数")
    params = st.select_slider("模型参数量(十亿)", options=[0.5, 1.0, 3.0, 7.0, 13.0, 32.0], value=7.0)
    batch = st.slider("批量(batch)", 1, 64, 8, 1)
    prompt_tokens = st.slider("Prompt 长度(token)", 128, 8192, 1024, 128)
    threads = st.slider("等效算力(模拟,个 NPU AICore 单位)", 4, 64, 16, 1)
    st.caption("时延为「量级估算」,用于建立直觉,非真实硬件数据。")

prefill_ms, decode_ms, per_tok_ms, kv_gb = est_latency(params, batch, prompt_tokens, threads)
total = prefill_ms + decode_ms * (prompt_tokens // 32 + 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prefill 时延", f"{prefill_ms:.1f} ms")
c2.metric("单步 Decode 时延", f"{decode_ms:.1f} ms")
c3.metric("估算 KV Cache", f"{kv_gb:.2f} GB")
c4.metric("链路合计(约)", f"{total:.0f} ms")

# ---------------- 时延构成图 ----------------
fig = go.Figure()
fig.add_trace(go.Bar(x=["Prefill", f"Decode ×{prompt_tokens // 32 + 1} 步", "总时延"],
                     y=[prefill_ms, decode_ms * (prompt_tokens // 32 + 1), total],
                     marker_color=["#4C78A8", "#F58518", "#E45756"],
                     text=[f"{prefill_ms:.0f} ms", f"{decode_ms*(prompt_tokens//32+1):.0f} ms", f"{total:.0f} ms"],
                     textposition="outside"))
fig.update_layout(title="推理链路时延构成:Prefill vs Decode", yaxis_title="毫秒(ms)",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ Prompt 越长,Decode 步数越多;批量越大,每步时延越高——两把钳子夹住吞吐。")

# ---------------- 批量 vs 时延曲线 ----------------
bs = np.arange(1, 65)
lat = [est_latency(params, b, prompt_tokens, threads)[1] for b in bs]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=bs, y=lat, mode="lines+markers",
                          name="decode 时延", line=dict(color="#4C78A8", width=3)))
fig2.add_vline(x=batch, line_dash="dash", line_color="#E45756")
fig2.update_layout(title=f"{params:.1f}B 模型:Decode 时延随批量增长", xaxis_title="批量",
                   yaxis_title="单步 decode 时延 (ms)", height=380,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:MindSpore 推理与 vLLM 推理是"同一件事、两套栈":
> 导出/转换负责把训练图固化,加载/推理负责把固化图搬上硬件跑。
> 差别主要在**调度器**:vLLM/MindIE 这类服务引擎还会叠加连续批处理、
> PagedAttention、预分配 KV 池——这些正是第 84/85/88 课的主角。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 81 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
