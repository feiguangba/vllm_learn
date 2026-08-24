# -*- coding: utf-8 -*-
# app_84_ascend_vllm.py — 昇腾推理引擎 vs vLLM:吞吐与显存对比 ⚖️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="⚖️ 84 · 昇腾引擎 vs vLLM", layout="wide")
st.title("⚖️ 第 84 课 · 昇腾推理引擎 vs vLLM:同一目标,两套栈")

st.markdown("""
GPU 上有 vLLM,昇腾上有什么?答案是两条路:
**vLLM-Ascend**(vLLM 官方插件,开源,复用 vLLM 的调度框架)
和 **MindIE**(华为自研高性能推理引擎,对标 TensorRT-LLM)。
两者都做**连续批处理 + PagedAttention**,只是实现栈不同。
下方拖动**并发数 / 平均输出长度 / 显存总量**,对比三种“引擎形态”的**吞吐与显存占用**。
""")

def calc(conc, out_len, gpu_gb, prefill=512):
    kv_per_req = 2.0 * 4096 * 2.0 / 1e9 * (prefill + out_len) * 0.5   # GB/请求
    # 三种引擎:批效率(单位算力产出 token/s)示意
    engine_eff = {"vLLM (CUDA)": 100, "vLLM-Ascend": 92, "MindIE (昇腾)": 95}
    names = list(engine_eff)
    kv_total = kv_per_req * conc
    thr = [eff * conc * 6.0 / max(out_len, 1) for eff in engine_eff.values()]
    mem_ratio = kv_total / gpu_gb * 100
    return names, thr, kv_total, mem_ratio

with st.sidebar:
    st.header("🎛️ 参数")
    conc = st.slider("并发请求数", 1, 256, 32, 1)
    out_len = st.slider("平均输出长度(token)", 64, 2048, 512, 64)
    gpu_gb = st.slider("显存总量(GB)", 16, 128, 64, 16)
    st.caption("吞吐为“示意效率”,用于直观对比,非真实 benchmark。")

names, thr, kv_total, mem_ratio = calc(conc, out_len, gpu_gb)
c1, c2, c3 = st.columns(3)
c1.metric("KV Cache 总量", f"{kv_total:.2f} GB")
c2.metric("KV 占显存比例", f"{mem_ratio:.1f} %", "可超 100% → 需分页/置换")
c3.metric("输出 token/批", f"{conc * out_len // 1:.0f}")

st.subheader("⚡ 三种引擎形态:吞吐对比")
fig = go.Figure(go.Bar(x=names, y=thr, marker_color=["#4C78A8", "#F58518", "#E45756"],
                       text=[f"{t:.0f}" for t in thr], textposition="outside"))
fig.update_layout(title=f"并发 {conc} · 输出 {out_len} token:吞吐对比(示意)", yaxis_title="吞吐(相对单位)",
                  height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 三者的差距来自算子实现与图优化策略,但都建立在“连续批处理 + 分页 KV”这两个地基上。")

st.subheader("📈 吞吐 vs 并发")
cs = np.arange(1, conc + 1)
lines = []
for name, eff in engine_eff.items():
    y = [eff * c * 6.0 / max(out_len, 1) for c in cs]
    fig2 = go.Figure() if name == list(engine_eff)[0] else fig2
    if name == list(engine_eff)[0]:
        fig2.add_trace(go.Scatter(x=cs, y=y, mode="lines+markers", name=name,
                                  line=dict(color="#4C78A8", width=3)))
    else:
        fig2.add_trace(go.Scatter(x=cs, y=y, mode="lines", name=name,
                                  line=dict(color="#F58518" if "Ascend" in name else "#E45756", width=3)))
fig2.add_vline(x=conc, line_dash="dash", line_color="#333")
fig2.update_layout(title="吞吐随并发增长(线性理想模型)", xaxis_title="并发请求数",
                   yaxis_title="吞吐(相对单位)", height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("现实中吞吐不会无限线性上涨——超过算力/显存后开始排队,这就是调度器的战场。")

st.markdown("""
> 💡 **结论**:**vLLM-Ascend** 让你把在 GPU 上写的 vLLM 代码“原样”搬到昇腾;
> **MindIE** 则是一套为昇腾深度定制的独立引擎。选谁,取决于你要“生态兼容”还是“榨干算力”。
> 两者共享同一套现代推理心法:连续批处理 + PagedAttention(第 85/88 课)。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 84 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
