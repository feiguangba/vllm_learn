# -*- coding: utf-8 -*-
# app_48_serve.py — LLM 推理服务与 vllm serve 启动参数 🚀
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚀 48 · 服务部署", layout="wide")
st.title("🚀 第 48 课 · 端到端部署:vllm serve 与 OpenAI 兼容接口")

st.markdown("""
把训练好的大模型"端上桌",需要一个**推理服务**:它常驻内存、等待客户端发来请求,
并返回模型生成的结果。vLLM 提供了现成的入口 `vllm serve`,底层对外暴露的是
**OpenAI 兼容 API**。下方拖一拖启动参数,实时拼出对应的 `vllm serve` 命令,
并估算这条命令需要多少显存来装 KV cache。
""")

# ---------------------------------------------------------------- 模型配置(估算用)
MODELS = {
    "Qwen2-1.5B":  dict(layers=28, kv_heads=2,  head_dim=128, vram_gb=6.0),
    "Qwen2-7B":    dict(layers=28, kv_heads=4,  head_dim=128, vram_gb=16.0),
    "Llama3-8B":   dict(layers=32, kv_heads=8,  head_dim=128, vram_gb=16.0),
    "DeepSeek-R1-7B": dict(layers=32, kv_heads=8, head_dim=128, vram_gb=16.0),
}

with st.sidebar:
    st.header("🎛️ vllm serve 启动参数")
    model = st.selectbox("--model", list(MODELS.keys()))
    served_name = st.text_input("--served-model-name", "qwen2-7b")
    max_len = st.slider("--max-model-len", 1024, 131072, 32768, 1024)
    util = st.slider("--gpu-memory-utilization", 0.5, 1.0, 0.90, 0.01)
    tp = st.selectbox("--tensor-parallel-size", [1, 2, 4])
    quant = st.selectbox("--quantization", ["auto", "awq", "gptq", "fp8", "none"])
    port = st.number_input("--port", 1024, 65535, 8000, 1)
    st.caption("以上参数会拼成一条可复现的 vllm serve 启动命令。")

cfg = MODELS[model]
# KV cache 每 token 占用:2(K+V) × 层数 × KV头 × 头维度 × 2字节
kv_per_token = 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * 2
kv_max_bytes = kv_per_token * max_len

# ---------------------------------------------------------------- 拼出 vllm serve 命令
cmd = (f"vllm serve {model} "
       f"--served-model-name {served_name} "
       f"--max-model-len {max_len} "
       f"--gpu-memory-utilization {util:.2f} "
       f"--tensor-parallel-size {tp} "
       f"--quantization {quant} "
       f"--port {port}")
st.subheader("🧾 生成的 vllm serve 命令")
st.code(cmd, language="bash")

# ---------------------------------------------------------------- 指标
c1, c2, c3, c4 = st.columns(4)
c1.metric("模型", model)
c2.metric("KV cache / token", f"{kv_per_token/1024:.1f} KiB")
c3.metric("最大 KV cache 总量", f"{kv_max_bytes/1024**3:.2f} GiB")
c4.metric("可用显存(估算)", f"{cfg['vram_gb']*util:.1f} / {cfg['vram_gb']:.0f} GiB")
st.caption("⚠️ 若“最大 KV cache 总量”超过“可用显存”,长上下文请求就会因显存不足而失败(OOM)。")

# ---------------------------------------------------------------- 显存 vs max_len 图
st.subheader("📊 KV cache 显存随上下文长度变化")
seq = list(range(1024, max_len + 1, 1024))
kv = [kv_per_token * s / 1024 ** 3 for s in seq]
budget = cfg["vram_gb"] * util
fig = go.Figure()
fig.add_trace(go.Scatter(x=seq, y=kv, mode="lines", name="KV cache 显存",
                         line=dict(width=3, color="#4C78A8")))
fig.add_hline(y=budget, line_dash="dash", line_color="#E45756",
              annotation_text=f"显存预算 {budget:.1f} GiB", annotation_position="top right")
fig.update_layout(title=f"{model} · KV cache 显存 vs max-model-len(max_len={max_len})",
                  xaxis_title="上下文长度 max-model-len", yaxis_title="KV cache 显存 (GiB)",
                  height=420, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("💡 观察:KV cache 随上下文长度**线性**增长。max-model-len 拉得越大,"
           "能同时服务的并发请求越少;拉得太小,长文本会被硬生生截断。")

# ---------------------------------------------------------------- 启动参数表
st.subheader("📋 vllm serve 常用参数速查")
st.dataframe(pd.DataFrame([
    {"参数": "必填", "默认": "-", "说明": "vllm serve"},
    {"参数": "--model", "默认": "-", "说明": "模型名或路径(唯一必填项)"},
    {"参数": "--served-model-name", "默认": "同模型名", "说明": "对外暴露的名字"},
    {"参数": "--max-model-len", "默认": "模型上限", "说明": "最长上下文"},
    {"参数": "--gpu-memory-utilization", "默认": "0.90", "说明": "显存使用比例"},
    {"参数": "--tensor-parallel-size", "默认": "1", "说明": "张量并行卡数"},
    {"参数": "--dtype / --quantization", "默认": "auto", "说明": "精度与量化"},
    {"参数": "--host / --port", "默认": "0.0.0.0:8000", "说明": "监听地址与端口"},
]), use_container_width=True)

st.markdown("""
> 🔗 **真机部署**:以上命令在 **Linux / Docker(WSL2)** 上直接可用。Windows 不原生支持 vLLM,
> 推荐在 WSL2 或 Docker Desktop 里跑官方镜像:
> `docker run --runtime nvidia -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen2-7B-Instruct`
> 详见 [vLLM 官方文档](https://docs.vllm.ai) 与
> [Docker 安装指南](https://docs.docker.com/engine/install/)。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 8 章 · 第 48 课配套演示")
