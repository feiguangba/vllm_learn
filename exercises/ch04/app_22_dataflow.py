# -*- coding: utf-8 -*-
"""🚰 app_22_dataflow.py — 模型前向数据流浏览器(minivllm 第 4 章 · 第 22 课)

运行: streamlit run app_22_dataflow.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚰 22 · 模型前向数据流", layout="wide")
st.title("🚰 第 22 课配套 App · 模型前向数据流浏览器")
st.markdown(
    "以一台「迷你 GPT」(隐层 256、2 层 Transformer、4 头注意力、词表 5000)为例,把 `ModelRunner.execute_model`"
    "的前向过程拆成 **8 步**。每一步都有一组输入/输出张量,形状随 **批大小 × 序列长度** 变化。"
    "本 App 让你任意调参并逐步骤查看中间张量的形状与显存占用。"
)

H, N_HEADS, HEAD_DIM, VOCAB = 256, 4, 64, 5000
DTYPE_BYTES = 4


def dataflow_shapes(batch: int, seq: int, mode: str) -> pd.DataFrame:
    if mode == "prefill":
        tokens, kv_len = batch * seq, seq
    else:
        tokens, kv_len = batch, seq - 1
    steps = [
        ("1 输入组装", "input_ids", (tokens,), "拼接后的词元流 [T]"),
        ("1 输入组装", "positions", (tokens,), "每个词元的绝对位置 [T]"),
        ("1 输入组装", "seq_lens", (batch,), "各序列长度 [B]"),
        ("2 词嵌入", "hidden_states", (tokens, H), "embedding 输出 [T, H]"),
        ("3 QKV 投影", "qkv_proj", (tokens, 3 * H), "fused QKV 线性投影 [T, 3H]"),
        ("3 QKV 投影", "q/k/v 分头", (tokens, N_HEADS, HEAD_DIM), "reshape 成多头 [T, heads, head_dim]"),
        ("4 KV Cache 读写", "kv_cache 写入", (tokens, N_HEADS, HEAD_DIM), "新 KV 写入 cache,旧 KV 共 kv_len 个位置"),
        ("5 注意力", "attn_scores", (tokens, N_HEADS, kv_len), f"q·k^T,历史长度 kv_len={kv_len}"),
        ("5 注意力", "attn_output", (tokens, N_HEADS, HEAD_DIM), "softmax 加权求和"),
        ("6 MLP", "mlp_gate_up", (tokens, 4 * H), "gate/up 双路线性 [T, 4H]"),
        ("6 MLP", "mlp_down", (tokens, H), "down 线性投影 [T, H]"),
        ("7 残差与输出", "logits", (tokens, VOCAB), "LM Head 输出 [T, 词表]"),
        ("8 采样", "next_tokens", (batch,), "每序列采样出一个新词元 [B]"),
    ]
    out = []
    for step, name, shape, note in steps:
        n = int(np.prod(shape))
        out.append({"步骤": step, "张量": name, "形状": str(shape), "元素数": n,
                    "显存(KB)": round(n * DTYPE_BYTES / 1024, 1), "说明": note})
    return pd.DataFrame(out)


st.sidebar.header("🎛️ 输入配置")
batch = st.sidebar.slider("批大小 batch(B)", 1, 64, 8)
seq = st.sidebar.slider("序列长度 seq(S)", 2, 256, 32)
mode_label = st.sidebar.radio("解码阶段", ["prefill", "decode"], index=0)
mode = "prefill" if mode_label == "prefill" else "decode"
step_filter = st.sidebar.selectbox(
    "选择步骤", ["全部", "1 输入组装", "2 词嵌入", "3 QKV 投影", "4 KV Cache 读写",
                "5 注意力", "6 MLP", "7 残差与输出", "8 采样"])
log_scale = st.sidebar.checkbox("显存用对数坐标", value=True)

df = dataflow_shapes(batch, seq, mode)
total_kb = df["显存(KB)"].sum()
peak_row = df.loc[df["显存(KB)"].idxmax()]
tokens_now = batch * seq if mode == "prefill" else batch

c1, c2, c3, c4 = st.columns(4)
c1.metric("当前批总词元数 T", f"{tokens_now}")
c2.metric("全部中间张量显存", f"{total_kb:.0f} KB")
c3.metric("最大张量", f"{peak_row['张量']} · {peak_row['显存(KB)']:.0f} KB")
c4.metric("阶段", "prefill" if mode == "prefill" else f"decode(每序列 1 新词元)")

st.markdown("### 🏭 数据流全景:每步张量显存占用")
st.caption("prefill 阶段一口气处理整段输入,张量巨大;decode 阶段每序列只有 1 个新词元,张量小而注意力随 kv_len 增长。")
fig = px.bar(
    df, x="张量", y="显存(KB)", color="步骤", text="显存(KB)",
    hover_data=["形状", "说明"], log_y=log_scale)
fig.update_layout(height=430, xaxis_tickangle=-30)
st.plotly_chart(fig, width="stretch")

if step_filter != "全部":
    sub = df[df["步骤"] == step_filter]
else:
    sub = df
st.markdown(f"### 🔍 步骤明细: `{step_filter}`")
st.caption("形状随批大小与序列长度实时重算 —— 这正是 vLLM 每次 execute_model 前都要'组装输入'的原因。")
st.dataframe(sub, width="stretch", hide_index=True)

st.markdown("---")
st.markdown(
    "💡 **直觉**:整条前向像一条流水线 🚰,词元流从「输入组装」进入,经过词嵌入、QKV 投影、KV Cache 读写、"
    "注意力、MLP、残差输出,最后在「采样」端吐出新词元。每台机器的水位(张量大小)都取决于批大小与序列长度。"
    "vLLM 的 `ModelRunner.execute_model` 做的就是:组装输入 → 循环各层 → 采样输出。"
)
st.caption("《minivllm: 图解 vLLM 推理引擎》第 4 章 · 第 22 课配套演示")

# ============ PyCharm / 直接运行入口 ============
# 说明: 在 PyCharm 里直接 Run 本文件,即可启动 Streamlit 服务(浏览器打开 http://localhost:8501)。
# 与命令行 `streamlit run app_XX.py` 完全等价(用子进程方式, 避免与顶层 st 调用冲突)。
if __name__ == "__main__":
    # 如果已在 Streamlit Runtime 中(AppTest/嵌入时加载),不再启动子进程。
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
