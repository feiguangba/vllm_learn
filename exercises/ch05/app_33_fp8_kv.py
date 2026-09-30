# -*- coding: utf-8 -*-
# app_33_fp8_kv.py — FP8 位拆解与 KV Cache 显存计算器 🚀
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚀 33 · FP8 与 KV Cache 显存", layout="wide")
st.title("🚀 第 33 课 · FP8 与 KV Cache:省一半显存的魔法")

st.markdown("""
FP8 用 8 bit 装「科学计数法」:**E4M3**(4 位指数 + 3 位尾数)精度高、范围小,适合权重与前向;
**E5M2**(5 位指数 + 2 位尾数)范围大、精度低,适合梯度与 KV cache。
把 KV cache 从 fp16 换成 fp8,显存直接减半——同样一张卡能装下双倍的并发上下文。

下方左侧调模型配置,右侧实时计算 KV cache 显存:
""")

def fp8_decode(code, exp_bits, mant_bits, exp_bias):
    # 把 8 位整数编码还原成浮点值(含 subnormal)
    sign = (code >> (exp_bits + mant_bits)) & 1
    exp = (code >> mant_bits) & ((1 << exp_bits) - 1)
    mant = code & ((1 << mant_bits) - 1)
    if exp == 0:                      # subnormal / 零:隐含位为 0
        val = (mant / (2 ** mant_bits)) * (2.0 ** (1 - exp_bias))
    else:                             # normal:隐含位为 1
        val = (1.0 + mant / (2 ** mant_bits)) * (2.0 ** (exp - exp_bias))
    return -val if sign else val

DTYPES = {"fp16 / bf16(16 bit)": 2, "fp8 E4M3 / E5M2(8 bit)": 1}

PRESETS = {
    "Llama-3-8B 类(32 层 · 8 KV头 × 128 维)": dict(layers=32, kv_heads=8, head_dim=128),
    "Qwen2.5-7B 类(28 层 · 4 KV头 × 128 维)": dict(layers=28, kv_heads=4, head_dim=128),
    "7B 老 Llama(32 层 · 32 KV头 × 128 维)": dict(layers=32, kv_heads=32, head_dim=128),
    "自定义": dict(layers=32, kv_heads=8, head_dim=128),
}

with st.sidebar:
    st.header("🎛️ 模型配置")
    preset = st.selectbox("预设模型", list(PRESETS.keys()))
    cfg = PRESETS[preset]
    layers = st.slider("层数 L", 4, 80, cfg["layers"], 1)
    kv_heads = st.slider("KV 头数(GQA)", 1, 32, cfg["kv_heads"], 1)
    head_dim = st.slider("每头维度", 64, 256, cfg["head_dim"], 16)
    batch = st.slider("并发请求数", 1, 128, 32, 1)
    ctx = st.slider("每请求上下文长度", 256, 131072, 4096, 256)
    st.caption("💡 KV = 2(K和V) × L × kv_heads × head_dim × 每元素字节 × token 数")

total_tokens = batch * ctx
kv_elems = 2 * layers * kv_heads * head_dim
bytes_fp16 = kv_elems * 2 * total_tokens
bytes_fp8 = kv_elems * 1 * total_tokens

c1, c2, c3, c4 = st.columns(4)
c1.metric("每 token KV(fp16)", f"{kv_elems * 2 / 1024:.1f} KB")
c2.metric("KV 显存 fp16/bf16", f"{bytes_fp16 / 1024 ** 3:.2f} GB")
c3.metric("KV 显存 fp8", f"{bytes_fp8 / 1024 ** 3:.2f} GB")
c4.metric("节省", f"{(1 - bytes_fp8 / bytes_fp16):.0%}", "fp16 → fp8")

st.subheader(f"📊 KV 显存 vs 并发规模(批量 {batch} × 上下文 {ctx} = {total_tokens:,} tokens)")
toks = np.linspace(1000, 2000000, 60)
fig = go.Figure()
for name, nb, color in [("fp16/bf16(2 B/元素)", 2, "#4C78A8"),
                        ("fp8(1 B/元素)", 1, "#F2C14E"),
                        ("int8 KV(1 B/元素)", 1, "#72B7B2")]:
    fig.add_trace(go.Scatter(x=toks, y=kv_elems * nb * toks / 1024 ** 3, mode="lines",
                             name=name, line=dict(color=color, width=2)))
fig.add_vline(x=total_tokens, line_dash="dot", line_color="#E45756",
              annotation_text=f"当前 = {total_tokens:,} tokens")
fig.update_layout(title="同样一张 24 GB 卡:fp16 撑死 200 万 token,fp8 能翻倍",
                  xaxis_title="并发 token 总数", yaxis_title="KV cache 显存(GB)",
                  height=420, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔬 加餐:E4M3 vs E5M2 可表示值(对数轴)")
e4m3 = sorted({fp8_decode(c, 4, 3, 7) for c in range(256) if fp8_decode(c, 4, 3, 7) > 0})
e5m2 = sorted({fp8_decode(c, 5, 2, 15) for c in range(256) if fp8_decode(c, 5, 2, 15) > 0})
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=e4m3, y=[1] * len(e4m3), mode="markers", name="E4M3",
                          marker=dict(size=7, color="#F2C14E")))
fig2.add_trace(go.Scatter(x=e5m2, y=[0] * len(e5m2), mode="markers", name="E5M2",
                          marker=dict(size=7, color="#72B7B2")))
fig2.update_layout(title="FP8 两种格式的正数刻度点:E4M3 密(精度)vs E5M2 宽(范围)",
                   xaxis_title="数值(对数轴)", yaxis=dict(showticklabels=False),
                   xaxis_type="log", height=300,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **vLLM 用法**:`vllm serve <model> --kv-cache-dtype fp8_e4m3`(或 fp8_e5m2)。
> 注意 KV 量化对长上下文检索类任务可能有轻微精度损失,建议先小规模评测。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 5 章 · 第 33 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
