"""app_08_kv_memory.py — KV Cache 内存计算器(真实显存分配版)
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_08_kv_memory.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
from real_ops import kv_bytes_real, cuda_available, device_str
st.set_page_config(page_title="KV Cache 内存计算器", layout="wide")

st.title("🧮 KV Cache 的内存账本:一个 token 占多少显存?")
st.markdown(
    "KV Cache 为每个 token 保存**全部层**的 K 与 V,占用公式:\n\n"
    "$$ \\text{KV bytes/token}= 2 \\times L \\times \\text{kv\\_heads} \\times \\text{head\\_dim} \\times \\text{element\\_bytes} $$\n\n"
    "### 🔬 为什么是这个公式?\n"
    "对一个自回归序列,每个 token 在每个头都要预留 **K 与 V 各一份**(因 2);每层都有独立的 KV 投影(因 $L$);"
    "GQA/MQA 里多个 query 头**共享**同一组 KV 头,所以用 $\\text{kv\\_heads}$ 而非全部 $H$ 个头——这正是**省显存的关键旋钮**。\n\n"
    "**本文会真的在 GPU 上按该形状分配一个 KV 张量,量它到底占多少显存**,并与理论公式对照,而不是只敲计算器。"
)
st.caption("拖动滑杆,右侧柱状图对比 MHA/GQA/MQA,饼图展示显存去向;KV 显存数字来自真实 torch.cuda 分配。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    n_layers = st.slider("🧱 层数 num_layers (L)", 1, 200, 32)
with c2:
    n_heads = st.slider("🎯 注意力头数 num_heads (H)", 1, 128, 32)
with c3:
    kv_heads = st.slider("🔑 kv_heads(GQA)", 1, 32, 8)
with c4:
    head_dim = st.select_slider("📏 head_dim", options=[64, 128, 256], value=128)

c5, c6, c7, c8 = st.columns(4)
with c5:
    dtype = st.selectbox("🎛️ 数据类型", ["bf16/fp16 (2 字节)", "fp8 (1 字节)", "fp32 (4 字节)"], index=0)
with c6:
    seq_len = st.slider("📜 单请求序列长度", 128, 16384, 2048, step=128)
with c7:
    batch = st.slider("👥 并发请求数", 1, 256, 16)
with c8:
    run_kv = st.button("🚀 在 GPU 上真实分配 KV 张量", type="primary")

bytes_per_elem = {"bf16/fp16 (2 字节)": 2, "fp8 (1 字节)": 1, "fp32 (4 字节)": 4}[dtype]
dtype_key = {"bf16/fp16 (2 字节)": "bf16", "fp8 (1 字节)": "fp16", "fp32 (4 字节)": "fp32"}[dtype]
per_token = 2 * n_layers * kv_heads * head_dim * bytes_per_elem
per_req = per_token * seq_len
total_kv = per_req * batch
gpu_gb = 16.0
kv_gb = total_kv / 1e9
weights_gb = 7.0 * 1e9 * bytes_per_elem / 1e9
act_gb = weights_gb * 0.02

# 当用户点按钮且是 CUDA,才真的分配
kv_real = None
if run_kv and cuda_available():
    with st.spinner("分配真实 KV 张量并测量显存增量…"):
        kv_real = kv_bytes_real(n_layers, kv_heads, head_dim, seq_len, batch, dtype_key)
elif run_kv and not cuda_available():
    st.warning("未检测到 CUDA,退化为理论值。")

def to_gb(x):
    return x / 1e9

m1, m2, m3, m4 = st.columns(4)
m1.metric("每 token 占用", f"{per_token/1024:.1f} KiB", help="2 × L × kv_heads × head_dim × bytes")
m2.metric("单请求 KV", f"{to_gb(per_req):.2f} GB")
m3.metric("全部并发 KV", f"{kv_gb:.2f} GB",
          delta=f"占显存 {kv_gb/gpu_gb*100:.0f}%" + (" ⚠️" if kv_gb/gpu_gb > 0.5 else ""))
m4.metric("权重(估)", f"{weights_gb:.1f} GB")

if kv_real:
    st.info(
        f"📡 **真实分配结果**:在 {kv_real['device']} 上分配形状为 "
        f"(2, {batch}, {n_layers}, {kv_heads}, {seq_len}, {head_dim}) 的 {dtype_key} 张量,"
        f"实测增量 **{kv_real['alloc_bytes']/1e9:.2f} GB**,与理论公式 **{kv_real['theory_bytes']/1e9:.2f} GB** "
        f"误差 {(kv_real['alloc_bytes']-kv_real['theory_bytes'])/max(kv_real['theory_bytes'],1)*100:.2f}%。")
real = bool(kv_real and not kv_real.get("simulated"))

fig1 = go.Figure()
labels = ["MHA (kv=全部头)", "GQA (kv=%d)" % kv_heads, "MQA (kv=1)"]
vals = [2 * n_layers * n_heads * head_dim * bytes_per_elem * seq_len,
        2 * n_layers * kv_heads * head_dim * bytes_per_elem * seq_len,
        2 * n_layers * 1 * head_dim * bytes_per_elem * seq_len]
fig1.add_trace(go.Bar(x=labels, y=[to_gb(v) for v in vals],
                      marker_color=["#e74c3c", "#2ecc71", "#3498db"], text=[f"{to_gb(v):.2f} GB" for v in vals],
                      textposition="outside"))
fig1.update_layout(title="📊 同配置下 MHA / GQA / MQA 的单请求 KV 显存", yaxis_title="GB",
                    template="plotly_white")
st.plotly_chart(fig1, width="stretch")

fig2 = go.Figure(go.Pie(
    labels=["KV Cache", "模型权重", "激活(估)"],
    values=[kv_gb, weights_gb, act_gb], hole=0.45,
    marker=dict(colors=["#2ecc71", "#3498db", "#f39c12"]),
    textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} GB (%{percent})"))
fig2.update_layout(title="🥧 显存去向:KV Cache 常常是最大的那一块", template="plotly_white")
st.plotly_chart(fig2, width="stretch")
st.markdown("### 🏷️ 关键结论")
st.success(f"当前配置每 token 需 **{per_token/1024:.1f} KiB**;{batch} 个请求、各 {seq_len} token 共需 **{kv_gb:.2f} GB**,"
           f"约占 16 GB 显存的 **{kv_gb/gpu_gb*100:.1f}%**。")
st.markdown(
    "### 💡 为什么 GQA 这么香?\n"
    "MHA 里每个 query 头都带一组专属 KV(**2·H·L·d·seq**),显存随头数线性爆炸;"
    "GQA 让一组 KV 被 4~8 个 query 头共享,**KV 显存直觉上除以 query-per-kv**;"
    "MQA 更是退化为单 KV 组。代价是表达能力略降,但**decode 是 memory-bound**,"
    "省下的带宽直接换吞吐——这就是 LLaMA-2/3 系列普遍用 GQA、Gemma 用 MQA 的原因。\n\n"
    "再把 dtype 换成 fp8/e4m3,元素字节再砍一半——量化与 GQA 是工业界省显存的两大杀器。"
)
real_badge(real=True)
terminology([
    ("KV bytes/token", "一个 token 全部层、K/V 共占的字节数,是显存规划的最小单元。"),
    ("GQA / MQA", "Group-Query Attention 与 Multi-Query Attention:让多个 query 头共享 KV 头以省显存/带宽。"),
    ("e4m3 / e5m2", "FP8 的两种指数长度编码;e4m3 精度更高,vLLM 默认用它做 KV/权重量化。"),
    ("memory access granularity", "对 KV 的读写以固定字节对齐,block 内可能产生 padding(见 09)。"),
    ("vLLM CacheConfig.DEFAULT_BLOCK_SIZE=16", "见 docs/02;块大小决定页粒度,与 half-warp 对齐。"),
] + [("出处2", f"真实分配来源:{device_str()}@{dtype_key}")])

foot_note()

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os
    import subprocess
    import sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])