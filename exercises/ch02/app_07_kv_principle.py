"""app_07_kv_principle.py — KV Cache 原理动态演示(真实 GPU 算子实测版)
运行: D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_07_kv_principle.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app_common import terminology, real_badge, foot_note
from real_ops import bench_qkv_attn, cuda_available, device_str
st.set_page_config(page_title="KV Cache 原理 · 计算量对比", layout="wide")

st.title("🔑 KV Cache 原理:无缓存 vs 有缓存,差多少计算量?")
st.markdown(
    "LLM 是**自回归(autoregressive)**模型:每步只生成 1 个新 token,但要让新 token 与**全部历史 token** 交互。\n\n"
    "两个阶段的本质差异:\n"
    "- **prefill(预填充)**:对整段 prompt 做**并行**前向,是 **compute-bound(算力受限)**;\n"
    "- **decode(解码)**:每步 1 个 token,逐步自回归,是 **memory-bound(内存受限)**。\n\n"
    "💡 **无缓存**:每一步都把历史所有 token 的 K、V 重新投影一遍,计算量随序列长度**平方级( O(T²) )**膨胀;\n"
    "💾 **有缓存(KV Cache)**:K、V 只投影一次就存进显存复用,每步只算新 token 的 QKV,总计算量降到**线性( O(T) )**。\n\n"
    "下方先看 **真实 GPU 算子微基准**(直接对注意力核心算子打点计时),再对比两条计算曲线——数据不是理想的 `np.arange`,"
    f"而是跑在 **{device_str()}** 上的实测时延。"
)
st.caption(
    "微基准说明:构造与真实自回归 decode 相同的形态——单 query 对全部历史 key/value 做打分+softmax+加权(async on CUDA)。"
    "「无缓存」模拟把 t 个历史 token 的 QKV **全部当作新算出**并做完整注意力;「有缓存」只用一个新 query 做注意力。"
    "每档重复多次,取其均值。未装 CUDA 时优雅回落为理论估算。"
)

with st.sidebar:
    st.header("⚙️ 基准参数")
    L = st.slider("🧱 层数 num_layers (L)", 1, 80, 32)
    H = st.slider("🎯 注意力头数 num_heads (H)", 1, 64, 32)
    D = st.select_slider("📏 head_dim (D)", options=[64, 128, 256], value=128)
    dtype = st.selectbox("🎛️ 精度 dtype", ["bf16", "fp16"], index=0)
    seq_len = st.slider("📜 已生成 token 数 (T)", 64, 2048, 1024, step=64)
    run = st.button("🚀 跑真实 CUDA 基准", type="primary")
    st.caption(f"GPU: {device_str() if cuda_available() else '未检测到 CUDA,将用理论曲线'}")

T_list = tuple(sorted(set([64, 128, 256, 512, 1024, seq_len])))
if run or not cuda_available():
    bench = bench_qkv_attn(L=L, H=H, D=D, T_list=T_list, dtype=dtype)
else:
    bench = None

if bench is None:
    st.info("点击左侧「🚀 跑真实 CUDA 基准」在 GPU 上实测耗时(约几秒到十几秒)。")
st.session_state.setdefault("bench", bench if bench is not None else None)

# ---------- 真相数据 vs 理论 ----------
if st.session_state["bench"] is not None:
    b = st.session_state["bench"]
    results = b["results"]
    ts = np.array([r[0] for r in results])
    nocache_ms = np.array([r[1] for r in results])
    cache_ms = np.array([r[2] for r in results])
    inst_real = nocache_ms / np.maximum(cache_ms, 1e-9)
    real = not b.get("simulated")

    m1, m2, m3 = st.columns(3)
    m1.metric("数据来源设备", (b["device"] if real else "CPU(理论)"),
              delta="真实算子实测" if real else "估算")
    m2.metric("无缓存末档每步耗时", f"{nocache_ms[-1]:.2f} ms" if real else f"{nocache_ms[-1]:.1e}")
    m3.metric("有缓存末档每步耗时", f"{cache_ms[-1]:.2f} ms" if real else f"{cache_ms[-1]:.1e}")
    st.session_state["real"] = real
else:
    nocache_ms = cache_ms = ts = inst_real = None
    real = False
    st.session_state["real"] = False

# ---------- 理论 FLOPs 曲线(作为对照轴) ----------
ts_full = np.arange(1, seq_len + 1)
qkv = 6 * L * H * D * D
att_per_t = 4 * L * H * D * ts_full
step_cache = qkv + att_per_t
step_nocache = qkv * ts_full + att_per_t
tot_cache = np.cumsum(step_cache)
tot_nocache = np.cumsum(step_nocache)
cum_speedup = tot_nocache / tot_cache

st.subheader("📊 两条计算曲线:平方级 vs 线性")
c1, c2, c3, c4 = st.columns(4)
c1.metric("无缓存累计 FLOPs", f"{tot_nocache[-1]:.3e}")
c2.metric("有缓存累计 FLOPs", f"{tot_cache[-1]:.3e}")
c3.metric("累计加速比(理论)", f"{cum_speedup[-1]:.1f}×")
c4.metric("末步单步加速比(理论)", f"{step_nocache[-1]/step_cache[-1]:.1f}×")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ts_full, y=tot_nocache, name="无缓存(累计)", mode="lines",
                          line=dict(color="#e74c3c", width=2), fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"))
fig1.add_trace(go.Scatter(x=ts_full, y=tot_cache, name="有缓存(累计)", mode="lines",
                          line=dict(color="#2ecc71", width=2), fill="tozeroy", fillcolor="rgba(46,204,113,0.15)"))
if inst_real is not None and len(ts) > 0:
    fig1.add_trace(go.Scatter(x=ts, y=(nocache_ms * 1e15), name="真实·无缓存(注意力核心)", mode="markers+lines",
                              line=dict(color="#c0392b", width=1, dash="dot"),
                              marker=dict(symbol="x", size=6)))
    fig1.add_trace(go.Scatter(x=ts, y=(cache_ms * 1e15), name="真实·有缓存(注意力核心)", mode="markers+lines",
                              line=dict(color="#27ae60", width=1, dash="dot"),
                              marker=dict(symbol="cross", size=6)))
fig1.update_layout(
    title="📊 计算量曲线(实线=理论 FLOPs;虚线标记=该形状下单次注意力算子的实测耗时×缩放)",
    xaxis_title="已生成 token 数", yaxis_title="累计 FLOPs / 相对量度", template="plotly_white",
    hovermode="x unified", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig1, width="stretch")

# ---------- 加速比 ----------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ts_full, y=step_nocache / step_cache, name="每步加速比(理论)", mode="lines",
                          line=dict(color="#3498db", width=2)))
fig2.add_trace(go.Scatter(x=ts_full, y=cum_speedup, name="累计加速比(理论)", mode="lines",
                          line=dict(color="#9b59b6", width=2, dash="dash")))
if inst_real is not None and len(ts) > 0:
    fig2.add_trace(go.Scatter(x=ts, y=inst_real, name="真实·单步加速比", mode="markers",
                              marker=dict(symbol="diamond", size=8, color="#e67e22"),
                              error_y=dict(type="data", array=np.ones_like(ts) * 0.05, visible=True)))
fig2.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="无缓存基准 = 1×")
fig2.update_layout(title="📈 加速比曲线:越往后差距越大", xaxis_title="已生成 token 数",
                    yaxis_title="加速比(×)", template="plotly_white", hovermode="x unified",
                    legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, width="stretch")
st.markdown("### 🏷️ 关键结论")
st.success(
    f"理论上序列长度为 **{seq_len}** 时,无缓存累计计算量约是 **{cum_speedup[-1]:.1f} 倍**于有缓存;"
    "这不依赖于模型规模,只取决于序列长度—— token 越多,差距越大。"
)
st.markdown(
    "**为什么 decode 快不起来?** 瓶颈并不在 QKV 那点 FLOPs,而在**内存**:每步只产出 1 个 token,"
    "却要把注意力完整读一遍 K、V(half-warp 逐 token)。真正让 vLLM 在 decode 下跑满吞吐的,"
    "是 **PagedAttention + CUDA Graph + 连续批处理** 的组合拳。本课先记住「KV Cache 把 O(T²) 降成 O(T)」这一记重拳。"
)
real_badge(real=st.session_state.get("real", False))
terminology([
    ("autoregressive / self-attention", "自回归每一新 token 都要与全部历史位置交互,是 KV Cache 存在的前提;attention 让位置可并行,见 Attention Is All You Need。"),
    ("compute-bound vs memory-bound", "prefill 算力受限,decode 受内存带宽限制,二者主导的硬件瓶颈完全不同。"),
    ("FLOPs 与 O(T²)/O(T)", "无缓存把 QKV 投影乘以 t 次故为平方;有缓存该部分恒为常数故线性。"),
    ("KV Cache", "缓存已算出的 K/V 张量,按 层×头×序列 存储;历史 K/V 与未来 token 无关,故数学上无损复用。"),
    ("prefill / TTFT", "预填充阶段对该 prompt 做一次并行前向并得到首个 token;TTFT 是衡量其快慢的指标。"),
    ("decode / TPOT", "逐 token 自回归阶段;TPOT 是每个输出 token 的平均耗时,memory-bound 主导。"),
    ("数据来源", f"本次真实注意力算子实测跑在 {device_str()}(CUDA 微基准)。参考文献:PagedAttention SOSP'23 · vLLM docs。"),
])

foot_note()

# ============ PyCharm / 直接运行入口 ============
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