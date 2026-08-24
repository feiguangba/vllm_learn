# -*- coding: utf-8 -*-
"""🎬 app_25_capture.py — CUDA Graph 捕获与重放演示(VLLM_learn 第 4 章 · 第 25 课)

运行: streamlit run app_25_capture.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🎬 25 · CUDA Graph 捕获与重放", layout="wide")
st.title("🎬 第 25 课配套 App · CUDA Graph 捕获与重放")

DEFAULT = {
    "eager_mean_ms": 0.48, "graph_mean_ms": 0.31, "replay_lat_ms": [0.30, 0.33, 0.29],
    "speedup": 1.55, "capture_ms": 9.2, "warmup": 3, "batch": 8, "hidden": 256,
}
meas_file = Path(__file__).parent / "cudagraph_result_25.json"
if meas_file.exists():
    try:
        data = json.loads(meas_file.read_text(encoding="utf-8"))
        DEFAULT.update(data)
        st.success(f"已加载第 25 课 notebook 生成的数据 {meas_file.name}(本机 CPU 类比 + 启动开销模型估算)")
    except Exception:
        st.warning("加载实测数据失败,使用内置示例数据")
else:
    st.warning(f"未找到 {meas_file.name},使用内置示例数据(运行第 25 课 notebook 可生成真实测量)")

st.markdown(
    "CUDA Graph 的捕获流程就像**录像 🎬**:先把固定形状的前向跑几遍热身(warmup),再用流捕获(stream capture)"
    "把整串 kernel **录**下来,之后每次推理只需**重放**(replay)这张图 —— 单次提交、全图执行。"
    "注意捕获有**硬约束**:形状必须固定、捕获期间不能分配显存/不能同步。"
)

st.sidebar.header("🎛️ 展示参数")
batch = st.sidebar.slider("展示批大小 batch", 1, 64, int(DEFAULT.get("batch", 8)))
replay_n = st.sidebar.slider("重放次数(分析延迟分布)", 20, 500, int(len(DEFAULT.get("replay_lat_ms", [100]))),
                             step=10)
chart_mode = st.sidebar.radio("图表", ["延迟对比", "重放延迟分布", "耗时累计曲线"])
show_capture = st.sidebar.checkbox("显示捕获开销明细", value=True)

mean_e = DEFAULT["eager_mean_ms"]
mean_g = DEFAULT["graph_mean_ms"]
speedup = DEFAULT.get("speedup", mean_e / max(mean_g, 1e-9))
rng = np.random.default_rng(7)
lat = rng.normal(mean_g, mean_g * 0.06, replay_n).clip(0.05, None)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Eager 单步延迟(ms)", f"{mean_e:.3f}")
c2.metric("Graph 单步延迟(ms)", f"{mean_g:.3f}")
c3.metric("加速比", f"{speedup:.2f}x")
c4.metric("捕获开销(一次性,ms)", f"{DEFAULT.get('capture_ms', 0):.1f}",
          help="捕获开销只付一次,之后每次重放都几乎为零")

if chart_mode == "延迟对比":
    df = pd.DataFrame({"方案": ["Eager(逐 kernel 提交)", "CUDA Graph(重放)"],
                       "单步延迟(ms)": [mean_e, mean_g]})
    fig = px.bar(df, x="方案", y="单步延迟(ms)", text="单步延迟(ms)", color="方案",
                 color_discrete_sequence=["#e67e22", "#16a085"])
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("小模型 + 小批时,kernel 执行时间短,启动开销占比大,Graph 收益最明显。")
elif chart_mode == "重放延迟分布":
    fig2 = go.Figure(go.Histogram(x=lat, nbinsx=30, marker_color="#16a085"))
    fig2.add_vline(x=mean_g, line_dash="dash", line_color="black",
                   annotation_text=f"均值 {mean_g:.3f} ms")
    fig2.update_layout(height=360, xaxis_title="单次重放延迟(ms)", yaxis_title="次数")
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"重复重放 {replay_n} 次,延迟围绕均值小幅波动 —— 无启动开销,方差主要来自 GPU 调度噪声。")
else:
    ks = np.arange(1, replay_n + 1)
    cum_e = ks * mean_e
    cum_g = ks * mean_g
    df3 = pd.DataFrame({"重放次数": ks, "Eager 累计(ms)": cum_e, "Graph 累计(ms)": cum_g})
    fig3 = px.line(df3, x="重放次数", y=["Eager 累计(ms)", "Graph 累计(ms)"], markers=True)
    fig3.update_layout(height=360, yaxis_title="累计耗时(ms)")
    st.plotly_chart(fig3, width="stretch")
    st.caption("decode 阶段每步只多 1 个词元,却要跑一遍整网络 —— 每步省下的零点几毫秒,乘上几百步就非常可观。")

if show_capture:
    st.markdown("### 🎬 捕获流程 5 步")
    steps = [
        ("1 热身 warmup", f"固定形状跑 {DEFAULT.get('warmup', 3)} 次,让 CUDA 上下文就绪", "🫀"),
        ("2 分配固定缓冲", "输入/输出缓冲一次性分配,形状锁死", "🔒"),
        ("3 开启流捕获", "with torch.cuda.graph(g): 开始录像", "📹"),
        ("4 录制前向", "正常跑一遍前向,GPU 按图记录所有 kernel 依赖", "🎞️"),
        ("5 重放循环", "g.replay() 单次提交,之后每步零启动开销", "▶️"),
    ]
    for i, (t, d, e) in enumerate(steps, 1):
        with st.expander(f"{e} 第 {i} 步: {t}", expanded=(i == 5)):
            st.markdown(d)
            if i == 5:
                st.code("for _ in range(n):\n    g.replay()", language="python")

st.markdown("---")
st.markdown(
    "⚠️ **坑**:捕获后输入张量地址必须保持不变(重放即重算图里的旧地址),所以 vLLM 用**静态输入缓冲区**"
    "把每轮数据先拷贝进去再重放;批大小一变化就要**重新捕获** —— 这也是 vLLM 按 batch 桶(fixed shapes)"
    "捕获多张图的原因。"
)
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 4 章 · 第 25 课配套演示")

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
