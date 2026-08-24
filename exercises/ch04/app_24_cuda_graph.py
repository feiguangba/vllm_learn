# -*- coding: utf-8 -*-
"""📸 app_24_cuda_graph.py — CPU-GPU 启动间隙与 CUDA Graph 原理(VLLM_learn 第 4 章 · 第 24 课)

运行: streamlit run app_24_cuda_graph.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📸 24 · CPU-GPU 启动间隙", layout="wide")
st.title("📸 第 24 课配套 App · CPU-GPU 启动间隙与 CUDA Graph")

MEAS = {
    "launch_us": 14.59, "kernel_us": 0.16, "gap_pct": 98.9,
    "n_kernels": 200, "speedup": 94.2,
}
meas_file = Path(__file__).parent / "launch_measure_24.json"
if meas_file.exists():
    try:
        MEAS.update(json.loads(meas_file.read_text(encoding="utf-8")))
        st.success(f"已加载本机 GPU 实测数据 {meas_file.name}")
    except Exception:
        st.warning("加载实测数据失败,使用内置示例数据")
else:
    st.warning(f"未找到 {meas_file.name},使用内置示例数据(运行第 24 课 notebook 可生成真实测量)")

st.markdown(
    "**关键事实**:GPU 是异步设备,CPU 提交一个 kernel 只需几微秒,但每次提交都有 **启动开销(launch overhead)**。"
    "`torch.add` 这种小 kernel GPU 执行只要几微秒,CPU 提交也要几微秒 —— 两者旗鼓相当时,CPU 就成了瓶颈,"
    "GPU 在等待中**空转**。CUDA Graph 把一串 kernel 的启动信息**捕获**成一张图,之后**重放**一次提交即完成全部,"
    "彻底抹掉中间的启动间隙。"
)

st.sidebar.header("🎛️ 仿真参数")
n_kernels = st.sidebar.slider("kernel 数量(个)", 10, 100000, 2000, step=10)
launch_us = st.sidebar.slider("单次 CPU 启动开销(µs)", 0.5, 20.0, MEAS["launch_us"], 0.5)
gpu_us = st.sidebar.slider("单 kernel GPU 执行(µs)", 0.5, 50.0, MEAS["kernel_us"], 0.5)
view_mode = st.sidebar.radio("视图", ["总耗时对比", "间隙占比分解", "间隙随规模增长"])
log_y = st.sidebar.checkbox("Y 轴对数坐标", value=True)

launch_total = n_kernels * launch_us / 1e3
gpu_total = n_kernels * gpu_us / 1e3
eager_ms = launch_total + gpu_total
graph_ms = gpu_total + (MEAS.get("capture_overhead_ms", 1.5) * 1000 / n_kernels)
gap_pct = launch_total / eager_ms * 100 if eager_ms > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Eager 总耗时(仿真)", f"{eager_ms:.2f} ms")
c2.metric("CUDA Graph 总耗时(仿真)", f"{graph_ms:.2f} ms")
c3.metric("启动间隙占比", f"{gap_pct:.1f} %",
          help="启动间隙占比越高,越值得用 CUDA Graph;小 kernel 场景通常 > 50%")

if view_mode == "总耗时对比":
    df = pd.DataFrame({
        "方案": ["Eager(每次提交)", "CUDA Graph(捕获+重放)"],
        "耗时(ms)": [eager_ms, graph_ms],
    })
    fig = px.bar(df, x="方案", y="耗时(ms)", text="耗时(ms)", color="方案",
                 color_discrete_sequence=["#e74c3c", "#27ae60"])
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"本机真实 GPU 微基准({MEAS['n_kernels']} 次小 matmul):eager 逐次提交总耗时 {MEAS['n_kernels']*MEAS['launch_us']:.0f} µs 手续费,"
               f"合并成 1 次 batched matmul 仅 {MEAS['n_kernels']*MEAS['kernel_us']:.0f} µs,"
               f"单次 launch 开销 {MEAS['launch_us']:.2f} µs,启动占比 {MEAS['gap_pct']:.1f}%,加速 {MEAS['speedup']:.1f}x。")
elif view_mode == "间隙占比分解":
    df2 = pd.DataFrame({
        "成分": ["CPU 启动间隙(可消除)", "GPU 实际执行(不可省)"],
        "耗时(ms)": [launch_total, gpu_total],
    })
    fig2 = go.Figure(go.Pie(labels=df2["成分"], values=df2["耗时(ms)"],
                            hole=0.45, marker=dict(colors=["#f39c12", "#2980b9"])))
    fig2.update_layout(height=360)
    fig2.update_traces(textinfo="label+percent")
    st.plotly_chart(fig2, width="stretch")
    st.caption("黄色部分(启动间隙)在 CUDA Graph 方案中被一次性摊平;kernel 数越多,单次提交节省越显著。")
else:
    ks = np.logspace(1, 5, 60).astype(int)
    eager_arr = ks * (launch_us + gpu_us) / 1e3
    graph_arr = ks * gpu_us / 1e3
    df3 = pd.DataFrame({
        "kernel 数": ks,
        "Eager 累计耗时(ms)": eager_arr,
        "CUDA Graph 累计耗时(ms)": graph_arr,
    })
    fig3 = px.line(df3, x="kernel 数", y=["Eager 累计耗时(ms)", "CUDA Graph 累计耗时(ms)"],
                   log_x=True, log_y=log_y, markers=True)
    fig3.update_layout(height=380, yaxis_title="累计耗时(ms)")
    st.plotly_chart(fig3, width="stretch")
    st.caption("两条线之间的距离 = 启动间隙随 kernel 数线性累积的规模效应。")

st.markdown("---")
st.markdown(
    "💡 **直觉**:像点外卖 📸 —— eager 模式是「点一单、做一份、送一份」,每次下单都有固定手续费(启动开销);"
    "CUDA Graph 是「一次性拍好菜谱,之后照单全炒」,手续费只付一次。kernel 越碎越小,收益越大。"
    "vLLM 正是靠 CUDA Graph 让 decode 阶段(每步只有几个小 kernel)的吞吐大幅提升。"
)
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 4 章 · 第 24 课配套演示")

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
