# -*- coding: utf-8 -*-
"""🧊 app_26_compile.py — torch.compile 与 Kernel 融合对比(minivllm 第 4 章 · 第 26 课)

运行: streamlit run app_26_compile.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧊 26 · torch.compile 与 Kernel 融合", layout="wide")
st.title("🧊 第 26 课配套 App · torch.compile 与 Kernel 融合")

DEFAULT = {
    "ops": [
        {"op": "GELU+线性+残差", "eager_kernels": 7, "compiled_kernels": 2,
         "eager_ms": 0.11, "compiled_ms": 0.06},
        {"op": "LayerNorm+线性", "eager_kernels": 6, "compiled_kernels": 2,
         "eager_ms": 0.09, "compiled_ms": 0.05},
        {"op": "注意力 QKV 投影+分头", "eager_kernels": 8, "compiled_kernels": 3,
         "eager_ms": 0.15, "compiled_ms": 0.08},
    ],
}
meas_file = Path(__file__).parent / "compile_measure_26.json"
if meas_file.exists():
    try:
        DEFAULT = json.loads(meas_file.read_text(encoding="utf-8"))
        st.success(f"已加载第 26 课 notebook 数据 {meas_file.name}(本机无 MSVC 编译器,编译侧为模型估算)")
    except Exception:
        st.warning("加载实测数据失败,使用内置示例数据")
else:
    st.warning(f"未找到 {meas_file.name},使用内置示例数据(运行第 26 课 notebook 可生成真实测量)")

ops = DEFAULT["ops"]
st.markdown(
    "`torch.compile` 用 **TorchInductor** 把多个相邻的 elementwise 算子**融合**成更少的 kernel:"
    "eager 模式下每个 aten 算子各启动一个 kernel,编译后几个算子合并进一个 Triton kernel。"
    "kernel 数量变少 → 启动开销变少 → 数据不再反复进出显存。vLLM 的 `CompilationMode`"
    "就包含 `inductor` 与 `cudagraph` 等层级,与本章的 CUDA Graph 组合使用。"
)

st.sidebar.header("🎛️ 对比参数")
op_name = st.sidebar.selectbox("算子组合", [o["op"] for o in ops])
metric = st.sidebar.radio("查看指标", ["kernel 数量", "单步耗时(ms)"])
show_fusion_detail = st.sidebar.checkbox("显示融合示意图", value=True)
log_y = st.sidebar.checkbox("对数坐标", value=False)

row = next(o for o in ops if o["op"] == op_name)
reduction = 1 - row["compiled_kernels"] / max(row["eager_kernels"], 1)
speedup = row["eager_ms"] / max(row["compiled_ms"], 1e-9)

c1, c2, c3 = st.columns(3)
c1.metric("Kernel 数: eager → 编译", f"{row['eager_kernels']} → {row['compiled_kernels']}")
c2.metric("Kernel 削减率", f"{reduction * 100:.0f}%")
c3.metric("耗时加速比", f"{speedup:.2f}x")

if metric == "kernel 数量":
    df = pd.DataFrame({
        "方案": [f"eager({row['op']})", f"torch.compile({row['op']})"],
        "kernel 数": [row["eager_kernels"], row["compiled_kernels"]],
    })
    fig = px.bar(df, x="方案", y="kernel 数", text="kernel 数", color="方案",
                 color_discrete_sequence=["#8e44ad", "#2e86c1"])
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("kernel 数可以借 torch.profiler 的 key_averages 统计——算子表行数大幅下降。")
else:
    df2 = pd.DataFrame({
        "方案": [f"eager({row['op']})", f"torch.compile({row['op']})"],
        "耗时(ms)": [row["eager_ms"], row["compiled_ms"]],
    })
    fig2 = px.bar(df2, x="方案", y="耗时(ms)", text="耗时(ms)", color="方案",
                  color_discrete_sequence=["#c0392b", "#27ae60"], log_y=log_y)
    fig2.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig2, width="stretch")
    st.caption("小模型 + 小输入时编译收益明显;若输入太大,单 kernel 内存带宽可能成为新瓶颈,加速比会收敛。")

if show_fusion_detail:
    st.markdown("### 🧊 融合示意:eager 的 kernel 序列 vs 编译后的 kernel 序列")
    st.caption("黄色 = 仍独立的 kernel,绿色 = 被融合进同一条 Triton kernel 的算子。")
    n_e = row["eager_kernels"]
    n_c = row["compiled_kernels"]
    gantt = []
    for i in range(n_e):
        gantt.append(dict(方案="eager", 阶段=i, 块长=1, 颜色="#f39c12"))
    per_fused = n_e // max(n_c, 1)
    for i in range(n_c):
        gantt.append(dict(方案="torch.compile", 阶段=i, 块长=min(per_fused * (i + 1), n_e) - per_fused * i, 颜色="#2e86c1"))
    gdf = pd.DataFrame(gantt)
    fig3 = px.bar(gdf, y="方案", x="块长", color="颜色", orientation="h",
                  color_discrete_map={"#f39c12": "#f39c12", "#2e86c1": "#2e86c1"},
                  custom_data=[gdf["阶段"]])
    fig3.update_layout(height=240, showlegend=False, bargap=0.3,
                       xaxis_title="kernel 序号(示意)", yaxis_title="")
    fig3.for_each_trace(lambda t: t.update(hovertemplate="kernel 序号 %{customdata[0]}<extra></extra>"))
    st.plotly_chart(fig3, width="stretch")

st.markdown("---")
st.markdown(
    "⚠️ **注意**:torch.compile 首次运行要**编译+缓存**,有一次性开销(几百 ms 到几秒),且要求输入形状稳定;"
    "vLLM 默认在 `CompilationMode` 中把编译后的图再包进 CUDA Graph,双管齐下。Windows 上 CPU Inductor"
    "需要 MSVC 编译器(cl),GPU Inductor 还需 Triton 的 Windows 移植版 —— 本机两者皆缺,故编译侧数字为模型估算。"
)
st.caption("《minivllm: 图解 vLLM 推理引擎》第 4 章 · 第 26 课配套演示")

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
