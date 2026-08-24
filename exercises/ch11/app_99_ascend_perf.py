# -*- coding: utf-8 -*-
# app_99_ascend_perf.py — 昇腾推理性能调优仪表盘 📊
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾推理性能调优 📊", layout="wide")
st.title("📊 第 99 课 · 昇腾推理性能调优:吞吐 / 时延仪表盘")

st.markdown("""
性能调优盯三个数:**TTFT(首字延迟)、TPOT(每字延迟)、吞吐(tokens/s)**。
下面拖一拖并发与输出长度、开一开优化开关,实时看指标怎么变 —— 就像第 50 课的仪表盘,
这次是昇腾版。
""")

st.sidebar.header("🎛️ 参数")
conc = st.sidebar.slider("并发请求数", 1, 256, 32, 1)
out_len = st.sidebar.slider("输出 token 数", 16, 1024, 128, 16)
opt_level = st.sidebar.radio("优化等级", ["baseline(全关)", "图模式+融合", "全部(KV量化+动态shape+TP)"])
npu = st.sidebar.selectbox("设备", ["昇腾 910B", "昇腾 310P", "RTX 5060(GPU 对照)"])
st.sidebar.caption("优化等级越高,单请求越快,但基线吞吐越低 —— 需要实测选甜蜜点。")

BASE = {"昇腾 910B": (3.5, 12.0), "昇腾 310P": (7.0, 25.0), "RTX 5060(GPU 对照)": (4.5, 16.0)}
base_ttft, base_tpot = BASE[npu]
alpha = 0.12                                  # 争抢系数
opt_scale = {"baseline(全关)": 1.0, "图模式+融合": 0.75, "全部(KV量化+动态shape+TP)": 0.55}
scale = opt_scale[opt_level]

ttft = base_ttft * (1 + alpha * (conc - 1)) * scale
tpot = base_tpot * (1 + alpha * (conc - 1) * 0.6) * scale
e2e = ttft + out_len * tpot
thr = conc * out_len / e2e

c1, c2, c3, c4 = st.columns(4)
c1.metric("TTFT(首字延迟)", f"{ttft:.1f} ms")
c2.metric("TPOT(每字延迟)", f"{tpot:.2f} ms")
c3.metric("E2E(端到端)", f"{e2e/1000:.2f} s")
c4.metric("吞吐", f"{thr:,.0f} tokens/s")

st.subheader("📈 吞吐 vs 并发(甜蜜点)")
cs = np.arange(1, 257)
thrs = []
for c in cs:
    t_ttft = base_ttft * (1 + alpha * (c - 1)) * scale
    t_tpot = base_tpot * (1 + alpha * (c - 1) * 0.6) * scale
    thrs.append(c * out_len / (t_ttft + out_len * t_tpot))
fig = go.Figure()
fig.add_trace(go.Scatter(x=cs, y=thrs, mode="lines", name="吞吐",
                         line=dict(color="#2e86c1", width=3)))
fig.add_vline(x=conc, line_dash="dash", line_color="#888")
fig.update_layout(title=f"吞吐曲线({npu} · {opt_level})", xaxis_title="并发",
                  yaxis_title="吞吐(tokens/s)", height=340,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🥧 单请求延迟组成")
fig2 = go.Figure(go.Pie(labels=["Prefill(TTFT)", "Decode(输出×TPOT)", "排队/调度"],
                        values=[ttft, out_len * tpot * 0.97, out_len * tpot * 0.03],
                        hole=0.5, marker_colors=["#e74c3c", "#2e86c1", "#95a5a6"]))
fig2.update_layout(title="E2E 延迟构成:输出越长,Decode 越占大头", height=320,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🛠️ 优化收益瀑布(相对 baseline)")
levels = ["baseline", "+图模式/融合", "+动态shape", "+KV量化", "+TP并行"]
ratio = [1.0, 0.8, 0.68, 0.55, 0.42]
fig3 = go.Figure(go.Waterfall(x=levels, y=[ratio[0]] + [ratio[i]-ratio[i-1] for i in range(1, 5)],
                              measure=["absolute"] + ["relative"]*4,
                              decreasing=dict(marker_color="#27ae60"),
                              connector=dict(line=dict(color="#888"))))
fig3.update_layout(title="调优瀑布:每步优化把单请求耗时再砍一截(示意)", yaxis_title="相对耗时",
                   height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
> 💡 **结论**:调优第一步永远是**上 profiler 量**(昇腾用 msprof / Ascend Insight Toolkit),
> 分清瓶颈是『算得慢 / 搬得等 / 调度空转』;然后按 图模式→融合→动态shape→KV 量化→并行 的
> 顺序逐项开,每开一项复测一次,别一口气全开。
""")
