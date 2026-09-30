# -*- coding: utf-8 -*-
"""生成 100_summary_roadmap.ipynb 与 app_100_summary.py(全书收尾)"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_100 = D('''
# -*- coding: utf-8 -*-
# app_100_summary.py — 全书总结:学习路线交互浏览 🏁
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="minivllm 全书路线图 🏁", layout="wide")
st.title("🏁 第 100 课 · 全书总结:vLLM 推理引擎全景与学习路线")

st.markdown("""
从第 1 课『认识 Token』到第 100 课(就是本页),我们走完了 **LLM 推理的完整地图**:
**推理基础 → KV Cache → 批处理调度 → 执行加速 → 量化 → 分布式 → Attention →
部署监控 → Triton → AI 编译器 → 华为昇腾生态**。下方可任选章节 / 拖动课号范围 /
切换主题主线,把全书"地图"翻来覆去看。
""")

CHS = [
    (1, "LLM 推理基础", 1, 6), (2, "KV Cache 与 PagedAttention", 7, 13),
    (3, "Continuous Batching 与调度", 14, 20), (4, "模型执行与 CUDA Graph", 21, 27),
    (5, "量化", 28, 34), (6, "分布式并行", 35, 41),
    (7, "Attention Kernel 实战", 42, 47), (8, "端到端 vLLM 部署", 48, 50),
    (9, "Triton 编程", 51, 60), (10, "AI 编译器原理", 61, 70),
    (11, "华为昇腾与 MindSpore 生态", 71, 100),
]
LINES = {
    "KV Cache 主线": [7, 8, 9, 10, 11, 12, 13, 45, 85, 96],
    "批处理与调度主线": [14, 15, 16, 17, 18, 19, 20],
    "量化主线": [28, 29, 30, 31, 32, 33, 34, 87, 97],
    "编译器主线": [26, 51, 56, 61, 64, 66, 67, 70, 79, 95],
    "并行与通信主线": [35, 36, 37, 38, 39, 40, 41, 80, 98],
    "部署与监控主线": [24, 25, 27, 48, 49, 50, 81, 99],
}

st.sidebar.header("🎛️ 参数")
sel_chs = st.sidebar.multiselect("选择章节", [f"第{c[0]}章 {c[1]}" for c in CHS],
                                 default=["第11章 华为昇腾与 MindSpore 生态"])
rng = st.sidebar.slider("课号范围", 1, 100, (1, 100))
line = st.sidebar.radio("主题主线", list(LINES.keys()))
show_scatter = st.sidebar.checkbox("显示全书课程分布散点", value=True)
st.sidebar.caption("每一条主线都是『自洽的进阶路径』 —— 跳着学也能成体系。")

sel_idx = {f"第{c[0]}章 {c[1]}": c[0] for c in CHS}
covered = [c for c in CHS if f"第{c[0]}章 {c[1]}" in sel_chs]
total_lessons = sum(c[3] - c[2] + 1 for c in covered)

c1, c2, c3, c4 = st.columns(4)
c1.metric("全书课数", 100)
c2.metric("选中章节覆盖课数", total_lessons)
c3.metric("主题主线数", len(LINES))
c4.metric("当前主线课数", len(LINES[line]))

st.subheader("🏔️ 全书地图:课程分布")
if show_scatter:
    pts_x, pts_y, pts_c = [], [], []
    for ch, name, lo, hi in CHS:
        for n in range(lo, hi + 1):
            if ch in sel_idx.values() or f"第{ch}章 {name}" in sel_chs:
                if rng[0] <= n <= rng[1]:
                    pts_x.append(n); pts_y.append(ch)
    fig = go.Figure(go.Scatter(x=pts_x, y=pts_y, mode="markers",
                               marker=dict(size=11, color=pts_y, colorscale="Viridis",
                                           showscale=True, colorbar=dict(title="章节")),
                               text=[f"第{n}课" for n in pts_x], hovertemplate="%{text}<br>章节 %{y}<extra></extra>"))
    for ch, name, lo, hi in CHS:
        fig.add_vline(x=lo - 0.5, line_dash="dot", line_color="#ccc")
        fig.add_annotation(x=(lo + hi) / 2, y=11.6, text=f"第{ch}章", showarrow=False,
                           font=dict(size=9, color="#888"))
    fig.update_layout(title="全书 100 课按章节分布(横轴=课号,纵轴=章节)",
                      xaxis=dict(range=[0, 101]), yaxis=dict(range=[0.5, 11.5], tickvals=list(range(1, 12))),
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧵 当前主线:各课连起来看")
l_lessons = [n for n in LINES[line] if rng[0] <= n <= rng[1]]
fig2 = go.Figure(go.Scatter(x=l_lessons, y=[0.5] * len(l_lessons), mode="markers+text",
                            marker=dict(size=15, color="#c0392b"),
                            text=[f"L{n}" for n in l_lessons], textposition="bottom center"))
fig2.add_trace(go.Scatter(x=l_lessons, y=[0.5] * len(l_lessons), mode="lines",
                          line=dict(color="#c0392b", width=2), showlegend=False))
fig2.update_layout(title=f"主线『{line}』的进阶路径(可再叠加范围筛选)", yaxis=dict(range=[0, 1], showticklabels=False),
                   xaxis=dict(range=[0, 101], title="课号"), height=260,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 各章课数统计")
fig3 = go.Figure(go.Bar(x=[f"第{c[0]}章" for c in CHS], y=[c[3] - c[2] + 1 for c in CHS],
                        text=[c[3] - c[2] + 1 for c in CHS], textposition="outside",
                        marker_color="#2e86c1"))
fig3.update_layout(title="每章课数:1-6 章打基础,9-11 章深入工具链", yaxis_title="课数",
                   height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
> 💡 **给毕业生的三句话**:① 会用 vLLM 只是起点,懂它**为什么快**(内存/调度/编译)才是
> 面试与调优的分水岭;② 手写 Triton 让你从『调包』变『造轮子』;③ 多硬件视野(GPU + 昇腾)
> 是 2026 年的加分项 —— 你已经在路上了。
""")
''')

NB = new_nb("第 100 课 · 全书总结:vLLM 推理引擎全景与学习路线",
            subtitle="全书 100 课总览 · 各章关系图 · 六大主线串讲 · 最终学习路线图",
            emoji="🏁")

chapter_cover(NB,
    objectives=[
        "回望全书 100 课:从 Token 到昇腾,一条完整的推理引擎知识链",
        "画出全书 11 章的关系图,看懂『谁支撑了谁』",
        "串讲六条主题主线:KV Cache / 批处理 / 量化 / 编译器 / 并行 / 部署",
        "用数据统计全书:各章课数、主线覆盖、难度分布",
        "给出毕业后的进阶学习路线图与行动建议",
        "配 App:交互浏览全书地图与主题主线",
    ],
    toc=[
        ("登顶仪式:回望 100 课", "从 L1 到 L100,一条怎样的路"),
        ("全书四段主线", "基础引擎 → 优化加速 → 工具链 → 多硬件生态"),
        ("11 章关系图", "谁支撑了谁:依赖关系一张图"),
        ("数据统计全书", "课数 / 主线 / 章节分布的可视化"),
        ("六条主题主线", "挑一条,跳着学也能成体系"),
        ("毕业路线图", "从『会用』到『会造』到『会调优』"),
        ("配套 App", "app_100_summary.py:学习路线交互浏览"),
    ],
    links=[
        ("vLLM 官方文档", "https://docs.vllm.ai"),
        ("PyTorch 文档", "https://pytorch.org/docs/stable/index.html"),
        ("Triton 文档", "https://triton-lang.org"),
        ("昇腾社区", "https://www.hiascend.com"),
        ("MindSpore", "https://www.mindspore.cn"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 登顶仪式:回望 100 课 ⛰️",
D('''
恭喜!你翻到了全书的最后一课。回想第 1 课,我们还在问"模型为什么只认 token 不认字";
一百课之后,你已经能从**编译器、内存、调度、通信、多硬件**的视角拆解一个推理引擎。

这一百课可以浓缩成一张地图:

- **第 1-4 章(01-27)**:推理基础 —— token → Transformer → prefill/decode → KV Cache →
  PagedAttention → 连续批处理 → CUDA Graph,把 vLLM"为什么快"的骨架搭起来;
- **第 5-8 章(28-50)**:优化与落地 —— 量化、分布式并行、Attention Kernel、部署与监控;
- **第 9-10 章(51-70)**:编程与编译 —— Triton 手写 kernel、AI 编译器全栈;
- **第 11 章(71-100)**:多硬件生态 —— 华为昇腾 + MindSpore,从架构、算子、图引擎到
  集群通信与性能调优,最后回到你正在读的这课。

下面先把这四大段画出来:
'''))

NB.code(D('''
phases = [
    ("推理基础", "01-27", "Token · Transformer · KV Cache\\nPagedAttention · Batching · CUDA Graph", "#dbe7f4", "#1f4e79"),
    ("优化与落地", "28-50", "量化 · 分布式 · Attention\\n部署服务 · 性能监控", "#d9ead3", "#38761d"),
    ("编程与编译", "51-70", "Triton 手写 kernel\\nAI 编译器全景与自动调优", "#fff2cc", "#7f6000"),
    ("多硬件生态", "71-100", "昇腾 · MindSpore · CANN\\nAscend C · HCCL · 云服务", "#e2d5f1", "#5b2c8f"),
]
fig, ax = plt.subplots(figsize=(10.5, 3.6))
ax.axis("off")
x = 0.3; bw, bh = 2.35, 2.2
for i, (t, rng, sub, fill, edge) in enumerate(phases):
    ax.add_patch(plt.Rectangle((x, 0.7), bw, bh, facecolor=fill, edgecolor=edge, lw=2))
    ax.text(x + bw/2, 0.7 + bh*0.72, f"{t}  {rng}", ha="center", fontsize=12, fontweight="bold", color=edge)
    ax.text(x + bw/2, 0.7 + bh*0.3, sub, ha="center", va="center", fontsize=8, color="#333")
    if i < len(phases) - 1:
        ax.annotate("", xy=(x + bw + 0.1, 1.8), xytext=(x + bw - 0.05, 1.8),
                    arrowprops=dict(arrowstyle="->", lw=2.4, color="#555"))
    x += bw + 0.5
ax.text(0.5, 3.2, "minivllm 全书四段主线(第 1-100 课)", fontsize=14, fontweight="bold", ha="center", color="#1f4e79")
ax.set_xlim(0, x); ax.set_ylim(0, 3.7)
plt.tight_layout()
'''),
"🗺️ 四段主线:每一步都建立在前一段之上 —— 不懂 KV Cache 就读不懂调度,不会编译器就看不懂昇腾 GE。")

NB.md("## 2️⃣ 11 章关系图:谁支撑了谁 🕸️",
D('''
把 11 章画成一张依赖图:箭头表示"前者是后者的基础"。看懂这张图,你就看懂了整本书的
**学习顺序**:基础章节打底,工具链章节建立在基础之上,华为生态是"多硬件视野"的延伸。
'''))

NB.code(D('''
nodes = [
    (1, "第1章\\n推理基础", 0.5, 8.5), (2, "第2章\\nKV Cache", 2.2, 8.2),
    (3, "第3章\\nBatching", 4.0, 8.0), (4, "第4章\\n执行加速", 5.8, 7.6),
    (5, "第5章\\n量化", 5.8, 5.6), (6, "第6章\\n分布式", 5.8, 3.6),
    (7, "第7章\\nAttention", 3.8, 6.6), (8, "第8章\\n部署监控", 2.4, 5.6),
    (9, "第9章\\nTriton", 0.6, 6.0), (10, "第10章\\nAI 编译器", 0.4, 3.8),
    (11, "第11章\\n华为生态", 2.6, 2.0),
]
edges = [(1, 2), (2, 3), (3, 4), (4, 7), (7, 8), (4, 9), (9, 10), (10, 11),
         (6, 8), (6, 11), (5, 11), (2, 7), (8, 11)]
pos = {n: (x, y) for n, _, x, y in nodes}
fig, ax = plt.subplots(figsize=(9.5, 7))
for a, b in edges:
    xa, ya = pos[a]; xb, yb = pos[b]
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#888",
                                connectionstyle="arc3,rad=0.15"))
for n, name, x, y in nodes:
    ax.add_patch(plt.Circle((x, y), 0.62, facecolor="#dbe7f4", edgecolor="#1f4e79", lw=1.8, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold", color="#1f4e79", zorder=4)
ax.set_xlim(0, 7); ax.set_ylim(0.5, 9.6); ax.axis("off")
ax.text(3.5, 9.3, "全书 11 章依赖关系:箭头 = 前者是后者的基础", fontsize=13, fontweight="bold", ha="center")
plt.tight_layout()
'''),
"🕸️ 依赖图:主线是 1→2→3→4→9→10→11(推理基础→工具链→多硬件),注意第 5、6、8 章像『支线』一样汇入主线与终点。")

NB.md("## 3️⃣ 数据统计:全书到底讲了什么 📊",
D('''
用数据给全书做个"体检":每章课数、课程在章节间的分布。你会发现两个特点 ——
**基础章节(1-6 章)循序渐进但课数均匀,工具链章节(9-11 章)讲得最深最密**。
'''))

NB.code(D('''
CHS = [
    ("第1章 推理基础", 1, 6), ("第2章 KV Cache", 7, 13), ("第3章 Batching", 14, 20),
    ("第4章 执行加速", 21, 27), ("第5章 量化", 28, 34), ("第6章 分布式", 35, 41),
    ("第7章 Attention", 42, 47), ("第8章 部署监控", 48, 50), ("第9章 Triton", 51, 60),
    ("第10章 AI 编译器", 61, 70), ("第11章 华为生态", 71, 100),
]
names = [c[0] for c in CHS]; counts = [c[2] - c[1] + 1 for c in CHS]
fig, ax = plt.subplots(figsize=(9.5, 4))
bars = ax.bar(names, counts, color="#2e86c1", width=0.6)
for b, v in zip(bars, counts):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, str(v), ha="center", fontsize=9)
ax.set_xticklabels(names, rotation=35, ha="right")
ax.set_ylabel("课数")
ax.set_title(f"全书共 {sum(counts)} 课:第 11 章(华为生态)占 30 课,是全书最厚的一章")
plt.tight_layout()
print(f"全书课数 = {sum(counts)};最厚章节 = 第11章({counts[-1]} 课);最薄 = 第8章({counts[7]} 课)")
'''),
"📊 统计:第 11 章 30 课最厚(涵盖框架、算子、编译器、推理、生态五大块),第 8 章(部署)只有 3 课 —— 因为部署与监控的底层已在前面讲透。")

NB.md("## 4️⃣ 六条主题主线:跳着学也能成体系 🧵",
D('''
不必从头啃到尾 —— 全书藏着几条**自洽的主线**,任选一条跟着学,都能形成体系:

1. **KV Cache 主线**:L7→L8→L9→L10→L11→L12→L13(原理与内存)→ L45(PagedAttention)
   → L85(昇腾实现)→ L96(内存账本);
2. **批处理与调度主线**:L14→L15→L16→L17→L18→L19→L20;
3. **量化主线**:L28→L29→L30→L31→L32→L33→L34(GPU)→ L87/L97(昇腾);
4. **编译器主线**:L26(torch.compile)→ L51/56(Triton)→ L61→L64→L66→L67→L70 →
   L79(图算融合)/L95(GE);
5. **并行与通信主线**:L35→L36(AllReduce)→ L37→L38→L39→L40→L41 → L80/L98(HCCL);
6. **部署与监控主线**:L24→L25→L27→L48→L49→L50 → L81/L99。

下面把它们画成一张"主线地图":
'''))

NB.code(D('''
LINES = {
    "KV Cache": [7, 8, 9, 10, 11, 12, 13, 45, 85, 96],
    "Batching": [14, 15, 16, 17, 18, 19, 20],
    "量化": [28, 29, 30, 31, 32, 33, 34, 87, 97],
    "编译器": [26, 51, 56, 61, 64, 66, 67, 70, 79, 95],
    "并行": [35, 36, 37, 38, 39, 40, 41, 80, 98],
    "部署": [24, 25, 27, 48, 49, 50, 81, 99],
}
fig, ax = plt.subplots(figsize=(10.5, 5))
colors = ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad", "#e67e22", "#16a085"]
for k, (name, lessons) in enumerate(zip(LINES, LINES.values())):
    y = k
    ax.plot(lessons, [y]*len(lessons), "-o", color=colors[k], lw=2.4, ms=6, label=name)
    for i, L in enumerate(lessons):
        ax.annotate(str(L), (lessons[i], y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=colors[k])
ax.set_yticks(range(len(LINES))); ax.set_yticklabels(list(LINES.keys()))
ax.set_xlim(0, 101); ax.set_xlabel("课号")
ax.set_title("六条主题主线:每个小圆点是一课,沿着任一条线跳着学,都能自成体系")
ax.legend(loc="upper left", ncol=2, fontsize=8); ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
'''),
"🧵 主线地图:同一课可能属于多条主线(L45 既在 KV 线也在 Attention 线) —— 主线是帮助你『按兴趣切入』的导航,不是约束。")

NB.md("## 5️⃣ 毕业后路线图:从会用、会造到会调优 🗺️",
D('''
把这本书读完,只是"拿到地图"。接下来四步把知识变成能力:

1. **会读源码**:挑 vLLM 的 scheduler / attention / cache 三个模块精读,对应 L16/L17/L45;
2. **会造轮子**:用 Triton 复刻 FlashAttention 与 PagedAttention 的 kernel(对应 L55/L58),
   再试着给昇腾写一个 Ascend C 算子(对应 L74/L92);
3. **会调优**:在生产环境把 TTFT/TPOT/吞吐调到位(GPU 用 vLLM,昇腾用 MindIE/vLLM-Ascend);
4. **会做多硬件决策**:同一套服务在 GPU 与昇腾上的迁移、量化、并行方案怎么选。

把这四条画成"毕业路线图":
'''))

NB.code(D('''
steps = [
    ("精读源码", "scheduler · attention\\ncache 三块(vLLM)"),
    ("手写 kernel", "Triton FA / PagedAttn\\nAscend C 算子"),
    ("生产调优", "TTFT/TPOT/吞吐\\nGPU + 昇腾双栈"),
    ("多硬件决策", "迁移 · 量化 · 并行\\n选型与成本"),
]
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.axis("off")
x = 0.4; bw, bh = 2.15, 1.6
for i, (t, sub) in enumerate(steps):
    ax.add_patch(plt.Rectangle((x, 0.6), bw, bh, facecolor="#eaf2f8", edgecolor="#1f4e79", lw=2))
    ax.text(x + bw/2, 0.6 + bh*0.7, t, ha="center", fontsize=12, fontweight="bold", color="#1f4e79")
    ax.text(x + bw/2, 0.6 + bh*0.3, sub, ha="center", va="center", fontsize=8, color="#333")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + bw + 0.1, 1.4), xytext=(x + bw - 0.05, 1.4),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="#555"))
    x += bw + 0.45
ax.text(0.5, 2.6, "毕业路线图:读源码 → 造轮子 → 调优 → 多硬件决策", fontsize=13,
        fontweight="bold", ha="center", color="#1f4e79")
ax.set_xlim(0, x); ax.set_ylim(0, 3.2)
plt.tight_layout()
'''),
"🗺️ 四步进阶:每一步都能从书里找到对应课程 —— 这本练习册不是终点,而是你工程能力的起点。")

NB.md("## 6️⃣ 结语:真正的路在脚下 🎉",
D('''
最后一句话送给读到这里的你:

> **懂 vLLM 的人很多,懂『为什么快』的人很少;会用 GPU 的人很多,能从容切换多硬件的人更少。**
> 你已经把这两件事都做了 —— 从 token 到昇腾,从手写 kernel 到整图下沉。

本书 100 课到此结束。愿你带着这份"编译器 + 内存 + 调度 + 多硬件"的思维,去读论文、
调生产、造工具,在 LLM 推理的浪潮里,越走越远。🏁 后会有期!
'''))

NB.md("## 7️⃣ 配套 App:学习路线交互浏览 🎛️",
D('''
运行同目录的 `app_100_summary.py`,**多选章节、拖动课号范围、切换六条主题主线**,
实时查看全书地图、主线路径与各章统计:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_100_summary.py
```

浏览器打开 **http://localhost:8501**(也可 `--server.port 8700`)。完整源码如下:
'''))

NB.code(app_cell("app_100_summary.py", APP_100),
"📜 运行本 cell 会覆盖写入 `app_100_summary.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "全书 100 课四段主线:推理基础(01-27)→ 优化落地(28-50)→ 编程编译(51-70)→ 华为生态(71-100)",
        "11 章依赖图:主线 1→2→3→4→9→10→11,量化/分布式/部署是汇入主线的支流",
        "六条主题主线:KV Cache / Batching / 量化 / 编译器 / 并行 / 部署,可跳着学",
        "毕业后四步:精读源码 → 手写 kernel → 生产调优 → 多硬件决策",
        "你已具备『编译器 + 内存 + 调度 + 多硬件』的完整思维 —— 路在脚下",
    ],
    practice=[
        "把第 2 节依赖图里第 11 章的入边单独画出来,说出它从哪几章『吸收』了什么",
        "设计你自己的第 7 条主线(如『内存优化』:L8→L9→L23→L63→L96),标出课号",
        "用 App 的『课号范围』筛选器,圈出你已掌握的课程,制定 30 天补课计划",
        "写一篇 300 字的毕业感想:书里哪一课改变了你对推理引擎的理解",
    ],
    links=[
        ("vLLM 官方文档", "https://docs.vllm.ai"),
        ("PyTorch 文档", "https://pytorch.org/docs/stable/index.html"),
        ("Triton 语言文档", "https://triton-lang.org"),
        ("昇腾开发者社区", "https://www.hiascend.com"),
        ("MindSpore", "https://www.mindspore.cn"),
    ])

out = str(Path(CH11) / "100_summary_roadmap.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_100_summary.py"
app_path.write_text(APP_100 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
