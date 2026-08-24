# -*- coding: utf-8 -*-
"""生成 93_modelarts.ipynb 与 app_93_modelarts.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_93 = D('''
# -*- coding: utf-8 -*-
# app_93_modelarts.py — ModelArts 与昇腾云:云上 AI 流程交互 ☁️
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ModelArts 与昇腾云 ☁️", layout="wide")
st.title("☁️ 第 93 课 · ModelArts 与昇腾云:云服务流程交互")

st.markdown("""
ModelArts 是华为云的 AI 一站式平台:像把整条"AI 生产线"外包给云厂商 ——
**要训练就开训练作业、要推理就发推理服务、不想写代码就交给自动学习**。
下面选一种服务、拖一下卡数与时长,实时算出费用并看流程时间线。
""")

st.sidebar.header("🎛️ 参数")
svc = st.sidebar.selectbox("服务类型", ["训练作业", "在线推理服务", "自动学习"])
pool = st.sidebar.radio("资源池", ["公共资源池", "专属资源池", "裸金属服务器"])
cards = st.sidebar.slider("昇腾 910 卡数", 1, 16, 8, 1)
hours = st.sidebar.slider("使用时长(小时)", 0.5, 24.0, 6.0, 0.5)
st.sidebar.caption("公共池按量计费最贵但零门槛;专属池包年便宜但要规划;裸金属完全独占最灵活。")

# 单价:公共池按卡时(¥/卡时),专属池按包年折算,裸金属按整机(¥/时)
if pool == "公共资源池":
    unit, label = 42.0, "¥/卡时(按量)"
elif pool == "专属资源池":
    unit, label = 26.0, "¥/卡时(包年折算)"
else:
    unit, label = 260.0, "¥/整机时"

if svc == "训练作业":
    total = unit * cards * hours
    steps = ["数据准备", "作业排队", "训练运行", "产物归档"]
    times = [0.5, 0.3, hours, 0.2]
else:
    total = unit * cards * hours * 1.6
    steps = ["模型发布", "服务拉起", "持续推理", "监控扩缩"]
    times = [0.3, 0.4, hours, 0.3]

c1, c2, c3, c4 = st.columns(4)
c1.metric("单价", f"{unit:.1f} {label}")
c2.metric("估算费用", f"¥{total:,.0f}")
c3.metric("昇腾 910 卡数", cards)
c4.metric("服务类型", svc)

st.subheader("🛠️ 作业流程时间线")
fig = go.Figure(go.Bar(x=steps, y=times, marker_color="#2e86c1",
                       text=[f"{t:.1f} h" for t in times], textposition="outside"))
fig.update_layout(title=f"{svc}({pool})的流程时间分布", yaxis_title="时长(小时)",
                  height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 费用 vs 使用时长(三种资源池)")
hs = np.arange(1, 25)
costs = {
    "公共资源池": 42.0 * cards * hs,
    "专属资源池": 26.0 * cards * hs,
    "裸金属服务器": 260.0 * hs,
}
fig2 = go.Figure()
colors = {"公共资源池": "#e74c3c", "专属资源池": "#27ae60", "裸金属服务器": "#2e86c1"}
for name, c in costs.items():
    fig2.add_trace(go.Scatter(x=hs, y=c, mode="lines", name=name,
                              line=dict(color=colors[name], width=3)))
fig2.add_vline(x=hours, line_dash="dash", line_color="#888")
fig2.update_layout(title="三种资源池的费用随使用时长增长(昇腾 910)",
                   xaxis_title="使用时长(小时)", yaxis_title="估算费用(¥)",
                   height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:云上的核心优势是**弹性** —— 想用 1000 卡就开 1000 卡,用完即还;
> 本地部署则胜在**可控与数据不出域**。选公共池跑实验、专属池跑生产、裸金属跑特殊负载,是常见分工。
""")
''')

NB = new_nb("第 93 课 · ModelArts 与昇腾云:把训练和推理搬上云",
            subtitle="训练作业 / 推理服务 / 自动学习 / 资源池 · 本地 vs 云端全面对比",
            emoji="☁️")

chapter_cover(NB,
    objectives=[
        "认识 ModelArts 全家桶:开发环境、数据管理、训练作业、模型管理、推理服务、自动学习",
        "理解资源池的三种形态:公共资源池、专属资源池、裸金属服务器",
        "掌握训练作业的完整生命周期与状态机:排队 → 调度 → 运行 → 完成/失败",
        "看懂自动学习(AutoML):不写代码如何完成从数据到模型的闭环",
        "会做本地部署 vs 云端部署的对比:成本、弹性、数据合规",
        "用 torch/随机过程模拟作业生命周期,matplotlib / plotly 可视化,配 App 交互",
    ],
    toc=[
        ("直觉:云食堂 vs 自家厨房", "云 = 弹性外包,本地 = 自建可控"),
        ("ModelArts 全家桶", "训练/推理/自动学习/资源池全景结构图"),
        ("本地 vs 云端对比", "成本、弹性、运维、合规一张表"),
        ("训练作业生命周期", "状态机模拟 + 时间线可视化"),
        ("自动学习:不写代码的 AI", "AutoML 五步流程"),
        ("成本账本", "三种资源池的费用对比"),
        ("配套 App", "app_93_modelarts.py:云服务流程交互"),
    ],
    links=[
        ("华为云 ModelArts 官网", "https://www.huaweicloud.com/product/modelarts.html"),
        ("ModelArts Studio 昇腾云", "https://www.huaweicloud.com/product/modelarts.html"),
        ("昇腾云服务文档", "https://www.hiascend.com"),
        ("MindSpore 云上安装指南", "https://www.mindspore.cn"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:云食堂 vs 自家厨房 🍚",
D('''
想象你要办一场千人宴会(训一个大模型):

- **自家厨房(本地部署)**:设备(GPU/昇腾服务器)自己买、厨师(运维)自己养、锅碗瓢盆
  (软件栈)自己配。前期投入巨大,但一切尽在掌握;
- **云食堂(ModelArts)**:直接租云厂商的厨房 —— 想用 100 口锅开 100 口,用完就退,
  厨师(托管运维)也由平台备好。贵在"按量付费",胜在"弹性无限"。

ModelArts 就是华为云的"AI 食堂",而且支持昇腾 NPU:你在本地没有昇腾硬件?
云上开几台昇腾 910 训练实例,马上就能跑。这就是本课的意义 —— 没有硬件也能用上昇腾生态。
'''))

NB.md("## 2️⃣ ModelArts 全家桶:一图看懂 🗺️",
D('''
ModelArts 把 AI 开发的完整流程都装进了一个平台:

1. **数据管理**:数据集标注、版本管理、增强;
2. **开发环境**:在线 Notebook(JupyterLab / VS Code),自带昇腾镜像,即开即用;
3. **训练作业**:提交训练任务,平台调度资源池,自动失败重试;
4. **模型管理**:把训练产物注册成模型,做版本与评估;
5. **推理服务**:一键发布为在线服务(OpenAI 兼容 API)或批量推理任务;
6. **自动学习(AutoML)**:填数据、选任务、点"开始",平台自动调参出模型;
7. **资源池**:公共(按量)、专属(包年)、裸金属(独占)三种形态。

下面把这张全家桶画出来:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.axis("off")
ax.text(4.8, 4.9, "ModelArts 全家桶:数据 → 开发 → 训练 → 模型 → 推理", fontsize=13,
        fontweight="bold", ha="center", color="#1f4e79")
comps = [
    ("数据管理", "标注 / 版本 / 增强", "#dbe7f4", "#1f4e79"),
    ("开发环境", "Notebook / VS Code", "#d9ead3", "#38761d"),
    ("训练作业", "昇腾卡调度 / 失败重试", "#fff2cc", "#7f6000"),
    ("模型管理", "注册 / 评估 / 版本", "#fce5cd", "#a64d17"),
    ("推理服务", "在线 API / 批量推理", "#e2d5f1", "#5b2c8f"),
]
x = 0.5; bw, bh = 1.55, 1.7
for i, (t, sub, fill, edge) in enumerate(comps):
    ax.add_patch(plt.Rectangle((x, 2.4), bw, bh, facecolor=fill, edgecolor=edge, lw=2))
    ax.text(x + bw/2, 2.4 + bh*0.68, t, ha="center", fontsize=11, fontweight="bold", color=edge)
    ax.text(x + bw/2, 2.4 + bh*0.35, sub, ha="center", va="center", fontsize=8, color="#333")
    if i < len(comps) - 1:
        ax.annotate("", xy=(x + bw + 0.12, 3.25), xytext=(x + bw - 0.05, 3.25),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="#555"))
    x += bw + 0.45
ax.add_patch(plt.Rectangle((0.5, 0.5), 5.2, 1.15, facecolor="#eaf2f8", edgecolor="#2c3e50", lw=1.8))
ax.text(3.1, 1.35, "资源池(昇腾 910/310P)", fontsize=11, fontweight="bold", ha="center", color="#2c3e50")
ax.text(3.1, 0.78, "公共(按量) | 专属(包年) | 裸金属(独占)", fontsize=9, ha="center", color="#333")
ax.annotate("", xy=(3.1, 2.4), xytext=(3.1, 1.65), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.set_xlim(0, 10.2); ax.set_ylim(0, 5.5)
plt.tight_layout()
'''),
"🎨 全景图:七个组件 + 资源池底座,一条流水线打通『数据到服务』。本课重点讲『训练作业 / 推理服务 / 自动学习』三块。")

NB.md("## 3️⃣ 本地 vs 云端:账要算清楚 ⚖️",
D('''
要不要上云,本质上是一笔账。我们用一张表 + 一组"打分"对比:

| 维度 | 本地自建 | ModelArts 云 |
|---|---|---|
| 硬件 | 一次性购买(GPU/昇腾) | 按量租用,无沉没成本 |
| 弹性 | 扩容要采购,按周/月计 | 开卡即用,用完即退 |
| 运维 | 自己管驱动/镜像/网络 | 平台托管,自带昇腾镜像 |
| 数据合规 | 数据不出域 | 需评估出域策略 |
| 昇腾可及性 | 需自购昇腾服务器 | 一键开通昇腾 910 |
'''))

NB.code(D('''
labels = ["前期投入", "弹性扩容", "运维负担", "昇腾可及性", "数据可控"]
local = [2, 1, 1, 1, 5]        # 本地:投入重、弹性差、但可控(1-5 打分,越高越好)
cloud = [5, 5, 4, 5, 3]
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, local, w, label="本地自建", color="#8e9aaf")
ax.bar(x + w/2, cloud, w, label="ModelArts 云", color="#2e86c1")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("优势打分 (1-5,越高越好)")
ax.set_title("本地 vs 云:五维度对比")
ax.legend(); plt.tight_layout()
'''),
"📊 打分对比:云在『弹性/昇腾可及性』完胜,本地只在『数据可控』占优 —— 这就是多数团队选择『云上昇腾』的原因。")

NB.md("## 4️⃣ 训练作业生命周期:一台状态机 🔁",
D('''
一个训练作业从提交到结束,要经过一系列**状态**,ModelArts 会全程跟踪:

- **QUEUING 排队**:作业在池子里排队等资源;
- **SCHEDULING 调度**:平台为你分配昇腾实例;
- **RUNNING 运行**:训练脚本在昇腾 910 上跑(可实时看日志);
- **COMPLETED / FAILED 完成 / 失败**:正常结束或异常退出(可配置失败自动重试)。

我们用随机过程模拟一次作业的完整生命周期,并画出时间线:
'''))

NB.code(D('''
import random
random.seed(7)
states = ["QUEUING", "SCHEDULING", "RUNNING", "COMPLETED"]
durations = [random.uniform(1, 3), random.uniform(0.5, 1.5), random.uniform(15, 25), 0.2]
print("作业生命周期(单位:秒,模拟):")
t = 0.0
for s, d in zip(states, durations):
    print(f"  {s:10s} {t:6.2f}s → {t+d:6.2f}s (耗时 {d:5.2f}s)")
    t += d

# 时间线甘特图
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.barh([0]*len(states), durations, left=np.cumsum([0] + durations[:-1]),
        height=0.5, color=["#95a5a6", "#f39c12", "#27ae60", "#2e86c1"])
for i, (s, d) in enumerate(zip(states, durations)):
    left = sum(durations[:i])
    ax.text(left + d/2, 0, s, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.set_yticks([]); ax.set_xlabel("时间 →"); ax.set_xlim(0, t)
ax.set_title("一次昇腾训练作业的完整生命周期")
plt.tight_layout()
'''),
"🔄 状态机跑一遍:注意 RUNNING 之前有等待与调度 —— 抢卡高峰时 QUEUING 可能等几小时,这是云上训练的『隐藏成本』。")

NB.md("## 5️⃣ 自动学习:不写代码的 AI 🤖",
D('''
自动学习(AutoML)面向"不想/不会写代码"的用户:只负责**喂数据 + 选任务 + 点开始**。
平台自动完成:数据切分 → 选网络结构 → 超参搜索 → 训练 → 评估 → 出模型,全程无人值守。

它和手写训练的区别就像**外卖点单 vs 自己做菜** —— 方便、快,但可定制性低。
下面画它的五步流水线,并用一个模拟"超参搜索"的小实验体会 AutoML 在做什么:
'''))

NB.code(D('''
# 模拟 AutoML 的超参搜索:随机采样几组配置,挑验证集分数最高的
import random
random.seed(0)
results = []
for i in range(12):
    lr = 10 ** random.uniform(-4, -2)                    # 学习率
    acc = random.uniform(0.5, 0.98) - abs(math.log10(lr) + 3) * 0.05   # 越靠近 1e-3 越好
    results.append((lr, max(acc, 0.4)))
best = max(results, key=lambda r: r[1])
lrs = [r[0] for r in results]; accs = [r[1] for r in results]
print(f"AutoML 共尝试 {len(results)} 组超参,最优 lr = {best[0]:.2e},验证分数 = {best[1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter([math.log10(l) for l in lrs], accs, s=70, color="#2e86c1", alpha=0.8)
ax.scatter(math.log10(best[0]), best[1], s=160, facecolors="none", edgecolors="#c0392b", lw=2.5, label="最优")
ax.set_xlabel("log10(学习率)"); ax.set_ylabel("验证分数(示意)")
ax.set_title("AutoML 超参搜索:尝 12 组,挑最好的")
ax.legend(); plt.tight_layout()
'''),
"🤖 AutoML 的本质:把『调参』变成『搜索』 —— 上面的随机采样只是最朴素的搜索,平台还内置贝叶斯等高级策略。")

NB.md("## 6️⃣ 成本账本:三种资源池怎么选 🧮",
D('''
云上昇腾的费用差异很大,核心是**资源池形态**:

- **公共资源池**:按量计费,随时开随时退,适合实验与短期任务(单价最贵);
- **专属资源池**:包年/包月预定,单价低不少,但要提前规划容量(生产常用);
- **裸金属服务器**:整机独占、性能可预期,适合大规模分布式训练与特殊场景。

下面按"8 卡昇腾 910"算一笔账,对比不同使用强度下的月费用:
'''))

NB.code(D('''
hours_per_day = np.arange(0, 25, 2)                      # 每天用几小时
n_days = 30
public = 42.0 * 8 * hours_per_day * n_days              # 公共:¥42/卡时
dedicated = 26.0 * 8 * hours_per_day * n_days           # 专属:¥26/卡时
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(hours_per_day, public/10000, "o-", color="#e74c3c", lw=2.2, label="公共资源池")
ax.plot(hours_per_day, dedicated/10000, "s-", color="#27ae60", lw=2.2, label="专属资源池")
ax.set_xlabel("每天使用小时数")
ax.set_ylabel("30 天估算费用(万元)")
ax.set_title("8 卡昇腾 910:公共 vs 专属资源池月费对比(示意单价)")
ax.legend(); ax.grid(True, ls="--", alpha=0.5); plt.tight_layout()
print(f"每天 8 小时:公共池 ≈ {public[-1]/10000:.1f} 万元 | 专属池 ≈ {dedicated[-1]/10000:.1f} 万元")
print("→ 常用即用专属,偶尔用用公共更划算。")
'''),
"🧮 账本:使用越频繁,专属池的优势越大;偶尔用一次,公共池零门槛更香 —— 云上省钱的核心是『按需匹配形态』。")

NB.md("## 7️⃣ 配套 App:云服务流程交互 🎛️",
D('''
运行同目录的 `app_93_modelarts.py`,**切换服务类型与资源池、拖卡数和时长**,
实时看流程时间线与三种资源池的费用对比曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_93_modelarts.py
```

浏览器打开 **http://localhost:8501**(也可 `--server.port 8693`)。完整源码如下:
'''))

NB.code(app_cell("app_93_modelarts.py", APP_93),
"📜 运行本 cell 会覆盖写入 `app_93_modelarts.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "ModelArts 全家桶:数据 → 开发环境 → 训练作业 → 模型管理 → 推理服务 → 自动学习",
        "资源池三形态:公共(按量)、专属(包年)、裸金属(独占),按使用强度选型",
        "训练作业是一台状态机:QUEUING → SCHEDULING → RUNNING → COMPLETED/FAILED",
        "自动学习 = 数据 + 选任务 + 点开始,平台自动做超参搜索与训练",
        "云上昇腾让『没有硬件也能用昇腾』成为现实,弹性是云的核心优势",
    ],
    practice=[
        "改第 4 节随机种子与时长分布,再跑一次作业生命周期,观察排队时长的波动",
        "给成本模型加一个『裸金属』档,把第 6 节的三条曲线画全并比较拐点",
        "模拟一次『失败重试』:RUNNING 后随机失败,自动重启,更新状态机并画时间线",
        "查 ModelArts 官网的计费页,更新本课的示意单价为当前真实价格",
    ],
    links=[
        ("华为云 ModelArts 产品页", "https://www.huaweicloud.com/product/modelarts.html"),
        ("ModelArts 文档中心", "https://support.huaweicloud.com/modelarts/index.html"),
        ("昇腾云服务", "https://www.hiascend.com"),
        ("MindSpore 云上部署", "https://www.mindspore.cn"),
    ])

out = str(Path(CH11) / "93_modelarts.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_93_modelarts.py"
app_path.write_text(APP_93 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
