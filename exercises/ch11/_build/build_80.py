# -*- coding: utf-8 -*-
"""生成 80_ms_distributed.ipynb 与 app_80_ms_dist.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_80 = D('''
# -*- coding: utf-8 -*-
# app_80_ms_dist.py — MindSpore 分布式训练:并行策略选择 🌐
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="MindSpore 分布式 🌐", layout="wide")
st.title("🌐 第 80 课 · MindSpore 分布式训练:并行策略选择")

st.markdown("""
MindSpore 支持**数据并行(DP)、模型并行(MP)、流水线并行(PP)与自动并行**。
下方选择模型规模与并行策略,对比**每卡显存、AllReduce 通信时间与理论吞吐**,
体会『切数据 vs 切模型』的成本差异。
""")

model = st.sidebar.selectbox("模型规模", ["7B", "70B", "405B"])
strategy = st.sidebar.radio("并行策略", ["数据并行 DP", "模型并行 MP(切权重)", "流水线并行 PP(切层)", "自动并行"])
gpus = st.sidebar.slider("卡数", 1, 64, 8, 1)
batch = st.sidebar.slider("batch(每卡)", 1, 64, 16, 1)
st.sidebar.caption("数据并行切『数据』、模型并行切『权重』、流水线切『层』 —— 三把剪刀切三处。")

params_b = {"7B": 7, "70B": 70, "405B": 405}[model]
bytes_el = 2                       # fp16
weights_total = params_b * bytes_el
bandwidth_gbps = 100               # 示意:昇腾 HCCS/网络带宽

if strategy == "数据并行 DP":
    w_per = weights_total
    kv_per = 0
    grad_comm = weights_total * 2 * (gpus - 1) / gpus    # AllReduce 通信量(每卡)
    comm_ms = grad_comm / (bandwidth_gbps / 8)
    speed = min(gpus, 16)          # 吞吐近线性但通信占用
elif strategy == "模型并行 MP(切权重)":
    w_per = weights_total / gpus
    kv_per = 0
    grad_comm = 0                  # 切权重的卡间通信为 allreduce 每算子
    comm_ms = weights_total * 2 / (bandwidth_gbps / 8)
    speed = gpus
elif strategy == "流水线并行 PP(切层)":
    w_per = weights_total / gpus
    kv_per = 0
    comm_ms = weights_total / gpus / (bandwidth_gbps / 8) * 0.5
    speed = gpus * 0.7             # 有气泡
else:
    w_per = weights_total / max(gpus, 2)
    kv_per = 0
    comm_ms = weights_total * 1.5 / (bandwidth_gbps / 8)
    speed = gpus * 0.9

c1, c2, c3, c4 = st.columns(4)
c1.metric("模型权重(总量)", f"{weights_total:.0f} GB")
c2.metric("每卡权重", f"{w_per:.1f} GB")
c3.metric("通信时间(示意)", f"{comm_ms:.1f} ms")
c4.metric("相对吞吐", f"{speed:.0f}")

st.subheader("📊 每卡显存 vs 通信时间")
strategies = ["数据并行 DP", "模型并行 MP", "流水线并行 PP"]
wps = [weights_total, weights_total / gpus, weights_total / gpus]
comms = [weights_total * 2 * (gpus - 1) / gpus, weights_total * 2, weights_total / gpus * 0.5]
fig = go.Figure()
fig.add_trace(go.Bar(x=strategies, y=wps, name="每卡权重(GB)", marker_color="#4C78A8",
                     text=[f"{v:.0f}" for v in wps], textposition="outside"))
fig.add_trace(go.Bar(x=strategies, y=comms, name="通信量(GB,示意)", marker_color="#E45756",
                     text=[f"{v:.0f}" for v in comms], textposition="outside"))
fig.update_layout(barmode="group", title=f"{model} 在 {gpus} 卡上:三策略的显存与通信",
                  yaxis_title="GB", height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 观察:DP 每卡都要装整份权重(显存最大、通信跟卡数走);MP/PP 摊薄权重但引入通信。")

st.subheader("📈 AllReduce 通信量随卡数变化")
cards = list(range(2, 65))
comm_dp = [weights_total * 2 * (g - 1) / g for g in cards]
fig2 = go.Figure(go.Scatter(x=cards, y=comm_dp, mode="lines+markers",
                            line=dict(color="#B279A2", width=3), name="DP AllReduce/卡"))
fig2.add_hline(y=weights_total * 2, line_dash="dash", line_color="#4C78A8",
               annotation_text="MP 每次权重同步量(固定)", annotation_position="top right")
fig2.update_layout(title=f"{model}:DP 通信量随卡数趋近于『两倍权重』,MP 则固定",
                   xaxis_title="卡数", yaxis_title="通信量(GB)", height=360)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Ring-AllReduce 下每卡通信量 = 2(N-1)/N × 数据量,卡越多单卡通信反而越小,但总通信线性涨。")

st.markdown("""
> 💡 **结论**:MindSpore 的自动并行帮你决定『三把剪刀怎么下』;推理场景(vLLM 的 TP/PP)
> 用的是同一套思想 —— 切权重省显存,切层降通信,数据并行提吞吐。
""")
''')

NB = new_nb("第 80 课 · MindSpore 分布式训练:数据并行、模型并行与自动并行",
            subtitle="三把剪刀切三处:数据 / 权重 / 层 —— AllReduce 在昇腾(HCCL),与 vLLM 的 TP/PP 同源",
            emoji="🌐")

chapter_cover(NB,
    objectives=[
        "理解数据并行 / 模型并行 / 流水线并行的『切法』与适用场景",
        "看懂 Ring-AllReduce 在昇腾(HCCL)上的通信量公式",
        "认识 MindSpore 自动并行(auto_parallel)的定位与好处",
        "用 matplotlib 画三种并行的架构示意与通信成本曲线",
        "与 vLLM 的 TP/PP/EP 对照,打通『训练并行 → 推理并行』的认知",
        "跑通配套 App:并行策略选择",
    ],
    toc=[
        ("直觉:三把剪刀切三处", "数据 / 权重 / 层,分别怎么切"),
        ("三种并行与自动并行", "DP / MP / PP + MindSpore auto_parallel"),
        ("AllReduce 在昇腾:HCCL", "Ring-AllReduce 通信量公式与计算"),
        ("通信成本曲线", "DP 通信随卡数趋近 2×权重,MP 固定"),
        ("与 vLLM 的 TP/PP 对照", "训练并行与推理并行的同源思想"),
        ("配套 Streamlit 演示", "app_80_ms_dist.py:并行策略选择"),
    ],
    links=[
        ("MindSpore 分布式训练文档", "https://www.mindspore.cn/docs/zh-CN/master/design/distributed_training_design.html"),
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
        ("vLLM 分布式推理文档", "https://docs.vllm.ai/en/latest/serving/distributed_serving.html"),
        ("昇腾 HCCL 文档", "https://www.hiascend.com/document"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:三把剪刀切三处 ✂️",
D('''
还记得第 35 课的四把剪刀吗?昇腾训练场景(以及 MindSpore)把它们简化成三把 + 一个"自动":

- **数据并行(DP)**:每卡一份完整模型,各吃各的 batch,每步梯度 **AllReduce** 同步 ——
  切的是**数据**;
- **模型并行(MP)**:把一层里的权重矩阵按列切开,每卡拿一半,矩阵乘时协同 —— 切的是**权重**;
- **流水线并行(PP)**:把网络按层切开,卡与卡像流水线一样接力 —— 切的是**层**;
- **自动并行(auto_parallel)**:你只写串行代码,框架自动搜索最省钱的切法(常是
  DP + MP + PP 的组合)。

用生活类比:数据并行是**开连锁店**(每家店菜谱一样,各接各的单);模型并行是**切菜分工**
(这家切葱那家切蒜,拼成一盘);流水线并行是**流水线车间**(裁缝缝纫质检分段接力)。
'''))

NB.md("## 2️⃣ 三种并行与自动并行:一张表看懂 🧭",
D('''
| 并行 | 切什么 | 每卡显存 | 通信模式 | 适用 |
|------|--------|----------|----------|------|
| **DP** | 数据/batch | 整份模型(最重) | AllReduce 梯度,每步一次 | 模型放得下、要吞吐 |
| **MP** | 权重矩阵 | 1/卡数(省最多) | 每算子 AllReduce | 单卡放不下 |
| **PP** | 网络层 | 1/卡数 | 层间点对点(稀疏) | 超深模型 |
| **auto_parallel** | 自动组合 | 自动 | 自动 | 不想手调并行度 |

**MindSpore 的 auto_parallel** 是它的招牌:你只管写 `model(train_data)` 这种串行代码,
框架分析通信量与显存,自动决定把哪个维度切到哪张卡 —— 相当于把三把剪刀交给机器人。
'''))

NB.code(D('''
# 三种并行 × 显存/通信 的示意对比(matplotlib)
labels = ["数据并行 DP", "模型并行 MP", "流水线并行 PP"]
per_card = [100, 12.5, 12.5]        # 每卡权重占比(示意,8 卡时 MP/PP=1/8)
comm = [15, 100, 5]                 # 通信强度(相对示意)
x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w/2, per_card, w, label="每卡权重占比(100=整份)", color="#4C78A8")
ax.bar(x + w/2, comm, w, label="通信强度(相对)", color="#E45756")
for xi, v in list(zip(x - w/2, per_card)) + list(zip(x + w/2, comm)):
    ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("三把剪刀的权衡:MP 最省显存但通信最勤,PP 通信最稀疏")
ax.legend(); plt.tight_layout()
'''),
"📊 权衡图:DP 显存最重通信适中,MP 显存最省但每算子都要 AllReduce,PP 通信最稀疏 —— 没有银弹。")

NB.md("## 3️⃣ AllReduce 在昇腾:HCCL 与 Ring-AllReduce 🔄",
D('''
数据并行的灵魂是**梯度 AllReduce**。昇腾上的集合通信库叫 **HCCL**(对标 NCCL),
默认采用 **Ring-AllReduce**:N 张卡排成一圈,数据切块沿环流动,两轮(Reduce-Scatter +
All-Gather)完成同步。每卡通信量:

$$\\text{每卡通信量} = \\dfrac{2(N-1)}{N} \\times \\text{数据量}$$

卡越多,$\\tfrac{N-1}{N}$ 越接近 1 —— **单卡通信量趋近『两倍数据量』**,不会随卡数爆涨,
这是 Ring 结构比朴素 AllReduce(每卡发 N-1 份)省得多的原因。算一下:
'''))

NB.code(D('''
def ring_comm(data_gb, n):
    """Ring-AllReduce 每卡通信量(GB): 2(N-1)/N × 数据量"""
    return 2 * (n - 1) / n * data_gb

weights_gb = 70 * 2       # 70B 模型 fp16 = 140GB(梯度同样大小)
for n in [2, 4, 8, 16, 32]:
    print(f"  N={n:>3}: 每卡通信 {ring_comm(weights_gb, n):7.1f} GB   (朴素 AllReduce: {weights_gb*(n-1):8.1f} GB)")
'''),
"✅ 结论:卡越多,单卡 Ring 通信越接近 2×数据量,而朴素 AllReduce 是 (N-1)× 数据量线性爆炸 —— HCCL/NCCL 都用 Ring(或更优的 Tree/Hierarchical)。")

NB.md("## 4️⃣ 通信成本曲线:卡数的两面性 📉",
D('''
画一张曲线,看清『加卡』的代价:DP 的梯度同步量随卡数**趋近饱和**(对数线),而 MP 的
权重同步是**固定量**;但注意,通信总量(Σ)其实随卡数线性增长 —— 所以卡不是越多越好,
带宽才是天花板:
'''))

NB.code(D('''
cards = np.arange(2, 33)
weights_gb = 140                      # 70B fp16 的梯度/权重
comm_dp = [ring_comm(weights_gb, n) for n in cards]       # 每卡
total_dp = [comm_dp[i] * cards[i] for i in range(len(cards))]  # 全网总和

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(cards, comm_dp, "o-", color="#4C78A8", lw=2.5, label="DP 每卡通信量(饱和)")
ax.plot(cards, total_dp, "s--", color="#E45756", lw=2, label="DP 全网通信总量(线性涨)")
ax.set_xlabel("卡数 N"); ax.set_ylabel("通信量(GB)")
ax.set_title("Ring-AllReduce 的『单卡饱和,总量线性』")
ax.legend(); plt.tight_layout()
'''),
"📊 双曲线:单卡通信趋近 2×权重(好事),但全网总量随卡数线性涨(带宽压力)—— 这就是『卡越多越快』的物理边界。")

NB.md("## 5️⃣ 与 vLLM 的 TP/PP 对照:训练并行 × 推理并行 🧭",
D('''
好消息是:训练并行的思想,推理引擎原样继承。对照表如下:

| 维度 | MindSpore 训练 | vLLM 推理 |
|------|----------------|-----------|
| 数据并行 | DP(梯度 AllReduce) | 多 worker 负载均衡(推理无梯度) |
| 模型并行 | MP(切权重) | **TP**(切矩阵,同一种切法) |
| 流水线并行 | PP(切层) | **PP**(切层 + KV 传递) |
| 专家并行 | —(MoE 训练) | **EP**(切专家) |
| 通信库 | HCCL | NCCL / HCCL(vllm-ascend) |

**关键差异**:训练每步都要 AllReduce 梯度;推理只要前向,TP 的 AllReduce 发生在
矩阵乘之间、KV cache 还要按 TP 切分(见第 35 课)。所以昇腾上跑 vLLM,
MP↔TP 是直接映射 —— 你学到的分布式直觉,两头通用。
'''))

NB.code(D('''
# 训练 vs 推理:通信频率与数据流差异示意图
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axis("off")
ax.add_patch(plt.Rectangle((0.4, 2.4), 3.6, 0.9, facecolor="#fdebd0", edgecolor="#7f5f01", lw=1.5))
ax.text(2.2, 2.85, "训练:每步 前向→反向→AllReduce 梯度", ha="center", fontsize=10)
ax.add_patch(plt.Rectangle((0.4, 0.4), 3.6, 0.9, facecolor="#d6eaf8", edgecolor="#1f4e79", lw=1.5))
ax.text(2.2, 0.85, "推理:只有前向,TP 矩阵间 AllReduce", ha="center", fontsize=10)
ax.add_patch(plt.Rectangle((5.2, 2.4), 3.6, 0.9, facecolor="#fdebd0", edgecolor="#7f5f01", lw=1.5))
ax.text(7.0, 2.85, "HCCL:集合通信库(昇腾)", ha="center", fontsize=10)
ax.add_patch(plt.Rectangle((5.2, 0.4), 3.6, 0.9, facecolor="#d6eaf8", edgecolor="#1f4e79", lw=1.5))
ax.text(7.0, 0.85, "NCCL / HCCL:vLLM 通用", ha="center", fontsize=10)
ax.set_xlim(0, 9.3); ax.set_ylim(0, 3.6)
ax.set_title("训练与推理:同一套通信骨架,不同的通信节奏", fontsize=12)
plt.tight_layout()
'''),
"🎨 骨架图:训练多一步反向梯度同步;推理则全靠前向里的 TP AllReduce —— 昇腾的 HCCL 两头都在用。")

NB.md("## 6️⃣ 配套 Streamlit 演示:并行策略选择 🎛️",
D('''
运行同目录下的 `app_80_ms_dist.py`,可以**选择模型规模、并行策略、卡数与 batch**,
实时看每卡权重、通信时间与相对吞吐:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_80_ms_dist.py
```

浏览器打开 **http://localhost:8501**。建议选 70B、切到模型并行,拖动卡数从 1 到 64,
观察每卡权重从 140GB 摊到 ~2GB;再切回数据并行对比通信量。完整源码如下
(与同目录 `app_80_ms_dist.py` 一字不差):
'''))

NB.code(app_cell("app_80_ms_dist.py", APP_80),
"📜 运行本 cell 会覆盖写入 `app_80_ms_dist.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "三把剪刀切三处:DP 切数据、MP 切权重、PP 切层;auto_parallel 让框架替你下剪刀",
        "Ring-AllReduce 每卡通信 2(N-1)/N × 数据量,卡越多单卡越趋近 2× 数据量",
        "HCCL 是昇腾的集合通信库(对标 NCCL),训练梯度与推理 TP 都靠它",
        "通信总量的物理边界:单卡饱和、总量线性 —— 卡越多,带宽是天花板",
        "vLLM 的 TP/PP 与 MindSpore 的 MP/PP 同源:训练并行的直觉,推理直接复用",
    ],
    practice=[
        "把 ring_comm 的 N 换成 128,对比每卡通信与朴素 AllReduce,体会 Ring 的优势",
        "用 matplotlib 画 7B / 70B / 405B 三种模型在 DP 下每卡显存随卡数的曲线,标注 16GB 卡能装下哪几档",
        "查 MindSpore auto_parallel 的搜索维度(策略候选:按维度切、recompute 等),写 3 条要点",
        "结合第 35 课,把 vLLM 的 TP/PP/EP 与本章 MP/PP 做一张完整的跨章对照表",
    ],
    links=[
        ("MindSpore 分布式训练设计文档", "https://www.mindspore.cn/docs/zh-CN/master/design/distributed_training_design.html"),
        ("昇腾 HCCL 文档", "https://www.hiascend.com/document"),
        ("vLLM 分布式推理文档", "https://docs.vllm.ai/en/latest/serving/distributed_serving.html"),
        ("Ring-AllReduce 论文", "https://arxiv.org/abs/1611.05736"),
    ])

out = str(Path(CH11) / "80_ms_distributed.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_80_ms_dist.py"
app_path.write_text(APP_80 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
