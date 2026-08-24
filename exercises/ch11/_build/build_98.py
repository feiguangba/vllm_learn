# -*- coding: utf-8 -*-
"""生成 98_ascend_cluster.ipynb 与 app_98_ascend_cluster.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_98 = D('''
# -*- coding: utf-8 -*-
# app_98_ascend_cluster.py — 昇腾集群通信:HCCL AllReduce 交互 🌐
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾集群通信 HCCL 🌐", layout="wide")
st.title("🌐 第 98 课 · 昇腾集群通信:HCCL 与 AllReduce 交互")

st.markdown("""
多卡推理(张量并行)需要把每张卡的梯度/中间结果**求和汇总**,这就是集合通信。
HCCL 是昇腾的集合通信库(对标 NCCL),最经典的是 **Ring AllReduce**:把数据切成 N 片,
沿环走 **2(N-1) 步**,每步每卡只搬 1/N 的数据。拖一拖卡数,看步数与通信量怎么变。
""")

st.sidebar.header("🎛️ 参数")
n_rank = st.sidebar.slider("卡数(rank)", 2, 32, 8, 1)
algo = st.sidebar.radio("通信算法", ["Ring AllReduce", "Tree(Recursive Halving)"])
data = st.sidebar.slider("每卡数据量(MB)", 1, 512, 128, 1)
lat = st.sidebar.slider("单步延迟(us)", 5, 50, 20, 5)
st.sidebar.caption("Ring 步骤多但每步搬得少;Tree 步骤少但每步搬得多 —— 大数据量 Ring 胜,小数据量 Tree 胜。")

steps = 2 * (n_rank - 1) if algo == "Ring AllReduce" else 2 * int(np.ceil(np.log2(n_rank)))
per_step = data / n_rank if algo == "Ring AllReduce" else data / 2
total_t = steps * (lat + per_step * 12)          # 时间 = 步数 × (延迟 + 每步数据 × 带宽系数)
t_other = (2 * int(np.ceil(np.log2(n_rank))) * (lat + data / 2 * 12)) if algo == "Ring AllReduce" \
    else (2 * (n_rank - 1) * (lat + data / n_rank * 12))

c1, c2, c3, c4 = st.columns(4)
c1.metric("通信步数", steps)
c2.metric("每步数据量", f"{per_step:.1f} MB")
c3.metric("估算总耗时", f"{total_t/1000:.2f} ms")
c4.metric("另一算法耗时", f"{t_other/1000:.2f} ms")

st.subheader("📈 卡数 → 通信步数与数据量")
ns = np.arange(2, 33)
if algo == "Ring AllReduce":
    s = 2 * (ns - 1); d = data / ns
else:
    s = 2 * np.ceil(np.log2(ns)); d = data / 2
fig = go.Figure()
fig.add_trace(go.Scatter(x=ns, y=s, mode="lines", name="步数",
                         line=dict(color="#2e86c1", width=3)))
fig.add_trace(go.Scatter(x=ns, y=d, mode="lines", name="每步数据量(MB)",
                         line=dict(color="#e67e22", width=3), yaxis="y2"))
fig.update_layout(title=f"{algo}:卡数增多时,步数与每步数据量的取舍",
                  xaxis_title="rank 数", yaxis=dict(title="通信步数"),
                  yaxis2=dict(title="每步数据量(MB)", overlaying="y", side="right"),
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🛰️ 环形拓扑示意图")
fig2 = go.Figure()
theta = np.linspace(0, 2*np.pi, n_rank, endpoint=False)
for i in range(n_rank):
    x0, y0 = np.cos(theta[i]), np.sin(theta[i])
    x1, y1 = np.cos(theta[(i+1) % n_rank]), np.sin(theta[(i+1) % n_rank])
    fig2.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                              line=dict(color="#bbb", width=2), showlegend=False))
    fig2.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers+text",
                              marker=dict(size=22, color="#2e86c1"), text=[f"R{i}"],
                              textposition="top center", showlegend=False))
fig2.update_layout(title=f"{n_rank} 卡环形拓扑(数据沿环流动)",
                   height=340, xaxis=dict(showticklabels=False, range=[-1.4, 1.4]),
                   yaxis=dict(showticklabels=False, range=[-1.4, 1.4]),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:Ring 的通信量(每卡)是 `2S(N-1)/N`,随卡数增多趋近 `2S`;Tree 的步数
> 只随 `log2(N)` 增长。工程上:**大张量、卡数多 → Ring;小张量、延迟敏感 → Tree/Hierarchical**。
> vLLM-Ascend 的张量并行与数据并行正是靠 HCCL 这些原语跑起来的。
""")
''')

NB = new_nb("第 98 课 · 昇腾集群通信:HCCL 与 AllReduce",
            subtitle="集合通信原语 · Ring AllReduce 算法模拟 · 与 NCCL 对照",
            emoji="🌐")

chapter_cover(NB,
    objectives=[
        "理解集合通信原语:AllReduce / AllGather / ReduceScatter / Broadcast 各干什么",
        "认识 HCCL:昇腾的集合通信库,与 NCCL 一一对照",
        "吃透 Ring AllReduce:2(N-1) 步,每步每卡只搬 1/N 的数据",
        "用 numpy 亲自动手模拟 4 卡 Ring AllReduce 并验证结果",
        "对比 Ring 与 Tree(Recursive Halving)的步数与通信量",
        "理解 HCCL 在 vLLM-Ascend 张量并行 / MindSpore 分布式中的角色",
    ],
    toc=[
        ("直觉:传话游戏与合奏", "多卡通信 = 一群人交换信息"),
        ("集合通信原语", "AllReduce / AllGather / Broadcast 一图看懂"),
        ("Ring AllReduce 详解", "ReduceScatter + AllGather 两段式"),
        ("动手模拟(4 卡)", "numpy 实现,验证结果正确"),
        ("Ring vs Tree", "步数、通信量、延迟三张牌怎么打"),
        ("HCCL 与 vLLM-Ascend", "张量并行/EP 如何用上 HCCL"),
        ("配套 App", "app_98_ascend_cluster.py:集群通信交互"),
    ],
    links=[
        ("昇腾 HCCL 文档", "https://www.hiascend.com/document"),
        ("NCCL 官方文档(对照)", "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html"),
        ("NCCL 论文(ring allreduce 出处)", "https://arxiv.org/abs/1807.05253"),
        ("vLLM-Ascend", "https://github.com/vllm-project/vllm-ascend"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:传话游戏与班级合奏 🎶",
D('''
一个模型拆到 8 张昇腾卡上(张量并行),每张卡只存了"一部分模型",但最终输出需要**全部卡
的信息**。多卡之间的这种"交换信息",就叫**集合通信(Collective Communication)**。

- **AllReduce**:所有人的数加起来,结果发给大家(训练里算梯度均值、推理里算注意力汇总);
- **AllGather**:每个人的"独有部分"广播给所有人,大家凑齐完整数据;
- **ReduceScatter**:先归约(求和)再分片发回;
- **Broadcast**:一个人把数据广播给所有人。

这些动作听起来像**传话游戏** —— 怎么传最省时间,就是本课的核心。昇腾上干这活儿的库叫
**HCCL**(Huawei Collective Communication Library),和 GPU 上的 **NCCL** 是一对孪生兄弟。
'''))

NB.md("## 2️⃣ 集合通信原语:一图看懂 🗺️",
D('''
把四个常用原语画成"谁给谁发什么"的关系图(箭头 = 数据流向):
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8))
prims = [
    ("AllReduce", "sum → 所有人", ["R0", "R1", "R2", "R3"]),
    ("AllGather", "每人的片 → 全人", ["R0", "R1", "R2", "R3"]),
    ("ReduceScatter", "sum → 每人一片", ["R0", "R1", "R2", "R3"]),
    ("Broadcast", "R0 → 所有人", ["R0", "R1", "R2", "R3"]),
]
for ax, (name, desc, ranks) in zip(axes, prims):
    for i, r in enumerate(ranks):
        ax.plot(i, 0, "o", color="#2e86c1", ms=16)
        ax.text(i, -0.08, r, ha="center", va="top", fontsize=8)
    for i in range(len(ranks) - 1):
        ax.annotate("", xy=(i+1, 0), xytext=(i, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#e67e22"))
    ax.set_xlim(-0.6, len(ranks) - 0.4); ax.set_ylim(-0.45, 0.55)
    ax.axis("off")
    ax.set_title(f"{name}\\n{desc}", fontsize=10)
fig.suptitle("集合通信四大原语(箭头=数据流向,从 R0 视角)", fontsize=13, y=1.02)
plt.tight_layout()
'''),
"🗺️ 原语对照:AllReduce 是最常用的『求和广播』;Ring 算法把 AllReduce 拆成 ReduceScatter + AllGather 两段。")

NB.md("## 3️⃣ Ring AllReduce:两段式传话 🔗",
D('''
最优雅的 AllReduce 实现是 **Ring AllReduce**(NCCL 与 HCCL 的默认主力):

**把 N 张卡排成一个环,每卡的数据切成 N 片**,然后走两段:

1. **ReduceScatter 段(N-1 步)**:每步每卡把自己的第 k 片发给下一卡,并从上一卡收到
   第 k-1 片就地相加 —— N-1 步后,每卡持有一片"全局和";
2. **AllGather 段(N-1 步)**:把这片"全局和"沿环再转一圈,每步再发一片给下一卡 ——
   N-1 步后,所有卡都拿到了完整结果。

总步数 **2(N-1)**,每步每卡只搬 **S/N** 的数据 —— 通信量被平摊到所有卡上,不堵车。
下面用 numpy 真跑一遍 4 卡:
'''))

NB.code(D('''
def ring_allreduce(tensors):
    """Ring AllReduce:输入 N 个同尺寸张量(每卡一个),返回全 N 个『总和张量』
    每卡数据切成 N 片 → ReduceScatter 段(N-1 步) + AllGather 段(N-1 步)。"""
    N = len(tensors)
    S = tensors[0].numel()
    C = S // N
    data = [t.reshape(N, C).clone() for t in tensors]      # [rank][chunk]
    for s in range(N - 1):                                 # 段1:ReduceScatter
        new = [d.clone() for d in data]
        for i in range(N):
            recv_c = (i - s - 2) % N                       # 本卡累加的块号
            src = (i - 1) % N
            new[i][recv_c] = data[i][recv_c] + data[src][recv_c]   # 收到上家并就地相加
        data = new
    for s in range(N - 1):                                 # 段2:AllGather
        new = [d.clone() for d in data]
        for i in range(N):
            send_c = (i - s - 1) % N                       # 本卡已归约好的块号
            src = (i - 1) % N
            new[i][send_c] = data[src][send_c]             # 收到上家的完整块,覆盖
        data = new
    return [d.reshape(S).clone() for d in data]

N = 4; S = 8
torch.manual_seed(0)
tensors = [torch.randn(S) for _ in range(N)]
result = ring_allreduce(tensors)
truth = sum(tensors)
err = max((r - truth).abs().max().item() for r in result)
print(f"4 卡 Ring AllReduce 完成:总步数 = 2×({N}-1) = {2*(N-1)}")
print(f"每卡结果与真实总和的最大误差 = {err:.2e} (≈0 即正确)")
'''),
"🔗 亲手实现:4 卡、8 元素,ReduceScatter + AllGather 两段各 3 步,最终每卡都拿到『全局和』。")

NB.md("## 4️⃣ 把每一步画出来:数据怎么流动 🎨",
D('''
上面算法的正确性不容易一眼看穿。把"每一卡每一块数据归约到哪一步完成"画成热力图:
横轴是时间步,纵轴是卡,颜色表示"该卡此刻正对哪一块数据做归约" —— 能直观看到
ReduceScatter 阶段的和逐步"合拢":
'''))

NB.code(D('''
steps = 2 * (N - 1)
grid = np.zeros((N, steps), dtype=int)
for step in range(N - 1):                    # ReduceScatter 段
    for i in range(N):
        block_off = (i - step - 1) % N
        grid[i, step] = block_off + 1        # 正在对第 block_off 片做累加
for step in range(N - 1, 2 * (N - 1)):       # AllGather 段
    for i in range(N):
        grid[i, step] = -1                   # 广播片

fig, ax = plt.subplots(figsize=(8.5, 3.4))
im = ax.imshow(grid, cmap="tab20c", aspect="auto")
ax.set_xticks(range(steps)); ax.set_xticklabels([f"t{s+1}" for s in range(steps)])
ax.set_yticks(range(N)); ax.set_yticklabels([f"R{i}" for i in range(N)])
ax.set_xlabel("时间步 →"); ax.set_ylabel("卡")
ax.axvline(N - 1.5, color="#c0392b", ls="--", lw=2)
ax.text(N - 1.0, -1.3, "ReduceScatter 段", fontsize=9, color="#c0392b", ha="center")
ax.text(N + (N - 1) / 2 - 0.5, -1.3, "AllGather 段", fontsize=9, color="#2e86c1", ha="center")
ax.set_title("4 卡 Ring AllReduce:每卡在每个时间步处理的块(色块=块号)")
plt.tight_layout()
'''),
"🎨 热力图:左半段(红虚线左侧)数据片沿环求和,右半段把完整和广播给所有人 —— 两段各 3 步,清晰可见。")

NB.md("## 5️⃣ Ring vs Tree:三张牌怎么打 🃏",
D('''
除了 Ring,还有 **Tree / Recursive Halving** 算法。两者的取舍:

| 维度 | Ring AllReduce | Tree(Recursive Halving) |
|---|---|---|
| 步数 | 2(N-1),随卡数线性增长 | 约 2·log2(N),随卡数对数增长 |
| 每步每卡数据量 | S/N(越多人越省) | 约 S/2(固定) |
| 适合场景 | 大张量、卡数多(训练/推理) | 小张量、延迟敏感(短消息) |

建一个简单时间模型 `总耗时 = 步数 × (延迟 + 每步数据量 / 带宽)`,扫卡数对比:
'''))

NB.code(D('''
lat_us = 20.0; bw_mb = 40.0          # 单步延迟 20us,带宽 40MB/us(示意)
S = 128.0                            # 每卡 128MB
ns = np.arange(2, 33)
ring = 2 * (ns - 1) * (lat_us + S / ns / bw_mb * 1000)      # us
tree = 2 * np.log2(ns) * (lat_us + S / 2 / bw_mb * 1000)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ns, ring / 1000, "o-", color="#2e86c1", lw=2, label="Ring AllReduce")
ax.plot(ns, tree / 1000, "s-", color="#e67e22", lw=2, label="Tree(Recursive Halving)")
ax.set_xlabel("rank 数"); ax.set_ylabel("估算总耗时 (ms)")
ax.set_title("大数据量(128MB/卡):卡越多 Ring 优势越明显")
ax.legend(); ax.grid(True, ls="--", alpha=0.5); plt.tight_layout()

# 小数据量场景:Tree 反而赢
Ssmall = 1.0
ring_s = 2 * (ns - 1) * (lat_us + Ssmall / ns / bw_mb * 1000)
tree_s = 2 * np.log2(ns) * (lat_us + Ssmall / 2 / bw_mb * 1000)
fig2, ax2 = plt.subplots(figsize=(7.5, 4))
ax2.plot(ns, ring_s / 1000, "o-", color="#2e86c1", lw=2, label="Ring AllReduce")
ax2.plot(ns, tree_s / 1000, "s-", color="#e67e22", lw=2, label="Tree(Recursive Halving)")
ax2.set_xlabel("rank 数"); ax2.set_ylabel("估算总耗时 (ms)")
ax2.set_title("小数据量(1MB/卡):Tree 靠『少步数』胜出")
ax2.legend(); ax2.grid(True, ls="--", alpha=0.5); plt.tight_layout()
'''),
"🃏 两张对比图:大消息 Ring 赢,小消息 Tree 赢 —— 这就是 HCCL/NCCL 都提供多种算法、按消息大小自动切换的原因。")

NB.md("## 6️⃣ HCCL 与 vLLM-Ascend:多卡推理的底座 🏗️",
D('''
HCCL 在昇腾生态里干三件大事:

1. **张量并行(TP)**:模型权重切到多卡,每步 forward 需要跨卡交换 —— AllReduce / AllGather
   天天用(vLLM-Ascend 的 `--tensor-parallel-size`);
2. **数据并行(DP)**:每卡一份完整模型,训练时 AllReduce 梯度;
3. **专家并行(EP)**:MoE 模型的专家路由,All-to-All 类通信(vLLM-Ascend 0.9 起支持大规模 EP)。

把"哪些原语用在哪儿"画一张对照表:
'''))

NB.code(D('''
scenarios = ["张量并行(TP)", "数据并行(DP)", "专家并行(EP)", "流水并行(PP)"]
prims_used = ["AllReduce / AllGather", "AllReduce(梯度)", "All-to-All", "Point-to-point"]
fig, ax = plt.subplots(figsize=(8, 3.4))
y = np.arange(len(scenarios))
ax.barh(y, [5, 4, 3, 2], color=["#2e86c1", "#27ae60", "#e67e22", "#8e44ad"])
for i, (sc, pr) in enumerate(zip(scenarios, prims_used)):
    ax.text(5.1, i, pr, va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(scenarios)
ax.set_xlim(0, 6.6); ax.set_xticks([])
ax.set_title("vLLM-Ascend / MindSpore 分布式场景用到的主要 HCCL 原语")
plt.tight_layout()
'''),
"🏗️ 场景对照:HCCL 是昇腾多卡推理的『通信血管』 —— 没有它,张量并行的大模型推理就跑不起来。")

NB.md("## 7️⃣ 配套 App:集群通信交互 🎛️",
D('''
运行同目录的 `app_98_ascend_cluster.py`,**拖卡数、切算法、调数据量与单步延迟**,
实时看通信步数、每步数据量、两种算法的耗时对比,以及环形拓扑图:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_98_ascend_cluster.py
```

浏览器打开 **http://localhost:8501**(也可 `--server.port 8698`)。完整源码如下:
'''))

NB.code(app_cell("app_98_ascend_cluster.py", APP_98),
"📜 运行本 cell 会覆盖写入 `app_98_ascend_cluster.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "集合通信原语:AllReduce / AllGather / ReduceScatter / Broadcast,AllReduce 最常用",
        "HCCL = 昇腾版 NCCL,原语与拓扑思路一一对应",
        "Ring AllReduce = ReduceScatter(N-1 步) + AllGather(N-1 步),每步每卡搬 S/N",
        "大消息 Ring 优(平摊带宽),小消息 Tree 优(步数少) —— 库会按大小自动切换",
        "HCCL 是 vLLM-Ascend 张量并行/专家并行与 MindSpore 分布式的通信底座",
    ],
    practice=[
        "把第 3 节的 N 改成 8、S 改成 16,重跑并核对最大误差仍为 0",
        "给 ring_allreduce 加一个『统计每卡通信量』的计数器,验证总量 = 2S(N-1)/N",
        "在第 5 节加一个 8MB 的中间场景,找出 Ring 与 Tree 的交叉点在哪",
        "查 HCCL 文档,列出它支持的算法与 NCCL 支持算法的对照表",
    ],
    links=[
        ("昇腾 HCCL 集合通信文档", "https://www.hiascend.com/document"),
        ("NCCL 文档", "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html"),
        ("《Bringing HPC Techniques to Deep Learning》Ring AllReduce", "https://arxiv.org/abs/1807.05253"),
        ("vLLM-Ascend", "https://github.com/vllm-project/vllm-ascend"),
    ])

out = str(Path(CH11) / "98_ascend_cluster.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_98_ascend_cluster.py"
app_path.write_text(APP_98 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
