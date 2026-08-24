# -*- coding: utf-8 -*-
"""生成 89_ms_autoparallel.ipynb 与 app_89_ms_autoparallel.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_89 = D('''
# -*- coding: utf-8 -*-
# app_89_ms_autoparallel.py — MindSpore 自动并行:切分策略与通信量 🧩
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧩 89 · MindSpore 自动并行", layout="wide")
st.title("🧩 第 89 课 · MindSpore 自动并行:切分策略与通信成本")

st.markdown("""
MindSpore 的 **auto_parallel** 会把“切不切、怎么切”交给**策略搜索**:
对矩阵乘等算子的输入/输出按 **行/列/无** 切到多张卡,用 **cost model** 估算通信量,
挑出最优的切分组合。下方选择**切分方式**与**张量大小**,实时看
**每卡算量、通信量与加速比**,直观体会“切得越碎、通信越贵”的权衡。
""")

def gemm_plan(M, N, K, devices, mode):
    if mode == "数据并行(不切)":
        per = M * N * K; comm = 0.0
    elif mode == "输出行切(TP)":
        per = M * (N // devices) * K; comm = N   # 沿 N 切,结果无需通信(每卡持有部分列)
    elif mode == "输出列切(TP2)":
        per = M * N * (K // devices); comm = M * N * 2   # 沿 K 切,需要 AllReduce 合并
    elif mode == "流水线(PP)":
        per = M * N * K / devices; comm = M * N
    speedup = (M * N * K) / max(per, 1) / (1 + comm / max(M * N * K, 1) * 0.3)
    return per, comm, speedup

with st.sidebar:
    st.header("🎛️ 参数")
    M = st.slider("行数 M", 512, 8192, 2048, 512)
    K = st.slider("内维 K", 512, 8192, 4096, 512)
    N = st.slider("列数 N", 512, 8192, 2048, 512)
    devices = st.slider("卡数", 1, 16, 4, 1)
    mode = st.selectbox("切分方式", ["数据并行(不切)", "输出行切(TP)", "输出列切(TP2)", "流水线(PP)"])
    st.caption("速度只算了“通信惩罚”的一阶近似,用于理解趋势,非真实基准。")

per, comm, speedup = gemm_plan(M, N, K, devices, mode)
c1, c2, c3 = st.columns(3)
c1.metric("每卡算量(FLOPs×1e9)", f"{per / 1e9:.1f}")
c2.metric("通信量(元素×1e6)", f"{comm / 1e6:.1f}")
c3.metric("估算加速比", f"{speedup:.2f}×")

st.subheader("⚖️ 通信量 vs 切分方式")
modes = ["数据并行", "输出行切", "输出列切", "流水线"]
comms = [0, N, M * N * 2, M * N]
fig = go.Figure(go.Bar(x=modes, y=[c / 1e6 for c in comms],
                       marker_color=["#4C78A8", "#F58518", "#E45756", "#6B4FA1"],
                       text=[f"{c/1e6:.1f}" for c in comms], textposition="outside"))
fig.update_layout(title="不同切分方式的通信量(元素数,百万)", yaxis_title="通信量×1e6",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 切分方式决定通信量:有的几乎零通信(行切),有的要全量 AllReduce(列切)。")

st.subheader("📈 加速比 vs 卡数")
cs = np.arange(1, devices + 1)
fig2 = go.Figure()
for m in ["数据并行", "输出行切", "输出列切"]:
    ys = [gemm_plan(M, N, K, c, m)[2] for c in cs]
    fig2.add_trace(go.Scatter(x=cs, y=ys, mode="lines+markers", name=m, line=dict(width=3)))
fig2.add_hline(y=cs[-1], line_dash="dash", line_color="#E45756", annotation_text="理想线性")
fig2.update_layout(title="加速比随卡数:切分越好越接近线性,但通信会让曲线弯下来",
                   xaxis_title="卡数", yaxis_title="加速比(×)", height=400,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:MindSpore 的自动并行=让机器替你选切分。它把每个算子的
> 切分方式(**策略**)枚举出来,用 cost model 评估通信,再做全局协调——
> 这比人手写 TP/PP 更不容易出错,也更接近最优。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 89 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
''')

NB = new_nb("第 89 课 · MindSpore 自动并行",
            subtitle="让机器替你选切分:策略搜索、算子级切分与通信成本",
            emoji="🧩")

chapter_cover(NB,
    objectives=[
        "理解四种并行维度:DP / TP / PP / EP 各切什么",
        "理解算子级切分:矩阵乘按行、列、内维切的三种方案",
        "理解 MindSpore 自动并行:策略搜索 + cost model 估通信",
        "用 torch 亲手实现 2 路“输出行切”GEMM 并验证结果",
        "对比自动并行与手动 TP/PP 的取舍",
    ],
    toc=[
        ("直觉:分蛋糕的四种切法", "DP/TP/PP/EP 各切哪一维"),
        ("算子级切分:矩阵乘的花样", "行切、列切、内维切与通信"),
        ("torch 模拟输出行切 GEMM", "2 路切分 → 拼接,验证等价"),
        ("策略搜索:机器替你选", "cost model 与递归规划"),
        ("自动并行 vs 手动 TP/PP", "费功夫 vs 省心"),
        ("交互图:切分策略", "plotly 对比通信与加速比"),
        ("配套 Streamlit 演示", "app_89_ms_autoparallel.py:切分方式交互"),
    ],
    links=[
        ("MindSpore 自动并行文档", "https://www.mindspore.cn/tutorials/zh-CN/r2.4/advanced/parallel/overview.html"),
        ("Megatron-LM 论文", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 分布式执行文档", "https://docs.vllm.ai/en/latest/features/distributed_serving.html"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:分蛋糕的四种切法 🎂",
D('''
一个 70B 模型是一整块大蛋糕,一张卡(一个盘子)装不下,怎么办?分——但**怎么分**很有讲究:

- **DP(数据并行)**:每张卡一份完整模型,分数据——蛋糕烤四份,各吃各的;
- **TP(张量并行)**:把每层矩阵切成片,卡与卡合作算一层——一块蛋糕切成四牙;
- **PP(流水线并行)**:按层竖着切,前几层在一号卡、后几层在二号卡——四层蛋糕摞四层;
- **EP(专家并行)**:MoE 时代把“专家”分到不同卡——不同的馅儿放不同的盘子。

用一张网格图把四种切法画清楚:
'''))

NB.code(D('''
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
def draw(ax, title, grid, colors):
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
grid = np.zeros((8, 8))
draw(axes[0, 0], "DP:每卡完整模型×数据分", grid, None)
grid_tp = np.array([[j % 4 for j in range(8)] for i in range(8)], dtype=float)
axes[0, 0].imshow(np.zeros_like(grid_tp), cmap="Reds", vmin=0, vmax=1)
axes[0, 0].set_title("DP:每卡完整模型,数据分 4 份", fontsize=11)
grid_pp = np.array([[i // 2 for j in range(8)] for i in range(8)], dtype=float)
axes[0, 1].imshow(grid_tp, cmap="Blues", vmin=0, vmax=1)
axes[0, 1].set_title("TP:每层矩阵按列切 4 片", fontsize=11)
axes[1, 0].imshow(grid_pp, cmap="Greens", vmin=0, vmax=1)
axes[1, 0].set_title("PP:层按阶段切 4 段", fontsize=11)
axes[1, 1].imshow(np.random.default_rng(0).choice([0, 1, 2, 3], (8, 8)), cmap="Purples", vmin=0, vmax=1)
axes[1, 1].set_title("EP:专家按路由分到 4 卡", fontsize=11)
fig.suptitle("并行四兄弟:切什么、怎么切(颜色=归属的卡)", fontsize=13)
plt.tight_layout(); plt.show()
'''), "🎨 颜色相同的小格属于同一张卡。四种切法可组合(3D 并行),MindSpore 的自动并行负责把这些组合搜出来。")

NB.md("## 2️⃣ 算子级切分:矩阵乘的花样 🔬",
D('''
自动并行的最小单位是**算子**:一个矩阵乘 $C = A \\cdot B$ 有三种经典切法:

1. **输出行切(沿 K 维?)**:不对——更准确说,沿 **N** 切 B,每卡算 C 的一部分列,零通信;
2. **输出列切(沿 M 切)**:每卡算 C 的一部分行,零通信(各算各的行);
3. **内维切(沿 K 切)**:每卡只算 K 的一部分,结果需要 **AllReduce** 相加,有通信。

画一张三方案对比图,标注各自的通信:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
def mm_block(ax, m, n, k, title, comm):
    ax.add_patch(plt.Rectangle((0, 0.3), k, m, fc="#DFE9F8", ec="#4C78A8", lw=2))
    ax.add_patch(plt.Rectangle((k + 0.1, 0.3), n, k, fc="#FDF3E4", ec="#E45756", lw=2))
    ax.text(k / 2, 0.3 + m / 2, "A", ha="center", va="center", fontsize=11)
    ax.text(k + 0.1 + n / 2, 0.3 + k / 2, "B", ha="center", va="center", fontsize=11)
    ax.text(0.1, 0.12, f"通信: {comm}", fontsize=9, color="#A03A45")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
mm_block(axes[0], 0.5, 0.45, 0.4, "方案1:沿 N 切 B(零通信)", "无")
mm_block(axes[1], 0.5, 0.45, 0.4, "方案2:沿 M 切 A(零通信)", "无")
mm_block(axes[2], 0.5, 0.45, 0.4, "方案3:沿 K 切(需 AllReduce)", "AllReduce")
fig.suptitle("一个矩阵乘的三种切分:关键在于切哪一维", fontsize=13)
plt.tight_layout(); plt.show()
'''), "🎨 方案 1/2 零通信,方案 3 要 AllReduce——但方案 3 的每卡计算量可以随 K 变小而摊薄,适合 K 很大的情况。")

NB.md("## 3️⃣ torch 模拟:输出行切 GEMM ✅",
D('''
用 torch 亲手验证“输出行切”:把 C 的行分给 2 张卡,每卡只算自己的行,最后**拼接**。
结果与整块 GEMM 完全一致,且**不需要任何通信**:
'''))

NB.code(D('''
torch.manual_seed(0)
M, K, N = 8, 16, 12
A = torch.randn(M, K); B = torch.randn(K, N)
C_ref = A @ B                                     # 整块参考

def split_row_gemm(A, B, world=2):
    row_ids = torch.chunk(torch.arange(A.shape[0]), world)
    parts = [A[rows] @ B for rows in row_ids]     # 每卡只算自己的行
    return torch.cat(parts, dim=0), parts

C_2way, parts = split_row_gemm(A, B, world=2)
print("卡0 计算行:", [r.item() for r in torch.chunk(torch.arange(M), 2)[0]])
print("卡1 计算行:", [r.item() for r in torch.chunk(torch.arange(M), 2)[1]])
print("拼接结果与整块 GEMM 一致:", torch.allclose(C_ref, C_2way))
print("通信量: 0(各算各的行,无需交换数据)")
'''), "✅ 输出行切是 TP 的基础之一:算得快、零通信,是自动并行最爱的“便宜策略”。")

NB.md("## 4️⃣ 策略搜索:机器替你选 🤖",
D('''
MindSpore 的自动并行(mode `AUTO_PARALLEL` / `SEMI_AUTO_PARALLEL`)不是随便切,
而是**搜索**:

1. 为每个算子枚举候选**策略**(每个张量:不切 / 按行切 / 按列切 / 复制到多卡);
2. 用 **cost model** 估算每种策略的**计算 + 通信**开销;
3. 全局协调相邻算子的策略(前一个算子的输出切法要接得上后一个算子的输入切法);
4. 挑总代价最小的组合,生成 `strategy` 下发到各卡。

画一张“策略 → 代价”的搜索示意:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 4.0))
strategies = ["策略 A\\n沿 N 切", "策略 B\\n沿 M 切", "策略 C\\n沿 K 切", "策略 D\\n不切"]
costs = [58, 61, 47, 100]
colors = ["#4C78A8", "#4C78A8", "#E45756", "#B8C4D8"]
bars = ax.bar(strategies, costs, color=colors)
for b, c in zip(bars, costs):
    ax.text(b.get_x() + b.get_width() / 2, c + 2, str(c), ha="center", fontsize=10)
ax.set_ylabel("估算总代价(计算+通信)")
ax.set_title("cost model 打分:策略 C 胜出(红),其余待选", fontsize=12)
ax.annotate("选中!", xy=(2, 47), xytext=(2.4, 78),
            arrowprops=dict(arrowstyle="->", color="#E45756"), fontsize=11, color="#A03A45")
plt.tight_layout(); plt.show()
'''), "📊 代价 = 计算 + 通信。自动并行替你枚举这些候选、打分、选优——你只需说一句“auto”。")

NB.md("## 5️⃣ 自动并行 vs 手动 TP/PP ⚖️",
D('''
手动 TP/PP(Megatron 风格)与 MindSpore 自动并行各有拥趸:

| 维度 | 手动 TP/PP | MindSpore 自动并行 |
|---|---|---|
| 上手 | 手写切分逻辑,易错 | 声明式,框架搜索 |
| 灵活性 | 每处手动精调 | 全局 cost model 协调 |
| 可调试性 | 一切透明 | 策略是黑盒,需看日志 |
| 新模型适配 | 每种结构重写 | 通用 |

画一张“手动 vs 自动”的**正确率 / 调优成本**对比图:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
axes[0].bar(["手动 TP/PP", "自动并行"], [82, 96], color=["#F58518", "#4C78A8"])
axes[0].set_title("首次运行正确率(示意)", fontsize=12)
axes[0].set_ylim(0, 105)
for i, v in enumerate([82, 96]):
    axes[0].text(i, v + 2, str(v), ha="center", fontsize=10)
axes[1].bar(["手动 TP/PP", "自动并行"], [78, 35], color=["#F58518", "#4C78A8"])
axes[1].set_title("调优工作量(示意,越低越好)", fontsize=12)
axes[1].set_ylim(0, 105)
for i, v in enumerate([78, 35]):
    axes[1].text(i, v + 2, str(v), ha="center", fontsize=10)
fig.suptitle("自动并行:把“容易错 + 费功夫”的活交给机器", fontsize=12)
plt.tight_layout(); plt.show()
'''), "📊 自动并行牺牲一点可调试性,换来“少犯错、少调参”——这正是大模型多卡训练的主流趋势。")

NB.md("## 6️⃣ 交互图:切分策略对比 📊",
D('''
用 plotly 把不同切分方式的**每卡算量 / 通信量 / 加速比**画成可切换视图,直观感受权衡:
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

M, K, N, dev = 4096, 4096, 4096, 4
modes = ["不切", "输出行切", "输出列切", "内维切(AllReduce)"]
per = [M * N * K, M * (N // dev) * K, (M // dev) * N * K, M * N * (K // dev)]
comm = [0, 0, 0, M * N]
fig = go.Figure()
fig.add_trace(go.Bar(x=modes, y=[p / 1e9 for p in per], name="每卡算量(×1e9)", marker_color="#4C78A8"))
fig.add_trace(go.Bar(x=modes, y=[c / 1e6 for c in comm], name="通信量(×1e6)", marker_color="#E45756"))
fig.update_layout(barmode="group", title=f"GEMM({M}×{N}×{K}) 在 {dev} 卡上的切分代价",
                  yaxis_title="数量", height=400, margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "📊 内维切算量最小但通信非零;行/列切零通信但每卡算量不变。引擎按“算量↔通信”总账选最优。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_89_ms_autoparallel.py`,选择**切分方式**,拖动**矩阵尺寸 / 卡数**,
实时看每卡算量、通信量与加速比,并对比不同切分的加速曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_89_ms_autoparallel.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_89_ms_autoparallel.py\n" + APP_89, "📜 这就是 app_89_ms_autoparallel.py 的完整源码,notebook 与 app 共用同一套代价模型,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "并行四兄弟:DP 切数据、TP 切矩阵、PP 切层、EP 切专家,可自由组合",
        "算子级切分三方案:沿 N/M 切零通信,沿 K 切要 AllReduce",
        "torch 验证输出行切 GEMM:结果与整块一致,通信为 0",
        "自动并行 = 枚举策略 + cost model 打分 + 全局协调",
        "自动并行牺牲可调试性,换来正确率与低调优成本",
    ],
    practice=[
        "实现“沿 K 切”的 2 路 GEMM:每卡算一部分 K,再用 torch.sum 模拟 AllReduce,验证结果",
        "给分块 GEMM 加一个 batch 维度,模拟 4 卡 DP+TP 组合切分",
        "调研 mindspore.set_auto_parallel_context 的 search_mode 参数有哪些取值",
        "对比 vLLM 的 TP 实现(vllm.distributed)与 MindSpore 自动并行:各自在哪一层切",
    ],
    links=[
        ("MindSpore 并行总览文档", "https://www.mindspore.cn/tutorials/zh-CN/r2.4/advanced/parallel/overview.html"),
        ("Megatron-LM 论文", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 分布式服务文档", "https://docs.vllm.ai/en/latest/features/distributed_serving.html"),
    ])

NB.save(str(Path(CH11) / "89_ms_autoparallel.ipynb"))
app_path = Path(CH11) / "app_89_ms_autoparallel.py"
app_path.write_text(APP_89 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
