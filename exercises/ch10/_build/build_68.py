# -*- coding: utf-8 -*-
"""生成 68_codegen_schedule.ipynb(本课不配 app)"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH10, finalize
from pathlib import Path

NB = new_nb("第 68 课 · 代码生成与调度:决定循环怎么走",
            subtitle="loop 分块 · 向量化 · tiling 决策 · Halide 风格调度 —— 同一算法,不同调度性能天差地别",
            emoji="🏭")

chapter_cover(NB,
    objectives=[
        "理解『算法』与『调度』分离的思想(Halide 风格):算法说算什么,调度说怎么算",
        "掌握核心调度原语:split(分块)、reorder(重排)、vectorize(向量化)、parallel(并行)",
        "理解 tiling 决策:分块大小影响缓存/寄存器局部性,过大过小都不好",
        "理解 CPU 与 GPU 调度的差异:缓存分块 vs 线程/块分块",
        "用 matplotlib 画出循环分块、向量化、CPU/GPU 调度示意与性能曲线",
        "用一个『调度生成器』亲手生成不同调度下的伪代码",
    ],
    toc=[
        ("直觉:调度 = 怎么切菜", "算法说『切土豆』,调度说『怎么切、切多细、几个人一起切』"),
        ("算法与调度分离", "Halide 的核心思想:同一算法,换调度即换性能"),
        ("四大调度原语", "split / reorder / vectorize / parallel"),
        ("tiling 决策与局部性", "分块大小 vs 缓存命中,过大过小都不好"),
        ("CPU vs GPU 调度", "缓存分块 vs 线程/块分块,机制不同"),
        ("调度生成器实战", "给定参数生成不同调度的伪代码"),
    ],
    links=[
        ("Halide 论文", "https://people.csail.mit.edu/nickolai/papers/ragan-kelley-halide.pdf"),
        ("TVM TensorIR 调度", "https://tvm.apache.org/docs/"),
        ("LLVM 循环优化", "https://llvm.org/docs/LoopTerminology.html"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:调度 = 怎么切菜 🥔",
D('''
把一道菜写清楚需要两件事:**算法** 和 **调度**。

- **算法(algorithm)**:算什么。比如"把 1024 个土豆都切成丁";
- **调度(schedule)**:怎么算。比如"横着切还是竖着切、每次切多大一块、几个人同时切"。

关键洞察:**同样一道菜,切法不同,时间天差地别**。编译器里也一样 —— **同一段计算,换个循环
怎么走(调度),性能能差好几倍甚至一个数量级**。Halide 把这种思想做到极致:**算法与调度分离**,
你只需要改一行调度描述,编译器就生成另一种循环结构。这一课就讲调度里的几个核心"切法"。
'''))

NB.md("## 2️⃣ 算法与调度分离(Halide 风格)🧩",
D('''
Halide 的公式是:

$$\\text{output}(x, y) = \\text{some computation}$$(算法)
$$\\text{pipeline.vectorize}(x, 4).\\text{parallel}(y)$$(调度)

算法描述"每个输出像素怎么算",调度描述"循环怎么组织"。同一个算法,可以:
- 朴素嵌套循环: `for y { for x { ... } }`
- 分块(tile): `for y_t { for x_t { for y_in { for x_in {...} } } }`
- 向量化: `for y { for x_vec { 一次处理 4 个 } }`
- 并行: `parallel for y { ... }`

下面我们写一个"调度生成器":输入分块大小与开关,输出对应的伪代码循环结构,亲眼看到调度如何改变代码形态。
'''))

NB.code(D('''
def schedule(N=8, T=4, reorder_=True, vectorize_=True, parallel_=True):
    """给定 N(矩阵边长)、T(分块大小)与开关,生成一段伪代码"""
    lines = []
    indent = "    "
    if not reorder_:
        # 朴素:一行行算
        lines.append(f"for y in 0..{N}:")
        lines.append(f"{indent}for x in 0..{N}:")
        op = "y*N + x"
        if vectorize_:
            lines.append(f"{indent*2}vectorize: 一次算 4 个 x 的元素(x_vec)")
        else:
            lines.append(f"{indent*2}A[y*N+x] = f(x, y)")
    else:
        # 分块 tiling:先块、再块内
        lines.append(f"for y_t in 0..{N} step {T}:     # 块循环(y)")
        lines.append(f"{indent}for x_t in 0..{N} step {T}: # 块循环(x)")
        lines.append(f"{indent}{indent}for y_in in 0..{T}:   # 块内")
        if vectorize_:
            lines.append(f"{indent*3}for x_vec in 0..{T} step 4: # 向量化")
            lines.append(f"{indent*4}A[(y_t+y_in)*N + (x_t+x_vec)] = f(...)  # 一次 4 个元素")
        else:
            lines.append(f"{indent*3}for x_in in 0..{T}:  # 块内")
            lines.append(f"{indent*4}A[(y_t+y_in)*N + (x_t+x_in)] = f(x, y)")
        if parallel_:
            lines.insert(0, f"parallel for y_t in 0..{N} step {T}:  # 外层并行(多核/多线程)")
    return "\\n".join(lines)

print("=== 朴素调度(不分块、不向量化)===")
print(schedule(reorder_=False, vectorize_=False, parallel_=False))
print("\\n=== 分块 + 向量化 + 并行(现代调度)===")
print(schedule(N=8, T=4, reorder_=True, vectorize_=True, parallel_=True))
'''),
"🏭 同一算法,两种调度生成两种截然不同的循环结构 —— 这就是『调度』对代码形态的影响。")

NB.code(D('''
# 可视化:循环分块(tiling)的示意 —— 把大方块切成小块,块内连续访问
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
N, T = 8, 4
# 左:朴素按行遍历
ax = axes[0]
ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal")
for y in range(N):
    for x in range(N):
        ax.add_patch(plt.Rectangle((x, N - 1 - y), 1, 1, facecolor="#a8c6e8" if x % 2 == y % 2 else "#cfe0f3"))
ax.set_title(f"朴素调度:按行扫描(N={N})")
# 右:分块后按块遍历
ax = axes[1]
ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal")
cmap = plt.get_cmap("tab20")
bi = 0
for yt in range(0, N, T):
    for xt in range(0, N, T):
        color = cmap(bi % 20); bi += 1
        for yy in range(T):
            for xx in range(T):
                ax.add_patch(plt.Rectangle((xt + xx, N - 1 - (yt + yy)), 1, 1,
                                           facecolor=color, edgecolor="white", lw=0.4))
ax.set_title(f"分块调度:tile={T}x{T},块内连续")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()
plt.tight_layout()
'''),
"🎨 左边按行线性扫描,右边先走『块』再走『块内』 —— 分块让同一块数据在缓存里被反复使用。")

NB.md("## 3️⃣ tiling 决策与局部性:不是越大越好 📐",
D('''
分块(tile)大小不是越大越好:块太小,循环开销占比高;块太大,一块放不进缓存/寄存器,
反而频繁换出。存在一个**甜点区间**。我们用模型估算『分块大小 vs 缓存未命中』的关系:
块越小越能留在缓存(命中率高),但块太小又浪费并行与向量宽度。
'''))

NB.code(D('''
# 模型:分块大小 vs 相对性能(合成曲线:太小循环开销高、太大缓存失效)
tile = np.arange(4, 65, 2)
# 两个相反因素:循环开销随 tile 增大而降低;缓存命中率随 tile 增大而下降
loop_overhead = 8.0 / tile                       # 小块时循环占比高
cache_miss = (tile / 32.0) ** 2 * 2.0            # 大块时缓存放不下
perf = 1.0 / (1.0 + loop_overhead + cache_miss)  # 归一化性能
best_tile = tile[np.argmax(perf)]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(tile, loop_overhead, "--", color="#e07a5f", label="循环开销(随 tile 减)")
ax.plot(tile, cache_miss, "--", color="#3d405b", label="缓存未命中(随 tile 增)")
ax.plot(tile, perf, "-", color="#81b29a", lw=2.5, label="综合性能")
ax.axvline(best_tile, ls=":", color="#c0392b")
ax.text(best_tile, perf.max(), f" 甜点 tile={best_tile}", color="#c0392b", fontsize=10)
ax.set_xlabel("分块大小 tile"); ax.set_ylabel("相对性能 / 代价")
ax.set_title("tiling 决策:太小循环开销高,太大缓存失效 —— 中间有甜点")
ax.legend(); plt.tight_layout()
'''),
"📐 性能曲线呈『先升后降』:tiling 决策就是在循环开销与缓存局部性之间找平衡。")

NB.md("## 4️⃣ 向量化:一次算多个元素 ⚡",
D('''
**向量化(vectorize)** 是另一种调度:让一条指令同时处理多个元素(如一次处理 4 个 float32)。
对访存密集的逐元素运算,向量化能显著摊薄指令取指与访存成本。现代 CPU 有 SSE/AVX,
GPU 天然按线程束(Warp)并行 —— 编译器自动向量化是后端的重要工作。
'''))

NB.code(D('''
# 向量化示意:一条指令处理 4 个元素 vs 逐个处理
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.axis("off")
lanes = 8
# 标量
for i in range(lanes):
    ax.add_patch(plt.Rectangle((i, 0.6), 0.8, 0.8, facecolor="#d9ead3", edgecolor="#38761d"))
ax.text(-0.4, 1.0, "标量(逐元素):", ha="right", fontsize=10)
for i in range(lanes):
    ax.text(i + 0.4, 1.0, f"LD{i}", ha="center", fontsize=7)
# 向量(4 宽)
for b in range(0, lanes, 4):
    ax.add_patch(plt.Rectangle((b, -0.9), 3.4, 0.8, facecolor="#fce5cd", edgecolor="#a64d17", lw=1.5))
    ax.text(b + 1.7, -0.5, f"vload×4", ha="center", fontsize=8, color="#a64d17")
ax.text(-0.4, -0.5, "向量(4 宽):", ha="right", fontsize=10)
ax.set_xlim(-3.5, lanes); ax.set_ylim(-1.3, 1.6)
ax.set_title("向量化:一条指令一次处理 4 个元素,摊薄指令与访存成本", fontsize=11)
plt.tight_layout()
'''),
"⚡ 示意图:标量一次一个,向量一次 4 个 —— 同样访存,指令与调度开销更少。")

NB.code(D('''
# 真实感受向量化:同一加法,让 torch 内部向量化 vs 手工逐元素,看速度差异
a = torch.randn(1_000_000); b = torch.randn(1_000_000)

def vectorized():
    return a + b                      # torch 内部自动向量化(1 条 kernel)

def scalar_loop():
    out = torch.empty_like(a)
    for i in range(len(a)):
        out[i] = a[i] + b[i]          # Python 逐元素(极慢,示意用)
    return out

import time
t0 = time.perf_counter(); vectorized(); tv = time.perf_counter() - t0
t0 = time.perf_counter(); _ = scalar_loop(); ts = time.perf_counter() - t0
print(f"torch 向量化加法 : {tv*1e3:.2f} ms")
print(f"Python 逐元素循环: {ts*1e3:.1f} ms  ({ts/tv:.0f}x 慢)")
print("→ 编译器/库帮你做的向量化,是性能的关键来源之一")
'''),
"📊 真实验证:同样 100 万元素相加,向量化比逐元素循环快几个数量级 —— 这就是向量化调度的价值。")

NB.md("## 5️⃣ CPU vs GPU 调度:机制不同 🖥️",
D('''
同样的循环,CPU 与 GPU 的"最优切法"不同:

| 维度 | CPU | GPU |
|---|---|---|
| 并行单元 | 少数多核 + SIMD 向量化 | 上千线程(线程束) |
| 主要瓶颈 | 缓存局部性(DRAM 往返) | 线程并行 + 共享内存带宽 |
| tiling 目的 | 把块装进缓存(L2/L1) | 把块装进共享内存/寄存器 |
| 调度手段 | split + vectorize + parallel | split + 线程/块映射 + 软件流水 |

GPU 的 tiling 要额外决定"每个线程算哪几行、块怎么映射到线程束",这就是第 65 课自动调优要搜的
`tile × num_warps × num_stages`。下面画一张 CPU/GPU 调度差异示意:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
# CPU:缓存分块(一个大矩阵被切块,块装进 L2)
ax = axes[0]
for yt in range(0, 4):
    for xt in range(0, 4):
        c = "#a8c6e8" if (yt * 4 + xt) % 2 == 0 else "#cfe0f3"
        ax.add_patch(plt.Rectangle((xt * 2, yt * 2), 1.8, 1.8, facecolor=c, edgecolor="#1f4e79"))
ax.add_patch(plt.Rectangle((7.2, 0.5), 1.6, 1.0, facecolor="#f4cccc", edgecolor="#c0392b"))
ax.text(8.0, 1.0, "L2", ha="center", va="center", fontsize=9)
ax.annotate("", xy=(7.1, 1.0), xytext=(1.5, 1.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))
ax.set_title("CPU:分块尽量装进 L2 缓存"); ax.axis("off")
# GPU:线程块映射到 SM(含共享内存)
ax = axes[1]
for sm in range(6):
    ax.add_patch(plt.Rectangle((sm * 1.3, 0.2), 1.0, 2.2, facecolor="#e2d5f1", edgecolor="#5b2c8f"))
    ax.text(sm * 1.3 + 0.5, 2.5, f"SM{sm}", ha="center", fontsize=8, color="#5b2c8f")
ax.text(7.9, 1.3, "线程块→SM\\n共享内存", ha="left", va="center", fontsize=9)
ax.set_title("GPU:tile 映射到 SM + 共享内存"); ax.axis("off")
plt.tight_layout()
'''),
"🎨 左 CPU 面向缓存、右 GPU 面向线程块/共享内存 —— 调度原语一样,优化目标不同。")

NB.md("## 6️⃣ 小结:调度 = 性能的杠杆 🏭",
D('''
这一课没有跑"快慢对比"的 GPU 实验,因为它讲的是**机制**:算法与调度分离、四大调度原语、
tiling 的甜点、向量化、CPU/GPU 差异。这些正是第 65 课"自动调优"在搜的东西 —— 自动调优
本质就是在**调度空间**里找最优组合。理解了调度,你就理解了为什么"同一个模型,换个编译器
能快好几倍"。
'''))

wrapup(NB,
    summary=[
        "算法(算什么)与调度(怎么算)分离是 Halide 的核心思想,同一算法换调度即换性能",
        "四大调度原语:split(分块)、reorder(重排)、vectorize(向量化)、parallel(并行)",
        "tiling 决策:太小循环开销高、太大缓存失效,存在甜点区间",
        "向量化让一条指令处理多个元素,对访存密集运算收益巨大",
        "CPU 调度面向缓存局部性,GPU 调度面向线程块/共享内存 —— 目标不同",
    ],
    practice=[
        "修改调度生成器的 T 与开关,生成 4 种不同调度的伪代码,对比循环层数",
        "把『tiling 甜点』曲线改成不同缓存大小参数,看甜点位置如何移动",
        "用 torch 对一个大矩阵分别做逐元素循环与向量化加法,画出规模 → 加速比曲线",
        "思考:为什么自动调优(65 课)要在调度空间里搜?把两者串起来写一段 100 字总结",
    ],
    links=[
        ("Halide 论文", "https://people.csail.mit.edu/nickolai/papers/ragan-kelley-halide.pdf"),
        ("TVM TensorIR 调度", "https://tvm.apache.org/docs/"),
        ("LLVM 循环优化术语", "https://llvm.org/docs/LoopTerminology.html"),
    ])

out = str(Path(CH10) / "68_codegen_schedule.ipynb")
NB.save(out)
finalize(out)
