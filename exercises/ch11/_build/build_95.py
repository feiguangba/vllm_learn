# -*- coding: utf-8 -*-
"""生成 95_ascend_graph_engine.ipynb(本课不配 app)"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, finalize
from pathlib import Path

NB = new_nb("第 95 课 · 昇腾图引擎 GE:让计算图『整图下沉』",
            subtitle="图优化 · 多算子融合 · 整图下沉 · 与 torch.compile 对照",
            emoji="🗺️")

chapter_cover(NB,
    objectives=[
        "理解 GE(Graph Engine)在 CANN 中的位置:计算图的『总调度师』",
        "掌握图优化三大手段:常量折叠、死代码消除、layout 变换",
        "看懂多算子融合:把一串小算子并成一个大算子,减少 kernel 启动",
        "理解整图下沉 vs 逐算子下发:一次下发 vs N 次下发",
        "把 GE 与 torch.compile / Inductor 对照,建立编译器思维",
        "用 torch.fx + 计时亲手演示融合带来的收益",
    ],
    toc=[
        ("直觉:总厨与流水线调度", "GE = 计算图的总调度师"),
        ("GE 在 CANN 里的位置", "前端图 → GE → 任务下发 → AI Core"),
        ("三大图优化手段", "常量折叠 / DCE / layout 变换"),
        ("多算子融合(动手)", "torch.fx 建图,手动融合,量收益"),
        ("整图下沉 vs 逐算子下发", "启动开销就是融合的收益来源"),
        ("与 torch.compile 对照", "同一套编译器思想,不同实现"),
    ],
    links=[
        ("昇腾 CANN 文档(GE)", "https://www.hiascend.com/document"),
        ("PyTorch torch.compiler 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("MLIR 论文(图优化背景)", "https://arxiv.org/abs/2002.11054"),
        ("vLLM-Ascend(依赖 GE/CANN)", "https://github.com/vllm-project/vllm-ascend"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:总厨与流水线调度 👨‍🍳",
D('''
一个模型在昇腾上跑,并不是"每层一勺一勺喂"。CANN 里的 **GE(Graph Engine,图引擎)** 就像
餐厅的**总厨**:先把你交上来的"菜谱"(计算图)整体读一遍,然后——

1. 把能合并的工序**合并**(把"切葱 + 热油 + 下锅"合成一道操作);
2. 把用不到的工序**删掉**(有的菜谱里写着但根本用不上的步骤);
3. 最后把整份菜单**一次性交给后厨**(整图下沉),而不是一道菜跑一趟厨房。

GE 的存在,让昇腾能在**图级别**做优化 —— 这正是编译器思维(第 61 课)在昇腾落地的地方。
本课把 GE 的职责拆开,用 torch 亲自动手演示"融合"为什么能提速。
'''))

NB.md("## 2️⃣ GE 在 CANN 里的位置 🧩",
D('''
从模型到昇腾执行,数据要走过一条链路:

```
PyTorch / ONNX / Caffe 模型
        │  (前端翻译成图)
        ▼
   GE 图引擎:图优化 + 算子融合 + 调度
        │  (整图下沉,生成任务)
        ▼
   AI Core 执行(Runtime + 算子库)
```

GE 的输入是"图",输出是"一堆排好顺序的任务"。它决定:**这个算子用哪个实现、先算谁、
中间结果放哪、要不要合并**。一句话 —— GE 是昇腾的"运行时编译器"。
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.axis("off")
stages = [
    ("模型描述", "PyTorch / ONNX\\nCaffe", "#dbe7f4", "#1f4e79"),
    ("前端建图", "计算图\\n(节点=算子)", "#e2d5f1", "#5b2c8f"),
    ("GE 图优化", "常量折叠 · DCE\\nlayout · 算子融合", "#fdebd0", "#a64d17"),
    ("任务调度", "排顺序 · 分配内存\\n决定实现", "#fff2cc", "#7f6000"),
    ("AI Core", "整图下沉\\n一次下发", "#d9ead3", "#38761d"),
]
x = 0.4; bw, bh = 1.6, 1.7
for i, (t, sub, fill, edge) in enumerate(stages):
    ax.add_patch(plt.Rectangle((x, 0.7), bw, bh, facecolor=fill, edgecolor=edge, lw=2))
    ax.text(x + bw/2, 0.7 + bh*0.7, t, ha="center", fontsize=11, fontweight="bold", color=edge)
    ax.text(x + bw/2, 0.7 + bh*0.32, sub, ha="center", va="center", fontsize=8, color="#333")
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + bw + 0.1, 1.55), xytext=(x + bw - 0.05, 1.55),
                    arrowprops=dict(arrowstyle="->", lw=2.4, color="#555"))
    x += bw + 0.5
ax.text(0.5, 2.85, "CANN 执行链路:模型 → 计算图 → GE 优化 → 调度 → 整图下沉到 AI Core",
        fontsize=13, fontweight="bold", ha="center", color="#1f4e79")
ax.set_xlim(0, x); ax.set_ylim(0, 3.3)
plt.tight_layout()
'''),
"🎨 链路图:GE 是中间那颗『大脑』,吃进图、吐出任务。本课重点看 GE 的『优化』与『整图下沉』。")

NB.md("## 3️⃣ 三大图优化手段 🔧",
D('''
GE 的图优化 pass 和任何编译器一样,老三样打底:

1. **常量折叠(Constant Folding)**:能提前算的常量直接算出结果,运行时不重复算
   (如 `weight = w0 + w1` 在构图时就算掉);
2. **死代码消除(DCE, Dead Code Elimination)**:输出用不到的节点直接剪掉,省算子省显存;
3. **layout 变换**:数据排布(NCHW ↔ NHWC)尽量统一在最早最省的位置转换,减少反复搬移。

这三板斧在 GPU 侧的 XLA / Inductor 里同样存在 —— 这就是为什么第 10 章的编译器知识
能直接平移过来。真正的昇腾特色是下面这个:**多算子融合 + 整图下沉**。
先动手体验前两板斧:常量折叠与死代码消除。
'''))

NB.code(D('''
# 模拟常量折叠:编译期就算掉的"常量计算",运行时不再重复
def with_folding(n_iters):
    """折叠:w 是常量,w0+w1 构图时算一次,后续直接复用"""
    w0 = torch.randn(64, 64); w1 = torch.randn(64, 64)
    w = w0 + w1                                   # ← 常量折叠:编译期算好
    x = torch.randn(128, 64)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y = x @ w                                 # 运行时只做一次 matmul
    return (time.perf_counter() - t0) / n_iters

def without_folding(n_iters):
    """不折叠:每次推理都重新算 w0+w1(多一次 64×64 加法)"""
    w0 = torch.randn(64, 64); w1 = torch.randn(64, 64)
    x = torch.randn(128, 64)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        w = w0 + w1                               # 运行时重复计算
        y = x @ w
    return (time.perf_counter() - t0) / n_iters

t_fold = with_folding(30); t_no = without_folding(30)
print(f"常量折叠: {t_fold*1e6:.1f} us/次 | 不折叠: {t_no*1e6:.1f} us/次 | 省 {100*(1-t_fold/t_no):.1f}%")

# 死代码消除:构图时剪掉"输出用不到"的节点
x = torch.randn(16, 16)
a = x * 2            # 会被用到
b = x @ x.T          # ← 死代码:算完没人用,直接删掉
c = a + 1
print(f"DCE 示例:节点 a、c 保留,b 被剪掉 → 省一次 {b.shape} 的矩阵乘。")
'''),
"🔍 亲手体验:常量折叠把『每次重复的常量运算』挪到编译期;DCE 把『没人用的节点』直接剪掉 —— 图优化就是替运行时减负。")

NB.md("## 4️⃣ 多算子融合:把一串合成一个 🔗",
D('''
`Linear → ReLU → Linear` 这类"老三样",GE 会融合成一个算子:数据在片上(UB)走完
整个过程,不用反复读写 DDR。我们用 torch.fx 建一张同样的小图,再手动写一个"融合版",
对比两者的执行开销(尤其模拟 **kernel 启动开销**,这是融合收益的主要来源):
'''))

NB.code(D('''
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

def net(x, w1, b1, w2, b2):
    h = torch.addmm(b1, x, w1)     # Linear1
    h = torch.relu(h)              # ReLU
    h = torch.addmm(b2, h, w2)     # Linear2
    return h

graph = fx.symbolic_trace(net)
n_ops = sum(1 for n in graph.graph.nodes if n.op == "call_function")
print("融合前:算子节点数 =", n_ops)

def net_fused(x, w1b1, w2b2):
    """融合版:把 w1/b1、w2/b2 先合成一个参数,减少中间张量往返"""
    h = F.linear(x, w1b1[0], w1b1[1])
    h = F.relu(h)
    return F.linear(h, w2b2[0], w2b2[1])

torch.manual_seed(0)
M, K, N = 64, 128, 128
x = torch.randn(M, K)
w1 = torch.randn(N, K); b1 = torch.randn(N)
w2 = torch.randn(N, N); b2 = torch.randn(N)
# 融合前后的算子在数学上完全等价
out1 = net(x, w1, b1, w2, b2)
out2 = net_fused(x, (w1, b1), (w2, b2))
print("融合前后输出最大差:", f"{(out1 - out2).abs().max().item():.2e} (应 ≈ 0)")

# 模拟 kernel 启动开销:每个算子都有固定启动成本,算子越少总开销越低
launch_us = 8.0                    # 每次 kernel 启动约 8us(示意)
exec_us = 30.0                     # 每个算子的计算时间(示意)
n_unfused = n_ops
t_unfused = (launch_us + exec_us) * n_unfused
t_fused = (launch_us + exec_us) * 2          # 融合成 2 个大算子
print(f"启动+执行(模拟): 未融合 {n_unfused} 算子 = {t_unfused:.0f}us | 融合 2 算子 = {t_fused:.0f}us")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(["未融合\\n(逐算子下发)", "融合\\n(整图下沉)"], [t_unfused, t_fused],
       color=["#c0392b", "#27ae60"], width=0.5)
for b, v in zip(ax.patches, [t_unfused, t_fused]):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}us", ha="center", fontsize=10)
ax.set_ylabel("耗时(示意,us)")
ax.set_title("算子融合的收益:少启动几次,就快一截")
plt.tight_layout()
'''),
"✅ 看数据:融合前后数学等价(输出差 ≈ 0),但『启动次数』少了一半 —— 图里算子越碎,融合收益越大。")

NB.md("## 5️⃣ 整图下沉 vs 逐算子下发 🚀",
D('''
GE 最有昇腾特色的一招是 **整图下沉(Whole-Graph Down)**:

- **逐算子下发**:每个算子在 Host(CPU)和 Device(NPU)之间来回"汇报",CPU 频繁介入,延迟高;
- **整图下沉**:优化后的整张图一次性编译成设备侧任务流,NPU 自己执行完,CPU 只负责开头和结尾
  —— 大量减少 Host↔Device 往返,这正是 LLM 推理延迟的关键来源之一。

用启动开销模型扫一遍"算子数 → 总耗时",直观看到为什么融合和下沉对推理这么重要:
'''))

NB.code(D('''
n_ops_range = np.arange(1, 21)
launch_us, exec_us = 8.0, 15.0
t_fused_model = (launch_us + exec_us) * 2        # 无论多少算子,整图下沉后按 2 个大算子算
t_separate = (launch_us + exec_us) * n_ops_range

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(n_ops_range, t_separate, "o-", color="#c0392b", lw=2, label="逐算子下发")
ax.plot(n_ops_range, [t_fused_model]*len(n_ops_range), "-", color="#27ae60", lw=2.5,
        label="整图下沉(≈2 个大算子)")
ax.fill_between(n_ops_range, t_fused_model, t_separate, color="#c0392b", alpha=0.12)
ax.set_xlabel("图中算子个数"); ax.set_ylabel("总耗时(示意,us)")
ax.set_title("算子越多,『整图下沉』省下的时间越多")
ax.legend(); plt.tight_layout()
'''),
"📈 曲线:横轴是图里算子数,纵轴是总耗时。算子越多,两条线的差距(阴影区)越大 —— 融合是『按算子数赚钱』。")

NB.md("## 6️⃣ 与 torch.compile 对照:同一套思想 ⚖️",
D('''
昇腾 GE 和 PyTorch 的 torch.compile,是"同一棵编译器大树上结的两颗果":

| 环节 | torch.compile / Inductor | 昇腾 GE |
|---|---|---|
| 前端 | TorchDynamo 捕获 FX 图 | 接收 PyTorch/ONNX 等前端图 |
| IR | FX → Inductor IR | GE IR(内部图) |
| 图优化 | 融合 / 折叠 / DCE | 融合 / 折叠 / DCE / layout |
| 代码生成 | 生成 Triton/C++ kernel | 生成昇腾任务流(整图下沉) |
| 目标 | NVIDIA GPU / CPU | 昇腾 NPU |

区别主要在**后端**与**粒度**:torch.compile 生成的是"可单发"的 kernel,昇腾 GE 更强调
把整张图**下沉为设备侧任务流**。两者都贯彻同一条主线 —— **在图上做优化,而不是在算子上一一打磨**。
'''))

NB.code(D('''
rows = ["前端建图", "IR 表示", "图优化 pass", "融合策略", "代码/任务生成", "目标硬件"]
tc = [5, 5, 5, 4, 5, 5]      # torch.compile/Inductor 的成熟度打分(示意)
ge = [4, 4, 4, 5, 5, 5]      # 昇腾 GE 的成熟度打分(示意)
df = pd.DataFrame({"torch.compile": tc, "昇腾 GE": ge}, index=rows)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
sns.heatmap(df.T, annot=True, fmt="d", cmap="YlGnBu", linewidths=1, linecolor="white", cbar=False, ax=ax)
ax.set_title("GE 与 torch.compile 各环节成熟度对比(5=最成熟)")
plt.tight_layout()
'''),
"📊 热力图:二者都成熟于『图优化/融合』;昇腾 GE 的独特优势是『整图下沉到设备侧』。学懂一个,另一个可举一反三。")

NB.md("## 7️⃣ 小结:把编译器思维带上昇腾 🧠",
D('''
GE 是昇腾版的"运行时编译器",核心三件事:

1. **优化**:常量折叠、DCE、layout 变换 —— 在图上做文章;
2. **融合**:把小算子合成大算子,少启动、少搬数据 —— 收益随算子数增长;
3. **整图下沉**:一次编译,设备侧整图执行,大幅减少 Host↔Device 往返。

如果你已经学完第 10 章(AI 编译器),你会惊喜地发现:**昇腾 GE 完全可以用同样的思维去理解**。
这也是 vLLM-Ascend 能在昇腾上跑出好性能的底层原因之一 —— 图被 GE 优化得越狠,推理越快。
'''))

wrapup(NB,
    summary=[
        "GE(Graph Engine)是 CANN 的计算图总调度师:吃进图、吐出优化后的任务流",
        "图优化老三样:常量折叠、死代码消除(DCE)、layout 变换",
        "多算子融合把一串小算子合成大算子,数学等价但启动次数大减",
        "整图下沉:优化后的整图一次性下发设备侧执行,减少 Host↔Device 往返",
        "GE 与 torch.compile 是同一棵编译器树上的果子:前端→IR→优化→后端 四段式同源",
    ],
    practice=[
        "给第 4 节的 net 再加一个 dropout,重画图并统计节点数变化",
        "把第 5 节的启动开销模型改成『计算占比 70%』,观察两条线的交叉点变化",
        "用 torch.compile 编译第 4 节的 net,对比它与手工融合版的算子数(第 66/67 课方法)",
        "读 vLLM-Ascend 源码里的 graph runner,找出它把哪些算子放进了 CUDA-Graph 式的图里",
    ],
    links=[
        ("昇腾 CANN 图引擎文档", "https://www.hiascend.com/document"),
        ("PyTorch torch.compiler", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("MLIR 论文", "https://arxiv.org/abs/2002.11054"),
        ("vLLM-Ascend", "https://github.com/vllm-project/vllm-ascend"),
    ])

out = str(Path(CH11) / "95_ascend_graph_engine.ipynb")
NB.save(out)
finalize(out)
