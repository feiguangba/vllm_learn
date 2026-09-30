# -*- coding: utf-8 -*-
"""生成 77_ms_dynamic_static.ipynb(本课不配 app,纯概念对比)"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, finalize
from pathlib import Path

NB = new_nb("第 77 课 · 动态图 vs 静态图:灵活、性能与调试的三方博弈",
            subtitle="从 TensorFlow 1 的手写静态图,到 torch.compile 的自动捕获,再到 MindSpore 的双模式 —— 一条业界进化史",
            emoji="⚖️")

chapter_cover(NB,
    objectives=[
        "从灵活性 / 性能 / 调试体验三个维度对比动态图与静态图",
        "理解『为什么训练与部署对图模式的偏好不同』",
        "梳理业界演进:TF1 手写静态图 → PyTorch 动态图 → torch.compile → MindSpore 双模式",
        "用 matplotlib 画累计耗时曲线与业界时间线",
        "体会静态图的『可读调试』:用 torch.fx 打印图节点",
    ],
    toc=[
        ("直觉:两把尺子", "灵活好改 vs 跑得快,没有免费的午餐"),
        ("三维对比表", "灵活性 / 性能 / 调试逐项 PK"),
        ("累计耗时曲线", "编译成本摊薄 vs 逐轮解释的数学账"),
        ("静态图怎么调试", "看图说话:打印图的每个节点"),
        ("业界演进时间线", "TF1 → PyTorch → torch.compile → MindSpore 双模式"),
        ("对 vLLM 的启发", "CUDA Graph 捕获与静态推理策略"),
    ],
    links=[
        ("MindSpore 动态图/静态图说明", "https://www.mindspore.cn/docs/zh-CN/master/design/mindspore/mindspore.html"),
        ("PyTorch torch.compile 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:两把尺子 📏",
D('''
动态图与静态图之争,本质是**两把尺子的较量**:

- **灵活尺**:能不能今天改这个层、明天换个激活、随时 print 中间结果?动态图说"能",
  静态图说"先重新编译";
- **性能尺**:同一模型跑一千轮,谁更省时间?静态图说"我",动态图说"那是你编译完了才快"。

用生活类比:动态图像**即兴爵士**,乐手随手发挥、随时solo;静态图像**交响乐总谱**,
谱子定好了,每场演出都精准高效。**爵士不适合交响乐大厅,总谱不适合 jam session** ——
没有谁绝对好,只有谁适合哪个场景。

训练大模型时,你既要灵活调结构,又要在同样的结构上跑成百上千轮 → 于是业界走出了
第三条路:**"先动态调试,再静态编译"**(torch.compile / MindSpore 双模式)。
'''))

NB.md("## 2️⃣ 三维对比:灵活、性能、调试 ⚖️",
D('''
把三张牌摊开对比:

| 维度 | 动态图 | 静态图 |
|------|--------|--------|
| **灵活性** | 结构随意改、条件分支自然 | 结构基本定死,改结构要重编译 |
| **性能** | 每轮逐算子解释 + 启动 | 编译一次,每轮整体执行 |
| **调试体验** | 像普通 Python,print / 断点 | 需要『看图』或打印图节点 |
| **内存** | 中间张量即时分配 | 编译期统一规划、buffer 复用 |
| **动态 shape** | 天然支持 | 需要按 shape 特化或加 padding |
| **部署/推理** | 逐算子启动,慢 | 整图下沉,极快 |

**结论很朴素**:研究阶段要的是灵活性 → 动态图;线上阶段要的是性能与确定性 → 静态图。
MindSpore 干脆两个都给(PyNative / Graph),torch 则用 `torch.compile` 把动态图
"二次编译"成静态图 —— 殊途同归。
'''))

NB.md("## 3️⃣ 累计耗时曲线:编译成本的数学账 📉",
D('''
静态图不是一开始就赢:它先付一笔**编译成本**。设动态图每轮耗时为 $T_d$,静态图编译
成本为 $C$、每轮耗时为 $T_s$,则跑 $N$ 轮后:

$$\\text{动态图累计}=N\\cdot T_d,\\qquad \\text{静态图累计}=C+N\\cdot T_s$$

交叉点在 $N^* = \\dfrac{C}{T_d-T_s}$。**只要跑超过 $N^*$ 轮,静态图就反超**。
用代码把这条数学规律画出来:
'''))

NB.code(D('''
def crossover(compile_ms, t_dynamic_ms, t_static_ms):
    return compile_ms / max(t_dynamic_ms - t_static_ms, 1e-9)

compile_ms, td, ts = 50.0, 0.24, 0.015      # 编译 50ms;动态每轮 0.24ms;静态每轮 0.015ms
n_star = crossover(compile_ms, td, ts)
print(f"每轮动态图 {td*1000:.0f}μs, 静态图 {ts*1000:.0f}μs, 编译成本 {compile_ms}ms")
print(f"交叉点 N* = {n_star:.0f} 轮:跑超过 {n_star:.0f} 轮,静态图反超")

rng = np.arange(1, 601)
eager = td * rng
static = compile_ms + ts * rng
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(rng, eager, label="动态图(逐轮解释)", color="#E45756", lw=2.5)
ax.plot(rng, static, label="静态图(编译一次)", color="#4C78A8", lw=2.5)
ax.axvline(n_star, ls="--", color="#555")
ax.text(n_star + 3, eager[-1] * 0.55, f"N* ≈ {n_star:.0f} 轮", fontsize=10, color="#555")
ax.set_xlabel("轮数 N"); ax.set_ylabel("累计耗时(ms)")
ax.set_title("编译成本的摊薄:静态图靠『长期主义』取胜")
ax.legend(); plt.tight_layout()
'''),
"📊 数学账:静态图把固定成本 C 摊到 N 轮上,轮数越大单轮摊薄越小 —— 训练几百上千轮时,静态图稳赢。")

NB.md("## 4️⃣ 静态图怎么调试:看图说话 🔍",
D('''
静态图不能随便 print,但静态图有个动态图没有的优势:**整张图可见** —— 你可以
"俯瞰"整个计算过程,而不是钻进某个算子。用 torch.fx 打印图的每个节点,
体会这种『图级调试』:
'''))

NB.code(D('''
import torch.fx as fx

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(6, 4)
    def forward(self, x):
        h = torch.relu(self.fc(x))
        h = torch.sum(h, dim=1)
        return torch.log1p(h)

gm = fx.symbolic_trace(Net())
print("静态图中的每个节点(可读的调试视图):")
for node in gm.graph.nodes:
    print(f"  {node.op:<12} {node.name:<12} -> {node.target}")
'''),
"🔍 图级调试:不是问『这个算子输出多少』,而是问『这张图长什么样』 —— 优化师与编译器工程师都爱看这种视图。")

NB.md("## 5️⃣ 业界演进:一条从『手写』到『自动』的路 🗺️",
D('''
动态/静态之争,二十年走了四步:

1. **TensorFlow 1(2015)**:静态图**手写** —— 先 `tf.placeholder` 建图,再 `tf.Session` 跑,
   灵活度极差,被戏称"写两层都有罪";
2. **PyTorch(2017)**:动态图一统天下 —— 灵活到爆,性能靠 eager 兜底;
3. **torch.compile(2023)**:一条新路 —— **先动态后编译**:Dynamo 自动捕获你的 Python
   代码成图,Inductor 再编译成高效 kernel,灵活与性能"我都要";
4. **MindSpore 双模式(并行演进)**:同一个框架里 PyNative 与 Graph 共存,还把静态图
   与昇腾整图下沉绑定 —— 相当于把 TF1 的图和 PyTorch 的爽捏在一起。

画成时间线:
'''))

NB.code(D('''
events = [
    (2015, "TensorFlow 1:手写静态图", "#E45756"),
    (2017, "PyTorch:动态图复兴", "#54A24B"),
    (2020, "MindSpore:双模式(PyNative + Graph)", "#4C78A8"),
    (2023, "torch.compile:动态捕获 + 静态编译", "#B279A2"),
    (2024, "vLLM 等推理引擎:静态化(CUDA Graph)", "#E2C62E"),
]
fig, ax = plt.subplots(figsize=(9.5, 3.6))
ax.axhline(0, color="#333", lw=1.2)
for i, (year, label, color) in enumerate(events):
    y = 1 if i % 2 == 0 else -1
    ax.plot([year, year], [0, y * 0.6], color=color, lw=1.8)
    ax.plot(year, y * 0.6, "o", color=color, ms=9)
    ax.text(year, y * 0.82, label, ha="center", va="bottom" if y > 0 else "top",
            fontsize=9, color=color)
ax.set_xlim(2014, 2026); ax.set_ylim(-2.1, 2.1); ax.set_yticks([])
ax.set_title("动态 vs 静态的业界演进:从手写静态图到『先动态后编译』")
plt.tight_layout()
'''),
"📈 时间线:主线是『手写静态图(难用)→ 动态图(好用)→ 自动捕获 + 静态编译(又要用又要快)』 —— MindSpore 双模式正是这条主线的集大成者。")

NB.md("## 6️⃣ 对 vLLM 的启发:推理引擎也在『静态化』 🚀",
D('''
回到《minivllm》的主线:LLM 推理为什么也要静态图思想?

- **CUDA Graph**:vLLM 支持把解码的固定计算图**捕获(Capture)成 CUDA Graph**,反复回放,
  省去每步逐 kernel 启动 —— 这就是"静态图"在 GPU 上的翻版;
- **固定形状**:LLM decode 阶段每个 token 的计算形状几乎不变,天然适合静态化;
- **昇腾场景**:vLLM 跑在昇腾上时(vllm-ascend),CANN 的 GE 整图下沉正是同一个道理。

所以动态/静态之争的答案,在推理引擎里已经写死:**能静态就静态,不能静态就用
动态图兜底动态形状**(如批大小变化时重新捕获)。这也是第 79 课『图算融合』的先声。
'''))

wrapup(NB,
    summary=[
        "动态图重灵活、静态图重性能、调试体验各有侧重 —— 三把尺子各有长短",
        "数学账:静态图累计耗时 = C + N·Ts,跑过交叉点 N* = C/(Td−Ts) 即反超",
        "静态图可『图级调试』:俯瞰整张图,而不是钻单个算子",
        "业界演进:TF1 手写静态图 → PyTorch 动态图 → MindSpore 双模式 → torch.compile",
        "推理引擎的答案:能静态就静态(CUDA Graph / 昇腾整图下沉),动态形状用动态图兜底",
    ],
    practice=[
        "把 crossover 参数改成 动态图每轮 0.5ms、静态图 0.01ms、编译 200ms,重算交叉点并解释结论",
        "用 torch.fx 打印一个带 if-else 的函数,观察分支如何体现在图中(体会静态图对动态控制的处理)",
        "给时间线补两个事件:2021 年 JAX jit 普及、2022 年 Dynamo 原型发布",
        "查 vLLM 文档中 CUDA Graph 的开启条件(padding、max_num_seqs),写 3 条要点",
    ],
    links=[
        ("MindSpore 动态图/静态图说明", "https://www.mindspore.cn/docs/zh-CN/master/design/mindspore/mindspore.html"),
        ("PyTorch torch.compile 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("vLLM CUDA Graph 文档", "https://docs.vllm.ai"),
    ])

out = str(Path(CH11) / "77_ms_dynamic_static.ipynb")
NB.save(out)
finalize(out)
