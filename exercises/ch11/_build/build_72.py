# -*- coding: utf-8 -*-
"""生成 72_davinci_arch.ipynb(本课不配 app,纯概念讲解)"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, finalize
from pathlib import Path

NB = new_nb("第 72 课 · 达芬奇架构:AI Core 的三单元与分级缓存",
            subtitle="Cube 算矩阵、Vector 算向量、Scalar 算标量 —— 一颗专为『张量』而生的芯片里长什么样",
            emoji="🏗️")

chapter_cover(NB,
    objectives=[
        "理解达芬奇架构『统一指令集 + 三单元分工』的设计思想",
        "认识 AI Core 的三大计算单元:Cube(矩阵)、Vector(向量)、Scalar(标量)",
        "看懂 L0/L1/L2/HBM 分级存储层次与『分块搬入片内』的数据流",
        "用 torch 分块 GEMM 模拟 Cube 的 tile 计算范式并验证正确性",
        "对比 NPU 与 GPU 在核心组织、存储、编程模型上的差异",
    ],
    toc=[
        ("直觉:一间分工明确的大厨房", "三单元三把刀:Cube / Vector / Scalar"),
        ("AI Core 三单元", "矩阵、向量、标量各算各的,同一套指令下发"),
        ("分级存储与矩阵数据流", "HBM → L2 → L1 → L0,分块搬入算完再搬回"),
        ("真机模拟:分块 GEMM", "用 torch 把大矩阵乘拆成 16×16 tile,验证一致"),
        ("NPU vs GPU", "核心组织、缓存、编程模型的三层对比"),
        ("为什么 LLM 推理爱昇腾", "大矩阵 GEMM 正对 Cube,FP16 高算力 + 大带宽"),
    ],
    links=[
        ("昇腾 AI 处理器架构白皮书(华为文档)", "https://www.hiascend.com/document"),
        ("昇腾 CANN 文档中心", "https://www.hiascend.com/document"),
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:一间分工明确的大厨房 🍳",
D('''
把昇腾的 **AI Core**(AI 计算核心)想象成一间餐厅后厨,里面站着三位各有绝活的大厨:

- **Cube 单元**(矩阵大厨):专做"大锅菜"—— **矩阵乘法**。一次能算 16×16×16 的乘加,
  就像一次蒸一整屉包子;
- **Vector 单元**(向量大厨):擅长"一锅端"—— **逐元素运算**(加法、激活、归一化),
  一次处理一整条向量,就像一勺捞出整锅汤圆;
- **Scalar 单元**(小炒师傅):负责"单点精确"—— **标量运算与控制流**(循环、跳转、地址计算),
  像颠勺定火候,量小但每道菜都离不开它。

三位大厨共用**同一本菜谱(统一指令集)**,由 Scalar 师傅念菜名指挥。这就是**达芬奇架构**
最核心的思想:**把一个 AI 计算需要的能力(矩阵 / 向量 / 标量)集成进一个核心,各司其职、
协同作战**,而不是像 GPU 那样用一堆通用单元硬扛所有活。
'''))

NB.md("## 2️⃣ AI Core 三单元:一芯三用 👾",
D('''
昇腾芯片由大量 **AI Core** 组成(如 Ascend 910 有几十个),每个 AI Core 内部就是上面
说的三单元结构。为什么这么分?因为神经网络的计算里,**90% 的算力消耗在矩阵乘**上,
剩下的是各种逐元素操作与少量标量逻辑 —— 把三种计算类型分别用**专用硬件**实现,
每一类都能做到极致能效:

| 单元 | 负责什么 | 计算强度 | 典型指令 |
|------|----------|----------|----------|
| **Cube** | 矩阵乘/卷积累加 | 极高(一次 16×16×16) | 矩阵乘加 |
| **Vector** | 逐元素/规约运算 | 高(一次一整条向量) | Add、Exp、激活 |
| **Scalar** | 标量、控制流、寻址 | 低(单点) | 跳转、比较、算术 |

把三单元画进一颗 AI Core 里,长这样:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.axis("off")

def box(x, y, w, h, fc, ec, title, sub, tw=12, fs=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.8))
    ax.text(x + w/2, y + h*0.7, title, ha="center", va="center", fontsize=tw,
            fontweight="bold", color=ec)
    ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=fs, color="#333")

def flow(x, y0, y1, color="#555", ls="-"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", lw=2, color=color, linestyle=ls))

box(2.9, 5.5, 4.2, 1.1, "#fdebd0", "#7f5f01", "HBM 全局内存", "A / B / C 矩阵整块")
box(2.9, 4.1, 4.2, 1.0, "#fcf3cf", "#7d6608", "L2 Cache", "多 AI Core 共享")
box(2.9, 2.7, 4.2, 1.0, "#d6eaf8", "#1f4e79", "L1 Buffer", "本 AI Core 独享")
box(0.7, 1.2, 2.0, 1.0, "#e8daef", "#5b2c8f", "L0 A", "A 分块")
box(5.3, 1.2, 2.0, 1.0, "#e8daef", "#5b2c8f", "L0 B", "B 分块")
box(3.0, 1.2, 2.0, 1.0, "#f9ebea", "#a93226", "L0 C", "累加输出")
box(1.7, 0.0, 5.0, 0.9, "#aed6f1", "#1f4e79", "Cube 单元:16×16×16 矩阵乘",
    "A × B → C 累加(Systolic 流水)", tw=11)
box(6.0, 0.0, 3.0, 0.9, "#d5f5e3", "#1e8449", "Vector 单元", "逐元素 / 规约", tw=11)
box(0.2, 0.0, 1.3, 0.9, "#fce4d6", "#c55a11", "Scalar", "控制流 / 标量", tw=11)

flow(5.0, 5.5, 4.1)     # HBM -> L2
flow(5.0, 4.1, 2.7)     # L2 -> L1
flow(1.7, 2.7, 1.2)     # L1 -> L0A
flow(6.3, 2.7, 1.2)     # L1 -> L0B
flow(4.2, 1.2, 0.9)     # L0C -> Cube? (L0C 在 Cube 下方,流回 Cube 累加)
ax.annotate("", xy=(4.0, 0.9), xytext=(4.2, 1.2),
            arrowprops=dict(arrowstyle="->", lw=2, color="#a93226"))
flow(1.7, 0.9, 0.0)     # L0A -> Cube
ax.annotate("", xy=(4.0, 0.0), xytext=(1.7, 0.0), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
flow(6.3, 0.9, 0.0)     # L0B -> Cube
ax.annotate("", xy=(4.0, 0.0), xytext=(6.3, 0.0), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.annotate("", xy=(4.0, 2.7), xytext=(4.0, 1.2),
            arrowprops=dict(arrowstyle="->", lw=1.6, ls="--", color="#a93226"))
ax.annotate("", xy=(5.0, 4.1), xytext=(5.0, 2.7),
            arrowprops=dict(arrowstyle="->", lw=1.6, ls="--", color="#a93226"))
ax.annotate("", xy=(5.0, 5.5), xytext=(5.0, 4.1),
            arrowprops=dict(arrowstyle="->", lw=1.6, ls="--", color="#a93226"))
ax.text(0.8, 4.6, "数据流入", fontsize=9, rotation=90, color="#555")
ax.text(8.2, 4.6, "结果写回(虚线)", fontsize=9, rotation=90, color="#a93226")
ax.set_xlim(0, 9.3); ax.set_ylim(-0.35, 6.9)
ax.set_title("一颗 AI Core 的内部结构:三单元 + 分级缓存 + 矩阵数据流", fontsize=13)
plt.tight_layout()
'''),
"🎨 架构图:同一颗 AI Core 里,Cube 吃矩阵、Vector 吃向量、Scalar 管控制;数据从 HBM 一路搬到 L0,算完再原路写回 —— 这就是『分块搬运』的全貌。")

NB.md("## 3️⃣ 分级存储:从 HBM 到 L0 的『物流链』 🗄️",
D('''
芯片上的存储像一栋楼的电梯系统,越靠近计算单元越快、越小:

| 层级 | 位置 | 大小(示意) | 作用 |
|------|------|-----------|------|
| **HBM** | 芯片外 | 几十 ~ 百 GB | 全局内存,放整个权重与激活 |
| **L2 Cache** | 芯片内共享 | 数十 MB | 多 AI Core 共享的中转站 |
| **L1 Buffer** | 每 AI Core 独享 | 几百 KB | 本核工作台 |
| **L0 A / L0 B / L0 C** | Cube 门口 | 各几十 KB | 输入分块与累加结果 |

**为什么这么麻烦?** 因为矩阵乘要"反复用同一块数据":一次 GEMM 里,一个 16×16 的
A 分块会被 B 的多个分块反复相乘。把这块数据**搬进离 Cube 最近的 L0**,一次搬入、多次使用,
能省下天文数字的 HBM 访问 —— 这正是 Cube 用 **Systolic(脉动)阵列** 实现矩阵乘的原因:
数据像水流一样在阵列里流动复用。我们用 torch 真实验证"分块 GEMM"的正确性:
'''))

NB.code(D('''
def matmul_naive(a, b):
    return a @ b

def matmul_tiled(a, b, tile=16):
    """模拟达芬奇 Cube 的分块范式:把矩阵按 tile×tile 拆开,
    每块搬入『片内』算完累加(这里用 torch 直接做分块乘)。"""
    n = a.shape[0]
    c = torch.zeros(n, n, dtype=a.dtype)
    for i in range(0, n, tile):
        for j in range(0, n, tile):
            for k in range(0, n, tile):
                c[i:i+tile, j:j+tile] += a[i:i+tile, k:k+tile] @ b[k:k+tile, j:j+tile]
    return c

n, tile = 64, 16
a = torch.randn(n, n); b = torch.randn(n, n)
c_naive = matmul_naive(a, b)
c_tiled = matmul_tiled(a, b, tile)
flops = 2 * n ** 3
n_tiles = (n // tile) ** 3
print(f"矩阵 {n}×{n}: 总计算量 ≈ {flops/1e6:.1f} MFLOPs")
print(f"分块数: 每维 {n//tile} 块 → 共 {n_tiles} 次 16×16×16 的 Cube 操作")
print("分块结果 与 整体结果 一致?", torch.allclose(c_naive, c_tiled, atol=1e-5))
'''),
"✅ 验证:同一笔矩阵乘,拆成 16×16 tile 逐个算,与整体结果一致 —— 这就是 Cube 能『分块算』的数理基础。")

NB.md("## 4️⃣ NPU vs GPU:两种路线的芯片哲学 ⚖️",
D('''
GPU(NVIDIA)与昇腾 NPU 的差别,本质是"通用 vs 专精"的路线之争:

| 维度 | 昇腾 NPU(达芬奇) | GPU(CUDA) |
|------|-----------------|-----------|
| 核心组织 | AI Core 内显式分 Cube/Vector/Scalar | SM 内大量统一 CUDA Core + Tensor Core |
| 矩阵计算 | 专用 Cube(FP16/BF16 高密度) | Tensor Core(与通用核并存) |
| 编程模型 | Ascend C 矢量编程、算子下沉 | CUDA kernel、thread/block/grid |
| 存储层次 | L0/L1/L2 + HBM,多级搬运显式可见 | 寄存器/SMEM/L2 + HBM |
| 指令集 | 昇腾自定义指令集 | PTX / SASS |
| 生态 | CANN + MindSpore | CUDA + cuDNN/cuBLAS |

用柱状图直观对比两种芯片在"矩阵 / 向量 / 标量 / 缓存带宽"上的倾向性:
'''))

NB.code(D('''
labels = ["矩阵算力", "向量算力", "标量/控制", "缓存带宽"]
npu = [5, 4, 2, 4]      # 示意:专精矩阵
gpu = [4, 3, 4, 5]      # 示意:通用与带宽
x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w/2, npu, w, label="昇腾 NPU(达芬奇)", color="#4C78A8")
ax.bar(x + w/2, gpu, w, label="GPU(CUDA)", color="#E45756")
for xi, v, lab in [(x - w/2, npu, "npu"), (x + w/2, gpu, "gpu")]:
    for a, b in zip(xi, v):
        ax.text(a, b + 0.06, f"{b}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("相对强弱(1-5,示意)")
ax.set_title("芯片哲学:NPU 专精矩阵,GPU 兼顾通用与带宽")
ax.legend(); plt.tight_layout()
'''),
"📊 对比图:昇腾把火力集中在矩阵(Cube)上,GPU 则保持通用性 —— 这也决定了 LLM 推理这类『矩阵密集』任务在昇腾上能发挥出高能效。")

NB.md("## 5️⃣ 为什么 LLM 推理爱昇腾 🚀",
D('''
LLM 推理的工作量,绝大部分是 **GEMM(矩阵乘)**:每个 token 都要和权重矩阵做乘加。
这颗大心脏正好是 Cube 单元的主场 —— 所以昇腾芯片对 LLM 格外友好:

1. **矩阵密集**:Prefill 是巨型 GEMM,Decode 是小 GEMM × 海量并发,Cube 都能高速处理;
2. **FP16/BF16 高算力**:LLM 训练推理都用 fp16/bf16,昇腾 910 系列正是按这两个精度设计的;
3. **KV Cache 大**:长上下文下 KV 反复读写,昇腾的 L2/HBM 带宽与多级缓存能扛住访存压力;
4. **整图下沉**:CANN 把计算图一次性编译下沉到芯片,省去逐算子 Host↔Device 往返(下两课展开)。

所以昇腾 + vLLM(vllm-ascend)的组合,是把『矩阵大、访存重』的 LLM 推理压进
专用芯片的典型路径。看懂这颗芯片,后面 CANN 软件栈(73)、算子开发(74)就有了底。
'''))

wrapup(NB,
    summary=[
        "达芬奇架构的核心:统一指令集下,Cube(矩阵)/ Vector(向量)/ Scalar(标量)三单元分工",
        "存储分层 HBM → L2 → L1 → L0,数据分块搬入片内、反复复用,是 Cube 高效的原因",
        "Cube 用 16×16×16 的 Systolic 阵列做矩阵乘;分块 GEMM 在数学上等价于整体 GEMM",
        "NPU 专精矩阵(Cube)、GPU 通用(CUDA Core + Tensor Core),对应两种芯片哲学",
        "LLM 推理矩阵密集 + FP16 + KV 访存重,正是昇腾芯片与整图下沉擅长的场景",
    ],
    practice=[
        "把 matmul_tiled 的 tile 改成 8 / 32,对比结果精度与浮点误差,体会 tile 大小的影响",
        "画一张『AI Core 布局图』:假设芯片有 32 个 AI Core,画出它们如何共享 L2 与 HBM 的拓扑",
        "查一下 Cube 单元处理 BF16 与 FP16 的算力是否翻倍,并解释为什么低精度能『一鱼两吃』",
        "用 time 对比 n=256 时分块与不分组 GEMM 的耗时,分析瓶颈在计算还是搬运(访存)",
    ],
    links=[
        ("昇腾 AI 处理器架构白皮书", "https://www.hiascend.com/document"),
        ("华为昇腾社区", "https://www.hiascend.com"),
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
    ])

out = str(Path(CH11) / "72_davinci_arch.ipynb")
NB.save(out)
finalize(out)
