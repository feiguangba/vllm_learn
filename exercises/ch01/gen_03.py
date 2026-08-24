# -*- coding: utf-8 -*-
"""生成第 03 课 notebook:采样策略(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:打分→选字直觉 -> softmax/温度/top-k/top-p 公式 -> 每个策略逐行实现 -> 四策略对比与数值验证 -> 真实规模 -> vLLM 关联
2. 每一行代码都有 inline 注释
3. 每个中间张量(概率/mask)打印内容与 shape
4. 论文支撑:top-k(Fan et al. arXiv:1805.04833)、nucleus/top-p(Holtzman et al. arXiv:1904.09751)、温度 softmax、HF how-to-generate 博客
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_03_sampling_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 03 课 · 采样策略:把 logits 变成一句话",
    subtitle="greedy / temperature / top-k / top-p —— 公式推导 + 逐行实现 + 数值验证 + vLLM 关联",
    emoji="🎲", chapter="第 1 章 · LLM 推理基础",
)

chapter_cover(
    nb,
    objectives=[
        "理解 logits 的含义:模型给整个词表打的未归一化分数,采样策略负责把它变成输出的 token",
        "掌握带温度的 softmax 公式 P_i = e^{z_i/T} / Σe^{z_j/T},理解温度如何控制分布的陡峭度",
        "实现并理解 top-k 采样(Fan et al., arXiv:1805.04833):固定保留 logits 最大的 k 个候选",
        "实现并理解 top-p / nucleus 采样(Holtzman et al., arXiv:1904.09751):按累积概率动态截断",
        "数值验证:四种策略在同一组 logits 上的行为差异,以及采样频次趋近理论概率",
        "用真实规模数字理解大词表下的实现代价,并建立与 vLLM SamplingParams / logits 处理管线的联系",
    ],
    toc=[
        ("直觉与动机", "从『打分』到『选字』"),
        ("核心定义与公式", "logits / softmax / 温度 / top-k / top-p 的严格定义"),
        ("greedy 解码", "每次都选最高分,确定但易重复"),
        ("温度 softmax", "T 是随机性的旋钮,逐行推演分布变化"),
        ("top-k 采样", "固定人数截断,Fan et al. 2018"),
        ("top-p 采样", "按质量动态截断,Holtzman et al. 2019"),
        ("四策略对比与数值验证", "采样频次趋近理论概率"),
        ("真实规模数字", "大词表下的实现代价与 vLLM 默认参数"),
        ("与 vLLM 的关系", "SamplingParams 与 logits 处理管线"),
        ("Streamlit 动态演示", "动手拧旋钮"),
    ],
    links=[
        ("Fan et al.: Hierarchical Neural Story Generation (top-k)", "https://arxiv.org/abs/1805.04833"),
        ("Holtzman et al.: The Curious Case of Neural Text Degeneration (nucleus)", "https://arxiv.org/abs/1904.09751"),
        ("HuggingFace: How to generate (greedy/beam/sampling)", "https://huggingface.co/blog/how-to-generate"),
        ("vLLM: SamplingParams 文档", "https://docs.vllm.ai/en/latest/features/sampling_params.html"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉与动机:从『打分』到『选字』\n\n"
    "上一课的自回归循环里,模型每步输出的其实不是「一个字」,而是**整个词表**的分数——"
    "叫 **logits**(未归一化的对数几率,可正可负)。分数不是概率,更不能直接拿来选字。\n\n"
    "**采样策略(sampling strategy)** 负责这最后一步:把 logits 变成真正吐出来的那个 token。"
    "不同的策略产生不同的性格:\n\n"
    "- **greedy**:每次都选最高分——确定、可复现,但容易陷入「好的、很好、非常好」的重复空洞;\n"
    "- **temperature**:软化/锐化整个分布——随机性的总旋钮;\n"
    "- **top-k**:只保留分数前 k 名——[Fan et al., 2018](https://arxiv.org/abs/1805.04833) 提出;\n"
    "- **top-p(nucleus)**:按概率质量动态截断——[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 提出。\n\n"
    "> 📄 Holtzman 等人论文的标题就叫 *The Curious Case of Neural Text Degeneration*,"
    "指出贪心/束搜索会退化出重复文本,而「截断不可靠的尾巴(unreliable tail)」能同时保住通顺与多样性。"
    "本课把这四个旋钮逐一实现、逐一验证。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式\n\n"
    "设模型输出 logits 向量 $\\mathbf{z} \\in \\mathbb{R}^{|\\mathcal{V}|}$(每个词一个分数)。\n\n"
    "**softmax(温度 T=1)** 把分数变成概率:\n\n"
    "$$ P_i = \\frac{e^{z_i}}{\\sum_{j=1}^{|\\mathcal{V}|} e^{z_j}} $$\n\n"
    "**带温度的 softmax** 先除以温度 $T$ 再归一化:\n\n"
    "$$ P_i^{(T)} = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}} $$\n\n"
    "$T<1$ 拉大分数差距(分布变尖、更确定);$T>1$ 压缩差距(分布变平、更随机);"
    "$T\\to 0$ 时退化为 greedy(argmax);$T=1$ 保持原分布。\n\n"
    "**top-k** 固定保留分数最高的 $k$ 个:$\\mathcal{V}^{(k)}=\\arg\\max_{|S|=k}\\sum_{x\\in S}P(x)$,"
    "再在其中重新归一化采样。\n\n"
    "**top-p(nucleus)** 从概率最大的 token 往下累积,直到累计概率 $\\ge p$,"
    "得到最小集合 $\\mathcal{V}^{(p)}$,再重新归一化采样。\n\n"
    "| 符号 | 含义 | 本课取值 |\n"
    "|---|---|---|\n"
    "| $\\mathbf{z}$ | logits 向量,长度 $=|\\mathcal{V}|$ | `LOGITS` (len 6) |\n"
    "| $T$ | 温度 | 0.3 / 0.8 / 1.0 / 2.0 |\n"
    "| $k$ | top-k 保留个数 | 1 / 3 / 5 |\n"
    "| $p$ | top-p 概率阈值 | 0.5 / 0.8 / 0.95 |\n"
    "| $\\mathcal{V}^{(k)}, \\mathcal{V}^{(p)}$ | 截断后的候选集合 | `mask` 布尔数组 |\n\n"
    "**统一视角**:top-k 用「固定人数」截断,top-p 用「固定质量」截断——"
    "两者都是 [Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 说的"
    "「*determining the generative model's trustworthy prediction zone*」。"
)

# =====================================================================
# 第 3 节 · greedy
# =====================================================================
nb.md(
    "## 3. greedy 解码:每次都选最高分\n\n"
    "**greedy(贪心)** 是最朴素的策略:直接取 logits(或概率)最大的那个 token。"
    "优点是完全确定、可复现;缺点是**一步错、步步错**——一旦选了当下最顺口的字,"
    "后续就困在同一条路上,容易输出空洞的重复。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Windows 下 OpenMP 库重复加载报错
import numpy as np                                      # numpy 数值运算

VOCAB = ["猫", "狗", "鱼", "鸟", "马", "牛"]             # 玩具词表,|V| = 6
LOGITS = np.array([3.2, 2.1, 1.5, 0.6, 0.2, -0.5], dtype=np.float64)  # 模型给的原始分数(长度=|V|)

print("候选:", VOCAB)
print("logits:", LOGITS)                                # 分数越高,模型越『倾向』这个 token
greedy_idx = int(np.argmax(LOGITS))                     # argmax:取分数最大的下标
print(f"greedy 选中: {VOCAB[greedy_idx]}  (logits={LOGITS[greedy_idx]})")''',
    "🏆 **greedy 只看分数最高那位**——这里是「猫」。注意它完全确定:同一输入永远同一输出。",
)

# =====================================================================
# 第 4 节 · 温度 softmax
# =====================================================================
nb.md(
    "## 4. 温度:softmax 的『放大镜』\n\n"
    "要让低分候选也有机会被选中,先把 logits 变成概率。标准 softmax 见第 2 节;"
    "加温度 $T$ 后,分数先除以 $T$。下面逐行实现并打印不同温度下的分布。"
)

nb.code(
    '''def softmax(logits, temperature=1.0):
    """带温度的 softmax:P_i = exp(z_i/T) / Σexp(z_j/T)。
    返回与 logits 同形状的概率向量(和为 1)。"""
    logits = np.asarray(logits, dtype=np.float64) / temperature   # 除以温度 T:拉大/压缩分数差距
    logits = logits - logits.max()                       # 减去最大值:数值稳定,exp 不溢出
    e = np.exp(logits)                                   # 逐元素指数
    return e / e.sum()                                   # 归一化为概率(和=1)

print(f"{'温度':>4} | " + "  ".join(f"{v:>7}" for v in VOCAB))   # 表头
for T in [0.3, 0.8, 1.0, 2.0]:                           # 遍历 4 个温度
    p = softmax(LOGITS, T)                               # 算温度 T 下的概率分布
    row = "  ".join(f"{x:.4f}" for x in p)               # 每行 4 位小数
    print(f"T={T:<3.1f} | {row}")

print()
for T in [0.3, 1.0, 3.0]:                                # 验证:温度越高,最高概率越低(分布越平)
    p = softmax(LOGITS, T)
    print(f"T={T}: 最高候选「{VOCAB[int(np.argmax(p))]}」概率 = {p.max():.4f}")''',
    "🔥 **看表**:T=0.3 时「猫」几乎垄断;T=2.0 时大家趋近均分。"
    "温度就是随机性的旋钮——这也是 [Ackley et al., 1985] 引入的『模拟退火』式缩放思想。",
)

nb.code(
    '''# 可视化:不同温度下整条概率曲线的形状(柱状图叠加)
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"                # 静态渲染器,notebook 内联展示
import plotly.graph_objects as go

fig = go.Figure()                                        # 新建画布
for T in [0.3, 0.8, 1.0, 2.0, 4.0]:                      # 5 条曲线
    p = softmax(LOGITS, T)                               # 每个温度一个分布
    fig.add_trace(go.Bar(x=VOCAB, y=p, name=f"T={T}"))   # 加一组柱
fig.update_layout(title="不同温度下的概率分布(温度越高越平坦)",
                  yaxis_title="概率", barmode="group", height=360,
                  margin=dict(l=10, r=10, t=50, b=10))
fig.show()                                               # 展示''',
    "🎨 **图形直观**:温度越高,整组柱子越矮越齐(平坦);温度越低,柱子越尖(集中到最高分)。",
)

# =====================================================================
# 第 5 节 · top-k
# =====================================================================
nb.md(
    "## 5. top-k 采样:只看前 k 个\n\n"
    "**top-k** 非常直白:只保留 logits 最大的 $k$ 个候选,其余概率归零,再重新归一化采样。"
    "它固定截断**人数**,实现简单,由 [Fan et al., 2018](https://arxiv.org/abs/1805.04833)"
    "在故事生成中提出并流行(OpenAI GPT-2 的默认采样就是它)。"
    "缺点是 $k$ 对所有上下文一样大:模型很确定时 $k=50$ 引入了垃圾候选,"
    "很发散时 $k=10$ 又砍掉了合理的候选。"
)

nb.code(
    '''def top_k_mask(logits, k):
    """top-k 掩码:保留 logits 最大的 k 个下标,其余 False。返回布尔数组(长度=|V|)。"""
    mask = np.zeros_like(logits, dtype=bool)             # 全 False 掩码
    if k <= 0 or k >= len(logits):                       # k 无效或 ≥ 词表大小
        mask[:] = True                                   # 不过滤(全部保留)
    else:
        mask[np.argsort(logits)[-k:]] = True             # argsort 升序,取末 k 个下标置 True
    return mask                                          # 返回掩码(True=保留)

def demo_topk(k):
    """演示 top-k=k 的效果:打印保留的候选与重归一化后的概率。"""
    mask = top_k_mask(LOGITS, k)                         # 计算掩码(长度=|V|=6)
    keep = [v for v, m in zip(VOCAB, mask) if m]         # 被保留的候选 token
    probs = softmax(LOGITS)                              # 原 softmax 概率(T=1)
    re = probs * mask                                    # 被裁掉的候选概率归零
    re = re / re.sum()                                   # 在保留集内重新归一化(和=1)
    print(f"top-k={k}: 保留 {keep}")
    print(f"          重归一化概率: {[f'{float(x):.3f}' for x in re]}")  # 纯 Python float,显示更干净
    print(f"          mask = {mask.astype(int).tolist()}")  # 打印掩码本身

for k in [1, 3, 5]:                                      # 三种 k 值
    demo_topk(k)''',
    "✂️ **top-k 是『固定人数』的裁切**:无论上下文多复杂,永远只看前 k 名。"
    "掩码(mask)是 $\\mathcal{V}^{(k)}$ 的机器表示。",
)

# =====================================================================
# 第 6 节 · top-p
# =====================================================================
nb.md(
    "## 6. top-p(nucleus)采样:按累积概率截断\n\n"
    "**top-p(也叫 nucleus,核心采样)** 不固定人数,而是**按质量**截断:"
    "从概率最大的 token 开始往下累积,直到累计概率超过 $p$。候选集合的大小是**动态的**:\n\n"
    "- 若某个 token 概率极高(模型很有把握),候选只有 1 个,几乎就是 greedy;\n"
    "- 若分布很平坦(模型很纠结),候选可能多达几十个。\n\n"
    "这正是 [Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 的核心洞察:"
    "『*the size of the candidate set rises and falls dynamically, corresponding to changes "
    "in the model's confidence region*』。"
)

nb.code(
    '''def top_p_mask(logits, p, temperature=1.0):
    """top-p 掩码:在『温度缩放后』的概率上按降序累积到 ≥ p 的最小集合。返回布尔数组(长度=|V|)。"""
    probs = softmax(logits, temperature)                 # 先做与采样完全相同的温度缩放(保证 T+top-p 一致)
    order = np.argsort(probs)[::-1]                      # 按概率降序的下标排列
    cum = np.cumsum(probs[order])                        # 降序累计和
    keep = cum - probs[order] <= p                       # 累计超过 p 之前的都保留(含越过 p 的首个)
    mask = np.zeros_like(probs, dtype=bool)              # 全 False 掩码
    mask[order[keep]] = True                             # 只把『核心集合』的下标置 True
    return mask                                          # 返回掩码

def demo_topp(p):
    """演示 top-p=p 的效果:打印保留的候选个数与内容。"""
    mask = top_p_mask(LOGITS, p)                         # 计算掩码
    keep = [v for v, m in zip(VOCAB, mask) if m]         # 保留的候选 token
    print(f"top-p={p}: 保留 {len(keep)} 个候选 {keep}")   # 集合大小随 p 变化

for p in [0.5, 0.8, 0.95]:                               # 三种阈值
    demo_topp(p)''',
    "🎯 **top-p 的候选数量自动伸缩**:分布集中时少留,分散时多留——"
    "这就是它与 top-k 的本质区别,也是它成为现代推理引擎默认截断的原因。",
)

# =====================================================================
# 第 7 节 · 四策略对比与数值验证
# =====================================================================
nb.md(
    "## 7. 四策略对比与数值验证\n\n"
    "实践中这些策略**叠加**使用:先 temperature 缩放,再 top-k / top-p 截断,"
    "最后在剩余候选上重新归一化并采样。HuggingFace 的 `generate()` 与 vLLM 都遵循"
    "「温度 → top-k → top-p → 归一化 → 采样」的管线"
    "([HF how-to-generate](https://huggingface.co/blog/how-to-generate))。\n\n"
    "> ⚠️ 一个容易忽略的实现细节:**top-p 的掩码必须建立在『温度缩放后』的分布上**。"
    "温度改变了每个候选的概率,也就改变了「累积到 p」的边界——"
    "所以 `sample_from` 里 `top_p_mask` 与 `softmax` 用的是**同一个 `temperature`**,二者始终一致。"
)

nb.code(
    '''def sample_from(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    """完整采样链路:温度 → top-k → top-p → 归一化 → 多项式采样。返回选中的下标。"""
    if rng is None:                                      # 未提供随机数生成器
        rng = np.random.default_rng()                    # 新建一个
    probs = softmax(logits, temperature)                 # ① 温度缩放
    mask = np.ones_like(probs, dtype=bool)               # ② 初始全保留
    if top_k > 0:                                        # ③ 若启用 top-k
        mask &= top_k_mask(logits, top_k)                #    与 top-k 掩码取交集
    if top_p < 1.0:                                      # ④ 若启用 top-p
        mask &= top_p_mask(logits, top_p, temperature)   #    与『温度缩放后』的 top-p 掩码取交集
    filtered = probs * mask                              # ⑤ 被裁掉的概率归零
    filtered = filtered / filtered.sum()                 # ⑥ 在保留集内重新归一化
    return int(rng.choice(len(filtered), p=filtered))    # ⑦ 多项式采样

# ---- 四种策略各采样 10 次,看输出差异 ----
rng = np.random.default_rng(42)                          # 固定种子,可复现
print(f"{'策略':<22} | 10 次采样结果")
configs = [                                              # (名称, 参数)
    ("greedy", dict(top_k=1)),                           # greedy = top-k 保留 1 个
    ("temperature=0.5", dict(temperature=0.5)),          # 低温:偏好高分
    ("top-k=3", dict(top_k=3)),                          # 只看前 3 名
    ("top-p=0.7", dict(top_p=0.7)),                      # 按质量截断
    ("T=0.5 + top-p=0.7", dict(temperature=0.5, top_p=0.7)),  # 叠加:掩码基于缩放后分布
]
for name, kw in configs:                                 # 遍历每种策略
    picks = []                                           # 收集 10 次结果
    for _ in range(10):                                  # 采 10 次
        idx = sample_from(LOGITS, rng=rng, **kw)         # 采样一次
        picks.append(VOCAB[idx])                         # 记录选中 token
    print(f"{name:<22} | " + " ".join(picks))            # 打印序列''',
    "🎲 **对比**:greedy 永远是「猫猫猫…」;temperature/top-k/top-p 给出有变化但符合分布的采样。"
    "注意 greedy 在实现上就是 `top_k=1`;最后一行演示**温度与 top-p 叠加**——"
    "掩码在缩放后的分布上计算,候选集随温度自适应。",
)

nb.code(
    '''# 数值验证:采样频次应趋近理论概率(大数定律)
def empirical_freq(config, n_samples=20000):
    """按配置采样 n 次,返回每个 token 被抽中的频率分布。"""
    rng = np.random.default_rng(0)                       # 固定种子
    counts = np.zeros(len(VOCAB))                        # 频次数组(长度=|V|)
    for _ in range(n_samples):                           # 采 n 次
        idx = sample_from(LOGITS, rng=rng, **config)     # 采样
        counts[idx] += 1                                 # 计数 +1
    return counts / n_samples                            # 归一化为频率

def check_consistency(config):
    """核对某配置下『理论分布(掩码后重归一化)』与『采样频率』是否一致。"""
    T = config.get("temperature", 1.0)                   # 该配置的温度(默认 1.0)
    p = config.get("top_p", 1.0)                         # 该配置的 top-p(默认 1.0)
    mask = top_p_mask(LOGITS, p, T)                      # 与采样『同温度』的理论截断掩码
    theory = softmax(LOGITS, T) * mask                   # 理论分布(温度缩放后)
    theory = theory / theory.sum()                       # 重归一化
    freq = empirical_freq(config)                        # 实测频率
    return theory, freq

# ① 纯 top-p(T=1):掩码在原始分布上算
theory, freq = check_consistency(dict(top_p=0.7))
print(f"{'token':>4} | {'理论概率':>8} | {'采样频率':>8} | {'偏差':>8}")
for i, v in enumerate(VOCAB):                            # 逐个 token 对比
    print(f"{v:>4} | {theory[i]:8.4f} | {freq[i]:8.4f} | {abs(theory[i]-freq[i]):8.4f}")
print(f"\\n纯 top-p 最大偏差 = {np.abs(theory - freq).max():.4f} (样本数 20000,偏差应趋近 0)")

# ② 温度+top-p 叠加(T=0.5):掩码必须建立在缩放后的分布上,才能对上采样频率
theory2, freq2 = check_consistency(dict(temperature=0.5, top_p=0.7))
print(f"T=0.5+top-p 最大偏差 = {np.abs(theory2 - freq2).max():.4f} (掩码基于缩放分布,才会一致)")''',
    "✅ **大数定律验证**:采样 2 万次后,每个 token 被抽中的频率与理论概率几乎重合——"
    "证明 `sample_from` 的整条管线(温度→截断→归一化→采样)数学上正确。"
    "第 ② 组专门验证「温度+top-p」叠加:掩码与采样用**同一个温度**,偏差才趋近 0。",
)

# =====================================================================
# 第 8 节 · 真实规模数字
# =====================================================================
nb.md(
    "## 8. 真实规模数字:大词表下的实现代价\n\n"
    "玩具词表只有 6 个,真实模型词表 3 万~15 万。规模放大后有两个工程点:"
)

nb.code(
    '''# 真实词表规模下,每个策略的成本与行为
V_REAL = 128256                                         # LLaMA-3 词表大小
print(f"真实词表 |V| = {V_REAL:,}")

# 成本:top-p 需要一次排序 + 一次 cumsum,与词表大小线性
import time
big_logits = np.random.randn(V_REAL)                    # 随机 128k 维 logits
t0 = time.perf_counter()                                 # 计时开始
m = top_p_mask(big_logits, 0.9)                          # 对 128k 词表做 top-p
dt = (time.perf_counter() - t0) * 1e3                    # 毫秒
print(f"对 {V_REAL:,} 个词做一次 top-p 截断耗时: {dt:.3f} ms,保留 {int(m.sum())} 个候选")

# 行为:同一分布下,top-p 的候选数 vs top-k 的固定人数
toks3 = softmax(big_logits[:8])                          # 取 8 个候选的小分布做对照
print(f"\\n对照(8 个候选): top-p=0.9 保留 {int(top_p_mask(big_logits[:8], 0.9).sum())} 个;"
      f"top-k=5 永远保留 5 个")''',
    "🚀 **要点**:top-p 在工程上是一个排序 + 一次 `cumsum`,开销与词表大小线性(毫秒级),"
    "因此成了 vLLM 的默认截断。而「候选数随分布自适应」正是它在真实词表上更好用的原因。",
)

nb.md(
    "**常用参数速查**(vLLM `SamplingParams` 与 HF `generate` 对齐):\n\n"
    "| 参数 | 含义 | 常见默认 | 说明 |\n"
    "|---|---|---|---|\n"
    "| `temperature` | 温度 | 1.0(greedy 用 0) | 0 走 argmax 快路径 |\n"
    "| `top_k` | 保留前 k 名 | 关闭/50 | 固定人数截断 |\n"
    "| `top_p` | 累积概率阈值 | 0.9~1.0 | 动态截断,主流默认 |\n"
    "| `seed` | 随机种子 | 关闭 | 固定后输出可复现 |\n"
    "| `repetition_penalty` | 重复惩罚 | 1.0 | 对已出现的 token 压分 |\n\n"
    "> ⚠️ greedy 在工程上**不走采样**:直接把 logits 过 `argmax`,省去 softmax 与随机数。"
    "这也是 `temperature=0` 语义的真相。"
)

# =====================================================================
# 第 9 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 9. 与 vLLM 的关系:SamplingParams 与 logits 处理管线\n\n"
    "vLLM 把采样参数封装成 `SamplingParams`,在 `vllm/v1/engine/__init__.py` 定义;"
    "模型每步输出 logits 后,由 `vllm/model_executor/layers/sampler.py` 的 `Sampler` 处理:\n\n"
    "1. **logits 处理器(logits processors)**:按参数依次施加 temperature、top-k、top-p、"
    "penalties —— 与本课 `sample_from` 的「温度→top-k→top-p」顺序一致;\n"
    "2. **采样**:greedy 走 `torch.argmax`;采样走 `torch.multinomial`(等价于多项式采样);\n"
    "3. **确定性**:`seed` 参数会重置 CUDA RNG,使同 prompt 同参数可复现。\n\n"
    "**为什么本课重要**:采样策略不改变模型本身,却完全决定用户看到的文本。"
    "推理引擎层(如 vLLM)只是把「选字」这一步做成可配置、可批处理的算子——"
    "一条 batch 里不同请求可以用不同的 temperature/top-p,互不干扰。"
)

# =====================================================================
# 第 10 节 · Streamlit
# =====================================================================
nb.md(
    "## 10. 🖥️ Streamlit 动态演示:动手拧旋钮\n\n"
    "运行 `app_03_sampling_demo.py`:温度 / top-k / top-p 三根滑杆任意拖动,"
    "概率分布柱状图实时重绘,右侧还能看到采样几百次后的命中频次。\n\n"
    "### 📜 App 完整源码(`app_03_sampling_demo.py`)"
)

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + textwrap.indent(APP_CODE, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_03_sampling_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_03_sampling_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_03_sampling_demo.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_03_sampling_demo.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,拖动滑杆观察分布与采样频次。\n\n"
    "🔍 试试:温度拉到 3.0 再配合 top-p=0.7,看分布先变平、再被截断尾巴。"
)

wrapup(
    nb,
    summary=[
        "logits 是模型给整个词表打的原始分;采样策略负责把 logits 变成最终输出的 token",
        "带温度 softmax P_i = e^{z_i/T}/Σe^{z_j/T}:T<1 变尖(更确定),T>1 变平(更随机),T→0 退化为 greedy",
        "top-k(Fan et al. 2018)固定保留前 k 名,简单但 k 不随上下文自适应",
        "top-p / nucleus(Holtzman et al. 2019)按累积概率 ≥p 动态截断,候选数随模型置信度伸缩",
        "工程上「温度→top-k→top-p→归一化→采样」是标准管线;greedy 走 argmax 快路径",
        "vLLM 把策略封装成 SamplingParams,由 Sampler 统一施加 logits 处理器并支持 per-request 参数",
    ],
    practice=[
        "实现 repetition_penalty:对已出现的 token 把 logits 除以惩罚系数,看重复如何被抑制",
        "把 top_p_mask 改成「严格越过 p 才算」(keep = cum <= p),对比与当前实现的候选数差异",
        "用种子固定采样,验证同一 seed 下多次运行输出完全一致;再换 seed 观察变化",
        "思考题:为什么 top-p 在模型很确定时≈greedy,而 top-k 不会?结合分布形状解释",
    ],
    links=[
        ("Fan et al.: Hierarchical Neural Story Generation", "https://arxiv.org/abs/1805.04833"),
        ("Holtzman et al.: The Curious Case of Neural Text Degeneration", "https://arxiv.org/abs/1904.09751"),
        ("HuggingFace: How to generate", "https://huggingface.co/blog/how-to-generate"),
        ("vLLM: SamplingParams", "https://docs.vllm.ai/en/latest/features/sampling_params.html"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch01\03_sampling_strategies.ipynb")