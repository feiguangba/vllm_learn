# -*- coding: utf-8 -*-
"""生成第 02 课 notebook:自回归生成(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:接龙直觉 -> 语言模型/链式法则/n-gram 定义 -> n-gram 统计逐行 -> 自回归循环 -> L次前向验证 -> 真实规模 -> vLLM 关联
2. 每一行代码都有 inline 注释
3. 每个中间结构(词表/条件分布/路径)打印长度与内容
4. 论文支撑:Shannon 1948(n-gram 起源)、Jurafsky & Martin SLP3、Attention Is All You Need(训练并行 vs 推理串行)
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_02_generate_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 02 课 · 自回归生成:一个 token 接一个 token",
    subtitle="语言模型与链式法则 · n-gram 玩具模型逐行实现 · 「生成长度 L = L 次前向」的串行本质",
    emoji="🔁", chapter="第 1 章 · LLM 推理基础",
)

chapter_cover(
    nb,
    objectives=[
        "理解语言模型(LM)的准确定义:在 token 序列上定义的概率分布,以及链式法则分解 P(x_1..x_T)=∏P(x_t|x_<t)",
        "掌握 n-gram 模型(Shannon 1948 提出)与 Markov 假设:下一个 token 只依赖最近 n-1 个 token",
        "逐行实现 bigram/trigram 的频次统计与条件概率查询(next_candidates),每个中间结构打印长度与内容",
        "实现自回归生成循环:预测 → 采样 → 拼接回上下文,亲眼看到「生成结果反过来影响后续预测」",
        "数值验证「生成长度 L 需要 L 次前向」的串行链条,并用 plotly 可视化逐步解码路径",
        "用真实规模数字理解 decode 为什么慢、TTFT 从哪里来,建立与 vLLM decode 循环的联系",
    ],
    toc=[
        ("直觉与动机", "接龙游戏与自动补全"),
        ("核心定义与公式", "语言模型、链式法则、n-gram、Markov 假设"),
        ("最小实现 · n-gram 统计", "build_ngram / next_candidates 逐行注释"),
        ("自回归生成循环", "预测 → 采样 → 拼接回上下文"),
        ("数值验证", "「L 个 token = L 次前向」计数实验"),
        ("解码路径可视化", "plotly 逐步铺开,看串行追加"),
        ("真实规模数字", "GPT 规模下的 decode 成本与 TTFT"),
        ("与 vLLM 的关系", "decode 循环、KV Cache 与生成配置"),
        ("Streamlit 动态演示", "动手玩接龙"),
    ],
    links=[
        ("Shannon: A Mathematical Theory of Communication (1948)", "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x"),
        ("Jurafsky & Martin: SLP3 Ch.3 N-gram Language Models", "https://web.stanford.edu/~jurafsky/slp3/3.pdf"),
        ("Vaswani et al.: Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("GPT-2: Language Models are Unsupervised Multitask Learners", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉与动机:接龙游戏与自动补全\n\n"
    "想象你在玩**词语接龙**:我说「机器」,你接「器学」,我再接「学习」……"
    "大模型生成文本就是这一回事——它一次只看**当前已生成的整段话**,"
    "只预测**下一个** token,然后把这个新 token 拼到末尾,作为下一次预测的输入。"
    "这个过程叫**自回归(autoregressive)生成**,意思是「输出回归于它自己之前的结果」。\n\n"
    "**为什么必须是这个形状?** 语言是序列的,一句话的意思由 token 的顺序承载。"
    "顺序地、一次一个地预测,才能保证第 $t$ 个 token 依赖前 $t-1$ 个——"
    "这与人类读写的方式一致,也与 [Vaswani et al., 2017]"
    "(https://arxiv.org/abs/1706.03762) 里 Transformer 的**因果掩码(causal mask)** 严格对应:\n\n"
    "> 「*we modify the self-attention sub-layer in the decoder stack to prevent positions "
    "from attending to subsequent positions … the predictions for position $i$ can depend "
    "only on the known outputs at positions less than $i$*」\n\n"
    "本课先用一个**玩具语言模型(n-gram)** 把这个过程逐行跑通——"
    "它没有神经网络,却抓住了「统计语言模型」的本质,"
    "也是 Shannon 1948 年开创信息论时就用过的模型。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式:语言模型与链式法则\n\n"
    "**定义(语言模型)** 设词表 $\\mathcal{V}$,语言模型是定义在 token 序列上的概率分布 $P$,"
    "它回答「这句话有多像人话」:$P(x_1,\\dots,x_T)$ 越大,句子越自然。\n\n"
    "**链式法则(chain rule)** 联合概率总能拆成条件概率的乘积——这是**精确**的:\n\n"
    "$$ P(x_1,\\dots,x_T) = \\prod_{t=1}^{T} P(x_t \\mid x_1, \\dots, x_{t-1}) $$\n\n"
    "于是「建模整句话」等价于「逐位建模『看到什么,接着说什么』」。"
    "问题来了:$P(x_t \\mid x_{1:t-1})$ 的条件太长了,无法统计。\n\n"
    "**Markov 假设** 做一个简化:假设第 $t$ 个 token 只依赖它**前面最近的 $n-1$ 个** token。"
    "这时的模型叫 **n-gram**(n=1 unigram,n=2 bigram,n=3 trigram),用最大似然估计:\n\n"
    "$$ P(x_t \\mid c) = \\frac{\\text{count}(c,\\, x_t)}{\\sum_{x'} \\text{count}(c,\\, x')},"
    "\\qquad c = x_{t-n+1:t-1} $$\n\n"
    "| 符号 | 含义 | 本课取值 |\n"
    "|---|---|---|\n"
    "| $n$ | 阶数:上下文长度 + 1 | 2(bigram)/3(trigram) |\n"
    "| $c$ | 上下文(context),长度为 n-1 的 token 元组 | `tuple` |\n"
    "| $\\text{count}(c, x)$ | 语料中(c 后接 x)的出现次数 | `Counter` |\n"
    "| $P(x_t\\mid c)$ | 条件概率分布,是采样/贪心的输入 | 归一化计数 |\n\n"
    "> 📄 这一统计语言模型的框架正是 Shannon 1948 年在《A Mathematical Theory of "
    "Communication》里引入的,见 [Jurafsky & Martin, SLP3 Ch.3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) "
    "的「Historical Notes」。现代 LLM 用神经网络估计同样的条件概率——本质是同一个目标。"
)

# =====================================================================
# 第 3 节 · 最小实现:n-gram 统计
# =====================================================================
nb.md(
    "## 3. 最小实现 · n-gram 统计:逐行注释\n\n"
    "用中文小语料实现 bigram 统计。核心数据结构是 `defaultdict(Counter)`:"
    "上下文元组 → {下一个 token: 次数}。每一步都打印结构,方便核对。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Windows 下 OpenMP 库重复加载报错
from collections import defaultdict, Counter             # 嵌套字典:上下文 -> 计数器

def build_ngram(corpus, n):
    """统计 n-gram:建立 (前 n-1 个 token) -> {下一个 token: 次数} 的字典。
    返回 defaultdict(Counter)。n=2 即 bigram,n=3 即 trigram。
    """
    model = defaultdict(Counter)                         # 上下文 -> Counter{下一 token: 次数}
    for text in corpus:                                  # 遍历语料里每条文本
        toks = list(text)                                # 文本拆成 token 序列(这里按字符)
        for i in range(len(toks) - n + 1):               # 滑动窗口,产生所有 n-gram
            ctx = tuple(toks[i:i + n - 1])               # 前 n-1 个 token 组成上下文元组
            model[ctx][toks[i + n - 1]] += 1             # 记录「上下文后面接了什么」,计数 +1
    return model                                         # 返回条件计数表

def next_candidates(model, context, n):
    """给定已生成的上下文,返回 (候选 token 列表, 归一化概率列表),按次数降序。
    上下文不足 n-1 个时退化为「只看着已有的几个」。"""
    ctx = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)  # 取最近 n-1 个
    counts = model.get(ctx, Counter())                   # 查到该上下文的条件计数(没有则空)
    total = sum(counts.values())                         # 总次数 = 归一化分母
    if total == 0:                                       # 语料里从没见过这个上下文
        return [], []                                    # 无候选,调用方应停止
    items = sorted(counts.items(), key=lambda kv: -kv[1])  # 按次数降序排序(概率高的在前)
    toks = [t for t, _ in items]                         # 候选 token 列表
    probs = [c / total for _, c in items]                # MLE 条件概率 count/∑count
    return toks, probs                                   # 返回候选与概率''',
    "🛠️ **两块积木**。`build_ngram` 扫一遍语料,建立条件计数表;"
    "`next_candidates` 按公式 $P(x_t|c)=\\text{count}/\\sum\\text{count}$ 输出条件分布。"
    "注意 `total==0` 的兜底:没见过的上下文直接返回空——真实 LLM 也会遇到类似「没见过」的情况。",
)

nb.code(
    '''# 在 4 条中文句子上训练 bigram,观察条件计数表
corpus = ["我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉", "小狗喜欢奔跑和玩耍", "小猫喜欢睡觉"]  # 迷你语料
model = build_ngram(corpus, n=2)                         # n=2:bigram(上下文=1 个 token)
print(f"语料里出现过的上下文种类: {len(model)} 种")       # 每种上下文 = 1 个键
for ctx in [("我",), ("喜",), ("猫",)]:                   # 注意:model 的键是元组 tuple
    print(f"  上下文「{''.join(ctx)}」 -> {dict(model[ctx])}")  # {下一 token: 次数}

toks, probs = next_candidates(model, ["我"], 2)           # 查「我」后面接什么
print(f"\\n上下文「我」-> 候选 {toks}")
print(f"              概率 {[round(p, 3) for p in probs]} (和 = {round(sum(probs), 3)})")''',
    "📖 **观察**:「喜」后面跟着「欢」(3 次)的概率远高于其他;"
    "每一行条件概率和都为 1——这就是一个离散条件分布,也是采样策略(第 03 课)的输入。",
)

# =====================================================================
# 第 4 节 · 自回归生成循环
# =====================================================================
nb.md(
    "## 4. 自回归生成循环:预测 → 采样 → 拼回上下文\n\n"
    "玩具模型就位,自回归生成是一个循环:\n\n"
    "1. 拿当前已生成的 token 序列作**上下文**;\n"
    "2. 查表得到「下一个 token 的条件概率分布」;\n"
    "3. 按策略选一个(这里先引入第 03 课会细讲的采样管线:温度/top-k/top-p);\n"
    "4. 把新 token **追加**到上下文末尾;\n"
    "5. 回到第 1 步,直到生成够 $L$ 个。\n\n"
    "先定义采样管线(第 03 课会展开成独立函数),再实现 `generate_ngram`。"
)

nb.code(
    '''# ---- 采样管线(第 03 课会逐一展开,这里先给出完整版)----
import numpy as np

def softmax(logits, temperature=1.0):
    """带温度的 softmax:P_i = exp(z_i/T) / Σexp(z_j/T)。T>1 变平(更随机),T<1 变尖。"""
    logits = np.asarray(logits, dtype=np.float64) / temperature   # 除以温度,缩放分布陡峭度
    logits = logits - logits.max()                       # 减最大值,数值稳定(防 exp 溢出)
    e = np.exp(logits)                                   # 指数化
    return e / e.sum()                                   # 归一化成概率,形状同 logits

def top_k_mask(logits, k):
    """保留 logits 最大的 k 个候选,其余置 False。"""
    mask = np.zeros_like(logits, dtype=bool)             # 全 False 掩码
    if k <= 0 or k >= len(logits):                       # k 无效或全覆盖
        mask[:] = True                                   # 不过滤
    else:
        mask[np.argsort(logits)[-k:]] = True             # 排序取最大的 k 个下标置 True
    return mask                                          # 返回布尔掩码

def top_p_mask(logits, p):
    """top-p(nucleus):从概率最大的候选往下累积,直到累计概率 ≥ p。"""
    probs = softmax(logits)                              # 先算概率
    order = np.argsort(probs)[::-1]                      # 按概率降序的下标
    cum = np.cumsum(probs[order])                        # 降序累计和
    keep = cum - probs[order] <= p                       # 超过 p 之前的都保留(含刚超过)
    mask = np.zeros_like(probs, dtype=bool)              # 全 False 掩码
    mask[order[keep]] = True                             # 只保留核心集合
    return mask                                          # 返回布尔掩码

def sample_from(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    """完整采样链路:温度 → top-k → top-p → 归一化 → 多项式采样。返回选中的下标。"""
    if rng is None:                                      # 未提供随机数生成器
        rng = np.random.default_rng()                    # 新建一个
    probs = softmax(logits, temperature)                 # 温度缩放后的概率
    mask = np.ones_like(probs, dtype=bool)               # 初始全保留
    if top_k > 0:                                        # 若启用 top-k
        mask &= top_k_mask(logits, top_k)                # 与 top-k 掩码取交集
    if top_p < 1.0:                                      # 若启用 top-p
        mask &= top_p_mask(logits, top_p)                # 与 top-p 掩码取交集
    filtered = probs * mask                              # 被裁掉的候选概率归零
    filtered = filtered / filtered.sum()                 # 在剩余候选中重新归一化
    return int(rng.choice(len(filtered), p=filtered))    # 按最终概率多项式采样''',
    "🎲 **采样管线**。温度控制整体陡峭度,top-k 固定人数、top-p 按质量动态截断——"
    "三者可叠加。这里先封装好,第 03 课会逐一拆开推演。",
)

nb.code(
    '''# ---- 自回归生成主循环 ----
def generate_ngram(model, n, start, steps, temperature=1.0, top_k=0, top_p=1.0, seed=0):
    """自回归生成:每步把刚生成的 token 追加进上下文再预测下一个。
    返回 (完整 token 列表, 逐步路径)。"""
    rng = np.random.default_rng(seed)                    # 固定随机种子,结果可复现
    context = list(start)                                # 当前上下文,初始为起始文本
    out = list(start)                                    # 输出序列,初始也含起始文本
    path = []                                            # 记录每一步的中间量(供可视化)
    for _ in range(steps):                               # 循环生成 steps 个 token
        toks, probs = next_candidates(model, context, n) # 查表得条件分布
        if not toks:                                     # 上下文无候选(语料没见过)
            break                                        # 提前停止
        logits = np.log(np.array(probs, dtype=np.float64) + 1e-12)  # 概率转 logits(对数空间)
        idx = sample_from(logits, temperature, top_k, top_p, rng)   # 采样选一个
        chosen = toks[idx]                               # 选中的 token
        out.append(chosen)                               # 拼进输出序列
        path.append((list(context), toks, probs, chosen))  # 记录该步(上下文/候选/概率/选中)
        context.append(chosen)                           # 新 token 成为上下文一部分(关键!)
    return out, path                                     # 返回生成结果与路径''',
    "🔁 **核心循环**。注意第 12 行 `context.append(chosen)`:刚生成的 token 立刻成为"
    "下一次预测的上下文——「生成结果反过来影响后续预测」,这正是自回归的『自』字。",
)

nb.code(
    '''# 实际生成 8 个 token,逐步打印
out, path = generate_ngram(model, n=2, start="我喜欢", steps=8)  # bigram,起点「我喜欢」
print("生成结果:", "".join(out))                          # 完整序列(含起始文本)
print("模型新增:", "".join(out[len("我喜欢"):]))           # 模型自己续写的部分
print()
print("逐步解码路径(上下文 -> 候选分布 -> 选中):")
for step, (ctx, toks, probs, chosen) in enumerate(path, 1):    # 遍历每步
    cand = " ".join(f"{t}({p:.2f})" for t, p in zip(toks, probs))  # 候选与概率
    print(f"  第 {step} 步:上下文「{''.join(ctx)}」 -> 候选[{cand}] -> 选中「{chosen}」")''',
    "💡 **你会看到**:每一步只吐一个字,且从第 2 步起上下文里多了上一步刚生成的字——"
    "生成的序列在『喂养』自己,这就是自回归。",
)

# =====================================================================
# 第 5 节 · 数值验证
# =====================================================================
nb.md(
    "## 5. 数值验证:「L 个 token = L 次前向」\n\n"
    "自回归循环**无法并行**:要算第 3 个 token,必须等第 2 个生成;"
    "要算第 2 个,又得等第 1 个。所以生成长度 $L$ 是一条**串行链**——"
    "恰好需要 $L$ 次「预测下一个」的前向。下面用计数实验实锤,"
    "并画出逐步路径。"
)

nb.code(
    '''# 计数实验:统计 generate 内部实际做了几次「预测」
def count_forward(model, n, start, steps):
    """模拟生成 steps 步,统计实际调用 next_candidates 的次数(= 前向次数)。"""
    context, calls = list(start), 0                      # 上下文 + 计数器
    for _ in range(steps):                               # 每步一次预测
        toks, _ = next_candidates(model, context, n)     # 查表(相当于一次前向)
        if not toks:                                     # 无候选
            break                                        # 停止
        calls += 1                                       # 前向次数 +1
        context.append(toks[0])                          # greedy 取最高概率,拼回上下文
    return calls                                         # 返回前向次数

for L in [1, 3, 5, 8]:                                   # 想生成几个 token
    c = count_forward(model, 2, "我喜欢", L)              # 实际需要几次前向
    print(f"想生成 {L} 个 token -> 需要 {c} 次前向 {'✅' if c == L else '❌'}")''',
    "✅ **结论实锤**:生成几个 token,就做几次前向。生成长度**线性**决定前向次数,"
    "无法靠加大 batch 缩短这条串行链(但可以靠批处理摊薄单次成本——第 06 课)。",
)

nb.code(
    '''# 可视化:把整条解码路径用 plotly 铺开(蓝=提示词,红=模型生成)
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"                # 静态渲染器,notebook 内联展示
import plotly.graph_objects as go

def viz_decode(tokens, start_len):
    """画一条解码路径:tokens 是完整序列,start_len 之前是提示词。"""
    n = len(tokens)                                      # 序列总长
    colors = ["#9ECAE1" if i < start_len else "#E45756" for i in range(n)]  # 蓝=提示/红=生成
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=[0] * n, mode="lines",
                             line=dict(color="#BBBBBB", width=1)))          # 底部连线
    fig.add_trace(go.Scatter(x=list(range(n)), y=[0] * n, mode="markers+text",
                             text=tokens, textposition="top center",        # 每个 token 标在上方
                             marker=dict(size=20, color=colors)))
    fig.update_layout(title="自回归解码路径(蓝=提示词,红=模型生成)",
                      xaxis_title="token 位置", yaxis=dict(showticklabels=False, range=[-0.6, 0.6]),
                      height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig                                          # 返回 Figure 对象

out2, _ = generate_ngram(model, n=2, start="我喜欢", steps=8)  # 重新生成一次
viz_decode(out2, len("我喜欢"))                          # 展示:红字一个个接在蓝字后面''',
    "🎨 **可视化**:每个新 token 都依赖它左边所有 token,所以只能从左到右依次产生——"
    "一眼看出生成是**串行追加**的。",
)

# =====================================================================
# 第 6 节 · 真实规模数字
# =====================================================================
nb.md(
    "## 6. 真实规模数字:decode 为什么慢\n\n"
    "把玩具模型换成真实 LLM,串行本质不变,只是每次「前向」的代价变了。"
    "记 $N$ 为参数量,经验上每处理一个 token 的前向约需 $2N$ 次浮点运算"
    "([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 的 scaling-law 口径),"
    "于是:"
)

nb.code(
    '''# 计算真实规模下的前向次数与 FLOPs 账本
def flops_for_decode(N, L_prompt, K_gen):
    """自回归生成 K 个 token:prefill 一次并行处理 L 个,decode 逐 token K 次前向。
    总 FLOPs ≈ 2N*(L + K)。"""
    return 2 * N * (L_prompt + K_gen)                    # 每 token ~2N,总 token 数 L+K

# 例:GPT-2 Medium(N=345M),prompt 512,生成 128
N = 345e6                                               # GPT-2 Medium 参数量 3.45 亿
L_prompt, K_gen = 512, 128                               # 提示词 512 个,生成 128 个
total = flops_for_decode(N, L_prompt, K_gen)             # 总 FLOPs
print(f"GPT-2 Medium (N={N/1e6:.0f}M): prefill 1 次前向({L_prompt} token) + decode {K_gen} 次前向")
print(f"  总 FLOPs ≈ 2N(L+K) = {total/1e12:.3f} TFLOPs")
print(f"  decode 独占 {K_gen} 次串行前向(占前向次数的 {K_gen/(K_gen+1)*100:.1f}%)")
print()
print("含义:prefill 是 1 次『大』并行前向,decode 是 K 次『小』串行前向。")
print("同一模型,同样的总 token 数,decode 阶段的墙钟时间往往远超 prefill —— 第 05 课实测。")''',
    "🚀 **要点**:decode 的 $K$ 次前向每次只算 1 个新 token,矩阵很小,"
    "GPU 算力吃不满;真正慢的原因是**串行依赖 + 每步都要搬运全部权重/KV**。"
    "用户感知上,首字等待时间叫 **TTFT(prefill 决定)**,之后每字间隔叫 **ITL/TPOT(decode 决定)**。",
)

# =====================================================================
# 第 7 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 7. 与 vLLM 的关系:decode 循环的工程化\n\n"
    "vLLM 的生成引擎做的正是本课的循环,只是每次「前向」被拆成两条流水线:\n\n"
    "1. **prefill**:一次性吃下整段 prompt,生成全部 K/V 并写入 KV Cache;\n"
    "2. **decode 循环**:每步只喂**最新 1 个 token**,模型算它的 Q,"
    "与缓存中的全部历史 K/V 做注意力,再输出下一个 token 的概率。\n\n"
    "本课玩具循环里 `context.append(chosen)` 这一步,"
    "在 vLLM 里对应 **KV Cache 的追加写**:历史 token 的 K/V 不需要重算,"
    "只需把新 token 的 K/V 追加进缓存(第 2 章第 07 课详细展开)。\n\n"
    "**与采样策略的关系**:模型每步输出的是**整个词表**的 logits(而非 1 个字),"
    "真正「选谁」由 `SamplingParams` 决定——这正是第 03 课的主题。\n\n"
    "**与批处理的关系**:串行链无法缩短,但**多条请求的 decode 可以并成一趟前向**——"
    "这是 vLLM 连续批处理(continuous batching)的核心动机,第 06 课会亲手实测「批大小 → 吞吐」曲线。"
)

# =====================================================================
# 第 8 节 · Streamlit
# =====================================================================
nb.md(
    "## 8. 🖥️ Streamlit 动态演示:动手玩接龙\n\n"
    "运行 `app_02_generate_demo.py`:选语料主题、切换 bigram/trigram、改起始文本与生成步数,"
    "每一步的候选概率柱状图与最终生成结果实时刷新。\n\n"
    "### 📜 App 完整源码(`app_02_generate_demo.py`)"
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
    "    print(\"    请把上方源码保存为 app_02_generate_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_02_generate_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_02_generate_demo.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_02_generate_demo.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,体验逐字接龙。\n\n"
    "🔍 试试:trigram 的候选更「有依据」,但也更容易跑出语料外的死路。"
)

wrapup(
    nb,
    summary=[
        "语言模型是在 token 序列上定义的概率分布,链式法则把它精确分解成逐位条件概率的乘积",
        "n-gram 用 Markov 假设近似:P(x_t|x_1..x_{t-1}) ≈ P(x_t|最近 n-1 个),用计数做最大似然估计(Shannon 1948 起源)",
        "自回归生成 = 预测 → 采样 → 拼回上下文 的循环,生成结果反过来影响后续预测(上下文自喂养)",
        "生成长度 L 需要恰好 L 次串行前向,这是 decode 慢的结构性根源,无法靠并行消除",
        "真实 LLM 里该循环对应 prefill(建 KV Cache)+ decode(每步只算新 token)两阶段",
        "vLLM 的生成引擎就是本课循环的工程化:历史 K/V 缓存复用 + 采样策略 + 连续批处理",
    ],
    practice=[
        "把 n 从 2 改成 3(trigram),对比同一语料上生成结果与候选分布的差异,解释为什么 trigram 更『有依据』但更容易断供",
        "给 next_candidates 加上 add-1(拉普拉斯)平滑,让没见过的上下文也有非零概率,观察生成有什么变化",
        "用 count_forward 验证「语料没见过的上下文会提前 break」:把 start 改成语料外的词观察停止行为",
        "思考题:为什么 GPT 训练时可以一次并行预测整句的每个位置(teacher forcing),而推理必须逐个生成?提示:因果掩码",
    ],
    links=[
        ("Shannon 1948: A Mathematical Theory of Communication", "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x"),
        ("Jurafsky & Martin: SLP3 Ch.3", "https://web.stanford.edu/~jurafsky/slp3/3.pdf"),
        ("Vaswani et al.: Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("Kaplan et al.: Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch01\02_autoregressive_generation.ipynb")