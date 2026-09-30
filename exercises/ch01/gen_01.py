# -*- coding: utf-8 -*-
"""生成第 01 课 notebook:认识 Token 与分词器(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:直觉动机 -> 核心定义与符号表 -> 字符级分词 -> BPE 逐行推演 -> 数值验证 -> 真实规模 -> vLLM 关联
2. 每一行代码都有 inline 注释(说明这行做什么/为什么/产出什么)
3. 每个序列/词表打印长度与内容,标注维度含义
4. 论文支撑:BPE(Sennrich et al. arXiv:1508.07909)、GPT-2 byte-level BPE(Radford et al. 2019)、minbpe
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_01_token_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 01 课 · 认识 Token 与分词器:从字符到子词",
    subtitle="字符级分词 · token→id 双向映射 · BPE 训练与编码逐行推演 · 词表大小 vs 序列长度的权衡",
    emoji="🏷️", chapter="第 1 章 · LLM 推理基础",
)

chapter_cover(
    nb,
    objectives=[
        "理解 token 与词表(vocabulary)的精确含义:模型为什么只认数字、不认字",
        "建立 token 化函数的数学定义 τ(text)→tokens 与双向映射 id(·)/decode(·),搞清楚 <unk> 兜底与词尾标记 </w> 的作用",
        "字符级分词起步:手写 char_tokenize / build_vocab / tokenize_to_ids,每个序列打印长度与内容",
        "逐行实现 BPE(字节对编码)训练循环与编码函数,跟踪每一步合并规则、序列长度与词表大小",
        "数值验证:编码-还原往返一致、合并让序列变短、词表随合并单调增长",
        "用真实规模数字对比(GPT-2 词表 50257、byte-level BPE、中文 token 成本),并联系 vLLM 请求管线中的分词位置",
    ],
    toc=[
        ("直觉与动机", "为什么模型只认 token 不认字 —— 编码与数字化"),
        ("核心定义与公式", "token 化函数、词表、id 映射的形式定义与符号表"),
        ("字符级分词", "最朴素约定,每个中间序列打印长度"),
        ("token → id 映射", "<unk> 兜底、逆映射,双向翻译"),
        ("最小 BPE 实现", "统计相邻对 → 合并最高频对 → 编码,逐行注释"),
        ("数值验证", "还原一致性、压缩率、词表增长"),
        ("真实规模数字", "GPT 词表、byte-level BPE、中文 token 成本"),
        ("与 vLLM 的关系", "分词器在请求预处理管线中的位置"),
        ("Streamlit 动态演示", "输入文字实时看切分"),
    ],
    links=[
        ("BPE 论文: Neural Machine Translation of Rare Words with Subword Units", "https://arxiv.org/abs/1508.07909"),
        ("GPT-2: Language Models are Unsupervised Multitask Learners", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
        ("HuggingFace: Summary of the tokenizers", "https://huggingface.co/docs/transformers/en/tokenizer_summary"),
        ("karpathy/minbpe(byte-level BPE 教学实现)", "https://github.com/karpathy/minbpe"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉与动机:为什么模型只认 token 不认字\n\n"
    "你发给大模型一句话,它并不能直接「读」——它的神经网络只能做矩阵运算,"
    "矩阵里装的全是**数字**。把文字变成数字要分两步:\n\n"
    "1. **分词(tokenization)**:把文字切成一个个小片段,每个片段叫一个 **token**;\n"
    "2. **查表映射**:查一张叫**词表(vocabulary)** 的对照表,把每个 token 换成整数 id。\n\n"
    "这个过程像**搭乐高**:文字是一堆积木,每块积木(token)都有一个编号(id),"
    "模型只认得编号,不认得积木本身的样子。\n\n"
    "**为什么不能直接按「单词」切?** 因为自然语言是开放的:总有生僻词、新词、"
    "拼写变体。若词表只收单词,遇到词表外的词就无解;若只收字符,序列又太长。"
    "**子词(subword)** 是中间路线:用高频片段(如「机器」「学习」)当积木,"
    "任何生词都能拆成已知积木的组合。2016 年 Sennrich 等人把数据压缩算法"
    " **BPE(Byte Pair Encoding,字节对编码)** 改造成子词分词法,"
    "[Sennrich et al., 2016](https://arxiv.org/abs/1508.07909),"
    "成为 GPT 系列等主流模型的默认选择。本课就把它逐行实现出来。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式:把「分词」写成数学\n\n"
    "先把概念写严谨,后面代码里的每个变量都能对上号。\n\n"
    "**定义(分词器)** 给定词表 $\\mathcal{V}=\\{t_1,\\dots,t_{|\\mathcal{V}|}\\}$,"
    "分词器是一个函数 $\\tau$,把文本串映射成 token 序列:\n\n"
    "$$ \\tau: \\text{str} \\rightarrow \\mathcal{V}^* ,\\qquad "
    "\\tau(\\text{“机器翻译”}) = [\\text{“机器”}, \\text{“翻”}, \\text{“译”}, \\text{“</w>”}] $$\n\n"
    "**定义(查表映射)** 词表给出 $\\text{id}: \\mathcal{V} \\rightarrow \\{0,1,\\dots,|\\mathcal{V}|-1\\}$ 的**双射**,"
    "把每个 token 编号。于是整条流水线是:\n\n"
    "$$ \\text{text} \\xrightarrow{\\tau} \\text{tokens} \\xrightarrow{\\text{id}(\\cdot)} \\text{ids} $$\n\n"
    "| 符号 | 含义 | 在本课的取值 |\n"
    "|---|---|---|\n"
    "| $|\\mathcal{V}|$ | 词表大小(去重 token 种类数) | 随合并次数增长 |\n"
    "| $\\tau$ | 分词函数 text→tokens | `char_tokenize` / `apply_bpe` |\n"
    "| $\\text{id}(\\cdot)$ | token→整数 双射 | `vocab` 字典 |\n"
    "| $\\text{decode}(\\cdot)$ | 整数→token 的逆映射 | 构造 `inverse_vocab` |\n"
    "| $<unk>$ | 词表外 token 的兜底(unknown) | 映射到最大 id+1 |\n"
    "| $</w>$ | 词尾标记(end-of-word),保留词边界信息 | 每词末尾追加 |\n\n"
    "**关键指标**:给定一段文本,`token 数`决定模型输入序列长度 $S$,"
    "而 $S$ 直接决定注意力计算的规模($O(S^2)$)与 KV Cache 的显存。"
    "所以「同样的语义,用几个 token 表示」是推理成本的直接来源——"
    "这正是第 6 节要算的真实账。"
)

# =====================================================================
# 第 3 节 · 字符级分词
# =====================================================================
nb.md(
    "## 3. 字符级分词:最朴素的约定\n\n"
    "先实现最「无脑」的分词:每个字符(含空格、标点、换行)单独成一个 token。"
    "它不需要训练、永远不会「不认识」任何字符,但代价是:一个英文单词被拆成一串字符,"
    "序列被拉得很长,模型难以学到「单词」概念。\n\n"
    "**每一行都加注释**,把输出是什么、为什么这么做写清楚。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Windows 下 OpenMP 库重复加载报错

def char_tokenize(text):
    """字符级分词:每个字符(含空格/标点/换行)单独成一个 token。
    返回 list[str],长度 = 输入字符数。
    """
    return list(text)                                    # str 拆成字符列表:text -> [字符, 字符, ...]

def build_vocab(token_lists):
    """把若干 token 序列汇总成词表 {token: id},按首次出现顺序从 0 编号。"""
    vocab = {}                                           # 词表字典:token -> id
    for tokens in token_lists:                           # 遍历每条 token 序列
        for token in tokens:                             # 遍历序列里的每个 token
            if token not in vocab:                       # 该 token 尚未收录
                vocab[token] = len(vocab)                # 分配下一个 id(当前词表大小)
    return vocab                                         # 返回 {token: id}

def tokenize_to_ids(text, vocab):
    """文本 -> id 序列;词表外的字符统一落到 <unk> 兜底。"""
    unk = vocab.get("<unk>")                             # 尝试查 <unk> 的 id
    if unk is None:                                      # 词表里没显式放 <unk>
        unk = max(vocab.values()) + 1 if vocab else 0    # 用「最大 id+1」充当兜底 id
    return [vocab.get(ch, unk) for ch in char_tokenize(text)]  # 逐个字符查表,查不到用 unk''',
    "🛠️ **三个基础函数**。`char_tokenize` 产出 token 序列(长度=字符数);"
    "`build_vocab` 产出词表(大小=去重字符数);`tokenize_to_ids` 产出 id 序列。"
    "注意 `tokenize_to_ids` 里 `<unk>` 的兜底逻辑:任何词表外的字符都不会让程序崩溃。",
)

nb.code(
    '''# 用一句话观察字符级分词:中英混排 + 空格 + 标点
text = "LLM 爱学习 deep learning"                        # 一句中英混排文本
tokens = char_tokenize(text)                             # -> ['L','L','M',' ','爱',...] 长度=字符数
print(f"原文      : {text}")
print(f"token 序列 : {tokens}")
print(f"token 数   : {len(tokens)}  (= 字符数 {len(text)})")   # 字符级:1 字符 = 1 token''',
    "✅ **看输出**:空格、字母、汉字被一一拆开,`token 数 = 字符数`。"
    "这就是最细粒度的分词——信息无损,但序列最长。",
)

# =====================================================================
# 第 4 节 · token → id 映射
# =====================================================================
nb.md(
    "## 4. token → id:模型只认数字\n\n"
    "模型内部的矩阵运算只吃整数 id,所以必须有一张词表把 token 换成 id;"
    "反过来生成时要把 id 翻译回文字,需要**逆映射**(inverse vocab)。"
    "词表里还会塞几个特殊 token:`<unk>`(词表外兜底)、`</w>`(词尾标记)。"
)

nb.code(
    '''# 在小语料上建词表,并验证双向映射
corpus_texts = ["你好世界", "深度学习", "hello world"]    # 迷你语料(3 条文本)
vocab = build_vocab([char_tokenize(s) for s in corpus_texts])  # 词表大小 = 去重字符数
print(f"词表大小 |V| = {len(vocab)}")
print("词表内容:", vocab)                                 # token -> id 双射

inverse_vocab = {vid: tok for tok, vid in vocab.items()}  # 构造逆映射 id -> token
print("逆映射(前 5 项):", dict(list(inverse_vocab.items())[:5]))  # 生成时靠它把 id 翻译回文字

ids = tokenize_to_ids("你好", vocab)                      # 词表内的词:正常查表
print(f"「你好」-> id 序列: {ids}")
ids2 = tokenize_to_ids("你好啊", vocab)                   # 含词表外字符「啊」
print(f"「你好啊」-> id 序列: {ids2}  (最后 1 个 id 是 <unk> 兜底)")''',
    "🔢 **观察**:词表内字符正常映射;词表外字符落到兜底 id。"
    "把词表导出成文件,就是模型「字典」的雏形——真实模型词表就是这个字典的巨型版。",
)

# =====================================================================
# 第 5 节 · 最小 BPE 实现
# =====================================================================
nb.md(
    "## 5. 最小 BPE 实现:逐行推演\n\n"
    "字符级分词太碎。**BPE(Byte Pair Encoding,字节对编码)** 让模型自动「学会」"
    "把高频相邻字符合并成一个 token。原论文里的算法是 [Gage, 1994]"
    "(https://en.wikipedia.org/wiki/Byte_pair_encoding) 提出的数据压缩技巧,"
    "Sennrich 等人 [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) 把它改造成子词分词:\n\n"
    "1. **初始化**:每个词拆成字符,词尾加 `</w>`;\n"
    "2. **反复合并**:统计所有相邻 token 对的频次,把**最高频那对**合并成一个新 token;\n"
    "3. **编码**:对新词按学到的合并顺序(先合并的优先)贪心套用。\n\n"
    "下面把 4 个函数逐行实现,每个中间序列都打印长度,顺着读就能看到子词怎么一步步长出来。"
)

nb.code(
    '''# ---- BPE 的三块积木:统计 / 合并 / 训练 ----
from collections import Counter                          # 计数器,统计相邻对频次

def get_pair_counts(token_lists):
    """统计所有 token 序列里相邻二元对的出现次数。返回 Counter{(a,b): 次数}。"""
    pairs = Counter()                                    # 键为 (a,b) 的计数器
    for tokens in token_lists:                           # 遍历每条序列
        for a, b in zip(tokens, tokens[1:]):             # 相邻元素两两成对(a,后一个 b)
            pairs[(a, b)] += 1                           # 该相邻对计数 +1
    return pairs                                         # 例如 {("机","器"): 3, ...}

def merge_pair(tokens, pair):
    """把序列中所有相邻的 pair(a,b) 合并成一个新 token(ab),从左到右、不重叠。"""
    out, i = [], 0                                       # 输出列表 + 扫描指针
    while i < len(tokens):                               # 遍历原序列
        if (i + 1 < len(tokens) and                     # 还有下一个元素
                tokens[i] == pair[0] and tokens[i + 1] == pair[1]):
            out.append(pair[0] + pair[1])                # 命中:合并成新 token
            i += 2                                       # 跳过已被合并的两个
        else:
            out.append(tokens[i])                        # 未命中:原样保留
            i += 1                                       # 前进一个位置
    return out                                           # 返回合并后的序列

def train_bpe(corpus, num_merges):
    """BPE 训练:每词先拆字符、词尾加 </w>,再反复合并最高频相邻对。
    返回 (merges 列表[(a,b),...], 训练后的 vocab 列表)。"""
    vocab = [list(word) + ["</w>"] for word in corpus]   # 每个词 -> [字符,..., "</w>"]
    merges = []                                          # 记录合并顺序(先合并的优先级高)
    for _ in range(num_merges):                          # 执行 num_merges 次合并
        pairs = get_pair_counts(vocab)                   # 统计当前所有相邻对频次
        if not pairs:                                    # 无对可并(语料太小,已被合并完)
            break
        best = max(pairs, key=pairs.get)                 # 频次最高的相邻对
        vocab = [merge_pair(word, best) for word in vocab]  # 对每条序列应用合并
        merges.append(best)                              # 记录这步合并规则
    return merges, vocab                                 # 训练产物:合并规则 + 新词表

def apply_bpe(word, merges):
    """用学到的合并顺序对新词做 BPE 编码(按 merges 顺序贪心套用)。"""
    tokens = list(word) + ["</w>"]                       # 拆字符 + 词尾标记
    for pair in merges:                                  # 依次应用每条合并规则
        tokens = merge_pair(tokens, pair)                # 只要相邻且匹配就合并
    return tokens                                        # 得到子词序列''',
    "🛠️ **四个函数**。`get_pair_counts` 统计频次、`merge_pair` 做一次合并、"
    "`train_bpe` 训练出合并规则本(merges)、`apply_bpe` 用规则编码新词。"
    "注意:训练只输出**合并规则**,不输出固定词表——任何新词都能用它拆解,这就是「开放词表」。",
)

nb.code(
    '''# 在 4 个中文词的迷你语料上训练 BPE,观察子词怎么长出来
corpus = ["机器学习", "机器翻译", "学习效率", "翻译机器"]   # 迷你语料(4 个词)
merges, trained_vocab = train_bpe(corpus, num_merges=4)  # 学 4 条合并规则
print("学到的合并规则(按顺序,越靠前优先级越高):")
for i, (a, b) in enumerate(merges, 1):                   # 逐条打印
    print(f"  第 {i} 次合并: {a} + {b} -> {a + b}")
print()
for word in ["机器翻译", "学习效率"]:                      # 两个验证样本
    pieces = apply_bpe(word, merges)                     # BPE 编码
    print(f"「{word}」切分 -> {pieces}  ({len(pieces)} 个 token)")''',
    "🎯 **你会看到**:“机+器”率先被合并成“机器”(频次最高),"
    "随后“学+习”“翻+译”再合并——子词一步步长出来,"
    "这正是 [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) 论文 Figure 1 的中文版。",
)

# =====================================================================
# 第 6 节 · 数值验证
# =====================================================================
nb.md(
    "## 6. 数值验证:还原一致性 · 压缩率 · 训练一致性\n\n"
    "BPE 是**无损**的:任何词都能由子词序列唯一还原。下面做三个数值实验验证正确性:"
)

nb.code(
    '''# 验证 1:编码-还原往返一致(去掉 </w> 后拼接必须等于原文)
words = ["机器翻译", "学习效率", "翻译机器", "机器学习"]    # 测试样本
ok_all = True                                            # 全部还原成功标志
for w in words:                                          # 逐个测试
    pieces = apply_bpe(w, merges)                        # BPE 编码
    restored = "".join(p for p in pieces if p != "</w>")  # 去掉词尾标记再拼接
    match = restored == w                                # 与原文比较
    ok_all = ok_all and match                            # 汇总结果
    print(f"「{w}」 -> {pieces} -> 还原「{restored}」 {'✅' if match else '❌'}")
print(f"全部无损还原: {ok_all}")                          # 结论:BPE 编码无损

# 验证 2:合并让序列变短(压缩率)
word = "机器翻译"                                        # 测一个词
char_len = len(list(word))                               # 字符级 token 数
bpe_len = len(apply_bpe(word, merges))                   # BPE token 数
print(f"「{word}」:字符级 {char_len} 个 token -> BPE {bpe_len} 个 token (压缩 {char_len/max(bpe_len,1):.2f}×)")

# 验证 3:训练产出的 vocab 与「逐条套用 merges 规则」的结果必须完全一致(内部一致性)
for word in corpus:                                      # 遍历训练语料里的每个词
    expect = apply_bpe(word, merges)                     # 用学到的 merges 手工重新编码
    got = trained_vocab[corpus.index(word)]              # 训练过程直接产出的表示
    ok = expect == got                                   # 两条路径是否一致
    print(f"「{word}」: train 产出 {got} vs apply 产出 {expect} -> {'✅' if ok else '❌'}")
print("提示:在大型语料上,BPE 词表大小 = 基础字符数 + 合并次数(原始字符仍会单独出现);"
      "这里玩具语料因单字被合并后不再单独出现,种类数才会波动。")''',
    "✅ **三个结论**:① BPE 编码-还原往返一致,信息无损;"
    "② 合并让 token 数减少、序列变短;③ 训练直接产出的表示与「回放合并规则」完全一致,"
    "证明训练与编码两条路径用的是同一套逻辑。",
)

# =====================================================================
# 第 7 节 · 真实规模数字
# =====================================================================
nb.md(
    "## 7. 真实规模数字:GPT 的词表与中文的代价\n\n"
    "真实模型的词表远大于玩具示例,而且有一个关键升级——**byte-level BPE**:\n\n"
    "- **GPT-2** 采用 byte-level BPE:基础词表是 256 个 UTF-8 **字节**,"
    "再往上叠加合并出的子词,最终词表大小 **50257**,"
    "见 [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。\n"
    "- 因为基础单位是字节,任意文本(emoji、任何语言)都能表示——**永远不会出现 `<unk>`**;"
    "- OpenAI 的 `tiktoken`(Rust 实现)就是它的工程版;"
    "Karpathy 的 [minbpe](https://github.com/karpathy/minbpe) 是干净的教学版。\n\n"
    "**中文为什么更「贵」?** 英文词被空格天然切分,一个常见单词常常 1 个 token;"
    "中文没有空格,同一个语义往往要拆成更多 token。token 数越多,"
    "输入序列越长,注意力计算与 KV Cache 显存就越大。"
)

nb.code(
    '''# 在更大的中文语料上训练 BPE,量化「合并次数 -> token 长度」的关系
corpus2 = ["人工智能", "机器学习", "人工成本", "智能家居", "语言模型"]  # 5 个词
test_word = "人工智能"                                    # 用来测切分粒度的词
print(f"合并次数 | 「{test_word}」切分结果 | token 数")
for k in [0, 1, 3, 5]:                                   # 不同合并次数
    m, _ = train_bpe(corpus2, num_merges=k)              # 重新训练
    pieces = apply_bpe(test_word, m)                     # 对测试词编码
    shown = "+".join(pieces)                             # 用 + 连接显示切分
    print(f"{k:4d}     | {shown:<28} | {len(pieces)}")

# 量化「中文 vs 英文」的 token 密度(用字符近似,真实模型用词表切)
en_sample = "machine learning is popular"                # 英文句子
zh_sample = "机器学习很受欢迎"                            # 同义中文句子
print(f"\\n英文「{en_sample}」: 字符数 {len(en_sample)}")
print(f"中文「{zh_sample}」: 字符数 {len(zh_sample)}")
print("提示:真实 tokenizer 里英文空格常被拆成独立 token,中文一个汉字常≈1 个 token,"
      "所以同义内容中文的 token 数往往不少于英文。")''',
    "⚠️ **坑**:同一个词在不同词表里 token 数不同。评价分词器时,"
    "常看「平均每个词拆成几个 token」——它直接决定序列长度,进而决定推理成本。",
)

nb.md(
    "**真实配置表**:\n\n"
    "| 模型 | 分词算法 | 词表大小 | 说明 |\n"
    "|---|---|---|---|\n"
    "| GPT-2 / GPT-3 | byte-level BPE | 50257 | 256 字节 + 50001 次合并 |\n"
    "| LLaMA-1/2 | SentencePiece BPE | 32000 | 字节级,无 `<unk>` |\n"
    "| LLaMA-3 | tiktoken BPE | 128256 | 更大词表、更短序列 |\n"
    "| Qwen2 | tiktoken BPE | 151936 | 对中文做了优化 |\n\n"
    "一句话:词表越大,单 token 表达力越强、序列越短,但 embedding 矩阵也越大。"
    "分词器是在「词表大小」与「序列长度」之间做权衡的引擎。"
)

# =====================================================================
# 第 8 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 8. 与 vLLM 的关系:分词器在请求管线中的位置\n\n"
    "vLLM 本身**不实现分词器**,它通过 HuggingFace `transformers` 的 `AutoTokenizer` "
    "加载模型配套的分词器(`vllm/transformers_utils/tokenizer/`)。一条请求的生命周期:\n\n"
    "1. **tokenize**:`TokenizerGroup.tokenize()` 把用户文本 prompt 切成 token id 序列;\n"
    "2. **预处理**:`engine/input_preprocess`(v0)/`v1/engine/input_processor`(v1)"
    " 补上 special tokens、按 `max_model_len` 截断;\n"
    "3. **前向**:模型吃掉 id 序列,输出下一个 token 的 logits;\n"
    "4. **detokenize**:`TokenizerGroup.detokenize()` 把生成的 id 序列翻译回文本流式返回。\n\n"
    "**为什么本课重要**:token 序列长度 $S$ 直接决定三层开销——\n"
    "- 注意力分数矩阵 $O(S^2)$ 的计算量;\n"
    "- KV Cache 的显存占用(每多一个 token 就多存一层 K/V);\n"
    "- 上下文窗口能否容纳(batch 里的请求受 `max_model_len` 约束)。\n\n"
    "所以「同样的 prompt,不同 tokenizer 切成几个 token」是推理系统预算的第一笔账。"
)

# =====================================================================
# 第 9 节 · Streamlit
# =====================================================================
nb.md(
    "## 9. 🖥️ Streamlit 动态演示:亲手验证\n\n"
    "把上面的分词器做成交互 App:输入任意文字、切换字符级/简化 BPE、拖动合并次数,"
    "实时看切分结果与频次柱状图。\n\n"
    "### 📜 App 完整源码(`app_01_token_demo.py`)"
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
    "    print(\"    请把上方源码保存为 app_01_token_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_01_token_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_01_token_demo.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_01_token_demo.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,输入文字观察切分。\n\n"
    "🔍 试试:把合并次数从 0 拉到 20,红色柱(多字符 token)会变多变长。"
)

wrapup(
    nb,
    summary=[
        "token 是词表里的最小单位;分词器 τ 把文字切成 token,词表把 token 映射成 id,模型只吃 id",
        "字符级分词最简单但序列最长;BPE 通过反复合并最高频相邻对,在词表大小与序列长度之间取平衡",
        "BPE 训练产出合并规则(merges)而非固定词表,因此是开放词表,任意新词都能无损编码",
        "特殊 token 承担兜底与边界职责:<unk> 处理词表外字符,</w> 保留词边界",
        "真实模型用 byte-level BPE(GPT-2 词表 50257),永不出现 <unk>;token 数直接决定推理成本",
        "vLLM 通过 HF AutoTokenizer 完成 tokenize→前向→detokenize,分词是请求管线第一步",
    ],
    practice=[
        "给 BPE 增加「合并规则回放」验证:把训练语料里的每个词都编码-还原,确认全部一致",
        "构造 inverse_vocab 并实现 decode(ids)->text,再验证 tokenize(decode(ids))==ids(要求无 <unk> 时成立)",
        "把 corpus 换成英文(带空格),观察 BPE 如何保留空格信息(提示:GPT-2 用 'Ġ' 表示空格)",
        "对比不同 num_merges 下同一句中文的 token 数,画一条「合并次数 → token 数」曲线",
    ],
    links=[
        ("BPE 论文 (arXiv)", "https://arxiv.org/abs/1508.07909"),
        ("GPT-2 论文 (OpenAI)", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
        ("HF: Summary of the tokenizers", "https://huggingface.co/docs/transformers/en/tokenizer_summary"),
        ("karpathy/minbpe", "https://github.com/karpathy/minbpe"),
        ("tiktoken (OpenAI)", "https://github.com/openai/tiktoken"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch01\01_token_and_tokenizer.ipynb")