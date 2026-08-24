# -*- coding: utf-8 -*-
"""ch01 生成公共组件:Notebook 构建 + 各课共享的源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH01 = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch01"
CHAPTER = "第 1 章 · LLM 推理基础"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 字符级分词器
# 第 01 / 02 / 06 课共用:把文本切成 token 列表,并建立 token ↔ id 双向映射。
TOK_CHAR = D('''
def char_tokenize(text):
    """字符级分词:每个字符(含空格、标点)都单独成一个 token。
    这是最简单、最“无脑”的分词方式,却是理解一切分词器的起点。"""
    return list(text)

def build_vocab(token_lists):
    """把若干 token 序列汇总成一个词表:token -> id(从 0 起)。
    按首次出现顺序编号,保证可复现。"""
    vocab = {}
    for tokens in token_lists:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

def tokenize_to_ids(text, vocab):
    """字符级分词 + 查表转 id;遇到词表外的字符统一映射为 <unk>。"""
    unk = vocab.get("<unk>", max(vocab.values()) + 1 if vocab else 0)
    return [vocab.get(c, unk) for c in text]
''')

# ---------------------------------------------------------------- 简化 BPE
# 第 01 课核心:BPE 的“合并最频繁相邻对”训练循环,麻雀虽小五脏俱全。
TOK_BPE = D('''
from collections import Counter

def get_pair_counts(token_lists):
    """统计所有 token 序列里相邻 token 对的出现次数。"""
    pairs = Counter()
    for tokens in token_lists:
        for a, b in zip(tokens, tokens[1:]):
            pairs[(a, b)] += 1
    return pairs

def merge_pair(tokens, pair):
    """把序列里所有相邻的 pair 合并成一个新 token(从左到右,非重叠)。"""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def train_bpe(corpus, num_merges):
    """简化 BPE 训练:每个词先拆成字符(词尾加 </w> 标记),
    然后反复合并出现次数最多的相邻对,记录合并顺序(merges)。"""
    vocab = [list(w) + ["</w>"] for w in corpus]
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)   # 出现最多的相邻对
        vocab = [merge_pair(w, best) for w in vocab]
        merges.append(best)
    return merges, vocab

def apply_bpe(word, merges):
    """用学到的合并顺序对新词做 BPE 编码(贪心按 merges 顺序合并)。"""
    tokens = list(word) + ["</w>"]
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens
''')

# ---------------------------------------------------------------- n-gram 玩具语言模型
# 第 02 / 06 课共用:统计"看到什么上下文,接着说什么",构成玩具版条件概率 P(next|context)。
NGRAM = D('''
from collections import defaultdict, Counter

def build_ngram(corpus, n):
    """n=2 bigram、n=3 trigram... 统计 (前 n-1 个 token) -> {下一个 token: 次数}。
    这是玩具版的条件概率: P(下一个 | 最近 n-1 个 token)。"""
    model = defaultdict(Counter)
    for text in corpus:
        toks = list(text)
        for i in range(len(toks) - n + 1):
            ctx = tuple(toks[i:i + n - 1])
            model[ctx][toks[i + n - 1]] += 1
    return model

def next_candidates(model, context, n):
    """给定上下文, 返回候选 token、次数与归一化概率(按次数降序)。
    上下文不足 n-1 个时退化为只看已有的那几个。"""
    ctx = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)
    counts = model.get(ctx, Counter())
    total = sum(counts.values())
    if total == 0:
        return [], []
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    toks = [t for t, _ in items]
    probs = [c / total for _, c in items]
    return toks, probs

def generate_ngram(model, n, start, steps, temperature=1.0, top_k=0, top_p=1.0, seed=0):
    """自回归生成: 每一步把刚生成的 token 追加进上下文, 再预测下一个。
    返回 (生成的完整 token 列表, 逐步路径)。"""
    rng = np.random.default_rng(seed)
    context = list(start)
    out = list(start)
    path = []
    for _ in range(steps):
        toks, probs = next_candidates(model, context, n)
        if not toks:
            break
        logits = np.log(np.array(probs, dtype=np.float64) + 1e-12)
        idx, filtered, mask = sample_from(logits, temperature, top_k, top_p, rng)
        chosen = toks[idx]
        out.append(chosen)
        path.append((context.copy(), toks, probs, chosen, filtered, mask))
        context.append(chosen)
    return out, path
''')

# ---------------------------------------------------------------- 采样策略
# 第 03 / 06 课共用: 温度、top-k、top-p 过滤 + 多项式采样, 一条完整采样管线。
SAMPLING = D('''
import numpy as np

def softmax(logits, temperature=1.0):
    """带温度的 softmax: 先把 logits 除以温度, 再归一化成概率。
    温度>1 变平(更随机), 温度<1 变尖(更确定)。"""
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()

def top_k_mask(logits, k):
    """保留 logits 最大的 k 个候选, 其余概率被置零。"""
    mask = np.zeros_like(logits, dtype=bool)
    if k <= 0 or k >= len(logits):
        mask[:] = True
    else:
        mask[np.argsort(logits)[-k:]] = True
    return mask

def top_p_mask(logits, p):
    """从概率最大的 token 往下逐个累积, 直到累计概率超过 p, 其余置零。"""
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = cum - probs[order] <= p
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    return mask

def sample_from(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    """完整采样管线: 温度 -> top-k -> top-p -> 归一化 -> 多项式采样。
    返回 (选中下标, 最终概率分布, 过滤掩码)。"""
    if rng is None:
        rng = np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    probs = softmax(logits, temperature)
    mask = np.ones_like(probs, dtype=bool)
    if top_k > 0:
        mask &= top_k_mask(logits, top_k)
    if top_p < 1.0:
        mask &= top_p_mask(logits, top_p)
    filtered = probs * mask
    filtered = filtered / filtered.sum()
    idx = rng.choice(len(filtered), p=filtered)
    return idx, filtered, mask
''')
