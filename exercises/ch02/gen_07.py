# -*- coding: utf-8 -*-
"""生成第 07 课 notebook: KV Cache 原理(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:注意力公式复习 -> 维度符号表 -> 手工张量逐行推演 -> 缓存实现 -> 复杂度推导 -> GPU 实测 -> 工程关联
2. 每一行代码都有 inline 注释
3. 每个中间张量打印 shape 并标注维度含义,脉络清晰
4. 论文支撑:PagedAttention (arXiv:2309.06180)、GQA (arXiv:2305.13245)、KV 内存公式
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_07_kv_principle.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 07 课 · KV Cache 原理:把 O(T²) 的重算降成 O(T) 的追加写",
    subtitle="注意力维度分析 · 无缓存 vs 有缓存逐行推演 · 复杂度公式 · 真实 GPU 实测",
    emoji="🔑", chapter="第 2 章 · KV Cache 与 PagedAttention",
)

chapter_cover(
    nb,
    objectives=[
        "复习缩放点积注意力公式,并建立统一的张量维度符号表(batch/head/seq/dim)",
        "用手工构造的小张量逐行推演:无缓存时每步要重算哪些 K/V,有缓存时只算哪些",
        "亲手实现一个最小 KV Cache(纯 PyTorch),打印每一步中间张量的 shape 并核对维度含义",
        "从逐行实现反推出 FLOPs 公式,验证无缓存 O(T²) vs 有缓存 O(T) 的复杂度差",
        "在真实 GPU 上实测两种方式耗时,理解 compute-bound 与 memory-bound 的工程真相",
        "建立与 PagedAttention 论文(arXiv:2309.06180)和 vLLM 实现的联系",
    ],
    toc=[
        ("注意力公式复习", "从 Attention(Q,K,V) 出发,建立维度符号表"),
        ("维度符号表", "B/H/S/D 四个维度每个代表什么,贯穿全课"),
        ("手工张量逐行推演", "batch=2, seq=4, heads=2, dim=8,一步步打印 shape"),
        ("自回归的两阶段", "prefill 并行算 vs decode 逐个算,K/V 谁在变、谁不变"),
        ("最小 KV Cache 实现", "纯 PyTorch 写 forward,逐行注释+shape 核对"),
        ("复杂度推导", "从实现反推 FLOPs,验证 O(T²)→O(T);全量加速比 140×/162×,上界 (T+1)/2"),
        ("真实 GPU 实测", "无缓存 vs 有缓存毫秒级耗时,compute/memory-bound 分析"),
        ("与 vLLM 的关系", "GPUBlockPool / PagedAttention 怎么用这个缓存"),
        ("Streamlit 动态演示", "拖动滑杆验证结论"),
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("GQA: Multi-Query Transformer (arXiv)", "https://arxiv.org/abs/2305.13245"),
        ("TensorTonic: KV Cache in LLMs Explained", "https://www.tensortonic.com/llm-internals/kv-cache"),
        ("vLLM 官方博客: 10x faster inference", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ],
)

# =====================================================================
# 第 1 节 · 注意力公式复习
# =====================================================================
nb.md(
    "## 1. 注意力公式复习:先把每个符号搞清楚\n\n"
    "KV Cache 的一切都围绕注意力展开。先复习标准**缩放点积注意力**(Scaled Dot-Product Attention):\n\n"
    "$$ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^{\\top}}{\\sqrt{d_k}}\\right) V $$\n\n"
    "这个公式只有四个字母 $Q, K, V, d_k$,但它们在大模型里的真实形状是什么?**"
    "后面所有代码都要打印这些 shape,所以必须先建立一个符号表**:\n\n"
    "| 符号 | 含义 | 形状 |\n"
    "|---|---|---|\n"
    "| $B$ | batch size(一次处理几个序列) | — |\n"
    "| $H$ | 注意力头数 num_heads | — |\n"
    "| $S$ | 序列长度 seq_len(一个序列有几个 token) | — |\n"
    "| $D$ | 每头维度 head_dim(头向量有多长) | — |\n"
    "| $Q$ | 查询矩阵(query) | $(B, H, S, D)$ |\n"
    "| $K$ | 键矩阵(key),被查询的目标 | $(B, H, S, D)$ |\n"
    "| $V$ | 值矩阵(value),最终要加权求和的内容 | $(B, H, S, D)$ |\n"
    "| $d_k$ | 缩放因子,就是 $D$ | $D$ |\n\n"
    "> ⚠️ 注意:工程实现里通常用 `[B, H, S, D]` 的 4D 布局(每头一段),而不是论文里的 3D 写法。"
    "PyTorch 的 `torch.nn.functional.scaled_dot_product_attention` 用的就是 `[B, H, S, D]`。\n\n"
    "**关键洞察(本课核心)** 自回归生成时,序列在**不断变长**:\n"
    "- 第 1 步生成后序列长 $S{=}1$,第 2 步 $S{=}2$,…,第 $T$ 步 $S{=}T$;\n"
    "- 那么 $K$ 的形状是 $(B, H, T, D)$,$V$ 的形状也是 $(B, H, T, D)$ —— **随 $T$ 增长**;\n"
    "- 但 $Q$ 永远只关心**最新那个位置**,形状始终是 $(B, H, 1, D)$。\n\n"
    "这就是 KV Cache 的出发点:**K 和 V 会越攒越多,而 Q 永远只有一行。**"
)

# =====================================================================
# 第 2 节 · 手工张量逐行推演
# =====================================================================
nb.md(
    "## 2. 手工构造小张量:亲手看维度变化\n\n"
    "先造一个迷你场景,把上面的符号变成真实的数字。我们取:\n\n"
    "```\n"
    "batch B = 2      (同时处理 2 条对话)\n"
    "num_heads H = 2  (2 个注意力头)\n"
    "head_dim D = 8   (每头 8 维,方便心算)\n"
    "seq_len S = 4    (当前序列已有 4 个 token)\n"
    "```\n\n"
    "**每一行都加了注释,把 shape 和维度含义直接写出来**,这样顺着读就能建立直觉。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch

# --------------------------------------------------------------------------
# 1. 定义迷你模型的超参数(刻意取小值,让每个数字都能心算验证)
# --------------------------------------------------------------------------
batch_size = 2          # B: 同时处理的序列条数(2 条对话)
num_heads = 2            # H: 注意力头数(每头独立关注不同的子空间)
head_dim = 8             # D: 每个注意力头的向量维度(向量越长,表达能力越强)
seq_len = 4              # S: 当前序列长度(已有 4 个 token)

# --------------------------------------------------------------------------
# 2. 模拟"某一步"算出的 Q、K、V(真实模型里由 Linear 投影得到)
#    torch.randn 生成标准正态分布,这里只关心 shape,不关心具体值
# --------------------------------------------------------------------------
q = torch.randn(batch_size, num_heads, 1, head_dim)      # Q: 只有"最新"1 个 token 的 query
k = torch.randn(batch_size, num_heads, seq_len, head_dim) # K: 全部 seq_len 个 token 的 key
v = torch.randn(batch_size, num_heads, seq_len, head_dim) # V: 全部 seq_len 个 token 的 value

# 打印 shape,并把每个维度的含义标注清楚
print(f"Q shape = {tuple(q.shape)}  <- (batch={batch_size}, heads={num_heads}, 当前token=1, dim={head_dim})")
print(f"K shape = {tuple(k.shape)}  <- (batch={batch_size}, heads={num_heads}, 全部token={seq_len}, dim={head_dim})")
print(f"V shape = {tuple(v.shape)}  <- (batch={batch_size}, heads={num_heads}, 全部token={seq_len}, dim={head_dim})")

# --------------------------------------------------------------------------
# 3. 手写缩放点积注意力(对照公式 Attention = softmax(QK^T / sqrt(d)) V)
# --------------------------------------------------------------------------
# 3.1 打分:Q 与 K 做点积 —— 新 token 对每个历史 token 有多相关
#     Q: (B, H, 1, D), K: (B, H, S, D)
#     K.transpose(-1, -2) 把最后两维(D, S)对调,变 (B, H, D, S)
#     matmul 后得到 (B, H, 1, S):每个 query 对每个 key 的注意力分数
scores = torch.matmul(q, k.transpose(-1, -2))            # (B, H, 1, S) = (2, 2, 1, 4)
print(f"\\nscores shape = {tuple(scores.shape)}  <- 打分矩阵:1 个 query × {seq_len} 个 key")

# 3.2 缩放:除以 sqrt(head_dim),防止点积过大导致 softmax 梯度消失(来源:原版 Transformer 论文)
scaled = scores / (head_dim ** 0.5)                      # 除 sqrt(D)=sqrt(8)≈2.83,形状不变

# 3.3 归一化:对最后一个维度(所有 key)做 softmax,让分数和为 1,成为"注意力权重"
weights = torch.softmax(scaled, dim=-1)                  # (B, H, 1, S) = (2, 2, 1, 4)
print(f"weights shape = {tuple(weights.shape)}  <- softmax 后的权重,每行和为 1")

# 3.4 加权求和:用权重对 V 加权 —— 输出是把"相关历史"的信息汇总到新 token
#     weights: (B, H, 1, S)  ×  V: (B, H, S, D)  ->  (B, H, 1, D)
out = torch.matmul(weights, v)                           # (B, H, 1, D) = (2, 2, 1, 8)
print(f"out    shape = {tuple(out.shape)}    <- 新 token 融合历史后的输出,形状与 Q 相同")

# 验证数值正确性:用 PyTorch 官方实现对照(结果应几乎一致)
import torch.nn.functional as F
ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
print(f"与官方实现最大误差 = {(out - ref).abs().max().item():.2e}  (≈0 说明手写正确)")''',
    "✅ **逐行推演**。请跟着注释读一遍:每个张量怎么变形、每步 shape 怎么来的。"
    "注意 `out` 的形状回到了 $(B,H,1,D)$ —— 这正是下一步**为什么能缓存**的关键。",
)

# =====================================================================
# 第 3 节 · 自回归两阶段
# =====================================================================
nb.md(
    "## 3. 自回归解码:prefill 与 decode 的维度视角\n\n"
    "LLM 生成文本是**自回归**(autoregressive)的:每次只生成 1 个 token,把新 token 拼到序列尾巴,\n"
    "再拿**整个变长序列**跑一遍前向,预测下一个。于是推理天然分成两阶段:\n\n"
    "| 阶段 | 输入序列长度 | K/V 形状 | 计算特点 |\n"
    "|---|---|---|---|\n"
    "| **prefill(预填充)** | 整个 prompt,如 $S{=}2048$ | 一次性算出全部 | 高度并行,compute-bound |\n"
    "| **decode(解码)** | 每步只 +1,如 $S{=}2048, 2049, ...$ | 每步只算新 1 个 | 串行,memory-bound |\n\n"
    "关键问题在 decode:**每步 $K$ 都会变长**。\n\n"
    "### 🔍 无缓存(naive):每步重算全部历史\n\n"
    "第 $t$ 步,序列长 $t$。朴素做法是把 $t$ 个 token 的 $Q,K,V$ **全部重算一遍**,再做注意力。\n"
    "可前一步明明已经算过前 $t{-}1$ 个 token 的 $K,V$ 了——**重算就是浪费**。\n\n"
    "### 💡 关键观察:K 和 V 一旦算出就不变\n\n"
    "第 $j$ 个 token 的 key $k_j$ 只依赖它自己的 embedding 和模型权重,与「后面来了谁」无关:\n\n"
    "$$ k_j = \\text{Linear}_K(\\text{emb}(x_j)), \\qquad v_j = \\text{Linear}_V(\\text{emb}(x_j)) $$\n\n"
    "所以第 1 步算出的 $k_0, k_1$ 到第 100 步依然是同样的 $k_0, k_1$。\n"
    "**把它们存起来,decode 每步只算新 token 的 $k_{t}, v_{t}$ 再追加进去** —— 这就是 KV Cache。\n\n"
    "> 📄 这一洞察是 KV cache 一切优化的起点,见 TensorTonic 教程"
    "「*exactly which vectors get cached and why they are safe to cache*」。"
)

# =====================================================================
# 第 4 节 · 最小 KV Cache 实现
# =====================================================================
nb.md(
    "## 4. 最小 KV Cache 实现:每行代码都注释\n\n"
    "现在用一个玩具模型动手实现。做法:实现一个**单个注意力层**的 forward,\n"
    "它接受 `(k_cache, v_cache)` 并返回更新后的缓存 —— 这正是 vLLM 里 attention backend 每步做的三件事:\n"
    "**新 token 的 QKV 投影 → 写入缓存 → 与缓存中的历史 KV 做注意力**。\n\n"
    "我们故意不用 `torch.cat` 重拼(那是 O(S²) 的,08 课会专门讲),而是用**预分配 + 原位写入**的思路。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# 定义一个极简的单层注意力,展示 KV Cache 的写入与读取
# --------------------------------------------------------------------------
class AttentionWithCache(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        # 调用父类 nn.Module 的初始化(必须)
        super().__init__()
        # 保存超参供 forward 使用
        self.num_heads = num_heads            # H: 注意力头数
        self.head_dim = head_dim              # D: 每头维度
        # Q/K/V 三个投影矩阵:把隐藏向量 hidden_size 投影成 num_heads*head_dim 维
        # 注意:这里的 w_q 把 (B, S, hidden) -> (B, S, num_heads*head_dim),再 reshape 成多头
        self.w_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)  # Q 投影
        self.w_k = nn.Linear(hidden_size, num_heads * head_dim, bias=False)  # K 投影
        self.w_v = nn.Linear(hidden_size, num_heads * head_dim, bias=False)  # V 投影

    def forward(self, x, k_cache, v_cache, cache_len):
        # x: (B, new_tokens, hidden) 当前步新增的 token(可能是 1 个,也可能是整个 prompt)
        # k_cache / v_cache: 已预分配的缓存,shape (B, H, max_seq, D)
        # cache_len: 当前已缓存了多少个 token(历史长度)

        batch, new_tokens, _ = x.shape                     # 解出 B 和新增 token 数

        # ---- 第 1 步:新 token 的 QKV 投影(每行注释)----
        q = self.w_q(x)                                    # (B, new, H*D) 查询投影
        k = self.w_k(x)                                    # (B, new, H*D) 键投影
        v = self.w_v(x)                                    # (B, new, H*D) 值投影

        # reshape 成多头布局:(B, new, H*D) -> (B, new, H, D) -> 转置为 (B, H, new, D)
        q = q.view(batch, new_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, new_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, new_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- 第 2 步:把新 K/V 原位写进缓存 ----
        # 写入区间 [cache_len, cache_len+new_tokens),这是"追加写"而非"重算"
        k_cache[:, :, cache_len:cache_len + new_tokens, :] = k   # K 写进缓存对应位置
        v_cache[:, :, cache_len:cache_len + new_tokens, :] = v   # V 写进缓存对应位置

        # ---- 第 3 步:注意力:新 Q 与缓存里全部历史 K/V 交互 ----
        # 取缓存中真正有效的部分(前 cache_len+new_tokens 个位置)
        k_all = k_cache[:, :, :cache_len + new_tokens, :]        # (B, H, S', D) S'=已缓存长度
        v_all = v_cache[:, :, :cache_len + new_tokens, :]        # (B, H, S', D)

        # 打分:Q(B,H,new,D) × K^T(B,H,D,S') -> (B,H,new,S')
        scores = torch.matmul(q, k_all.transpose(-1, -2)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)                  # (B,H,new,S') 每行和=1
        out = torch.matmul(weights, v_all)                       # (B,H,new,D) 加权汇总
        return out.transpose(1, 2)                               # 回到 (B, new, H, D)


# --------------------------------------------------------------------------
# 演练:prefill(一次性处理整个 prompt)+ decode(每步 1 个 token)
# --------------------------------------------------------------------------
hidden_size = 16            # 隐藏层维度(embedding 长度)
num_heads = 2               # H = 2
head_dim = 8                # D = 8
max_seq = 8                 # 缓存最大容量(预分配,不随 token 增长)
batch_size = 1              # B = 1,先跑单序列便于观察

model = AttentionWithCache(hidden_size, num_heads, head_dim)

# 预分配 KV 缓存:一次性申请 (B, H, max_seq, D),避免反复 resize
k_cache = torch.zeros(batch_size, num_heads, max_seq, head_dim)  # K 缓存缓冲
v_cache = torch.zeros(batch_size, num_heads, max_seq, head_dim)  # V 缓存缓冲
print(f"预分配 K 缓存 shape = {tuple(k_cache.shape)} <- (batch={batch_size}, heads={num_heads}, max_seq={max_seq}, dim={head_dim})")

cache_len = 0   # 当前缓存长度初始为 0

# ---- PREFILL:一次喂整个 prompt(4 个 token)----
prompt = torch.randn(batch_size, 4, hidden_size)   # (B=1, S=4, hidden=16) 模拟 prompt 的 4 个 embedding
out = model(prompt, k_cache, v_cache, cache_len)               # 前向,内部会把 4 个 K/V 写入缓存
cache_len += 4                                                 # 缓存长度更新为 4
print(f"prefill 后:输出 shape = {tuple(out.shape)}, 缓存长度 cache_len = {cache_len}")
print(f"           此时 K 缓存前 {cache_len} 个位置已有内容,其余仍为 0(预分配的空位)")

# ---- DECODE:每步只喂 1 个新 token ----
for step in range(1, 4):                                        # 再解码 3 步
    new_tok = torch.randn(batch_size, 1, hidden_size)           # (B, 1, hidden) 仅 1 个新 token
    out = model(new_tok, k_cache, v_cache, cache_len)           # 前向:Q 只算新的,K/V 追加写
    cache_len += 1                                              # 缓存 +1
    print(f"decode 第 {step} 步:输出 shape = {tuple(out.shape)}, 缓存长度 = {cache_len}")

print(f"\\n最终缓存长度 = {cache_len},K/V 每步只追加,从未重算 —— 这就是 KV Cache。")''',
    "🛠️ **最小实现**。重点观察两件事:\n"
    "1. `k_cache[:, :, cache_len:cache_len+new_tokens, :] = k`:每次是**切片写入**(追加),不是重新计算;\n"
    "2. `cache_len` 每次递增,`K` 缓存第 `cache_len` 位之前永远不变。"
    "这正对应 vLLM attention backend 每步的「新 QKV → 写缓存 → 注意力」三件事。",
)

# =====================================================================
# 第 5 节 · 复杂度推导
# =====================================================================
nb.md(
    "## 5. 复杂度推导:为什么是 O(T²) vs O(T)\n\n"
    "设模型 $L$ 层,每层 $H$ 头,每头 $D$ 维;已经生成了 $T$ 个 token。\n\n"
    "**每 token 每层的 QKV 投影 FLOPs**(3 个矩阵乘,每个 $2 \\cdot hidden \\cdot H D$):\n\n"
    "$$ \\text{QKV} = 6 L H D^2 $$\n\n"
    "**第 $t$ 步注意力 FLOPs**:新 query($1$ 个)与 $t$ 个历史 key 打分($2HDt$)再与 $t$ 个 value 加权($2HDt$):\n\n"
    "$$ \\text{attn}(t) = 4 L H D t $$\n\n"
    "### 🔢 无缓存:每步重算全部历史\n\n"
    "第 $t$ 步要重算 $t$ 个 token 的 QKV($\\times t$),注意力照常:\n"
    "$$ \\text{FLOPs}_{\\text{no-cache}} = \\sum_{t=1}^{T} (6LHD^2 \\cdot t + 4LHDt) = O(T^2) $$\n\n"
    "### 🔢 有缓存:每步只算新 token\n\n"
    "第 $t$ 步只算 1 个新 token 的 QKV(常数),注意力照常($4LHDt$ 是硬刚需,两种方式一样):\n"
    "$$ \\text{FLOPs}_{\\text{cache}} = \\sum_{t=1}^{T} (6LHD^2 + 4LHDt) = O(T) $$\n\n"
    "### ⚖️ 加速比\n\n"
    "差距全在 QKV 投影项:无缓存把它乘了 $t$,有缓存是常数。于是,**仅 QKV 投影部分**的理论上界为:\n\n"
    "$$ \\text{speedup}_{\\text{QKV}} \\approx \\frac{6LHD^2 \\cdot T(T+1)/2}{6LHD^2 \\cdot T} \\approx \\frac{T+1}{2} $$\n\n"
    "`T=1024` 时这个上界约 $\\approx 512$×,`T=2048` 时约 $\\approx 1024$×。\n"
    "**但这只是『仅 QKV 部分』的上界**——注意力打分($4LHDt$ 一项)两种方式都要做、省不掉。"
    "把它计入后的**全量实际加速比**要小得多:`T=1024` 约 **140×**,`T=2048` 约 **162×**(下一步代码会打印验证)。\n"
    "> ⚠️ 注意:以上仍是**计算量(FLOPs)视角**的理论值。真实 GPU 上 decode 是 memory-bound,"
    "实测加速比往往更小 —— 第 6 节实测见分晓。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np

def total_flops(L: int, H: int, D: int, T: int, use_cache: bool) -> int:
    # 计算生成 T 个 token 的总 FLOPs
    # qkv: 每 token 每层的 QKV 投影 FLOPs = 6*L*H*D^2
    qkv = 6 * L * H * D * D                       # 常数项:1 个 token 的 QKV
    t = np.arange(1, T + 1)                       # 1..T 每个解码步的序列长度
    if use_cache:
        # 有缓存:每步 QKV 恒定(只算新 token),注意力 4*L*H*D*t 随 t 线性增长
        per_step = qkv + 4 * L * H * D * t
    else:
        # 无缓存:每步 QKV 也乘 t(重算全部历史),注意力相同
        per_step = qkv * t + 4 * L * H * D * t
    return int(per_step.sum())                    # 返回总 FLOPs(int)

# 用 LLaMA-7B 的真实规模验证:L=32 层, H=32 头, D=128 维, 上下文 T=1024
L, H, D, T = 32, 32, 128, 1024
no_cache = total_flops(L, H, D, T, use_cache=False)   # 无缓存总 FLOPs
use_cache = total_flops(L, H, D, T, use_cache=True)   # 有缓存总 FLOPs
print(f"无缓存总 FLOPs = {no_cache:.3e}")
print(f"有缓存总 FLOPs = {use_cache:.3e}")
print(f"全量加速比(含注意力) = {no_cache / use_cache:.1f}×")
print(f"仅 QKV 部分的理论上界 (T+1)/2 = {(T + 1) / 2:.0f}×  (不含注意力,故大于全量值)")

# 额外验证:全量加速比只取决于 T,与模型规模(L,H)无关;上界 (T+1)/2 恒更高
for Lm, Hm in [(32, 32), (80, 64), (96, 96)]:            # LLaMA-7B / 70B / GPT-3 175B
    na = total_flops(Lm, Hm, 128, 2048, use_cache=False)
    ca = total_flops(Lm, Hm, 128, 2048, use_cache=True)
    print(f"L={Lm:<3} H={Hm:<3} 全量加速比 = {na / ca:6.1f}×  (T=2048;仅QKV上界≈{(2048+1)/2:.0f}×)")''',
    "✅ **数值验证**。把上节的公式直接写成函数,代入真实模型规模:**全量(含注意力)加速比约 140×(T=1024)、"
    "162×(T=2048)**,都明显小于仅 QKV 的上界 $(T{+}1)/2$;且全量值只由序列长度 $T$ 决定,"
    "与模型大小无关——KV Cache 的收益是普适的。",
)

# =====================================================================
# 第 6 节 · 真实 GPU 实测
# =====================================================================
nb.md(
    "## 6. 真实 GPU 实测:纸上公式 vs 硬件真相\n\n"
    "前面的复杂度推导是**计算量**视角(FLOPs)。但真实 GPU 上还有第二个维度:**内存带宽**。\n"
    "decode 每步只算 1 个新 token,却要把整个历史的 K/V 从 HBM 读一遍——\n"
    "**读得比算得多**,算子被带宽卡住,GPU 算力吃不满。这就是为什么即便全量理论加速比(140×/162×)"
    "也常常达不到,实测往往只有几十倍。\n\n"
    "下面复用 `real_ops.bench_qkv_attn`:在 GPU 上真实跑注意力算子,分别测「无缓存(重算全部历史的 QKV+注意力)」\n"
    "和「有缓存(只算新 query 的注意力)」的毫秒耗时。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os
# 设定环境变量避免 OpenMP 冲突(Windows 上 torch 常见问题)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 把 ch02 目录加入模块搜索路径,好导入 real_ops
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises\\ch02")
from real_ops import bench_qkv_attn                       # 真实 GPU 微基准

# 用 LLaMA 的注意力配置:H=32 头, D=128 维, 历史长度从 32 测到 512
r07 = bench_qkv_attn(H=32, D=128, T_list=(32, 64, 128, 256, 512), dtype="bf16")
print("设备:", r07["device"], "· 是否仿真(simulated) =", r07["simulated"])
print(f"\\n{'历史T':>6} | {'无缓存(重算全部, ms)':>22} | {'有缓存(只算新, ms)':>20} | {'加速比':>8}")
for t, nc, c in r07["results"]:                          # 遍历每个历史长度的结果
    print(f"{t:6d} | {nc:22.3f} | {c:20.3f} | {nc / max(c, 1e-12):8.1f}x")

# 结论:真实加速比远小于 (T+1)/2 的公式值,原因是 decode 是 memory-bound(读 K/V 比算 Q 贵)''',
    "🚀 **真实 GPU 数字**。重点:无缓存/有缓存的实测加速比随 $T$ 上升,但**远小于公式的 $(T{+}1)/2$**。\n"
    "原因正是 memory-bound:decode 阶段带宽瓶颈压制了 QKV 那部分省下的 FLOPs。",
)

nb.md(
    "### 🔬 为什么实测比公式小?compute-bound vs memory-bound\n\n"
    "看每步的数据流:有缓存时,第 $t$ 步要做\n\n"
    "```\n"
    "新 token 的 QKV 投影  (小,算得快)\n"
    "+ 读全部历史 K/V       (大,从 HBM 读 (B,H,t,D)×2 个元素)\n"
    "+ 与全部历史做注意力   (读得越多越慢)\n"
    "```\n\n"
    "当 $t$ 很大时,**读 K/V 的带宽成本**成为主导,GPU 算力(FLOPs)根本不是瓶颈。\n"
    "所以 KV Cache 的真实价值不只是「省 FLOPs」,更是:**把不可并行的 decode 变成轻量 append,\n"
    "并避免反复搬运历史 KV**。这正是 PagedAttention 论文(SOSP'23)要解决的核心矛盾——\n"
    "缓存本身的**内存管理**问题(碎片、预分配浪费),我们 08、10 课展开。"
)

# =====================================================================
# 第 7 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 7. 与 vLLM 的关系:缓存从哪里来、到哪里去\n\n"
    "vLLM 把「存 K/V」这件事做成了工业级。关键路径:\n\n"
    "1. **预分配物理块**:`v1/core/block_pool.py` 的 `GPUBlockPool` 启动时按显存预算一次划出\n"
    "   成百上千个**固定大小**的物理块(默认 16 token/块);\n"
    "2. **页表映射**:`v1/core/kv_cache_utils.py` 维护 逻辑块→物理块 的映射与读写下标;\n"
    "3. **调度器分配**:`sched/scheduler.py` 为每个请求逐 token 分配缓存块;\n"
    "4. **attention backend**:每步只做「新 QKV → 写缓存 → 与缓存做注意力」三件事。\n\n"
    "这就是本课第 4 节最小实现的**工程放大版**。区别在于:\n"
    "- 我们是 1 个预分配 buffer;vLLM 是**分页的**块池(借鉴 OS 虚拟内存);\n"
    "- 我们用 `cache_len` 记录位置;vLLM 用 **block table**(逻辑块→物理块)记录;\n"
    "- 我们每步 append;vLLM 按块分配,内部碎片小、可跨请求共享(前缀复用)。\n\n"
    "> 📄 详见 **PagedAttention (SOSP'23)** arXiv:2309.06180 第 4.2 节「KV Cache Manager」。\n\n"
    "### 🔗 和下一课(08)的衔接\n\n"
    "本课证明了 KV Cache 省计算。但缓存本身要**吃显存**,而且吃得很凶:\n\n"
    "$$ \\text{KV bytes} = 2 \\times \\text{batch} \\times \\text{seq} \\times \\text{layers} \\times \\text{kv\\_heads} \\times \\text{head\\_dim} \\times \\text{dtype\\_bytes} $$\n\n"
    "L=32, kv_heads=8(GQA), D=128, seq=32K, batch=1, bf16:\n\n"
    "$$ 2 \\times 1 \\times 32768 \\times 32 \\times 8 \\times 128 \\times 2 \\approx 4.3\\,\\text{GB} $$\n\n"
    "一个请求就 4GB 显存——这就是 08 课要算的「显存账本」,也是 GQA(MQA)能把缓存缩小 $\\frac{H}{\\text{kv\\_heads}}$ 倍的动机。"
)

# =====================================================================
# 第 8 节 · Streamlit
# =====================================================================
nb.md(
    "## 8. 🖥️ Streamlit 动态演示:亲手验证\n\n"
    "把上面的公式做成交互 App:拖动**层数 / 头数 / 序列长度**滑杆,实时计算两种方式的总 FLOPs 并对比。\n\n"
    "### 📜 App 完整源码(`app_07_kv_principle.py`)"
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
    "    print(\"    请把上方源码保存为 app_07_kv_principle.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_07_kv_principle.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_07_kv_principle.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_07_kv_principle.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,拖动滑杆观察 FLOPs 变化。\n\n"
    "🔍 试试:把序列长度拉到 4096,全量加速比接近 180×;把层数拉到 200,两条线「剪刀差」更大。"
)

wrapup(
    nb,
    summary=[
        "注意力中 K、V 只依赖各自 token 的 embedding,一旦算出就固定不变——这是能缓存的前提",
        "无缓存每步重算全部历史 K/V,总计算量 O(T²);有缓存每步只算新 token 并追加写入,总计算量 O(T)",
        "加速比 ≈ (T+1)/2(仅 QKV 部分),与模型规模无关,只取决于序列长度",
        "真实 GPU 上 decode 是 memory-bound,实测加速比远小于公式——KV Cache 更核心的价值是避免反复搬运历史 KV",
        "vLLM 用 GPUBlockPool 管理分页物理块 + block table 记录逻辑到物理的映射,是本课最小实现的工程放大版",
    ],
    practice=[
        "把 AttentionWithCache 改成支持 GQA:共享 KV 头(把 k 的 H 从 2 变 1,再用 repeat 扩回),看缓存体积变化",
        "在 forward 里手动打印每次写入缓存前后的 k_cache[:,0,0,:,:],验证「追加写」确实只改了新增位置",
        "把 decode 循环改成每步打印 K 缓存第 cache_len 列的值,确认历史位置从未被覆盖",
    ],
    links=[
        ("PagedAttention 论文", "https://arxiv.org/abs/2309.06180"),
        ("GQA 论文", "https://arxiv.org/abs/2305.13245"),
        ("vLLM 博客: Continuous Batching", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("TensorTonic: KV Cache Explained", "https://www.tensortonic.com/llm-internals/kv-cache"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch02\07_kv_cache_principle.ipynb")