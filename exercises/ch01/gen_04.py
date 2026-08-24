# -*- coding: utf-8 -*-
"""生成第 04 课 notebook:手写单层 Transformer 块(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:信息交换直觉 -> 注意力/多头/FFN/LayerNorm/残差 公式 -> ToyBlock 逐行推演(打印每个中间张量 shape) -> 数值验证 -> 真实模型配置 -> vLLM 关联
2. 每一行代码都有 inline 注释
3. 每个中间张量打印 shape 并标注维度含义(B/T/H/head/dim)
4. 论文支撑:Attention Is All You Need(arXiv:1706.03762)、LayerNorm(arXiv:1607.06450)、GELU(arXiv:1606.08415)、RoPE(arXiv:2104.09864)
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_04_transformer_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 04 课 · 手写单层 Transformer 块",
    subtitle="embedding → QKV → 缩放点积注意力 → 多头 → FFN → LayerNorm → 残差,每个张量打印 shape",
    emoji="🧠", chapter="第 1 章 · LLM 推理基础",
)

chapter_cover(
    nb,
    objectives=[
        "理解缩放点积注意力 Attention(Q,K,V)=softmax(QK^T/√d_k)V 的每个符号与维度",
        "理解多头注意力:为什么把 hidden 切成 n_heads 个头、每个头看不同的子空间",
        "理解 FFN / LayerNorm / 残差连接的作用,以及它们与注意力的分工",
        "手写单层 transformer block,前向时打印每一步的张量 shape 并核对维度含义",
        "数值验证:softmax 行和为 1、与 PyTorch 官方实现最大误差、缩放因子 √d_k 的作用",
        "用真实模型配置表(LLaMA/Qwen)计算参数规模,建立与 vLLM 中注意力实现的联系",
    ],
    toc=[
        ("直觉与动机", "token 之间如何『交换名片』"),
        ("核心定义与公式", "注意力 / 多头 / FFN / LayerNorm / 残差的严格定义"),
        ("搭建 ToyBlock", "QKV 投影与多头拆分,逐行注释"),
        ("缩放点积注意力", "打分 → 缩放 → softmax → 加权求和"),
        ("FFN + LayerNorm + 残差", "各自消化 + 稳定训练"),
        ("9 步前向逐形状打印", "一个 cell 打印所有中间张量 shape"),
        ("数值验证", "行和=1、官方实现对照、缩放因子"),
        ("真实规模数字", "LLaMA/Qwen 配置表与参数公式"),
        ("与 vLLM 的关系", "QKV 线性层 / attention 算子 / RoPE"),
        ("Streamlit 动态演示", "调头数与序列长度看注意力热图"),
    ],
    links=[
        ("Vaswani et al.: Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("Ba et al.: Layer Normalization", "https://arxiv.org/abs/1607.06450"),
        ("Hendrycks & Gimpel: Gaussian Error Linear Units (GELU)", "https://arxiv.org/abs/1606.08415"),
        ("Su et al.: RoFormer (RoPE)", "https://arxiv.org/abs/2104.09864"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉与动机:token 之间如何『交换名片』\n\n"
    "想象一个会议室,里面坐着一段话的每个 token。它们要**互相交流信息**:"
    "比如「它」需要知道「猫」是谁,才能把「猫在追老鼠,它跑得很快」里的指代弄清楚。"
    "**注意力机制(attention)** 就是让每个 token 去「看」其他所有 token,"
    "决定自己该从谁那里汲取多少信息——像交换名片时,有人递得勤(权重大),有人只是点头(权重小)。\n\n"
    "一个 **Transformer 块(block)** 把这件事组装成标准流水线:\n\n"
    "```\n"
    "输入 x → [多头注意力] → +残差 → LayerNorm → [FFN] → +残差 → LayerNorm → 输出\n"
    "```\n\n"
    "**注意力负责 token 之间交流,FFN 负责每个 token 独立消化**——"
    "这套设计出自 [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)"
    "的《Attention Is All You Need》,是现代所有 LLM 的基本积木。"
    "本课把它在 PyTorch 里逐行写出来,每个中间张量都打印 shape。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式\n\n"
    "**缩放点积注意力(scaled dot-product attention)** 就三步([Vaswani et al., 2017]"
    "(https://arxiv.org/abs/1706.03762) 的式 (1)):\n\n"
    "$$ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^{\\top}}{\\sqrt{d_k}}\\right) V $$\n\n"
    "1. **打分**:$QK^{\\top}$ 计算每个 query 与每个 key 的相似度;\n"
    "2. **缩放**:除以 $\\sqrt{d_k}$,防止点积过大把 softmax 推向饱和区(梯度消失);\n"
    "3. **归一化 + 加权**:softmax 把分数变成权重(行和为 1),再对 $V$ 加权求和。\n\n"
    "**多头注意力(multi-head)**:把 hidden 切成 $H$ 个 `head_dim` 的子空间,"
    "每头独立做注意力再拼接。论文原话是:*«instead of performing a single attention function "
    "with d_model-dimensional keys, values and queries, we found it beneficial to linearly "
    "project the queries, keys and values h times»*。\n\n"
    "**FFN(前馈网络)**:逐位置独立的两层线性 + 激活,论文式 (2) 用 ReLU:\n\n"
    "$$ \\text{FFN}(x) = \\max(0,\\, xW_1 + b_1) W_2 + b_2 $$\n\n"
    "现代模型把 ReLU 换成 GELU([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415))。\n\n"
    "**残差 + LayerNorm**:每个子层输出为 $\\text{LayerNorm}(x + \\text{Sublayer}(x))$——"
    "残差让梯度能穿过深层,LayerNorm([Ba et al., 2016](https://arxiv.org/abs/1607.06450))稳定分布。\n\n"
    "| 符号 | 含义 | 本课取值 |\n"
    "|---|---|---|\n"
    "| $B$ | batch(一次几条序列) | 1 |\n"
    "| $T$ | 序列长度 seq_len | 8 |\n"
    "| $H_{hidden}$ | 隐藏维 hidden | 32 |\n"
    "| $H$ | 注意力头数 num_heads | 4 |\n"
    "| $d_k$ | 每头维度 head_dim = hidden/H | 8 |\n"
    "| $Q,K,V$ | 投影后的查询/键/值 | (B,H,T,d_k) |\n"
    "| $\\text{FFN}_{hidden}$ | FFN 中间层宽度 | 2×hidden=64 |\n\n"
    "> ⚠️ 注意区分:$H_{hidden}$(隐藏维)与 $H$(头数)是两个量。"
    "工程里张量布局用 `(B, T, hidden)` 进、`(B, H, T, d_k)` 多头拆、再拼回 `(B, T, hidden)`。"
)

# =====================================================================
# 第 3 节 · 搭建 ToyBlock
# =====================================================================
nb.md(
    "## 3. 搭建 ToyBlock:QKV 从哪来\n\n"
    "输入是一段序列的 **embedding** 张量,形状 $(B, T, H_{hidden})$。"
    "注意力需要三个角色:Q(我来查)、K(被查的索引)、V(查到了取什么内容)——"
    "它们都由输入 $x$ 分别过一个线性投影得到。下面逐行实现。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Windows 下 OpenMP 库重复加载报错
import torch
import torch.nn as nn
import math                                            # 提供 sqrt

torch.manual_seed(0)                                    # 固定随机种子,结果可复现
HIDDEN, N_HEADS = 32, 4                                 # 隐藏维 32,注意力头数 4
HEAD_DIM = HIDDEN // N_HEADS                            # 每头维度 = 32/4 = 8

class ToyBlock(nn.Module):
    """单层 Transformer 块:注意力 + 残差 + LayerNorm + FFN + 残差 + LayerNorm。
    forward 里每个子步骤都打印张量 shape,方便对照公式。
    """

    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS):
        super().__init__()                               # 调用父类 nn.Module 初始化(必须)
        self.hidden = hidden                             # 保存隐藏维
        self.n_heads = n_heads                           # 保存头数
        self.head_dim = hidden // n_heads                # 每头维度(必须能整除)
        # 四个投影:Q/K/V 各一个,输出再投影(O)一个,都是 hidden->hidden
        self.wq = nn.Linear(hidden, hidden, bias=False)  # Q 投影:(B,T,h) -> (B,T,h)
        self.wk = nn.Linear(hidden, hidden, bias=False)  # K 投影
        self.wv = nn.Linear(hidden, hidden, bias=False)  # V 投影
        self.wo = nn.Linear(hidden, hidden, bias=False)  # 注意力输出投影
        self.norm1 = nn.LayerNorm(hidden)                # 注意力子层后的 LayerNorm
        self.ffn = nn.Sequential(                        # 两层 FFN
            nn.Linear(hidden, hidden * 2),               # 升维:h -> 2h
            nn.GELU(),                                   # 激活函数(现代模型的默认选择)
            nn.Linear(hidden * 2, hidden),               # 降维回:2h -> h
        )
        self.norm2 = nn.LayerNorm(hidden)                # FFN 子层后的 LayerNorm

    def forward(self, x, verbose=False):
        """前向:x 是 (B, T, hidden) 的 embedding 张量。
        verbose=True 时打印每一步的 shape。返回 (输出, attention 权重)。"""
        B, T, H = x.shape                                # 解出 batch B、序列长 T、隐藏维 H
        hd = self.head_dim                               # 每头维度 d_k
        if verbose: print(f"[1] 输入 embedding       {tuple(x.shape)} <- (B={B}, T={T}, hidden={H})")

        # ---- ① QKV 投影 + 多头拆分 ----
        q = self.wq(x).view(B, T, self.n_heads, hd).transpose(1, 2)  # (B,T,h)->(B,T,H,dk)->(B,H,T,dk)
        k = self.wk(x).view(B, T, self.n_heads, hd).transpose(1, 2)  # 同样的多头布局
        v = self.wv(x).view(B, T, self.n_heads, hd).transpose(1, 2)  # 同样的多头布局
        if verbose: print(f"[2] Q/K/V 多头拆分       {tuple(q.shape)} <- (B={B}, heads={self.n_heads}, T={T}, dk={hd})")

        # ---- ② 缩放点积注意力 ----
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)  # Q(B,H,T,dk)@K^T(B,H,dk,T) -> (B,H,T,T),除以√dk
        if verbose: print(f"[3] 缩放点积 scores     {tuple(scores.shape)} <- (B, heads, query T, key T)")

        weights = torch.softmax(scores, dim=-1)           # 对最后一个维度(key)归一化,行和为 1
        if verbose: print(f"[4] attention 权重       {tuple(weights.shape)} <- 每行是合法概率分布")

        attn = weights @ v                                # 加权求和:weights(B,H,T,T)@V(B,H,T,dk) -> (B,H,T,dk)
        attn = attn.transpose(1, 2).reshape(B, T, H)      # 多头拼回:(B,H,T,dk) -> (B,T,hidden)
        out_attn = self.wo(attn)                          # 输出投影,回到 hidden 维
        if verbose: print(f"[5] 注意力输出(拼回)     {tuple(out_attn.shape)} <- 形状与输入相同")

        # ---- ③ 残差 + LayerNorm ----
        h = x + out_attn                                  # 残差连接:原值 + 子层输出(同 shape 相加)
        if verbose: print(f"[6] 残差 (x + attn)      {tuple(h.shape)}")
        h = self.norm1(h)                                 # LayerNorm 归一化(沿最后一维)
        if verbose: print(f"[7] LayerNorm           {tuple(h.shape)}")

        # ---- ④ FFN + 残差 + LayerNorm ----
        h = h + self.ffn(h)                               # FFN 逐 token 独立变换 + 残差
        if verbose: print(f"[8] FFN + 残差           {tuple(h.shape)}")
        h = self.norm2(h)                                 # 第二个 LayerNorm
        if verbose: print(f"[9] LayerNorm           {tuple(h.shape)}")
        return h, weights                                 # 返回输出与注意力权重(供可视化)''',
    "🛠️ **ToyBlock**。核心是 `view` + `transpose` 的多头拆分:`(B,T,hidden)` 先切成 `(B,T,H,dk)`,"
    "再转置成 `(B,H,T,dk)`——这正是 PyTorch `F.scaled_dot_product_attention` 要求的 4D 布局。",
)

# =====================================================================
# 第 4 节 · 9 步前向
# =====================================================================
nb.md(
    "## 4. 9 步前向:把形状全部打出来\n\n"
    "跑一次 `verbose=True` 的前向,输入一段 T=8 的随机 embedding,"
    "把 9 步的张量形状全部打印出来,对照第 2 节的公式逐个核对。"
)

nb.code(
    '''# 实例化块并跑一次 verbose 前向
block = ToyBlock(HIDDEN, N_HEADS)                        # 建块:hidden=32, heads=4, dk=8
x = torch.randn(1, 8, HIDDEN)                            # 输入 (B=1, T=8, hidden=32)
out, weights = block(x, verbose=True)                    # 前向,打印每一步 shape
print()
print("最终输出形状:", tuple(out.shape))                   # 应与输入 (1,8,32) 完全一致''',
    "📐 **你会看到**:输入 (1,8,32) → QKV 拆成 4 个头 (1,4,8,8) → scores (1,4,8,8) → "
    "权重 (1,4,8,8) → 拼回 (1,8,32) → 一路到输出 (1,8,32)。"
    "**形状没变,信息却被『重排』过了**——这正是 Transformer 块「恒等维度、复杂变换」的设计。",
)

# =====================================================================
# 第 5 节 · 数值验证
# =====================================================================
nb.md(
    "## 5. 数值验证:正确性三连\n\n"
    "做三个数值实验:① 注意力权重每行和为 1;② 与 PyTorch 官方实现对照最大误差;"
    "③ 缩放因子 $\\sqrt{d_k}$ 为什么必要。"
)

nb.code(
    '''# ① 每行 softmax 权重和为 1:合法概率分布
row_sums = weights[0, 0].detach().sum(dim=-1)            # 取 head 0,对 key 维求和(每行)
print(f"① 权重每行之和: {row_sums.numpy()}")               # 应全是 1.0

# ② 与官方实现对照(手写 vs F.scaled_dot_product_attention)
import torch.nn.functional as F
q = torch.randn(1, 4, 8, 8)                              # 构造 (B,H,T,dk) 的 Q
k = torch.randn(1, 4, 8, 8)                              # K
v = torch.randn(1, 4, 8, 8)                              # V
scores2 = q @ k.transpose(-2, -1) / math.sqrt(8)         # 手写:打分 + 缩放
w2 = torch.softmax(scores2, dim=-1)                      # 手写:softmax
out2 = w2 @ v                                            # 手写:加权求和
ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None)  # 官方实现
print(f"② 与官方实现最大误差: {(out2 - ref).abs().max().item():.2e}  (≈0 说明手写正确)")

# ③ 缩放因子 √d_k:不缩放时点积方差 = d_k,过大导致 softmax 饱和
big_d = 512                                              # 假设 dk 很大(真实模型常用)
num_samples = 200000                                     # 抽很多样本估计方差(单样本 var 无意义)
q3 = torch.randn(num_samples, 1, 1, big_d)               # 随机 Q,批量采样
k3 = torch.randn(num_samples, 1, 1, big_d)               # 随机 K,批量采样
dot_nos = (q3 @ k3.transpose(-2, -1))                    # 不缩放的点积:(S,1,1,1)
print(f"③ 不缩放时点积方差 ≈ {dot_nos.var().item():.1f}(理论 = dk = {big_d});"
      f"缩放 √dk={math.sqrt(big_d):.1f} 后方差 ≈ {(dot_nos/math.sqrt(big_d)).var().item():.2f}")''',
    "✅ **三个结论**:① 每行和为 1,是合法概率分布;② 与 PyTorch 官方实现误差 ~1e-7,"
    "证明手写正确;③ 点积方差随 $d_k$ 线性增长,不缩放会饱和——"
    "这正是论文里 *«the dot products grow large in magnitude, pushing the softmax function "
    "into regions where it has extremely small gradients»* 的量化验证。",
)

# =====================================================================
# 第 6 节 · 真实规模数字
# =====================================================================
nb.md(
    "## 6. 真实规模数字:LLaMA / Qwen 配置\n\n"
    "玩具块只有 32 维,真实模型把它堆成几十层、几千维。"
    "看两张真实配置表,并验证「参数从哪来」。"
)

nb.code(
    '''# 真实模型配置表与参数公式
configs = [
    {"name": "LLaMA-2 7B",     "hidden": 4096, "layers": 32, "heads": 32, "ffn": 11008},
    {"name": "Qwen2.5-7B",     "hidden": 3584, "layers": 28, "heads": 28, "ffn": 18944},
    {"name": "GPT-3 175B",     "hidden": 12288, "layers": 96, "heads": 96, "ffn": 49152},
]
print(f"{'模型':<14} | {'hidden':>6} | {'层数':>4} | {'头数':>4} | {'每头维度':>6} | {'每层参数':>10}")
for c in configs:                                        # 遍历配置
    hd = c["hidden"] // c["heads"]                       # 每头维度 = hidden / heads
    # 每层参数:注意力 4*hidden²(QKV+O) + FFN 3*hidden*ffn(SwiGLU 三个矩阵)
    per_layer = 4 * c["hidden"] ** 2 + 3 * c["hidden"] * c["ffn"]
    print(f"{c['name']:<14} | {c['hidden']:6d} | {c['layers']:4d} | {c['heads']:4d} | {hd:6d} | {per_layer/1e9:8.2f}B")

# 验证:LLaMA-2 7B 的近似总参数量(不含 embedding)
c = configs[0]                                           # LLaMA-2 7B
total = c["layers"] * (4 * c["hidden"] ** 2 + 3 * c["hidden"] * c["ffn"])
print(f"\\nLLaMA-2 7B 近似总参数量(不含词表) ≈ {total/1e9:.1f}B  <-> 官方口径 6.7B")''',
    "🚀 **要点**:attention 贡献 $4H^2$(Q/K/V/O 四个矩阵),FFN 贡献 $3H\\cdot\\text{ffn}$"
    "(SwiGLU 三个门控矩阵)。每头维度恒为 `hidden/heads`(LLaMA/Qwen 都是 128),"
    "所以**头数增加不增加参数**,只是让注意力看更多子空间。",
)

nb.md(
    "| 模型 | hidden | layers | heads | kv_heads | head_dim | 位置编码 |\n"
    "|---|---|---|---|---|---|---|\n"
    "| LLaMA-2 7B | 4096 | 32 | 32 | 32 | 128 | RoPE |\n"
    "| Qwen2.5-7B | 3584 | 28 | 28 | 4(GQA) | 128 | RoPE |\n"
    "| GPT-3 175B | 12288 | 96 | 96 | 96 | 128 | 学习式 |\n\n"
    "> 📄 现代模型几乎都用 **RoPE(旋转位置编码)** 给 token 注入位置信息——"
    "它不是加一个向量,而是把 Q/K 按位置**旋转**一个角度,让注意力分数只依赖相对位置。"
    "见 [Su et al., 2021](https://arxiv.org/abs/2104.09864) 的 RoFormer。"
    "GQA(共享 KV 头)则是把 `kv_heads` 从 `heads` 缩小来省 KV Cache 显存,第 2 章展开。"
)

# =====================================================================
# 第 7 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 7. 与 vLLM 的关系:本课块是引擎里的一个算子\n\n"
    "vLLM 不自己发明 Transformer,而是用高度优化的算子实现本课的同一条前向:\n\n"
    "1. **QKV 投影**:`vllm/model_executor/layers/linear.py` 的 `QKVParallelLinear`,"
    "把三个 Linear 合并成一个算子(减少内核启动);\n"
    "2. **注意力**:推理时用 **PagedAttention** 内核(第 2 章 08-13 课),"
    "训练/前向对照可用 PyTorch 官方 `F.scaled_dot_product_attention`(FlashAttention 路径);\n"
    "3. **位置编码**:`vllm/model_executor/layers/rotary_embedding.py` 实现 RoPE,"
    "推理时在 K/V 上逐位置旋转;\n"
    "4. **层栈**:`TransformerDecoderLayer` 堆叠多层,与第 6 节配置表一一对应。\n\n"
    "**本课的价值**:理解了这一层的每个张量 shape,就能看懂 vLLM 里任意一个 attention "
    "kernel 的输入输出契约——因为它们都在算同一个公式。"
)

# =====================================================================
# 第 8 节 · Streamlit
# =====================================================================
nb.md(
    "## 8. 🖥️ Streamlit 动态演示:动手调头数与长度\n\n"
    "运行 `app_04_transformer_demo.py`:拖动**序列长度**与**注意力头数**滑杆,"
    "attention 权重热图实时重绘,还能切换查看不同头。\n\n"
    "### 📜 App 完整源码(`app_04_transformer_demo.py`)"
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
    "    print(\"    请把上方源码保存为 app_04_transformer_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_04_transformer_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_04_transformer_demo.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_04_transformer_demo.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,拖动滑杆观察热图变化。\n\n"
    "🔍 试试:序列拉到 16、头数在 1/2/4/8 之间切换,观察不同头关注的位置模式不同。"
)

wrapup(
    nb,
    summary=[
        "Transformer 块 = 多头注意力(token 间交流)+ FFN(各自消化)+ LayerNorm + 残差,维度恒为 hidden",
        "缩放点积注意力 Attention=softmax(QK^T/√d_k)V:打分、缩放防饱和、softmax 归一化、加权求和",
        "多头把 hidden 切成 H 个 head_dim 子空间,每头独立关注,最后拼回——头数不增加参数",
        "残差(LayerNorm(x+Sublayer(x)))让梯度可穿深,LayerNorm 稳定分布,GELU 是现代激活默认",
        "数值验证:权重行和为 1、与官方实现误差 ~1e-7、点积方差∝d_k 解释了 √d_k 的必要性",
        "真实模型(LLaMA/Qwen)每层 ≈ 4hidden² + 3·hidden·ffn 参数,head_dim 恒为 128,RoPE 做位置编码",
        "vLLM 用 QKVParallelLinear + PagedAttention + RoPE 实现同一条前向,本课 shape 契约全适用",
    ],
    practice=[
        "把 head_dim 改成不整除的值(hidden=32, heads=5),观察 view 报错并解释 reshape 的约束",
        "给 forward 加一个参数 use_sdp=True,内部改调 F.scaled_dot_product_attention,对照输出与手写是否一致",
        "把 GELU 换成 ReLU,看张量 shape 不变、值分布变化——验证激活函数只改非线性",
        "实现一个 2 层 ToyBlock 堆叠,打印层间 hidden state 的 shape 与 norm 的 scale 变化",
    ],
    links=[
        ("Vaswani et al.: Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("Ba et al.: Layer Normalization", "https://arxiv.org/abs/1607.06450"),
        ("Hendrycks & Gimpel: GELU", "https://arxiv.org/abs/1606.08415"),
        ("Su et al.: RoFormer (RoPE)", "https://arxiv.org/abs/2104.09864"),
        ("The Annotated Transformer (Harvard)", "https://nlp.seas.harvard.edu/annotated-transformer/"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch01\04_transformer_quickstart.ipynb")