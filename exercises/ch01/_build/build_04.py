# -*- coding: utf-8 -*-
"""生成 04_transformer_quickstart.ipynb 与 app_04_transformer_demo.py"""
from helpers import D, chapter_cover, wrapup, new_nb, CH01
from pathlib import Path

APP_04 = D('''
# -*- coding: utf-8 -*-
# app_04_transformer_demo.py — 手写单层 Transformer 演示 🧠
import streamlit as st
import torch
import torch.nn as nn
import math
import plotly.graph_objects as go

st.set_page_config(page_title="单层 Transformer 🧠", layout="wide")
st.title("🧠 第 04 课 · 手写单层 Transformer 块")

st.markdown("""
Transformer 是现代大模型的核心积木。这一课用手写一个**单层 transformer block**:
embedding → QKV → 缩放点积注意力 → FFN → LayerNorm → 残差,并在 CPU 上跑前向。
下方调整**序列长度**与**注意力头数**,实时看 attention 权重热图——每个 token 在关注谁。
""")

HIDDEN = 32

class ToyBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, hidden, bias=False)
        self.wv = nn.Linear(hidden, hidden, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, verbose=False):
        B, T, H = x.shape
        hd = self.head_dim
        if verbose: print(f"[1] embedding        {tuple(x.shape)}")
        q = self.wq(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        if verbose: print(f"[2] Q/K/V 多头拆分    {tuple(q.shape)}")
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        if verbose: print(f"[3] 缩放点积 scores   {tuple(scores.shape)}")
        weights = torch.softmax(scores, dim=-1)
        if verbose: print(f"[4] attention 权重    {tuple(weights.shape)}")
        attn = weights @ v
        attn = attn.transpose(1, 2).reshape(B, T, H)
        out_attn = self.wo(attn)
        if verbose: print(f"[5] 注意力输出       {tuple(out_attn.shape)}")
        h = x + out_attn
        if verbose: print(f"[6] + 残差           {tuple(h.shape)}")
        h = self.norm1(h)
        if verbose: print(f"[7] LayerNorm        {tuple(h.shape)}")
        h = h + self.ffn(h)
        if verbose: print(f"[8] FFN + 残差        {tuple(h.shape)}")
        h = self.norm2(h)
        if verbose: print(f"[9] LayerNorm        {tuple(h.shape)}")
        return h, weights

with st.sidebar:
    st.header("🎛️ 参数")
    seq_len = st.slider("序列长度 T", 4, 16, 8, 1)
    n_heads = st.slider("注意力头数", 1, 4, 4, 1)
    head_idx = st.slider("查看第几个头", 0, n_heads - 1, 0, 1)
    st.caption("head_dim = hidden / n_heads,头数必须能整除 hidden=32。")

torch.manual_seed(0)
block = ToyBlock(HIDDEN, n_heads)
x = torch.randn(1, seq_len, HIDDEN)
out, weights = block(x)

w = weights[0, head_idx].detach().numpy()
st.subheader(f"🔍 Attention 权重热图(head {head_idx}, T={seq_len})")
fig = go.Figure(go.Heatmap(
    z=w, x=[f"t{i}" for i in range(seq_len)],
    y=[f"t{i}" for i in range(seq_len)],
    colorscale="Blues", zmin=0, zmax=w.max(),
    text=[[f"{v:.2f}" for v in row] for row in w], texttemplate="%{text}"))
fig.update_layout(title="第 j 行 = token t_j 对前面所有 token 的关注权重(行和为 1)",
                  xaxis_title="被关注的 key token", yaxis_title="查询 query token",
                  height=460, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 每一行是 softmax 的结果,加起来为 1:它表示“当前 token 把注意力花在谁身上”。"
           "头数越多,模型能从不同视角关注不同位置。")
''')

NB = new_nb("第 04 课 · 手写单层 Transformer 块",
            subtitle="embedding → QKV → 缩放点积注意力 → FFN → LayerNorm → 残差,一步步打印张量形状,再看清 attention 权重热图",
            emoji="🧠")

chapter_cover(NB,
    objectives=[
        "理解 Transformer 块的完整前向:embedding、QKV、缩放点积注意力、FFN、LayerNorm、残差",
        "在 torch CPU 上手写一个单层 block,并逐步打印张量形状",
        "看懂 scaled dot-product attention 的公式与多头拆分的张量变换",
        "用 plotly 热图可视化 attention 权重:每个 token 在关注谁",
    ],
    toc=[
        ("直觉:信息在'交换名片'", "注意力让序列里的 token 互相交流信息"),
        ("搭建玩具块:QKV 从哪来", "用三个线性层把 embedding 拆成 Q/K/V"),
        ("缩放点积注意力", "scores = QKᵀ/√d,softmax 成权重,再加权求和 V"),
        ("多头拆分与合并", "把 hidden 切成多个头,并行注意力后拼回"),
        ("FFN + LayerNorm + 残差", "逐位置的前馈网络与稳定训练的三件套"),
        ("逐步打印形状:9 步前向", "用 verbose 前向看清每一步张量变化"),
        ("Attention 权重热图", "plotly 热图看 token 间的关注关系"),
        ("配套 Streamlit 演示", "app_04_transformer_demo.py:序列长度/头数滑杆看热图"),
    ],
    links=[
        ("Attention Is All You Need(Transformer 原始论文)", "https://arxiv.org/abs/1706.03762"),
        ("PyTorch nn.Transformer 文档", "https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html"),
        ("Jay Alammar: The Illustrated Transformer", "https://jalammar.github.io/illustrated-transformer/"),
    ])

NB.md("## 1️⃣ 直觉:信息在'交换名片' 🤝",
D('''
想象一个会议室,里面坐着一段话的每个 token。它们要**互相交流信息**:比如“它”需要知道
“猫”是谁,才能把“猫在追老鼠,它跑得很快”里的“它”指代清楚。**注意力机制**就是让每个
token 去“看”其他所有 token,决定自己该从谁那里汲取多少信息——像交换名片时,
有人递得勤(权重高),有人被忽略(权重低)。

一个 **Transformer 块**就是这套“开会 + 讨论 + 总结”的完整流程。本课用 torch 在
**CPU** 上手写一个单层块,把张量形状一步步打印出来,真正弄懂每个数字从哪来到哪去。
'''))

NB.code("import os\nos.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", \"TRUE\")\n"
        "import torch\nimport torch.nn as nn\nimport numpy as np\nimport pandas as pd\n"
        "import math\n\ntorch.manual_seed(0)\ndevice = \"cpu\"",
        "🧪 老规矩:先解决 Windows 下 torch 的 OMP 库冲突。注意本课所有张量都在 **CPU** 上运行(纯教学,无需 GPU)。")

NB.md("## 2️⃣ 搭建玩具块:QKV 从哪来 🔑",
D('''
输入是一段序列的 **embedding** 张量,形状 `(B, T, H)`:

- **B**(batch):一批几段话;
- **T**(seq_len):每段话几个 token;
- **H**(hidden):每个 token 用多少维向量表示(本课取 32)。

注意力需要三个角色:**Q(查询)**、**K(键)**、**V(值)**。直觉上:

- Q 是“我在找什么”;
- K 是“我有什么可被找到”;
- V 是“我真正能提供的内容”。

用三个线性层把同一个 `x` 分别投影成 Q/K/V。先把块定义出来:
'''))

NB.code(D('''
HIDDEN, N_HEADS = 32, 4
HEAD_DIM = HIDDEN // N_HEADS

class ToyBlock(nn.Module):
    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, hidden, bias=False)
        self.wv = nn.Linear(hidden, hidden, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, verbose=False):
        B, T, H = x.shape
        hd = self.head_dim
        if verbose: print(f"[1] embedding        {tuple(x.shape)}")
        q = self.wq(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        if verbose: print(f"[2] Q/K/V 多头拆分    {tuple(q.shape)}")
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        if verbose: print(f"[3] 缩放点积 scores   {tuple(scores.shape)}")
        weights = torch.softmax(scores, dim=-1)
        if verbose: print(f"[4] attention 权重    {tuple(weights.shape)}")
        attn = weights @ v
        attn = attn.transpose(1, 2).reshape(B, T, H)
        out_attn = self.wo(attn)
        if verbose: print(f"[5] 注意力输出       {tuple(out_attn.shape)}")
        h = x + out_attn
        if verbose: print(f"[6] + 残差           {tuple(h.shape)}")
        h = self.norm1(h)
        if verbose: print(f"[7] LayerNorm        {tuple(h.shape)}")
        h = h + self.ffn(h)
        if verbose: print(f"[8] FFN + 残差        {tuple(h.shape)}")
        h = self.norm2(h)
        if verbose: print(f"[9] LayerNorm        {tuple(h.shape)}")
        return h, weights
'''), "🎨 这个类就是完整的单层块。`forward` 里每一步都顺手打印形状,方便对照下面的讲解。")

NB.md("## 3️⃣ 缩放点积注意力 ⚖️",
D('''
**缩放点积注意力(scaled dot-product attention)** 的核心就三步:

1. 计算相似度:`scores = Q @ Kᵀ`(每个 query 与每个 key 点积,衡量“匹配程度”);
2. 缩放:`÷ √d_k`(防止点积过大把 softmax 推向饱和,梯度消失);
3. 归一化 + 加权:`weights = softmax(scores)`,再 `weights @ V` 加权求和。

$$\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V$$

点积大 = 相似 → softmax 给高权重 → 该 token 的信息被更多地吸收。这就是“关注”的数学含义。
'''))

NB.md("## 4️⃣ 多头拆分与合并 🪄",
D('''
现实中的 Transformer 用**多头注意力(multi-head)**:不只用一组 Q/K/V,而是把 hidden
拆成 `n_heads` 个 `head_dim` 的“子空间”,让每个头从**不同角度**去关注。

张量变换如下(`x` 是 `(B, T, H)`):

1. `view(B, T, n_heads, head_dim)` 把 H 切成头;
2. `transpose(1, 2)` 变成 `(B, n_heads, T, head_dim)`,好让所有头一起并行算注意力;
3. 注意力输出再 `transpose` 回来、`reshape(B, T, H)` 拼回原来的 hidden;
4. 最后过一个 `wo` 线性层投影回 hidden。

这样既保留了“每个 token 关注谁”的能力,又让模型有多个观察视角。vLLM 推理时,
GPU 里正是这样批量并行处理所有头。
'''))

NB.md("## 5️⃣ FFN + LayerNorm + 残差 🔧",
D('''
注意力负责**让 token 之间交流**;而 **FFN(前馈网络)** 在每个 token 的位置上独立地
做一次非线性变换,相当于“每个人都把收集到的信息自己消化一下”。

为了训练稳定,还要两个组件:

- **残差连接(residual)**:`h = x + attention_output`,让梯度能顺畅地流过深层网络;
- **LayerNorm**:对每个 token 的 hidden 做归一化(均值 0、方差 1),再缩放平移,
  避免激活值过大过小。

本课用 `nn.LayerNorm` 与 `nn.GELU` 直接调用,你只需理解它们“做了什么”即可。
'''))

NB.md("## 6️⃣ 逐步打印形状:9 步前向 📐",
D('''
现在跑一次 `verbose=True` 的前向,输入一段长度 T=8 的随机 embedding,
把 9 步的张量形状全部打出来。对照前面的讲解,逐个数字核对:
'''))

NB.code(D('''
block = ToyBlock(HIDDEN, N_HEADS)
x = torch.randn(1, 8, HIDDEN)          # (B, T, H)
out, weights = block(x, verbose=True)
print()
print("最终输出形状:", tuple(out.shape))
'''), "📐 你会看到:输入 (1,8,32) → QKV 拆成 4 个头 (1,4,8,8) → scores (1,4,8,8) → 权重 (1,4,8,8) → 拼回 (1,8,32) → 一路到输出 (1,8,32)。形状没变,信息却被'重排'过了。")

NB.code(D('''
# 验证缩放因子 √d_k:温度太高 softmax 会饱和,缩放让它保持在合理区间
print(f"head_dim d_k = {HEAD_DIM},缩放因子 √d_k = {math.sqrt(HEAD_DIM):.3f}")
print("权重每行之和(应为 1):", weights[0, 0].detach().sum(dim=-1).numpy())
'''), "✅ 每行 softmax 权重加起来为 1,证明它是合法概率分布;缩放因子保证了分布不会过于尖锐。")

NB.md("## 7️⃣ Attention 权重热图 🎨",
D('''
把第 0 个头的权重矩阵画成热图:行是 query token,列是 key token,颜色越深代表
“这个 token 越关注那个 token”。对角线通常较深(每个 token 都最关注自己),
而深色块往往揭示了词与词之间的指代关系。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

def viz_attn(w, title=""):
    T = w.shape[0]
    fig = go.Figure(go.Heatmap(
        z=w, x=[f"t{i}" for i in range(T)], y=[f"t{i}" for i in range(T)],
        colorscale="Blues", zmin=0, zmax=w.max(),
        text=[[f"{v:.2f}" for v in row] for row in w], texttemplate="%{text}"))
    fig.update_layout(title=title or "Attention 权重热图(head 0)",
                      xaxis_title="被关注的 key", yaxis_title="查询 query",
                      height=460, margin=dict(l=10, r=10, t=50, b=10))
    return fig

viz_attn(weights[0, 0].detach().numpy())
'''), "🎨 每个 token 的注意力分布在它的那一行;颜色越深,权重越高。把序列换长一点、头数换多,热图会呈现更丰富的结构。")

NB.md("## 8️⃣ 配套 Streamlit 演示:动手调头数与长度 🎛️",
D('''
运行同目录下的 `app_04_transformer_demo.py`,拖动**序列长度**与**注意力头数**滑杆,
attention 权重热图实时重绘,还能切换查看不同头:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_04_transformer_demo.py
```

浏览器打开 **http://localhost:8501**。把序列长度调大、头数调多,观察不同头关注模式如何分化。
完整源码如下(与同目录 `app_04_transformer_demo.py` 一致):
'''))

NB.code("%%writefile app_04_transformer_demo.py\n" + APP_04,
        "📜 这就是 app_04_transformer_demo.py 的完整源码。notebook 与 app 共用同一个 ToyBlock,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "Transformer 块 = 注意力(让 token 交流) + FFN(各自消化) + LayerNorm + 残差",
        "缩放点积注意力:scores=QKᵀ/√d_k,softmax 成权重再加权 V,刻画 token 间关注关系",
        "多头把 hidden 切成 head_dim 子空间并行注意力,再拼回,提供多角度观察",
        "残差让梯度顺畅、LayerNorm 稳定激活,是深层网络能训练的关键",
        "用 verbose 前向打印 9 步形状 + plotly 热图,把'关注谁'看得一清二楚",
    ],
    practice=[
        "把 HIDDEN 改成 64、N_HEADS 改成 8,重跑前向,核对 head_dim 是否仍等于 hidden//heads",
        "给 ToyBlock 加一个 dropout 层(放在 attention 输出后),观察形状是否受影响",
        "取 T=16 的输入,画出第 1 个头与第 3 个头的热图,比较它们的关注模式差异",
        "读《Attention Is All You Need》图 2,对照本课代码的每一层,把论文与代码一一对应",
    ],
    links=[
        ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("Jay Alammar: The Illustrated Transformer", "https://jalammar.github.io/illustrated-transformer/"),
        ("PyTorch nn.LayerNorm / nn.MultiheadAttention 文档", "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"),
    ])

NB.save(str(Path(CH01) / "04_transformer_quickstart.ipynb"))

app_path = Path(CH01) / "app_04_transformer_demo.py"
app_path.write_text(APP_04 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
