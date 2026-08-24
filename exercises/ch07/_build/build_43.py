# -*- coding: utf-8 -*-
"""生成 43_online_softmax.ipynb 与 app_43_online_softmax.py(教材级重写版)

对齐 REWRITE_STANDARD.md。论文支撑:
Online normalizer calculation for softmax (Milakov & Gimelshein, NVIDIA, 2018, arXiv:1805.02867);
FlashAttention (arXiv:2205.14135) 把在线 softmax 用作分块 attention 的数学地基。
"""
from pathlib import Path
from helpers import D, new_nb, chapter_cover, wrapup, CH07, CPU_HEADER
from helpers import SOFTMAX_NAIVE, SOFTMAX_TWOPASS, SOFTMAX_ONLINE

APP_FILE = "app_43_online_softmax.py"

APP_43 = D('''
# -*- coding: utf-8 -*-
# app_43_online_softmax.py — 在线 softmax:逐块状态可视化 🧮
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧮 43 · 在线 Softmax", layout="wide")
st.title("🧮 第 43 课 · 在线 Softmax:边扫边修正")

st.markdown("""
softmax 的分母(所有 $e^{x_i}$ 之和)需要 **“看完一整行”** 才算得出来。可 FlashAttention 偏偏要
**分块扫描**、边算边丢。怎么办?用**在线 softmax**:维护一个**运行时最大值 $m$** 和**运行时总和 $l$**,
每看到一个块,就用「旧 $m$ 与当前块最大值」去**校正**已经累积的 $l$。
下方拖动**块大小**与**序列长度**,逐块观察 $m$、$l$ 如何一步步逼近全局值。
""")

def full_softmax(x):
    # 整体 softmax:减 max 再 exp 归一(数值稳定,但要读两遍)
    e = np.exp(x - x.max())
    return e / e.sum()

def online_softmax_trace(x, block):
    # 逐块记录 (m, l) 快照,用于可视化;m:running max, l:running sum
    m = -np.inf
    l = 0.0
    trace = []
    n = len(x)
    for i in range(0, n, block):
        xb = x[i:i + block]                    # 读一个块
        m_new = max(m, xb.max())               # 更新 running max
        l = l * np.exp(m - m_new) + np.exp(xb - m_new).sum()   # 用旧 m 与块内值校正再累加
        m = m_new
        trace.append((m, l, i // block + 1))
    return trace

def online_softmax(x, block):
    # 一趟扫描得到 m/l;最终再读一遍 x 做归一
    m = -np.inf
    l = 0.0
    n = len(x)
    for i in range(0, n, block):
        xb = x[i:i + block]
        m_new = max(m, xb.max())
        l = l * np.exp(m - m_new) + np.exp(xb - m_new).sum()
        m = m_new
    return np.exp(x - m) / l

with st.sidebar:
    st.header("🎛️ 参数")
    block = st.slider("块大小 Bm", 1, 16, 4, 1)
    n = st.slider("序列长度 N", 8, 64, 24, 1)
    mode = st.radio("数值分布", ["正态", "右偏(有尖峰)"])
    st.caption("块越小,在线 softmax 的 m/l 越“勤快”地更新;尖峰分布更能看出校正的过程。")

rng = np.random.default_rng(42)
if mode == "正态":
    x = rng.normal(0, 1, n)
else:
    x = rng.normal(0, 1, n)
    x[n // 2] = 20.0
    x[n // 2 + 1] = 15.0

if block > n:
    block = n

p_full = full_softmax(x)
p_on = online_softmax(x, block)
err = float(np.max(np.abs(p_full - p_on)))
trace = online_softmax_trace(x, block)
nblocks = len(trace)

c1, c2, c3 = st.columns(3)
c1.metric("块数", f"{nblocks}")
c2.metric("最大绝对误差 vs 整体", f"{err:.2e}")
c3.metric("最终 m(全局 max)", f"{trace[-1][0]:.3f}")

bs = [t[2] for t in trace]; ms = [t[0] for t in trace]; ls = [t[1] for t in trace]
fig = go.Figure()
fig.add_trace(go.Bar(x=bs, y=ms, name="running max m", marker_color="#4C78A8"))
fig.add_trace(go.Bar(x=bs, y=ls, name="running sum l", marker_color="#F2C14E", yaxis="y2"))
fig.add_hline(y=max(ms), line_dash="dot", line_color="#4C78A8",
              annotation_text=f"全局 max={max(ms):.3f}")
fig.update_layout(title=f"每处理一个块后,running m / l 的状态(块大小 {block})",
                  xaxis_title="已处理的块序号", yaxis_title="m", height=420,
                  yaxis2=dict(title="l", overlaying="y", side="right"),
                  legend=dict(orientation="h", y=1.12), barmode="overlay",
                  margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(range(n)), y=p_full, mode="lines+markers",
                          name="整体 softmax", line=dict(color="#4C78A8", width=2)))
fig2.add_trace(go.Scatter(x=list(range(n)), y=p_on, mode="markers",
                          name="在线 softmax", marker=dict(color="#E45756", size=5)))
fig2.update_layout(title="整体 vs 在线 softmax 的概率分布(应几乎重合)",
                   xaxis_title="位置 i", yaxis_title="概率", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.caption("💡 观察:m 单调不减(每次取 max),l 会先被校正(乘上旧最大值的衰减系数)再累加当前块;"
           "块越小,中间状态越多、但最终误差同样极小——在线 softmax 与整体 softmax 数值等价。")

st.markdown(r"""
> 💡 **结论**:在线 softmax 通过 $m \\leftarrow \\max(m, \\text{块内max})$、
> $l \\leftarrow l \\cdot e^{m-m_{\\text{new}}} + \\sum e^{x-m_{\\text{new}}}$ 的两行更新,
> 在不重新读回整行的情况下精确算出分母。这就是 FlashAttention 能分块扫描的理论地基。
> 来源:[Milakov & Gimelshein, 2018 (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 7 章 · 第 43 课配套演示")
''')

NB = new_nb("第 43 课 · 在线 Softmax",
            subtitle="softmax 分母必须看完整行,分块怎么办?running max / sum 边走边修正——FlashAttention 的数学地基",
            emoji="🧮")

chapter_cover(NB,
    objectives=[
        "指出 softmax 分母需要全行信息这一分块难题,建立 x/m/l 符号表",
        "复现朴素 softmax 的 exp 溢出问题与两遍 softmax 的稳定性代价",
        "从 Milakov & Gimelshein(arXiv:1805.02867)推导在线 softmax 的 m/l 更新公式,讲清校正系数 e^(m-m_new) 的来历",
        "手工构造小向量,逐行打印每步 m/l 的演化,验证最终等价于整体 softmax",
        "用 torch 实现分块在线 softmax 并与 torch.softmax 数值对比(allclose)",
        "理解在线 softmax 在 FlashAttention 与 vLLM kernel 中的角色",
    ],
    toc=[
        ("直觉:分批数钱也要算总额", "分母必须看完全部,分块怎么破"),
        ("朴素 softmax:溢出与 NaN", "直接 exp 在输入偏大时的灾难"),
        ("两遍 softmax:稳定但读两遍", "先找 max 再 exp 归一"),
        ("在线 softmax:running max/sum", "公式推导 + 校正系数从哪来(论文支撑)"),
        ("逐行推演:打印每一步 m/l", "小向量手推,看 running 状态如何逼近全局"),
        ("数值验证:分块 ≡ 整体", "torch 分块实现 vs torch.softmax,allclose"),
        ("与 FlashAttention / vLLM 的关系", "在线 softmax 是分块 attention 的理论地基"),
        ("小结 + 练习 + 延伸阅读", "要点、动手题、论文链接"),
    ],
    links=[
        ("Online normalizer calculation for softmax (NVIDIA, 2018)", "https://arxiv.org/abs/1805.02867"),
        ("FlashAttention (NeurIPS 2022, 含在线 softmax 用法)", "https://arxiv.org/abs/2205.14135"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.md("## 1. 直觉:分批数钱也要算总额 💰\n\n"
      "你开了一家小店,一天卖了很多货,想把**每笔销售额占全天总销售额的比例**算出来。\n"
      "问题是:你只有到了晚上关店时,才知道**全天总销售额是多少**。可顾客是随到随走的,"
      "总不能把所有账单都堆在桌上、关店后再一次性算吧?\n\n"
      "softmax 就遇到一模一样的困境:它的分母 $\\sum e^{x_i}$ 是**这一整行的总和**,"
      "必须**把整行都看一遍**才知道。而 FlashAttention 偏偏要**分块扫描**、边算边丢——那分母从哪来?\n\n"
      "答案:**边走边修正(running correction)**。先用看到的块维护一个「大致总额」,"
      "每看到新块,就把旧总额按新信息**打个折扣**再继续累加。这就是 **online softmax(在线 softmax)**,"
      "由 NVIDIA 的 [Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867) 提出。\n\n"
      "先建立符号表:\n\n"
      "| 符号 | 含义 | 形状 |\n|---|---|---|\n"
      "| $x$ | 输入向量(一行打分) | $(N,)$ |\n"
      "| $m$ | running max(目前已看到的最大值) | $(1,)$ 标量 |\n"
      "| $l$ 或 $d$ | running sum(校正后的 $\\sum e^{x-m}$) | $(1,)$ 标量 |\n"
      "| $B$ | 分块大小(一次读几个元素) | — |\n\n"
      "> 论文里归一化项记作 $d$(denominator),FlashAttention 与后续文献常用 $l$(log-sum-exp 相关)。本课两者通用。")

NB.code(CPU_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed。有 CUDA 就走真实 GPU——下面的在线/整体 softmax 对比都是 GPU 上真算的。")

# =====================================================================
NB.md("## 2. 朴素 softmax:溢出与 NaN 💥\n\n"
      "先看最简单(也最危险)的 softmax:直接 `exp(x)` 再归一。当输入偏大(比如 1000),"
      "`exp(1000)` 直接溢出成 `inf`,`inf / inf` 就变成 `NaN`——整个输出毁于一旦。")

NB.code(SOFTMAX_NAIVE, "**softmax_naive**:直接 exp 再归一。输入偏大时 `exp` 溢出成 inf,结果变 NaN。")

NB.code(D('''
# 构造一个偏大的输入,演示朴素 softmax 的溢出
x = torch.tensor([1000.0, 1000.0, 999.0, 1000.0])   # 每个元素都接近 1000,exp 会爆炸
print("输入 x       :", x.tolist())
print("softmax_naive:", softmax_naive(x))           # 结果出现 NaN
print("→ exp(1000)=inf,inf/inf=NaN。这就是朴素实现不稳定的根源。")
'''), "⚠️ 看到 NaN 了吗?`exp(1000)` 已超过 float 上限(≈1.8e308)变成 inf。")

# =====================================================================
NB.md("## 3. 两遍 softmax:稳定但读两遍 📖\n\n"
      "修正办法很简单:先减掉最大值 $m=\\max_i x_i$,再 exp。因为 $x_i - m \\le 0$,exp 永远不会溢出:\n\n"
      "$$\\text{softmax}(x)_i = \\frac{e^{x_i - m}}{\\sum_j e^{x_j - m}}$$\n\n"
      "代价是要**读两遍数据**:第一遍找 $m$,第二遍算 exp 和求和。这在大矩阵上意味着 HBM 访存翻倍。")

NB.code(SOFTMAX_TWOPASS, "**softmax_twopass**:先减最大值(第一遍),再 exp 归一(第二遍)。数值稳定,但要读两遍数据。")

NB.code(D('''
print("softmax_twopass:", softmax_twopass(x))   # 数值稳定,正常输出
print("softmax_naive  :", softmax_naive(x))     # 还是 NaN(对照)
print("\\n→ 减掉 max 之后永不溢出,但需要先扫一遍找 max(读两遍)。")
'''), "✅ 减掉 max 之后,Never 溢出。但注意:它需要先扫一遍找 max,数据要读两遍。")

# =====================================================================
NB.md("## 4. 在线 softmax:running max / sum 🚂\n\n"
      "两遍 softmax 要读两遍数据,可 FlashAttention 只想**读一遍**、分块扫描。在线 softmax 用两个 running 状态解决。\n\n"
      "论文 Algorithm 3([Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867)):\n\n"
      "$$ m_0 = -\\infty, \\quad l_0 = 0 $$\n"
      "$$ m_j = \\max(m_{j-1},\\, x_j), \\qquad l_j = l_{j-1}\\,e^{m_{j-1} - m_j} + e^{x_j - m_j} $$\n\n"
      "**Theorem 1**:第 1–6 行计算得 $m_V = \\max_k x_k$ 与 $l_V = \\sum_j e^{x_j - m_V}$——与整体 softmax 的分母完全一致。\n\n"
      "**校正系数的来历**:当新元素 $x_j$ 让 $m$ 增大到 $m_j$ 时,旧累积的 $l_{j-1}$ 是以 $m_{j-1}$ 为基准的"
      "($e^{x - m_{j-1}}$)。要换成以 $m_j$ 为基准,必须整体乘上 $e^{m_{j-1} - m_j}$。这就是「打个折扣」。\n\n"
      "论文还指出,该更新可写成**结合律 + 交换律**的二元算子,从而能并行归约:\n\n"
      "$$ (m_i, l_i) \\oplus (m_j, l_j) = \\big(\\max(m_i,m_j),\\; l_i e^{m_i - \\max} + l_j e^{m_j - \\max} \\big) $$\n\n"
      "这让整行可被拆成多块并行处理——正是 GPU kernel 需要的性质。")

NB.code(SOFTMAX_ONLINE, "**softmax_online**:一趟扫描,维护 running max(m) 与 running sum(l)。每看到一个块就用「旧 m 与当前块最大值」校正已累积的 l。")

# =====================================================================
NB.md("## 5. 逐行推演:打印每一步 m/l 🧮\n\n"
      "用手工构造的小向量,逐块打印 $m$ 与 $l$ 的演化,亲眼看到「校正 + 累加」的过程。"
      "输入取 $x=[3,\\,1,\\,-2,\\,4]$,块大小 $B=2$。")

NB.code(D('''
# 逐块推演:记录每个块处理完后的 (m, l)
def online_trace(x, block):
    """逐块返回 (m, l, 块序号) 快照,便于观察 running 状态演化。"""
    m = torch.tensor(float("-inf"))            # running max 初始为 -inf
    l = torch.tensor(0.0)                      # running sum 初始为 0
    trace = []
    for i in range(0, x.shape[-1], block):     # 沿 x 方向分块扫描
        xb = x[i:i + block]                    # 读一个块 (B,)
        m_new = torch.maximum(m, xb.max())     # 更新 running max
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new).sum()   # 校正旧和 + 累加当前块
        m = m_new
        trace.append((float(m), float(l), i // block + 1))
        print(f"  块 {i//block+1}: xb={xb.tolist()}, m={float(m):.3f}, l={float(l):.3f}")
    return trace

x = torch.tensor([3.0, 1.0, -2.0, 4.0])        # 手算友好的小向量
print(f"输入 x = {x.tolist()}, 全局 max = {x.max().item()}")
trace = online_trace(x, block=2)               # 每块 2 个元素
print(f"\\n最终 m = {trace[-1][0]:.3f}(应等于全局 max={x.max().item():.3f}),l = {trace[-1][1]:.3f}")
print(f"在线 softmax = {torch.exp(x - trace[-1][0]) / trace[-1][1]}")

# 对照整体 softmax
full = torch.softmax(x, dim=-1)
print(f"整体 softmax = {full}")
print(f"两者最大误差 = {(torch.exp(x - trace[-1][0]) / trace[-1][1] - full).abs().max().item():.2e}")
'''), "🎯 看演化的两个规律:① $m$ 只升不降(每次取 max);② 当新块出现更大的值时,$l$ 先被校正(乘衰减系数)再累加。最终 $m$、$l$ 与整体一致。")

# =====================================================================
NB.md("## 6. 数值验证:分块 ≡ 整体 ✅\n\n"
      "数学上等价还不够,得在 **GPU** 上把「分块在线 softmax」和「完整 softmax」摆到同一起跑线真算一遍,"
      "并用 `torch.allclose` 断言两者一致。输入刻意加进**尖峰**(每行有值到 +40),"
      "正是要考验 running max 的数值稳定性——等效于 FlashAttention 处理「注意力打分里有一个很大的值」。")

NB.code(D('''
# GPU 上验证「分块在线 softmax ≡ 完整 softmax」,块大小从 64 试到 2048
def softmax_online_batched(x, block):
    """逐行分块在线 softmax:x:(B,N)。返回与 torch.softmax 同形状。"""
    B, N = x.shape                            # B 行、N 列
    m = torch.full((B, 1), float("-inf"), device=x.device)   # running max,形状 (B,1)
    l = torch.zeros((B, 1), device=x.device)                 # running sum,形状 (B,1)
    for i in range(0, N, block):              # 沿列分块扫描
        xb = x[:, i:i + block]                # 读一个块 (B, B)
        m_new = torch.maximum(m, xb.max(dim=-1, keepdim=True).values)   # 更新 max (B,1)
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new).sum(dim=-1, keepdim=True)   # 校正+累加
        m = m_new
    return torch.exp(x - m) / l               # 最终归一(再读一遍 x)

torch.manual_seed(1)
xbig = torch.randn(64, 4096, device=dev) * 5.0                              # 64 行 × 4096 列
xbig[torch.arange(64), torch.randint(0, 4096, (64,), device=dev)] += 40.0   # 每行一个尖峰
print(f"输入: {tuple(xbig.shape)} @ {dev},每行含尖峰(检验数值稳定)。")
full = torch.softmax(xbig, dim=-1)            # 整体 softmax 作参照 (64, 4096)
for block in [64, 128, 512, 2048]:            # 不同块大小
    on = softmax_online_batched(xbig, block)  # 分块在线 softmax
    same = bool(torch.allclose(full, on, atol=1e-5, rtol=1e-5))
    err = (full - on).abs().max().item()      # 最大绝对误差
    print(f"  block={block:5d}:  allclose={same},  最大误差={err:.2e}")
del xbig; torch.cuda.empty_cache()
'''), "🎯 GPU 真相:四个块大小全部 allclose=True,最大误差 ~1e-6——分块在线 softmax 与整体 softmax 数值一致,分块正确性实锤。")

# =====================================================================
NB.md("## 7. 与 FlashAttention / vLLM 的关系 🔗\n\n"
      "在线 softmax 不是个孤立的技巧,它是**分块注意力一切正确的根基**:\n\n"
      "1. **FlashAttention 的前向**(论文 Algorithm 1)在每读入一个 $K,V$ 块后,都用这两行更新 running 状态:\n\n"
      "$$ m^{\\text{new}} = \\max(m,\\, \\tilde m), \\qquad l^{\\text{new}} = l\\,e^{m - m^{\\text{new}}} + \\tilde l\\,e^{\\tilde m - m^{\\text{new}}} $$\n\n"
      "并同时用同一系数缩放输出累加器 $O$(第 42 课已实现);\n"
      "2. **vLLM 的 kernel**(flash_attn / triton_attn)里,每个线程块都维护自己的 $m_i, l_i, acc$(寄存器/SRAM),"
      "沿 $K/V$ 分块循环更新——本课第 6 节的 `softmax_online_batched` 就是这个逻辑的逐行版;\n"
      "3. 工程路径参考:`vendor/vllm/vllm/v1/attention/backends/flash_attn.py` 与 triton 教程的 `06-fused-attention.py`。\n\n"
      "> 📄 简言之:第 42 课讲了「为什么要分块」,本课讲了「分块后分母怎么算」。两者合起来就是 FlashAttention。")

# =====================================================================
NB.md("## 8. 🖥️ Streamlit 动态演示:拖块大小看状态 🎛️\n\n"
      "运行同目录下的 `app_43_online_softmax.py`,拖动**块大小 / 序列长度**滑块,逐块看 $m$、$l$ 如何演化,"
      "并实时对比整体与在线 softmax 的概率分布:\n\n"
      "```bash\nD:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_43_online_softmax.py\n```\n\n"
      "浏览器打开 **http://localhost:8501**。建议把块大小从 1 拖到 16,观察块数变化与中间状态;"
      "再切换到「右偏(有尖峰)」分布,看 $l$ 在校正时的「下跌再猛涨」。完整源码如下(与同目录 app 一字不差):")

NB.code(f"%%writefile {APP_FILE}\n" + APP_43, "📜 这就是 app_43_online_softmax.py 的完整源码,notebook 与 app 共用同一套 online 逻辑,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "softmax 分母需要全行信息,与分块扫描矛盾——在线 softmax 用 running max/sum 解决",
        "朴素 softmax 直接 exp 会溢出成 NaN;减 max 后数值稳定",
        "两遍 softmax 稳定但要读两遍数据;在线 softmax 一趟扫描只需读约 3 次/元素",
        "核心更新:m←max(m,块内max),l←l·e^(m-m_new)+Σe^(x-m_new);系数 e^(m-m_new) 即校正因子",
        "该更新是结合律+交换律的二元算子,可并行归约(论文 Theorem 1 保证最终等于整体分母)",
        "GPU 实测:各块大小下在线 softmax 与整体误差 ~1e-6、allclose=True——分块正确性实锤",
        "在线 softmax 是 FlashAttention 与 vLLM 分块 attention kernel 的数学地基",
    ],
    practice=[
        "把 online_trace 改成记录每一步的校正系数 e^(m-m_new),观察它随块变化的规律",
        "给 softmax_online_batched 加一个 bf16 版本,比较 fp32 与 bf16 下的误差",
        "在 2D 矩阵上实现逐行 online softmax,并用 torch.softmax(dim=-1) 验证",
        "推导:为什么 m 单调不减,而 l 在校正时可能下降?写出那一步的数值例子",
        "读论文 Algorithm 3 与 Theorem 1,用自己的话证明 l_V=Σ e^(x_j-m_V)",
    ],
    links=[
        ("Online normalizer calculation for softmax (2018)", "https://arxiv.org/abs/1805.02867"),
        ("FlashAttention (2022, 用在线 softmax 做分块)", "https://arxiv.org/abs/2205.14135"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH07) / "43_online_softmax.ipynb"))

app_path = Path(CH07) / APP_FILE
app_path.write_text(APP_43 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
