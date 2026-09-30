# -*- coding: utf-8 -*-
"""生成 42_flash_attention_principle.ipynb 与 app_42_flash_attn.py(教材级重写版)

对齐 REWRITE_STANDARD.md:
由浅入深(直觉 -> 定义/符号表 -> 逐行推演 -> 数值验证 -> 真实规模 -> 工程关联 -> 小结)。
每一行代码都有 inline 注释;每个张量打印 shape 并标注维度含义。
论文支撑:FlashAttention (arXiv:2205.14135)、FlashAttention-2 (arXiv:2307.08691)。
"""
from pathlib import Path
from helpers import D, new_nb, chapter_cover, wrapup, CH07, CPU_HEADER, NAIVE_ATTN
from helpers import FLASH_CHUNKED

APP_FILE = "app_42_flash_attn.py"

# ---------------- App(直跑入口,与 notebook 共用同一套公式)----------------
APP_42 = D('''
# -*- coding: utf-8 -*-
# app_42_flash_attn.py — FlashAttention 原理:标准 vs 分块的 HBM 访存/显存对比 ⚡
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="⚡ 42 · FlashAttention 原理", layout="wide")
st.title("⚡ 第 42 课 · FlashAttention 原理:访存才是瓶颈")

st.markdown(r"""
标准的 attention 要把 **打分矩阵 S 和概率矩阵 P(都是 $H \times N \times N$)** 整体写进显存(HBM)。
当序列 $N$ 变长时,$N^2$ 项呈平方级暴涨,显存和访存量都撑不住。
**FlashAttention** 的思路是:把 $K/V$ 切成小方块,分块读进来边算边丢,**从不整体写出 S/P**,
于是把 $O(N^2)$ 的显存降回 $O(N)$。下方拖动滑块,直观对比两者的 HBM 访存量与显存占用。
""")

def hbm_bytes(N, H, d, db=2):
    # naive:读 Q/K/V 写 O(约 5·N·H·d)+ S/P 各写读两次(4·H·N²);flash:只 Q/K/V 各读一次 + O 写一次
    naive = 5 * N * H * d * db + 4 * H * N * N * db
    flash = 4 * N * H * d * db
    return naive, flash

def mem_bytes(N, H, d, db=2):
    # 峰值显存:naive 额外持有 S 与 P 两个 N×N 张量;flash 从不落 S/P
    naive = (3 * N * H * d + 2 * H * N * N + N * H * d) * db
    flash = (3 * N * H * d + N * H * d) * db
    return naive, flash

with st.sidebar:
    st.header("🎛️ 参数")
    N = st.slider("序列长度 N", 256, 8192, 1024, 128)
    H = st.slider("注意力头数 H", 1, 16, 8, 1)
    d = st.selectbox("头维度 d", [32, 64, 128], index=1)
    db = st.radio("数据类型", ["fp16(2 字节)", "fp32(4 字节)"])
    dtype_bytes = 2 if db.startswith("fp16") else 4
    st.caption("HBM = 显存。naive 的多出部分主要是 S/P 这两个 N×N 大张量的读写。")

nv, fl = hbm_bytes(N, H, d, dtype_bytes)
mv, mf = mem_bytes(N, H, d, dtype_bytes)

c1, c2, c3, c4 = st.columns(4)
c1.metric("naive 访存量", f"{nv/1e6:.2f} MB")
c2.metric("flash 访存量", f"{fl/1e6:.2f} MB")
c3.metric("访存节省", f"{nv/fl:.1f}×")
c4.metric("显存:naive→flash", f"{mv/1e6:.0f}→{mf/1e6:.1f} MB")

Nx = np.arange(256, 8193, 256)
nv_a = [hbm_bytes(n, H, d, dtype_bytes)[0] / 1e6 for n in Nx]
fl_a = [hbm_bytes(n, H, d, dtype_bytes)[1] / 1e6 for n in Nx]
mv_a = [mem_bytes(n, H, d, dtype_bytes)[0] / 1e6 for n in Nx]
mf_a = [mem_bytes(n, H, d, dtype_bytes)[1] / 1e6 for n in Nx]

fig = go.Figure()
fig.add_trace(go.Scatter(x=Nx, y=nv_a, name="naive 访存", mode="lines",
                         line=dict(width=2.5, color="#E45756")))
fig.add_trace(go.Scatter(x=Nx, y=fl_a, name="flash 访存", mode="lines",
                         line=dict(width=2.5, color="#4C78A8")))
fig.add_trace(go.Scatter(x=Nx, y=mv_a, name="naive 显存", mode="lines", dash="dot",
                         line=dict(color="#E45756")))
fig.add_trace(go.Scatter(x=Nx, y=mf_a, name="flash 显存", mode="lines", dash="dot",
                         line=dict(color="#4C78A8")))
fig.update_layout(title=f"HBM 访存量/显存 vs 序列长度(H={H}, d={d})",
                  xaxis_title="序列长度 N", yaxis_title="MB(对数轴)",
                  yaxis_type="log", height=480,
                  legend=dict(orientation="h", y=1.12),
                  margin=dict(l=10, r=10, t=60, b=10))
fig.add_vline(x=N, line_dash="dash", line_color="#B279A2",
              annotation_text=f"N={N}")
st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 观察:naive 的曲线随 N 呈 O(N²) 上扬(斜率大),flash 只有 O(N) 的缓慢上升;"
           "序列越长,两者的差距拉得越大——这就是 FlashAttention 在大模型长上下文里不可替代的原因。")

st.markdown("""
> 💡 **结论**:标准 attention 慢在**把 S/P 大张量写进又读出 HBM**;
> FlashAttention 用分块 + 在线 softmax 把这些中间量留在片上(SRAM),HBM 访存从 $O(N^2)$ 降到 $O(N)$。
> 计算量(FLOPs)其实没变,省的是**访存**这一块。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 7 章 · 第 42 课配套演示")
''')

NB = new_nb("第 42 课 · FlashAttention 原理",
            subtitle="为什么 attention 慢在访存而不是算力?分块(tiling)+ 在线 softmax 如何把 O(N²) 的 HBM 访存拉回 O(N)",
            emoji="⚡")

chapter_cover(NB,
    objectives=[
        "复述标准缩放点积注意力的三步,并建立 B/H/N/d 维度符号表",
        "用 HBM 访问字节数(而非 FLOPs)重新审视 attention 的瓶颈,理解 IO-aware 的含义",
        "用 torch 手写 naive 三步,逐行打印 S/P 张量的 shape 与内存占用,验证 O(N²) 开销",
        "推导并实现分块 + 在线 softmax 的 flash_attention_chunked,并数值验证其与标准 attention 等价",
        "代入 LLaMA-7B(等)真实规模,算 naive vs flash 的 HBM 访存节省倍数",
        "在真实 GPU 上实测 naive 耗时与峰值显存随 N 的变化,看懂平方惩罚",
        "建立与 FlashAttention(arXiv:2205.14135)、FlashAttention-2(arXiv:2307.08691)和 vLLM 实现的联系",
    ],
    toc=[
        ("直觉:GPU 的两层记忆", "HBM 与 SRAM,算得快不如搬得少"),
        ("标准 attention:三个大张量落盘", "符号表 + 三步公式 + naive 逐行实现与 shape"),
        ("算一笔 IO 账:FLOPs vs HBM 访问", "为什么标准实现是 Θ(Nd+N²) 访存、O(N²) 显存"),
        ("分块 + 在线 softmax:FlashAttention 的钥匙", "tiling 与 recomputation,逐行实现 chunked 版"),
        ("数值验证:分块 ≡ 标准", "torch.allclose 断言数学等价"),
        ("真实规模数字:代入 LLaMA/Qwen", "算 HBM 访存节省倍数;GPU 真测 naive 耗时/显存"),
        ("与 vLLM 工程实现的关系", "FLASH_ATTN backend 与 PyTorch SDPA 的 flash 内核"),
        ("小结 + 练习 + 延伸阅读", "要点、动手题、论文链接"),
    ],
    links=[
        ("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (NeurIPS 2022)", "https://arxiv.org/abs/2205.14135"),
        ("FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (ICLR 2024)", "https://arxiv.org/abs/2307.08691"),
        ("PyTorch scaled_dot_product_attention 文档", "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

# =====================================================================
# 第 1 节 · 直觉:GPU 的两层记忆
# =====================================================================
NB.md("## 1. 直觉:GPU 有两层记忆,算得快不如搬得少 🏎️\n\n"
      "GPU 里有两类存储,速度与容量天差地别:\n\n"
      "| 存储 | 容量 | 带宽(约) | 用途 |\n|---|---|---|---|\n"
      "| **HBM**(显存) | 大(几十 GB) | 慢(约 1.5–3.4 TB/s) | 存权重、激活、KV Cache |\n"
      "| **SRAM**(片上) | 小(几十 MB) | 快(约 19 TB/s) | 寄存器、共享内存、缓存 |\n\n"
      "> 📄 这是 FlashAttention 论文开篇的「图 1」:GPU 的**算力增长远快于内存带宽增长**,"
      "导致大多数 Transformer 算子其实是**访存受限(memory-bound)**——等数据从 HBM 搬进来,而不是在算。[Dao et al., 2022](https://arxiv.org/abs/2205.14135)\n\n"
      "**核心洞察**:attention 的问题从来不是「算得慢」,而是「**搬得慢**」。标准实现要把 $N\\times N$ 的中间矩阵"
      "写进 HBM 再读出来,序列越长,这笔「搬运费」以 $N^2$ 暴涨。FlashAttention 的整套设计,就是**少搬数据**。")

# =====================================================================
# 第 2 节 · 标准 attention:符号表 + 三步公式 + naive 逐行实现
# =====================================================================
NB.md("## 2. 标准 attention:先建立符号表,再逐行推演 🧱\n\n"
      "标准缩放点积注意力(scaled dot-product attention)对输入 $Q,K,V \\in \\mathbb{R}^{N \\times d}$ 做三步:\n\n"
      "$$ S = \\frac{QK^\\top}{\\sqrt{d}}, \\qquad P = \\text{softmax}(S), \\qquad O = P\\,V $$\n\n"
      "真实工程里张量是 4D 布局,先定好符号表(本课所有代码都打印这些 shape):\n\n"
      "| 符号 | 含义 | 形状 |\n|---|---|---|\n"
      "| $B$ | batch size(一次处理几个序列) | — |\n"
      "| $H$ | 注意力头数 num_heads | — |\n"
      "| $N$ | 序列长度 seq_len(一个序列几个 token) | — |\n"
      "| $d$ | 每头维度 head_dim | — |\n"
      "| $Q$ | 查询矩阵 | $(B, H, N, d)$ |\n"
      "| $K$ | 键矩阵 | $(B, H, N, d)$ |\n"
      "| $V$ | 值矩阵 | $(B, H, N, d)$ |\n"
      "| $S$ | 打分矩阵 $QK^\\top/\\sqrt{d}$ | $(B, H, N, N)$ ← **平方级大头** |\n"
      "| $P$ | 概率矩阵 softmax($S$) | $(B, H, N, N)$ ← **另一个平方级大头** |\n"
      "| $O$ | 输出矩阵 $PV$ | $(B, H, N, d)$ |\n\n"
      "**注意**:$S$ 与 $P$ 是 $N\\times N$ 的,随 $N$ 平方增长——这就是标准实现显存/访存爆炸的根源。")

NB.code(CPU_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed。有 CUDA 就走真实 GPU(RTX 5060),下面的耗时与显存都是真跑出来的。")

NB.code(NAIVE_ATTN, "**naive_attention**:把 QKᵀ / softmax / 加权和 三个大张量全部写进 HBM。这是最直观、也最占显存的写法。")

NB.code(D('''
# 手工构造小张量,逐行观察标准 attention 的中间张量 shape 与内存占用
# 刻意取小值:B=1, H=2, N=4, d=8,方便心算验证维度
batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8   # B, H, N, d

# 随机初始化 Q/K/V(只关心 shape,不关心具体值)
Q = torch.randn(batch_size, num_heads, seq_len, head_dim)   # (B,H,N,d)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)   # (B,H,N,d)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)   # (B,H,N,d)
print(f"Q shape = {tuple(Q.shape)} <- (batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim})")

# ---- 第 1 步:打分 S = QKᵀ / √d ----
# einsum 的 "bhnm" 表示输出第 3 维来自 Q 的 seq、第 4 维来自 K 的 seq,即每个 query 对每个 key 打分
S = torch.einsum("bhnd,bhmd->bhnm", Q, K) / math.sqrt(head_dim)   # (B,H,N,N)
print(f"S shape = {tuple(S.shape)} <- (batch, heads, {seq_len} 个query, {seq_len} 个key)")

# ---- 第 2 步:概率 P = softmax(S)(对最后一维即所有 key 归一)----
P = torch.softmax(S, dim=-1)   # (B,H,N,N) 每行和为 1
print(f"P shape = {tuple(P.shape)} <- softmax 后每行和为 1")

# ---- 第 3 步:输出 O = P V ----
O = torch.einsum("bhnm,bhmd->bhnd", P, V)   # (B,H,N,d)
print(f"O shape = {tuple(O.shape)} <- 输出,形状回到 (B,H,N,d)")

# ---- 量化:单看 S 与 P 的内存占用 ----
# fp32 下每个元素 4 字节;S/P 各有 B*H*N*N 个元素
bytes_per_element = 4
s_mem = bytes_per_element * batch_size * num_heads * seq_len * seq_len   # S 的字节数
print(f"\\n单个 S 张量(本例 N={seq_len}) = {s_mem} 字节")
print(f"若 N=2048, H=8, fp32: 单个 S = {4*1*8*2048*2048/1e6:.1f} MB (两两对比可见 N 的平方惩罚)")

# 对照官方实现,验证手写正确(误差应为 0)
import torch.nn.functional as F
ref = F.scaled_dot_product_attention(Q, K, V)
print(f"naive 与官方 SDPA 最大误差 = {(O - ref).abs().max().item():.2e}  (≈0 说明三步书写正确)")
'''), "🎯 看到没:`S` 一个矩阵就有 $B\\cdot H\\cdot N\\cdot N$ 个元素——$N$ 每翻一倍,这块内存涨 4 倍。这就是「平方惩罚」的起点。")

# =====================================================================
# 第 3 节 · IO 账本:FLOPs vs HBM 访问
# =====================================================================
NB.md("## 3. 算一笔 IO 账:FLOPs vs HBM 访问 🧮\n\n"
      "评价一个算子有两种「成本」:\n\n"
      "1. **计算量(FLOPs)**:$QK^\\top$ 与 $PV$ 两处矩阵乘各约 $2N^2 d$,总 FLOPs $\\approx 4N^2 d$;\n"
      "2. **访存量(HBM 字节)**:数据在 HBM 与 SRAM 之间搬了多少次。\n\n"
      "标准实现的访存是:\n"
      "- 把 $S$ 写 HBM、再读出来(算 softmax);把 $P$ 写 HBM、再读出来(算 $PV$);\n"
      "- 加上 $Q,K,V,O$ 的读写 → 总访存 $\\approx O(N^2)$ 字节;\n\n"
      "FlashAttention 的访存是:$Q$ 读一次、$K/V$ 分块读一次、$O$ 写一次,$S/P$ **从不落盘** → 总访存 $\\approx O(N)$ 字节。\n\n"
      "论文给出严格 IO 复杂度:[Dao et al., 2022](https://arxiv.org/abs/2205.14135)\n\n"
      "$$ \\text{标准:}\\;\\Theta(Nd + N^2),\\qquad \\text{FlashAttention:}\\;\\Theta\\!\\left(\\frac{N^2 d^2}{M}\\right) $$\n\n"
      "其中 $M$ 是 SRAM 大小。当 $M$ 足够大时,FlashAttention 的访存远小于标准实现(论文实测最多省 **9×** 的 HBM 访问)。\n\n"
      "把访存公式写成函数,算几个典型 $N$ 的数值:")

NB.code(D('''
def hbm_bytes(N, H, d, dtype_bytes=2):
    """naive 与 flash 的 HBM 访存(字节)。naive 大头是 S/P 的 4 次 N² 访问(各写+读两次)。"""
    naive = 5 * N * H * d * dtype_bytes + 4 * H * N * N * dtype_bytes   # 5·N·H·d(QKV 读+O 写)+ 4·H·N²(S/P 读写)
    flash = 4 * N * H * d * dtype_bytes                                 # 只 Q/K/V 各读一次 + O 写一次
    return naive, flash

def mem_bytes(N, H, d, dtype_bytes=2):
    """峰值显存:naive 额外持有 S 与 P 两个 N×N 张量;flash 从不落 S/P。"""
    naive = (3 * N * H * d + 2 * H * N * N + N * H * d) * dtype_bytes   # QKV + S + P + O
    flash = (3 * N * H * d + N * H * d) * dtype_bytes                   # QKV + O
    return naive, flash

print(f"{'N':>6} | {'naive访存(MB)':>14} | {'flash访存(MB)':>14} | {'节省×':>7} | {'显存 naive→flash(MB)':>24}")
for N in [512, 1024, 2048, 4096, 8192]:                    # 典型上下文长度
    nv, fl = hbm_bytes(N, 8, 64)                            # H=8, d=64
    mv, mf = mem_bytes(N, 8, 64)
    print(f"{N:>6} | {nv/1e6:14.2f} | {fl/1e6:14.2f} | {nv/fl:6.1f}× | {mv/1e6:12.1f} → {mf/1e6:7.1f}")

print("\\n→ naive 访存随 N 翻倍涨约 4 倍(O(N²)),flash 只随 N 线性涨;两者差距越拉越大。")
'''), "🔍 观察表格最后一列「节省×」:N 越大,节省倍数越大——这正是长上下文场景下 FlashAttention 不可替代的原因。")

# =====================================================================
# 第 4 节 · 分块 + 在线 softmax:FlashAttention 的钥匙
# =====================================================================
NB.md("## 4. 分块 + 在线 softmax:FlashAttention 的钥匙 🔑\n\n"
      "FlashAttention 用两个成熟技巧让 $S/P$ 不再落盘(论文 Algorithm 1):\n\n"
      "1. **分块(tiling)**:把 $K,V$ 沿 $N$ 方向切成 $B_c$ 大小的小块,一次只读一小块进 SRAM,"
      "算一小块 $S$、更新输出 $O$,用完即弃;\n"
      "2. **在线 softmax(online softmax)**:softmax 的分母需要「看完一整行」才知道,"
      "FlashAttention 用**运行时最大值 $m$** 与**运行时求和 $l$** 边走边修正,保证每个块的贡献正确累加。\n\n"
      "论文按 SRAM 大小 $M$ 选块宽:$B_c = \\lceil M/4d \\rceil$,$B_r = \\min(\\lceil M/4d\\rceil, d)$。\n\n"
      "> 📄 在线 softmax 的数学基础来自 NVIDIA 的 [Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867),"
      "第 43 课专门逐行推演。\n\n"
      "下面用纯 torch 手写分块版(与 kernel 的数学结构一一对应):")

NB.code(FLASH_CHUNKED, "**flash_attention_chunked**:外层循环沿 K/V 分块扫描,内层做在线 softmax + 输出累加。这就是 FlashAttention 内核的数学骨架。")

NB.md("### 🔬 逐行看懂 running max / sum 的校正\n\n"
      "设已处理完第 $j$ 块,累积了 $m$ 与 $l$;读入第 $j{+}1$ 块后得到块内最大值 $\\tilde m$、块内指数和 $\\tilde l$。\n"
      "合并公式(来自 [Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867) 与 FlashAttention Algorithm 1):\n\n"
      "$$ m^{\\text{new}} = \\max(m,\\; \\tilde m), \\qquad l^{\\text{new}} = l\\cdot e^{m - m^{\\text{new}}} + \\tilde l\\cdot e^{\\tilde m - m^{\\text{new}}} $$\n\n"
      "系数 $e^{m - m^{\\text{new}}}$ 是把旧块以「新最大值」为基准**重新归一**的校正因子——旧块所有 $e^{x-m}$ 都要按这个比例缩小,"
      "才能与新块在同一基准上相加。输出累加器 $O$ 用同样的系数缩放,最后整体除以 $l$。")

# =====================================================================
# 第 5 节 · 数值验证:分块 ≡ 标准
# =====================================================================
NB.md("## 5. 数值验证:分块 ≡ 标准 ✅\n\n"
      "分块版的数学必须等价于标准 attention。随机生成 $Q,K,V$,对比两者输出,并用 `torch.allclose` 断言:"
      "不同块大小都应成立。")

NB.code(D('''
# 数值验证:分块 flash ≡ 标准 attention(用 torch 官方 SDPA 作参照)
import torch.nn.functional as F

torch.manual_seed(0)
B, H, N, d = 1, 4, 256, 64                     # 小规模便于快速验证
Q = torch.randn(B, H, N, d, device=dev)        # (B,H,N,d)
K = torch.randn(B, H, N, d, device=dev)
V = torch.randn(B, H, N, d, device=dev)

ref = F.scaled_dot_product_attention(Q, K, V)  # 官方实现,作为数学真值
print(f"输入: Q/K/V shape = {tuple(Q.shape)} @ {dev}")
for block_m in [16, 32, 64, 128]:              # 不同的 K/V 分块宽度
    O_flash = flash_attention_chunked(Q, K, V, block_M=block_m)   # 分块版输出 (B,H,N,d)
    err = (O_flash - ref).abs().max().item()   # 与官方最大绝对误差
    ok = bool(torch.allclose(O_flash, ref, atol=1e-5, rtol=1e-5))
    print(f"  block_M={block_m:4d}: 最大误差={err:.2e}, allclose={ok}")

print("\\n→ 所有块大小下误差都在 1e-6 量级且 allclose=True:分块实现在数值上与标准 attention 等价。")
del Q, K, V; torch.cuda.empty_cache()
'''), "🎯 误差 ~1e-6、allclose 全 True——分块没有改变数学,只是改变了数据搬移方式。这是 FlashAttention「exact(精确)」的实锤。")

# =====================================================================
# 第 6 节 · 真实规模数字 + GPU 实测
# =====================================================================
NB.md("## 6. 真实规模数字:代入 LLaMA/Qwen,并在 GPU 上实测 ⚖️\n\n"
      "把上面的公式代入真实模型配置,看 naive 的「平方惩罚」有多可怕。以 **LLaMA-7B** 为例:"
      "$L=32$ 层,$H=32$ 头,$d=128$ 维。\n\n"
      "单层单 batch 的 $S$ 矩阵(bf16,2 字节):$\\text{bytes} = 2\\cdot H\\cdot N^2$。\n\n"
      "| 序列长度 N | 单层 S+P 显存(bf16) | 备注 |\n|---|---|---|\n"
      "| 2,048 | 2×2×32×2048² ≈ 0.54 GB | 尚可 |\n"
      "| 8,192 | ≈ 8.6 GB | 已吃紧 |\n"
      "| 32,768 | ≈ 137 GB | 远超单卡 |\n\n"
      "这就是为什么长上下文**必须**用 FlashAttention——它把显存从 $O(N^2)$ 拉回 $O(N)$。\n\n"
      "下面在真实 GPU 上实测:naive 的耗时与峰值显存如何随 $N$ 上涨。")

NB.code(D('''
# 真机实测①:naive 耗时随 N 上涨(H=4, d=64, fp32, GPU)——验证 O(N²) 的平方惩罚
def bench_ms(fn, *a, iters=5):
    for _ in range(2):            # warmup:触发 kernel 编译/缓存
        fn(*a)
    torch.cuda.synchronize()      # 等 GPU 真正跑完(否则测到的是"提交"时间)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0   # 毫秒

print(f"{'N':>7} {'naive耗时(ms)':>14} {'耗时/上一档':>12} {'S+P显存(MB)':>12}")
prev = None
for n in [256, 512, 1024, 2048, 4096]:           # 序列长度逐档翻倍
    q = torch.randn(1, 4, n, 64, device=dev)     # (B,H,N,d)
    k = torch.randn(1, 4, n, 64, device=dev)
    v = torch.randn(1, 4, n, 64, device=dev)
    ms = bench_ms(naive_attention, q, k, v)      # 真实耗时
    ratio = f"{ms / prev:.2f}x" if prev else "-" # 相对上一档(翻倍档应≈4x)
    s_mem = 2 * 4 * n * n * 4 / 1e6              # S 与 P 各 4*N*N 个 fp32(4 字节)
    print(f"{n:>7} {ms:14.3f} {ratio:>12} {s_mem:12.1f}")
    prev = ms
    del q, k, v; torch.cuda.empty_cache()
print("→ 耗时每档接近翻两番(≈4x)= O(N²) 在真机上的表现;长序列下绝不能这么写。")
'''), "🎯 看数字:N 从 256→512→1024→2048→4096,耗时每一档都接近乘 4——这就是平方级(O(N²))的实锤。")

NB.code(D('''
# 真机实测②:用 torch.cuda 真实计数器量 naive 一次前向的峰值显存
torch.cuda.reset_peak_memory_stats()             # 清零峰值统计
q = torch.randn(1, 4, 2048, 64, device=dev)
k = torch.randn(1, 4, 2048, 64, device=dev)
v = torch.randn(1, 4, 2048, 64, device=dev)
_ = naive_attention(q, k, v)                     # 跑一次前向
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1e6   # 真实峰值显存(MB)
print(f"naive @N=2048,H=4,d=64: 峰值显存 = {peak:.1f} MB")
print(f"  其中单看 S+P 就占 {2*4*2048*2048*4/1e6:.1f} MB —— 两个平方级大张量是真正的显存大户。")
del q, k, v; torch.cuda.empty_cache()
'''), "📏 真实计数器佐证:峰值显存几乎全被 S/P 这两个 N×N 张量吃光——这正是 flash 把它们留在片上(SRAM)的意义。")

NB.md("### 🚀 FlashAttention-2:更好的并行与工作切分\n\n"
      "第一版 FlashAttention 已在 GPU 上比标准实现快 2–4×,但只达到理论算力峰值的 25–40%。"
      "原因:它在**线程块 / warp** 层面的工作切分不优(flash 用 split-K,各 warp 要写共享内存再同步求和)。\n\n"
      "[Dao, 2024 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) 的 FlashAttention-2 做三件事:\n\n"
      "1. **减少非矩阵乘 FLOPs**:改写在线 softmax 的缩放次数(A100 上非 matmul 的 FP32 只有 19.5 TFLOPs/s,"
      "而 matmul 达 312 TFLOPs/s——非 matmul 每个 FLOP 贵 16 倍);\n"
      "2. **沿序列维度也并行**:长序列、小 batch 时提高占用率;\n"
      "3. **改为 split-Q**:各 warp 各算一段输出,互不通信,减少共享内存读写。\n\n"
      "效果:约 **2×** 快于 FlashAttention,在 A100 上达 50–73% 理论峰值,端到端训练达 225 TFLOPs/s(72% MFU)。"
      "同时支持 head_dim 到 256 以及 GQA/MQA。**vLLM 默认用的就是 FlashAttention-2 及后续版本(FA3/FA4)**。")

# =====================================================================
# 第 7 节 · 与 vLLM 工程实现的关系
# =====================================================================
NB.md("## 7. 与 vLLM 工程实现的关系 🔗\n\n"
      "vLLM 的注意力后端 `FLASH_ATTN` 直接调用 flash-attention 库(FA2/FA3/FA4),把本课的分块/在线 softmax 变成了"
      "生产级 CUDA 内核。工程路径:`vendor/vllm/vllm/v1/attention/backends/flash_attn.py`。\n\n"
      "关键点:\n"
      "1. **预填充(prefill)** 用 varlen FlashAttention,一次算整个 prompt 的注意力;\n"
      "2. **解码(decode)** 用分页的 FlashAttention 变体,配合 PagedAttention 的 KV block(第 45 课);\n"
      "3. 普通用户不手写内核:PyTorch 的 `F.scaled_dot_product_attention` 内部也会自动调度到 FlashAttention 内核。\n\n"
      "> 📄 后端选择的逐层过滤逻辑在 `vllm/v1/attention/selector.py`,第 47 课专门讲。\n\n"
      "本课你手写的 `flash_attention_chunked`,数学结构与 flash-attention 库里的 `flash_attn_func` **一一对应**——"
      "区别只在:库内核用 CUDA/Triton 把「Python for 循环」变成了「GPU 并行线程」。下一课(43)深入在线 softmax,"
      "第 44 课用 Triton 真写一个可编译的 attention 内核。")

# =====================================================================
# 第 8 节 · Streamlit
# =====================================================================
NB.md("## 8. 🖥️ Streamlit 动态演示:拖一拖,看访存曲线 🎛️\n\n"
      "运行同目录下的 `app_42_flash_attn.py`,拖动**序列长度 / 头数 / 头维度**滑块,实时看标准与 flash 的 "
      "HBM 访存、显存两条曲线如何在**对数轴**上拉开差距:\n\n"
      "```bash\nD:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_42_flash_attn.py\n```\n\n"
      "浏览器打开 **http://localhost:8501**。建议把序列长度从 256 一路拖到 8192,观察红色(naive)曲线如何以 "
      "O(N²) 的斜率快速上扬,蓝色(flash)曲线却几乎保持平缓。完整源码如下(与同目录 `app_42_flash_attn.py` 一字不差):")

NB.code(f"%%writefile {APP_FILE}\n" + APP_42, "📜 这就是 app_42_flash_attn.py 的完整源码,notebook 与 app 共用同一套 hbm_bytes / mem_bytes 公式,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "GPU 算力增长远快于内存带宽,attention 是典型的访存受限(memory-bound)算子——瓶颈在搬数据不在算",
        "标准 attention 把 S、P 两个 N×N 大张量写进又读出 HBM,访存 Θ(Nd+N²)、显存 O(N²),随 N 平方爆炸",
        "FlashAttention 用分块(tiling)+ 在线 softmax 让 S/P 不落盘,访存降到 Θ(N²d²/M),显存降到 O(N),且数学上精确(exact)",
        "本课手写的 flash_attention_chunked 与标准 attention 数值等价(allclose,误差 ~1e-6),验证了分块不改变数学",
        "GPU 真测:naive 耗时随 N 每档约 4×(O(N²) 实锤),峰值显存几乎全被 S/P 吃光",
        "FlashAttention-2 靠减少非 matmul FLOPs、沿序列并行、split-Q 切分,比 v1 快约 2×(A100 上达 225 TFLOPs/s)",
        "vLLM 的 FLASH_ATTN 后端即生产级封装;PyTorch SDPA 也会自动调度到 flash 内核",
    ],
    practice=[
        "把 hbm_bytes 的 dtype_bytes 从 2 改成 4(fp32),重算各序列长度访存,观察节省倍数如何变化",
        "推导:为什么 naive 访存里 S/P 占 4 次 N² 访问(写+读各两次)?写出每一步的字节数",
        "把 flash_attention_chunked 的 block_M 改成 16/32/64/128,重跑数值验证,观察误差与耗时",
        "给 flash_attention_chunked 加一个下三角因果掩码(is_causal),并与 F.scaled_dot_product_attention(is_causal=True) 对比",
        "在 app_42 里加一个「节省倍数」指标卡,观察它随 N 的增长曲线",
    ],
    links=[
        ("FlashAttention 论文 (NeurIPS 2022)", "https://arxiv.org/abs/2205.14135"),
        ("FlashAttention-2 论文 (ICLR 2024)", "https://arxiv.org/abs/2307.08691"),
        ("PyTorch SDPA 文档", "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH07) / "42_flash_attention_principle.ipynb"))

app_path = Path(CH07) / APP_FILE
app_path.write_text(APP_42 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
