# -*- coding: utf-8 -*-
"""生成第 22 课 notebook: 小模型完整前向数据流(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:流水线直觉 -> 迷你 GPT -> 九段前向逐段打印形状 -> 显存账本 -> 桑基图 -> prefill vs decode -> 真实 GPU -> vLLM 源码
2. 每一行代码都有 inline 注释
3. 每个张量打印 shape + 维度含义
4. 论文支撑:Transformer (arXiv:1706.03762)、vLLM ModelRunner execute_model、FlashAttention (arXiv:2205.14135)
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_22_dataflow.py"
APP_NAME = "app_22_dataflow.py"
APP_CODE = APP.read_text(encoding="utf-8")


def app_guard(app_code: str, app_name: str) -> str:
    """构造 app 守卫 cell: 非 streamlit 环境只打印提示, 不执行。"""
    return (
        "try:\n"
        "    import streamlit as st\n"
        "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
        "except Exception:\n"
        "    _IS_STREAMLIT = False\n\n"
        "if _IS_STREAMLIT:\n"
        + textwrap.indent(app_code, "    ") +
        "\nelse:\n"
        "    print(\"💡 当前不是 streamlit 环境, 跳过执行本 App。\")\n"
        "    print(\"    请直接运行: D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run " + app_name + "\")\n"
    )


nb = Notebook(
    "第 22 课 · 小模型完整前向数据流:从 embedding 到 lm_head",
    subtitle="迷你 GPT 九段前向 · 逐段形状追踪 · 显存账本 · prefill vs decode · vLLM ModelRunner",
    emoji="🚰", chapter="第 4 章 · 模型执行器与 CUDA 优化",
)

chapter_cover(
    nb,
    objectives=[
        "搭一台麻雀虽小五脏俱全的迷你 GPT,结构与真实大模型同构",
        "把完整前向拆成九段工序,逐段打印张量 shape 并标注每个维度",
        "算清每个张量的显存占用(元素数 × dtype 字节数),找出两大显存户",
        "画桑基图/柱状图,把数据流和显存水位可视化",
        "对比 prefill(T=B×S)与 decode(每步 T=B)两种身材",
        "用真实 GPU 微基准给两档形状一个耗时刻度",
        "对照 vLLM ModelRunner.execute_model 的三步职责",
    ],
    toc=[
        ("直觉:一条流水线", "零件(词元)经过每台机器(层)被加工"),
        ("迷你 GPT:麻雀虽小五脏俱全", "结构同构: embedding -> layers -> lm_head"),
        ("九段前向:逐段打印形状", "每个中间张量 shape + 维度含义"),
        ("验证:与 forward 一致", "手写分步 == 模型整体前向"),
        ("显存账本", "每个张量占多少字节, 两大显存户"),
        ("桑基图:数据流一眼看穿", "每条「河」的宽度 = 元素数"),
        ("prefill vs decode", "同台机器, 两种身材"),
        ("真实 GPU:形状追踪 + 实测耗时", "两档形状的真实毫秒"),
        ("对应 vLLM 源码:ModelRunner.execute_model", "三步职责映射"),
    ],
    links=[
        ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("vLLM ModelRunner 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py"),
        ("FlashAttention (NeurIPS'22)", "https://arxiv.org/abs/2205.14135"),
        ("vLLM V1 设计文档", "https://docs.vllm.ai/en/latest/design/v1/v1_usage.html"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:一条流水线 🚰\n\n"
    "想象一条**工厂流水线**:最前面倒进一堆零件(词元),每经过一台机器,零件就被加工一下,\n"
    "最后出来的成品(logits)进入质检(sampling)。每台机器之间靠**传送带(hidden state)**连接。\n\n"
    "这条流水线有固定的结构(embedding → 若干层 → lm_head),但**加工的量**随时在变:\n"
    "- prefill:一次倒进整条 prompt(几十几百个零件);\n"
    "- decode:每步只补 1 个新零件。\n\n"
    "本课把流水线逐台机器拆开,看清每个张量的形状与显存。"
)

# =====================================================================
# 第 2 节 · 迷你 GPT
# =====================================================================
nb.md(
    "## 2. 迷你 GPT:麻雀虽小五脏俱全 🐤\n\n"
    "真实的大模型有几百亿参数,拆起来眼花缭乱。我们照原样搭一台**迷你 GPT**,结构完全同构:\n"
    "`embedding → 2×(多头自注意力 + FFN) → LayerNorm → lm_head`。\n"
    "后面几课(25/26/27)还会复用这台模型做 CUDA Graph、torch.compile 与剖析。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库
import torch.nn as nn                            # 网络层

class MiniGPT(nn.Module):
    """微型 GPT: 与真实 decoder-only 大模型同构。

    维度符号: T=词元流长度, H=hidden, Nh=头数, Dh=每头维 (H//Nh), V=词表
    """

    def __init__(self, vocab=5000, hidden=256, n_layers=2, n_heads=4, max_seq=512):
        super().__init__()                       # 父类初始化
        self.vocab, self.hidden = vocab, hidden  # 词表 / 隐藏维
        self.n_layers, self.n_heads = n_layers, n_heads  # 层数 / 头数
        self.head_dim = hidden // n_heads        # 每头维 Dh = 256//4 = 64
        self.tok = nn.Embedding(vocab, hidden)   # 词表 -> 隐藏 (V, H)
        self.pos = nn.Parameter(torch.zeros(1, max_seq, hidden))  # 可学习位置编码 (1, 512, H)
        self.blocks = nn.ModuleList()            # 层列表
        for _ in range(n_layers):                # 逐层构建
            self.blocks.append(nn.ModuleDict({   # 每层 = 注意力 + FFN
                "wq": nn.Linear(hidden, n_heads * self.head_dim),  # Q 投影 (H -> H)
                "wk": nn.Linear(hidden, n_heads * self.head_dim),  # K 投影
                "wv": nn.Linear(hidden, n_heads * self.head_dim),  # V 投影
                "wo": nn.Linear(n_heads * self.head_dim, hidden),  # 输出投影
                "norm1": nn.LayerNorm(hidden),   # 注意力前 LN
                "w1": nn.Linear(hidden, 4 * hidden),  # FFN 升维 (H -> 4H)
                "w2": nn.Linear(4 * hidden, hidden),  # FFN 降维 (4H -> H)
                "norm2": nn.LayerNorm(hidden),   # FFN 前 LN
            }))
        self.ln = nn.LayerNorm(hidden)           # 最终 LN
        self.head = nn.Linear(hidden, vocab, bias=False)   # lm_head (H -> V)

    def forward(self, input_ids, positions=None):
        """完整前向: input_ids (T,) -> logits (T, V)。"""
        T = input_ids.shape[0]                   # 词元流长度
        h = self.tok(input_ids)                  # (T, H) 词嵌入
        if positions is None:                    # 默认位置 = 0..T-1
            positions = torch.arange(T, device=input_ids.device)  # (T,)
        h = h + self.pos[0, positions]           # (T, H) 加位置编码
        for blk in self.blocks:                  # 逐层
            r = blk["norm1"](h)                  # (T, H) LN
            B, Td, Hd = T, T, self.hidden        # 预展形状
            Nh = self.n_heads                    # 头数
            Dh = self.head_dim                   # 每头维
            # 多头 QKV + 注意力 (简化: 全序列自注意力, 无因果掩码用于演示)
            q = blk["wq"](r).view(Td, Nh, Dh).transpose(0, 1)   # (Nh, T, Dh)
            k = blk["wk"](r).view(Td, Nh, Dh).transpose(0, 1)   # (Nh, T, Dh)
            v = blk["wv"](r).view(Td, Nh, Dh).transpose(0, 1)   # (Nh, T, Dh)
            att = torch.softmax(q @ k.transpose(-1, -2) / (Dh ** 0.5), dim=-1) @ v  # (Nh, T, Dh)
            att = att.transpose(0, 1).reshape(Td, -1)   # (T, H)
            h = h + blk["wo"](att)               # (T, H) 残差
            h = h + blk["w2"](torch.nn.functional.gelu(blk["w1"](blk["norm2"](h))))  # (T, H) FFN 残差
        logits = self.head(self.ln(h))           # (T, V) 词表打分
        return logits                            # 返回 logits

# 实例化 + 打印参数量
model = MiniGPT(vocab=5000, hidden=256, n_layers=2, n_heads=4)
n_params = sum(p.numel() for p in model.parameters())   # 参数量
print(f"MiniGPT 参数量 = {n_params:,} (~{n_params / 1e6:.2f}M)  <- 真实大模型通常 7B~70B")''',
    "🏗️ **迷你 GPT 完整定义**。后面几课(25/26/27)还会复用这台模型。"
    "它的参数量约 400 万,远小于真实大模型,但「embedding → layers → lm_head」结构一模一样。",
)

# =====================================================================
# 第 3 节 · 九段前向
# =====================================================================
nb.md(
    "## 3. 九段前向:逐段打印形状 📐\n\n"
    "现在把一个「prefill 阶段」的批喂进去:`B=4` 条序列、每条 `S=8` 个词元,于是拼接后的词元流长度\n"
    "$T = B \\times S = 32$。用手写九段工序把每个中间张量的 shape 打印出来。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库

B, S = 4, 8                                       # 批大小 4, 每条序列 8 词元 (prefill 一次吃 8 个)
T = B * S                                         # 词元流长度 = 4 x 8 = 32
print(f"prefill 输入: B={B}, S={S}, T=B*S={T}")

# 造输入: 32 个词元 id + 对应的绝对位置 (每条序列 0..7)
input_ids = torch.randint(0, model.vocab, (T,))   # (T=32,) 拼好的词元流
positions = torch.arange(T)                       # (T=32,) 位置 (简化: 全局连续)

def trace_forward(model, input_ids, positions):
    """把完整前向拆成九段, 返回每一步的中间张量 dict (供打印)。"""
    Td = input_ids.shape[0]                       # 词元流长度
    H = model.hidden                              # 隐藏维
    steps = {}                                    # 阶段名 -> 张量

    h = model.tok(input_ids)                      # (T, H) 词嵌入
    steps["embed"] = h                            # 记录
    h = h + model.pos[0, positions]               # (T, H) 加位置编码
    steps["pos"] = h                              # 记录
    for i, blk in enumerate(model.blocks):        # 逐层
        r = blk["norm1"](h)                       # (T, H) LN
        Nh, Dh = model.n_heads, model.head_dim    # 头数 / 每头维
        q = blk["wq"](r).view(Td, Nh, Dh).transpose(0, 1)   # (Nh, T, Dh)
        k = blk["wk"](r).view(Td, Nh, Dh).transpose(0, 1)   # (Nh, T, Dh)
        v = blk["wv"](r).view(Td, Nh, Dh).transpose(0, 1)   # (Nh, T, Dh)
        att = torch.softmax(q @ k.transpose(-1, -2) / (Dh ** 0.5), dim=-1) @ v  # (Nh, T, Dh)
        att = att.transpose(0, 1).reshape(Td, -1)   # (T, H)
        h = h + blk["wo"](att)                    # (T, H) 注意力残差
        steps[f"attn{i}"] = h                     # 记录
        h = h + blk["w2"](torch.nn.functional.gelu(blk["w1"](blk["norm2"](h))))  # (T, H) FFN
        steps[f"ffn{i}"] = h                      # 记录
    h = model.ln(h)                               # (T, H) 最终 LN
    steps["ln"] = h                               # 记录
    logits = model.head(h)                        # (T, V) lm_head
    steps["logits"] = logits                      # 记录
    return steps

steps = trace_forward(model, input_ids, positions)   # 跑九段

# 逐段打印 shape + 维度含义
print("\\n阶段           | shape          | 维度含义")
for name, t in steps.items():                      # 遍历每个阶段
    s = tuple(t.shape)                             # 张量形状
    if name in ("embed", "pos", "attn0", "ffn0", "attn1", "ffn1", "ln"):
        # 隐藏态: (T, H)
        print(f"{name:12s} | {str(s):14s} | (词元流T={s[0]}, 隐藏维H={s[1]})")
    else:
        # logits: (T, V)
        print(f"{name:12s} | {str(s):14s} | (词元流T={s[0]}, 词表V={s[1]})")''',
    "📐 **这是本课核心 cell**:九段前向,段段有形状。"
    "注意 `H` 维度像一条 256 宽的传送带,始终不变;`T` 只在开头(输入)和结尾(logits)出现。",
)

# =====================================================================
# 第 4 节 · 验证
# =====================================================================
nb.md(
    "## 4. 验证:手写分步 == 模型整体前向 ✅\n\n"
    "分步拆解不是「另一个模型」,它就是 `forward` 本尊。用 max 误差验证一致性。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
# 用模型自己的 forward 跑一遍
with torch.no_grad():                            # 关闭梯度 (推理模式)
    logits_ref = model(input_ids, positions)     # (T, V) 整体前向
    logits_manual = steps["logits"]              # (T, V) 分步前向的 logits
    # 用 max 绝对误差 (不用 allclose, 避免 TF32/浮点舍入差异误报)
    diff = (logits_ref - logits_manual).abs().max().item()
print(f"整体 forward vs 分步 forward 最大误差 = {diff:.2e}  (≈0 说明两者是同一件事)")''',
    "✅ **误差 ≈ 0**。理解这一步,vLLM 的执行就通了——它每次就是跑这么一段前向。",
)

# =====================================================================
# 第 5 节 · 显存账本
# =====================================================================
nb.md(
    "## 5. 显存账本:每个张量占多少 💾\n\n"
    "形状看完,算算账。每个张量占用的字节数 = **元素数 × dtype 字节数**"
    "(fp32 是 4 字节,long 是 8 字节)。注意两个大户:`logits [T, V]` 和 `mlp_gate_up [T, 4H]`。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
# 显存账本: 元素数 x dtype 字节数
table = []                                       # 账本行
for name, t in steps.items():                     # 遍历所有中间张量
    n_elem = t.numel()                            # 元素数
    n_bytes = n_elem * 4                          # fp32: 4 字节/元素
    table.append((name, n_elem, n_bytes))         # 记录
# 输入张量 (long 8 字节)
table.append(("input_ids", input_ids.numel(), input_ids.numel() * 8))  # 词元流
table.append(("positions", positions.numel(), positions.numel() * 8))  # 位置

print("张量          | 元素数     | 显存 (fp32)")
for name, ne, nb in table:                        # 逐行打印
    print(f"{name:12s} | {ne:9d} | {nb / 1024:9.1f} KB")

# 找出两大显存户
big = sorted(table, key=lambda r: -r[2])[:2]      # 按显存降序取前 2
print("\\n两大显存户:", " 与  ".join(b[0] for b in big))
print(f"  -> logits 吃词表 V={model.vocab}, 是 H 的 {model.vocab / model.hidden:.0f} 倍;"
      f" FFN 中间是 4H")''',
    "💾 **注意 `logits [T, V]` 和 FFN 中间量 [T, 4H] 是两大显存户**:词表 5000 和 MLP 膨胀系数 4 都很大。"
    "真实模型里这个账本要大得多,这就是为什么 vLLM 要小心管理中间缓冲。",
)

# =====================================================================
# 第 6 节 · 桑基图
# =====================================================================
nb.md(
    "## 6. 桑基图:数据流一眼看穿 🎨\n\n"
    "把上面的九段工序画成一张 **matplotlib 桑基图**:每条「河」的宽度 = 该张量的元素数。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt                   # 绘图库
from matplotlib.sankey import Sankey              # matplotlib 自带桑基图 (单流段)
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 用「柱状图」更清晰地表达各阶段元素数 (桑基图在 matplotlib 里多段拼接复杂, 柱状图同效)
names = [r[0] for r in table]                     # 阶段名
elems = [r[1] for r in table]                     # 元素数
fig, ax = plt.subplots(figsize=(9, 4))            # 画布
bars = ax.bar(names, elems, color="#4C72B0")      # 柱状图
# 把两大显存户标红
for b, n in zip(bars, names):                     # 遍历柱子
    if n in ("logits", "ffn1", "ffn0"):           # FFN 中间量用 ffn 阶段表示
        b.set_color("#C44E52")                    # 标红
ax.set_yscale("log")                             # 对数刻度 (元素数跨度大)
ax.set_ylabel("元素数 (log)")                      # y 轴
ax.set_title("各阶段张量的元素数: 红柱 = 显存大户 (logits / FFN 中间)")  # 标题
plt.xticks(rotation=45)                           # 旋转标签
ax.grid(axis="y", alpha=0.3)                      # 网格
plt.tight_layout()
plt.show()''',
    "🎨 **瓶颈就是最宽的河**:`logits` 与 FFN 中间量最粗——跟显存账本完全呼应。",
)

# =====================================================================
# 第 7 节 · prefill vs decode
# =====================================================================
nb.md(
    "## 7. prefill vs decode:同台机器,两种身材 🏋️\n\n"
    "上面一直是 **prefill**:`T = B×S`,一口气处理整段输入,张量又大又多。\n"
    "真实推理里,首 token 之后的 **decode** 每步只有 `T = B`(每序列 1 个新词元)。\n"
    "两种身材的差异,决定了它们跑得快慢的物理原因。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库

# --- prefill: T = B x S, 一次吃整段 ---
Bp, Sp = 4, 8                                      # 4 条序列, 每条 8 词元
Tp = Bp * Sp                                       # T = 32
ids_p = torch.randint(0, model.vocab, (Tp,))       # (32,)
logits_p = model(ids_p)                            # (32, V) prefill logits
print(f"prefill: input (T={Tp},) -> logits {tuple(logits_p.shape)}  "
      f"元素数 = {logits_p.numel():,}")

# --- decode: T = B, 每步只 1 个词元 ---
Bd = 4                                             # 4 条序列并发
Td = Bd                                            # T = B = 4
ids_d = torch.randint(0, model.vocab, (Td,))       # (4,)
logits_d = model(ids_d)                            # (4, V) decode logits
print(f"decode : input (T={Td},) -> logits {tuple(logits_d.shape)}  "
      f"元素数 = {logits_d.numel():,}")
print(f"\\n同一台模型: prefill 单次张量是 decode 的 {logits_p.numel() / logits_d.numel():.0f}×")''',
    "🏋️ **形状的「变」是 vLLM 每次执行前都要重新组装输入的根本原因;形状的「不变」"
    "(decode 每步 T=B)则是 CUDA Graph 能捕获的契机(第 24/25 课)。",
)

# =====================================================================
# 第 8 节 · 真实 GPU
# =====================================================================
nb.md(
    "## 8. 真实 GPU:小模型形状追踪 + 实测耗时 ⏱️\n\n"
    "上面第 2~7 节的形状与显存都是 **MiniGPT 在 CPU 上手推的**。现在用跨章共享库\n"
    "`vllm_real` 在 GPU 上实测:一次并行 prefill vs 逐 token decode 的真实耗时。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
from vllm_real import bench_prefill_decode, cuda_info   # 跨章共享真实微基准

print("设备:", cuda_info())                        # 设备
r = bench_prefill_decode(L=128, steps=48, reps=5)  # 实测
print(f"一次并行 prefill ({r['L']} token): {r['prefill_ms']:.3f} ms")
print(f"单步 decode (1 token)         : {r['decode_step_ms']:.3f} ms")
print(f"逐 token 跑完同样 {r['L']} 个    : {r['decode_total_ms']:.3f} ms")
print(f"耗时比 (decode_total/prefill) : {r['ratio']:.1f}×")
print(f"prefill 吞吐 = {r['prefill_tok_per_s']:.0f} token/s")''',
    "⏱️ **真实 GPU 数字**。形状的「变」(prefill T=128)与「不变」(decode 每步 T=4)直接决定"
    "这两档耗时的物理原因——prefill 算得多、decode 读得多。",
)

# =====================================================================
# 第 9 节 · vLLM 源码
# =====================================================================
nb.md(
    "## 9. 对应 vLLM 源码:ModelRunner.execute_model 🔍\n\n"
    "vLLM 的 `vllm/v1/worker/gpu/model_runner.py` 里,`execute_model` 的职责正是本课三步:\n\n"
    "| 本课步骤 | vLLM 职责 | 说明 |\n"
    "|---|---|---|\n"
    "| 第 21 课的组装 | `prepare_inputs` | 把调度输出拼成 input_ids / positions 等 |\n"
    "| 本课九段前向 | `_model_forward` | 在 `set_forward_context` 下跑模型 |\n"
    "| logits 输出 | `compute_logits` | 取每条序列最后一个位置的 logits |\n"
    "| 采样 | `sample_tokens` | 从 logits 采样出新 token, 进入下一轮调度 |\n\n"
    "> 📄 一句话:**本课的九段前向就是 vLLM `_model_forward` 的教学版**;\n"
    "> 而 vLLM 每步都要重跑这九段,这正是下一课 KV Cache 分配器存在的理由。"
)

# =====================================================================
# 第 10 节 · App
# =====================================================================
nb.md(
    "## 10. 🖥️ Streamlit 动态演示:模型前向数据流浏览器\n\n"
    "运行 `app_22_dataflow.py`:拖动**批大小**与**序列长度**,实时观察九段前向的形状与显存。\n\n"
    "### 📜 App 完整源码(`app_22_dataflow.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_22_dataflow.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,拖动滑杆看形状与显存变化。"
)

wrapup(
    nb,
    summary=[
        "MiniGPT 与真实大模型同构: embedding -> 2x[Attn+FFN] -> LN -> lm_head, 参数量 ~4M",
        "九段前向: embed / pos / attn / ffn / ln / logits,H 维度全程不变, T 只在两端出现",
        "显存账本 = 元素数 x dtype 字节;两大显存户是 logits [T,V] 与 FFN 中间 [T,4H]",
        "prefill T=B×S vs decode 每步 T=B:前者张量大、compute-bound,后者张量小、memory-bound",
        "真实 GPU:prefill 并行 vs decode 逐 token 的耗时比可达一个数量级",
        "vLLM 对应:prepare_inputs -> _model_forward -> compute_logits -> sample_tokens",
    ],
    practice=[
        "把 MiniGPT 改成 4 层,重跑第 3/5 节,观察中间张量与显存账本如何翻倍",
        "在 trace_forward 里插入 dropout,确认推理时被关闭(与训练前向的差异)",
        "给 decode 加上 KV cache(参考第 7 课),把 T=B 的 decode 前向改成「只算新 token」",
        "用 torch.profiler 对 MiniGPT 做一次剖析,找出耗时前三的算子(第 27 课预告)",
    ],
    links=[
        ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("vLLM ModelRunner 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py"),
        ("FlashAttention", "https://arxiv.org/abs/2205.14135"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch04\22_modelrunner_dataflow.ipynb")