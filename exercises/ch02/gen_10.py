# -*- coding: utf-8 -*-
"""生成第 10 课 notebook: PagedAttention 核心(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md 与 gen_07.py 金标准):
1. 由浅入深:OS 虚拟内存类比 -> 虚拟/物理块定义 -> 手工页表逐行推演 -> torch 物理池 scatter/gather 数值验证 ->
   real_ops.paged_map 真实页表 -> 论文 §4.1 块级注意力公式 -> 与 vLLM kernel 的关系 -> 小结
2. 每一行代码都有 inline 注释;每个张量打印 shape 并标注维度含义
3. 论文支撑:PagedAttention (SOSP'23, arXiv:2309.06180) §4.1/§4.2/§4.3、vLLM Paged Attention 设计文档
4. 复用 real_ops.paged_map 生成随机物理页表
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_10_paged_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 10 课 · PagedAttention 核心:用操作系统的虚拟内存管理 KV",
    subtitle="虚拟块 vs 物理块 · 手工页表逐行推演 · torch 物理池数值验证 · 论文块级注意力",
    emoji="📖", chapter="第 2 章 · KV Cache 与 PagedAttention",
)

chapter_cover(
    nb,
    objectives=[
        "用操作系统虚拟内存类比建立直觉:块=页、token=字节、请求=进程、块表=页表",
        "严格定义虚拟块与物理块,推导逻辑块数公式 N_logical = ⌈S/B⌉ 与 slot 定位公式",
        "手写 PageTable,把逻辑序列切成块并映射到物理块,逐行推演每个中间量",
        "用 torch 构造真实形状的物理 KV 池,验证「scatter 散落写入 → 按块表 gather 读回」数据不丢",
        "复用 real_ops.paged_map 生成随机映射的真实页表,复核 slot = P×B+off 对每个 token 成立",
        "读懂论文 §4.1 的块级注意力公式与 vLLM kernel 的指针寻址,理解「不连续也能高效计算」",
    ],
    toc=[
        ("直觉:操作系统的虚拟内存", "块=页、token=字节、请求=进程的一一对应"),
        ("核心定义与公式", "虚拟块/物理块/块表,与两个关键公式"),
        ("手工页表逐行推演", "PageTable 类 + slot 计算,打印每个中间量"),
        ("torch 数值验证:散落写入不丢数据", "物理池 scatter → 按块表 gather → 逐位一致"),
        ("真实页表", "real_ops.paged_map 随机映射 + slot 公式全量复核"),
        ("可视化:逻辑→物理映射", "plotly 表格与热图,pyecharts 占用热图"),
        ("论文视角:块级注意力", "§4.1 公式与 vLLM kernel 的指针寻址"),
        ("Streamlit 动态演示", "交互式分页映射"),
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM Paged Attention 设计文档", "https://docs.vllm.ai/en/latest/design/paged_attention.html"),
        ("vLLM 官方博客: 10x faster inference", "https://blog.vllm.ai/2023/06/20/vllm.html"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉:操作系统早在 1960 年代就解决了这个问题\n\n"
    "上一课停车场告诉我们:要求「一整片相邻车位」太奢侈。那怎么办?\n"
    "操作系统早就有一模一样的矛盾——**进程看到连续地址空间,物理内存却随意分散**——"
    "它的解法就是 **分页 + 页表**。PagedAttention 论文(SOSP'23)把这个思想原封不动搬到 KV Cache:\n\n"
    "| 操作系统 | PagedAttention |\n"
    "|---|---|\n"
    "| 进程 process | 请求 request(一个 prompt + 生成) |\n"
    "| 页 page(通常 4KB) | **KV 块**(默认 16 token) |\n"
    "| 页表 page table | **块表 block table**(逻辑块 → 物理块) |\n"
    "| 物理页框 physical frame | 共享物理块池里的物理块 |\n"
    "| 缺页 / 分配 | 按需分配新块 |\n"
    "| fork() 写时复制 | 前缀共享 / COW(第 13 课) |\n\n"
    "论文原文的一一对应:「**one can think of blocks as pages, tokens as bytes, and requests as processes**」。\n\n"
    "> 🏷️ **为什么这个类比成立?** 因为 KV Cache 和进程内存有同样的三个特征:① 动态增长(每步 +1 token);"
    "② 生命周期与最终长度不可预知;③ 请求之间可以共享前缀。这三个特征让「按最大长度预分配连续内存」注定浪费,"
    "让「分页」注定高效。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式\n\n"
    "给定序列长度 $S$ 与块大小 $B$(vLLM 默认 16),逻辑块数为:\n\n"
    "$$ N_{logical} = \\left\\lceil \\frac{S}{B} \\right\\rceil $$\n\n"
    "第 $i$ 个逻辑块装 token $[iB,\\ (i{+}1)B)$;**最后一个块可以装不满**(内部碎片 ≤ B-1)。\n\n"
    "每个逻辑块被映射到**任意一个空闲物理块**,这张映射就是 **block table(块表,即页表)**。\n"
    "论文 §4.1 把第 $j$ 个 KV 块写成(下标按论文习惯从 1 开始):\n\n"
    "$$ K_j = (k_{(j-1)B+1}, \\ldots, k_{jB}), \\qquad V_j = (v_{(j-1)B+1}, \\ldots, v_{jB}) $$\n\n"
    "即每个块里装 **B 个连续位置的 K 向量(或 V 向量)**。块内是连续的,块间可以散落。\n\n"
    "**寻址公式(slot 定位)**:物理块 $P$ 内第 $o$ 个槽位的线性下标为\n\n"
    "$$ \\text{slot} = P \\times B + o $$\n\n"
    "每个符号: $P$=物理块号(块表查到), $B$=块大小, $o$=块内偏移 = $k \\bmod B$。"
    "任意 token $k$ 两步定位:$lb = k // B$(逻辑块号)→ $P = \\text{table}[lb]$(查块表)→ $o = k \\bmod B$。"
)

# =====================================================================
# 第 3 节 · 手工页表逐行推演
# =====================================================================
nb.md(
    "## 3. 手工页表逐行推演\n\n"
    "用字典 `logical -> physical` 就是一张最朴素的块表。看代码前先心算一个例子:\n\n"
    "```\n"
    "seq_len=10, block_size=4  →  N_logical = ceil(10/4) = 3\n"
    "token 0~3 → 逻辑块 L0 → 物理块 P2\n"
    "token 4~7 → 逻辑块 L1 → 物理块 P0\n"
    "token 8~9 → 逻辑块 L2 → 物理块 P4 (只装 2 个, 块尾 2 个槽空闲)\n"
    "token 9 的 slot = P4 × 4 + 1 = 17\n"
    "```"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import math

class PageTable:
    # 最朴素的块表(页表): 逻辑块号 -> 物理块号
    def __init__(self, seq_len: int, block_size: int, free_blocks: list):
        self.seq_len = seq_len                     # S: 序列长度(token 数)
        self.block_size = block_size               # B: 每块装几个 token
        self.n_logical = math.ceil(seq_len / block_size)   # N_logical = ceil(S/B)
        self.table = {}                            # 块表本体: {逻辑块号: 物理块号}
        for lb in range(self.n_logical):           # 逐个逻辑块分配
            if not free_blocks:                    # 物理块池耗尽
                raise MemoryError("物理块不足")     # 抛错提示
            self.table[lb] = free_blocks.pop(0)    # 从池里顺序取一块(first-fit 简化)

    def slot_of(self, token_idx: int):
        # 任意 token 的定位: 逻辑块 -> 物理块 -> 块内偏移 -> 绝对 slot
        lb, off = divmod(token_idx, self.block_size)  # lb=k//B, off=k%B
        pb = self.table[lb]                          # 查块表拿物理块号
        return lb, pb, off, pb * self.block_size + off  # 返回(逻辑块, 物理块, 偏移, slot)

# --------------------------------------------------------------------------
# 推演: seq_len=10, block_size=4, 物理块池给 5 块(编号 0~4)
# --------------------------------------------------------------------------
pt = PageTable(10, 4, [2, 0, 4, 1, 3])               # 池顺序: 先发 P2, 再 P0, 再 P4
print(f"逻辑块数 N_logical = {pt.n_logical}  <- ceil({pt.seq_len}/{pt.block_size})")
print(f"块表 logical -> physical = {pt.table}")

# 抽查几个关键 token: 首、块边界、末
for tok in [0, 3, 4, 9]:
    lb, pb, off, slot = pt.slot_of(tok)              # 解包定位结果
    print(f"token {tok:2d} -> 逻辑块 L{lb} -> 物理块 P{pb} -> 块内偏移 {off} -> slot = {slot}")

# 公式复核: 每个 token 的 slot 都等于 P×B+off
ok = all(pt.slot_of(k)[3] == pt.table[k // pt.block_size] * pt.block_size + (k % pt.block_size)
         for k in range(10))                         # 对 0..9 全部 token 断言
print(f"\\nslot = P×B + off 对全部 {pt.seq_len} 个 token 成立? {ok} ✅")''',
    "🎨 **图示**。逻辑块 0、1、2 分别落在物理块 2、0、4——物理地址完全打乱,"
    "但只要块表在手,每个 token 的 slot 都能 O(1) 算出。",
)

# =====================================================================
# 第 4 节 · torch 数值验证
# =====================================================================
nb.md(
    "## 4. torch 数值验证:散落写入,逐位读回不丢数据 🔬\n\n"
    "上一节的块表只是「编号游戏」。现在用 **torch 真张量**把 KV 数据写进一个物理池,\n"
    "再按块表 gather 回逻辑顺序,验证**数据在物理块间散落存放后仍然逐位一致**——\n"
    "这正是 PagedAttention kernel 每次前向在做的事:按 block table 从 `k_cache`/`v_cache` 取数。\n\n"
    "张量布局:物理池 `(num_blocks, block_size, num_heads, head_dim)` 对应 vLLM 的 `k_cache`。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch

# --------------------------------------------------------------------------
# 1. 参数: 迷你规模, 便于心算
# --------------------------------------------------------------------------
seq_len = 10                # S: 逻辑序列 10 个 token
block_size = 4              # B: 每块 4 个 token
num_blocks = 5              # 物理块池共 5 块
num_heads = 2               # H: 注意力头数
head_dim = 8                # D: 每头维度
table = {0: 2, 1: 0, 2: 4}  # 块表: 逻辑块 0/1/2 -> 物理块 2/0/4(复用第 3 节)

# --------------------------------------------------------------------------
# 2. 物理 KV 池: 形状 (物理块数, 块内token数, 头数, 头维度)
# --------------------------------------------------------------------------
k_pool = torch.zeros(num_blocks, block_size, num_heads, head_dim)  # K 物理池
print(f"K 物理池 shape = {tuple(k_pool.shape)}")
print("  维度含义: (num_blocks=物理块数, block_size=块内token数, num_heads=头数, head_dim=头维度)")

# --------------------------------------------------------------------------
# 3. 模拟「刚算出的逻辑 K」: 形状 (seq_len, num_heads, head_dim)
#    用 arange 保证每个元素唯一, 方便验证读回一致
# --------------------------------------------------------------------------
k_logical = torch.arange(seq_len * num_heads * head_dim, dtype=torch.float32).reshape(
    seq_len, num_heads, head_dim)
print(f"\\n逻辑 K shape = {tuple(k_logical.shape)}  <- (逻辑token数=S, 头数, 头维度)")

# --------------------------------------------------------------------------
# 4. scatter: 把逻辑序列按块表「散落」写进物理池
#    第 lb 个逻辑块的内容写到物理块 table[lb] 的块内位置 0..(该块实际token数-1)
# --------------------------------------------------------------------------
for lb in range(len(table)):                     # 遍历 3 个逻辑块
    start = lb * block_size                      # 该逻辑块在序列中的起始 token 下标
    end = min(start + block_size, seq_len)       # 结束下标(最后一块可能不满)
    n_tok = end - start                          # 该块实际装的 token 数
    pb = table[lb]                               # 查块表: 物理块号
    k_pool[pb, :n_tok] = k_logical[start:end]    # 把这块 KV 放进物理池对应槽位
print(f"写入后: 物理块 {sorted(table.values())} 已有内容, 其余块保持全 0")

# --------------------------------------------------------------------------
# 5. gather: 按块表把物理池读回逻辑顺序
# --------------------------------------------------------------------------
k_gathered = torch.zeros_like(k_logical)         # 与逻辑 K 同形状的收集结果
for tok in range(seq_len):                       # 逐个 token
    lb, off = divmod(tok, block_size)            # 逻辑块号与块内偏移
    pb = table[lb]                               # 物理块号
    k_gathered[tok] = k_pool[pb, off]            # 从物理池对应槽位取回
print(f"gather 后 shape = {tuple(k_gathered.shape)}  <- (S, H, D), 与逻辑 K 完全同形")

# --------------------------------------------------------------------------
# 6. 断言: 散落存放 + 块表读回 = 原数据逐位一致
# --------------------------------------------------------------------------
identical = torch.equal(k_gathered, k_logical)   # 逐元素相等?
print(f"scatter → gather 后与原始逻辑 K 逐位一致? {identical} ✅")
print(f"(验证维度示例) k_gathered[9, 1, :3] = {k_gathered[9, 1, :3].tolist()}")
print(f"                k_logical[9, 1, :3] = {k_logical[9, 1, :3].tolist()}  <- 相同则说明没丢")''',
    "🔬 **数值验证**。K 数据散落在物理块 2/0/4,经块表读回后与原始逻辑序列**逐位一致**——"
    "这就是「不连续存储也能正确计算」的证明,也是 PagedAttention kernel 每次都做的事。",
)

# =====================================================================
# 第 5 节 · 真实页表
# =====================================================================
nb.md(
    "## 5. 真实页表:paged_map 生成一张随机映射 🗂️\n\n"
    "手写的 `PageTable` 顺序取块,偏教科书。仓库的 `real_ops.paged_map` 生成**随机映射**的页表:\n"
    "逻辑块号仍连续 0,1,2,…,物理块号却从池中**随机**弹出、散落各处——"
    "这正是 OS 的 MMU/页表做的事,也是 vLLM attention backend 真正遇到的形态。\n\n"
    "关键洞察:无论物理块怎么乱序,只要有块表,`slot = P × block_size + off` 对**任意 token** 都成立。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 OpenMP 冲突
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch02")
from real_ops import paged_map                         # 生成随机物理页表

B = 16                                                 # 块大小(对齐 vLLM 默认)
pm = paged_map(seq=50, block_size=B, n_phys=10, seed=42)  # 50 token, 10 物理块
print(f"逻辑块 -> 物理块 页表: {pm['mapping']}")          # 打印随机映射
print(f"逻辑块数 = {pm['n_logical']}(ceil(50/16)=4), 已映射 {len(pm['mapping'])} 块")

# 空闲(未映射)物理块
all_phys = set(range(10))                              # 全部物理块编号
free = sorted(all_phys - set(pm["mapping"].values()))  # 减去已用 = 空闲
print(f"空闲物理块: {free}")

# 抽查 4 个关键 token 的落点, 展示「逻辑块 -> 物理块 -> 偏移 -> slot」链路
print("\\n抽查 token 落点:")
for s in pm["slots"]:                                  # 遍历每个 token 的落点记录
    if s["token"] in (1, 16, 17, 50):                  # 只打印 4 个代表
        print(f"  token {s['token']:2d} -> 逻辑块 L{s['logical']} -> 物理块 P{s['physical']}"
              f" -> 偏移 {s['offset']:2d} -> slot = {s['slot']}")

# 全量复核: 每个 token 的 slot 都严格等于 物理块×16 + 偏移
ok = all(s["slot"] == s["physical"] * B + s["offset"] for s in pm["slots"])
print(f"\\nslot = 物理块×{B} + 偏移 对全部 {pm['seq']} 个 token 成立? {ok} ✅")''',
    "🗂️ **真实页表**。物理块被打乱,但每个 token 的 slot 仍能 O(1) 算出——"
    "这就是 PagedAttention 能「拒绝连续」却不付出寻址代价的原因。",
)

# =====================================================================
# 第 6 节 · 可视化
# =====================================================================
nb.md(
    "## 6. 可视化:逻辑 → 物理 映射 📊\n\n"
    "把块表画成表格与热图:表格看清每个映射,热图看清「哪些物理块被哪个逻辑块占用」。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"                     # 内嵌渲染

# 表格: 逻辑块 -> 物理块 + token 区间
rows = [{"逻辑块": f"L{lb}", "物理块": f"P{pb}",
         "token 区间": f"{lb*B+1} ~ {min((lb+1)*B, 50)}"}   # token 1 起, 末块截断
        for lb, pb in sorted(pm["mapping"].items())]          # 按逻辑块序排列
fig1 = go.Figure(go.Table(
    header=dict(values=["逻辑块", "物理块", "token 区间"], fill_color="#2c3e50",
                font=dict(color="white"), align="center"),
    cells=dict(values=[[r["逻辑块"] for r in rows], [r["物理块"] for r in rows],
                       [r["token 区间"] for r in rows]], align="center")))
fig1.update_layout(title="🗺️ 块表:逻辑块 → 物理块", height=200, template="plotly_white")
fig1.show()

# 热图: 每个逻辑块格子的颜色 = 它映射到的物理块号(-1=未映射)
z = [[pm["mapping"].get(lb, -1) for lb in range(pm["n_logical"])]]   # 一行 4 个逻辑块
fig2 = go.Figure(go.Heatmap(
    z=z, x=[f"L{lb}" for lb in range(pm["n_logical"])], y=["物理块号"],
    colorscale=[[0, "#f5f5f5"], [0.1, "#2c3e50"], [1, "#27ae60"]],
    zmin=-1, zmax=9, showscale=False,
    hovertemplate="逻辑块 %{x}: 物理块 P%{z}<extra></extra>"))
fig2.update_layout(title="🧩 逻辑块 → 物理块 热图", height=200, template="plotly_white",
                   xaxis=dict(side="top"), margin=dict(t=60))
fig2.show()''',
    "📊 表格与热图双视角:逻辑序列连续(0,1,2,3),物理位置散落(如 P9, P0, P6…)——一目了然。",
)

# =====================================================================
# 第 7 节 · 论文视角
# =====================================================================
nb.md(
    "## 7. 论文视角:块级注意力与 kernel 寻址\n\n"
    "PagedAttention 的注意力计算也是**块粒度**的(论文 §4.1)。对第 $i$ 个 query $q_i$,\n"
    "把它的历史 KV 按块切分后逐块计算并合并:\n\n"
    "$$ A_{ij} = \\frac{\\exp(q_i^{\\top} K_j / \\sqrt{d})}{\\sum_{t=1}^{\\lceil i/B \\rceil} \\exp(q_i^{\\top} K_t / \\sqrt{d})},"
    "\\qquad o_i = \\sum_{j=1}^{\\lceil i/B \\rceil} V_j A_{ij}^{\\top} $$\n\n"
    "其中 $K_j, V_j$ 是第 $j$ 个 **KV 块**($B$ 个 token 的 K/V 向量矩阵)。"
    "每个块内部连续(访存友好),块之间靠块表跳转——这是它区别于「指针式拼接」方案的本质。\n\n"
    "vLLM 的 PagedAttention kernel(`csrc/attention/attention_kernels.cu`)把物理块号直接算进指针:\n\n"
    "```\n"
    "k_ptr = k_cache + physical_block_number * kv_block_stride\n"
    "               + kv_head_idx * kv_head_stride\n"
    "               + physical_block_offset * x\n"
    "```\n\n"
    "`physical_block_number` 正是块表里查到的物理块号,`physical_block_offset` 是块内偏移——"
    "与本课 `slot = P × B + off` 完全同构。论文 §4.3 的 decode 流程:prefill 只为 prompt 分配必要的块,\n"
    "decode 每步只在新块空位写 KV、块满再申请新物理块、更新块表。"
)

nb.md(
    "## 8. 🖥️ Streamlit 动态演示:交互式分页映射\n\n"
    "输入序列长度与块大小,实时得到:块表表格(逻辑块 → 物理块 + token 区间)、物理块占用热图、\n"
    "以及任意 token 的 slot 落点(逻辑块 → 物理块 → 偏移 → 绝对 slot)。\n\n"
    "### 📜 App 完整源码(`app_10_paged_demo.py`)"
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
    "    print(\"当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_10_paged_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_10_paged_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_10_paged_demo.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 。\n\n"
    "🔍 试试:序列 100、块大小 16 → 7 个逻辑块;把物理块总数调到 5,观察「物理块不足」的报错。"
)

wrapup(
    nb,
    summary=[
        "PagedAttention 把 OS 虚拟内存整套搬到 KV Cache:块=页、token=字节、请求=进程、块表=页表",
        "N_logical = ⌈S/B⌉;第 i 个逻辑块映射到任意物理块,块内连续、块间散落",
        "slot = P × B + off 两步 O(1) 定位任意 token;torch 实测 scatter→gather 数据逐位一致",
        "paged_map 生成的真实随机页表验证:物理块越乱,块表越不可少,但寻址代价恒为 O(1)",
        "论文 §4.1 块级注意力逐块计算合并;vLLM kernel 用 physical_block_number 直接算指针,与 slot 公式同构",
    ],
    practice=[
        "给 PageTable 加 free():释放物理块并支持二次分配,验证 slot 定位在新映射下仍然正确",
        "把 PageTable 的取块策略改成 random-fit(随机选空闲块),看热图更「凌乱」但寻址依然正确",
        "给 torch 验证加第 2 层:同时 scatter K 和 V,再按块表做一次真实的注意力(softmax(Q K^T)V)并对比连续实现",
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM Paged Attention 设计文档", "https://docs.vllm.ai/en/latest/design/paged_attention.html"),
        ("vLLM 源码: block_pool.py", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch02\10_pagedattention_core.ipynb")