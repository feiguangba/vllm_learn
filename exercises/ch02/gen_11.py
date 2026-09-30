# -*- coding: utf-8 -*-
"""生成第 11 课 notebook: Block Table 与 Slot Mapping(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md 与 gen_07.py 金标准):
1. 由浅入深:每请求一张表的直觉 -> BlockTable 三字段定义 -> compute_slot_mapping 手工推演 ->
   torch 张量版数值验证(多请求并发) -> real_ops.paged_map 真实映射 -> 调度器生命周期 -> 小结
2. 每一行代码都有 inline 注释;每个张量打印 shape 并标注维度含义
3. 论文支撑:PagedAttention (arXiv:2309.06180) §4.2 块表/§4.3 解码流程、vLLM v1/worker/block_table.py 源码
4. 复用 real_ops.paged_map 生成 req_to_blocks 与 slot_mapping
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_11_block_table.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 11 课 · Block Table 与 Slot Mapping:多请求的并发内存账",
    subtitle="BlockTable 三字段 · slot_mapping 算法 · torch 多请求验证 · 调度器生命周期",
    emoji="🔗", chapter="第 2 章 · KV Cache 与 PagedAttention",
)

chapter_cover(
    nb,
    objectives=[
        "理解 BlockTable 的三个核心字段:block_ids(块表本体)、num_full_slots(有效槽位)、slot_mapping(逐 token 槽位)",
        "对照 vLLM v1/worker/block_table.py 的 tensor 布局:block_table (num_reqs, max_num_blocks) 与 slot_mapping (num_tokens)",
        "手写 compute_slot_mapping,与 vLLM 的 Triton kernel 公式完全对齐:slot = block_table[req, pos//B] × B + pos%B",
        "用 torch 张量做多请求并发验证:公共物理池 + 每请求独立块表,按 slot_mapping gather 数据逐位一致",
        "复用 real_ops.paged_map 生成 req_to_blocks / slot_mapping / num_full_slots 三个工程字段",
        "讲清调度器视角:块按需「长」出来、满了再申请、请求结束整表归还",
    ],
    toc=[
        ("直觉:每个人的行程单", "每请求一张表,互不干扰"),
        ("BlockTable 三字段", "block_ids / num_full_slots / slot_mapping 的精确定义"),
        ("手工实现 slot_mapping", "与 vLLM kernel 对齐的一行公式"),
        ("公共块池 + 多请求并发", "BlockPool 分配/归还,每请求一张表"),
        ("torch 数值验证", "tensor 版块表 + 按 slot gather,逐位一致"),
        ("真实映射三字段", "paged_map 生成 req_to_blocks / slot_mapping / num_full_slots"),
        ("可视化", "plotly 表格 + 热图"),
        ("调度器视角:块怎么长大", "vLLM 调度器的分配/归还/抢占"),
        ("Streamlit 动态演示", "多请求并发分配模拟器"),
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 源码: v1/worker/block_table.py", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/block_table.py"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:每个人的行程单\n\n"
    "一个旅行团住酒店:每个人一张**行程单**,上面写着「第 1 天住 301 房、第 2 天住 205 房……」\n"
    "房间(物理块)由酒店统一调配,**每天可能都换房**,但只要行程单在手,任何时候都能找到这个人。\n\n"
    "vLLM 完全一样:**每个请求一张 block table**,记录它的每个逻辑块落在哪个物理块。\n"
    "请求之间互不干扰,物理块池由调度器统一分配。第 10 课讲「一张表管一个序列」,\n"
    "本课升级到**多个请求同时并发**,还要实现逐 token 的 **slot_mapping**——attention kernel 靠它寻址。\n\n"
    "> 🏷️ **BlockTable = 请求的「行程单」:逻辑块 → 物理块的映射 + 已写满槽位数。**\n\n"
    "decode 每生成 1 个 token,就往最后一块里多占 1 个槽位;最后一块满了,再向调度器要一个新物理块——\n"
    "块是**逐块按需增长**的,而不是一次性按最大长度预分配(那会重蹈第 9 课的预留浪费)。"
)

# =====================================================================
# 第 2 节 · BlockTable 三字段
# =====================================================================
nb.md(
    "## 2. BlockTable 的三个核心字段\n\n"
    "对照 vLLM `v1/worker/block_table.py`(V1)与论文 §4.2 的概念,一张 BlockTable 至少包含:\n\n"
    "1. **`block_ids: list[int]`** —— 逻辑块 $i$ 对应的物理块号(这就是块表本体,论文里的 block table);\n"
    "2. **`num_full_slots: int`** —— 已写满 KV 的槽位数($0 \\le$ num_full_slots $\\le$ 逻辑块数×B);\n"
    "3. **`slot_mapping: list[int]`** —— 长度 = 当前 token 数;第 $k$ 个 token 的 KV 落在物理池哪个槽位。\n\n"
    "V1 工程实现里它们是**预分配的 tensor**:\n\n"
    "```\n"
    "block_table  : (num_reqs, max_num_blocks_per_req)  int32   # 每行 = 一个请求的物理块号序列\n"
    "slot_mapping : (num_batched_tokens)                int64   # 本轮批量里每个 token 的物理槽位\n"
    "num_blocks_per_row : (num_reqs)                    int32   # 每行实际用了几块(块表长度可变)\n"
    "```\n\n"
    "> ⚠️ 关键约定:`num_full_slots` 决定「这个请求的 KV 到底有多少有效」——块虽然预分配了,"
    "> **没写入的槽位不算数**(这正是后续块复用、slot 对齐的基础)。"
)

# =====================================================================
# 第 3 节 · 手工实现 slot_mapping
# =====================================================================
nb.md(
    "## 3. 手工实现 slot_mapping:与 vLLM kernel 对齐\n\n"
    "vLLM 的 `compute_slot_mapping` 在 Triton kernel 里做(`_compute_slot_mapping_kernel`),核心逻辑极朴素:\n"
    "对位置 `pos` 的 token,先除出逻辑块号、查块表拿物理块号、再模出块内偏移:\n\n"
    "$$ \\text{slot} = \\text{block\\_table}[\\text{req},\\ pos // B] \\times B + (pos \\bmod B) $$\n\n"
    "先手写一个纯 Python 版本,再验证与显式展开一致。"
)

nb.code(
    '''# -*- coding: utf-8 -*-

def compute_slot_mapping(block_ids, block_size, num_tokens):
    # 与 vLLM compute_slot_mapping 对齐的逐 token slot 计算
    # block_ids: 该请求的逻辑块 -> 物理块 序列(块表一行)
    # block_size: B, 每块 token 数
    # num_tokens: 该请求当前的有效 token 数
    slot_mapping = []                       # 逐 token 槽位结果
    for k in range(num_tokens):             # 遍历每个有效 token
        lb, off = divmod(k, block_size)     # lb = k//B(逻辑块号), off = k%B(块内偏移)
        slot_mapping.append(block_ids[lb] * block_size + off)  # slot = P×B + off
    return slot_mapping

# 演示: 3 个逻辑块映射到物理块 [5, 9, 2], 块大小 B=4, 共 10 个 token
block_ids = [5, 9, 2]                       # 块表一行: L0->P5, L1->P9, L2->P2
B = 4                                       # 块大小
sm = compute_slot_mapping(block_ids, B, 10) # 生成 10 个 token 的 slot_mapping
print(f"slot_mapping = {sm}  <- 长度 {len(sm)} = 有效 token 数")

# 展开检查: token 8 -> 逻辑块 2, 偏移 0 -> 物理块 2 -> slot = 2*4+0 = 8
print(f"token 8 展开: lb=8//{B}={8//B}, off=8%{B}={8%B}, P={block_ids[2]}, slot={block_ids[2]*B+0}")
assert sm[8] == block_ids[2] * B + 0, "映射不一致!"   # 断言与公式一致
print("✅ 断言通过: slot_mapping 与公式完全一致")''',
    "🎨 **图示**。一行 `divmod` + 一行查表,就是 vLLM 里 slot 映射的全部秘密——"
    "每个 token 的物理槽位由「它自己的逻辑位置」与「块表」共同决定。",
)

# =====================================================================
# 第 4 节 · 公共块池 + 多请求并发
# =====================================================================
nb.md(
    "## 4. 公共块池 + 多请求并发\n\n"
    "现在让多个请求**同时**向同一个块池申请物理块(first-fit),各自维护自己的 block table。\n"
    "注意关键分工:**块池是公共的,表是私有的**——这正是并发安全的基础。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import random

class BlockPool:
    # 公共物理块池: 按需发放 / 归还空闲块
    def __init__(self, n_blocks: int):
        self.free = list(range(n_blocks))  # 空闲块列表(初始全部空闲)
        self.used = []                     # (物理块号, 请求号) 记录占用

    def alloc(self, rid: int, n: int):
        # 为请求 rid 分配 n 块, 返回物理块号列表(不足则返回 None)
        if len(self.free) < n:             # 空闲块不够
            return None
        picks = self.free[:n]              # first-fit 取前 n 块
        del self.free[:n]                  # 从空闲列表移除
        self.used += [(p, rid) for p in picks]  # 记入占用表
        return picks                       # 返回分配的物理块号

    def free(self, rid: int):
        # 请求 rid 结束时归还它占用的全部物理块
        self.free += [p for p, r in self.used if r == rid]   # 归还的块回空闲表
        self.used = [(p, r) for p, r in self.used if r != rid]  # 清理占用记录

# 多请求并发分配: 4 个请求, 长度随机, 共享 32 块的公共池
random.seed(3)                             # 固定种子保证可复现
pool = BlockPool(32)                       # 公共池共 32 个物理块
tables = {}                                # 每请求的块表: rid -> (seq, block_ids)
B = 8                                      # 块大小 8
n_req = 4                                  # 4 个并发请求
for r in range(n_req):                     # 逐个请求分配
    seq = random.randint(16, 40)           # 该请求长度随机 16~40 token
    n_blocks = (seq + B - 1) // B          # 逻辑块数 = ceil(seq/B)
    ids = pool.alloc(r, n_blocks)          # 从公共池领块
    assert ids, "块池不足"                  # 池子要够用
    tables[r] = (seq, ids)                 # 记录该请求的块表
    print(f"请求 {r}: 长度 {seq:2d} token, 逻辑块 {n_blocks} 块, 物理块 {ids}")

print(f"\\n块池已用 {len(pool.used)}/32 块, 空闲 {len(pool.free)} 块")
sm0 = compute_slot_mapping(tables[0][1], B, 12)   # 请求 0 前 12 个 token 的 slot
print(f"请求 0 前 12 个 token 的 slot_mapping = {sm0}")

# 并发安全验证: 不同请求占用的物理块互不重叠
all_blocks = [pb for _, ids in tables.values() for pb in ids]   # 拼出全部占用
print(f"占用块总数 = {len(all_blocks)}, 去重后 = {len(set(all_blocks))} -> 无重叠? {len(all_blocks)==len(set(all_blocks))} ✅")''',
    "✅ **验证**。4 个请求、4 张独立块表、1 个公共块池——不同请求的物理块**零重叠**,"
    "这就是「池公共、表私有」的并发安全模型。",
)

# =====================================================================
# 第 5 节 · torch 数值验证
# =====================================================================
nb.md(
    "## 5. torch 数值验证:tensor 版块表 + 按 slot gather 🔬\n\n"
    "工程里块表和 slot_mapping 都是 **tensor**(vLLM V1:`block_table (num_reqs, max_blocks)`、"
    "`slot_mapping (num_tokens)`)。本课用 torch 复现这两者,并按 slot_mapping 从物理池 gather 出\n"
    "**一个 batch 里所有请求的全部 token** 的 KV——这正是 attention 前向前的数据准备。\n\n"
    "张量形状:物理池 `(num_blocks, block_size, num_heads, head_dim)`,gather 结果 `(total_tokens, num_heads, head_dim)`。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch

# --------------------------------------------------------------------------
# 1. 参数与数据: 4 个请求, 每个 3~5 个 token, 用 arange 保证可验证
# --------------------------------------------------------------------------
num_reqs = 4                       # R: 并发请求数
seq_lens = [3, 5, 4, 3]            # 每请求的 token 数(长度不一, 无需 pad)
num_heads, head_dim = 2, 8         # H, D
B = 4                              # 块大小
max_blocks = 2                     # 单请求最多 2 块(够 4 个 token 一段)
num_blocks = 8                     # 物理池 8 块

# 每请求的块表(手填): block_table[rid] = [物理块号, ...]
block_tables = [[0, 3], [1, 5], [2], [6]]   # 每请求 1~2 块, 物理块各自不同

# 物理 KV 池: (num_blocks, block_size, num_heads, head_dim)
k_pool = torch.zeros(num_blocks, B, num_heads, head_dim)  # 全部置零
print(f"物理 KV 池 shape = {tuple(k_pool.shape)}  <- (num_blocks, block_size, num_heads, head_dim)")

# 逻辑 KV: 每个 token 一个唯一值, 顺序排列(方便验证读回)
total_tokens = sum(seq_lens)                       # 全部请求 token 数 = 15
k_logical = torch.arange(total_tokens * num_heads * head_dim, dtype=torch.float32).reshape(
    total_tokens, num_heads, head_dim)             # (15, 2, 8)
print(f"逻辑 K(拼成一串) shape = {tuple(k_logical.shape)}  <- (total_tokens, num_heads, head_dim)")

# --------------------------------------------------------------------------
# 2. 逐请求逐 token 计算 slot_mapping(tensor 版), 并把 KV 写入物理池
# --------------------------------------------------------------------------
slot_mapping = []                        # 收集所有请求的 slot
token_start = 0                          # 请求在「拼串」中的起始下标
for rid in range(num_reqs):              # 遍历每个请求
    for k in range(seq_lens[rid]):       # 遍历该请求的每个 token
        lb, off = divmod(k, B)           # 逻辑块号与块内偏移
        pb = block_tables[rid][lb]       # 查该请求块表 -> 物理块号
        slot = pb * B + off              # slot = P×B + off
        slot_mapping.append(slot)        # 记入 slot_mapping
        src = token_start + k            # 该 token 在拼串中的全局下标
        k_pool[pb, off] = k_logical[src] # 把 KV 写入物理池对应槽位
    token_start += seq_lens[rid]         # 更新下一请求的起始下标
slot_map_t = torch.tensor(slot_mapping, dtype=torch.int64)   # 转 tensor
print(f"\\nslot_mapping tensor shape = {tuple(slot_map_t.shape)}  <- 长度 {len(slot_map_t)} = 全部 token 数")

# --------------------------------------------------------------------------
# 3. 按 slot_mapping gather: 把物理池里的 KV 按槽位线性索引取回
# --------------------------------------------------------------------------
kv_flat = k_pool.reshape(-1, num_heads, head_dim)      # (num_blocks*B, H, D) 线性化
print(f"物理池线性化 shape = {tuple(kv_flat.shape)}  <- (num_blocks*block_size, H, D)")
gathered = kv_flat[slot_map_t]                          # 高级索引: 按 slot 取行
print(f"gather 结果 shape   = {tuple(gathered.shape)}  <- (total_tokens, H, D)")

# --------------------------------------------------------------------------
# 4. 断言: 按 slot 取回 == 原始逻辑 KV(逐位一致)
# --------------------------------------------------------------------------
identical = torch.equal(gathered, k_logical)           # 逐元素相等?
print(f"\\n按 slot_mapping gather 与原始逻辑 K 逐位一致? {identical} ✅")''',
    "🔬 **数值验证**。slot_mapping 把「请求 × 逻辑位置」两维拍成物理池的一维下标;"
    "torch 高级索引 `kv_flat[slot_mapping]` 一次取回全部 15 个 token——数据逐位一致,"
    "这正是 vLLM attention 前向拿 KV 的方式。",
)

# =====================================================================
# 第 6 节 · 真实映射三字段
# =====================================================================
nb.md(
    "## 6. 真实映射三字段:paged_map 拼出工程字段 🧾\n\n"
    "真实物理块号来自**块池随机发放**,正好用 `real_ops.paged_map` 给每个请求生成一张真实(打乱)物理页表,\n"
    "再拼出 vLLM 的三个关键字段:\n\n"
    "- **`req_to_blocks[rid]`**:请求 → 该请求按序占用的物理块号列表(块表本体);\n"
    "- **`slot_mapping`**:每个 token 落在物理池的槽位($P\\times B+\\text{off}$),attention 靠它 gather;\n"
    "- **`num_full_slots`**:已写满 KV 的有效 token 数(未写的不算数,是块复用/对齐的地基)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 OpenMP 冲突
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch02")
from real_ops import paged_map                         # 随机物理页表生成器

B = 16                                                 # 块大小(对齐 vLLM 默认)
seqs = {0: 20, 1: 40, 2: 5}                            # rid -> 序列长度

# ---- 字段 1: req_to_blocks(每请求一张真实打乱的块表) ----
req_to_blocks = {}                                     # 请求 -> 物理块号列表
for rid, seq in seqs.items():                          # 逐个请求
    pm = paged_map(seq=seq, block_size=B, n_phys=12, seed=rid)   # 随机映射
    req_to_blocks[rid] = [pm["mapping"][lb] for lb in range(pm["n_logical"])]  # 按序取物理块
    print(f"请求 {rid}: seq={seq:2d} -> block_ids(物理) = {req_to_blocks[rid]}")

# ---- 字段 2: 请求 0 的 slot_mapping(逐 token) ----
rid0 = 0                                               # 挑请求 0
blocks = req_to_blocks[rid0]                           # 它的块表
slot_mapping = [blocks[k // B] * B + k % B for k in range(seqs[rid0])]  # slot = P×B + off
print(f"\\n请求 0 的 slot_mapping = {slot_mapping}")

# ---- 字段 3: num_full_slots = 已写满 KV 的有效 token 数 ----
num_full_slots = seqs[rid0]                            # 已生成 20 个 token 就是 20 个有效槽
print(f"请求 0 的 num_full_slots = {num_full_slots}  <- 20 个 token 已写, 未写的不算数")

# 抽查: 位置 k=19 -> 逻辑块 19//16=1 -> 物理块 blocks[1] -> 偏移 3 -> slot
k = 19                                                 # 最后一个 token
lb, off = divmod(k, B)                                 # 逻辑块号与偏移
print(f"\\n位置 k={k} -> 逻辑块 L{lb} -> 物理块 P{blocks[lb]} -> 偏移 {off} -> slot {slot_mapping[k]}")
assert slot_mapping[k] == blocks[lb] * B + off         # 断言与公式一致
print("断言通过: 与 slot = P×B+off 一致 ✅")''',
    "🧾 **三个工程字段**。每请求一张表、物理块各自打乱;`slot_mapping` 把任意 token 摊到固定槽位——"
    "attention 后端按它 gather,无需连续。",
)

# =====================================================================
# 第 7 节 · 可视化
# =====================================================================
nb.md(
    "## 7. 可视化:并发场景的表格与热图 📊\n\n"
    "把并发请求的块表汇总成表格,再用热图看物理块被谁占用。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

# 用第 4 节的并发结果: tables[rid] = (seq, block_ids), B=8
rows = []                                        # 表格行列表
for r, (seq, ids) in tables.items():             # 遍历 4 个请求
    for lb, pb in enumerate(ids):                # 每请求的每个逻辑块
        rows.append({"请求": f"请求{r}", "逻辑块": f"L{lb}", "物理块": f"P{pb}",
                     "token 区间": f"{lb*B+1}~{min((lb+1)*B, seq)}"})   # 块覆盖的 token 范围
fig = go.Figure(go.Table(
    header=dict(values=["请求", "逻辑块", "物理块", "token 区间"], fill_color="#2c3e50",
                font=dict(color="white"), align="center"),
    cells=dict(values=[[r_["请求"] for r_ in rows], [r_["逻辑块"] for r_ in rows],
                       [r_["物理块"] for r_ in rows], [r_["token 区间"] for r_ in rows]],
               align="center")))
fig.update_layout(title="🗺️ 4 个请求的 Block Table 汇总", height=300, template="plotly_white")
fig.show()''',
    "📊 每请求一张表、若干行;行数 = 该请求的逻辑块数;token 区间各不相同——各自独立。",
)

nb.code(
    '''# -*- coding: utf-8 -*-
from pyecharts.charts import HeatMap
from pyecharts import options as opts

# 物理块视角: x=物理块, y=请求, value=请求号(深色 = 被占用)
z = []
for r, (seq, ids) in tables.items():             # 遍历 4 个请求
    for lb, pb in enumerate(ids):                # 每个逻辑块
        z.append([pb, r, r])                     # (物理块, 请求, 值=请求号)
hm = (HeatMap()
      .add_xaxis([f"P{i}" for i in range(32)])   # 32 个物理块横轴
      .add_yaxis("请求", [f"请求{r}" for r in range(n_req)], z,  # 4 个请求纵轴
                 label_opts=opts.LabelOpts(is_show=False))
      .set_global_opts(title_opts=opts.TitleOpts(title="🔥 物理块占用热图:同一块只属于一个请求"),
                       visualmap_opts=opts.VisualMapOpts(max_=n_req - 1, is_show=False),
                       xaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=True))))
hm.render_notebook()''',
    "📊 每列一个物理块,深色 = 被占用;同一列的深色必然只属于同一个请求(池公共、表私有)。",
)

# =====================================================================
# 第 8 节 · 调度器视角
# =====================================================================
nb.md(
    "## 8. 调度器视角:块是怎么「长」出来的\n\n"
    "真实 vLLM(`v1/core/sched/scheduler.py` + 论文 §4.3)里块的生命周期大致是:\n\n"
    "1. **请求进入** → 按 prompt 一次性分配足额物理块(prefill 需要全部 prompt 的 KV);\n"
    "2. **decode 每步** → 检查最后一块是否写满:`num_full_slots` 已满则 `append` 申请新块、更新块表;\n"
    "3. **请求结束** → 整张块表的物理块**全部归还**公共池(论文 §4.3:按需分配、用完即还);\n"
    "4. **显存紧张** → 调度器**抢占(preempt)**低优先级请求,把它的块先还回来(swap 到 CPU 或重算)。\n\n"
    "这套机制让吞吐翻倍的原因很简单:块按需增长、空闲块立刻能给别人用,"
    "而第 9 课的连续分配会让大量显存「看得见摸不着」。\n\n"
    "> 📄 vLLM V1 里 `BlockTable.append_row` 把新块号写进 `block_table.np[row, start:start+n]`,"
    "> 并维护 `num_blocks_per_row`;`compute_slot_mapping` 在每次前向前把所有 token 的位置换算成槽位。"
    "> 本课 5 节的 torch 验证就是这个流程的简化镜像。"
)

nb.md(
    "## 9. 🖥️ Streamlit 动态演示:多请求并发分配模拟器\n\n"
    "拖动请求数、序列长度、块大小、物理块总数,实时生成:物理块占用热图、每个请求的 Block Table、\n"
    "以及选中请求的**逐 token slot 映射表**。\n\n"
    "### 📜 App 完整源码(`app_11_block_table.py`)"
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
    "    print(\"    请把上方源码保存为 app_11_block_table.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_11_block_table.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_11_block_table.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 。\n\n"
    "🔍 试试:请求 6 个、每请求 512 token、块大小 16 → 每请求 32 块;把物理块减到 100,观察分配失败提示。"
)

wrapup(
    nb,
    summary=[
        "BlockTable 三核心字段:block_ids(块表)、num_full_slots(有效槽位数)、slot_mapping(逐 token 物理槽位)",
        "slot_mapping 与 vLLM kernel 完全对齐:slot = block_table[req, pos//B] × B + pos%B,O(1) 定位",
        "torch 验证:4 请求并发、每请求独立块表,按 slot_mapping gather 全部 token 逐位一致——工程真相的镜像",
        "req_to_blocks / slot_mapping / num_full_slots 三个工程字段由 paged_map 可复现生成",
        "调度器生命周期:按需长块、满则 append、结束整表归还、紧张则抢占;池公共、表私有",
    ],
    practice=[
        "给 BlockPool 加 free_request(rid):模拟请求结束的全量归还,并验证归还的块能被新请求复用",
        "用 torch 把 5 节的验证扩成「每步 decode +1 token」:写新槽、块满自动申请新块,看 num_full_slots 与块表同步增长",
        "实现一个预分配(一次性按 max 长度分整块)的版本,对比两种策略的总块数与浪费",
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 源码: v1/worker/block_table.py", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/block_table.py"),
        ("vLLM 源码: v1/core/sched/scheduler.py", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch02\11_block_table_slots.ipynb")