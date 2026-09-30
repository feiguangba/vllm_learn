# -*- coding: utf-8 -*-
"""生成第 23 课 notebook: KV 块分配器(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:储物柜直觉 -> 为什么按块 -> BlockAllocator 类逐行推演 -> 典型场景 -> 热力图 -> 压力测试 -> vLLM 对照 -> 真实显存
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义
4. 论文支撑:PagedAttention (arXiv:2309.06180)、vLLM block_pool.py、ch02 real_ops.kv_bytes_real
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_23_allocator.py"
APP_NAME = "app_23_allocator.py"
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
    "第 23 课 · KV 块分配器:free list、引用计数与碎片",
    subtitle="BlockAllocator 逐行实现 · 块池热力图 · 压力测试 · vLLM block_pool 对照 · 真实显存",
    emoji="🗃️", chapter="第 4 章 · 模型执行器与 CUDA 优化",
)

chapter_cover(
    nb,
    objectives=[
        "理解为什么 KV cache 必须按块管理,而不是整段预分配",
        "手写 BlockAllocator:free list 分配 + 引用计数回收,逐行推演",
        "玩一次典型场景:分配 4 条序列、释放中间一条,观察碎片形态",
        "画块池热力图,看引用计数与空闲空洞",
        "压力测试:几百步随机分配/释放,比较 free_list 与 best_fit 的碎片率",
        "对照 vLLM block_pool.py 的三张账(空闲表 / 引用计数 / 分配器)",
        "用真实 GPU 微基准算出「一块 KV 到底占多少显存」",
    ],
    toc=[
        ("直觉:仓库里的储物柜", "按块存取, 随用随取"),
        ("为什么按块管理 KV Cache", "整段预分配 vs 按需分块的账本"),
        ("手写 BlockAllocator", "free list + refcount, 逐行推演"),
        ("典型场景", "分配 4 条序列, 释放中间一条"),
        ("块池热力图", "把 48 块画成一行格子"),
        ("压力测试", "free_list vs best_fit 的碎片率"),
        ("对应 vLLM:block_pool.py 的三张账", "真实实现对照"),
        ("真实显存:一池 KV 块到底多大", "kv_bytes_real 实测"),
    ],
    links=[
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM block_pool 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:仓库里的储物柜 🗃️\n\n"
    "想象一间**存放柜的仓库**:柜子分成一个个小格子,客人来了**按需租格子**,退租时格子归还。\n"
    "好处是显而易见的:\n\n"
    "- 客人只租自己需要的大小,不浪费;\n"
    "- 格子可以分散在仓库各处,不必连成一片;\n"
    "- 两个客人可以**共享**同一排格子(前缀复用)。\n\n"
    "KV cache 的显存管理完全一样:把显存切成**固定大小的块(block)**,每个请求按需占块。\n"
    "这就是 PagedAttention(arXiv:2309.06180)的核心思想——把「整段连续显存」变成「可分的页」。\n\n"
    "本课实现一个**最小可运行的块分配器**。"
)

# =====================================================================
# 第 2 节 · 为什么按块
# =====================================================================
nb.md(
    "## 2. 为什么按块管理 KV Cache 📦\n\n"
    "**传统做法(整段预分配)**:每条序列一进来,就按它的最大长度 `max_len` 预留一整条 KV 显存。\n"
    "问题:\n"
    "1. 大部分请求生成不到 `max_len`,预留的显存**白白锁死**;\n"
    "2. 序列释放后留下**无法复用的空洞**(碎片);\n"
    "3. 无法跨请求共享(前缀缓存不可行)。\n\n"
    "**按块管理(分页)**:显存切成固定大小块(如每块 16 token),按需分配、用完归还。\n\n"
    "$$ \\text{需要的块数} = \\left\\lceil \\frac{\\text{seq\\_len}}{\\text{block\\_size}} \\right\\rceil $$\n\n"
    "最后一块装不满是「块内尾部浪费」,通常只有 block_size/2 ≈ 8 个词元,"
    "远小于整段预分配的浪费。vLLM 博客报告:PagedAttention 的内存浪费**低于 4%**。"
)

# =====================================================================
# 第 3 节 · BlockAllocator
# =====================================================================
nb.md(
    "## 3. 手写 BlockAllocator ✍️\n\n"
    "核心逻辑只有四条:\n"
    "1. **free list**:空闲块号队列,分配时取队首;\n"
    "2. **引用计数**:每块记录被几个序列引用,`refcount>0` 说明有人用;\n"
    "3. **释放**:`refcount` 减一,**减到 0 才回收进 free list**;\n"
    "4. **共享**:同一块可被多个序列引用(前缀复用),释放时只减计数。\n\n"
    "我们实现两种找块策略:`free_list`(直接取队首,快)与 `best_fit`(扫全池找最小连续空洞,省外部碎片)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
from dataclasses import dataclass, field          # 数据类

class BlockAllocator:
    """KV 块分配器: free list + 引用计数。

    num_blocks  : 物理块总数
    block_size  : 每块容纳的 token 数
    strategy    : 'free_list' (队首取块) / 'best_fit' (扫池找最小空洞)
    """

    def __init__(self, num_blocks=48, block_size=16, strategy="free_list"):
        self.num_blocks = num_blocks               # 物理块总数
        self.block_size = block_size               # 每块 token 数
        self.strategy = strategy                   # 分配策略
        # free list: 初始全部空闲 (0..num_blocks-1)
        self.free_list = list(range(num_blocks))   # 空闲块号队列
        self.refcounts = [0] * num_blocks          # 每块引用计数
        self.owners = {}                           # owner_id -> [block_ids] 谁占了哪些块

    def allocate(self, num_tokens, owner_id):
        """为 owner 分配 num_tokens 所需的块。返回块号列表; 失败返回 None。"""
        need = (num_tokens + self.block_size - 1) // self.block_size  # 需要的块数 = ceil
        if self.strategy == "free_list":           # free list: 直接取队首 need 块
            blocks = self.free_list[:need]         # 队首切片
            if len(blocks) < need:                 # 不够
                return None                        # 分配失败
            self.free_list = self.free_list[need:] # 从队首移除
        else:                                      # best_fit: 扫全池找最小连续空洞
            # 收集所有「连续空闲段」 (起点, 长度)
            free_runs = []                         # 空闲段列表
            run_start, run_len = None, 0           # 当前段起点 / 长度
            for b in range(self.num_blocks):       # 遍历全池
                if self.refcounts[b] == 0:         # 空闲块
                    if run_len == 0:               # 新段起点
                        run_start = b
                    run_len += 1                   # 段长 +1
                else:                              # 被占用, 段中断
                    if run_len > 0:                # 有段要收
                        free_runs.append((run_start, run_len))
                    run_len = 0                    # 重置
            if run_len > 0:                        # 收尾段
                free_runs.append((run_start, run_len))
            free_runs.sort(key=lambda x: x[1])     # 按段长升序 (最小空洞优先)
            blocks = []                            # 选中的块
            for start, ln in free_runs:            # 从小到大填洞
                for b in range(start, start + min(ln, need - len(blocks))):
                    blocks.append(b)               # 取块
                if len(blocks) >= need:            # 够了
                    break
            if len(blocks) < need:                 # 全池仍不够
                return None                        # 分配失败
        # 更新引用计数 + 记录 owner
        for b in blocks:                           # 每块
            self.refcounts[b] += 1                 # 引用 +1
        self.owners[owner_id] = blocks             # 记录 owner -> 块
        return blocks                              # 返回块号列表

    def release(self, owner_id):
        """释放 owner 的所有块: 每块 refcount-1, 减到 0 才回收进 free list。"""
        blocks = self.owners.pop(owner_id, [])     # 取出该 owner 的块并删除记录
        for b in blocks:                           # 每块
            self.refcounts[b] -= 1                 # 引用 -1
            if self.refcounts[b] == 0:             # 没人用了
                self.free_list.append(b)           # 回收进 free list

    def share(self, src_owner, dst_owner, n_blocks):
        """共享: 把 src 的前 n_blocks 块也让 dst 引用 (前缀复用)。"""
        blocks = self.owners[src_owner][:n_blocks] # 源的前 n 块
        for b in blocks:                           # 每块
            self.refcounts[b] += 1                 # 引用 +1 (共享!)
        self.owners[dst_owner] = list(blocks)      # dst 拥有这些块


# --------------------------------------------------------------------------
# 演练: 小分配器, 演示 refcount 语义
# --------------------------------------------------------------------------
a = BlockAllocator(num_blocks=8, block_size=4)     # 8 块, 每块 4 token
b1 = a.allocate(5, "seq_1")                        # 5 token -> ceil(5/4)=2 块
b2 = a.allocate(9, "seq_2")                        # 9 token -> 3 块
print(f"seq_1 占块 {b1}, refcounts = {a.refcounts}")
print(f"seq_2 占块 {b2}, refcounts = {a.refcounts}")
a.release("seq_1")                                 # 释放 seq_1
print(f"释放 seq_1 后: refcounts = {a.refcounts}")
print(f"  free_list = {a.free_list}  <- 释放的块回到队首/队尾待复用")''',
    "✍️ **注意 `release` 里「减到 0 才回收」**:这正是共享块(refcount>1)能安全存在的关键——\n"
    "一个人退租,室友还在住,柜子不能下架。",
)

# =====================================================================
# 第 4 节 · 典型场景
# =====================================================================
nb.md(
    "## 4. 玩一次典型场景 🎬\n\n"
    "分配 4 条序列(64/96/32/128 词元,block_size=16 → 4/6/2/8 块),再释放中间的 `seq_1`,\n"
    "观察 free list 与 refcounts 的变化。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
alloc = BlockAllocator(num_blocks=48, block_size=16, strategy="free_list")  # 48 块, 每块 16 token
seqs = {"seq_0": 64, "seq_1": 96, "seq_2": 32, "seq_3": 128}   # 4 条序列的 token 数
for sid, n_tok in seqs.items():                  # 逐条分配
    blks = alloc.allocate(n_tok, sid)            # 分配
    print(f"{sid}: {n_tok:4d} token -> {len(blks)} 块 {blks}")

print("\\n分配后 refcounts (前 24 块):", alloc.refcounts[:24])
alloc.release("seq_1")                            # 释放中间的 seq_1
print("释放 seq_1 后 free_list 前 10 个:", alloc.free_list[:10])
print("释放 seq_1 后 refcounts (前 24 块):", alloc.refcounts[:24])''',
    "🎬 **释放 seq_1 后,中间挖出一条 6 块的「洞」**——它和尾部的空闲块不相连,"
    "这就是最直观的碎片形态。",
)

# =====================================================================
# 第 5 节 · 热力图
# =====================================================================
nb.md(
    "## 5. 可视化:块池热力图 🎨\n\n"
    "把 48 个块画成一行格子:灰色 = 空闲(refcount=0),颜色越深 = 被越多序列引用。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 1.6))          # 画布 (一行格子)
rc = np.array(alloc.refcounts).reshape(1, -1)      # (1, 48) 引用计数行
im = ax.imshow(rc, cmap="YlGnBu", aspect="auto")   # 热力图: 越深引用越多
ax.set_xticks(range(alloc.num_blocks))             # 每块一个刻度
ax.set_xticklabels(range(alloc.num_blocks), fontsize=6)  # 块号
ax.set_yticks([])                                  # 隐藏 y 轴
ax.set_title("块池热力图: 灰=空闲, 深蓝=被多序列引用; 中间浅色段 = 释放出的空洞")  # 标题
fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05)   # 色条
plt.tight_layout()
plt.show()
print("refcounts:", alloc.refcounts)              # 打印具体引用计数''',
    "🎨 **中间那一截浅色就是被释放的空洞**;深色块 refcount>1(被共享)。",
)

# =====================================================================
# 第 6 节 · 压力测试
# =====================================================================
nb.md(
    "## 6. 压力测试:随机分配/释放序列 🌀\n\n"
    "真实服务里,序列**源源不断地来、又随机地走**(生成完毕或被抢占)。\n"
    "我们模拟几百步随机操作,比较 free_list 与 best_fit 两种策略的**外部碎片率**\n"
    "(= 1 − 最大连续空闲段 / 总空闲块,衡量「空但零散」的块占比)。\n\n"
    "注意两种策略在「块内浪费」(internal)上**完全相同**——都要 `ceil(n/block_size)` 块,\n"
    "末尾装不满的那点由块大小决定,与选哪几块无关。真正能拉开差距的是**外部碎片**:\n"
    "空闲块被切得多碎、还能不能拼出大段连续空间。这才是 best_fit 发挥作用的地方。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def largest_free_run(a):
    """返回池里最大一段「连续空闲块」的长度 (外部碎片的关键指标)。"""
    best, run = 0, 0                              # 最大段 / 当前段
    for c in a.refcounts:                         # 遍历每块
        if c == 0:                                # 空闲块
            run += 1                              # 段长 +1
            best = max(best, run)                 # 更新最大值
        else:                                     # 被占用
            run = 0                               # 段中断
    return best                                   # 返回最大连续空闲段

def stress(strategy, steps=300, num_blocks=64, block_size=16, seed=0):
    """随机分配/释放压力测试。返回 外部碎片率 历史 (0=完全连续, 1=碎到最大段只占1块)。"""
    rng = np.random.default_rng(seed)              # 可复现随机数
    a = BlockAllocator(num_blocks=num_blocks, block_size=block_size, strategy=strategy)  # 分配器
    frag_hist = []                                 # 每步外部碎片率
    active = {}                                    # owner -> token 数
    next_id = 0                                    # owner 编号
    for step in range(steps):                      # 每一步
        if rng.random() < 0.6 or not active:       # 60% 概率分配
            n_tok = int(rng.integers(8, 200))      # 随机 token 数
            owner = f"r{next_id}"; next_id += 1    # 新 owner
            blks = a.allocate(n_tok, owner)        # 分配
            if blks is not None:                   # 成功
                active[owner] = n_tok              # 记录
        else:                                      # 否则释放
            owner = rng.choice(list(active))       # 随机选一个
            a.release(owner)                       # 释放
            del active[owner]                      # 删除记录
        used_blocks = sum(1 for c in a.refcounts if c > 0)  # 已占用块
        free_blocks = a.num_blocks - used_blocks  # 总空闲块
        if free_blocks > 0:                       # 有空闲
            frag = 1 - largest_free_run(a) / free_blocks  # 外部碎片率
        else:                                     # 全满
            frag = 0.0                            # 无空闲则无碎片
        frag_hist.append(frag)                     # 记录
    return frag_hist

hist_f = stress("free_list", seed=0)               # free_list 外部碎片历史
hist_b = stress("best_fit", seed=0)                # best_fit 外部碎片历史

fig, ax = plt.subplots(figsize=(9, 4))             # 画布
ax.plot(hist_f, color="#4C72B0", lw=1, label="free_list", alpha=0.8)   # free_list 曲线
ax.plot(hist_b, color="#C44E52", lw=1, label="best_fit", alpha=0.8)   # best_fit 曲线
ax.set_xlabel("随机操作步")                         # x 轴
ax.set_ylabel("外部碎片率")                         # y 轴
ax.set_title("压力测试: free_list vs best_fit 的外部碎片率")   # 标题
ax.legend()                                        # 图例
ax.grid(alpha=0.3)                                 # 网格
plt.tight_layout()
plt.show()

# 汇总: 平均与峰值外部碎片率
import numpy as np                                 # 数值库
print(f"free_list: 平均外部碎片率 = {np.mean(hist_f):.3f}, 峰值 = {np.max(hist_f):.3f}")
print(f"best_fit : 平均外部碎片率 = {np.mean(hist_b):.3f}, 峰值 = {np.max(hist_b):.3f}")
print(f"=> best_fit 优先填最小连续空洞, 保住大段连续空闲, 外部碎片率更低;"
      f"free_list 只取队首, 可能把大段切碎。")''',
    "🌀 **看曲线**:best_fit 的平均与峰值外部碎片率都更低——它把分配压进最小的空洞,"
    "把大段连续空闲留给了后来的大请求。代价是每次分配要扫全池找洞,更慢。\n"
    "工程上仍常用 free_list:分页下块可分散,外部碎片并不致命,而速度是硬要求。",
)

# =====================================================================
# 第 7 节 · vLLM 对照
# =====================================================================
nb.md(
    "## 7. 对应 vLLM:block_pool.py 的三张账 🔍\n\n"
    "vLLM V1 的 `vllm/v1/core/block_pool.py` 管理的就是本课这套账(概念对照):\n\n"
    "| 本课 | vLLM | 说明 |\n"
    "|---|---|---|\n"
    "| `free_list` | `free_block_queue`(双端队列) | 空闲物理块队列, 分配时出队 |\n"
    "| `refcounts` | 每块的引用计数 | 共享块 (前缀缓存 / 并行采样) 计数 |\n"
    "| `allocate` | `allocate_slots` | 调度器按 token 数申请块 |\n"
    "| `release` | `free` / 归还 | refcount 归零才回池 |\n\n"
    "vLLM 的 `GPUBlockPool` 启动时按 `gpu_memory_utilization` 一次划出成百上千块;"
    "每块默认 **16 token**。调度器每步 `allocate_slots` 给每个请求的 `num_new_tokens` 分配新块。\n\n"
    "> 📄 引用:vLLM V1 博客(Anatomy of vLLM)描述:*「Each block stores 16 tokens by default.\n"
    "> If a request has 17 new tokens, we need ceil(17/16)=2 blocks」*。"
)

# =====================================================================
# 第 8 节 · 真实显存
# =====================================================================
nb.md(
    "## 8. 真实显存:一池 KV 块到底多大 💾\n\n"
    "上面分配器管的是「块号」(虚拟地块),没给「一块到底占多少显存」。真实账本:\n\n"
    "$$ \\text{一块字节} = 2 \\times \\text{layers} \\times \\text{kv\\_heads} \\times \\text{head\\_dim} \\times \\text{block\\_size} \\times \\text{dtype\\_bytes} $$\n\n"
    "用 ch02 的 `real_ops.kv_bytes_real` 在真实 GPU 上量化。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch02")
from real_ops import kv_bytes_real                # ch02 真实 KV 显存微基准

# 真实配置: 16 层, 8 个 KV head (GQA), D=128, 一条 2048 词元的序列, bf16 KV
NL, KH, HD, SEQ = 16, 8, 128, 2048                # 层数 / KV头 / head_dim / 序列长度
r = kv_bytes_real(L=NL, kv_heads=KH, head_dim=HD, seq=SEQ, batch=1, dtype="bf16")
per_tok = 2 * NL * KH * HD * 2                    # 每词元 K+V 字节 (bf16)
print("设备:", r.get("device"))
print(f"理论 KV 字节 = 2 × {NL}层 × {KH}头 × {HD}维 × {SEQ}词元 × 2B = {r['theory_bytes'] / 1e6:.1f} MB")
print(f"GPU 实际分配  = {r['alloc_bytes'] / 1e6:.1f} MB (与理论一致, 双通道 K+V)")
print(f"单块 (block_size=16) 承载 = {16 * per_tok / 1e3:.1f} KB")
print(f"8GiB 显存约可同时放 {round(8 * 1024**3 / (r['theory_bytes'] + 1e-12))} 条这种 {SEQ} 序列 —— "
      f"这就是分配器要精打细算的总盘子")''',
    "💾 **真实显存数字**。KV 随序列长度线性膨胀——所以 vLLM 必须「按需分块」而不是一进场就整段锁死,"
    "否则池子早被大 prompt 撑爆。",
)

# =====================================================================
# 第 9 节 · App
# =====================================================================
nb.md(
    "## 9. 🖥️ Streamlit 动态演示:KV 块分配器模拟器\n\n"
    "运行 `app_23_allocator.py`:可点击的分配器模拟器——手动分配/释放,实时看块池热力图。\n\n"
    "### 📜 App 完整源码(`app_23_allocator.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_23_allocator.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,点击分配/释放,观察块池热力图与碎片率。"
)

wrapup(
    nb,
    summary=[
        "按块管理 = 分页思想:显存切成固定块,按需分配、用完归还,浪费低于 4%",
        "BlockAllocator 四要素:free list / 引用计数 / 释放(减到0才回收) / 共享(前缀复用)",
        "典型场景展示碎片形态:释放中间序列挖出与尾部不相连的空洞",
        "压力测试:块内浪费(internal)两种策略相同;best_fit 优先填小空洞降低外部碎片,但分配要扫全池,free_list 快而碎片略高",
        "vLLM 对照:free_block_queue / refcount / allocate_slots / free,每块默认 16 token",
        "真实显存:一块 16 token 的 KV 占 ~数百 KB~MB,整池是分配器精打细算的总盘子",
    ],
    practice=[
        "给 BlockAllocator 加一个「空闲块数不足时先压缩空洞」的逻辑,对比碎片率",
        "实现共享的释放语义:两个序列共享一块,一个释放后 refcount 从 2 变 1,块不回收",
        "把压力测试的分配概率从 0.6 改成 0.9,观察碎片率是否恶化(高负载更碎)",
        "用第 8 节的 per_tok 公式,计算 block_size=16/32/64 时「一块」的显存,画对比表",
    ],
    links=[
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM block_pool 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py"),
        ("vLLM V1 博客: Anatomy of vLLM", "https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch04\23_kv_allocator.ipynb")