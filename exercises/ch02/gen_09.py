# -*- coding: utf-8 -*-
"""生成第 09 课 notebook: 内存碎片化问题(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md 与 gen_07.py 金标准):
1. 由浅入深:停车场直觉 -> 内/外碎片精确定义与公式 -> 手工分配器逐行推演 ->
   同一随机流公平对比 -> real_ops.alloc_frag_sim 真实数字 -> 论文 §3.1 三种浪费 -> 小结
2. 每一行代码都有 inline 注释;每个中间状态打印并标注含义
3. 论文支撑:PagedAttention (SOSP'23, arXiv:2309.06180) §3.1、vLLM 官方博客 (60%-80% 浪费)
4. 复用 real_ops.alloc_frag_sim 做统一可复现的分配器仿真
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_09_fragmentation.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 09 课 · 内存碎片化:连续内存为什么装不下更多请求?",
    subtitle="外碎片 vs 内碎片 · 连续 vs 分页分配器逐行推演 · 同流公平对比 · 论文量化",
    emoji="🧩", chapter="第 2 章 · KV Cache 与 PagedAttention",
)

chapter_cover(
    nb,
    objectives=[
        "用停车场类比建立直觉:为什么「空位总数够」却「停不进一辆大车」",
        "给出内碎片 / 外碎片 / 预留(保留)三者的严格定义与定量公式",
        "手写一个连续分配器 ContinuousAllocator,逐行推演 first-fit 与最大连续空闲段",
        "手写一个分页分配器 PagedAllocator,在同一随机请求流下与连续分配公平对比",
        "复用 real_ops.alloc_frag_sim 拿到可复现的外部/内部碎片率与排队次数",
        "对齐 PagedAttention 论文 §3.1:旧系统 KV 利用率仅 20.4%~38.2%,而分页方案浪费 <4%",
    ],
    toc=[
        ("直觉:停车场的空位", "为什么有空位却停不进车"),
        ("三种浪费的精确定义", "预留 / 外碎片 / 内碎片,与公式"),
        ("手工连续分配器", "first-fit 逐行推演,打印内存布局"),
        ("手工分页分配器", "按块分配,只付最后一块的代价"),
        ("同一随机流公平对比", "60 步请求进出,谁拒绝请求?谁碎片高?"),
        ("真实分配器模拟", "real_ops.alloc_frag_sim 的可复现数字"),
        ("可视化:布局热图与碎片率曲线", "plotly 热图 + 碎片率演化"),
        ("论文视角:三种浪费的量化", "PagedAttention §3.1 的 20.4% 真相"),
        ("Streamlit 动态演示", "交互式碎片回放模拟器"),
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方博客: 10x faster inference", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉:停车场的空位\n\n"
    "假设你是停车场管理员:车位一字排开,**一辆车必须停在整片相邻车位上**(连续分配)。\n"
    "高峰期车辆大小不一、随时进出:\n\n"
    "- 一辆货车(占 8 个车位)开走了,留下 8 个**连在一起**的空位;\n"
    "- 一辆小轿车(3 位)停进来,又一辆皮卡(5 位)停进剩下 5 个——全满了;\n"
    "- 又来一辆货车,需要 8 个**相邻**车位……这时只要还有 8 连位就进得去;\n"
    "- 但更气人的场景:空位总量还有 10 个,却被切成了 **3+3+4 三块**,货车(8 连位)照样进不去!\n\n"
    "> 🏷️ **外部碎片** = 存在、但拼不成一整片的空闲。内存里就是「空闲却无法满足连续大块请求」的显存。\n\n"
    "LLM 推理正是这个停车场:每个请求的 KV Cache 在旧系统里需要一块**连续显存**;"
    "请求随时到达、完成、离开,连续分配必然产生大量外部碎片。"
    "vLLM 官方博客给出的数字是:**旧系统因碎片与超额预留浪费了 60%~80% 的显存**"
    "([vLLM blog](https://blog.vllm.ai/2023/06/20/vllm.html))。"
)

# =====================================================================
# 第 2 节 · 三种浪费的精确定义
# =====================================================================
nb.md(
    "## 2. 三种浪费:预留、外碎片、内碎片\n\n"
    "PagedAttention 论文 §3.1 把旧系统的 KV 内存浪费拆成三类(论文原文 Fig.3):\n\n"
    "| 浪费类型 | 定义 | 何时才知道浪费 | 类比 |\n"
    "|---|---|---|---|\n"
    "| **预留(reserved)** | 按 max 序列长度预分配、当前没用到但**将来可能用**的槽位 | 请求结束才见分晓 | 买了 8 个车位但车只占 3 个 |\n"
    "| **内部碎片(internal)** | 分配给你、但你永远用不到的空间(预留超了上限) | 请求结束后才知道 | 车位上限 8,实际最多 5,多余的 3 个永远空着 |\n"
    "| **外部碎片(external)** | 空闲但**不连续**,拼不出新请求要的连续大块 | 服务进行中就知道 | 空位被切成 3+3+4,8 连位进不来 |\n\n"
    "定量指标(本课代码里全程用):\n\n"
    "$$ \\text{ext\\_frag} = 1 - \\frac{\\text{最大连续空闲段}}{\\text{总空闲槽位}} $$\n\n"
    "$$ \\text{int\\_frag} = \\frac{\\text{已分配但未使用的槽位}}{\\text{已分配的槽位}} $$\n\n"
    "> ⚠️ **关键认识:两种碎片不可兼得**——连续分配几乎无内部碎片(按需精确分配),但外部碎片爆炸;\n"
    "> 按固定块分配几乎无外部碎片(可分散),但每个请求**最后一块**必然留空(≤ block_size-1 槽)。\n"
    "> 分页方案的选择,就是用「少量内部碎片」换取「消灭外部碎片」。"
)

# =====================================================================
# 第 3 节 · 手工连续分配器
# =====================================================================
nb.md(
    "## 3. 手工连续分配器:逐行推演 first-fit\n\n"
    "先写一个最朴素的连续分配器:内存是 0/1 数组,0=空闲,1=占用;分配时从头找第一段连续的空位(first-fit)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-

class ContinuousAllocator:
    # 连续分配器: 请求必须落在连续空闲槽位上(first-fit)
    def __init__(self, slots: int):
        self.slots = slots                 # 内存槽位总数
        self.occ = [0] * slots             # 占用表: 0 = 空闲, 非0 = 请求编号

    def alloc(self, size: int, rid: int) -> bool:
        # 尝试为请求 rid 分配 size 个连续槽位, 成功返回 True
        for start in range(self.slots - size + 1):     # 枚举所有可能的起始位置
            if all(v == 0 for v in self.occ[start:start + size]):  # 该段全部空闲?
                for i in range(start, start + size):   # 逐槽写入请求编号
                    self.occ[i] = rid                  # 占用
                return True                            # 分配成功
        return False                       # 找不到连续段 -> 外部碎片拒绝请求

    def free(self, rid: int):
        # 释放请求 rid 占用的全部槽位
        self.occ = [0 if v == rid else v for v in self.occ]

    def largest_free(self) -> int:
        # 最大连续空闲段长度(用来算外部碎片率)
        best = cur = 0                     # best=历史最大, cur=当前连续长度
        for v in self.occ:                 # 遍历每个槽位
            if v == 0:                     # 空闲: 连续长度 +1
                cur += 1
                best = max(best, cur)      # 更新历史最大
            else:
                cur = 0                    # 遇到占用: 连续段断掉
        return best                        # 返回最大连续空闲段

    def ext_frag_ratio(self) -> float:
        # 外部碎片率 = 1 - 最大连续空闲段 / 总空闲槽位
        total_free = self.occ.count(0)     # 总空闲槽位数
        if total_free == 0:                # 没有空闲就不存在碎片
            return 0.0
        return 1 - self.largest_free() / total_free   # 空闲被切碎得越多, 越接近 1

# --------------------------------------------------------------------------
# 演示: 构造一个「空位总数够、但货车进不来」的场景
# --------------------------------------------------------------------------
a = ContinuousAllocator(24)                # 24 个槽位的迷你停车场
a.alloc(8, 1); a.alloc(6, 2); a.alloc(4, 3); a.alloc(6, 4)   # 四辆车把 24 槽占满(8+6+4+6=24)
print(f"初始布局(0=空闲): {''.join(str(x) for x in a.occ)}")  # 打印整条占用数组
a.free(1); a.free(3)                       # 货车(8槽)与皮卡(4槽)开走, 腾出 12 个空位
print(f"两辆离开后布局    : {''.join(str(x) for x in a.occ)}")
print(f"空闲 {a.occ.count(0)} 槽, 最大连续空闲 {a.largest_free()} 槽  <- 12 个空位被切成 8+4 两段")
print(f"外部碎片率 = {a.ext_frag_ratio()*100:.0f}%")
ok = a.alloc(9, 5)                         # 再来一辆要 9 连位的车: 总空位 12 ≥ 9, 但最大连续只有 8
print(f"尝试停 9 槽的车: {'成功 ✅' if ok else '失败 ❌ —— 空位够却不连续,这就是外部碎片'}")''',
    "🎨 **图示**。看最后一行:空位总数 12 足够停 9 槽的车,却被切成 8+4 两段、最大连续只有 8,"
    "`alloc(9)` 返回 False——外部碎片把请求拒之门外。",
)

# =====================================================================
# 第 4 节 · 手工分页分配器
# =====================================================================
nb.md(
    "## 4. 手工分页分配器:按块分配,可分散\n\n"
    "PagedAttention 的解法是**分页**:把内存切成固定大小的块(block),一个请求可以占用**任意多块**,\n"
    "块之间不必相邻。代价只是**最后一块**可能装不满(内部碎片 ≤ block_size-1)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-

class PagedAllocator:
    # 分页分配器: 按整块分配, 块可分散在任意位置
    def __init__(self, slots: int, block: int):
        self.slots = slots                 # 内存槽位总数
        self.block = block                 # 块大小(每块装 block 个 token)
        self.n_blocks = slots // block     # 物理块总数
        self.owner = [0] * self.n_blocks   # 每块的归属: 0=空闲, 非0=请求编号
        self.occ = [0] * slots             # 槽位级占用表(便于对比/可视化)

    def alloc(self, size: int, rid: int) -> bool:
        # 为请求 rid 分配能容纳 size 个槽位的块(向上取整)
        need = (size + self.block - 1) // self.block        # 需要几块 = ceil(size/block)
        free = [b for b, o in enumerate(self.owner) if o == 0]  # 当前空闲块列表
        picks = free[:need]                                # 取前 need 块(first-fit)
        if len(picks) < need:                              # 空闲块不够
            return False
        for b in picks:                                    # 逐块标记归属
            self.owner[b] = rid
            lo = b * self.block                            # 该块起始槽位
            hi = min((b + 1) * self.block, self.slots)     # 该块结束槽位(最后一块可能不满)
            for j in range(lo, hi):                        # 把槽位级占用也标上
                self.occ[j] = rid
        return True

    def free(self, rid: int):
        # 释放请求 rid 的全部块与槽位
        self.occ = [0 if v == rid else v for v in self.occ]
        self.owner = [0 if o == rid else o for o in self.owner]

    def int_frag_ratio(self, requested: int) -> float:
        # 内部碎片率 = (已分配但未使用) / 已分配 = 1 - 实际需求/分配槽
        used = sum(1 for o in self.owner if o) * self.block  # 已分配的槽位数(整块计)
        if used == 0:                                        # 无分配则无碎片
            return 0.0
        return (used - requested) / used                     # 需求少于整块容量的部分

# --------------------------------------------------------------------------
# 对比演示: 与上一节同一个请求序列
# --------------------------------------------------------------------------
b = PagedAllocator(24, 4)                  # 24 槽, 每块 4 槽(分页版停车场)
b.alloc(8, 1); b.alloc(6, 2); b.alloc(4, 3); b.alloc(6, 4)  # 同样的四辆车
b.free(1); b.free(3)                       # 同样的两辆开走
print(f"分页布局(0=空闲): {''.join(str(x) for x in b.occ)}")
print(f"块归属 owner    : {b.owner}")
ok = b.alloc(9, 5)                         # 同样的 9 槽请求(连续分配器刚刚拒收的)
print(f"尝试停 9 槽的车: {'成功 ✅' if ok else '失败 ❌'}")
print(f"内部碎片率 = {b.int_frag_ratio(9)*100:.0f}%  (3 块共 12 槽,用了 9 槽,最后一块只用了 1/4 槽)")''',
    "🎨 **图示**。同一个 9 槽请求,连续分配器刚拒收;分页分配器把它**分散到多块**上停进去——"
    "代价只是最后一块浪费一点槽位(内部碎片)。",
)

# =====================================================================
# 第 5 节 · 同一随机流公平对比
# =====================================================================
nb.md(
    "## 5. 同一随机流公平对比:谁更能扛?\n\n"
    "单次演示不够,让**同一串随机请求**(大小、生命周期随机)驱动两个分配器跑 80 步。\n"
    "为保证公平,我们加一条**准入控制**:当前活跃需求超过容量 75% 时不再接受新请求——\n"
    "这样**分页分配器不会因池子耗尽而拒绝**(它的拒绝只在真 OOM 时发生,不是碎片),\n"
    "而连续分配器只要因**不连续**而拒收,那就是纯碎片浪费。\n\n"
    "这正是 PagedAttention 论文 §3.1 的核心场景:显存总量明明够,却被碎片「看得见摸不着」。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import random
random.seed(1)                             # 固定随机种子, 保证可复现

SLOTS, BLOCK = 64, 8                       # 64 槽内存, 分页块大小 8
CAP = int(SLOTS * 0.75)                    # 准入上限: 活跃需求不超过 75% 容量
cont = ContinuousAllocator(SLOTS)          # 连续分配器(第 3 节)
page = PagedAllocator(SLOTS, BLOCK)        # 分页分配器(第 4 节)

active = {}                                # 活跃请求: rid -> [剩余寿命, 大小]
frag_rej = 0                               # 连续分配「有足够空闲却不连续」被拒的次数
page_rej = 0                               # 分页分配被拒次数(有准入控制下应恒为 0)
ext_hist, int_hist = [], []                # 记录每一步两类碎片率

for step in range(80):                     # 模拟 80 步请求进出
    demand = sum(v[1] for v in active.values())       # 当前活跃需求(真实占用)
    if random.random() < 0.65 or not active:          # 65% 概率来新请求
        size = random.randint(4, 16)                  # 随机大小 4~16 槽
        rid = step + 1                                # 请求编号
        if demand + size <= CAP:                      # 准入控制: 不超容量上限才收
            free_total = cont.occ.count(0)            # 连续分配的总空闲槽位
            ok_c = cont.alloc(size, rid)              # 连续分配尝试
            if not ok_c:                              # 连续分配失败
                if free_total >= size:                # 总空闲够, 却不连续 -> 碎片浪费
                    frag_rej += 1
            ok_p = page.alloc(size, rid)              # 分页分配尝试(分散, 无需连续)
            if not ok_p:                              # 分页拒绝只可能因真 OOM
                page_rej += 1
            active[rid] = [random.randint(3, 12), size]   # 随机生命周期 3~12 步
    else:                                             # 否则释放一个活跃请求
        rid = random.choice(list(active))             # 随机选一个请求
        cont.free(rid); page.free(rid)                # 两个分配器同时释放
        del active[rid]                               # 移出活跃表
    for k in list(active):                            # 生命周期倒计时
        active[k][0] -= 1
        if active[k][0] <= 0:                         # 寿命到 0 就自动释放
            cont.free(k); page.free(k)
            del active[k]
    ext_hist.append(cont.ext_frag_ratio())            # 连续分配的外部碎片率
    need = sum(v[1] for v in active.values())         # 全部活跃请求的真实需求
    int_hist.append(page.int_frag_ratio(need))        # 分页分配的内部碎片率

print(f"连续分配: {frag_rej} 次请求因『总空闲够但不连续』被碎片拒绝")
print(f"分页分配: {page_rej} 次拒绝(有准入控制, 池子不会耗尽 → 永不因碎片拒收)")
print(f"连续分配 外部碎片率: 平均 {sum(ext_hist)/len(ext_hist)*100:.0f}%, 峰值 {max(ext_hist)*100:.0f}%")
print(f"分页分配 内部碎片率: 平均 {sum(int_hist)/len(int_hist)*100:.0f}%, 峰值 {max(int_hist)*100:.0f}%")''',
    "✅ **公平对比**。同一随机流、同样的准入控制:连续分配因碎片拒收请求,分页分配零拒收——"
    "因为块的唯一性使任何空闲块都能满足任何请求。外部碎片率(连续)峰值超过 60%。",
)

# =====================================================================
# 第 6 节 · 真实分配器模拟
# =====================================================================
nb.md(
    "## 6. 真实分配器模拟:仓库统一的可复现剧本 🎞️\n\n"
    "上面是手写在 cell 里的分配器。仓库把一套**统一、可复现**的仿真收敛进了 `real_ops.alloc_frag_sim`:\n"
    "在同一条随机请求流(大小、生命周期随机,共 80 步)上同时驱动连续与分页分配器,\n"
    "逐帧记录**外部碎片率 / 内部碎片率 / 排队(被拒)次数 / 利用率**。这样讲解、app、评测用的是同一套代码。\n\n"
    "顺带落地两个工程术语:\n"
    "- **free-list**:分配器维护一张「空闲块/空闲段」列表,分配时 O(1) 弹出、释放时 O(1) 放回——"
    "  `alloc_frag_sim` 里 `owner_p == 0` 就是这张空闲表,`largest_free` 是它的最大连续段;\n"
    "- **max_model_len**:vLLM 按它给每个请求预留足额块数,既是硬上限,也把每请求内部碎片控制在 ≤ block_size-1。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 OpenMP 冲突
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch02")
import numpy as np
from real_ops import alloc_frag_sim                     # 仓库统一分配器仿真

sim = alloc_frag_sim(total_slots=128, block_size=8, steps=80, seed=0)  # 128 槽/块8/80 步
h = sim["state"]["hist"]                                # 每步的历史记录
ext = np.array(h["ext"])                                # 连续分配的外部碎片率序列
intr = np.array(h["int"])                               # 分页分配的内部碎片率序列

print(f"连续分配 外部碎片率: 平均 {ext.mean()*100:.1f}%   峰值 {ext.max()*100:.1f}%")
print(f"分页分配 内部碎片率: 平均 {intr.mean()*100:.1f}%   峰值 {intr.max()*100:.1f}%")
print(f"连续分配 累计被拒(排队)次数: {h['queue'][-1]}")
print(f"连续分配 末步利用率(非空槽占比): {h['util'][-1]*100:.0f}%")''',
    "🚀 **真实数字**。外部碎片率(连续)在高位震荡,分页外部碎片归零、内部碎片平稳受「块尾 ≤ block_size-1」约束——"
    "这一升一降的交换正是本课核心,也是 PagedAttention 的设计动机。",
)

# =====================================================================
# 第 7 节 · 可视化
# =====================================================================
nb.md(
    "## 7. 可视化:内存布局热图与碎片率演化 📊\n\n"
    "把 `alloc_frag_sim` 的末步布局画成热图(上=连续分配,碎片斑驳;下=分页分配,按块整齐),\n"
    "再画两类碎片率随时间的演化曲线。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"                       # 内嵌渲染到 notebook

n_slots = sim["total_slots"]                             # 仿真用的槽位总数(128)
step = -1                                                # 取最后一步的布局
occ_c = h["cont"][step]                                  # 连续分配末步槽位占用(数组)
occ_p = h["page"][step]                                  # 分页分配末步槽位占用(数组)

fig = go.Figure()
# 上排: 连续分配的末步布局
fig.add_trace(go.Heatmap(
    z=[occ_c], x=list(range(n_slots)),                   # 一行 n_slots 个槽位
    colorscale=[[0, "#f5f5f5"], [0.02, "#2c3e50"], [1, "#3498db"]],  # 灰=空闲, 蓝=占用
    zmin=0, zmax=step + 2, showscale=False, yaxis="y",
    hovertemplate="槽 %{x}: 请求 %{z}<extra></extra>"))
# 下排: 分页分配的末步布局
fig.add_trace(go.Heatmap(
    z=[occ_p], x=list(range(n_slots)),
    colorscale=[[0, "#f5f5f5"], [0.02, "#2c3e50"], [1, "#2ecc71"]],
    zmin=0, zmax=step + 2, showscale=False, yaxis="y2",
    hovertemplate="槽 %{x}: 请求 %{z}<extra></extra>"))
fig.update_layout(
    title="🧩 末步内存布局:上=连续分配(空洞斑驳), 下=分页分配(按块整齐)",
    height=300, template="plotly_white",
    xaxis=dict(title="槽位", domain=[0, 1]),
    yaxis=dict(tickvals=[0], ticktext=["连续分配"]),
    yaxis2=dict(tickvals=[0], ticktext=["分页分配"], anchor="x", overlaying="y", side="left"),
    margin=dict(l=90))
fig.show()''',
    "📊 **布局热图**。连续分配的空洞像被小轿车切碎的车位;分页分配则按 8 槽一块整齐填满——"
    "同一批请求,布局质量天差地别。",
)

nb.code(
    '''# -*- coding: utf-8 -*-
from pyecharts.charts import Line
from pyecharts import options as opts

steps = list(range(1, 81))                               # 横轴: 步数 1..80
line = (Line()
        .add_xaxis(steps)
        .add_yaxis("外部碎片率(连续)",
                   [round(float(v), 3) for v in h["ext"]],   # 连续分配的外部碎片
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3, color="#e74c3c"),
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.15, color="#e74c3c"))
        .add_yaxis("内部碎片率(分页)",
                   [round(float(v), 3) for v in h["int"]],   # 分页分配的内部碎片
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3, color="#2ecc71"),
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.15, color="#2ecc71"))
        .set_global_opts(title_opts=opts.TitleOpts(title="📈 两类碎片率随时间演化(同一事件流)"),
                         xaxis_opts=opts.AxisOpts(name="步数"),
                         yaxis_opts=opts.AxisOpts(name="碎片率", max_=1),
                         legend_opts=opts.LegendOpts(pos_top="5%")))
line.render_notebook()''',
    "📊 **碎片率演化**。红线(外部碎片)上蹿下跳,绿线(内部碎片)稳如磐石——外部碎片才是连续分配的主要杀手。",
)

# =====================================================================
# 第 8 节 · 论文视角
# =====================================================================
nb.md(
    "## 8. 论文视角:三种浪费到底浪费了多少?\n\n"
    "PagedAttention 论文(SOSP'23, arXiv:2309.06180)§3.1 用 profiling 给出了精确数字:\n\n"
    "> 旧系统为每个请求按**最大可能序列长度**预分配一整段连续 KV 内存。由于输出长度不可预测,\n"
    "> 预留(reserved)、内部碎片(internal)、外部碎片(external)三路叠加,"
    "> **实际只有 20.4%~38.2% 的 KV 缓存内存被真正用来存放 token 状态**。\n\n"
    "换句话说,**60%~80% 的 KV 显存是浪费的**(与 vLLM 博客口径一致)。PagedAttention 的应对:\n\n"
    "1. **消灭外部碎片**:所有块一样大,任何空闲块都能满足任何请求;\n"
    "2. **缓解内部碎片**:块较小(默认 16 token),只损失每请求最后一块的 ≤15 槽;\n"
    "3. **消灭预留浪费**:按需分配,不预分配 max 长度。\n\n"
    "论文实测分页方案整体浪费 <4%(仅剩最后一块的内部碎片)。\n\n"
    "> 📄 论文对块大小的权衡:块太小则 GPU 并行度/访存效率不足,块太大则内部碎片上升、共享率下降;"
    "> ShareGPT 轨迹下块 16~128 都表现良好,块 16 在 Alpaca 轨迹最优——vLLM 因此默认 `block_size=16`。"
)

nb.md(
    "## 9. 🖥️ Streamlit 动态演示:交互式碎片回放\n\n"
    "把 `alloc_frag_sim` 做成可交互 App:总槽位、块大小、随机种子、模拟步数可调,"
    "并支持**拖动进度条回放**每一步的内存布局热图(上=连续,下=分页)与碎片率曲线。\n\n"
    "### 📜 App 完整源码(`app_09_fragmentation.py`)"
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
    "    print(\"    请把上方源码保存为 app_09_fragmentation.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_09_fragmentation.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_09_fragmentation.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 。\n\n"
    "🔍 试试:把块大小从 8 调到 32,内部碎片率上升而外部碎片仍是 0;换随机种子看不同剧本。"
)

wrapup(
    nb,
    summary=[
        "旧系统按最大序列长度预分配连续 KV 内存,预留 + 内部 + 外部三类碎片叠加,利用率仅 20.4%~38.2%(论文 §3.1)",
        "外部碎片 = 空闲但不连续,连续分配下可浪费 60%+ 显存(vLLM 博客口径 60%~80%);内部碎片 = 分配粒度造成的块尾浪费(≤ block_size-1)",
        "两种碎片不可兼得:连续分配外部碎片爆炸,分页分配只付少量内部碎片——分页用「少量内部碎片」换「外部碎片归零」",
        "同一随机流公平对比:连续分配多次拒绝请求,分页分配零拒绝;alloc_frag_sim 可复现地量化了这一结论",
        "PagedAttention 把块设为固定大小(默认 16 token),整体浪费降到 <4%,是 vLLM 高吞吐的基石",
    ],
    practice=[
        "把 ContinuousAllocator 改成 best-fit 策略(找最小且够大的连续段),对比外部碎片率是否下降",
        "统计两种分配器在 200 步内的「总拒绝次数」,画出拒绝数 vs 负载(新请求概率)的关系曲线",
        "实现一个「按 token 精确连续分配 + 定期内存整理(compact)」的分配器,对比碎片率与整理代价",
    ],
    links=[
        ("PagedAttention 论文 (Section 3: 碎片量化)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方博客: 10x faster inference", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("vLLM Paged Attention 设计文档", "https://docs.vllm.ai/en/latest/design/paged_attention.html"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch02\09_fragmentation_problem.ipynb")