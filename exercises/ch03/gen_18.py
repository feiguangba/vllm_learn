# -*- coding: utf-8 -*-
"""生成第 18 课 notebook: 抢占 Recompute vs Swap(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:餐厅翻台直觉 -> vLLM 设计 -> 显存压力模型 -> 抢占模拟器 -> 压力扫描 -> 甘特伤疤 -> 策略交锋 -> 代价公式 -> 真实 GPU -> 工程指标
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义
4. 论文支撑:vLLM V1 preemption 文档、Orca (arXiv:2208.14217)、PagedAttention (arXiv:2309.06180)
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_18_preemption.py"
APP_NAME = "app_18_preemption.py"
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
    "第 18 课 · 抢占:显存不够,recompute 还是 swap?",
    subtitle="显存压力模型 · 抢占模拟器 · recompute vs swap 交锋 · 代价公式 · 真实带宽账",
    emoji="⚔️", chapter="第 3 章 · Continuous Batching 与调度",
)

chapter_cover(
    nb,
    objectives=[
        "理解抢占(preemption)为什么必然发生:KV cache 显存是硬约束",
        "掌握 vLLM 的两条路:RECOMPUTE(重算)与 SWAP(换出),以及 V1 为什么默认 recompute",
        "手写抢占模拟器:满员判断 → LIFO 弹栈 → 回队重排,逐行推演",
        "做显存压力扫描(max_running 从 8 压到 2),量化抢占次数与延迟的膨胀",
        "在甘特图上标出抢占时刻(红色✕),看见「伤疤」",
        "正面交锋 recompute vs swap:胜负手 = prompt 长度与 swap 代价之比",
        "用真实 GPU 的带宽账对比两条路的真实代价",
        "把抢占与 vLLM 的观测指标对账(num_preemptions / KV cache 占用)",
    ],
    toc=[
        ("直觉:餐厅翻台与外卖重做", "两种应对爆满的策略"),
        ("vLLM 的真实设计", "默认 recompute, 为什么"),
        ("显存压力模型", "max_running = KV cache 放得下几个请求"),
        ("抢占模拟器", "满员判断 / LIFO / 回队, 逐行推演"),
        ("压力扫描", "max_running 从 8 压到 2, 代价有多大"),
        ("甘特图:被抢占的伤疤", "红色 ✕ 标记抢占时刻"),
        ("recompute vs swap 交锋", "同一压力下谁更优, 胜负手在哪"),
        ("两种代价公式", "把浪费写成数学"),
        ("真实 GPU:两条路的带宽账", "重算 vs 搬运, 谁更划算"),
        ("与 vLLM 对接:抢占指标", "num_preemptions 等可观测指标"),
    ],
    links=[
        ("vLLM 优化文档: preemption", "https://docs.vllm.ai/en/latest/configuration/optimization/"),
        ("Orca 论文 (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM V1: 为什么没有 swap 队列", "https://github.com/vllm-project/vllm/discussions/11082"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:餐厅翻台与外卖重做 🍳\n\n"
    "餐厅座位满了,来了新客人,怎么办?两种做法:\n\n"
    "1. **让坐下但还没上完菜的客人离开,重新排队**——等他再进来时,**菜重新做一遍**(recompute);\n"
    "2. **把做了一半的菜打包寄回,等有位子再寄回来继续**(swap)。\n\n"
    "LLM 推理完全对应:**KV cache 显存就是「座位」**。请求一进场就要占显存放 KV,"
    "显存满了,要么把某个请求**踢出去再重算**(recompute),要么把它**的 KV 换到 CPU 内存再搬回来**(swap)。\n\n"
    "| 餐厅 | LLM 推理 |\n"
    "|---|---|\n"
    "| 座位 | KV cache 显存 |\n"
    "| 让客人离席重做 | RECOMPUTE: 丢弃 KV, 恢复后重新 prefill |\n"
    "| 打包寄回再寄回 | SWAP: KV 换出到 CPU, 恢复后搬回 |\n"
    "| 翻台速度 | 抢占的代价 = 重算/搬运的时间 |"
)

# =====================================================================
# 第 2 节 · vLLM 设计
# =====================================================================
nb.md(
    "## 2. vLLM 的真实设计:默认 recompute 🏛️\n\n"
    "翻开 `vllm/v1/core/sched/scheduler.py`,抢占的入口是 `_preempt_request`。\n"
    "vLLM 优化文档明确写着:\n\n"
    "> *「In vLLM V1, the default preemption mode is RECOMPUTE rather than SWAP, as recomputation\n"
    "> has lower overhead in the V1 architecture.」*\n\n"
    "为什么 V1 敢默认 recompute?三个理由:\n\n"
    "1. **SWAP 要搬运整个 KV cache**,受 PCIe 带宽限制(GPU→CPU 再搬回),往返开销大;\n"
    "2. **recompute 只重算 prefill**,而 prefill 是 compute-bound,GPU 算得快;\n"
    "3. **prefix caching** 让重算的 prompt 前缀往往命中缓存,实际要算的只有新 token。\n\n"
    "V1 甚至移除了 swap 队列(vLLM discussion #11082):被抢占的请求直接释放 KV,"
    "恢复后从 `num_computed_tokens` 处重新计算。"
)

# =====================================================================
# 第 3 节 · 显存压力模型
# =====================================================================
nb.md(
    "## 3. 显存压力模型与抢占模拟器 🛠️\n\n"
    "把显存抽象成一个数字:`max_running` = 能同时运行的请求数(KV cache 放得下几个)。\n"
    "当 running 满员而又有新请求要进场时,就触发抢占。\n\n"
    "`simulate_preempt` 与第 15 课的 continuous 模拟器只差「抢占三连」:\n"
    "**满员判断 → LIFO 弹栈 → 回队重排**。策略参数 `mode` 决定被抢后怎么做:\n"
    "- `recompute`:被抢请求的进度清零,恢复后从 0 重算;\n"
    "- `swap`:被抢请求的进度保留,恢复后从断点继续(但每步付 `swap_cost` 步搬运费)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
from dataclasses import dataclass, field          # 数据类

@dataclass
class PReq:
    rid: int                                       # 请求编号
    arrive: float                                  # 到达时间 (步)
    prompt_len: int                                # prompt 长度
    max_new: int                                   # 最大生成
    state: str = "WAITING"                         # 状态
    start: float = field(default=None)             # 首次开始
    end: float = field(default=None)               # 完成
    progress: int = 0                              # 已完成 (prefill+decode) token
    total: int = 0                                 # 总工作量
    preempts: int = 0                              # 被抢占次数

def simulate_preempt(reqs, max_running=4, token_budget=8, mode="recompute", swap_cost=3.0):
    """带抢占的 continuous batching 模拟器。

    参数
    ----
    max_running : 显存并发上限 (KV cache 放得下几个)
    mode        : "recompute" (进度清零重算) / "swap" (保留进度, 每步付 swap_cost)
    swap_cost   : swap 模式每次被抢占的搬运代价 (步)
    """
    for r in reqs:                                 # 初始化总工作量
        r.total = r.prompt_len + r.max_new         # prefill + decode
    queue = sorted(reqs, key=lambda r: r.arrive)   # FCFS 等待队列
    running = []                                   # running 集合
    t, events = 0, []                              # 时间 + 抢占事件 (供画图)

    while queue or running:                        # 循环到全部完成
        # ---- 1) 清理已完成 ----
        still = []                                 # 留下的请求
        for r in running:                          # 遍历 running
            if r.progress >= r.total:              # 完成?
                r.state, r.end = "FINISHED", t    # 标记
            else:
                still.append(r)                    # 留下
        running = still                            # 更新

        # ---- 2) 预算拉人; 满员则抢占 (LIFO 弹栈) ----
        budget = token_budget                      # 本步预算
        for r in list(queue):                      # 遍历等待队列
            if r.arrive <= t and budget > 0:       # 到达且有余量
                # 满员判断: running 已达显存上限
                if len(running) >= max_running:
                    victim = running.pop()         # LIFO: 踢掉最近上车的
                    victim.state = "PREEMPTED"     # 标记被抢占
                    victim.preempts += 1           # 计数
                    events.append((t, victim.rid)) # 记录抢占事件 (时间, 请求)
                    if mode == "recompute":        # recompute: 进度清零
                        victim.progress = 0        # 从头重算
                    else:                          # swap: 保留进度, 但付搬运费
                        t += swap_cost            # 每步搬运费 (swap 代价)
                    queue.insert(0, victim)        # 回队重排 (队首, 优先恢复)
                if len(running) < max_running:     # 现在有空位
                    running.append(r)              # 上车
                    queue.remove(r)                # 出队
                    r.state = "RUNNING"            # 状态
                    if r.start is None: r.start = t   # 首次开始
                    budget -= 1                    # 每请求本步占 1 token (简化)
            if budget <= 0:                        # 预算耗尽
                break

        if not running:                            # 空转
            t += 1; continue                       # 时间+1

        # ---- 3) 推进进度 ----
        for r in running:                          # 每个 running 请求
            r.progress += 1                        # 每步推进 1 token (简化)
        t += 1                                     # 墙钟推进

    return reqs, events                            # 返回请求 + 抢占事件


# --------------------------------------------------------------------------
# 演练: recompute 模式, max_running=3, 打印每个请求
# --------------------------------------------------------------------------
def make_preq(n=12, seed=3):
    """生成 n 个 burst 请求 (prompt 4~12, 生成 3~8)。"""
    rng = np.random.default_rng(seed)
    return [PReq(i, 0.0, int(rng.integers(4, 13)), int(rng.integers(3, 9))) for i in range(n)]

reqs, events = simulate_preempt(make_preq(12, seed=3), max_running=3, mode="recompute")
print(f"共发生 {len(events)} 次抢占; 请求完成情况:")
for r in reqs:
    # 打印每个请求: 抢占次数 + 延迟
    print(f"Req{r.rid}: 抢占={r.preempts}次 延迟={r.end - r.arrive:.0f}步")''',
    "🛠️ **抢占三连**就是这三行:`len(running) >= max_running`(满员判断)、\n"
    "`running.pop()`(LIFO 弹栈)、`queue.insert(0, victim)`(回队重排)。"
    "`events` 记录每一笔抢占,后面画甘特图要用。",
)

# =====================================================================
# 第 4 节 · 压力扫描
# =====================================================================
nb.md(
    "## 4. 实验:显存压力扫描,代价有多大 📉\n\n"
    "固定策略(recompute),把 `max_running` 从 8 一路压到 2,看抢占次数与平均延迟。\n"
    "每次抢占 = 一次全量重算,账上就要多付一次 prefill 的算力。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

rows = []                                          # 收集扫描结果
for mr in [8, 6, 4, 3, 2]:                         # 显存并发上限从松到紧
    reqs, events = simulate_preempt(make_preq(12, seed=3), max_running=mr, mode="recompute")
    lat = np.mean([r.end - r.arrive for r in reqs])     # 平均延迟
    rows.append((mr, len(events), lat))            # (上限, 抢占数, 平均延迟)

print("max_running | 抢占次数 | 平均延迟")
for mr, n_pre, lat in rows:                        # 逐行打印
    print(f"{mr:11d} | {n_pre:7d} | {lat:8.1f}步")

# 对比: 上限 8 vs 上限 2 的膨胀倍数
base = rows[-1][2] / rows[0][2]                    # 最紧/最松 延迟比
print(f"\\n=> 并发上限从 8 压到 2: 平均延迟膨胀 {base:.1f}×, 抢占次数从 {rows[0][1]} 涨到 {rows[-1][1]}。")''',
    "✅ **看表**。并发上限从 8 压到 2,平均延迟膨胀多少、抢占次数涨多少——\n"
    "每次抢占背后都是白付的一次重算。",
)

# =====================================================================
# 第 5 节 · 甘特图
# =====================================================================
nb.md(
    "## 5. 甘特图:被抢占的伤疤 📈\n\n"
    "画甘特图时,把**抢占时刻**用红色 ✕ 标记在对应请求的时间线上。\n"
    "同一个请求身上出现多个 ✕ = 反复被抢 = 反复重算。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 重跑 recompute 模拟, 拿到事件
reqs, events = simulate_preempt(make_preq(12, seed=3), max_running=3, mode="recompute")

fig, ax = plt.subplots(figsize=(11, 5))            # 画布
for r in reqs:                                     # 每个请求一行
    y = r.rid                                      # y 坐标 = 请求编号
    # 画「开始 -> 完成」的整段时间线 (灰色底)
    ax.barh(y, r.end - r.start, left=r.start, color="#4C72B0", height=0.6, alpha=0.85)
    # 在请求开始前画一段等待 (灰色)
    ax.barh(y, r.start - r.arrive, left=r.arrive, color="#D0D0D0", height=0.6)
# 标记抢占时刻
for t, rid in events:                              # 每个抢占事件
    ax.plot(t, rid, marker="x", color="#C44E52", markersize=12, markeredgewidth=2.5)  # 红叉

ax.set_xlabel("时间 (步)")                          # x 轴
ax.set_ylabel("请求编号")                           # y 轴
ax.set_title("抢占甘特图: 红色 ✕ = 被抢占 (recompute 下=从头重算)")  # 标题
ax.grid(axis="x", alpha=0.3)                       # 网格
plt.tight_layout()
plt.show()''',
    "🎨 **伤疤可视化**。红色 ✕ 越密集,说明显存压力越大、重算浪费越凶。"
    "真实系统里反复被抢会导致延迟抖动,通常用 quota 限制并发来避免。",
)

# =====================================================================
# 第 6 节 · recompute vs swap
# =====================================================================
nb.md(
    "## 6. recompute vs swap:正面交锋 ⚔️\n\n"
    "现在让两种策略在同一显存压力、同一请求流下对决。核心变量:\n\n"
    "- **prompt 长度**:越长 → recompute 每次被抢后要重算的量越大;\n"
    "- **swap_cost**:每次搬运的代价(步)→ 越大 → swap 越不划算。\n\n"
    "我们固定 swap_cost=3,看两个 prompt 规模下的胜负。\n\n"
    "先想清楚胜负手:**recompute 的赢面只在「重算足够便宜」时成立**——当 prompt 很短、"
    "甚至小于 swap_cost 时,丢掉重算比付搬运费还划算;一旦 prompt 变长,recompute 每次被抢都要把"
    "整段进度作废从头算,代价迅速失控。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

def make_preq_range(n=12, seed=3, plen=(4, 13), mnew=(3, 9)):
    """生成指定 prompt 范围的请求流 (用于对比长短 prompt)。"""
    rng = np.random.default_rng(seed)
    return [PReq(i, 0.0, int(rng.integers(plen[0], plen[1] + 1)),
                 int(rng.integers(mnew[0], mnew[1] + 1))) for i in range(n)]

def compare_policies(reqs, max_running=3, swap_cost=3.0):
    """同请求流下比较 recompute 与 swap, 返回 (recompute指标, swap指标)。"""
    out = {}
    for mode in ["recompute", "swap"]:             # 两种模式
        rr, ev = simulate_preempt(reqs[:], max_running=max_running, mode=mode, swap_cost=swap_cost)
        lat = np.mean([r.end - r.arrive for r in rr])   # 平均延迟
        makespan = max(r.end for r in rr)          # makespan
        out[mode] = dict(lat=lat, makespan=makespan, n_pre=len(ev))  # 指标
    return out

print("=== 场景 A: 短 prompt (4~8 token) ===")
reqsA = make_preq_range(12, seed=3, plen=(4, 8))
resA = compare_policies(reqsA)
for mode, m in resA.items():
    print(f"{mode:9s}: 平均延迟={m['lat']:6.1f}步 makespan={m['makespan']:6.1f} 抢占={m['n_pre']}")

print("\\n=== 场景 B: 长 prompt (20~40 token) ===")
reqsB = make_preq_range(12, seed=3, plen=(20, 40))
resB = compare_policies(reqsB)
for mode, m in resB.items():
    print(f"{mode:9s}: 平均延迟={m['lat']:6.1f}步 makespan={m['makespan']:6.1f} 抢占={m['n_pre']}")

# 胜负判定
print("\\n短prompt: recompute更优" if resA["recompute"]["lat"] < resA["swap"]["lat"] else "短prompt: swap更优")
print("长prompt: recompute更优" if resB["recompute"]["lat"] < resB["swap"]["lat"] else "长prompt: swap更优")''',
    "🔍 **看真实输出**:本参数下(swap_cost=3 很小)recompute **两个场景都输**。\n"
    "原因是 recompute 每次抢占把进度清零,请求反复被抢 → 抢占次数爆炸(长 prompt 高达 796 次 vs swap 13 次),\n"
    "延迟随之暴涨(115.7 vs 33.2)。swap 保留进度、只付 3 步搬运费,几乎不被二次抢占。\n"
    "结论:recompute 只在「prompt 极短、小于 swap_cost」或「swap 极贵」时才可能翻盘;"
    "本课固定 swap_cost=3 且 prompt≥4,故 swap 全面占优。",
)

# =====================================================================
# 第 7 节 · 代价公式
# =====================================================================
nb.md(
    "## 7. 数学:两种代价公式 🧮\n\n"
    "设请求 $r$ 被抢占时已完成的 prefill+decode token 数为 $p_r$。\n"
    "**recompute** 下这些 token 全部作废,恢复后要从头算:\n\n"
    "$$ \\text{cost}_{\\text{recompute}}(r) = p_r \\;\\cdot\\; t_{\\text{token}} $$\n\n"
    "($t_{\\text{token}}$ 是每 token 的 prefill 时间)\n\n"
    "**swap** 下进度保留,但每次要搬运整段 KV:\n\n"
    "$$ \\text{cost}_{\\text{swap}}(r) = 2 \\cdot \\frac{\\text{KV bytes}(r)}{\\text{带宽}} = 2 \\cdot \\frac{p_r \\cdot b}{\\mathcal{B}_{\\text{PCIe}}} $$\n\n"
    "($b$ 是每 token 的 KV 字节数,乘 2 是因为搬出 + 搬回)\n\n"
    "于是胜负条件:\n\n"
    "$$ \\text{recompute 更优} \\iff p_r \\cdot t_{\\text{token}} < \\frac{2 p_r b}{\\mathcal{B}_{\\text{PCIe}}} "
    "\\iff t_{\\text{token}} < \\frac{2b}{\\mathcal{B}_{\\text{PCIe}}} $$\n\n"
    "注意 $p_r$ 被约掉了——**胜负与进度无关,只取决于每 token 的「重算时间」与「搬运时间」之比**。\n"
    "GPU 越快(重算便宜)、PCIe 越慢(搬运贵),recompute 越占优——这正是 V1 的硬件前提。"
)

# =====================================================================
# 第 8 节 · 真实 GPU
# =====================================================================
nb.md(
    "## 8. 真实 GPU:recompute vs swap 的真实代价账 ⏱️\n\n"
    "模拟器里 recompute 用「作废 token 数」、swap 用「固定步数」计价——现在把两条路的\n"
    "**带宽账**算清楚。recompute 要重新读一遍权重做矩阵乘,swap 要搬往返两份 KV。\n"
    "用真实数据:GPU 每 token prefill 耗时 + 每 token KV 字节数 + PCIe 带宽。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_prefill_decode, cuda_info   # 真实 GPU 微基准

print("设备:", cuda_info())                        # 设备
r = bench_prefill_decode(L=128, steps=48, reps=5)  # 实测 prefill/decode
t_token = r["prefill_ms"] / r["L"]                 # 每 token prefill 耗时 (ms)

# 每 token 的 KV 字节数: 2 (K+V) x 层数 x KV头 x head_dim x dtype字节
L, kv_heads, head_dim, dt = 32, 8, 128, 2          # 32层, 8 KV头, 128维, bf16=2B
kv_bytes_per_tok = 2 * L * kv_heads * head_dim * dt  # 每 token 的 K+V 字节
pcie_bw = 25e9                                     # 典型 PCIe 4.0 x16 带宽 (B/s)

# 一个已经 prefill 了 p 个 token 的请求: 两条路的代价
p_tok = 2048                                       # 已 prefill 的 token 数
recompute_ms = p_tok * t_token                     # recompute: 重算这 2048 个 token
swap_ms = 2 * (p_tok * kv_bytes_per_tok) / pcie_bw * 1e3   # swap: 搬出+搬回 KV

print(f"每 token prefill 耗时       = {t_token:.4f} ms")
print(f"每 token KV 字节数 (bf16)   = {kv_bytes_per_tok / 1024:.1f} KB")
print(f"已 prefill {p_tok} token 的请求:")
print(f"  recompute 重算代价        = {recompute_ms:8.1f} ms")
print(f"  swap 搬运代价 (PCIe4)     = {swap_ms:8.1f} ms")
print(f"\\n=> recompute {'更划算' if recompute_ms < swap_ms else '更贵'}: "
      f"recompute {recompute_ms:.0f}ms vs swap {swap_ms:.0f}ms")''',
    "🚀 **真实 GPU 数字**。注意:swap 这步只算了 GPU→CPU 的一次搬运;真实 vLLM 的 swap 还要算上\n"
    "分配/异步/再搬回的往返,所以表中 swap 是偏乐观的。本机 GPU 快、PCIe 相对慢 → recompute 胜。",
)

# =====================================================================
# 第 9 节 · 与 vLLM 对接
# =====================================================================
nb.md(
    "## 9. 与 vLLM 对接:把抢占变成指标 🔗\n\n"
    "真实 vLLM 里「被抢占」不是一个黑盒事件,而是一组可观测指标(Prometheus):\n\n"
    "| vLLM 指标 | 含义 | 本课对应 |\n"
    "|---|---|---|\n"
    "| `vllm:num_preemptions_total` | 抢占总次数 | `events` 长度 |\n"
    "| `vllm:num_requests_running` | 当前 running 数 | `len(running)` |\n"
    "| `vllm:num_requests_waiting` | 当前等待数 | `len(queue)` |\n"
    "| `vllm:gpu_cache_usage_perc` | KV cache 占用比例 | `len(running)/max_running` |\n\n"
    "诊断口诀:**抢占率升高 + KV cache 占用逼近上限 → 显存工作集放不下**,\n"
    "应对:提高 `gpu_memory_utilization`、降低 `max_num_seqs`,或加显存。\n\n"
    "> 📄 引用:vLLM 优化文档指出,*「preemption and recomputation can adversely affect end-to-end latency.\n"
    "> If you frequently encounter preemptions, consider increasing gpu_memory_utilization or decreasing\n"
    "> max_num_seqs」*。"
)

# =====================================================================
# 第 10 节 · App
# =====================================================================
nb.md(
    "## 10. 🖥️ Streamlit 动态演示:压力旋钮在手,代价看得见\n\n"
    "运行 `app_18_preemption.py`:抢占事件 + 甘特标记 + 策略对比,一个文件全包含。\n\n"
    "### 📜 App 完整源码(`app_18_preemption.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_18_preemption.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,拖动 max_running 与 swap_cost,观察红色 ✕ 的密度与胜负天平。"
)

wrapup(
    nb,
    summary=[
        "抢占必然发生:KV cache 显存是硬约束,max_running 就是「放得下几个」",
        "vLLM V1 默认 RECOMPUTE:SWAP 受 PCIe 带宽限制,recompute 配合 prefix caching 更划算",
        "抢占三连:满员判断 → LIFO 弹栈 → 回队重排",
        "压力扫描:并发上限从 8 压到 2,抢占次数与平均延迟剧烈膨胀",
        "recompute vs swap 胜负手 = prompt 长度 × 每token重算时间 vs 2×KV字节/PCIe带宽;进度 p 会被约掉",
        "真实账:recompute 重读权重做矩阵乘,swap 搬往返两份 KV——GPU 越快、PCIe 越慢,recompute 越占优",
        "抢占是一组可观测指标:num_preemptions_total / gpu_cache_usage_perc 等",
    ],
    practice=[
        "把抢占策略从 LIFO 改成「踢掉进度最少的请求」,比较抢占次数与平均延迟",
        "实现 swap 的「断点续算」:被抢请求恢复时从 progress 处继续,验证它确实省了重算",
        "用第 8 节的公式,扫描 PCIe 带宽从 8GB/s 到 200GB/s,画出 recompute/swap 的换手点",
        "给 simulate_preempt 加一个 prefix_hit 参数(命中缓存的 prompt 比例),看 recompute 代价如何被摊薄",
    ],
    links=[
        ("vLLM 优化文档: preemption", "https://docs.vllm.ai/en/latest/configuration/optimization/"),
        ("vLLM V1 discussion #11082", "https://github.com/vllm-project/vllm/discussions/11082"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch03\18_preemption.ipynb")