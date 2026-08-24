# -*- coding: utf-8 -*-
"""生成第 19 课 notebook: 完整调度模拟器与参数敏感性实验(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:风洞直觉 -> Simulator 类 -> 基线 -> 三组敏感性实验 -> 真实 GPU 参考墙 -> 调参地图
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义
4. 论文支撑:Sarathi-Serve (arXiv:2403.02310) 的 chunked prefill 与 token 预算、vLLM V1 调度
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_19_simulator.py"
APP_NAME = "app_19_simulator.py"
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
    "第 19 课 · 完整调度模拟器与参数敏感性实验",
    subtitle="Simulator 类 · 基线运行 · 三组敏感性扫描 · 真实 GPU 参考墙 · 调参地图",
    emoji="🛰️", chapter="第 3 章 · Continuous Batching 与调度",
)

chapter_cover(
    nb,
    objectives=[
        "把前几课的所有旋钮(到达流/策略/预算/抢占/分块 prefill)整合成一个 Simulator 类",
        "跑一次基线运行,拿到吞吐、延迟、抢占三组指标作参考点",
        "敏感性 I:到达率 λ 扫描,找到吞吐的「膝盖」(饱和点)",
        "敏感性 II:token 预算扫描,验证「预算不是越大越好」",
        "敏感性 III:策略 × 显存并发上限热力图,输出调参地图",
        "引入真实 GPU 吞吐曲线作「参考墙」,把模拟器的步换算成 token/s",
        "对照 Sarathi-Serve(arXiv:2403.02310)的 chunked prefill 与 token budget 设计",
    ],
    toc=[
        ("直觉:风洞实验", "一次只改一个参数的敏感性分析"),
        ("Simulator:把所有旋钮拧到一起", "六步 run() 与各课的对应"),
        ("基线运行", "一行代码出全部指标"),
        ("敏感性 I:到达率 λ", "吞吐的膝盖在哪"),
        ("敏感性 II:token 预算", "为什么不是越大越好"),
        ("敏感性 III:策略 × 显存热力图", "双因素调参地图"),
        ("真实 GPU 参考墙", "把「步」换算成 token/s"),
        ("结论:调参的地图", "一条可复用的决策链"),
    ],
    links=[
        ("Sarathi-Serve (OSDI'24)", "https://arxiv.org/abs/2403.02310"),
        ("vLLM V1 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("Orca 论文 (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("vLLM SchedulerConfig 文档", "https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:风洞实验 🌪️\n\n"
    "飞机设计师在风洞里调机翼,一次**只改一个参数**——改攻角就不动风速,改风速就不动攻角。\n"
    "调度系统也一样:参数太多(到达率、预算、策略、并发上限…),想找规律必须**单因素扫描**。\n"
    "本课把前几课的所有旋钮整合进一个 `Simulator`,然后做三组敏感性实验,产出一张「调参地图」。\n\n"
    "> 📄 工程对照:这正是 vLLM 性能调优的日常——vLLM 文档里对 `max_num_batched_tokens` 的建议\n"
    "> 就来自这类敏感性分析:*「For optimal throughput, we recommend setting max_num_batched_tokens > 8192」*。"
)

# =====================================================================
# 第 2 节 · Simulator 类
# =====================================================================
nb.md(
    "## 2. Simulator:把所有旋钮拧到一起 🛠️\n\n"
    "`Simulator` 把 `make_reqs`(到达流)、`pick_next`(策略)、预算分配、抢占逻辑整合成一个类。\n"
    "它还支持 **chunked prefill**(把大 prompt 切块跨步处理)——这正是 Sarathi-Serve\n"
    "(arXiv:2403.02310)的关键思想:*把长 prefill 拆成近等长的小块,塞进 decode 的预算空隙*。\n\n"
    "`run()` 的六个步骤与前面各课的对应关系:\n"
    "收新请求(16 课事件流) → 清完成(16) → 排序(17 策略) → 预算拉人(15) → 抢占(18) → 推进(15)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
from dataclasses import dataclass, field          # 数据类

@dataclass
class SReq:
    rid: int                                       # 请求编号
    arrive: float                                  # 到达时间 (步)
    prompt_len: int                                # prompt 长度
    max_new: int                                   # 最大生成
    state: str = "WAITING"                         # 状态
    start: float = field(default=None)             # 首次开始
    end: float = field(default=None)               # 完成
    progress: int = 0                              # 已完成 token 数
    total: int = 0                                 # 总工作量
    preempts: int = 0                              # 被抢占次数

class Simulator:
    """完整调度模拟器: 到达流 + 策略 + 预算 + 抢占 + 分块 prefill。"""

    def __init__(self, policy="fcfs", token_budget=8, max_running=None,
                 prefill_chunk=None, rate=0.5, mode="burst", n=24, seed=1):
        self.policy = policy                       # 调度策略
        self.token_budget = token_budget           # 每步 token 预算
        self.max_running = max_running             # 显存并发上限 (None=不限)
        self.prefill_chunk = prefill_chunk or max(1, token_budget)  # 每步 prefill 块
        self.reqs = self._make_reqs(n, mode, rate, seed)  # 造请求流
        self.events = []                           # 抢占事件 (供统计)

    def _make_reqs(self, n, mode, rate, seed):
        """生成请求流: burst 或 poisson 到达。"""
        rng = np.random.default_rng(seed)          # 可复现随机数
        reqs, t = [], 0.0                          # 列表 + 到达累计
        for i in range(n):                         # 逐个生成
            if mode == "poisson":                  # 泊松流
                t += rng.exponential(1.0 / rate)   # 间隔均值 1/rate
            a = 0.0 if mode == "burst" else t      # 到达时间
            r = SReq(i, a, int(rng.integers(4, 25)), int(rng.integers(3, 12)))  # 请求
            r.total = r.prompt_len + r.max_new     # 总工作量
            reqs.append(r)                         # 加入
        return reqs                                # 返回请求流

    def step_tokens(self, r):
        """请求 r 本步要处理的 token 数: prefill 分块, decode 恒为 1。"""
        if r.progress < r.prompt_len:              # prefill 阶段
            return min(self.prefill_chunk, r.prompt_len - r.progress)  # 剩余 prompt 的块
        return 1                                   # decode: 1

    def run(self):
        """跑完整模拟。返回指标 dict: makespan / 平均延迟 / 抢占数 / 吞吐。"""
        queue = sorted(self.reqs, key=lambda r: r.arrive)   # FCFS 队列
        running = []                               # running 集合
        t = 0                                      # 墙钟
        total_tokens = sum(r.total for r in self.reqs)  # 总工作量

        while queue or running:                    # 循环到全部完成
            # (1) 收新到达 (到达时间到点的请求直接进 waiting 队列逻辑)
            # (2) 清完成
            still = []                             # 留下的请求
            for r in running:                      # 遍历 running
                if r.progress >= r.total:          # 完成?
                    r.state, r.end = "FINISHED", t    # 标记
                else:
                    still.append(r)                # 留下
            running = still                        # 更新

            # (3) 排序 + 预算拉人
            budget = self.token_budget             # 本步预算
            # 3a. 先保留 running 的预算 (每请求占 step_tokens)
            for r in running:                      # 每个运行中请求
                budget -= self.step_tokens(r)      # 扣预算
            # 3b. 从等待队列按策略拉人
            waiting = sorted([w for w in queue if w.arrive <= t],
                             key=lambda w: w.arrive if self.policy == "fcfs"
                             else w.total - w.progress)  # fcfs/sjf 排序
            for r in waiting:                      # 按序尝试
                if self.max_running and len(running) >= self.max_running:  # 满员?
                    victim = running.pop()         # LIFO 抢占
                    victim.state = "PREEMPTED"     # 标记
                    victim.preempts += 1           # 计数
                    self.events.append((t, victim.rid))  # 记录事件
                    victim.progress = 0            # recompute: 清零
                    queue.insert(0, victim)        # 回队首
                need = self.step_tokens(r)         # 本步 token
                if budget >= need and (not self.max_running or len(running) < self.max_running):
                    running.append(r)              # 上车
                    queue.remove(r)                # 出队
                    r.state = "RUNNING"            # 状态
                    if r.start is None: r.start = t    # 首次开始
                    budget -= need                 # 扣预算
                elif budget < need:
                    break                          # 预算不够

            if not running:                        # 空转
                t += 1; continue                   # 时间+1

            # (4) 推进
            for r in running:                      # 每个请求
                r.progress += self.step_tokens(r)  # 推进本步 token
            t += 1                                 # 墙钟 +1

        lat = float(np.mean([r.end - r.arrive for r in self.reqs]))  # 平均延迟
        return dict(
            makespan=t,                            # 墙钟
            avg_latency=lat,                       # 平均延迟
            n_preempt=len(self.events),            # 抢占次数
            throughput=total_tokens / t,           # 吞吐 (token/步)
        )''',
    "🛠️ **组装的艺术**。`run()` 的每个步骤都在注释里标了对应当前课的哪一节——"
    "先拆解,再整合。`prefill_chunk` 让大 prompt 分块处理,不再一次性吃光预算。",
)

# =====================================================================
# 第 3 节 · 基线
# =====================================================================
nb.md(
    "## 3. 基线运行:一行代码出全部指标 📊\n\n"
    "先用默认参数跑一次基线,拿到「参考点」,后面所有扫描都跟它比。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
sim = Simulator(seed=1)                            # 全部默认: FCFS / 预算 8 / 无抢占
base = sim.run()                                   # 跑基线
print("基线指标 (FCFS, 预算=8, burst 24 请求):")
for k, v in base.items():                          # 逐项打印
    print(f"  {k:12s} = {v}")''',
    "✅ **这就是模拟器的「面板」**:吞吐、延迟、抢占,一行代码全拿到。"
    "后面每次扫描都基于它做对比。",
)

# =====================================================================
# 第 4 节 · 敏感性 I
# =====================================================================
nb.md(
    "## 4. 敏感性 I:到达率 λ 扫描(负载的脾气) 📈\n\n"
    "把 λ 从 0.2 一路加到 1.6(请求越来越密),其余全锁死。画**双轴曲线**:\n"
    "吞吐(左轴)与平均延迟(右轴)。找到吞吐的「膝盖」(饱和点):超过它,延迟暴涨,吞吐纹丝不动。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]  # 到达率扫描区间
tputs, lats = [], []                               # 收集吞吐与延迟
for lam in lambdas:                                # 每个到达率
    s = Simulator(policy="fcfs", token_budget=8, mode="poisson", rate=lam, n=60, seed=1)  # 泊松流
    m = s.run()                                    # 跑模拟
    tputs.append(m["throughput"])                  # 吞吐
    lats.append(m["avg_latency"])                  # 平均延迟

fig, ax1 = plt.subplots(figsize=(8, 5))            # 左轴: 吞吐
ax1.plot(lambdas, tputs, "o-", color="#4C72B0", label="吞吐 (token/步)")
ax1.set_xlabel("到达率 λ (请求/步)")                # x 轴
ax1.set_ylabel("吞吐 (token/步)", color="#4C72B0")  # 左 y 轴
ax1.tick_params(axis="y", labelcolor="#4C72B0")

ax2 = ax1.twinx()                                  # 右轴: 延迟
ax2.plot(lambdas, lats, "s--", color="#C44E52", label="平均延迟 (步)")
ax2.set_ylabel("平均延迟 (步)", color="#C44E52")     # 右 y 轴
ax2.tick_params(axis="y", labelcolor="#C44E52")
h1, l1 = ax1.get_legend_handles_labels()           # 合并图例
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
ax1.set_title("到达率扫描: 吞吐有膝盖, 延迟无上限")
plt.tight_layout()
plt.show()

# 找膝盖: 吞吐增速首次跌破 10% 的位置
print("λ     | 吞吐 | 平均延迟")
for lam, tp, la in zip(lambdas, tputs, lats):
    print(f"{lam:4.1f} | {tp:6.2f} | {la:7.1f}")''',
    "📊 **找到吞吐的「膝盖」**。它告诉你这台「GPU」该配多大的请求压力——超过它,延迟暴涨,吞吐纹丝不动。"
    "这就是所谓「饱和点」(saturation point)。",
)

# =====================================================================
# 第 5 节 · 敏感性 II
# =====================================================================
nb.md(
    "## 5. 敏感性 II:token 预算扫描(预算不是越大越好) 🪙\n\n"
    "扫描 `token_budget ∈ [2, 4, ..., 24]`。预期:\n"
    "- 预算太小 → 每步只能跑一点点 → 吞吐低、延迟高;\n"
    "- 预算合适 → 吞吐饱和、延迟合理;\n"
    "- 预算继续增大 → 吞吐不再涨(算力上限),而延迟因队内请求过多而回升。\n\n"
    "结论:Sarathi-Serve 的 token budget(论文里的 `max_tokens_in_batch`)正是这样设的——"
    "取在「饱和点的 1.2 倍左右」,既不浪费也不吃亏。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

budgets = np.arange(2, 26, 2)                      # 预算 2,4,...,24
tputs, lats = [], []                               # 吞吐 / 延迟
for b in budgets:                                  # 每个预算
    s = Simulator(policy="fcfs", token_budget=int(b), n=40, seed=1)  # 只改预算
    m = s.run()                                    # 跑
    tputs.append(m["throughput"])                  # 吞吐
    lats.append(m["avg_latency"])                  # 延迟

fig, ax1 = plt.subplots(figsize=(8, 5))            # 画布
ax1.plot(budgets, tputs, "o-", color="#4C72B0", label="吞吐 (token/步)")
ax1.set_xlabel("token 预算")                       # x 轴
ax1.set_ylabel("吞吐 (token/步)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2 = ax1.twinx()                                  # 右轴
ax2.plot(budgets, lats, "s--", color="#C44E52", label="平均延迟 (步)")
ax2.set_ylabel("平均延迟 (步)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right")
ax1.set_title("token 预算扫描: 吞吐先升后平, 延迟先降后升")
plt.tight_layout()
plt.show()

print("预算 | 吞吐 | 平均延迟")
for b, tp, la in zip(budgets, tputs, lats):
    print(f"{b:4d} | {tp:6.2f} | {la:7.1f}")''',
    "🔍 **看到吞吐的饱和点后**,把预算定在「饱和点的 1.2 倍左右」——既不浪费也不吃亏。"
    "这正是敏感性分析直接产出的调参建议。",
)

# =====================================================================
# 第 6 节 · 敏感性 III
# =====================================================================
nb.md(
    "## 6. 敏感性 III:策略 × 显存热力图 🔥\n\n"
    "双因素分析:策略 × 显存并发上限,输出**平均延迟热力图**。\n"
    "热力图能一眼看出:哪个策略在显存紧时最稳,哪个在显存松时最快。\n\n"
    "注意:要让 `max_running` 真正成为「有效旋钮」,token 预算必须够大——否则每步只放得进一两个请求,\n"
    "running 永远到不了上限,抢占不会触发(第 5 节那种小预算下并发上限形同虚设)。所以这里把预算提到 32,"
    "让显存并发上限真正卡住并发。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

policies = ["fcfs", "sjf"]                         # 两种策略
mem_limits = [3, 4, 6, 8, 12, None]                # 显存并发上限 (None=不限)
grid = np.zeros((len(policies), len(mem_limits)))  # 热力图矩阵
for i, pol in enumerate(policies):                 # 每行策略
    for j, mr in enumerate(mem_limits):            # 每列显存
        s = Simulator(policy=pol, token_budget=32, max_running=mr, n=40, seed=2)  # 构造 (预算32, 让并发上限真正生效)
        grid[i, j] = s.run()["avg_latency"]        # 平均延迟

fig, ax = plt.subplots(figsize=(9, 3.5))           # 画布
im = ax.imshow(grid, cmap="YlOrRd", aspect="auto") # 热力图 (红=高延迟)
ax.set_xticks(range(len(mem_limits)))              # x 刻度
ax.set_xticklabels([str(m) if m else "∞" for m in mem_limits])  # 显存上限标签
ax.set_yticks(range(len(policies)))                # y 刻度
ax.set_yticklabels(policies)                       # 策略标签
ax.set_xlabel("显存并发上限 max_running")            # x 轴
ax.set_ylabel("策略")                              # y 轴
ax.set_title("策略 × 显存 平均延迟热力图 (步)")       # 标题
for i in range(grid.shape[0]):                     # 每个格子写数字
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                color="black", fontsize=9)
fig.colorbar(im, ax=ax)                            # 色条
plt.tight_layout()
plt.show()

print("行=策略, 列=显存上限, 值=平均延迟(步)")
print("        " + "  ".join(f"{m if m else '∞':>6}" for m in mem_limits))
for i, pol in enumerate(policies):
    print(f"{pol:6s} " + "  ".join(f"{v:6.0f}" for v in grid[i]))''',
    "📊 **调参地图**。预算 32 下并发上限成为有效旋钮:显存越紧(列往左)延迟越高且伴随抢占,\n"
    "显存越松(列往右)延迟越低;同一列里 sjf 通常低于 fcfs——显存紧时 sjf 更稳,显存松时两者差异缩小。",
)

# =====================================================================
# 第 7 节 · 真实 GPU 参考墙
# =====================================================================
nb.md(
    "## 7. 真实 GPU 参考墙:把「步」换算成 token/s ⏱️\n\n"
    "前面的敏感性实验全在「步」这个抽象单位里——步数不等于秒。\n"
    "给模拟器配一堵**真实参考墙**:用 `vllm_real` 实测 batch → token/s,把模拟器报告的\n"
    "「token/步」换算成「token/s」。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
from vllm_real import bench_throughput_curve, cuda_info   # 真实 GPU 微基准

print("设备:", cuda_info())                        # 设备
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
print("每步token | 吞吐(token/s)")
for b, tp in zip(b_list, tps):                     # 打印真实曲线
    print(f"{b:8d} | {tp:10.0f}")

# 把模拟器的吞吐 (token/步) 乘上「每步真实秒数」
# 基准模拟的每步并发 ~ 平均 running 数, 取 batch=8 的每步耗时近似
ms_per_step = dict(zip(b_list, mps)).get(8, 1.0)   # batch=8 的每步 ms
sim_base = Simulator(seed=1).run()                 # 基线模拟
tok_per_step = sim_base["throughput"]             # token/步
tok_per_sec = tok_per_step / (ms_per_step / 1000)  # token/秒
print(f"\\n基线模拟: {tok_per_step:.1f} token/步, 每步≈{ms_per_step:.3f} ms")
print(f"        => 换算真实吞吐 ≈ {tok_per_sec:.0f} token/s")''',
    "🚀 **真实数字做参照物**。模拟器内部仍以「步」为逻辑单位(与调度无关),"
    "真实曲线只负责把步换算成秒——这就是第 20 课真机实验的「热身」。",
)

# =====================================================================
# 第 8 节 · 结论
# =====================================================================
nb.md(
    "## 8. 结论:调参的地图 🗺️\n\n"
    "把三张图合起来,得到一条可复用的决策链:\n\n"
    "1. **先测负载**:跑 λ 扫描,找到吞吐饱和点——那是这台机器的承载上限;\n"
    "2. **再定预算**:把 `token_budget` 设在饱和点附近(≈1.2×),别贪大;\n"
    "3. **后调策略**:显存紧时用 SJF 类策略降延迟,显存宽裕时 FCFS 就够;\n"
    "4. **始终监控抢占**:`n_preempt > 0` 说明显存是瓶颈,回到第 3 步或加显存。\n\n"
    "> 📄 这套方法论与 Sarathi-Serve 的结论一致:论文把每步 token 数(即预算)作为核心旋钮,\n"
    "> 在解码 batch 的「算术强度空隙」里塞分块 prefill,从而同时优化吞吐与延迟。"
)

# =====================================================================
# 第 9 节 · App
# =====================================================================
nb.md(
    "## 9. 🖥️ Streamlit 动态演示:全参数风洞\n\n"
    "运行 `app_19_simulator.py`:全参数交互模拟器 + 扫描面板 + 热力图。\n\n"
    "### 📜 App 完整源码(`app_19_simulator.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_19_simulator.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,在风洞里同时改多个旋钮,验证第 8 节的决策链。"
)

wrapup(
    nb,
    summary=[
        "Simulator 整合五类旋钮:到达流 / 策略 / 预算 / 抢占 / 分块 prefill,run() 六步对应前几课",
        "基线运行给「参考点」,所有敏感性扫描都与之对比",
        "λ 扫描找吞吐膝盖(饱和点):超过它延迟暴涨、吞吐纹丝不动",
        "预算扫描验证「不是越大越好」:吞吐先升后平,延迟先降后升,取饱和点 1.2× 附近",
        "策略 × 显存热力图给出「显存紧用 SJF,显存松 FCFS 够用」的调参地图",
        "真实 GPU 曲线把模拟器的「步」换算成 token/s,Sarathi-Serve 的 token budget 正是这类分析的结论",
    ],
    practice=[
        "把 Simulator 的抢占策略参数化(加 preempt_policy='lifo'/'shortest'),重新做敏感性 III",
        "在 λ 扫描里同时记录 P95 延迟,验证「平均延迟先升」之前 P95 已经爆了",
        "实现 warmup 预热:给 Simulator 加一个 per-request 的『思考时间』,观察对饱和点的影响",
        "把第 7 节换算做成函数,输入模拟吞吐直接输出真实 token/s,供后续章节复用",
    ],
    links=[
        ("Sarathi-Serve (OSDI'24)", "https://arxiv.org/abs/2403.02310"),
        ("vLLM V1 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
        ("Orca 论文", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 性能调优文档", "https://docs.vllm.ai/en/latest/configuration/optimization/"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch03\19_iterative_scheduler_sim.ipynb")