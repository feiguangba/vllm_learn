# -*- coding: utf-8 -*-
"""生成第 15 课 notebook: Continuous Batching(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:自助餐直觉 -> Orca 论文观察 -> token 预算 -> 模拟器逐行推演 -> 对比实验 -> 真实 GPU -> vLLM 关联
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义
4. 论文支撑:Orca (arXiv:2208.14217)、vLLM 博客、Anyscale 博客
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_15_continuous_batch.py"
APP_NAME = "app_15_continuous_batch.py"
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
    "第 15 课 · Continuous Batching:让 GPU 永远在干活",
    subtitle="iteration-level scheduling · token 预算 · 静态 vs 连续对比 · 真实 GPU 吞吐",
    emoji="🍳", chapter="第 3 章 · Continuous Batching 与调度",
)

chapter_cover(
    nb,
    objectives=[
        "理解 continuous batching 的核心:把调度粒度从「请求」降到「迭代(iteration)」",
        "读懂 Orca 论文(arXiv:2208.14217)的两个关键观察:请求完成即可离开、新请求立刻可进",
        "掌握 token 预算(token budget)的概念:每步能处理多少 token,而不是几个请求",
        "手写 simulate_continuous,逐行推演「每步动态换批」的机制",
        "同一请求流对比静态批 vs 连续批:墙钟、延迟、吞吐三项指标全面量化",
        "用真实 GPU 吞吐曲线验证「动态成员」把有效并发顶得更高",
        "把 token 预算与 vLLM 的 max_num_scheduled_tokens / max_num_batched_tokens 对上号",
    ],
    toc=[
        ("直觉:自助餐与翻台", "固定座位的「翻台」vs 随到随吃的「自助餐」"),
        ("Orca 论文:一句话点破", "iteration-level scheduling 的两个关键观察"),
        ("定义:token 预算", "为什么用 token 不用请求数——算力视角"),
        ("请求模型升级", "在 Req 上加 generated / prefilled 两个字段"),
        ("模拟器 simulate_continuous", "每步一个 token 预算,逐行推演"),
        ("对比实验", "同一请求流:静态 vs 连续,三项指标"),
        ("甘特图 + 累积完成曲线", "差距看得见的双图"),
        ("逐步动画:batch 成员怎么变", "running 集每步都在换血"),
        ("真实 GPU:动态成员 vs 固定 batch", "把「平均活跃成员」顶到吞吐曲线上更右的位置"),
        ("与 vLLM 对接:token 预算的真身", "max_num_scheduled_tokens / max_num_batched_tokens"),
    ],
    links=[
        ("Orca 论文 (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 官方博客: continuous batching", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Anyscale: 23x throughput via continuous batching", "https://www.anyscale.com/blog/continuous-batching-llm-inference"),
        ("vLLM V1 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:自助餐与翻台 🍽️\n\n"
    "第 14 课的大巴比喻里,最伤人的不是「等」,而是**整套规则**:凑满才发、整批同进退、\n"
    "到站才接下一批。把场景换成**自助餐厅**:客人随到随吃、吃完就走、位置立刻被下一位客人坐下——\n"
    "**没有「凑满一桌才上菜」这回事**。\n\n"
    "GPU 服务的类比对应如下:\n\n"
    "| 自助餐厅 | Continuous Batching |\n"
    "|---|---|\n"
    "| 客人随到随吃 | 新请求一到就有机会立刻进入 batch |\n"
    "| 吃完就走 | 生成完成的请求立刻从 batch 中移出、结果立刻返回 |\n"
    "| 位置立刻让给下一位 | 腾出的算力立刻被新请求填上 |\n\n"
    "这就是 **continuous batching(连续批处理)**,也叫 **iteration-level scheduling(迭代级调度)**\n"
    "或 **dynamic batching**。它最早由 Orca 论文正式提出并系统化。"
)

# =====================================================================
# 第 2 节 · Orca 论文
# =====================================================================
nb.md(
    "## 2. Orca 论文:一句话点破 23× 吞吐 ✍️\n\n"
    "2022 年微软的 Orca 论文(arXiv:2208.14217)提出 **iteration-level scheduling**:\n\n"
    "> 调度器在**每次迭代(一次模型前向)**结束时重新决定 batch 成员——\n"
    "> (1) 选择下一批要跑的请求; (2) 让引擎只对这些请求跑**一次迭代**; (3) 接收迭代结果。\n\n"
    "论文原文的三个关键动作: *「(1) selects requests to run next; (2) invokes the engine to execute\n"
    "one iteration for the selected requests; (3) receives execution results for the scheduled iteration」*。\n"
    "因为每个迭代结束后都会检查,**完成的请求立刻返回客户端、新到的请求立刻有机会进场**。\n\n"
    "论文另一个贡献是 **selective batching**:不同请求的 prompt 长度、已生成步数都不同,\n"
    "导致 Attention 的形状不兼容,不能直接拼成一个 batch。Orca 的做法是——**只把形状兼容的算子\n"
    "(非 Attention 的矩阵乘、LayerNorm)做批处理,Attention 按请求分别执行**。\n"
    "这个「不兼容就不强拼」的思想,是理解后面 token 预算的关键。\n\n"
    "在 GPT-3 175B 上,Orca 相比 NVIDIA FasterTransformer 取得 **36.9× 的吞吐提升**(同延迟水平)。\n"
    "> 💡 vLLM 的 continuous batching 正是这一机制的工程化实现(叠加 PagedAttention 显存优化),\n"
    "> 官方博客报告相比 HF 最高 24×、相比 TGI 2.2~2.5× 的吞吐提升。"
)

# =====================================================================
# 第 3 节 · token 预算
# =====================================================================
nb.md(
    "## 3. 定义:token 预算——为什么用 token 不用请求数 🪙\n\n"
    "如果只控制「每步最多跑几个请求」,会出问题:有的请求正在 prefill(prompt 有 1000 个 token),\n"
    "有的请求在 decode(每步 1 个 token)。同样 4 个请求,工作量可能差 250 倍。\n\n"
    "所以 Orca / vLLM 用 **token 预算** 控制每步的计算量:\n\n"
    "$$ \\sum_{r \\in \\mathcal{B}_t} n_r(t) \\;\\le\\; \\text{budget} $$\n\n"
    "- $\\mathcal{B}_t$:第 $t$ 步被调度的请求集合;\n"
    "- $n_r(t)$:请求 $r$ 在第 $t$ 步要处理的 token 数(prefill 阶段 > 1,decode 阶段 = 1);\n"
    "- budget:每步允许处理的总 token 数上界。\n\n"
    "这样,调度器对「算力」做预算,而不是对「人数」做预算。vLLM 里这个 budget 就是\n"
    "`max_num_scheduled_tokens`(通常等于 `max_num_batched_tokens`,默认 2048)。\n\n"
    "> 📄 详见 vLLM 文档 SchedulerConfig: *「Maximum number of tokens that the scheduler may issue\n"
    "> in a single iteration」*。"
)

# =====================================================================
# 第 4 节 · 请求模型
# =====================================================================
nb.md(
    "## 4. 请求模型升级:多两个字段 🎫\n\n"
    "和上一课几乎一样——新增 `generated`(已生成 token 数)与 `prefilled`(prefill 进度,分块用)。\n"
    "这两个字段让模拟器能回答:**这个请求现在处于哪个阶段、还需要几步**。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库: 生成随机请求
from dataclasses import dataclass, field          # 数据类

@dataclass
class Req:
    rid: int                                       # 请求编号 (唯一标识)
    arrive: float                                  # 到达时间 (单位: 步)
    prompt_len: int                                # prompt 长度 (prefill token 数)
    max_new: int                                   # 最大生成 token 数
    state: str = "WAITING"                         # WAITING / RUNNING / FINISHED
    start: float = field(default=None)             # 首次开始执行时间
    end: float = field(default=None)               # 完成时间
    generated: int = 0                             # 已生成的 token 数 (新增, decode 进度)
    prefilled: int = 0                             # 已完成 prefill 的 token 数 (新增, 分块用)

def make_reqs(n=10, mode="burst", rate=0.5, seed=42, plen=(5, 30), mnew=(5, 20)):
    """制造请求流 (与第 14 课相同)。"""
    rng = np.random.default_rng(seed)              # 可复现随机数发生器
    reqs, t = [], 0.0                              # 请求列表 + 累计到达时间
    for i in range(n):                             # 逐个生成请求
        if mode == "poisson":                      # 泊松流: 指数间隔
            t += rng.exponential(1.0 / rate)       # 间隔均值 = 1/rate
        a = 0.0 if mode == "burst" else t          # burst 全在 t=0
        reqs.append(Req(i, a,
                        int(rng.integers(plen[0], plen[1] + 1)),   # prompt 长度
                        int(rng.integers(mnew[0], mnew[1] + 1))))  # 最大生成
    return reqs

# 打印一批请求的完整字段, 注意新加的两个进度字段
reqs = make_reqs(n=5, seed=0)
for r in reqs:
    # 打印三要素 + 初始进度 (prefilled=0, generated=0)
    print(f"Req{r.rid}: 到达={r.arrive:.0f} prompt={r.prompt_len} 生成={r.max_new} | prefilled={r.prefilled} generated={r.generated}")''',
    "✅ **请求模型就绪**。`prefilled` 和 `generated` 就是模拟器判断「该请求这步要算几个 token」的依据:"
    "prompt 没算完 → 每步推进 prefill;prompt 算完了 → 每步生成 1 个 token。",
)

# =====================================================================
# 第 5 节 · 模拟器
# =====================================================================
nb.md(
    "## 5. 模拟器 simulate_continuous:每步一个 token 预算 🪙\n\n"
    "continuous batching 的模拟器只比静态版多了**一个概念**:token 预算。\n"
    "调度循环如下:\n\n"
    "1. **每步开始**:先把已完成/已满员的请求移出 running;\n"
    "2. **预算分配**:按到达顺序把 waiting 里的请求拉进来,直到预算用尽或请求不足;\n"
    "3. **执行一步**:每个 running 请求推进 `prefill`(若干 token)或 `decode`(1 个 token);\n"
    "4. **检查完成**:`prompt 已算完且 generated == max_new` 的请求标记 FINISHED 并移出。\n\n"
    "注意第 2 步里「预算按 token 计」:prefill 请求一次吃掉 `prompt 剩余 min(chunk, 预算剩余)` 个 token,"
    "decode 请求一次吃 1 个 token。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

def simulate_continuous(reqs, token_budget=8, prefill_chunk=None):
    """Continuous batching 模拟器: 每步按 token 预算动态换批。

    参数
    ----
    reqs          : list[Req] 请求流
    token_budget  : int       每步最多处理的 token 数 (预算)
    prefill_chunk : int|None  每步 prefill 最多处理的 chunk; None = 一次算完整个 prompt

    返回
    ----
    (reqs, history) : reqs 被改写; history 记录每步 {running, tokens}
    """
    if prefill_chunk is None:                      # 默认: prefill 不切块, 一次算完
        prefill_chunk = token_budget               # 每步最多一个完整 prompt
    queue = sorted(reqs, key=lambda r: r.arrive)   # 等待队列 (FCFS)
    running = []                                   # 正在运行的请求 (动态换血)
    t = 0                                          # 墙钟时间 (步)
    history = []                                   # 每步的 running 快照, 供画图

    while queue or running:                        # 只要还有人没完成就继续
        # ---- 第 1 步: 清理已完成/已到上限的请求 ----
        still = []                                 # 留下还要继续跑的
        for r in running:                          # 遍历当前 running
            # 完成条件: prompt 已算完 且 生成已达 max_new
            if r.prefilled >= r.prompt_len and r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t    # 标记完成并记录结束时间
            else:
                still.append(r)                    # 没完成: 继续留在 running
        running = still                            # 更新 running 集合

        # ---- 第 2 步: 用 token 预算把新请求拉进来 ----
        budget = token_budget                      # 本步剩余预算 (随请求入场递减)
        for r in list(queue):                      # 遍历等待队列
            if r.arrive <= t and budget > 0:       # 已到达 且 还有预算
                running.append(r)                  # 加入 running (随到随发)
                queue.remove(r)                    # 从等待队列移除
                r.state, r.start = "RUNNING", t if r.start is None else r.start  # 记录开始
                budget -= min(prefill_chunk, r.prompt_len)  # prefill 占用预算
            if budget <= 0:                        # 预算耗尽: 本步不再接新请求
                break

        if not running:                            # 没请求可跑 (还没到发车时间)
            t += 1                                 # 时间空转 1 步
            continue                               # 继续下一轮

        # ---- 第 3 步: 执行一步, 推进每个请求的进度 ----
        for r in running:                          # 遍历所有 running 请求
            if r.prefilled < r.prompt_len:         # 还在 prefill 阶段
                # 一次推进 min(剩余prompt, 本步可算的chunk) 个 token
                r.prefilled += min(prefill_chunk, r.prompt_len - r.prefilled)
            else:                                  # 进入 decode 阶段
                r.generated += 1                   # 每步只生成 1 个 token

        history.append({                           # 记录本步快照 (供画图)
            "t": t,                                # 时间
            "running": [r.rid for r in running],   # 当前 running 成员
            "tokens": sum(                         # 本步实际消耗的 token 数
                (min(prefill_chunk, r.prompt_len - r.prefilled) if r.prefilled < r.prompt_len
                 else 1) for r in running),
        })
        t += 1                                     # 墙钟推进 1 步
    return reqs, history                           # 返回改写后的请求 + 历史快照


# --------------------------------------------------------------------------
# 演练: burst 8 个请求, token_budget=8, 打印每个请求的开始/结束
# --------------------------------------------------------------------------
reqs, hist = simulate_continuous(make_reqs(n=8, seed=42), token_budget=8)
print("continuous batching: 每个请求的开始与完成时间")
for r in reqs:
    # 打印每个请求的调度结果 (注意 start 各不相同 = 随到随发)
    print(f"Req{r.rid}: start={r.start:.0f} end={r.end:.0f} 延迟={r.end - r.arrive:.0f} 步")''',
    "🛠️ **逐行推演**。对比第 14 课的 `simulate_static`:这里没有「整批同进退」——\n"
    "第 1 步的清理和第 2 步的补充让 running **每步都在换血**。"
    "`budget -= min(prefill_chunk, r.prompt_len)` 就是 token 预算的精髓。",
)

# =====================================================================
# 第 6 节 · 对比实验
# =====================================================================
nb.md(
    "## 6. 对比实验:同一请求流,两种调度 ⚔️\n\n"
    "同一批请求(同一个 seed 生成),分别喂给静态批(`batch_size=4`)和 continuous\n"
    "(`token_budget=8`)。比较三个指标:**墙钟时间(makespan)、平均延迟、吞吐(token/步)**。\n\n"
    "为保证公平:静态批的 batch_size=4 与连续批「预算 8 / 平均 prompt 15」对应相近的算力口径。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

def simulate_static(reqs, batch_size):
    """静态批模拟器 (第 14 课实现, 这里内联以便对照)。"""
    queue = sorted(reqs, key=lambda r: r.arrive)   # FCFS 队列
    t, batches = 0.0, []                           # 时间 + 批次记录
    while queue:                                   # 循环直到全部服务完
        batch = []                                 # 本批成员
        for r in list(queue):                      # 按到达顺序捞人
            if r.arrive <= t and len(batch) < batch_size:   # 已到达且有座位
                batch.append(r)                    # 上车
                queue.remove(r)                    # 出队
        if not batch:                              # 空转等待
            t += 1.0                               # 时间照走
            continue
        for r in batch:                            # 整批同时开始
            r.state, r.start = "RUNNING", t
        bt = max(r.prompt_len + r.max_new for r in batch)   # 木桶原理
        for r in batch:                            # 整批同时结束
            r.end, r.state = t + bt, "FINISHED"
        batches.append(dict(start=t, end=t + bt, members=[r.rid for r in batch]))
        t += bt                                    # 推进墙钟
    return reqs, batches

def report(reqs, label):
    """把一批请求汇总成指标字典。"""
    df = [(r.end - r.arrive, r.prompt_len + r.max_new) for r in reqs]  # (延迟, 工作量)
    lat = np.mean([d for d, _ in df])              # 平均延迟
    makespan = max(e for e, _ in df)               # makespan
    tput = sum(w for _, w in df) / makespan        # 吞吐 = 总token / makespan
    return dict(label=label, makespan=makespan, avg_latency=lat, tput=tput)

# 同一请求流 (seed=42, burst 16 个)
n_reqs, seed = 16, 42                              # 请求数与种子
r1 = make_reqs(n=n_reqs, seed=seed)                # 给静态批的副本
r2 = make_reqs(n=n_reqs, seed=seed)                # 给连续批的副本 (同一份请求)

simulate_static(r1, batch_size=4)                  # 静态批: batch_size=4
simulate_continuous(r2, token_budget=8)            # 连续批: token_budget=8

res = [report(r1, "静态批"), report(r2, "连续批")]  # 计算两组指标
for x in res:                                      # 打印对比
    print(f"{x['label']}: makespan={x['makespan']:.0f}步 平均延迟={x['avg_latency']:.1f}步 吞吐={x['tput']:.2f} token/步")
speed = res[1]["makespan"] / res[0]["makespan"]    # 连续批 vs 静态批 makespan 比
print(f"\\n=> 连续批 makespan 缩短 {speed:.2f}×, 平均延迟缩短 "
      f"{res[0]['avg_latency'] / res[1]['avg_latency']:.2f}×, 吞吐提升 {res[1]['tput'] / res[0]['tput']:.2f}×")''',
    "🏆 **同一批活,连续批的墙钟、平均延迟、吞吐三项全面占优**——这就是「翻台」的力量。"
    "数字会随 seed 变化,但方向几乎不变。",
)

# =====================================================================
# 第 7 节 · 甘特图 + 累积完成
# =====================================================================
nb.md(
    "## 7. 甘特图 + 累积完成曲线:差距看得见 📊\n\n"
    "数字之外,我们把两种调度画成**同一张图的两个子图**。再画一张累积完成曲线:\n"
    "横轴是时间,纵轴是「已完成请求数」。静态批是阶梯状(一批一批完成),\n"
    "连续批更平滑(随到随走)——两条线的「领先面积」就是吞吐优势。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False         # 负号

def run_sim(sim_fn, n=16, seed=42):
    """跑一种调度, 返回按时间排序的 (完成时刻) 列表。"""
    reqs = make_reqs(n=n, seed=seed)               # 造请求
    sim_fn(reqs)                                   # 执行调度
    return sorted(r.end for r in reqs)             # 每个请求的完成时刻 (升序)

# 两种调度的完成时刻序列
end_static = run_sim(lambda r: simulate_static(r, batch_size=4))        # 静态批
end_cont = run_sim(lambda r: simulate_continuous(r, token_budget=8))    # 连续批

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))   # 左右两个子图

# ---- 左图: 累积完成曲线 ----
x = np.arange(1, len(end_static) + 1)              # 1..N (第几个完成的)
ax1.step(end_static, x, where="post", color="#4C72B0", label="静态批", lw=2)
ax1.step(end_cont, x, where="post", color="#C44E52", label="连续批", lw=2)
ax1.set_xlabel("时间 (步)")                        # x 轴: 时间
ax1.set_ylabel("已完成的请求数")                   # y 轴: 完成数
ax1.set_title("累积完成曲线: 连续批领先")           # 标题
ax1.legend()                                       # 图例
ax1.grid(alpha=0.3)                                # 网格

# ---- 右图: 每个请求的延迟对比 ----
lat_s = np.array(end_static) - 0                   # 静态批延迟 (burst 到达=0)
lat_c = np.array(end_cont) - 0                     # 连续批延迟
ax2.bar(x - 0.2, lat_s, width=0.4, color="#4C72B0", label="静态批")   # 左柱
ax2.bar(x + 0.2, lat_c, width=0.4, color="#C44E52", label="连续批")   # 右柱
ax2.set_xlabel("请求编号")                          # x 轴: 请求
ax2.set_ylabel("延迟 (步)")                        # y 轴: 延迟
ax2.set_title("每个请求的延迟对比")                  # 标题
ax2.legend()                                       # 图例
ax2.grid(alpha=0.3)                                # 网格

plt.tight_layout()
plt.show()''',
    "🎨 **领先面积即吞吐优势**。左图连续批几乎全程领先;右图连续批几乎每个请求延迟都更低——"
    "因为它不用陪跑、也不用凑批。",
)

# =====================================================================
# 第 8 节 · 逐步动画
# =====================================================================
nb.md(
    "## 8. 逐步动画:batch 成员是怎么变的 🎬\n\n"
    "continuous batching 的「连续」体现在**每次迭代 batch 都不同**。我们把模拟器改成\n"
    "带日志版本,打印前 12 步的 running 成员,亲眼看到「有人下车、有人上车」。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

def simulate_continuous_log(reqs, token_budget=8, max_steps=12):
    """带日志的 continuous batching 模拟: 打印每步 running 成员。"""
    queue = sorted(reqs, key=lambda r: r.arrive)   # FCFS 等待队列
    running = []                                   # running 集合
    t = 0                                          # 时间
    while (queue or running) and t < max_steps:    # 最多跑 max_steps 步 (截断演示)
        # 清理已完成
        still = []                                 # 留下的请求
        for r in running:                          # 遍历 running
            if r.prefilled >= r.prompt_len and r.generated >= r.max_new:  # 完成?
                r.state, r.end = "FINISHED", t    # 标记完成
            else:
                still.append(r)                    # 未完成: 留下
        running = still                            # 更新集合

        # 预算拉人
        budget = token_budget                      # 本步预算
        for r in list(queue):                      # 遍历等待队列
            if r.arrive <= t and budget > 0:       # 到达且有预算
                running.append(r)                  # 上车
                queue.remove(r)                    # 出队
                if r.start is None: r.start = t    # 记录首次开始
                budget -= min(token_budget, r.prompt_len)   # prefill 占预算
            if budget <= 0:                        # 预算耗尽
                break

        if not running:                            # 空转
            t += 1; continue                       # 时间+1 跳过

        # 执行一步
        for r in running:                          # 推进每个请求
            if r.prefilled < r.prompt_len:         # prefill 阶段
                r.prefilled += min(token_budget, r.prompt_len - r.prefilled)
            else:                                  # decode 阶段
                r.generated += 1                   # 生成 1 个 token

        # 日志: 打印本步 running 成员 (前 6 个) + 各成员阶段
        members = " ".join(
            f"{r.rid}({'pre' if r.prefilled < r.prompt_len else 'dec'})"  # 阶段标记
            for r in running[:6])
        print(f"步 {t:2d}: running=[{members}] 成员数={len(running)}")
        t += 1                                     # 时间推进


simulate_continuous_log(make_reqs(n=8, seed=1), token_budget=8, max_steps=12)''',
    "🎬 **观察前 12 步**:某个请求完成(从 running 消失)的同时,新请求立刻被补进 running——"
    "batch 的成员每一行都在变。这就是「连续」二字的字面意思。",
)

# =====================================================================
# 第 9 节 · 真实 GPU
# =====================================================================
nb.md(
    "## 9. 真实 GPU:动态成员 vs 固定 batch 的吞吐差异 ⏱️\n\n"
    "模拟器说 continuous batching「吃得更饱」——但**吃饱能省多少真实算力**,只有 GPU 知道。\n"
    "我们用跨章共享库 `vllm_real` 实测:每步把 `batch` 个单 token 一起前向(连续批的雏形),\n"
    "得到 **batch → token/s** 的真实曲线。曲线上升的原因:更大的 batch 把「读权重、启动 kernel」\n"
    "等固定开销摊薄。\n\n"
    "然后做一件关键的事:统计同一请求流下,**静态批 vs 连续批的「每步平均活跃成员数」**——\n"
    "动态批赢就赢在把平均活跃成员顶在曲线上更右的位置。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info  # 跨章共享真实 GPU 微基准

print("设备:", cuda_info())                        # 打印设备信息
# 实测: batch -> (token/s, ms/步)
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
print("batch | 每步耗时(ms) | 吞吐(token/s)")
for b, ms, tp in zip(b_list, mps, tps):            # 逐行打印
    print(f"{b:5d} | {ms:10.3f} | {tp:10.0f}")''',
    "🚀 **真实 GPU 数字**。记住曲线的形状:batch 越大单位 token 越便宜。"
    "但要注意两个前提:(1) batch 真的凑得满;(2) 每一步都满载输出。",
)

nb.md(
    "### 9.2 把动态成员算进有效并发\n\n"
    "现在拼另一半:同一请求流,**动态成员**(continuous,完成一个补一个)vs **固定 batch**\n"
    "(静态批)。我们统计「每步实际活跃成员数」,看谁更贴近吞吐曲线的高位。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

# 真实曲线查表 (batch -> token/s)
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
tps_of = dict(zip(b_list, tps))                    # {batch: token/s}

# ---- 用模拟器统计「每步活跃成员数」 ----
def avg_active(sim_fn, n=16, seed=42):
    """跑模拟, 返回每步平均活跃成员数。sim_fn 接受 reqs 一个参数 (参数由闭包携带)。"""
    reqs = make_reqs(n=n, seed=seed)               # 造请求
    _, out = sim_fn(reqs)                          # 返回的第二个值: 连续批=history, 静态批=batches
    is_hist = isinstance(out, list) and bool(out) and "running" in out[0]  # 连续批判据: 有 running 键
    if is_hist:                                    # 连续批: 用 history 快照
        return float(np.mean([len(h["running"]) for h in out]))  # 平均成员数
    else:                                          # 静态批: 用 batches 记录
        steps = sum(b["end"] - b["start"] for b in out)   # 总执行步
        return float(sum(len(b["members"]) * (b["end"] - b["start"]) for b in out) / steps)

act_static = avg_active(lambda r: simulate_static(r, batch_size=4))       # 静态批平均并发
act_cont = avg_active(lambda r: simulate_continuous(r, token_budget=8))   # 连续批平均并发

# 把平均并发映射到真实吞吐曲线 (线性插值)
def tps_at(b):
    """在实测曲线上对 batch 做线性插值, 得到对应的 token/s。"""
    bs, ts = list(b_list), list(tps)               # 曲线点
    if b <= bs[0]:                                 # 小于最小 batch
        return ts[0]                               # 用最小 batch 的值
    if b >= bs[-1]:                                # 大于最大 batch
        return ts[-1]                              # 用最大 batch 的值
    for i in range(len(bs) - 1):                   # 找到夹住 b 的区间
        if bs[i] <= b <= bs[i + 1]:                # 落在这两个点之间
            frac = (b - bs[i]) / (bs[i + 1] - bs[i])   # 归一化位置
            return ts[i] + frac * (ts[i + 1] - ts[i])  # 线性插值

print(f"静态批: 平均活跃成员 = {act_static:.2f} -> 对应真实吞吐 ≈ {tps_at(act_static):.0f} token/s")
print(f"连续批: 平均活跃成员 = {act_cont:.2f} -> 对应真实吞吐 ≈ {tps_at(act_cont):.0f} token/s")
print(f"=> 连续批把平均并发从 {act_static:.1f} 抬到 {act_cont:.1f}, 真实吞吐提升 "
      f"{tps_at(act_cont) / tps_at(act_static):.2f}×")''',
    "🔢 **前因后果落到数字**。动态批赢在**把『平均活跃成员』顶在曲线上更右的位置**——"
    "静态批的值被巡航空转和整批陪跑拉低,同一个 batch 却要陪跑最长请求。"
    "配合真实曲线,这多出来的并发就是白捡的吞吐。",
)

# =====================================================================
# 第 10 节 · 与 vLLM 对接
# =====================================================================
nb.md(
    "## 10. 与 vLLM 对接:token 预算的真身 🔗\n\n"
    "模拟器里的 `token_budget` 在 vLLM 里有一个真实对应物:`max_num_scheduled_tokens`。\n"
    "它的兄弟参数有:\n\n"
    "| vLLM 参数 | 默认值 | 含义 | 对应本课 |\n"
    "|---|---|---|---|\n"
    "| `max_num_batched_tokens` | 2048 | 每步最多处理的 token 数 | `token_budget` |\n"
    "| `max_num_scheduled_tokens` | = 上一项 | 调度器每步最多**发放**的 token 数 | `budget` 发放逻辑 |\n"
    "| `max_num_seqs` | 128 | 每步最多同时跑的请求数 | 可选的并发上限 |\n\n"
    "vLLM V1 调度器(`vllm/v1/core/sched/scheduler.py`)的 `schedule()` 每步做三件事:\n"
    "先服务 running 的 decode(优先),再用剩余 token 预算补 prefill 请求,不足就**分块**\n"
    "(chunked prefill,本课 `prefill_chunk` 就是它的雏形)。这和本课模拟器的循环一一对应。\n\n"
    "> 📄 详见 [vLLM SchedulerConfig 文档](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)。\n\n"
    "下一课(16)我们会给这个模拟器加上**状态机**视角,看清每个请求在 WAITING/RUNNING/FINISHED 之间怎么流转。"
)

# =====================================================================
# 第 11 节 · App
# =====================================================================
nb.md(
    "## 11. 🖥️ Streamlit 动态演示:亲手推演每一笔\n\n"
    "运行 `app_15_continuous_batch.py`,用**步进控制**亲手推演每一笔:\n"
    "拖动 token 预算与请求参数,实时对比静态批与连续批的甘特图、累积完成曲线与指标。\n\n"
    "### 📜 App 完整源码(`app_15_continuous_batch.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_15_continuous_batch.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,拖动滑杆观察两条累积曲线的剪刀差。"
)

wrapup(
    nb,
    summary=[
        "continuous batching = iteration-level scheduling:每步迭代结束重新决定 batch 成员",
        "Orca (arXiv:2208.14217):完成的请求立刻返回、新请求立刻能进;selective batching 处理形状不兼容",
        "token 预算按算力而非人数计:∑每步 token ≤ budget,对应 vLLM 的 max_num_scheduled_tokens",
        "模拟器三步循环:清理完成 -> 预算拉人 -> 推进进度,每步都在换血",
        "同一请求流下连续批 makespan/延迟/吞吐三项全面占优,累积完成曲线全程领先",
        "真实 GPU 上动态批赢在把平均活跃成员顶到吞吐曲线的更右端",
    ],
    practice=[
        "把 simulate_continuous 的 token_budget 从 8 调到 2 和 32,观察吞吐曲线怎么变(太小饿、太大也无益)",
        "给 simulate_continuous 加一个 max_running 上限(并发数约束),模拟显存受限",
        "实现 prefill_chunk=4 的分块 prefill,观察 TTFT(第一个 token 时间)如何被摊平",
        "用第 9 节 avg_active 的插值法,分别算两种调度的「算力利用率」= 平均并发/理想并发",
    ],
    links=[
        ("Orca 论文 (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Anyscale: continuous batching", "https://www.anyscale.com/blog/continuous-batching-llm-inference"),
        ("vLLM V1 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch03\15_continuous_batching.ipynb")