# -*- coding: utf-8 -*-
"""ch03 生成公共组件:Notebook 构建 + 各课共享的模拟器源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH03 = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch03"
CHAPTER = "第 3 章 · Continuous Batching 与调度"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


def app_cell(nb, app_code, fname, note=""):
    """把 streamlit app 源码用守卫包起来嵌入 notebook。

    notebook 内核里执行时走 else 分支(仅提示怎么运行),streamlit 环境(st.streamlit run)
    才真正执行 app 主体——避免 App 代码的运行时依赖(st.session_state 等)在裸内核崩溃,
    同时源码仍完整保留在 cell 里供读者复制。
    """
    guard = (
        "try:\n"
        "    import streamlit as st\n"
        "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
        "except Exception:\n"
        "    _IS_STREAMLIT = False\n\n"
        "if _IS_STREAMLIT:\n"
        + textwrap.indent(textwrap.dedent(app_code), "    ") +
        "\nelse:\n"
        "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
        f"    print(\"    请把上方源码保存为 {fname} 后用以下命令运行:\")\n"
        f"    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run {fname}\")\n"
    )
    nb.code(guard, note)
    return nb


# ---------------------------------------------------------------- 模拟器源码
# 注:prefill 采用"分块"策略(chunked prefill,对应 vLLM 的 max_num_batched_tokens):
# 预算不够一个完整 prompt 时,先吃光剩余预算、记下进度,下步继续——避免小预算死锁。
SIM_BASE = D('''
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Req:
    """一个推理请求的完整画像"""
    rid: int                 # 请求编号
    arrive: float            # 到达时间(单位:步)
    prompt_len: int          # prefill 阶段要处理的 token 数
    max_new: int             # decode 阶段最多生成的 token 数
    priority: float = 0.0    # 优先级(0~1,越大越优先)
    state: str = "WAITING"   # WAITING / RUNNING / PREEMPTED / FINISHED
    start: float = None      # 首次开始执行的时间
    end: float = None        # 完成时间
    generated: int = 0       # 已生成的 token 数
    prefilled: int = 0       # 已完成 prefill 的 token 数(分块进度)
    preempted: int = 0       # 被抢占次数
    resume_left: int = 0     # swap 策略:搬回 KV 还需要多少步

def make_reqs(n=10, mode="burst", rate=0.5, seed=42, plen=(5, 30), mnew=(5, 20)):
    """制造一个请求流。
    mode="burst"  : 所有请求同时到达(压测场景)
    mode="poisson": 按泊松过程到达,rate 越大到达越密集
    """
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        if mode == "poisson":
            t += rng.exponential(1.0 / rate)
        a = 0.0 if mode == "burst" else t
        reqs.append(Req(i, a,
                        int(rng.integers(plen[0], plen[1] + 1)),
                        int(rng.integers(mnew[0], mnew[1] + 1)),
                        priority=float(rng.random())))
    return reqs
''')

SIM_STATIC = D('''
def simulate_static(reqs, batch_size):
    """静态批处理模拟器:GPU 一次服务一批,批内全部完成才换下一批。
    规则:
      1. 请求按到达顺序排队,凑满 batch_size 个才发车;
      2. 批内所有请求同时开始,耗时 = 批内最长请求耗时(队头阻塞!);
      3. 凑不满一批时 GPU 空转等待(尾部浪费!)。
    返回 (reqs, batches):batches 记录每一批的起止时间与成员,供甘特图使用。
    """
    queue = sorted(reqs, key=lambda r: r.arrive)   # 待发车队列
    t, batches = 0.0, []
    while queue:
        batch = []
        for r in list(queue):
            if r.arrive <= t and len(batch) < batch_size:
                batch.append(r)
                queue.remove(r)
        if not batch:              # 一个能发车的都没有 → GPU 空转等待
            t += 1.0
            continue
        for r in batch:            # 发车:整批同时开始
            r.state, r.start = "RUNNING", t
        batch_time = max(r.prompt_len + r.max_new for r in batch)  # 木桶原理
        for r in batch:
            r.end, r.state = t + batch_time, "FINISHED"
        batches.append(dict(start=t, end=t + batch_time, members=[r.rid for r in batch]))
        t += batch_time
    return reqs, batches
''')

SIM_CONT = D('''
def simulate_continuous(reqs, token_budget=8):
    """Continuous batching 模拟器(iteration-level scheduling,逐迭代调度)。
    每步(一个迭代)流程:
      1. 已到达且不在运行中的请求进入运行集;
      2. 用 token_budget 依次给请求分配 token:
         - prefill 请求分块消费预算,攒够 prompt_len 即进入 RUNNING;
         - decode 请求每步吃 1 个预算、生成 1 个 token;
      3. 完成的请求立刻离开,位置马上被新请求补上。
    """
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    while done < total:
        for r in list(waiting):            # 1) 补位
            if r.arrive <= t:
                running.append(r); waiting.remove(r)
        budget = token_budget
        for r in running:                  # 2a) 分块 prefill:吃预算,记进度
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:      # prefill 完成!
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in running:                  # 2b) decode:每个请求分 1 个 token
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):            # 3) 完成离场
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
                running.remove(r); done += 1
        t += 1.0
    return reqs
''')

SIM_LOG = D('''
def simulate_continuous_log(reqs, token_budget=8):
    """与 simulate_continuous 完全相同,但额外记录每一步的 batch 成员,
    供"逐步动画"观察每次迭代运行集的变化。"""
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    steps = []   # 每一步: (running 成员, waiting 成员, 已完成数)
    while done < total:
        for r in list(waiting):
            if r.arrive <= t:
                running.append(r); waiting.remove(r)
        budget = token_budget
        for r in running:
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in running:
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
                running.remove(r); done += 1
        steps.append(dict(step=int(t), running=[r.rid for r in running],
                          waiting=[r.rid for r in waiting], finished=done))
        t += 1.0
    return reqs, steps
''')

SIM_SCHED = D('''
def pick_next(waiting, policy):
    """按策略给等待队列排序(返回排序后的新列表)。
    policy = "FCFS"     先来先服务,按 arrive 排序
    policy = "SJF"      短作业优先,总耗时最短的先上
    policy = "Priority" 高优先级先上,priority 越大越优先
    """
    if policy == "FCFS":
        return sorted(waiting, key=lambda r: r.arrive)
    if policy == "SJF":
        return sorted(waiting, key=lambda r: r.prompt_len + r.max_new)
    if policy == "Priority":
        return sorted(waiting, key=lambda r: -r.priority)
    raise ValueError(f"未知策略: {policy}")

def simulate_sched(reqs, token_budget=8, policy="FCFS"):
    """在 continuous batching 之上引入调度策略:
    每步先按策略给等待队列排序,再按序补位、按预算分配 token。"""
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    schedule_log = []   # 每步记录 running 成员
    while done < total:
        waiting = pick_next(waiting, policy)   # ① 策略排序
        for r in list(waiting):                # ② 已到达请求按策略顺序补位
            if r.arrive <= t:
                running.append(r); waiting.remove(r)
        budget = token_budget
        for r in running:                      # ③ 分块 prefill
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in running:                      # ④ decode 分配
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):                # ⑤ 完成离场
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
                running.remove(r); done += 1
        schedule_log.append(dict(step=int(t), running=[r.rid for r in running], done=done))
        t += 1.0
    return reqs, schedule_log
''')

SIM_PREEMPT = D('''
def simulate_preempt(reqs, token_budget=8, max_running=4, policy="recompute", swap_cost=3.0):
    """带抢占的 continuous batching 模拟器。
    max_running : 显存决定的并发上限(KV cache 放不下就抢)
    policy      : "recompute" — 被抢占请求的 KV 全部作废(清空 prefilled/generated),
                               恢复后从头重新 prefill
                  "swap"      — KV 换到 CPU,恢复时花 swap_cost 步搬回,进度保留
    抢占规则    : LIFO —— 新请求到来而 running 已满时,抢占最后一个进入的请求。
    """
    waiting = sorted(reqs, key=lambda r: r.arrive)
    running, t, done, total = [], 0.0, 0, len(reqs)
    events = []   # (类型, 时间, 请求编号, 对方请求编号)
    while done < total:
        new = [r for r in waiting if r.arrive <= t]      # ① 本步新到达
        for r in new:
            waiting.remove(r)
        for r in list(new):                               # ② 被抢占请求优先复座
            if r.state == "PREEMPTED" and len(running) < max_running:
                running.append(r); new.remove(r)
        for r in list(new):                               # ③ 新请求进入,running 满则抢位
            if len(running) < max_running:
                running.append(r); new.remove(r)
            else:
                victim = running.pop()                    # LIFO:抢最后进来的
                victim.state, victim.preempted = "PREEMPTED", victim.preempted + 1
                if policy == "recompute":
                    victim.generated, victim.prefilled = 0, 0   # KV 作废,重来
                    victim.state = "WAITING"                    # 需要重新 prefill
                    victim.resume_left = 0
                else:
                    victim.resume_left = swap_cost              # KV 已搬去 CPU
                waiting.insert(0, victim)
                events.append(("抢占", t, victim.rid, r.rid))
                running.append(r); new.remove(r)
        for r in running:                                 # ④ swap:搬 KV 期间不产出 token
            if r.resume_left > 0:
                r.resume_left -= 1
                if r.resume_left == 0 and r.state == "PREEMPTED":
                    r.state = "RUNNING"                    # KV 搬回,继续 decode
        budget = token_budget
        for r in running:                                 # ⑤ 分块 prefill
            if r.state == "WAITING" and budget > 0 and r.prefilled < r.prompt_len:
                use = min(budget, r.prompt_len - r.prefilled)
                budget -= use
                r.prefilled += use
                if r.prefilled >= r.prompt_len:
                    r.state = "RUNNING"
                    if r.start is None:
                        r.start = t
        for r in running:                                 # ⑥ decode 分配
            if r.state == "RUNNING" and budget > 0 and r.generated < r.max_new:
                r.generated += 1; budget -= 1
        for r in list(running):                           # ⑦ 完成离场
            if r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
                running.remove(r); done += 1
        t += 1.0
    return reqs, events
''')

SIM_STATS = D('''
def reqs_df(reqs):
    """把请求对象汇总成 DataFrame,附带延迟 / 等待时间等指标"""
    df = pd.DataFrame([dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,
                            work=r.prompt_len + r.max_new, generated=r.generated,
                            preempted=r.preempted) for r in reqs])
    df["latency"] = df["end"] - df["arrive"]
    df["wait"] = df["start"] - df["arrive"]
    return df

def throughput(df):
    """吞吐指标:每单位时间(步)完成的请求数与 token 数"""
    makespan = float(df["end"].max())
    return dict(req_per_step=float(len(df) / makespan),
                token_per_step=float(df["work"].sum() / makespan),
                makespan=makespan,
                avg_latency=float(df["latency"].mean()),
                p95_latency=float(np.percentile(df["latency"], 95)))
''')

GANTT_PLOTLY = D('''
def gantt_plotly(df, color_map, title="调度甘特图"):
    """把请求时间线画成横向甘特图:横轴时间,每行一个请求。
    df 需要包含列: rid / start / end / kind(kind 用于着色)"""
    import plotly.graph_objects as go
    fig = go.Figure()
    df = df.sort_values(["start", "rid"])
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["end"] - row["start"]], y=[f"Req {row.rid}"],
            base=[row["start"]], orientation="h",
            marker_color=color_map.get(row.get("kind", "RUNNING"), "#888"),
            text=f"执行 {row['end'] - row['start']:.0f} 步",
            hoverinfo="x+y+text", showlegend=False, width=0.6))
    fig.update_layout(title=title, xaxis_title="时间(步)", yaxis_title="请求",
                      height=40 + 34 * len(df), margin=dict(l=10, r=10, t=40, b=10),
                      bargap=0.15)
    return fig
''')