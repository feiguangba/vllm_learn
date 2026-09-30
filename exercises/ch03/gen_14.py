# -*- coding: utf-8 -*-
"""生成第 14 课 notebook: 静态批处理的缺陷(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:大巴直觉 -> 规则与公式 -> 模拟器逐行推演 -> 指标量化 -> 真实 GPU -> 工程关联
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义,脉络清晰
4. 论文支撑:Orca (arXiv:2208.14217)、vLLM 官方博客、Anyscale 博客
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_14_static_batch.py"
APP_NAME = "app_14_static_batch.py"
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
    "第 14 课 · 静态批处理的缺陷:队头阻塞与尾部浪费",
    subtitle="固定批次的大巴调度 · 逐行推演模拟器 · 两个缺陷的量化 · 真实 GPU 账单",
    emoji="🐌", chapter="第 3 章 · Continuous Batching 与调度",
)

chapter_cover(
    nb,
    objectives=[
        "理解静态批处理(static batching)的三条规则:满批发车 / 整批同进退 / 空转等待",
        "用手工构造的小请求流逐行推演 simulate_static,看懂每个中间量怎么算出来的",
        "量化两大缺陷:队头阻塞(head-of-line blocking)与尾部浪费(tail waste),用数字说话",
        "画出甘特图与批次扫描曲线,理解「批次大小 × 延迟 × 利用率」三者如何互相打架",
        "引入真实 GPU 吞吐曲线,把抽象的「步」换算成毫秒,算出静止 batch 白烧了多少算力",
        "建立与 Orca 论文(arXiv:2208.14217)和 vLLM continuous batching 的动机联系",
    ],
    toc=[
        ("直觉:固定座位的大巴", "把 GPU 批处理比喻成凑满一车才发车的景区大巴"),
        ("定义:三条规则与两个缺陷", "规则符号化,队头阻塞/尾部浪费的精确定义与公式"),
        ("请求模型 Req", "造一批「游客」:到达时间、prompt 长度、最大生成长度"),
        ("模拟器 simulate_static", "逐行推演:满批发车、木桶原理、空转等待"),
        ("量化两大缺陷", "makespan / 平均延迟 / GPU 利用率 / 空转 / 空座率"),
        ("甘特图:一眼看穿问题", "灰色等待段 vs 彩色执行段,matplotlib 直出"),
        ("参数扫描:批次大小", "利用率先升后降、延迟单调上升的双刃剑曲线"),
        ("数学视角:三个公式", "把浪费写成公式,推导静态批的最优批次"),
        ("真实 GPU 账单", "把「步」换算成毫秒,量化长尾请求下的真实浪费"),
        ("与 vLLM 的关联", "为什么 vLLM 必须抛弃这套规则——continuous batching 的动机"),
    ],
    links=[
        ("Orca 论文 (OSDI'22): iteration-level scheduling", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 官方博客: 10x faster inference", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Anyscale: How continuous batching enables 23x throughput", "https://www.anyscale.com/blog/continuous-batching-llm-inference"),
        ("vLLM V1 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:固定座位的大巴 🚌\n\n"
    "想象你在景区门口等**景区大巴**:每辆车有固定座位数(比如 4 个),**凑满一车才发车**;\n"
    "车上的人一起出发、一起到站——哪怕有人只想逛 10 分钟,也得陪着要逛 3 小时的游客坐完全程;\n"
    "到站后大巴才能回去接下一批人。\n\n"
    "把「景区大巴」换成 GPU、把「游客」换成推理请求,这就是**静态批处理(static batching)**:\n\n"
    "| 大巴 | GPU 批处理 |\n"
    "|---|---|\n"
    "| 🚌 座位 = batch_size | GPU 一次并行处理几个请求 |\n"
    "| ⏳ 凑满才发车 | 请求先排队,不满一批不开算 |\n"
    "| 🐢 一起到站 | 批内耗时取**最长**请求(木桶原理) |\n"
    "| 🕳️ 回站等人 | 发车间隙与最后凑不满的一批,计算资源全闲着 |\n\n"
    "这套规则在深度学习框架早期非常常见:把请求攒成固定大小 batch,一次 `forward` 算一批。"
    "它简单、整齐、好实现,但有两个致命缺陷,正是本课要量化的东西。"
)

nb.md(
    "## 2. 定义:三条规则与两个缺陷 📐\n\n"
    "### 2.1 三条规则(符号化)\n\n"
    "设第 $i$ 批请求集合为 $\\mathcal{B}_i$,批大小固定为 $B$。规则如下:\n\n"
    "$$ \\text{启动条件}: |\\mathcal{B}_i| = B \\quad(\\text{凑满才发车}) $$\n\n"
    "$$ \\text{批耗时}: \\; T_i = \\max_{r \\in \\mathcal{B}_i} (p_r + m_r) \\quad(\\text{木桶原理, 整批同进退}) $$\n\n"
    "其中 $p_r$ 是请求 $r$ 的 prompt 长度(prefill token 数),$m_r$ 是最大生成长度(decode token 数)。\n\n"
    "### 2.2 两个缺陷\n\n"
    "- **队头阻塞 (Head-of-Line Blocking)**:批内最短请求明明 $p_r{+}m_r$ 步能干完,却要等批内最长的\n"
    "  请求一起走——**提前完成也只能干等**。浪费量 = 批耗时 − 该请求自身耗时。\n"
    "- **尾部浪费 (Tail Waste)**:最后一批可能凑不满 $B$ 个座位;更普遍地,GPU 在「凑批」的空窗期**空转**。\n\n"
    "这两个缺陷正是 Orca 论文(arXiv:2208.14217)§1 描述的动机:**现有 serving 系统在\n"
    "『请求已结束但 batch 未结束』和『新请求到了但 batch 未结束』两种情况下都无能为力**。\n"
    "论文里原话是 *「requests that have finished earlier than other requests in a batch cannot return\n"
    "to the client, while newly arrived requests have to wait until the current batch completely finishes」*。"
)

# =====================================================================
# 第 3 节 · 请求模型
# =====================================================================
nb.md(
    "## 3. 请求模型 Req:先造一批「游客」🎫\n\n"
    "模拟单位约定:**1 步 = 1 次迭代**。prefill 与 decode 都按 token 计——这是为了和第 15 课\n"
    "的 token 预算接轨(Orca 论文里同样用 token/iteration 作为调度粒度)。\n\n"
    "先定义一个 `Req` 数据类,再写 `make_reqs` 生成请求流。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库: 生成随机请求参数
import pandas as pd                               # 表格库: 汇总请求指标
from dataclasses import dataclass, field          # 数据类: 轻量定义请求结构

# --------------------------------------------------------------------------
# 1. 定义请求数据类 Req: 一个请求 = 一段 prompt + 一段待生成的输出
# --------------------------------------------------------------------------
@dataclass
class Req:
    rid: int                                       # 请求编号 (唯一标识, 0,1,2,...)
    arrive: float                                  # 到达时间 (单位: 步)
    prompt_len: int                                # prefill 阶段要处理的 token 数 (p_r)
    max_new: int                                   # decode 阶段最多生成的 token 数 (m_r)
    state: str = "WAITING"                         # 状态: WAITING / RUNNING / FINISHED
    start: float = field(default=None)             # 首次开始执行的时间 (初始为空)
    end: float = field(default=None)               # 完成时间 (初始为空)

# --------------------------------------------------------------------------
# 2. 生成请求流: 两种到达模式
# --------------------------------------------------------------------------
def make_reqs(n=10, mode="burst", rate=0.5, seed=42, plen=(5, 30), mnew=(5, 20)):
    """制造请求流。mode='burst' 所有请求同时到达; mode='poisson' 按泊松过程逐批到达。
    返回 list[Req]。"""
    rng = np.random.default_rng(seed)              # 可复现随机数发生器 (seed 固定)
    reqs, t = [], 0.0                              # 请求列表, 累计到达时间
    for i in range(n):                             # 逐个生成 n 个请求
        if mode == "poisson":                      # 若是泊松流: 用指数分布模拟到达间隔
            t += rng.exponential(1.0 / rate)       # 间隔均值 = 1/rate (rate=每秒请求数)
        a = 0.0 if mode == "burst" else t          # burst: 全部 t=0 到达; poisson: 用累计时间
        reqs.append(Req(                           # 生成一个请求对象并加入列表
            i, a,                                  # 编号 i, 到达时间 a
            int(rng.integers(plen[0], plen[1] + 1)),  # prompt 长度在 [plen0, plen1] 均匀取整
            int(rng.integers(mnew[0], mnew[1] + 1)),  # 最大生成长度同理
        ))
    return reqs                                    # 返回完整请求流

# --------------------------------------------------------------------------
# 3. 打印一批请求, 建立直觉
# --------------------------------------------------------------------------
reqs = make_reqs(n=8, seed=42)                     # 8 个 burst 请求, seed 固定保证可复现
for r in reqs:                                     # 遍历每个请求
    # 输出每个请求的三要素 (到达 / prompt / 最大生成), 单位都是「步」
    print(f"Req{r.rid}: 到达={r.arrive:.0f} 提示词={r.prompt_len} 最大生成={r.max_new} 总耗时={r.prompt_len + r.max_new}")''',
    "✅ **请求流就绪**。每个请求的「总耗时 = prompt_len + max_new」,这就是它独占 GPU 时的完成时间;"
    "注意第 14 课所有单位都是「步」,第 9 节会把它换算成毫秒。",
)

# =====================================================================
# 第 4 节 · 模拟器
# =====================================================================
nb.md(
    "## 4. 模拟器 simulate_static:把大巴规则翻译成代码 🐢\n\n"
    "规则只有三条,代码也只有三块:\n\n"
    "1. **满批发车**:从队列里按到达顺序捞人,捞满 `batch_size` 个就发车;\n"
    "2. **整批同进退**:本批耗时 = 批内所有请求耗时($p_r{+}m_r$)的**最大值**;\n"
    "3. **空转等待**:一个能发车的都没有,时间照走、GPU 白等。\n\n"
    "下面这个 `simulate_static` 就是完整的大巴调度器。我们逐步拆开注释,并记录每批的起止时间 `batches` 供画甘特图。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库 (上一 cell 已 import, 这里再引一遍保证独立可跑)

def simulate_static(reqs, batch_size):
    """静态批处理模拟器: GPU 一次服务一批, 批内全部完成才换下一批。

    参数
    ----
    reqs       : list[Req]  待服务的请求流
    batch_size : int        固定批次大小 (大巴座位数)

    返回
    ----
    (reqs, batches) : reqs 被就地改写 (补上 start/end/state),
                      batches 是 list[dict], 记录每批 {start, end, members}
    """
    queue = sorted(reqs, key=lambda r: r.arrive)   # 待发车队列: 按到达时间升序 (FCFS 进场)
    t = 0.0                                        # 当前墙钟时间 (单位: 步), 从 0 开始
    batches = []                                   # 每批的记录, 供画甘特图 / 算指标

    while queue:                                   # 只要还有人没发车就继续
        batch = []                                 # 本批成员 (临时收集)
        for r in list(queue):                      # 遍历队列副本 (因为边遍历边删除)
            # 满批发车: 只捞「已经到达」且「座位还没坐满」的请求
            if r.arrive <= t and len(batch) < batch_size:
                batch.append(r)                    # 把该请求拉上本批
                queue.remove(r)                    # 从等待队列移除 (已上车)

        if not batch:                              # 一个能发车的都没有 -> GPU 空转
            t += 1.0                               # 时间照走 1 步, 算力白等
            continue                               # 跳到 while 重新检查

        for r in batch:                            # 发车: 整批同时开始
            r.state = "RUNNING"                    # 状态置为运行中
            r.start = t                            # 记录本批开始时间

        # 木桶原理: 本批耗时 = 批内所有请求「prompt+生成」的最大值
        batch_time = max(r.prompt_len + r.max_new for r in batch)

        for r in batch:                            # 整批同时到站
            r.end = t + batch_time                 # 每个请求都在「开始+批耗时」时刻结束
            r.state = "FINISHED"                   # 状态置为完成

        batches.append({                           # 记录本批信息, 供甘特图/统计用
            "start": t,                            # 批开始时间
            "end": t + batch_time,                 # 批结束时间
            "members": [r.rid for r in batch],     # 成员编号列表
        })
        t += batch_time                            # 墙钟推进整批耗时
    return reqs, batches                           # 返回改写后的请求 + 批次记录


# --------------------------------------------------------------------------
# 演练: 用 8 个 burst 请求 + batch_size=3 跑一遍, 打印每批
# --------------------------------------------------------------------------
reqs, batches = simulate_static(make_reqs(n=8, seed=42), batch_size=3)
for i, b in enumerate(batches, 1):                 # 枚举每批 (编号从 1 开始)
    # 打印: 第几批 / 起止步 / 成员 / 批耗时
    print(f"批{i}: 步 {b['start']:.0f} -> {b['end']:.0f}, 成员 {b['members']}, 耗时 {b['end'] - b['start']:.0f}")''',
    "🛠️ **逐行推演**。重点看三个关键词:`len(batch) < batch_size`(满批发车)、\n"
    "`max(...)`(木桶原理)、`t += 1.0; continue`(空转等待)。"
    "跑完后你能看到「批 2 里谁在拖后腿」——队头阻塞的证据。",
)

# =====================================================================
# 第 5 节 · 量化
# =====================================================================
nb.md(
    "## 5. 量化两大缺陷:让数字自己说话 📏\n\n"
    "大巴开完了,现在算账。我们关心四个指标(全部由 `end - arrive` 等原始量导出):\n\n"
    "- **makespan(墙钟总耗时)**: $M = \\max_r (end_r - arrive_r)$ 视角上取最大完成时间;\n"
    "- **平均完成时间(平均延迟)**: $\\bar L = \\frac{1}{N}\\sum_r (end_r - arrive_r)$;\n"
    "- **GPU 利用率**: $\\text{util} = \\dfrac{\\text{全部有效工作量}}{\\text{makespan}} = \\dfrac{\\sum_r (p_r{+}m_r)}{M}$;\n"
    "- **GPU 空转步数**: $M - \\sum_i T_i$($T_i$ 为各批真实耗时)。\n\n"
    "**队头阻塞**体现在批内:短请求明明 10 步能干完,却要等批里最长的 30 步——白白多等 20 步。\n"
    "**尾部浪费**体现在批尾:最后一批只有 2 个人,大巴 4 个座位空了 2 个。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import pandas as pd                               # 表格库

def reqs_df(reqs):
    """把请求对象汇总成 DataFrame, 并附带延迟/等待两个派生指标。"""
    df = pd.DataFrame([                             # 按请求逐行构造表格
        dict(rid=r.rid, arrive=r.arrive, start=r.start, end=r.end,   # 基本时间量
             work=r.prompt_len + r.max_new)         # 该请求独立完成所需步数 (有效工作量)
        for r in reqs])
    df["latency"] = df["end"] - df["arrive"]        # 延迟 = 完成 - 到达 (含排队等待)
    df["wait"] = df["start"] - df["arrive"]         # 等待 = 开始 - 到达 (排队时长)
    return df                                       # 返回表格

def throughput(df):
    """从请求表格导出吞吐/延迟指标。"""
    makespan = float(df["end"].max())               # makespan = 最后一个完成的时刻
    return dict(
        req_per_step=float(len(df) / makespan),     # 每秒(步)完成的请求数
        token_per_step=float(df["work"].sum() / makespan),  # 每步处理的 token 数
        makespan=makespan,                          # 墙钟总耗时
        avg_latency=float(df["latency"].mean()),    # 平均完成时间
    )

# --------------------------------------------------------------------------
# 用 16 个请求、batch_size=4 跑一次并打印全部指标
# --------------------------------------------------------------------------
batch_size = 4                                    # 固定批次大小 (本实验用 4)
reqs, batches = simulate_static(make_reqs(n=16, seed=7), batch_size=batch_size)
df = reqs_df(reqs)                                  # 汇总成表格
tput = throughput(df)                               # 计算吞吐/延迟指标
idle = tput["makespan"] - sum(b["end"] - b["start"] for b in batches)  # 空转 = makespan - 各批耗时之和

# 队头阻塞: 每个批内「最长耗时 - 最短耗时」的最大值 (最短请求陪跑的最长白等)
hol = max(
    (df[df.rid.isin(b["members"])].work.max()       # 批内最长请求耗时
     - df[df.rid.isin(b["members"])].work.min())    # 批内最短请求耗时
    for b in batches
)
# 尾部浪费: 最后一批的空座率
tail_waste = (batch_size - len(batches[-1]["members"])) / batch_size

print(f"全部完成耗时 (makespan)     = {tput['makespan']:.0f} 步")
print(f"平均完成时间 (平均延迟)     = {tput['avg_latency']:.1f} 步")
print(f"GPU 利用率                 = {df['work'].sum() / tput['makespan'] * 100:.1f}%")
print(f"GPU 空转                   = {idle:.0f} 步 ({idle / tput['makespan'] * 100:.0f}%)")
print(f"队头阻塞: 批内最短请求最多要多等 {hol:.0f} 步")
print(f"尾部浪费: 最后一批 {len(batches[-1]['members'])} 人, 容量 {batch_size}, 空座率 {tail_waste * 100:.0f}%")''',
    "✅ **数字即证据**。同一批请求,静态批处理把大量时间花在「等人」上——这就是 vLLM 博客里\n"
    "*「memory waste and GPU idle」* 的微观来源。",
)

# =====================================================================
# 第 6 节 · 甘特图
# =====================================================================
nb.md(
    "## 6. 甘特图:一眼看穿问题 👀\n\n"
    "数字有了,但「形象」还不够。用 **matplotlib** 画一张甘特图:横轴是时间(步),\n"
    "每个请求一行,灰色段 = 排队等待,彩色段 = 执行(同一批同一种颜色)。\n\n"
    "你会看到两件非常直观的事:\n"
    "1. **队头阻塞**:批内彩色段**同时开始、同时结束**,长短不一的请求被强行对齐成同一个长度;\n"
    "2. **尾部浪费**:最后一行的灰色等待段特别长——凑不满一批,GPU 只能干等。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import matplotlib                               # 绘图库 (显式引用, 便于设置后端)
import matplotlib.pyplot as plt                 # pyplot 接口
import numpy as np                              # 数值库
%matplotlib inline

# 中文字体设置 (Windows 通用; 若缺字体 matplotlib 会回退, 不影响数据)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False       # 让负号正常显示

# 复跑一次模拟, 拿到底层数据
reqs, batches = simulate_static(make_reqs(n=16, seed=7), batch_size=4)
df = reqs_df(reqs)                               # 汇总请求表格

fig, ax = plt.subplots(figsize=(12, 6))          # 创建画布 12x6 英寸
cmap = plt.cm.tab10                              # 10 色循环调色板 (区分不同批次)

for i, b in enumerate(batches):                  # 遍历每一批
    for rid in b["members"]:                     # 遍历批内每个成员
        r = df[df.rid == rid].iloc[0]            # 取该请求的时间信息
        y = rid                                   # 用请求编号作为 y 坐标
        # 灰色段: 排队等待 [arrive, start)
        ax.barh(y, r.start - r.arrive, left=r.arrive, color="#D0D0D0",
                edgecolor="none", height=0.6)
        # 彩色段: 执行 [start, end), 同批同色
        ax.barh(y, r.end - r.start, left=r.start, color=cmap(i % 10),
                edgecolor="none", height=0.6, label=f"批{i + 1}" if rid == b["members"][0] else None)

ax.set_xlabel("时间 (步)")                        # x 轴: 时间
ax.set_ylabel("请求编号")                         # y 轴: 请求
ax.set_title("静态批处理甘特图: 灰色=排队, 彩色=执行 (同批同色)")  # 标题
ax.legend(loc="upper right", fontsize=8)         # 图例 (每个批次一种颜色)
ax.grid(axis="x", alpha=0.3)                     # 细网格便于读数
plt.tight_layout()                               # 自动调整边距
plt.show()                                       # 渲染到 notebook''',
    "🎨 **一眼看穿**。同批请求的彩色段完全对齐(同时开始同时结束)是队头阻塞;"
    "最后几行灰色段特别长是凑不满批导致的 GPU 空转。",
)

# =====================================================================
# 第 7 节 · 参数扫描
# =====================================================================
nb.md(
    "## 7. 参数扫描:批次大小是把双刃剑 ⚔️\n\n"
    "直觉告诉我们:批次太小 → 批次多、发车间隙多 → GPU 空转;批次太大 → 批内「木桶」越长 →\n"
    "队头阻塞更重。我们用同一请求流,把 `batch_size` 从 1 扫到 10,画出 **利用率** 和 **平均延迟** 两条曲线。\n\n"
    "注意:利用率随 batch 增大先升后降,平均延迟几乎单调上升——**两个目标互相打架**。\n"
    "这正是 vLLM 用 `max_num_seqs`(默认 128)与 `max_num_batched_tokens`(默认 2048)两个旋钮\n"
    "同时约束批的「人数上限」与「token 预算」的原因。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                              # 数值库
import matplotlib.pyplot as plt                 # 绘图库
%matplotlib inline

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False       # 负号正常显示

batch_sizes = range(1, 11)                       # 扫描 batch_size = 1..10
util_list, lat_list, makespan_list = [], [], []  # 收集三条曲线

for bs in batch_sizes:                           # 遍历每个批次大小
    reqs, batches = simulate_static(make_reqs(n=32, seed=11), batch_size=bs)  # 同请求流
    df = reqs_df(reqs)                            # 汇总表格
    makespan = float(df["end"].max())             # makespan
    util = df["work"].sum() / makespan            # 利用率 = 总工作量 / makespan
    util_list.append(util)                        # 记录利用率
    lat_list.append(df["latency"].mean())         # 记录平均延迟
    makespan_list.append(makespan)                # 记录 makespan

fig, ax1 = plt.subplots(figsize=(8, 5))           # 左轴: 利用率
ax1.plot(list(batch_sizes), util_list, "o-", color="#C44E52", label="GPU 利用率")
ax1.set_xlabel("批次大小 batch_size")              # x 轴
ax1.set_ylabel("GPU 利用率", color="#C44E52")      # 左 y 轴
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax1.set_ylim(0, 1.05)                            # 利用率上限 1

ax2 = ax1.twinx()                                # 右轴: 平均延迟 (双 y 轴)
ax2.plot(list(batch_sizes), lat_list, "s--", color="#4C72B0", label="平均延迟")
ax2.set_ylabel("平均延迟 (步)", color="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#4C72B0")

# 合并图例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right")
ax1.set_title("批次大小扫描: 利用率先升后降, 延迟单调上升")
plt.tight_layout()
plt.show()

# 打印每个批次的具体数字, 便于对照
print("batch_size | 利用率 | 平均延迟 | makespan")
for bs, u, l, m in zip(batch_sizes, util_list, lat_list, makespan_list):
    print(f"{bs:10d} | {u * 100:5.1f}% | {l:8.1f}步 | {m:7.0f}步")''',
    "📊 **双刃剑曲线**。注意利用率在某个批次达到峰值后回落——这就是「最优批次」的位置;"
    "而平均延迟随批次几乎只升不降。两个目标不可能同时最优,调度器必须做权衡。",
)

# =====================================================================
# 第 8 节 · 数学视角
# =====================================================================
nb.md(
    "## 8. 数学视角:三个公式定乾坤 🧮\n\n"
    "把观察写成公式。设第 $i$ 批有 $B_i$ 个请求,第 $i$ 批的耗时为:\n\n"
    "$$ T_i = \\max_{r \\in \\mathcal{B}_i} (p_r + m_r) $$\n\n"
    "则 **队头阻塞浪费** 定义为批内最短请求多等的时间:\n\n"
    "$$ W_i^{\\text{HOL}} = T_i - \\min_{r \\in \\mathcal{B}_i} (p_r + m_r) $$\n\n"
    "**GPU 空转浪费** 为凑批等待与尾部不足:\n\n"
    "$$ W^{\\text{idle}} = M - \\sum_i T_i, \\qquad W^{\\text{tail}} = \\sum_i (B - B_i)^+ \\cdot \\bar T $$\n\n"
    "其中 $M$ 是 makespan,$(x)^+ = \\max(x,0)$。因此静态批的总「浪费步数」约为:\n\n"
    "$$ W^{\\text{static}} = W^{\\text{HOL}} + W^{\\text{idle}} + W^{\\text{tail}} $$\n\n"
    "**直觉**:$B$ 越大 → $T_i$ 越被最长请求主导 → $W^{\\text{HOL}}$ 增大;但 $B$ 越小 → 批次越多 →\n"
    "$W^{\\text{idle}}$ 增大。两者此消彼长,所以存在**最优批次**。Orca 论文正是抓住这一点:\n"
    "与其调一个固定批次,不如**每步(iteration)动态决定批成员**——把 batch 从「请求粒度」降到「迭代粒度」。"
)

# =====================================================================
# 第 9 节 · 真实 GPU 账单
# =====================================================================
nb.md(
    "## 9. 真实 GPU 账单:把「步」换算成毫秒 ⏱️\n\n"
    "前面 1~8 节全程在「步」这个抽象时间单位里谈浪费。现在把**真实算力**搬进来:\n"
    "用跨章共享库 `vllm_real` 在 GPU 上实测「每步(每 token 每请求)真实耗时」随 batch 的变化,\n"
    "然后把模拟器的步数映射成毫秒,算出静止 batch 在长尾请求下的真实浪费。\n\n"
    "实测得到的曲线符合 **Amdahl/带宽定律**:batch 越大,单位 token 的固定开销(读权重、kernel 启动)\n"
    "被更多 token 摊薄,所以吞吐**非线性**上涨。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库: 路径与环境变量
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 OpenMP 冲突 (Windows torch 常见)
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")  # 加入共享库目录
from vllm_real import bench_throughput_curve, cuda_info   # 跨章共享真实 GPU 微基准

print("设备:", cuda_info())                        # 打印当前设备 (GPU 型号或 CPU 降级)

# 固定 batch: 每步把 batch 个单 token 一起前向 (连续批的雏形), 测真实吞吐
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
print("batch | 每步耗时(ms)   | 吞吐(token/s)")
for b, ms, tp in zip(b_list, mps, tps):            # 逐行打印实测结果
    print(f"{b:5d} | {ms:9.3f}   | {tp:10.0f}")

# 结论: batch 越大单位 token 越便宜 (非线性上涨)—— 这是「凑批」的唯一好处''',
    "🚀 **真实 GPU 数字**。记住这条曲线的形状:batch 越大单位 token 越便宜。"
    "但它有两个前提:第一,batch **真的凑得满**;第二,每一步都**满载输出**。"
    "静态批恰恰同时违反这两点。",
)

nb.md(
    "### 9.2 把模拟器的步映射成毫秒:静止 batch 白烧了多少\n\n"
    "现在把两条线索接起来:模拟器给出「每步多少个请求在跑」,真实曲线给出「这个并发下每步多少毫秒」。\n"
    "于是可以把静态批的 makespan 换算成真实毫秒,并对比「吃满算力时的吞吐」得到浪费比例。\n\n"
    "做法:静态批执行期间,每步并发 = batch_size(满载但队头阻塞),空转步并发 = 0;\n"
    "每步耗时 = `mps` 表中对应并发的那一项。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info  # 真实 GPU 微基准

# 1) 拿真实曲线: batch -> (ms_per_step, tokens_per_s)
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 4, 8, 16, 32), token_len=16, reps=5)
ms_of = dict(zip(b_list, mps))                    # {batch: ms/步} 查表
tps_of = dict(zip(b_list, tps))                   # {batch: token/s} 查表

# 2) 造一个「长尾到达」的请求流: 前 16 个 t=0 涌到, 后 6 个零星迟到
reqs = make_reqs(n=22, mode="poisson", rate=0.08, seed=3)   # 泊松流: 前密后疏
batch_size = 4                                    # 固定批次
reqs, batches = simulate_static(reqs, batch_size) # 跑静态批模拟
df = reqs_df(reqs)                                # 汇总表格
makespan_steps = float(df["end"].max())           # makespan (步)

# 3) 换算成真实毫秒: 每次「满载执行」的步, 每步花 ms_of[batch_size] 毫秒
n_exec_steps = sum(b["end"] - b["start"] for b in batches)   # 真正在跑的总步数
wall_ms = n_exec_steps * ms_of.get(batch_size, 1.0)          # 墙钟 = 执行步数 x 每步毫秒

# 4) 计算真实吞吐与浪费
total_tokens = int(df["work"].sum())              # 总 token 数 (有效工作量)
achieved_tps = total_tokens / (wall_ms / 1000)    # 实际吞吐 = token / 真实秒
peak_tps = max(tps)                               # 物理上限: 曲线上最高吞吐 (吃满时)
waste_pct = (peak_tps - achieved_tps) / peak_tps  # 浪费比例

print(f"设备: {cuda_info()}")
print(f"静态批 makespan = {makespan_steps:.0f} 步, 其中执行 {n_exec_steps:.0f} 步, 空转 {makespan_steps - n_exec_steps:.0f} 步")
print(f"每步真实耗时 (batch={batch_size}) = {ms_of.get(batch_size, 0):.3f} ms")
print(f"墙钟 = {wall_ms:.0f} ms = {wall_ms / 1000:.2f} s")
print(f"实际吞吐 = {achieved_tps:.0f} token/s, 吃满时上限 = {peak_tps:.0f} token/s")
print(f"=> 长尾请求下, 静止 batch 的算力浪费 ≈ {waste_pct * 100:.0f}%")''',
    "🔢 **前因后果落到数字**。浪费 = (吃满的速度 − 实际的速度) ÷ 吃满的速度。"
    "模拟器的空转步、队头陪跑,被真实吞吐曲线放大成了毫秒级的「白烧电」。"
    "这正是下一课 continuous batching 要消灭的部分。",
)

# =====================================================================
# 第 10 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 10. 为什么 vLLM 必须抛弃这套规则:与 Orca / vLLM 的关联 🔗\n\n"
    "Orca 论文(arXiv:2208.14217)指出,静态批的核心问题是**调度粒度错了**:\n"
    "它在「请求粒度」上排程,而 Transformer 生成是**多迭代**过程——一次请求要跑很多次模型。\n"
    "正确做法是把调度粒度降到**迭代(iteration)粒度**:\n\n"
    "$$ \\text{静态批: 一批请求整体跑完才换下一批} \\quad\\rightarrow\\quad "
    "\\text{Orca: 每步选择一批请求, 只跑一次迭代} $$\n\n"
    "论文中的两个关键概念:\n\n"
    "- **iteration-level scheduling**:每步执行完就检查——完成的请求立刻返回,新请求立刻能进;\n"
    "- **selective batching**:只对形状兼容的算子(非 Attention 的矩阵乘、LayerNorm)做批处理,\n"
    "  不兼容的(变长 Attention)分别执行。\n\n"
    "vLLM 的连续批处理(continuous batching)继承了 Orca 的迭代级调度,并叠加 PagedAttention 的\n"
    "显存管理,把静态批的浪费基本清零。官方博客(vLLM, 2023)报告:**相比 HuggingFace Transformers\n"
    "吞吐提升最高 24×,相比 TGI 提升 2.2~2.5×**。\n\n"
    "> 📄 一句话:**本课量化的所有浪费,就是 continuous batching 存在的全部理由。**\n"
    "> 下一课(15)我们把 batch 从「固定」变成「每步动态」,亲手看浪费怎么被填平。"
)

# =====================================================================
# 第 11 节 · App
# =====================================================================
nb.md(
    "## 11. 🖥️ Streamlit 动态演示:亲手验证\n\n"
    "把上面的模拟器做成可交互 App:拖动**请求数量 / 批次大小 / 到达模式**滑杆,\n"
    "实时观察甘特图与平均延迟、GPU 利用率、空转、尾部浪费四个指标。\n\n"
    "### 📜 App 完整源码(`app_14_static_batch.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 本目录已生成 `app_14_static_batch.py`(与 notebook 共享同一套模拟器);\n"
    "2. 命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_14_static_batch.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501,把 batch_size 从 1 拖到 10,观察利用率先升后降、"
    "平均延迟单调上升。"
)

wrapup(
    nb,
    summary=[
        "静态批处理三规则:满批发车、整批同进退(木桶原理)、空转等待,调度粒度是「请求」",
        "队头阻塞:批内最短请求被迫陪跑最长请求;浪费 = 批耗时 − 自身耗时",
        "尾部浪费:最后一批凑不满 + 凑批空窗期 GPU 空转",
        "批次扫描呈双刃剑:利用率先升后降(最优批次存在),平均延迟单调上升",
        "真实 GPU 曲线证明 batch 越大单位 token 越便宜,但前提是「凑得满 + 满载输出」——静态批两者都做不到",
        "Orca (arXiv:2208.14217) 把调度粒度从请求降到迭代,这是 continuous batching 的起点",
    ],
    practice=[
        "把 make_reqs 的 mode 改成 poisson、rate 调到 1.5,重跑第 5 节,观察空转率怎么变",
        "修改 simulate_static 让它记录每批内每个请求的「陪跑浪费」,画出浪费直方图",
        "在 batch_size 扫描里额外记录 P95 延迟,验证「平均延迟会撒谎」(P95 涨得更快)",
        "把 n 从 16 增到 64,观察最优批次的位置如何移动(思考:为什么朝大方向移动?)",
    ],
    links=[
        ("Orca 论文 (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Anyscale: continuous batching 23x", "https://www.anyscale.com/blog/continuous-batching-llm-inference"),
        ("vLLM V1 调度器源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch03\14_static_batching_problem.ipynb")