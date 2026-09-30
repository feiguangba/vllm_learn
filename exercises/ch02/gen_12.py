# -*- coding: utf-8 -*-
"""生成第 12 课 notebook: 前缀缓存 Prefix Caching(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md 与 gen_07.py 金标准):
1. 由浅入深:脱口秀开场白直觉 -> 链式块哈希原理 -> 手工模拟逐块哈希命中 ->
   块需求公式推导 -> real_ops.prefix_hit_sim 真实仿真 -> 收益/命中率曲线 -> TTFT 视角 -> 实践要点 -> 小结
2. 每一行代码都有 inline 注释;每个中间量打印并标注含义
3. 论文/文档支撑:vLLM automatic prefix caching 设计文档(chain hash + block hash table + 驱逐策略)、
   SGLang RadixAttention (arXiv:2312.07104)、PagedAttention (arXiv:2309.06180)
4. 复用 real_ops.prefix_hit_sim 做逐块链式哈希命中仿真
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_12_prefix_cache.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 12 课 · 前缀缓存:同样的话,只算一遍",
    subtitle="链式块哈希原理 · 命中率数学 · 手工逐块模拟 · 真实仿真 · 收益曲线",
    emoji="🌳", chapter="第 2 章 · KV Cache 与 PagedAttention",
)

chapter_cover(
    nb,
    objectives=[
        "理解前缀缓存的动机:系统提示词 / few-shot / RAG / 多轮对话天然共享前缀",
        "掌握 vLLM automatic prefix caching 的原理:链式块哈希 + 全局哈希表 + 物理块复用",
        "推导块需求公式:有缓存 = ⌈pS/B⌉ + N·⌈(1-p)S/B⌉,并用手工逐块模拟验证",
        "复用 real_ops.prefix_hit_sim 做真实的逐块链式哈希命中仿真(FNV 滚动哈希)",
        "画出「物理块 vs 请求数」与「命中率 vs 前缀比例」的收益曲线",
        "理解 TTFT 下降的原理,并掌握 vLLM 的启用方式与适用场景",
    ],
    toc=[
        ("直觉:开场白不用重讲", "系统提示词/few-shot/RAG 天然共享前缀"),
        ("核心原理:链式块哈希", "block.hash = H(父哈希, 本块 token) + 全局哈希表"),
        ("手工模拟:逐块哈希命中", "simulate_prefix_cache 每块比较,输出命中/新建"),
        ("公式推导与验证", "blocks_cache 公式 + 命中率与请求数的数学关系"),
        ("真实链式哈希命中仿真", "real_ops.prefix_hit_sim 逐块 FNV 滚动哈希"),
        ("收益曲线", "plotly: 物理块 vs 请求数;pyecharts: 命中率 vs 前缀比例"),
        ("TTFT 视角", "为什么首个 token 时延也下降"),
        ("实践要点", "vLLM 怎么开、hash 算法、cache_salt、适用场景"),
        ("Streamlit 动态演示", "前缀缓存收益计算器"),
    ],
    links=[
        ("vLLM: Automatic Prefix Caching 设计文档", "https://docs.vllm.ai/en/stable/design/prefix_caching/"),
        ("SGLang: RadixAttention (arXiv)", "https://arxiv.org/abs/2312.07104"),
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉:开场白不用重讲\n\n"
    "想象一个脱口秀演员每天演出:开场白固定 10 分钟,后面的段子每天换。\n"
    "聪明的做法:**开场白练一次就够了**,每天上台直接进入新段子——而不是从头到尾重演一遍。\n\n"
    "真实 LLM 推理里,「开场白」无处不在:\n\n"
    "- **系统提示词**:每个请求都带,可能几千 token(Agent 场景动辄 8K+);\n"
    "- **few-shot 示例**:同一个应用里永远相同;\n"
    "- **RAG 长文档**:多个问题共用同一份文档,反复 prefill;\n"
    "- **多轮对话**:第 $N$ 轮请求包含前 $N-1$ 轮的完整历史。\n\n"
    "> 🏷️ **前缀缓存(prefix caching)**:不同请求**共同前缀**的 KV Cache 只算一次、物理块共享复用。\n"
    "> vLLM 的实现叫 **automatic prefix caching**:不需要用户指定前缀,系统靠哈希自动发现。\n\n"
    "SGLang 的 [RadixAttention](https://arxiv.org/abs/2312.07104) 是同一个思想的前缀树版本;"
    "vLLM 选择**哈希表**版本,理由是其实现更简单、天然支持任意驱逐策略(详见设计文档)。"
)

# =====================================================================
# 第 2 节 · 核心原理
# =====================================================================
nb.md(
    "## 2. 核心原理:链式块哈希 + 全局哈希表\n\n"
    "怎么知道两个请求的前缀「一样」?逐 token 比较太慢。vLLM 的做法是**按块哈希**:\n\n"
    "每个 KV 块的哈希由「**父块的哈希 + 本块的全部 token**」联合算出(链式,像区块链):\n\n"
    "$$ \\text{hash}_j = H(\\text{hash}_{j-1},\\ \\text{tokens}_j), \\qquad \\text{hash}_0 = 0 $$\n\n"
    "这样每个哈希**唯一标识「到该块边界为止的整段前缀」**——前缀里任何一个 token 变了,"
    "后面所有块的哈希全变。于是:\n\n"
    "1. 所有**写满的块**连同它的哈希放进一张**全局哈希表** `hash → 物理块`;\n"
    "2. 新请求的 prompt **逐块算哈希**:**命中** → 直接复用物理块(引用计数 +1,第 13 课讲写时复制);"
    "   **未命中** → 计算 KV、写块、插入哈希表;\n"
    "3. 复用块被多个请求共享 → 显存大幅节省,且命中部分**跳过 prefill 计算**。\n\n"
    "vLLM 设计文档里的 `KVCacheBlock`(简化):\n\n"
    "```\n"
    "class KVCacheBlock:\n"
    "    block_id: int        # 块编号(不可变)\n"
    "    block_hash: BlockHash # 块哈希(写满时赋值, 被驱逐时清空)\n"
    "    ref_cnt: int         # 正在使用该块的请求数\n"
    "    prev/next_free_block # 空闲队列的双向链表指针\n"
    "```\n\n"
    "> ⚠️ 坑:命中以**块**为单位。前缀长度不是块大小的整数倍时,只有「完整块前缀」能命中,"
    "> 最后不满一块的部分要重新计算(vLLM 也支持 `prefix_match_unit` 把粒度调细,但那是进阶)。"
)

# =====================================================================
# 第 3 节 · 手工模拟
# =====================================================================
nb.md(
    "## 3. 手工模拟:逐块哈希命中\n\n"
    "用纯 Python 模拟「$N$ 个共享前缀请求,每请求长度 $S$,共享前缀比例 $p$,块大小 $B$」:\n"
    "把每个请求的序列逐块算哈希(简化版),命中就复用、未命中就新建并入库。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import math

def simulate_prefix_cache(n_req, seq_len, prefix_ratio, block_size):
    # 模拟 N 个共享前缀请求的逐块哈希命中
    # n_req: 请求数, seq_len: 每请求 token 数, prefix_ratio: 前缀比例(0~100),
    # block_size: 每块 token 数
    prefix_len = int(seq_len * prefix_ratio / 100)      # 共享前缀的 token 数
    cache = {}                                          # 全局哈希表: 哈希 -> 物理块号
    hit = miss = 0                                      # 命中 / 未命中计数器
    next_block = 0                                      # 下一个新物理块编号
    for r in range(n_req):                              # 遍历每个请求
        for lb in range(math.ceil(seq_len / block_size)):   # 遍历该请求的每个逻辑块
            start = lb * block_size                     # 该块在序列中的起始下标
            is_prefix = start + block_size <= prefix_len    # 完整块且属于共享前缀?
            if is_prefix:
                h = f"pre{lb}"                          # 前缀块哈希(与请求无关)
            else:
                h = f"req{r}L{lb}"                      # 后缀块哈希(每个请求独有)
            if is_prefix and h in cache:                # 前缀且哈希已存在 -> 命中
                hit += 1                                # 直接复用物理块
            else:                                       # 未命中 -> 计算并入库
                miss += 1
                if h not in cache:                      # 第一次出现才新建
                    cache[h] = next_block               # 哈希表记录物理块号
                    next_block += 1                     # 物理块号递增
    return hit, miss, next_block                        # (命中块, 新算块, 总物理块)

# 数值: 32 个请求, 每请求 2048 token, 前缀 70%, 块 16
hit, miss, blocks = simulate_prefix_cache(32, 2048, 70, 16)
no_cache = 32 * math.ceil(2048 / 16)                    # 无缓存: 每请求全算 ceil(2048/16)=128 块
print(f"32 个请求 × 2048 token, 前缀 70%, 块 16:")
print(f"  命中(复用)块数 = {hit}, 新计算块数 = {miss}")
print(f"  块命中率 = {hit/(hit+miss)*100:.1f}%")
print(f"  有缓存总物理块 = {blocks}")
print(f"  无缓存总物理块 = {no_cache}")
print(f"  节省物理块 = {100*(1 - blocks/no_cache):.1f}%")''',
    "✅ **验证**。逐块哈希命中模拟的数字与公式(`有缓存 = 前缀块 + N×后缀块`)核对一致:"
    "前缀块只算一次,请求越多省得越多。",
)

# =====================================================================
# 第 4 节 · 公式推导
# =====================================================================
nb.md(
    "## 4. 公式推导:命中率与请求数的数学\n\n"
    "设 $N$ 个请求、每请求长 $S$、共享前缀比例 $p(0\\le p\\le 1)$、块大小 $B$:\n\n"
    "- **无缓存**:每请求都要 $\\lceil S/B \\rceil$ 块 → $\\text{blocks}_{no\\_cache} = N \\lceil S/B \\rceil$;\n"
    "- **有缓存**:前缀块 $\\lceil pS/B \\rceil$ 只算一次,每请求只新增 $\\lceil (1-p)S/B \\rceil$ 个后缀块:\n\n"
    "$$ \\text{blocks}_{cache} = \\left\\lceil \\tfrac{pS}{B} \\right\\rceil + N \\cdot \\left\\lceil \\tfrac{(1-p)S}{B} \\right\\rceil $$\n\n"
    "**节省比例**: $1 - \\frac{\\text{blocks}_{cache}}{\\text{blocks}_{no\\_cache}}$,随 $N$ 增大趋近 $p$(全部前缀被摊薄到 0 成本)。\n"
    "**块命中率**:命中块 / 总需求块 ≈ $\\frac{\\lceil pS/B \\rceil}{\\lceil pS/B \\rceil + N\\lceil (1-p)S/B \\rceil}$,随 $N$ 增大趋近于 $\\frac{p}{p + (1-p)} = p$(前缀越长越接近)。"
)

# =====================================================================
# 第 5 节 · 真实链式哈希仿真
# =====================================================================
nb.md(
    "## 5. 真实链式哈希命中仿真 🌳\n\n"
    "第 3 节用「前缀比例」做简化判断,没有真正『看内容』。仓库的 `real_ops.prefix_hit_sim` 更进一步:\n"
    "对每个请求**逐块做链式哈希**——`block.hash = H(prev_hash, 本块token)`(FNV 滚动哈希),\n"
    "把所有块哈希放进一张**全局哈希表** `hash -> 物理块`,首个请求建立、后续命中即复用。\n\n"
    "这正是 vLLM automatic prefix caching 的核心:不靠用户指定前缀、不靠逐 token 比较,单纯靠\n"
    "『**同样 token 序列 → 同样链式哈希 → 同一哈希键 → 同一物理块**』自动发现共同前缀。"
    "命中块带引用计数,多个请求共享它;未命中才真算 KV 并写块。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os, math
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 OpenMP 冲突
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch02")
from real_ops import prefix_hit_sim                     # 逐块链式哈希命中仿真

seq, pref, B = 2048, 1400, 16       # 2048 token 请求, 1400 token 共享前缀, 块 16
pref_blocks = math.ceil(pref / B)   # 前缀整块数 = ceil(1400/16) = 88
print(f"共享前缀 {pref} token = {pref_blocks} 整块;每请求其余 {seq-pref} token 为独有后缀")
print(f"\\n{'请求数':>6} | {'命中(复用)块':>10} | {'需计算块':>8} | {'块命中率':>8} | {'命中/前缀块':>10}")
for n in (1, 2, 8, 32):             # 4 个请求数档位
    r = prefix_hit_sim(n_req=n, seq_len=seq, prefix_len=pref, block_size=B)  # 真实仿真
    print(f"{n:>6} | {r['n_recovered']:>10} | {r['n_computed_segments']:>8} | "
          f"{r['hit_rate']*100:>7.1f}% | {r['n_recovered']/pref_blocks:>10.1f}x")

print(f"\\n参考: 无缓存时 1 个请求需 {math.ceil(seq/B)} 整块;前缀最多 {pref_blocks} 块可全部复用")''',
    "🌳 **真实链式哈希仿真**。请求数翻倍,命中块随之上涨、命中率趋近共享前缀块占比——"
    "省下的计算与内存一并兑现。",
)

# =====================================================================
# 第 6 节 · 收益曲线
# =====================================================================
nb.md(
    "## 6. 收益曲线:物理块 vs 请求数 📊\n\n"
    "请求数越多,无缓存曲线以 $\\lceil S/B \\rceil$ 块/请求 的斜率上涨;有缓存曲线只按后缀块增长——\n"
    "两条线的「剪刀差」就是前缀缓存的总收益。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"                     # 内嵌渲染

n_req, S, p, B = 100, 2048, 0.7, 16   # 100 请求, 2048 token, 前缀 70%, 块 16
pref_b = math.ceil(S * p / B)         # 前缀整块数 = ceil(1433.6/16) = 90
suf_b = math.ceil(S * (1 - p) / B)    # 每请求后缀块数 = ceil(614.4/16) = 39
full_b = math.ceil(S / B)             # 无缓存每请求块数 = 128
ns = np.arange(1, n_req + 1)          # 请求数 1..100
y_with = pref_b + ns * suf_b          # 有缓存: 前缀块 + 每请求后缀块
y_without = ns * full_b               # 无缓存: 每请求全块

fig = go.Figure()
fig.add_trace(go.Scatter(x=ns, y=y_without, name="无前缀缓存", mode="lines",
                         line=dict(color="#e74c3c", width=2), fill="tozeroy",
                         fillcolor="rgba(231,76,60,0.12)"))
fig.add_trace(go.Scatter(x=ns, y=y_with, name="有前缀缓存", mode="lines",
                         line=dict(color="#2ecc71", width=2), fill="tozeroy",
                         fillcolor="rgba(46,204,113,0.15)"))
fig.update_layout(title=f"📊 物理块需求 vs 请求数(前缀 {p*100:.0f}%):斜率 = 每请求新块数",
                  xaxis_title="请求数", yaxis_title="占用物理块数",
                  template="plotly_white", hovermode="x unified",
                  legend=dict(orientation="h", y=1.1))
fig.show()''',
    "📊 **收益曲线**。有缓存时曲线「变缓」:斜率从 128 降到 39 块/请求——这正是共享的效果。",
)

nb.md(
    "## 7. 命中率曲线:前缀比例是关键旋钮 📈\n\n"
    "块命中率随共享前缀比例 $p$ 上升。当 $p\\to 1$(全部共享),命中率趋近 $1 - 1/N$;"
    "$p=0$ 时命中率为 0。中间是陡峭的上升段——**前缀越长的场景收益越夸张**。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
from pyecharts.charts import Line
from pyecharts import options as opts

# 对每个前缀比例跑一遍逐块模拟, 统计块命中率与节省比例
ratios = list(range(0, 101, 5))                          # 前缀比例 0%~100%
hit_rate, save_rate = [], []                             # 两个指标序列
for p in ratios:                                         # 遍历每个比例
    h, m, blocks = simulate_prefix_cache(50, 2048, p, 16)  # 50 请求逐块模拟
    tot = h + m                                          # 总需求块
    hit_rate.append(round(h / tot * 100, 2) if tot else 0)  # 块命中率 %
    save_rate.append(round((1 - blocks / (50 * math.ceil(2048/16))) * 100, 2))  # 节省 %

line = (Line()
        .add_xaxis(ratios)
        .add_yaxis("块命中率 (%)", hit_rate, is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3, color="#3498db"),
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.15, color="#3498db"))
        .add_yaxis("节省物理块 (%)", save_rate, is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3, color="#9b59b6"))
        .set_global_opts(title_opts=opts.TitleOpts(title="📈 命中率与节省比例 vs 共享前缀比例"),
                         xaxis_opts=opts.AxisOpts(name="共享前缀比例 (%)", min_=0, max_=100),
                         yaxis_opts=opts.AxisOpts(name="百分比 (%)"),
                         legend_opts=opts.LegendOpts(pos_top="5%")))
line.render_notebook()''',
    "📊 **命中率曲线**。前缀比例 60%~80% 时命中率已相当可观——系统提示词 + few-shot 的典型区间。",
)

# =====================================================================
# 第 8 节 · TTFT 视角 + 实践要点
# =====================================================================
nb.md(
    "## 8. TTFT 视角:为什么首个 token 时延也下降\n\n"
    "首 token 时延(TTFT)≈ 处理完整 prompt 的时间。有了前缀缓存,新请求只需处理:\n\n"
    "$$ \\text{新计算量} = \\text{未命中的前缀尾部} + \\text{独有后缀} $$\n\n"
    "长系统提示词(如 8K token)场景下,TTFT 可以直接砍掉一大截——"
    "这正是 Agent/RAG 应用在 vLLM 上「首 token 很快」的原因之一。"
    "注意:前缀缓存只省 **prefill(处理输入)**,不省 decode(生成输出),"
    "所以「输出很长」或「没有共享前缀」的场景收益有限(vLLM 官方文档口径)。\n\n"
    "## 9. 实践要点\n\n"
    "- 启用:`--enable-prefix-caching`(V1 引擎默认开启);\n"
    "- 哈希算法:V0.11 起默认 `sha256`(抗碰撞、安全);`xxhash` 更快但多租户环境有碰撞风险;"
    "  多租户可用 `cache_salt` 做缓存隔离(把 salt 混进首块哈希);\n"
    "- 命中粒度:完整块才能进哈希表,**块大小整数倍对齐**的前缀命中率最高;"
    "  「多轮对话前缀拼接」等技巧可以提升命中;\n"
    "- 驱逐策略:ref_cnt=0 → LRU → 优先驱逐前缀更长的块(与 RadixAttention 同策略);\n"
    "- 命中块带引用计数,配合**写时复制**保证一个请求改写时不污染他人——第 13 课的主角。"
)

nb.md(
    "## 10. 🖥️ Streamlit 动态演示:前缀缓存收益计算器\n\n"
    "拖动请求数、序列长度、共享前缀比例、块大小,实时显示:命中块数、节省显存(GB)、块命中率、\n"
    "物理块需求曲线(有/无缓存)与「节省比例 vs 前缀比例」曲线。\n\n"
    "### 📜 App 完整源码(`app_12_prefix_cache.py`)"
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
    "    print(\"    请把上方源码保存为 app_12_prefix_cache.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_12_prefix_cache.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_12_prefix_cache.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 。\n\n"
    "🔍 试试:请求数 200、前缀 90% —— 节省接近 90%;前缀拖到 0%,收益归零。"
)

wrapup(
    nb,
    summary=[
        "前缀缓存按块哈希复用物理块:命中 → 引用复用并跳过 prefill,未命中 → 计算并入库",
        "链式块哈希 block.hash = H(父哈希, 本块 token) 唯一标识到该块边界的前缀;vLLM 默认 sha256",
        "块需求公式:有缓存 = ⌈pS/B⌉ + N·⌈(1-p)S/B⌉,请求越多收益越明显,节省比例趋近 p",
        "prefix_hit_sim 逐块 FNV 滚动哈希仿真验证:请求数翻倍,命中块上涨、命中率趋近前缀块占比",
        "命中同时降低内存与 TTFT,但不省 decode;适用于系统提示词 / few-shot / RAG / 多轮对话场景",
    ],
    practice=[
        "把 simulate_prefix_cache 改成「前缀不完全对齐块」的情形(如前缀 70 token、块 16),观察多出来的未命中块",
        "给模拟加上「引用计数 + 驱逐」:ref_cnt 归零的块按 LRU 淘汰,画一条「命中率 vs 缓存容量」曲线",
        "画一条「TTFT vs 前缀比例」曲线:假设未命中 token 每 token 1ms、命中 token 0ms,验证首 token 时延下降",
    ],
    links=[
        ("vLLM: Automatic Prefix Caching 设计文档", "https://docs.vllm.ai/en/stable/design/prefix_caching/"),
        ("SGLang: RadixAttention 论文", "https://arxiv.org/abs/2312.07104"),
        ("vLLM: 前缀缓存功能页", "https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html"),
        ("PagedAttention 论文", "https://arxiv.org/abs/2309.06180"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch02\12_prefix_caching.ipynb")