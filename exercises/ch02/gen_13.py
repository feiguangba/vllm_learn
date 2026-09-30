# -*- coding: utf-8 -*-
"""生成第 13 课 notebook: Copy-on-Write 写时复制(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md 与 gen_07.py 金标准):
1. 由浅入深:图书馆参考书/OS fork 直觉 -> ref_count 三条规则 -> CowPool 手工实现逐行推演 ->
   块总数增长曲线 -> GPU cudaMemcpy 真实拷贝成本 -> 论文 §4.4 并行采样/beam search -> null_block -> 小结
2. 每一行代码都有 inline 注释;每个中间量打印并标注含义
3. 论文支撑:PagedAttention (SOSP'23, arXiv:2309.06180) §4.4、vLLM 前缀缓存设计文档(ref_cnt)、
   OS fork() 的经典语义
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_13_cow_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 13 课 · Copy-on-Write:共享的快乐,写入时才付代价",
    subtitle="引用计数 · 三条 COW 规则 · CowPool 逐行推演 · GPU 拷贝成本 · 论文共享机制",
    emoji="🖨️", chapter="第 2 章 · KV Cache 与 PagedAttention",
)

chapter_cover(
    nb,
    objectives=[
        "理解前缀缓存/并行采样带来的矛盾:多个请求共享物理块,谁有权写?",
        "掌握引用计数(ref_count)与 Copy-on-Write 的三条规则:读共享、写独享、写共享先复制",
        "手写 CowPool:共享 → 写入 → 复制 → 引用更新的完整事件流,逐行注释并打印",
        "画出「写事件数 vs 块总数」曲线,量化 COW 相对无共享方案的节省",
        "在 GPU 上实测一次块级 cudaMemcpy(COW 的真实拷贝动作)的耗时",
        "对齐论文 §4.4:并行采样、beam search、共享前缀分别怎么用 COW;理解 null_block",
    ],
    toc=[
        ("直觉:图书馆的公共参考书", "共用一本,做笔记前先复印(OS fork 的同款答案)"),
        ("COW 三条规则", "ref_count 决定:读/写独享/写共享,各怎么处理"),
        ("CowPool 手工实现", "共享 → 写入 → 复制 → 引用更新,事件日志全记录"),
        ("数值验证:块总数增长曲线", "写事件驱动,COW 比无共享省多少"),
        ("真实拷贝成本", "GPU 上测一次块级 cudaMemcpy 的耗时"),
        ("可视化:引用计数", "plotly 柱状图看 ref 从 3 → 2 → 1"),
        ("论文视角:并行采样与 beam search", "§4.4 的共享与 COW,以及 null_block 之谜"),
        ("vLLM 中的 COW", "调度器如何为 fork 与部分命中兜底"),
        ("Streamlit 动态演示", "交互式共享/写复制模拟器"),
    ],
    links=[
        ("PagedAttention 论文 (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM: Automatic Prefix Caching 设计文档", "https://docs.vllm.ai/en/stable/design/prefix_caching/"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:图书馆的公共参考书\n\n"
    "上一课,多个请求共享了同一个物理块(公共前缀)。但共享立刻带来一个尖锐的问题:\n\n"
    "> **如果某个请求想「写」这块内存,会不会改坏别人正在读的数据?**\n\n"
    "想象图书馆里的公共参考书:所有人都可以读同一本(省纸!)。但如果有同学想**在上面做笔记**,怎么办?\n"
    "总不能让大家各买一本(浪费!),也不能让笔记写进公共书里(污染别人!)。\n\n"
    "答案:📋 **写时复制(Copy-on-Write, COW)**——先复印一份给他,他在**副本**上做笔记;"
    "公共书继续供大家阅读。复印的代价只在「真有人要写」的那一刻发生。\n\n"
    "> 📄 这是操作系统 `fork()` 的经典语义:子进程与父进程共享内存页,任一进程**写入**共享页时才复制。"
    "> PagedAttention 论文(SOSP'23)§4.4 把这个机制按**块粒度**用在了 KV Cache 上——"
    "> 论文原文:「*the copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block*」。"
)

# =====================================================================
# 第 2 节 · COW 三条规则
# =====================================================================
nb.md(
    "## 2. COW 三条规则:一句话讲清\n\n"
    "每块内存带一个**引用计数 ref_count**(有多少请求正在读它):\n\n"
    "1. **读(只读共享)**:免费,`ref_count` 不动——共享不花钱;\n"
    "2. **写,且 ref_count == 1**:你是唯一读者,直接就地写,`ref_count` 不变;\n"
    "3. **写,且 ref_count > 1**:先**复制**一块新内存(新块 ref_count = 1),把你的引用**从旧块移到新块**,\n"
    "   旧块 ref_count 减一,然后在新块上写。\n\n"
    "> 🏷️ 复制发生在「写」的瞬间,而不是「共享」的瞬间——这就是「copy-**on-write**」名字的来历。\n\n"
    "代价分析:若 $N$ 个请求共享一块、最终只有 $K$ 个请求写入它,则最多复制 $K$ 次;"
    "若不共享,一开始就要 $N$ 份。COW 把成本从「共享时就付」推迟到「真正分歧时」,"
    "对长共享前缀场景往往省一个数量级的内存。"
)

# =====================================================================
# 第 3 节 · CowPool 手工实现
# =====================================================================
nb.md(
    "## 3. CowPool 手工实现:共享 → 写入 → 复制\n\n"
    "用纯 Python 实现 COW 块池:每个请求一张块表,若干共享前缀块 + 每请求独有后缀块。"
    "写入操作自动查 `ref_count`:大于 1 先复制。"
)

nb.code(
    '''# -*- coding: utf-8 -*-

class CowBlock:
    # 一块带引用计数的 KV 块(元数据)
    def __init__(self, ref: int = 1):
        self.ref = ref          # 引用计数: 正在读/持有它的请求数
        self.version = 1        # 版本号: 每次写入 +1(便于追踪)
        self.owners = set()     # 持有它的请求集合

class CowPool:
    # 带 COW 语义的块池
    def __init__(self, n_prefix: int, n_req: int):
        self.blocks = {}                     # 块 id -> CowBlock
        self.log = []                        # 事件日志
        self.next_id = 0                     # 下一个新块编号
        self.tables = {f"req{r}": [] for r in range(n_req)}   # 每请求一张块表
        for p in range(n_prefix):            # 共享前缀块: 所有请求共有
            b = CowBlock(ref=n_req)          # ref = 请求数(大家都读)
            b.owners = set(range(n_req))     # 所有请求都持有
            self.blocks[self.next_id] = b    # 登记块
            for r in range(n_req):           # 每个请求的表都包含前缀块
                self.tables[f"req{r}"].append(self.next_id)
            self.next_id += 1                # 下一个块号
        for r in range(n_req):               # 每请求独有后缀块
            b = CowBlock(ref=1)              # 只有自己读
            b.owners = {r}
            self.blocks[self.next_id] = b
            self.tables[f"req{r}"].append(self.next_id)   # 只进自己的表
            self.next_id += 1

    def write(self, owner: int, block_id: int):
        # owner 请求写入 block_id: ref>1 则先复制(COW), 否则就地写
        b = self.blocks[block_id]            # 取出目标块
        if b.ref > 1:                        # 被共享: 必须复制
            new_id = self.next_id            # 分配新块号
            self.next_id += 1
            self.blocks[new_id] = CowBlock(ref=1)       # 新块 ref=1
            self.blocks[new_id].owners = {owner}        # 只有 owner 持有
            self.blocks[new_id].version = b.version + 1 # 版本继承并 +1
            b.ref -= 1                       # 旧块引用 -1(owner 离开了)
            b.owners.discard(owner)          # owner 从旧块移除
            row = self.tables[f"req{owner}"] # owner 的块表
            row[row.index(block_id)] = new_id           # 把旧块换成新块
            self.log.append(
                f"COW: 请求{owner} 写共享块 {block_id}(ref={b.ref+1}) -> 复制为块 {new_id}")
        else:                                # 独享: 直接写
            b.version += 1                   # 版本 +1
            self.log.append(f"直接写: 请求{owner} 写独享块 {block_id}, 版本 -> {b.version}")

# --------------------------------------------------------------------------
# 演练: 3 个请求共享 2 个前缀块 + 各自 1 个后缀块
# --------------------------------------------------------------------------
pool = CowPool(2, 3)                         # 2 前缀块, 3 个请求
print("初始化块表:", {k: v for k, v in pool.tables.items()})
print("初始化引用:", {k: b.ref for k, b in pool.blocks.items()})
print("\\n执行写入:")
pool.write(0, 0)                             # 请求 0 写共享前缀块 0 -> 触发 COW
pool.write(1, 0)                             # 请求 1 也写它 -> 再触发 COW
pool.write(2, 4)                             # 请求 2 写自己的独享块 -> 直接写
print("\\n".join(pool.log))                   # 打印事件日志
print("\\n末态引用计数:", {k: b.ref for k, b in pool.blocks.items()})
print("请求 0 的块表:", pool.tables["req0"])''',
    "🎨 **图示**。事件日志清楚地展示「什么时候复制、什么时候直接写」:"
    "前缀块 0 被两个请求写过,产生 2 个 COW 副本;独享块 4 直接写。",
)

# =====================================================================
# 第 4 节 · 数值验证:块总数增长曲线
# =====================================================================
nb.md(
    "## 4. 数值验证:写事件 vs 块总数 📈\n\n"
    "观察:随着「写事件」不断发生,块总数从「共享基线」向「完全独立」逼近——"
    "曲线越缓,COW 省得越多。4 个请求共享 3 个前缀块,每个请求依次写前缀块 0:"
)

nb.code(
    '''# -*- coding: utf-8 -*-
from pyecharts.charts import Line
from pyecharts import options as opts

pool2 = CowPool(3, 4)                        # 4 个请求共享 3 个前缀块 + 各自 1 后缀块
n_blocks = [len(pool2.blocks)]               # 初始块数 = 3 + 4 = 7
for owner in [0, 1, 2, 3]:                   # 4 个请求依次写入共享块 0
    pool2.write(owner, 0)                    # 前 3 次 ref>1 触发 COW(新块 +1), 第 4 次 ref=1 就地写
    n_blocks.append(len(pool2.blocks))       # 记录当前块总数

naive = 4 * (3 + 1)                          # 无共享: 4 请求 × (3 前缀 + 1 后缀) = 16 块
print(f"COW 块总数序列 = {n_blocks}  <- 从 {n_blocks[0]} 起步, 前 3 次写各 +1, 末次 ref=1 就地写不再 +1")
print(f"无共享基线     = {naive} 块  <- 一开始就 16 块, 全部独立")

line = (Line()
        .add_xaxis(list(range(len(n_blocks))))       # 横轴: 写事件数
        .add_yaxis("块总数(COW)", n_blocks, is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3, color="#2ecc71"),
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.15, color="#2ecc71"))
        .add_yaxis("无共享基线(全部独立)", [naive] * len(n_blocks),
                   linestyle_opts=opts.LineStyleOpts(width=2, color="#e74c3c", type_="dashed"))
        .set_global_opts(title_opts=opts.TitleOpts(title="📈 写事件 vs 块总数: COW 让增长变缓"),
                         xaxis_opts=opts.AxisOpts(name="写事件数"),
                         yaxis_opts=opts.AxisOpts(name="块总数"),
                         legend_opts=opts.LegendOpts(pos_top="5%")))
line.render_notebook()''',
    "📊 **增长曲线**。红虚线是无共享的「全价」16 块;绿线从共享基线 7 块出发,前 3 次写入各 +1 块(到 10),"
    "第 4 次写入时共享块 ref 已降为 1、改为就地写,块数停在 10——这正是「复制成本推迟到分歧时」的直观体现。",
)

# =====================================================================
# 第 5 节 · 真实拷贝成本
# =====================================================================
nb.md(
    "## 5. COW 的拷贝成本:一次块级 cudaMemcpy 的真实耗时 ⏱️\n\n"
    "COW 的核心动作是「**写入时才复制**」,在 GPU 上对应一次设备到设备的 `cudaMemcpy`——"
    "把整块 KV 从旧物理块搬到新物理块。块通常很小(16 token × 多层 × 多头),"
    "所以单次拷贝应该极快;但共享一旦演变为**每次写入都分歧**,拷贝次数就会追上共享数,COW 的收益就消失。\n\n"
    "下面在真实 GPU 上量一块 bf16 KV 的拷贝耗时(`torch.copy_` 底层即 `cudaMemcpyAsync`);"
    "**未安装 CUDA 时自动回退到 CPU**,演示同一个拷贝动作。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 OpenMP 冲突
import torch

# 设备选择: 有 CUDA 用 GPU, 否则优雅回退到 CPU(拷贝动作照样可演示)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"运行设备: {device}  (无 CUDA 时自动回退到 CPU)")

# 一块 KV 的真实形状: (2, L, H_kv, block_size, head_dim)  bf16
# 2 = K/V 两层, L=32 层, H_kv=8 头, block_size=16 token, head_dim=128
L, Hkv, D, BS = 32, 8, 128, 16               # 层数 / KV 头数 / 头维度 / 块大小
kw = 2 * L * Hkv * BS * D * 2                # 一块 bf16 KV 的字节数
src = torch.randn(2, L, Hkv, BS, D, device=device, dtype=torch.bfloat16)  # 源块
dst = torch.empty_like(src)                  # 目标块(同形状)
print(f"源块 shape = {tuple(src.shape)}  <- (K/V=2, layers={L}, kv_heads={Hkv}, block_size={BS}, head_dim={D})")
print(f"一块字节数 = {kw/2**10:.0f} KiB")

for _ in range(5):                           # 预热 5 次(避免首次分配抖动)
    dst.copy_(src)                           # 拷贝操作
if device == "cuda":                         # GPU 上才需要同步(CPU 拷贝是同步的)
    torch.cuda.synchronize()                 # 等 GPU 完成
t0 = time.perf_counter()                     # 计时起点
for _ in range(200):                         # 连续拷贝 200 次(模拟 COW)
    dst.copy_(src)                           # 每次 = 一次块级 cudaMemcpy
if device == "cuda":                         # GPU 上等全部完成再计时
    torch.cuda.synchronize()
per_s = (time.perf_counter() - t0) / 200     # 单次平均耗时(秒)
print(f"单块 COW 拷贝: {kw/2**10:.0f} KiB -> 一次 {per_s*1e6:.1f} µs")

for n in (16, 128, 1024):                    # 连续分歧 N 块的情形
    print(f"  连续 COW {n:5d} 块 ≈ {per_s*n*1e3:.2f} ms")   # 拷贝 N 次的总耗时''',
    "🚀 **真实数字**。单块拷贝只要几十微秒——所以只要不是「每次写都分歧」,"
    "COW 的拷贝成本远低于它换来的内存节省。这也解释了为什么 block 不能太大:块越大单次拷贝越贵、浪费越多。",
)

# =====================================================================
# 第 6 节 · 可视化 + 论文视角
# =====================================================================
nb.md(
    "## 6. 可视化:引用计数的变化 📊\n\n"
    "把写入前后的引用计数画成柱状图:共享块 0 的 ref 从 3 降为 1(两个请求复制走了),"
    "新复制的块 ref=1。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

# 用第 3 节演练后的 pool 状态
blk_ids = sorted(pool.blocks)                # 所有块编号(含 COW 复制出的新块)
refs = [pool.blocks[i].ref for i in blk_ids]  # 每块当前引用计数
colors = ["#e74c3c" if pool.blocks[i].ref > 1 else "#3498db" for i in blk_ids]  # 红=仍共享, 蓝=独享
fig = go.Figure(go.Bar(
    x=[f"块 {i}" for i in blk_ids],          # 横轴: 块编号
    y=refs,                                  # 纵轴: ref_count
    marker_color=colors,                     # 颜色区分共享/独享
    text=[f"v{pool.blocks[i].version}" for i in blk_ids],   # 标版本号
    textposition="outside"))
fig.add_hline(y=1, line_dash="dot", line_color="gray")      # 参考线 y=1
fig.update_layout(title="📊 末态引用计数: 蓝=独享, 红=仍被共享", yaxis_title="ref_count",
                  template="plotly_white", height=300, margin=dict(t=60))
fig.show()''',
    "📊 前缀块 0 因为两次写入被复制,ref 从 3 降到 1;新块 5、6 都是 ref=1。"
    "共享的快乐:没被写过的块一直保持 ref>1,零成本共享。",
)

nb.md(
    "## 7. 论文视角:并行采样、beam search 与 null_block\n\n"
    "PagedAttention 论文 §4.4 的**内存共享**就是 COW 的三个典型场景:\n\n"
    "**① 并行采样(parallel sampling)** 一个请求生成多个输出,它们共享整个 prompt 的 KV。"
    "论文 Fig.8:两个 sample 的 prompt 逻辑块**都映射到同一组物理块**(ref=2),"
    "各自生成时才在最后一块触发 COW——「*we only reserve space for one copy of the prompt's state*」。\n\n"
    "**② beam search** 多个候选序列像进程树一样共享前缀块、中途分叉。"
    "论文实测在 beam 宽度 4~6 下 **37.6%~55.2%** 的块因共享而省下(ShareGPT 轨迹 44.3%~66.3%);"
    "旧系统需要频繁整段拷贝 KV,COW 只需按块复制。\n\n"
    "**③ 共享前缀(shared prefix)** 服务商把系统提示词等公共前缀预分配为**只读共享块**;"
    "新请求直接把逻辑块映射过去,末块标记 COW(论文原文: last block marked copy-on-write)。\n\n"
    "### null_block 之谜 🔍\n\n"
    "块池会预留一整块**全零、只读**的 `null_block`(vLLM `v1/core/kv_cache_utils.py` 里 `is_null=True`)。\n"
    "凡是调度器「要腾地方 / 还没填满」的位置就用它顶上,读到它返回安全的全零 KV,"
    "避免指到已释放内存造成悬垂读;它的 `ref_cnt` 被特殊标记、不可被写入。"
)

nb.md(
    "## 8. vLLM 中的 COW:fork 与部分命中兜底\n\n"
    "vLLM V1 中 COW 由调度器 + 块管理器配合完成:`KVCacheBlock.ref_cnt` 记录共享度;"
    "前缀缓存**部分命中**(命中边界落在块内部,`prefix_match_unit`)或某请求与邻居**分叉**时,"
    "调度器对该块执行「分配新块 + 拷贝数据 + 更新块表」——与我们 `CowPool.write` 完全同构。\n\n"
    "思考:decode 阶段每个请求的**后缀**几乎必然不同,因此后缀块会逐渐全部复制走;"
    "但只要**前缀块永远不改写**(只读共享),它就始终 ref>1、永不复制——"
    "这正是前缀缓存能安全落地、又几乎不付出拷贝成本的基石。"
)

nb.md(
    "## 9. 🖥️ Streamlit 动态演示:交互式 COW 模拟器\n\n"
    "构造多个共享前缀的请求,用按钮**让某个请求写入某个块**,实时观察:块引用计数柱状图、\n"
    "所有权矩阵热图、事件日志,以及「当前块数 vs 无 COW 的块数」的节省。\n\n"
    "### 📜 App 完整源码(`app_13_cow_demo.py`)"
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
    "    print(\"    请把上方源码保存为 app_13_cow_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_13_cow_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_13_cow_demo.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 。\n\n"
    "🔍 试试:3 个请求共享 3 个前缀块,让请求 0、1、2 依次写入块 0——注意 ref 从 3 → 2 → 1,并诞生两个新块。"
)

wrapup(
    nb,
    summary=[
        "共享不花钱:只读场景 ref_count 不动,物理块零成本复用——COW 只在「写共享块」的瞬间触发",
        "三条规则:读不动 ref;写 ref=1 就地写;写 ref>1 先复制(新块 ref=1)、旧块 ref-1、更新块表",
        "块总数曲线:COW 从共享基线出发、每次分歧 +1 块,远低于无共享的「全价」;论文实测 beam search 省 37.6%~55.2%",
        "单块 cudaMemcpy 只要几十微秒,拷贝成本远小于省下的内存——前提是不要「每次写都分歧」",
        "vLLM 在并行采样、beam search、前缀缓存部分命中处使用 COW,配合 null_block 防止悬垂读",
    ],
    practice=[
        "给 CowPool 加 free(owner):请求结束时释放其全部块,ref 归零的块删除并复用编号",
        "模拟 100 个写事件,统计「复制次数 vs 直接写次数」随共享请求数的变化,画两条曲线",
        "实现一个「无 COW 的朴素方案」(写入即全量复制整个序列),对比两种方案的块总数曲线与拷贝量",
    ],
    links=[
        ("PagedAttention 论文 (Section 4.4 内存共享)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM: Automatic Prefix Caching 设计文档", "https://docs.vllm.ai/en/stable/design/prefix_caching/"),
        ("vLLM 源码: v1/core/block_pool.py", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch02\13_cow_copy_on_write.ipynb")