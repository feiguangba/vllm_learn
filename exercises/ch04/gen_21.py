# -*- coding: utf-8 -*-
"""生成第 21 课 notebook: 变长序列组批量张量组装(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:拼桌餐厅直觉 -> 五个核心张量 -> 组装函数逐行推演 -> 一致性验证 -> padding 浪费 -> 真实 GPU -> vLLM 源码对照
2. 每一行代码都有 inline 注释
3. 每个张量打印 shape + 维度含义
4. 论文支撑:vLLM ModelRunner prepare_inputs、PagedAttention (arXiv:2309.06180) 的 block 概念
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_21_metadata.py"
APP_NAME = "app_21_metadata.py"
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
    "第 21 课 · 变长序列组批量张量组装",
    subtitle="input_ids · positions · slot_mapping · block_table · seq_lens 的拼接与索引",
    emoji="🧩", chapter="第 4 章 · 模型执行器与 CUDA 优化",
)

chapter_cover(
    nb,
    objectives=[
        "理解问题:一个调度批里每条序列长度不同,GPU 需要一组统一的张量",
        "逐个拆解五个核心张量:input_ids / positions / slot_mapping / block_table / seq_lens",
        "手写组装函数:拼接 + 记录索引,逐行推演每个张量的 shape 与语义",
        "用断言验证三条前置条件:slot 唯一、位置连续、块内落位正确",
        "量化 padding 浪费:如果偷懒 pad 到等长,会白烧多少算力",
        "用真实 GPU 的 prefill 吞吐把 padding 浪费换算成秒",
        "对照 vLLM ModelRunner 的 prepare_inputs 真实源码路径",
    ],
    toc=[
        ("问题:变长序列怎么塞进一个批次", "拼桌餐厅的比喻"),
        ("五个核心张量逐个拆解", "每个张量的 shape 与语义"),
        ("组装函数:拼接 + 记录索引", "逐行推演, 打印每个中间张量"),
        ("一致性验证", "三条前置条件断言"),
        ("padding 浪费有多大", "数值 + 可视化"),
        ("表格核对 + 热力图", "把组装结果验证一遍"),
        ("真实 GPU:拼好的词元流交到 GPU 的刻度", "prefill 吞吐 × padding 浪费 = 白烧秒数"),
        ("与 vLLM 源码结构对应", "prepare_inputs / slot_mapping 真实路径"),
    ],
    links=[
        ("vLLM ModelRunner 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM V1 设计文档", "https://docs.vllm.ai/en/latest/design/v1/v1_usage.html"),
        ("K4i.top: SchedulerOutput -> GPU 前向", "https://k4i.top/posts/model-runner-scheduler-output-to-gpu-forward/"),
    ],
)

# =====================================================================
# 第 1 节 · 问题
# =====================================================================
nb.md(
    "## 1. 问题:变长序列怎么塞进一个批次? 🍜\n\n"
    "想象你去一家**拼桌餐厅**:一桌客人先到,已经吃到一半;新客人后到,才刚开始点菜。\n"
    "服务员要把两桌的菜同时送,就得把「已上的菜 + 正在上的菜」按**座位**理清。\n\n"
    "vLLM 的 `ModelRunner.execute_model` 拿到调度输出后,面对的是同样的问题:\n"
    "一个**调度批(schedule batch)**里有若干条序列,每条的长度各不相同。\n"
    "GPU 前向需要一张扁平的**一维词元流**,但每条序列必须能**恢复现场**:\n"
    "我的词元在流的哪个区间?我的绝对位置在哪?我的 KV 寄存在哪个槽位?\n\n"
    "答案就是一组**元数据张量**(metadata),本课逐个拆解并亲手组装。"
)

# =====================================================================
# 第 2 节 · 五个张量
# =====================================================================
nb.md(
    "## 2. 五个核心张量逐个拆解 🧩\n\n"
    "设一个调度批有 $S$ 条序列,第 $s$ 条长度为 $l_s$,总词元数 $T = \\sum_s l_s$。\n\n"
    "| 张量 | shape | 语义 |\n"
    "|---|---|---|\n"
    "| `input_ids` | $(T,)$ | 所有序列的词元拼接成一维流,GPU 一次吃进去 |\n"
    "| `positions` | $(T,)$ | 每个词元在自己序列中的绝对位置(0,1,2,…) |\n"
    "| `slot_mapping` | $(T,)$ | 每个词元的 KV 在缓存中的**扁平槽位号** |\n"
    "| `block_table` | $(S, \\lceil l_s / B \\rceil)$ | 每条序列的逻辑块 → 物理块映射(块大小 B) |\n"
    "| `seq_lens` | $(S,)$ | 每条序列当前长度 |\n\n"
    "`input_ids` 是「流水线的主传送带」;后四个张量是「恢复现场」的索引。\n"
    "> 📄 vLLM 的 `prepare_inputs` 正是产出这些张量;`slot_mapping` 来自 PagedAttention\n"
    "> 论文(arXiv:2309.06180)的 block 表——每个词元都有一格扁平 KV 槽位。"
)

# =====================================================================
# 第 3 节 · 组装函数
# =====================================================================
nb.md(
    "## 3. 组装函数:拼接 + 记录索引 🛠️\n\n"
    "核心思想就八个字:**拼接 + 记录索引**。\n"
    "手工构造一个批次:4 条序列,长度 9 / 3 / 6 / 4,块大小 4。逐行推演组装过程。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import torch                                      # 深度学习库 (用 numpy 构造, 末尾转 torch)

# --------------------------------------------------------------------------
# 1. 构造输入: 4 条变长序列
# --------------------------------------------------------------------------
block_size = 4                                     # 每个 KV 块容纳 4 个词元 (vLLM 默认思路)
seq_lens = np.array([9, 3, 6, 4], dtype=np.int64)  # (S=4,) 每条序列的词元数
n_seq = len(seq_lens)                              # S = 4
total_tokens = int(seq_lens.sum())                 # T = 9+3+6+4 = 22
print(f"seq_lens shape = {seq_lens.shape}  <- (S={n_seq},) 每条序列长度")
print(f"total_tokens  = {total_tokens}  <- T, 拼接后的词元流长度")

# 每条序列的词元 id (随机, 只关心结构)
rng = np.random.default_rng(2026)                  # 可复现随机数
token_ids = [rng.integers(0, 50, size=s) for s in seq_lens]   # 4 段词元
print(f"每条序列的词元数 = {[len(t) for t in token_ids]}")

# --------------------------------------------------------------------------
# 2. 拼接 input_ids: 一维词元流
# --------------------------------------------------------------------------
input_ids = np.concatenate(token_ids).astype(np.int64)   # (T=22,) 拼起来
print(f"\\ninput_ids shape = {input_ids.shape}  <- (T={total_tokens},) 全部词元拼成一维流")
print(f"  input_ids = {input_ids.tolist()}")

# --------------------------------------------------------------------------
# 3. positions: 每个词元在自己序列里的绝对位置
# --------------------------------------------------------------------------
positions = np.concatenate([np.arange(s) for s in seq_lens]).astype(np.int64)  # (T,)
print(f"positions shape = {positions.shape}  <- (T={total_tokens},) 每条序列从 0 数起")
print(f"  positions = {positions.tolist()}")

# --------------------------------------------------------------------------
# 4. 拼接区间 offsets: 每条序列在流中的 [start, end)
# --------------------------------------------------------------------------
offsets = np.concatenate([[0], np.cumsum(seq_lens)])[:-1]   # (S,) 每条序列起点
print(f"\\noffsets shape = {offsets.shape}  <- (S={n_seq},) 每条序列的起始下标")
print(f"  offsets   = {offsets.tolist()}  (序列 s 的区间 = [offsets[s], offsets[s]+seq_lens[s]))")

# --------------------------------------------------------------------------
# 5. block_table: 每条序列的逻辑块 -> 物理块
# --------------------------------------------------------------------------
blocks_needed = np.ceil(seq_lens / block_size).astype(int)  # (S,) 每条序列需几块
block_table = []                                     # 每条序列的块号表
free_id = 0                                          # 物理块分配游标
for s in range(n_seq):                               # 逐序列分配
    blks = []                                        # 该序列的块表
    for _ in range(blocks_needed[s]):                # 分配所需块数
        blks.append(free_id)                         # 发一块物理块号
        free_id += 1                                 # 游标前进
    block_table.append(blks)                         # 记录
bt = np.full((n_seq, int(blocks_needed.max())), -1, dtype=np.int64)  # (S, max_blocks) 带 -1 pad
for s, blks in enumerate(block_table):               # 填表
    bt[s, :len(blks)] = blks                         # 有效块填入
print(f"\\nblock_table shape = {bt.shape}  <- (S={n_seq}, max_blocks={bt.shape[1]}) 行=序列 列=逻辑块")
for s in range(n_seq):                               # 打印每行
    print(f"  序列{s}: {bt[s].tolist()}  <- 逻辑块 0..{blocks_needed[s]-1} 映射到物理块")

# --------------------------------------------------------------------------
# 6. slot_mapping: 每个词元的扁平 KV 槽位号
# --------------------------------------------------------------------------
slot_mapping = np.full(total_tokens, -1, dtype=np.int64)  # (T,) 先填 -1
for s in range(n_seq):                               # 逐序列
    base = np.array(block_table[s]) * block_size     # 每块的基址 = 物理块号 × 块大小
    for i in range(int(seq_lens[s])):                # 序列内第 i 个词元
        # 槽位 = 所在物理块的基址 + 块内偏移
        slot_mapping[offsets[s] + i] = base[i // block_size] + (i % block_size)
print(f"slot_mapping shape = {slot_mapping.shape}  <- (T={total_tokens},) 每词元一个槽位")
print(f"  slot_mapping = {slot_mapping.tolist()}")''',
    "🛠️ **逐行推演**。关键一行是 `slot_mapping[offsets[s]+i] = base[i//block_size] + (i%block_size)`:"
    "把「流的第几号词元」映射到「物理块的哪个槽」。这就是 vLLM 的 `_get_slot_mapping` 在做的计算。",
)

# =====================================================================
# 第 4 节 · 一致性验证
# =====================================================================
nb.md(
    "## 4. 一致性验证:三条前置条件 ✅\n\n"
    "组装正确与否,可以用三条**硬性断言**验证——它们分别对应 vLLM 的三条前置条件:\n\n"
    "1. **槽位唯一**:`slot_mapping` 里没有重复(每个 KV 槽位只属于一个词元);\n"
    "2. **位置连续**:`positions` 按序列切分后,每条都是 0,1,2,…,lₛ−1;\n"
    "3. **块内落位正确**:词元 `i` 的槽位 = 块号×块大小 + 块内偏移。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

# 验证 1: 槽位唯一 (vLLM: slot_mapping 必须一对一)
uniq = len(np.unique(slot_mapping))                 # 去重后的槽位数
assert uniq == total_tokens, "槽位不唯一!"          # 断言: 与总词元数相等
print(f"✅ 槽位唯一: {uniq} 个唯一槽位 = {total_tokens} 个词元")

# 验证 2: 位置连续 (vLLM: positions 每序列从 0 连续递增)
for s in range(n_seq):                              # 逐序列检查
    seg = positions[offsets[s]:offsets[s] + seq_lens[s]]   # 该序列的 positions 段
    expected = np.arange(seq_lens[s])               # 期望: 0..l_s-1
    assert np.array_equal(seg, expected), f"序列{s} 位置不连续!"  # 断言
print("✅ 位置连续: 每条序列的 positions 都是 0,1,2,...,l_s-1")

# 验证 3: 块内落位正确 (vLLM: 槽位 = 物理块号×块大小 + 块内偏移)
for s in range(n_seq):                              # 逐序列检查
    for i in range(int(seq_lens[s])):               # 每个词元
        got = slot_mapping[offsets[s] + i]          # 实际槽位
        exp = block_table[s][i // block_size] * block_size + (i % block_size)  # 期望
        assert got == exp, f"序列{s} 词元{i} 落位错误!"   # 断言
print("✅ 块内落位正确: 槽位 = 物理块号 × block_size + 块内偏移")

# 汇总: 打印组装结果表
print("\\n汇总表: 序列 | 区间 | positions段 | 块表 | 槽位段")
for s in range(n_seq):                              # 逐序列打印
    start, end = offsets[s], offsets[s] + seq_lens[s]   # 拼接区间
    print(f"  seq{s} | [{start},{end}) | {positions[start:end].tolist()} | "
          f"{block_table[s]} | {slot_mapping[start:end].tolist()}")''',
    "✅ **三条断言全过**。这些断言不是装饰——vLLM 的 PagedAttention kernel 假设这些性质成立,\n"
    "违反任何一条都会产生错乱的结果。",
)

# =====================================================================
# 第 5 节 · padding 浪费
# =====================================================================
nb.md(
    "## 5. padding 浪费有多大? 📊\n\n"
    "如果我们偷懒,把所有序列 pad 到最长,矩阵形状就变成 `[S, max_len]`。\n"
    "真正的有效词元只有 $T$,而 pad 后的格子有 $S \\times \\max\\_len$ 个。\n\n"
    "$$ \\text{waste} = 1 - \\frac{T}{S \\times \\max\\_len} $$\n\n"
    "算账:批越大、长度差越大,padding 浪费越惊人。这是 vLLM 坚持紧凑拼接的核心理由。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库

max_len = int(seq_lens.max())                      # 批内最长序列
padded_cells = n_seq * max_len                     # pad 后总格子数
waste = 1.0 - total_tokens / padded_cells          # 浪费比例
print(f"最长序列 max_len  = {max_len}")
print(f"紧凑拼接的格子数   = {total_tokens} (T, 全部有效)")
print(f"pad 到等长的格子数 = {padded_cells} (S × max_len)")
print(f"padding 浪费      = {waste * 100:.1f}%  <- 这些格子白白占算力")

# 更大的批次更夸张: 模拟 64 条随机长度序列
rng = np.random.default_rng(0)                     # 可复现随机数
big_lens = rng.integers(4, 65, size=64)            # 64 条序列, 长度 4..64
big_T = int(big_lens.sum())                        # 总词元
big_waste = 1.0 - big_T / (len(big_lens) * int(big_lens.max()))  # 浪费比例
print(f"\\n64 条随机长度序列: T={big_T}, max_len={big_lens.max()}")
print(f"  padding 浪费 = {big_waste * 100:.1f}%  <- 批越大越夸张")''',
    "📊 **算账**。这个浪费比例乘上真实 prefill 吞吐,就是第 7 节要算的「白烧秒数」。",
)

nb.md(
    "### padding 浪费的静态柱状图 🎨\n\n"
    "灰色虚线 = `max_len` 水位线;红色段 = 若偷懒 pad 会白白多算的词元。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8, 4))             # 画布
x = np.arange(n_seq)                               # 序列编号
ax.bar(x, seq_lens, color="#4C72B0", label="有效词元")   # 有效长度柱
ax.bar(x, max_len - seq_lens, bottom=seq_lens, color="#C44E52", label="padding 浪费")  # 浪费段
ax.axhline(max_len, color="gray", ls="--", lw=1.2, label="max_len 水位线")   # 水位线
ax.set_xticks(x)                                   # x 刻度
ax.set_xticklabels([f"seq{s}\\nlen={seq_lens[s]}" for s in range(n_seq)])   # 标签
ax.set_ylabel("词元数")                             # y 轴
ax.set_title("若 pad 到等长: 红段 = 白烧的算力")      # 标题
ax.legend()                                        # 图例
ax.grid(axis="y", alpha=0.3)                       # 网格
plt.tight_layout()
plt.show()''',
    "📊 **红段即浪费**。紧凑拼接没有红段——这是 vLLM 坚持紧凑拼接(flatten + 元数据)的核心理由。",
)

# =====================================================================
# 第 6 节 · 表格核对 + 热力图
# =====================================================================
nb.md(
    "## 6. 表格核对 + 热力图:把组装结果验证一遍 🧭\n\n"
    "把组装结果以纯文本表格核对一遍(逐序列展示拼接区间与槽位),再看 `positions` 热力图。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
# 纯文本表格核对 (不用 pandas: 本机 torch 导入后再建 DataFrame 会在 IPython 内核崩溃,
# 这是 Windows 上 torch+pandas 的已知环境问题, 避开即可——表本身只是文本排版)
print("序列 | 长度 | 拼接区间 | positions                | block_table | slot_mapping")
for s in range(n_seq):                             # 逐序列
    start, end = offsets[s], offsets[s] + seq_lens[s]   # 拼接区间
    # 每列用 f-string 排版, 宽度对齐
    print(f"seq{s} | {int(seq_lens[s]):4d} | [{start:2d},{end:2d}) | "
          f"{str(positions[start:end].tolist()):24s} | {str(block_table[s]):12s} | {slot_mapping[start:end].tolist()}")
print("\\n核对: 每条序列的 slot_mapping 段都落在自己的块号 x 4 + [0..3] 上")''',
    "📋 **表格核对**。序列 1 长 3 词元,占 1 块,槽位 = 块号×4 + [0,1,2]。",
)

# =====================================================================
# 第 7 节 · 真实 GPU
# =====================================================================
nb.md(
    "## 7. 真实 GPU:拼好的词元流交到 GPU 的刻度 ⏱️\n\n"
    "上面我们把变长批拼成了 `input_ids + positions + slot_mapping + block_table`。\n"
    "这套「拼好的词元流」最终要被 GPU 前向吃掉。用跨章共享库 `vllm_real` 实测\n"
    "真实 prefill 每秒能处理多少词元,再把第 5 节的 padding 浪费换算成「白烧的秒数」。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import sys, os                                   # 系统库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 兼容
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_prefill_decode, cuda_info   # 跨章共享真实微基准

print("设备:", cuda_info())                        # 设备
r = bench_prefill_decode(L=256, steps=32, reps=5)  # 实测 prefill 吞吐
prefill_tok_s = r["prefill_tok_per_s"]             # 每秒能 prefill 多少词元
print(f"真实 prefill 吞吐 = {prefill_tok_s:.0f} token/s")

# 把 padding 浪费换算成白烧的秒数: 用「64 条随机序列」那组数字
wasted_tokens = padded_cells - total_tokens        # 本批多算的词元数
wasted_ms = wasted_tokens / prefill_tok_s * 1000   # 白烧的毫秒
print(f"本批 (4 序列) padding 浪费词元 = {wasted_tokens}")
print(f"  => 白烧 {wasted_ms:.2f} ms (若偷懒 pad 到等长)")

# 更大批次: 用 64 序列那组
big_wasted = int(big_lens.max() * len(big_lens) - big_T)  # 大批准浪费
big_ms = big_wasted / prefill_tok_s * 1000         # 大批准白烧毫秒
print(f"64 序列批 padding 浪费词元 = {big_wasted}")
print(f"  => 白烧 {big_ms:.1f} ms —— 这就是 vLLM 坚持紧凑拼接省下的算力")''',
    "🚀 **真实 GPU 数字**。把「padding 浪费比例」乘上「真实 prefill 每秒处理词元数」,"
    "就是 vLLM 坚持紧凑拼接省下的白烧算力——逻辑层结论在物理层得到了刻度。",
)

# =====================================================================
# 第 8 节 · vLLM 源码对照
# =====================================================================
nb.md(
    "## 8. 与 vLLM 源码结构对应 🔍\n\n"
    "在真实 vLLM 里,这些张量由 **调度器输出** 驱动生成。`vllm/v1/worker/gpu/model_runner.py` 的\n"
    "`execute_model()` 调用链:\n\n"
    "```\n"
    "GPUModelRunner.execute_model()\n"
    "  -> _update_states(...)        更新批状态 (哪些请求活跃)\n"
    "  -> prepare_inputs(...)        产出 input_ids / positions / query_start_loc / seq_lens\n"
    "  -> prepare_attn(input_batch)  产出 block_tables / slot_mappings\n"
    "  -> _model_forward(...)        跑模型\n"
    "  -> compute_logits / sample_tokens\n"
    "```\n\n"
    "| 本课张量 | vLLM 产出位置 | 用途 |\n"
    "|---|---|---|\n"
    "| `input_ids` | `prepare_inputs` | 拼好的词元流, 直接进模型 |\n"
    "| `positions` | `prepare_inputs` | RoPE 位置编码输入 |\n"
    "| `slot_mapping` | `prepare_attn` / `_get_slot_mappings` | 注意力 kernel 写 KV 的地址 |\n"
    "| `block_table` | `prepare_attn` | 注意力 kernel 读 KV 的页表 |\n"
    "| `seq_lens` | `prepare_inputs` | 掩码与分块边界 |\n\n"
    "> 📄 一句话:**本课的组装函数就是 vLLM `prepare_inputs` + `prepare_attn` 的教学版缩略。**\n"
    "> 下一课(22)把拼好的词元流喂进一台迷你 GPT,看它在九段前向里怎么流动。"
)

# =====================================================================
# 第 9 节 · App
# =====================================================================
nb.md(
    "## 9. 🖥️ Streamlit 动态演示:变长序列组批组装\n\n"
    "把本课的组装逻辑搬进 Streamlit,拖动滑块改变序列数量与长度范围,实时观察:\n"
    "五个张量的组装结果与 padding 浪费。\n\n"
    "### 📜 App 完整源码(`app_21_metadata.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_21_metadata.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,拖动序列数与长度范围,观察五个张量的变化。"
)

wrapup(
    nb,
    summary=[
        "问题:一个调度批里序列长度各异,GPU 需要一维词元流 + 恢复现场的元数据",
        "五个张量:input_ids(拼好的流) / positions(绝对位置) / slot_mapping(KV槽位) / block_table(块映射) / seq_lens(长度)",
        "组装 = 拼接 + 记录索引;核心公式:slot = 物理块号×block_size + 块内偏移",
        "三条硬性断言:槽位唯一、位置连续、块内落位正确,违反任一条 kernel 就错",
        "padding 浪费 = 1 - T/(S×max_len),批越大越夸张,真实 GPU 上等于白烧毫秒",
        "vLLM 对应:prepare_inputs 产出词元类张量,prepare_attn 产出 KV 类张量",
    ],
    practice=[
        "把 block_size 从 4 改成 16,重跑组装,观察 slot_mapping 与 block_table 怎么变",
        "加入一条 0 长度的空序列,验证组装函数是否还能正确处理(边界情况)",
        "把 slot_mapping 换成「跨层」版本(每层一个槽位偏移),模拟多层的真实场景",
        "用第 5 节公式,画 padding 浪费 vs 序列数(1~256)的曲线,找到浪费的拐点",
    ],
    links=[
        ("vLLM ModelRunner 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM V1 设计文档", "https://docs.vllm.ai/en/latest/design/v1/v1_usage.html"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch04\21_metadata_assembly.ipynb")