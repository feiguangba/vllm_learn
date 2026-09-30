# -*- coding: utf-8 -*-
"""生成 21_metadata_assembly.ipynb(app_21_metadata.py 已存在,仅嵌入源码)"""
from helpers import D, chapter_cover, wrapup, new_nb, CH04, app_src, finalize
from pathlib import Path

NB = new_nb("第 21 课 · 变长序列组批量张量组装",
            subtitle="把长短不一的序列,拼成一串让 GPU 喜欢的张量:input_ids / positions / slot_mapping / block_table",
            emoji="🧩")

chapter_cover(NB,
    objectives=[
        "理解为什么推理批处理里的序列长度各不相同(变长批 / 调度批的由来)",
        "亲手用 torch 组装五个核心张量:input_ids / positions / slot_mapping / block_table / seq_lens",
        "看懂每个张量的形状、语义与它们之间的对应关系(紧凑拼接 + 索引恢复)",
        "对比 vLLM 的紧凑拼接与朴素的 pad-to-max 做法,量化 padding 浪费",
        "在真实 GPU 上给「拼好的词元流」标一个刻度:prefill / decode 各自多少毫秒",
        "跑通配套 Streamlit App,交互式观察组装结果",
    ],
    toc=[
        ("问题:变长序列怎么塞进一个批次", "拼桌餐厅比喻 + seq_lens 的由来"),
        ("五个核心张量逐个拆解", "input_ids / positions / slot_mapping / block_table / seq_lens 的定义与直觉"),
        ("padding 浪费有多大", "对比 vLLM 紧凑拼接 vs 朴素定长 padding,量化浪费比例"),
        ("真实 GPU:拼好的词元流交到 GPU 的刻度", "用 vllm_real 实测 prefill 一次并行 vs decode 逐字"),
        ("与 vLLM 源码结构对应", "SchedulerOutput / ScheduledSequenceGroup 字段对照"),
        ("配套 App:🧩 变长序列组批组装", "Streamlit 交互演示"),
    ],
    links=[
        ("NVIDIA CUDA Graphs 博客", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
        ("PyTorch Tensor 文档", "https://pytorch.org/docs/stable/tensors.html"),
    ])

NB.md("## 1. 问题:变长序列怎么塞进一个批次? 🍜",
D('''
想象你去一家**拼桌餐厅**🍜:一桌客人先到,已经吃到一半,新客人后到,才刚开始点菜。
如果要求所有桌子同时吃完,先到的客人就得干等;如果一桌只能坐固定人数,那每次都得浪费几个空位。
大模型推理的批处理遇到的正是这个问题:**一个批次里,每条序列(相当于一桌客人)的长度不一样**。

在 vLLM 里,调度器(Scheduler)每步挑选若干序列组成一个**调度批(scheduled batch)**,这些序列:

- 有的处于 **prefill** 阶段:一次性要处理一长段输入(比如 2000 个词元);
- 有的处于 **decode** 阶段:每步只产生 1 个新词元;
- 有的正在做 **chunked prefill**:一大段输入被切成小块,分几步处理。

于是**每条序列在本步要处理的词元数各不相同**。朴素的做法是把所有序列 pad 到一样长,
但词表级别的矩阵运算里,pad 的位置也会被白白计算 —— 这就是我们稍后要量化的浪费。
vLLM 的做法是:**紧凑拼接(compaction)** —— 把当前步所有需要处理的词元头尾相连,拼成一维张量
`input_ids`,再用一组**辅助张量**记录'每个词元属于哪个序列、在序列的哪个位置、KV Cache 该放哪'。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch OMP 库冲突(Windows)
import torch
import numpy as np

torch.manual_seed(0)

# 模拟一个调度批:4 条序列,长度各不相同
# 语义上这来自 Scheduler 的输出:每条序列(seq_id)在本步需要处理的词元数
seq_lens = torch.tensor([5, 9, 3, 7])          # 第 i 条序列处理 5/9/3/7 个词元
num_seqs = len(seq_lens)
total_tokens = int(seq_lens.sum())
print(f"设备 = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"序列数 = {num_seqs},总词元数 = {total_tokens}")
print(f"seq_lens = {seq_lens.tolist()}")
print("💡 先看输入:一个 seq_lens 张量就是变长批的「调度指令」。")
'''),
"💡 组装是纯逻辑操作(CPU 也能跑),但它拼出来的张量最终要交到 GPU 前向 —— 第 4 节会标一个真实刻度。")

NB.md("## 2. 五个核心张量逐个拆解 🧩",
D('''
vLLM 的 `ModelRunner.execute_model` 拿到调度输出后,会把它组装成一组 GPU 张量
(在源码里由 `ModelInputForGPUBuilder` 完成)。我们这里只关心五个最重要的:

| 张量 | 形状 | 含义 | 生活比喻 |
|---|---|---|---|
| `input_ids` | `[total_tokens]` | 所有序列本步词元头尾拼接 | 流水线上拼接的零件 |
| `positions` | `[total_tokens]` | 每个词元在其序列中的绝对位置 | 每个零件在自家图纸上的坐标 |
| `slot_mapping` | `[total_tokens]` | 每个词元的 KV Cache 扁平槽位号 | 零件要放的寄存柜号 |
| `block_table` | `[num_seqs, max_blocks]` | 每条序列占用的 KV 块号表(pad 到等长) | 每桌客人的桌号清单 |
| `seq_lens` | `[num_seqs]` | 每条序列本步的词元数 | 每桌的入座人数 |

**直觉速记**:`input_ids` 告诉模型'算哪些词元',`positions` 告诉模型'每个词元在哪',
`slot_mapping` + `block_table` 告诉模型'KV Cache 从哪取、往哪存'。后面三个都是**变长拼接之后
用来'恢复现场'的索引信息** —— 这正是 PagedAttention 的精髓。
'''))

NB.md(D('''
组装函数:核心思想就是'拼接 + 记录索引'。注意 `slot_mapping` 是扁平槽位号,这正是 vLLM 的
KV Cache 张量展平后的索引——`槽位 = 块号 × block_size + 块内偏移`。
'''))

NB.code(D('''
block_size = 4  # 每个 KV 块容纳 4 个词元(和 vLLM 默认思路一致,可配置)

# 每条序列的 KV 块表:块号按顺序分配(物理块池从 0 开始)
block_table = [
    [0, 1],       # 序列 0:5 个词元 -> 需要 ceil(5/4)=2 块
    [2, 3, 4],    # 序列 1:9 个词元 -> 3 块
    [5],          # 序列 2:3 个词元 -> 1 块
    [6, 7],       # 序列 3:7 个词元 -> 2 块
]

def assemble_batch(seq_lens, block_table, block_size, vocab_size=100):
    """把变长序列组装成 vLLM 风格的一组张量(迷你版)。"""
    seq_lens = seq_lens.to(torch.long)
    total = int(seq_lens.sum())

    # 1) input_ids:所有词元头尾拼接(实际是 tokenizer 查表的结果)
    input_ids = torch.randint(0, vocab_size, (total,))

    # 2) positions:每个序列内部从 0 开始编号
    positions = torch.cat([torch.arange(s) for s in seq_lens])

    # 3) 各序列在拼接张量中的起点偏移
    offsets = torch.cumsum(seq_lens, dim=0) - seq_lens

    # 4) slot_mapping:词元 (seq s, 序列内位置 i) 的 KV 槽位号
    #    槽位号 = 所在块号 * block_size + 块内偏移
    slot_mapping = torch.empty(total, dtype=torch.long)
    for s in range(len(seq_lens)):
        base = torch.tensor(block_table[s]) * block_size   # 每块的起始槽位
        for i in range(int(seq_lens[s])):
            block_idx = i // block_size                    # 词元在块表中的第几块
            slot_mapping[offsets[s] + i] = base[block_idx] + (i % block_size)

    return input_ids, positions, slot_mapping, seq_lens, offsets

input_ids, positions, slot_mapping, seq_lens, offsets = assemble_batch(
    seq_lens, block_table, block_size)

print('input_ids    :', input_ids.tolist())
print('positions    :', positions.tolist())
print('slot_mapping :', slot_mapping.tolist())
print('offsets      :', offsets.tolist())
'''),
"🚰 拼出来的一维词元流 `input_ids` 就是下一课(第 22 课)模型前向吃进去的东西;索引张量帮模型'恢复现场'。")

NB.code(D('''
# 一致性验证:每个断言都对应一条 vLLM 的前置条件
assert positions.numel() == total_tokens, 'positions 长度必须等于总词元数'
assert slot_mapping.numel() == total_tokens
assert len(slot_mapping.unique()) == total_tokens, '槽位必须互不重叠(同一词元不能占两个槽)'

# 序列 s 的第 i 个词元,其 position 必等于 i
for s in range(num_seqs):
    seg = positions[offsets[s]: offsets[s] + seq_lens[s]]
    assert torch.equal(seg, torch.arange(seq_lens[s])), f'序列 {s} 的 positions 不连续'

# 槽位号必须落在 KV 块表对应的块内
for s in range(num_seqs):
    slots = slot_mapping[offsets[s]: offsets[s] + seq_lens[s]]
    for i, sl in enumerate(slots):
        blk = block_table[s][i // block_size]
        assert (blk * block_size) <= sl < (blk + 1) * block_size
print('✅ 全部断言通过:张量之间关系自洽(每条断言都对应 vLLM 内部的一个不变量)')
'''),
"✅ 槽位唯一、位置连续、块内落位正确 —— 这分别是 vLLM 的 slot_mapping 唯一性、positions 连续性、block_table 落位三条前置条件。")

NB.md("## 3. padding 浪费有多大? 📊",
D('''
如果我们偷懒,把所有序列 pad 到最长,矩阵形状就变成 `[num_seqs, max_len, H]`。
多余的 pad 位置虽然可以靠 mask 避免影响结果,但**显存和算力照付**。下面的代码直接量化这份浪费。

$$\\text{浪费比例} = 1 - \\frac{\\text{有效词元}(\\sum seq_lens)}{\\text{朴素 padding 词元}(B \\times \\max len)}$$
'''))

NB.code(D('''
max_len = int(seq_lens.max())
padded_tokens = num_seqs * max_len
waste_ratio = 1 - total_tokens / padded_tokens

print(f'紧凑拼接   :处理 {total_tokens} 个词元([total_tokens] 一维流)')
print(f'朴素 padding:处理 {padded_tokens} 个词元(形状 [num_seqs, max_len],白算 {padded_tokens - total_tokens} 个)')
print(f'浪费比例   = {waste_ratio * 100:.1f}%')
print(f'本例还好,真实生产里 decode 序列长度可以相差 100 倍,浪费会超过 90%')
'''),
"📊 算账:批越大、长度差越大,padding 浪费越惊人。这是 vLLM 坚持紧凑拼接的核心理由。")

NB.md("### 实现细节:padding 浪费的静态柱状图 🎨",
D('''
- 灰虚线 = `max_len` 水位线:朴素的 `[B, max_len]` 矩阵里每条序列都得塞满到这个高度;
- 红段 = 白白多算的词元:它们不影响结果(mask 掉),但**显存与算力照付**;
- **边界条件**:所有序列等长 → 浪费 0;长度悬殊 → 浪费趋近 $1 - \\frac{1}{B}$。
'''))

NB.code(D('''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

seqs = [f"seq {i}" for i in range(num_seqs)]
eff = seq_lens.tolist()
pad = [max_len - s for s in eff]

fig, ax = plt.subplots(figsize=(8.4, 4.6))
x = np.arange(num_seqs)
ax.bar(x - 0.2, eff, width=0.4, label="有效词元", color="#3498db", edgecolor="k", lw=0.4)
ax.bar(x + 0.2, pad, width=0.4, label="padding 浪费(朴素做法)", color="#e74c3c", edgecolor="k", lw=0.4)
for xi, e, p in zip(x, eff, pad):
    ax.text(xi - 0.2, e + 0.1, str(e), ha="center", fontsize=9)
    ax.text(xi + 0.2, p + 0.1, str(p), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(seqs)
ax.set_ylabel("词元数")
ax.axhline(max_len, color="0.4", ls="--", lw=1)
ax.text(num_seqs - 0.5, max_len + 0.15, f"max_len = {max_len}", ha="right", fontsize=9, color="0.3")
ax.set_title(f"变长批:有效词元 vs padding 浪费(浪费比例 {waste_ratio * 100:.1f}%)")
ax.legend(); plt.tight_layout(); plt.show()
'''),
"📊 pyecharts/matplotlib 双实现都可;matplotlib 版更适合打印成册。红段 = 若偷懒 pad 会白白多算的词元。")

NB.md(D('''
把组装结果以表格形式核对一遍(逐序列展示拼接区间与槽位),再看 `positions` 热力图
(`positions` = 每条序列**内部从 0 连续编号**,然后头尾拼接 —— 本课核心不变量)。
'''))

NB.code(D('''
# 把组装结果以纯文本表格核对一遍(规避在 GPU 已初始化后构造 DataFrame 的 HTML 渲染崩溃)
rows = []
for s in range(num_seqs):
    rows.append(dict(seq=f"seq {s}", lens=int(seq_lens[s]),
                     span=f"[{int(offsets[s])}, {int(offsets[s] + seq_lens[s])})",
                     positions=positions[offsets[s]: offsets[s] + seq_lens[s]].tolist(),
                     slots=slot_mapping[offsets[s]: offsets[s] + seq_lens[s]].tolist(),
                     block_table=block_table[s]))

fmt = "{:<6} {:<5} {:<14} {:<40} {:<52} {:<12}"
print(fmt.format("序列", "长度", "拼接区间", "positions", "slot_mapping", "block_table"))
for r in rows:
    print(fmt.format(r["seq"], r["lens"], r["span"],
                     str(r["positions"]), str(r["slots"]), str(r["block_table"])))
'''),
"📋 表格核对:序列 1 长 9 词元,占 3 块,槽位 = [8,9,10,11, 12,13,14,15, 16](8 号块起始)。")

NB.code(D('''
import plotly.express as px
import plotly.graph_objects as go

import torch as _t
mat = _t.full((num_seqs, max_len), -1, dtype=_t.long)
for s in range(num_seqs):
    mat[s, :seq_lens[s]] = positions[offsets[s]: offsets[s] + seq_lens[s]]

fig = go.Figure(go.Heatmap(
    z=mat.numpy(),
    x=[f'pos {i}' for i in range(max_len)],
    y=[f'seq {s} (len={seq_lens[s]})' for s in range(num_seqs)],
    colorscale='Viridis',
    hovertemplate='序列 %{y}<br>位置 %{x}<br>position=%{z}<extra></extra>',
))
fig.update_layout(title='positions 热力图:-1 表示该位置无词元(紧凑拼接下不存在)',
                  height=380)
fig
'''),
"🧭 白格在紧凑拼接下不存在(没有 pad)—— 画出来只为和「朴素 padding 矩阵」对比:同样是 `[B, max_len]` 的形状,紧凑版不用填满,不浪费一格算力。")

NB.md("## 4. 真实 GPU:拼好的词元流交到 GPU 的刻度 ⏱️",
D('''
上面我们把变长批拼成了 `input_ids + positions + slot_mapping + block_table`。这套「拼好的词元流」
最终要**交给 GPU 前向**(下一课 ModelRunner 的数据流)。这里用跨章共享库 `vllm_real.bench_prefill_decode`
在 RTX 5060 上标一个真实刻度:**同样一段词元流,一次并行 prefill 还是逐字 decode,GPU 花的时间完全不同。**

- **prefill**:一次把 `T` 个词元一起前向 —— 拼接后的整段 `input_ids` 正是这种「大矩阵一次算完」;
- **decode**:每步只喂 1 个新词元 —— 变长批里 decode 序列每步就贡献 1 个 token。

这正是调度器为什么要把「紧凑拼接的 T」交给 GPU,以及为什么 decode 每步是 memory-bound:
`decode / prefill` 的真实耗时比,几倍到几十倍甚至更高。
'''))

NB.code(D('''
import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_prefill_decode, cuda_info

b = bench_prefill_decode(d=256, layers=8, L=256, steps=64, reps=7)
print("设备:", cuda_info(), "· 参数", f"{b['params']/1e6:.1f}M", "· L=", b["L"])
print(f"prefill(一次并行 {b['L']} 词元流) : {b['prefill_ms']:.3f} ms  → {b['prefill_tok_per_s']/1e3:.0f} k tok/s")
print(f"decode(每步 1 个新词元)         : 每步 {b['decode_step_ms']:.3f} ms")
print(f"耗时比 decode/prefill(总量相同时) : {b['ratio']:.0f} 倍")
print("→ 紧凑拼接保住的是 prefill 的并行度;被拼接时白算的 pad 词元,在这里就是白烧的 GPU 时间。")

import gc, torch
torch.cuda.empty_cache(); gc.collect()
print("\\n真实 GPU 刻度已就位。拼词元是逻辑活,但拼多少、怎么拼,直接决定下面这个数字。")
'''),
"🚀 **真实 GPU 数字**。把第 3 节的「padding 浪费比例」乘上这里「真实 prefill 每秒能处理多少词元」,就是 vLLM 坚持紧凑拼接省下的白烧算力 —— 逻辑课从此有了物理刻度。")

NB.md("## 5. 与 vLLM 源码结构对应 🔍",
D('''
在真实 vLLM 里,这些张量由 **调度器输出** 驱动生成。vLLM 的 `vllm/v1/core/sched/output.py`
定义了 `SchedulerOutput`,其核心字段包括:

- `scheduled_seq_groups`:本轮要执行的序列组(每条含序列、token 块表、本轮要处理的词元区间);
- `num_scheduled_tokens`:本轮总共要处理的词元数(即我们的 `total_tokens`);
- `preempted_seq_ids` / `finished_seq_ids`:被抢占、已完成的序列。

其中序列组对应的结构 `ScheduledSequenceGroup` 携带 `token_chunk_size`(本轮处理多少词元)与
`block_table`(物理块表)等字段 —— 我们迷你版里的 `seq_lens` 与 `block_table` 就是从这些字段来的。

⚠️ 诚实说明:本练习环境的 `vendor/vllm` 目录尚未就绪,以上是基于 vLLM 公开源码的讲解;
等你配好 vendor 目录后,可以直接 `grep` 对应文件核对字段名。

`gpu_model_runner.py` 的 `execute_model` 拿到调度输出后,调用输入构建器把它们**同步组装**成上面
五个张量,再交给模型前向 —— 这正是我们下一课(第 22 课)要跟踪的数据流起点。
'''))

NB.md("## 6. 配套 App:🧩 变长序列组批组装 🎛️",
D('''
把本课的组装逻辑搬进 Streamlit,你可以拖动滑块改变序列数量与长度范围,实时观察:

- 五个张量的组装结果(热力图 + 明细表);
- 总词元数、需要的 KV 块数与 padding 浪费比例;
- 每个序列在拼接张量中的区间与槽位。

**运行方法**:在 `ch04` 目录执行:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_21_metadata.py
```

然后在浏览器打开 http://localhost:8501 。完整源码如下(与同目录 `app_21_metadata.py` 一字不差):
'''))

NB.code("%%writefile app_21_metadata.py\n" + app_src("app_21_metadata.py"),
"🧩 配套 App 完整源码(app_21_metadata.py):本 cell 只把源码写入文件(不执行),notebook 与 app 共享同一套组装规则。")

wrapup(NB,
    summary=[
        "变长批的根源:同一步内 prefill / decode / chunked prefill 序列的长度不同",
        "五个核心张量:input_ids(拼接词元)、positions(绝对位置)、slot_mapping(槽位)、block_table(块表)、seq_lens(长度)",
        "组装 = 紧凑拼接 + 索引恢复,slot_mapping 由 block_table 与块内偏移共同决定",
        "朴素 padding 的浪费 = 1 − total / (num_seqs × max_len),真实场景常超 90%",
        "真实 GPU 刻度:紧凑的词元流一次并行 prefill 远比逐字 decode 划算,白算的 pad 全是白烧的算力",
        "对应 vLLM 源码:SchedulerOutput / ScheduledSequenceGroup 是组装的数据源头,ModelRunner 负责这一步",
    ],
    practice=[
        "把 block_size 改成 2 和 16,重新运行组装函数,观察 slot_mapping 的变化规律",
        "给 assemble_batch 增加一个 max_seq_len 参数,支持部分序列被截断(chunked prefill 场景)",
        "统计不同长度分布(均匀/极差)下 padding 浪费率,画一张浪费率 vs 长度方差的关系图",
        "把第 4 节的 bench_prefill_decode 的 L 从 256 改到 2048,看 prefill 一次并行到底多划算(吞吐随 L 变化)",
        "在 App 里把序列数调到 8、长度范围拉到最大,对比浪费比例的变化",
    ],
    links=[
        ("PyTorch Tensor 索引文档", "https://pytorch.org/docs/stable/tensor_indexing.html"),
        ("vLLM 调度文档", "https://docs.vllm.ai/en/latest/design/scheduler.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

out = str(Path(CH04) / "21_metadata_assembly.ipynb")
NB.save(out)
finalize(out)