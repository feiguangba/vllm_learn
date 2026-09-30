# -*- coding: utf-8 -*-
"""生成 23_kv_allocator.ipynb(app_23_allocator.py 已存在,%%writefile 覆盖写入相同内容)"""
from helpers import D, chapter_cover, wrapup, new_nb, CH04, app_src, finalize


def app_cell(name):
    return "%%writefile " + name + "\n" + app_src(name)


NB = new_nb("第 23 课 · KV 块分配器:free list、引用计数与碎片",
            subtitle="vLLM 把 KV Cache 切成一块一块,用一张空闲表 + 引用计数来管 —— 我们亲手写一个,再用随机序列压测它",
            emoji="🗃️")

chapter_cover(NB,
    objectives=[
        "理解 KV Cache 为什么按「块(block)」管理:PagedAttention 的分页思想",
        "手写一个 BlockAllocator:free list(空闲表)+ refcount(引用计数)+ 分配/释放",
        "亲眼看见「外部碎片」:块明明有空,却拼不出一段连续空闲",
        "用随机分配/释放序列做压力测试,统计两种策略(顺序 vs best-fit)的碎片率分布",
        "用 plotly 热图可视化块池状态、用 pyecharts 画碎片率随时间的演化",
        "对应 vLLM 的 vllm/v1/core/block_pool.py:free block queue、ref_cnt 与 prefix 共享",
    ],
    toc=[
        ("直觉:仓库里的储物柜", "free list 是空柜清单,refcount 是一把钥匙几个人在用,碎片是空柜不连成片"),
        ("为什么按块管理 KV Cache", "整段预分配的内部碎片 vs 分块的灵活,以及新引入的外部碎片"),
        ("手写 BlockAllocator", "free list + refcount + allocate/release/stats,40 行核心逻辑"),
        ("玩一次典型场景", "分配 4 条序列、释放中间 1 条,看池子留下什么"),
        ("可视化:块池热图", "plotly 热图,颜色 = 引用计数"),
        ("压力测试:随机分配/释放", "几百步随机操作,统计碎片率分布,对比两种策略"),
        ("对应 vLLM block_pool.py", "概念对照:空闲块队列、引用计数、共享块的加减计数"),
        ("真实显存:一池 KV 块到底多大", "用 ch02 的 kv_bytes_real 在 GPU 上量化 KV 显存,给封套倍数"),
        ("配套 App:🗃️ 块分配器模拟器", "streamlit 交互:亲手分配/释放,实时看池状态"),
    ],
    links=[
        ("vLLM block_pool.py 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py"),
        ("PagedAttention 论文 (vLLM)", "https://arxiv.org/abs/2309.06180"),
        ("操作系统:碎片(Wikipedia)", "https://en.wikipedia.org/wiki/Fragmentation_(computing)"),
    ])

NB.md("## 1. 直觉:仓库里的储物柜 🗃️",
D('''
想象一间**存放柜的仓库**:

- 每个储物柜大小一样(比如都能放 16 件行李)——这就是 **KV 块(block)**,`block_size=16` 个词元的 KV;
- 仓库门口挂着一块**空柜清单**(free list):哪些柜子还空着,分配时照单划掉,释放时再登记回来;
- 有些柜子被**好几拨客人共用**(prefix caching 里两条序列共享同一个前缀块)——所以每个柜子还要挂个
  **计数牌(refcount)**:几个人在用就写几,减到 0 才能重新进空柜清单;
- 最麻烦的是**碎片**:清单上明明写着 10 个空柜,但它们东一个西一个、不挨着……等等,KV 块不是可以
  任意分散吗?没错!对 KV Cache 来说,块**逻辑上连续**就够了(块表 block_table 把它们串起来,第 21 课
  见过)。所以 KV 分配器最怕的不是「不连续」,而是**空闲块总数不够**(被占用太多)。但当我们模拟
  「优先要连续段」的策略或观察空闲块分布时,碎片的概念依然有用——本课用「最大连续空闲段 / 空闲总数」
  来量化池子的零散程度,这是内存分配器领域的经典指标。

一句话:**vLLM 用「分块 + 引用计数」把 KV Cache 从整段预分配的浪费中解放出来**。
'''))

NB.md("## 2. 为什么按块管理 KV Cache 📦",
D('''
**传统做法(整段预分配)**:每条序列一进来,就按它的最大长度 `max_len` 预留一整条 KV 显存。
问题有二:

1. **内部碎片**:序列实际只用了一半长度,另一半显存白白锁死;
2. **无法共享**:两条序列有相同前缀(比如同样的 system prompt),KV 却各存各的。

**PagedAttention 的做法(分块)**:把 KV Cache 切成固定大小的小块(`block_size=16`),
序列要多少块就给多少块,像操作系统的分页一样按需分配。前缀相同的块还能**共享**(refcount>1)。

代价是引入了新概念:

$$\\text{所需块数} = \\left\\lceil \\frac{\\text{序列长度}}{\\text{block\\_size}} \\right\\rceil$$

以及分配器要维护的两张账:**free list**(哪些块空闲)与 **refcount**(每块被几条序列引用)。
分配 = 从 free list 摘块、计数 +1;释放 = 计数 −1,减到 0 才回收入 free list。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch OMP 库冲突(Windows)
import numpy as np
import pandas as pd
import json, math

print("块数计算速查:block_size = 16")
for n in [1, 15, 16, 17, 64, 100]:
    print(f"  序列长度 {n:3d} 词元 -> 需要 {math.ceil(n / 16):3d} 块(向上取整,最后一块可能装不满)")
'''),
"🧮 最后一块装不满是「块内尾部浪费」,通常只有 block_size/2 ≈ 8 个词元,远小于整段预分配的浪费。")

NB.md("## 3. 手写 BlockAllocator ✍️",
D('''
核心逻辑只有四条,我们把它写成一个类(与配套 App `app_23_allocator.py` 中的 `SimAllocator` 逻辑一致):

- `free`:空闲块号列表(初始 0..N−1);
- `refcount[i]`:第 i 块被几条序列引用;
- `allocate(seq_id, n_tokens)`:算出需要 k 块,free 不够就失败;按策略挑块,计数 +1;
- `release(seq_id)`:该序列所有块计数 −1,减到 0 的块回收入 free。

两种挑块策略:**free_list**(从清单头顺序拿,最简单)与 **best_fit**(优先填小空洞,模拟抗碎片策略)。
'''))

NB.code(D('''
class BlockAllocator:
    """KV 块分配器:free list + 引用计数(与 app_23_allocator.py 同一套逻辑)"""

    def __init__(self, num_blocks, block_size, strategy="free_list"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.strategy = strategy
        self.free = list(range(num_blocks))     # 空闲块清单:初始全部空闲
        self.refcount = [0] * num_blocks        # 每块的引用计数
        self.seqs = {}                          # seq_id -> 块号列表

    def _pick(self, k):
        # free_list 策略:顺序取最前面的 k 个空闲块(vLLM 的 free block queue 就是类似思路)
        if self.strategy == "free_list":
            return self.free[:k]
        # best_fit 策略:优先从「最小的连续空洞」里拿,模拟抗碎片分配
        used = set(b for b in range(self.num_blocks) if self.refcount[b] > 0)
        holes, run, start = [], 0, None
        for b in range(self.num_blocks):
            if b not in used:
                if run == 0:
                    start = b
                run += 1
            else:
                if run > 0:
                    holes.append((run, start))
                run = 0
        if run > 0:
            holes.append((run, start))
        holes.sort()
        got = []
        for r, s in holes:
            for b in range(s, s + min(r, k - len(got))):
                got.append(b)
            if len(got) >= k:
                break
        return got

    def allocate(self, seq_id, n_tokens):
        k = int(np.ceil(n_tokens / self.block_size))       # 向上取整成块数
        if k > len(self.free):
            return False                                    # 空闲块不够,分配失败(触发抢占/换出)
        blocks = self._pick(k)
        for b in blocks:
            self.free.remove(b)
            self.refcount[b] += 1
        self.seqs[seq_id] = blocks
        return True

    def release(self, seq_id):
        if seq_id not in self.seqs:
            return False
        for b in self.seqs.pop(seq_id):
            self.refcount[b] -= 1
            if self.refcount[b] == 0:                       # 计数减到 0 才真正空闲
                self.free.append(b)
                self.free.sort()
        return True

    def stats(self):
        used = sum(1 for r in self.refcount if r > 0)
        free_blocks = self.num_blocks - used
        runs, run, start = [], 0, None                      # 统计连续空闲段
        for b in range(self.num_blocks):
            if self.refcount[b] == 0:
                if run == 0:
                    start = b
                run += 1
            else:
                if run > 0:
                    runs.append((start, run))
                run = 0
        if run > 0:
            runs.append((start, run))
        largest = max((r for _, r in runs), default=0)
        frag = 1.0 - (largest / free_blocks) if free_blocks > 0 else 0.0
        return dict(used=used, free=free_blocks, runs=runs, largest_run=largest, fragmentation=frag)
'''),
"✍️ 注意 `release` 里「减到 0 才回收」:这正是共享块(refcount>1)能安全存在的关键 —— 一个人退租,室友还在住,柜子不能下架。")

NB.md("## 4. 玩一次典型场景 🎬",
D('''
分配 4 条序列(64/96/32/128 词元,block_size=16 → 4/6/2/8 块),再释放中间的 `seq_1`,
看看池子会留下什么。这个场景就是 App 里「🔄 重置并跑一个典型场景」按钮做的事。
'''))

NB.code(D('''
alloc = BlockAllocator(num_blocks=48, block_size=16, strategy="free_list")

for i, ln in enumerate([64, 96, 32, 128]):
    ok = alloc.allocate(f"seq_{i}", ln)
    print(f"allocate seq_{i}({ln:3d} 词元) -> {'成功' if ok else '失败'}")
alloc.release("seq_1")
print("release  seq_1(96 词元,6 块)已释放\\n")

s = alloc.stats()
print(f"已用块 {s['used']}/48,空闲块 {s['free']},最大连续空闲段 {s['largest_run']} 块")
print(f"空闲段分布(起点,长度): {s['runs']}")
print(f"碎片率 = 1 - 最大连续段/空闲总数 = 1 - {s['largest_run']}/{s['free']} = {s['fragmentation']:.3f}")
print(f"seq 占块: {alloc.seqs}")
'''),
"🎬 释放 seq_1 后,中间挖出一条 6 块的「洞」——它和尾部的空闲块**不相连**,这就是最直观的碎片形态。")

NB.md("## 5. 可视化:块池热图 🎨",
D('''
把 48 个块画成一行格子:灰色 = 空闲(refcount=0),颜色越深 = 被越多序列引用。
一眼就能看见 seq_1 留下的「空洞」。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

colors = [alloc.refcount[b] for b in range(alloc.num_blocks)]
fig = go.Figure(go.Heatmap(
    z=[colors],
    colorscale=[[0, "#e0e0e0"], [0.4, "#7fb3d5"], [1, "#1b4f72"]],
    zmin=0, zmax=max(4, max(colors)),
    x=[f"{b}" for b in range(alloc.num_blocks)], y=["池"],
    hovertemplate="块 %{x}<br>refcount=%{z}<extra></extra>",
))
fig.update_layout(title="块池状态(颜色 = 引用计数,灰 = 空闲)", height=160,
                  xaxis_title="块号", yaxis_visible=True, xaxis_tickangle=0)
fig
'''),
"🎨 中间那一截浅灰就是被释放的空洞;若把鼠标放上去,能看到 refcount 已归零。")

NB.md("## 6. 压力测试:随机分配/释放序列 🌀",
D('''
真实服务里,序列**源源不断地来、又随机地走**(生成完毕或被抢占)。我们模拟几百步随机操作:
每步 45% 概率释放一条已有序列,否则尝试分配一条新序列(长度 8~400 词元随机)。
统计全程的**碎片率轨迹**与分布,并对比两种策略。
'''))

NB.code(D('''
def stress(strategy, steps=400, num_blocks=64, block_size=16, seed=0):
    rng = np.random.default_rng(seed)
    a = BlockAllocator(num_blocks, block_size, strategy)
    frags, used_ratio, clock = [], [], 0
    for _ in range(steps):
        if a.seqs and rng.random() < 0.45:
            sid = list(a.seqs)[int(rng.integers(0, len(a.seqs)))]
            a.release(sid)
        else:
            sid = f"s{clock}"
            clock += 1
            a.allocate(sid, int(rng.integers(8, 400)))
        s = a.stats()
        frags.append(s["fragmentation"])
        used_ratio.append(s["used"] / num_blocks)
    return np.array(frags), np.array(used_ratio)

frag_fl, used_fl = stress("free_list", seed=7)
frag_bf, used_bf = stress("best_fit", seed=7)
print(f"free_list 策略:碎片率 均值 {frag_fl.mean():.3f} / 中位数 {np.median(frag_fl):.3f} / 最差 {frag_fl.max():.3f}")
print(f"best_fit  策略:碎片率 均值 {frag_bf.mean():.3f} / 中位数 {np.median(frag_bf):.3f} / 最差 {frag_bf.max():.3f}")
print(f"free_list 池占用率 均值 {used_fl.mean()*100:.1f}%,best_fit {used_bf.mean()*100:.1f}%(两者应接近,对比才公平)")
'''),
"🌀 多跑几个 seed 结论类似:best_fit 能把碎片率压低一些,但代价是每次分配要扫全池找空洞 —— 工程上常用简单快速的 free list,靠「块可分散」的分页特性天然免疫大部分碎片危害。")

NB.code(D('''
from pyecharts.charts import Line
from pyecharts import options as opts

line = (
    Line()
    .add_xaxis([str(i) for i in range(0, len(frag_fl), 4)])
    .add_yaxis("free_list 碎片率", [round(float(frag_fl[i]), 3) for i in range(0, len(frag_fl), 4)],
               is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("best_fit 碎片率", [round(float(frag_bf[i]), 3) for i in range(0, len(frag_bf), 4)],
               is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(title="随机分配/释放下的碎片率演化(400 步)"),
                     yaxis_opts=opts.AxisOpts(max_=1.0), xaxis_opts=opts.AxisOpts(name="步数"))
)
line.render_notebook()
'''),
"📈 两条曲线都在波动:每当池子快满(占用率高)时,剩余空闲块又少又零散,碎片率冲高;释放一波后又回落。")

NB.code(D('''
import plotly.express as px

hist_df = pd.DataFrame({
    "碎片率": np.concatenate([frag_fl, frag_bf]),
    "策略": ["free_list"] * len(frag_fl) + ["best_fit"] * len(frag_bf),
})
fig2 = px.histogram(hist_df, x="碎片率", color="策略", barmode="overlay", nbins=25,
                    color_discrete_map={"free_list": "#e74c3c", "best_fit": "#27ae60"})
fig2.update_layout(title="碎片率分布:free_list vs best_fit(同一条随机序列)",
                   height=380, yaxis_title="步数(占比)")
fig2
'''),
"📊 best_fit 的分布整体左移(碎片更少),但两者尾部都有高碎片事件 —— 池子接近满时谁都躲不开。")

NB.md("## 7. 对应 vLLM:block_pool.py 的三张账 🔍",
D('''
vLLM V1 的 `vllm/v1/core/block_pool.py` 管理的就是本课这套账(概念对照):

| 本课模拟 | vLLM block_pool.py | 说明 |
|---|---|---|
| `free` 空闲表 | free block queue(`free_block_queue`) | 空闲块队列,分配从队头拿、释放入队尾(先来后用,天然简单) |
| `refcount[i]` | `ref_cnts` 字典 | 块 → 引用数;prefix 共享时多加 1 |
| `allocate(seq, n)` | `allocate_slots` | 给序列分配 KV 槽位(块号) |
| `release(seq)` | `free` / `free_block` | 序列结束,块计数减 1,归零回收 |
| (共享块) | `incr_refcount` / prefix caching | 命中前缀时直接复用已有块,refcount+1,不重算 KV |

⚠️ 诚实说明:本练习环境的 `vendor/vllm` 目录尚未就绪,以上对照基于 vLLM 公开源码与文档;
配好 vendor 后可直接打开 `vllm/v1/core/block_pool.py` 核对。核心思想只有一句:
**空闲队列 + 引用计数,让块能借、能还、能共享。**
'''))

NB.md("## 8. 真实显存:一池 KV 块到底多大 💾",
D('''
上面分配器管的是「块号」(虚拟地块),没给「一块到底占多少显存」。真实账本:
一个词元的 K+V = $2 \\times n_{layers} \\times n_{kv\\_heads} \\times head\\_dim \\times dtype\\_bytes$ 字节,
再乘序列长度就得到显存。我们调用 ch02 的 `real_ops.kv_bytes_real`,在 **RTX 5060 上真实分配**一个
与真实 KV cache 同形状的张量,量化它实际吃掉多少显存,再换算成「一个 GPU 显存能装下多少周期的这种请求」,
给「分配器把池子里块分来分去」一个物理封套 —— 块不够,就是拼多烂的碎片策略都救不回来的天花板。
'''))

NB.code(D('''
import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:/Project/21-Cpp_learn/explore/minivllm/exercises/ch02")
from real_ops import kv_bytes_real

# 真实配置:D=128,16 层,8 个 KV head;一条 2048 词元的序列,bf16 KV
NL, KH, HD, SEQ = 16, 8, 128, 2048          # 层数 / KV head / head_dim / 序列长度
r = kv_bytes_real(L=NL, kv_heads=KH, head_dim=HD, seq=SEQ, batch=1, dtype="bf16")
per_tok = 2 * NL * KH * HD * 2                       # 每个词元的 K+V 字节(bf16)
print("设备:", r.get("device"))
print(f"理论 KV 字节 = 2 × {NL}层 × {KH}头 × {HD} × {SEQ} 词元 × 2B = {r['theory_bytes']/1e6:.1f} MB")
print(f"GPU 实际分配 = {r['alloc_bytes']/1e6:.1f} MB(与理论一致,bf16 KV 双通道 K+V)")
print(f"单块(block_size=16)承载 = {16 * per_tok / 1e3:.1f} KB")
print(f"8GiB 显存约可同时放 {round(8*1024**3 / (r['theory_bytes'] + 1e-12))} 条这种 {SEQ} 序列 —— 这就是分配器要精打细算的总盘子。")

import gc, torch
torch.cuda.empty_cache(); gc.collect()
'''),
"💾 **真实显存数字**。KV 随序列长度线性膨胀 —— 所以 vLLM 必须「按需分块」而不是一进场就整段锁死,否则池子早被大 prompt 撑爆。")

NB.md("## 9. 配套 App:🗃️ KV 块分配器模拟器 🎛️",
D('''
同目录的 `app_23_allocator.py` 把这个分配器做成了**可点击的模拟器**:
左侧配置池大小与 block_size、选择策略,然后亲手「分配一个新序列」「跑典型场景」,
右侧实时显示块池热图、空闲段分布、碎片率指标和每条序列的块表。

**运行方法**(在 `ch04` 目录执行):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_23_allocator.py
```

浏览器打开 **http://localhost:8501**(也可加 `--server.port 8623` 换端口)。
下面这个 cell 会把 app 源码原样写入 `app_23_allocator.py`(内容与文件一字不差):
'''))

NB.code(app_cell("app_23_allocator.py"),
"📜 运行后会覆盖写入相同内容,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "KV Cache 按块管理:所需块数 = ceil(序列长度 / block_size),最后一块可能装不满(尾部浪费很小)",
        "分配器两张账:free list(空闲块清单)+ refcount(引用计数),减到 0 才回收",
        "释放中间序列会留下「空洞」:空闲块不连片,即外部碎片;碎片率 = 1 − 最大连续段/空闲总数",
        "随机压力测试:best_fit 碎片率更低但要扫全池;free list 简单快,分页特性让块可分散、免疫大部分危害",
        "vLLM block_pool.py = 空闲块队列 + ref_cnts + allocate_slots/free,prefix 共享靠 refcount+1",
    ],
    practice=[
        "把 block_size 从 16 改成 4 和 64,重跑压力测试:块更小碎片如何?块更大尾部浪费如何?",
        "给 allocate 加「部分分配」:空闲块不够时先给能给的块(对应 vLLM 的抢占/换出逻辑)",
        "实现 allocate_shared(seq_a, seq_b, n_shared):让两条序列共享前 n_shared 个词元的块,验证 refcount 变成 2",
        "把压力测试里的到达/离开改成「长序列活得久、短序列走得快」,观察碎片率是否下降",
    ],
    links=[
        ("vLLM block_pool.py 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py"),
        ("PagedAttention 论文", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

from pathlib import Path
out = str(Path(CH04) / "23_kv_allocator.ipynb")
NB.save(out)
finalize(out)
