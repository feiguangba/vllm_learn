# -*- coding: utf-8 -*-
"""生成 96_npu_memory_kv.ipynb 与 app_96_npu_memory.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_96 = D('''
# -*- coding: utf-8 -*-
# app_96_npu_memory.py — 昇腾 NPU 内存与 KV Cache 计算器 💾
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="NPU 内存与 KV Cache 💾", layout="wide")
st.title("💾 第 96 课 · 昇腾 NPU 内存与 KV Cache:显存计算器")

st.markdown("""
KV Cache 就像推理时的「**半成品库存**」:每个请求的键值对都要存下来,供后续生成使用。
它的大小 = `2 × 层数 × kv_heads × head_dim × batch × seq × dtype字节`。
拖一拖下面的参数,实时算出 KV Cache 要吃掉多少显存、在目标设备上占多大比例。
""")

st.sidebar.header("🎛️ 参数")
n_layers = st.sidebar.slider("层数", 8, 96, 32, 1)
n_heads = st.sidebar.slider("KV 头数(kv_heads, GQA)", 1, 32, 8, 1)
head_dim = st.sidebar.slider("head_dim", 32, 256, 128, 16)
batch = st.sidebar.slider("batch(并发请求数)", 1, 128, 16, 1)
seq = st.sidebar.slider("序列长度", 512, 32768, 8192, 512)
dtype = st.sidebar.radio("KV 存储精度", ["fp16/bf16(2B)", "fp8(1B)", "int8(1B)"])
dev = st.sidebar.selectbox("目标设备", ["昇腾 910B(64GB)", "RTX 5060(8GB)", "昇腾 310P(8GB)"])
st.sidebar.caption("KV 精度从 fp16 降到 fp8/int8,显存直接减半 —— 这是推理优化的头号杠杆。")

dbytes = 2 if "fp16" in dtype else 1
kv_gb = 2 * n_layers * n_heads * head_dim * batch * seq * dbytes / 1e9
mem_map = {"昇腾 910B(64GB)": 64, "RTX 5060(8GB)": 8, "昇腾 310P(8GB)": 8}
mem = mem_map[dev]
frac = kv_gb / mem * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("KV Cache 占用", f"{kv_gb:.2f} GB")
c2.metric("设备显存", f"{mem} GB")
c3.metric("显存占比", f"{frac:.1f}%")
c4.metric("每 token KV 量", f"{kv_gb / max(seq*batch, 1):.4f} GB")

st.subheader("📈 KV 占用 vs 序列长度(三种并发)")
seqs = np.arange(512, 32769, 512)
fig = go.Figure()
for b in [1, 16, 64]:
    y = 2 * n_layers * n_heads * head_dim * b * seqs * dbytes / 1e9
    fig.add_trace(go.Scatter(x=seqs, y=y, mode="lines", name=f"batch={b}",
                             line=dict(width=3)))
fig.add_hline(y=mem, line_dash="dash", line_color="#c0392b")
fig.update_layout(title=f"KV Cache 随序列长度线性增长({dev},红线=显存上限)",
                  xaxis_title="序列长度", yaxis_title="KV 占用(GB)",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🥧 推理显存账单(示意分配)")
pieces = ["KV Cache", "模型权重", "计算图/工作区", "碎片与预留"]
vals = [kv_gb, mem * 0.35, mem * 0.1, max(mem * 0.05, 0.1)]
vals[0] = min(vals[0], mem * 0.9)
fig2 = go.Figure(go.Pie(labels=pieces, values=vals, hole=0.45,
                        marker_colors=["#e74c3c", "#2e86c1", "#27ae60", "#95a5a6"]))
fig2.update_layout(title="推理显存构成(示意)", height=340,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:KV Cache 是「线性增长 + 按并发翻倍」的显存大头。三个杠杆最有效:
> ① 精度降到 fp8/int8(省一半);② GQA 减 KV 头数;③ PagedAttention 动态按需分配(不预占)。
""")
''')

NB = new_nb("第 96 课 · 昇腾 NPU 内存管理与 KV Cache",
            subtitle="内存层次 · 静态规划与内存复用 · KV Cache 公式与优化杠杆",
            emoji="💾")

chapter_cover(NB,
    objectives=[
        "掌握昇腾 NPU 的内存层次:HBM/DDR、L2、L1、UB,以及统一内存模型",
        "理解静态内存规划与内存复用:如何用『复用』把小池子装下大模型",
        "背熟 KV Cache 计算公式,会算任意配置下的显存占用",
        "对比 GPU 与昇腾 NPU 在 KV Cache 上的容量与策略差异",
        "掌握优化 KV Cache 的三大杠杆:精度、GQA、PagedAttention",
        "用 torch/numpy 实现 KV 计算器,配 App 交互调参",
    ],
    toc=[
        ("直觉:仓库与半成品库存", "KV Cache = 推理时的『半成品库存』"),
        ("NPU 内存层次", "HBM → L2 → L1 → UB,以及统一内存"),
        ("内存复用:小池子装大模型", "静态规划 + buffer reuse + 内存池"),
        ("KV Cache 公式(动手)", "手写计算器,算出真实数字"),
        ("GPU vs 昇腾对比", "同一公式,不同设备容量与策略"),
        ("三大优化杠杆", "精度 / GQA / PagedAttention"),
        ("配套 App", "app_96_npu_memory.py:显存计算器"),
    ],
    links=[
        ("昇腾 CANN 内存管理文档", "https://www.hiascend.com/document"),
        ("vLLM-Ascend(支持 PagedAttention)", "https://github.com/vllm-project/vllm-ascend"),
        ("vLLM KV Cache 文档(回顾)", "https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html"),
        ("GQA 论文", "https://arxiv.org/abs/2305.13245"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:仓库与半成品库存 📦",
D('''
想象一个**中央厨房**(NPU)同时给很多桌(请求)做菜:

- 每桌的点单内容(历史 token 的 Key/Value)要**留档**,方便后面"接着上菜" —— 这就是 **KV Cache**;
- 留档不能乱扔,得放在仓库(显存)里;仓库满了,就得上不了新单、甚至让服务 OOM;
- 仓库管理得讲究:常用料(权重)放货架(片上缓存),留档(KV)放冷库(HBM),还要学会"复用货位"。

KV Cache 的尺寸不是玄学,它有一个**精确的公式**。学会算它,你就能回答面试官最爱的
问题:"这个模型到底能塞下多长的上下文?" 本课先看内存长什么样,再动手算账。
'''))

NB.md("## 2️⃣ NPU 内存层次:冷库、货架与操作台 🏗️",
D('''
昇腾 NPU 的内存是一个**金字塔**:

- **HBM / DDR**:整块"大仓库",权重、KV Cache、中间结果都在这里,容量大但慢;
- **L2 / L1 缓存**:离 AI Core 更近的"货架",缓存热点数据;
- **UB(统一缓冲)**:AI Core 的"操作台",计算前数据必须搬到这里(92 课讲过);
- **统一内存(Unified Memory)**:Host 与 Device 共享地址空间,减少搬运(昇腾特色之一,
  对应 CUDA 的 Unified Memory 概念)。

一句话:**数据从冷库一层层搬到操作台,越靠近计算越快也越小**。推理优化的本质就是
"让数据尽量少跑远路"。画成金字塔:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.axis("off")
levels = [
    ("UB 统一缓冲", "操作台:计算前必经", 1.2, "#f8c471", 6.0),
    ("L1 / L2 缓存", "货架:热点数据", 1.8, "#f9e79f", 5.0),
    ("HBM / DDR", "冷库:权重 · KV Cache · 中间结果", 2.6, "#aed6f1", 4.0),
]
y = 0.4
for name, desc, h, c, w in levels:
    ax.add_patch(plt.Rectangle((3.2 - w/2, y), w, h, facecolor=c, edgecolor="#555", lw=1.8))
    ax.text(3.2, y + h/2, name, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(3.2, y + h*0.25, desc, ha="center", va="center", fontsize=8, color="#333")
    y += h - 0.25
ax.annotate("", xy=(0.4, 1.2), xytext=(0.4, 3.0),
            arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.text(0.3, 2.1, "更快\\n更小\\n←", ha="center", fontsize=9, rotation=90)
ax.text(3.2, 4.7, "NPU 内存金字塔:从冷库到操作台", fontsize=13, fontweight="bold", ha="center")
ax.set_xlim(0, 6.4); ax.set_ylim(0, 5.2)
plt.tight_layout()
'''),
"🎨 内存金字塔:KB 级 UB / MB 级缓存 / GB 级 HBM。KV Cache 住冷库(HBM),优化目标是让它『少占、快取』。")

NB.md("## 3️⃣ 内存复用:小池子装大模型 🧰",
D('''
显存永远不够用,于是昇腾像所有 AI 栈一样玩起了**内存复用**:

1. **静态内存规划**:编译/构图阶段就把每个张量的生命周期算好,按"时间段"分配,而非按"张量"分配;
2. **Buffer 复用**:生命周期不重叠的张量共用同一块内存(如某层用完的中间结果,下一层立刻复用其位置);
3. **内存池(Memory Pool)**:按大小分桶缓存,减少频繁 malloc/free 的开销与碎片。

复用能把"逻辑上需要 20GB"压到"物理上只占 12GB"。下面模拟一个简单场景:三个
生命周期不同的中间张量,用时间线展示它们如何共用同一块内存:
'''))

NB.code(D('''
# 模拟:张量 a、b、c 的生命周期,a 用完后 c 才登场 → 可复用同一块内存
life = [("a", 0, 4), ("b", 2, 6), ("c", 4, 8)]   # (名字, 开始, 结束)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(1, 10, left=0, height=0.35, color="#c0392b", alpha=0.35)
ax.text(5, 1, "朴素:独立分配 → 共 10GB(3 段)", ha="center", va="center", fontsize=9, color="#c0392b")
used = [("a", 0, 4, "#2e86c1"), ("b", 2, 6, "#27ae60"), ("c", 4, 8, "#e67e22")]
for name, s, e, c in used:
    ax.barh(2, e - s, left=s, height=0.35, color=c, alpha=0.85)
    ax.text((s+e)/2, 2, name, ha="center", va="center", fontsize=9, color="white")
ax.text(4.05, 2.4, "高峰并发占用 = 4+3 = 7GB → 省 30%", fontsize=8, color="#333")
ax.set_yticks([1, 2]); ax.set_yticklabels(["朴素", "复用"])
ax.set_xlabel("时间 →(GB 生命周期)")
ax.set_title("内存复用:让生命周期错开的张量共用同一块内存")
ax.set_xlim(0, 10.5); plt.tight_layout()
print("朴素方案:10GB;复用方案:高峰 7GB → 省 30%。")
'''),
"🧰 复用原理:a 用完后 b 才登场、b 用完前 c 登场 —— 只要『生命周期不重叠』,就能共用货位。昇腾的静态规划自动完成这件事。")

NB.md("## 4️⃣ KV Cache 公式:一口算出显存 📐",
D('''
KV Cache 的大小有一个**精确公式**(回顾第 8 章):

$$\\text{KV\\_bytes} = 2 \\times n_{\\text{layers}} \\times n_{\\text{kv\\_heads}} \\times d_{\\text{head}} \\times B \\times S \\times \\text{dtype\\_bytes}$$

- 系数 `2`:每个 token 的 Key 和 Value 各一份;
- `n_layers × n_kv_heads × d_head`:单个 token 的 KV 维度;
- `B × S`:batch × 序列长度(所有请求的所有位置);
- `dtype_bytes`:fp16/bf16=2,fp8/int8=1。

下面把它写成函数,算几个真实场景:
'''))

NB.code(D('''
def kv_cache_gb(n_layers, n_kv_heads, head_dim, batch, seq, dtype_bytes=2):
    """KV Cache 占用(GB)。公式:2 × layers × kv_heads × head_dim × batch × seq × dtype_bytes"""
    return 2 * n_layers * n_kv_heads * head_dim * batch * seq * dtype_bytes / 1e9

print("=== 场景 1:7B 级模型(batch=16, seq=8192, fp16)===")
g1 = kv_cache_gb(n_layers=32, n_kv_heads=8, head_dim=128, batch=16, seq=8192)
print(f"  KV Cache = {g1:.2f} GB")

print("=== 场景 2:同样配置但 KV 精度降到 fp8 ===")
g2 = kv_cache_gb(32, 8, 128, 16, 8192, dtype_bytes=1)
print(f"  KV Cache = {g2:.2f} GB(省 {100*(1-g2/g1):.0f}%)")

print("=== 场景 3:同场景但 batch 拉到 64 ===")
g3 = kv_cache_gb(32, 8, 128, 64, 8192)
print(f"  KV Cache = {g3:.2f} GB(batch 翻 4 倍,显存翻 4 倍)")

fig, ax = plt.subplots(figsize=(7, 3.8))
labels = ["fp16\\n(batch16)", "fp8\\n(batch16)", "fp16\\n(batch64)"]
vals = [g1, g2, g3]
bars = ax.bar(labels, vals, color=["#c0392b", "#27ae60", "#e67e22"], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f} GB", ha="center", fontsize=10)
ax.set_ylabel("KV Cache 占用 (GB)")
ax.set_title("KV Cache 的三个场景对比")
plt.tight_layout()
'''),
"📐 看数字:KV Cache 与 batch、seq 线性相乘 —— 并发翻倍显存翻倍;精度减半显存减半。这就是为什么『量化 KV』是最热门优化。")

NB.md("## 5️⃣ GPU vs 昇腾:同一公式,不同答卷 ⚖️",
D('''
公式在 GPU 和昇腾上完全一样,差异在**容量与策略**:

| 项目 | RTX 5060(8GB) | 昇腾 910B(64GB) |
|---|---|---|
| HBM 容量 | 8 GB | 64 GB(8 倍) |
| 典型模型 | 7B(fp16 权重 14GB 装不下 → 需量化) | 7B fp16 权重 + 大 KV 都能装 |
| KV Cache 优化 | PagedAttention 按需分配 | PagedAttention(vLLM-Ascend)同样支持 |
| 内存管理 | CUDA 显存池 | CANN 静态规划 + 统一内存 |

用上面的计算器模拟:同样"32 层 / 8 KV 头 / 128 维 / seq=8192",在 8GB 和 64GB 设备上,
KV 能撑到多高的并发?
'''))

NB.code(D('''
mem_5060, mem_910b = 8, 64            # GB
for dev, mem in [("RTX 5060(8GB)", mem_5060), ("昇腾 910B(64GB)", mem_910b)]:
    b = 0
    while kv_cache_gb(32, 8, 128, b + 1, 8192) < mem * 0.8:
        b += 1
        if b > 2000:
            break
    print(f"{dev}:seq=8192 时 KV 最多支撑并发 ≈ {b} (KV ≈ {kv_cache_gb(32,8,128,b,8192):.1f}GB)")

seqs = np.arange(1024, 32769, 1024)
fig, ax = plt.subplots(figsize=(7.5, 4))
for mem, label, c in [(8, "RTX 5060(8GB)", "#e74c3c"), (64, "昇腾 910B(64GB)", "#2e86c1")]:
    y = [mem * 0.8 / (2*32*8*128*s*2/1e9) for s in seqs]     # 80% 显存能撑的并发
    ax.plot(seqs, np.clip(y, 0, 500), "-", lw=2.2, color=c, label=label)
ax.set_xlabel("序列长度"); ax.set_ylabel("KV 能支撑的最大并发(示意)")
ax.set_title("同一公式下:显存越大,能撑的『并发 × 长度』越大")
ax.legend(); plt.tight_layout()
'''),
"⚖️ 对比结论:公式不变,容量变 —— 昇腾 910B 的 64GB HBM 让『长上下文 × 高并发』成为可能,这正是大模型推理选大显存的原因。")

NB.md("## 6️⃣ 三大优化杠杆:让 KV 更省 🔧",
D('''
显存不够时,从三个方向下手(按性价比排序):

1. **KV 精度量化**:fp16 → fp8/int8,显存直接减半,精度损失通常在可接受范围
   (昇腾/MindIE 支持 KV 量化);
2. **GQA(Grouped-Query Attention)**:多组 query 头共享一组 KV 头,KV 头数从 32 降到 8
   —— 显存省 4 倍(回顾第 21 课);
3. **PagedAttention 动态分配**:不再按最大长度预占,而是像操作系统分页一样按需分配,
   解决了"预分配浪费"与"碎片化"(vLLM-Ascend 已支持)。

画一张"三个杠杆叠加"的柱状图,直观看到它们如何逐级压缩 KV 占用:
'''))

NB.code(D('''
base = kv_cache_gb(n_layers=32, n_kv_heads=32, head_dim=128, batch=16, seq=8192)
l1 = kv_cache_gb(32, 32, 128, 16, 8192, dtype_bytes=1)        # fp8
l2 = kv_cache_gb(32, 8, 128, 16, 8192, dtype_bytes=1)         # + GQA 32→8
l3 = l2 * 0.7                                                  # + PagedAttention 免预分配(示意)
steps = ["基线 fp16\\n32 KV 头", "KV 量化 fp8", "+ GQA(8 头)", "+ PagedAttention"]
vals = [base, l1, l2, l3]
fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.bar(steps, vals, color=["#95a5a6", "#2e86c1", "#27ae60", "#e67e22"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f} GB", ha="center", fontsize=10)
ax.set_ylabel("KV Cache 占用 (GB)")
ax.set_title("三个优化杠杆叠加:KV 从 34GB 一路压到 5GB")
plt.tight_layout()
print(f"基线 {base:.1f} GB → 三杠杆叠加后 {l3:.1f} GB,压缩 {base/l3:.1f}×")
'''),
"🔧 组合拳威力:三个杠杆叠加能把 KV 压缩 6-7 倍 —— 相当于把『8GB 才能跑』变成『1GB 就能跑』。")

NB.md("## 7️⃣ 配套 App:显存计算器 🎛️",
D('''
运行同目录的 `app_96_npu_memory.py`,**拖层数/KV 头/并发/序列长度、选精度与设备**,
实时看 KV 占用、显存占比、增长曲线与显存账单饼图:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_96_npu_memory.py
```

浏览器打开 **http://localhost:8501**(也可 `--server.port 8696`)。完整源码如下:
'''))

NB.code(app_cell("app_96_npu_memory.py", APP_96),
"📜 运行本 cell 会覆盖写入 `app_96_npu_memory.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "NPU 内存是金字塔:HBM/DDR → L2/L1 → UB,数据越靠近计算越快也越小",
        "内存复用三招:静态规划、Buffer 复用、内存池,让小池子装下大模型",
        "KV Cache 公式:2 × layers × kv_heads × head_dim × batch × seq × dtype_bytes",
        "KV 与 batch、seq 线性相乘,与精度成反比 —— 量化 KV 直接减半",
        "三大杠杆:KV 量化、GQA、PagedAttention,叠加可压缩 6-7 倍",
    ],
    practice=[
        "把第 4 节的 kv_cache_gb 改成返回字节数(bytes),核对 1e9 与 2^30 的差别",
        "用第 6 节模型算『1 个 32 层 7B 模型 + KV』总共需要多少显存,选一台装得下的昇腾",
        "调研 vLLM-Ascend 的 kv-cache-dtype 参数,写出 fp8 用法示例",
        "画一张『并发 vs 可支持上下文长度』的权衡曲线,标注甜蜜点",
    ],
    links=[
        ("昇腾 CANN 内存文档", "https://www.hiascend.com/document"),
        ("vLLM-Ascend", "https://github.com/vllm-project/vllm-ascend"),
        ("vLLM 自动前缀缓存(回顾)", "https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html"),
        ("GQA 论文", "https://arxiv.org/abs/2305.13245"),
    ])

out = str(Path(CH11) / "96_npu_memory_kv.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_96_npu_memory.py"
app_path.write_text(APP_96 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
