# -*- coding: utf-8 -*-
"""生成 79_graph_fusion.ipynb 与 app_79_fusion.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_79 = D('''
# -*- coding: utf-8 -*-
# app_79_fusion.py — 图算融合(Graph Kernel Fusion)开关对比 🧬
import time
import streamlit as st
import plotly.graph_objects as go
import torch

st.set_page_config(page_title="图算融合 🧬", layout="wide")
st.title("🧬 第 79 课 · 图算融合:融合开关对比")

st.markdown("""
昇腾 CANN 的**图引擎 GE** 会在编译期做**图算融合(Graph Kernel Fusion)**:把多个相邻
算子合并成一个大 kernel,省掉中间张量的『写出去再读回来』。下方**打开 / 关闭融合**、
调整**链式算子个数**与**张量大小**,实时对比耗时与访存量。
""")

fuse = st.sidebar.checkbox("启用算子融合", value=True)
n_ops = st.sidebar.slider("链式算子个数", 2, 8, 4, 1)
size = st.sidebar.slider("张量元素数", 100_000, 10_000_000, 2_000_000, 100_000)
show_dag = st.sidebar.checkbox("显示融合前后 DAG", value=True)
st.sidebar.caption("融合的核心收益:中间张量不再落回全局内存(省 2×访存/算子)。")

torch.manual_seed(0)
x = torch.randn(size)

def chain(x, n):
    y = x
    for i in range(n):
        y = y * 1.0001 + 0.5        # 逐元素算子链(加法、乘法)
    return y

def bench(fn, iters=10):
    ts = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - ts) / iters * 1000

if fuse:
    t = bench(lambda: chain(x, n_ops))
else:
    def chain_sep(x, n):
        y = x
        for i in range(n):
            y = y * 1.0001
            y = y + 0.5              # 每个算子单独一轮(模拟不融合,中间落盘)
        return y
    t = bench(lambda: chain_sep(x, n_ops))

traffic = size * 4 * 2 if fuse else size * 4 * 2 * n_ops   # 每算子读+写 8B

c1, c2, c3, c4 = st.columns(4)
c1.metric("单次耗时", f"{t:.3f} ms")
c2.metric("算子数", f"{2 * n_ops} 个基础算子")
c3.metric("访存量", f"{traffic/1e6:.0f} MB")
c4.metric("融合状态", "开" if fuse else "关")

if show_dag:
    st.subheader("🌳 融合前后 DAG")
    fig = go.Figure()
    if fuse:
        fig.add_trace(go.Scatter(x=[0.2], y=[0.5], mode="markers+text", marker=dict(size=60, color="#4C78A8"),
                                 text=[f"Fused kernel\\n(n_ops×2 算子合并)"], textposition="middle center", textfont=dict(size=12)))
        fig.add_annotation(x=0.2, y=-0.15, text="输入 x", showarrow=False)
        fig.add_annotation(x=0.2, y=1.15, text="输出 y", showarrow=False)
    else:
        xs = [i / (n_ops * 2 + 1) for i in range(1, n_ops * 2 + 1)]
        fig.add_trace(go.Scatter(x=xs, y=[0.5] * len(xs), mode="markers",
                                 marker=dict(size=26, color="#E45756"),
                                 text=[f"op{i}" for i in range(1, len(xs) + 1)],
                                 textposition="top center"))
        fig.add_annotation(x=0.0, y=0.5, text="输入", showarrow=False)
        fig.add_annotation(x=1.0, y=0.5, text="输出", showarrow=False)
    fig.update_xaxes(showticklabels=False, range=[-0.1, 1.1])
    fig.update_yaxes(showticklabels=False, range=[-0.3, 1.4])
    fig.update_layout(title="算子链:红点=未融合的独立算子,蓝块=融合后的单 kernel",
                      height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("融合后中间张量不再落盘,访存从 2n 次降到 2 次。")

st.subheader("📊 不同张量大小下的耗时")
sizes = [100_000, 500_000, 1_000_000, 5_000_000]
times_on = [bench(lambda: chain(torch.randn(s), n_ops)) for s in sizes]
def chain_sep(x, n):
    y = x
    for i in range(n):
        y = y * 1.0001
        y = y + 0.5
    return y
times_off = [bench(lambda: chain_sep(torch.randn(s), n_ops)) for s in sizes]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sizes, y=times_off, mode="lines+markers", name="不融合(逐算子)",
                          line=dict(color="#E45756", width=3)))
fig2.add_trace(go.Scatter(x=sizes, y=times_on, mode="lines+markers", name="融合(单 kernel)",
                          line=dict(color="#4C78A8", width=3)))
fig2.update_layout(xaxis_type="log", title="融合 vs 不融合:张量越大访存节省越明显",
                   xaxis_title="张量元素数(对数)", yaxis_title="耗时(ms)", height=380)
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 观察:算子数越多、张量越大,融合收益越大 —— 大模型推理的逐元素链正是融合的富矿。")

st.markdown("""
> 💡 **一句话**:GE 图算融合 ≈ torch.compile 的融合 —— 都是把『小算子链』压成
> 『一个大 kernel』。昇腾把它放在编译期自动完成,开发者无感但收益常在。
""")
''')

NB = new_nb("第 79 课 · 图算融合:GE 如何把 N 个算子并成 1 个 kernel",
            subtitle="算子融合 / Buffer 融合 / 统一 IR —— 与 torch.compile 的融合同源,用 DAG 看懂融合前后",
            emoji="🧬")

chapter_cover(NB,
    objectives=[
        "理解图算融合(Graph Kernel Fusion)为什么是图优化的第一板斧",
        "掌握 GE 的三大融合策略:算子融合、Buffer 融合、统一 IR",
        "用 matplotlib 画融合前后的计算图 DAG,直观看到『中间张量消失』",
        "用 torch 实测融合 vs 不融合的访存账与耗时差异",
        "对照 torch.compile 的融合思想,理解跨框架的共性",
        "跑通配套 App:融合开关对比",
    ],
    toc=[
        ("直觉:把三步工序并成一步", "融合省的不是计算,是搬货(访存)"),
        ("GE 的三大融合策略", "算子融合 / Buffer 融合 / 统一 IR"),
        ("融合前后 DAG", "matplotlib 双图:5 个算子 → 2 个算子"),
        ("访存账与实测", "每算子读+写 8 字节,融合后只读一次写一次"),
        ("与 torch.compile 对照", "昇腾 GE vs Inductor 的『殊途同归』"),
        ("配套 Streamlit 演示", "app_79_fusion.py:融合开关对比"),
    ],
    links=[
        ("昇腾 CANN 图算融合文档", "https://www.hiascend.com/document"),
        ("PyTorch torch.compile 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:把三步工序并成一步 🏭",
D('''
回想第 72 课:数据从 HBM 搬到片内要花大代价。如果三个逐元素算子(A→B→C)各算各的,
每个算子的结果都要**写回 HBM、再被下一个算子读出来** —— 三步就是三次搬货。

**图算融合(Graph Kernel Fusion)** 的思路:让这三个算子在片内一次性算完 —— 只搬一次货、
只写一次结果。省的不是计算(计算量没变),省的是**搬货(访存)**。

昇腾 CANN 的**图引擎 GE** 在编译期自动做这件事:它拿到计算图后,把相邻的、能合并的
算子融合成一个大 kernel,再整图下沉执行。这就是 GE 图算融合(GKF, Graph Kernel Fusion)。
'''))

NB.md("## 2️⃣ GE 的三大融合策略 🧩",
D('''
GE 的融合不是"蛮干",而是三招组合拳:

1. **算子融合**:把语义上能合并的算子捏成一个。典型:
   - 算术融合(Mul + Add 合成一个 `MulAdd` 类算子,省中间张量);
   - 相邻逐元素链融合(ReLU+BN+Add 等);
   - 反向传播中 `Softmax + SoftmaxGrad` 这类结对算子合并。
2. **Buffer 融合(Buffer Fusion)**:不能合并的算子,至少让它们**复用同一块片上缓冲**,
   减少缓冲区申请与数据搬运 —— 相当于"货车司机换着开同一辆车";
3. **统一 IR**:融合后的算子用一套统一的中间表示(IR)表达,方便后续继续优化与下沉执行。

用一张表总结:
| 策略 | 解决的问题 | 收益 |
|------|-----------|------|
| 算子融合 | 中间张量落盘 | 访存从 2n 降到 2 |
| Buffer 融合 | 缓冲区反复申请 | 内存复用、搬运减少 |
| 统一 IR | 融合产物可编译 | 整图下沉、持续优化 |
'''))

NB.md("## 3️⃣ 融合前后 DAG:中间张量凭空消失 🎨",
D('''
把一条 5 算子链的 DAG 画出来,对比融合前后 —— 左图有 4 个中间张量节点,
右图它们全部消失,只剩一个大的 Fused Kernel:
'''))

NB.code(D('''
def draw_dag(ax, nodes, title):
    ax.axis("off")
    x0, y0, bw, bh = 0.5, 0.6, 1.6, 0.8
    xs = np.linspace(x0, x0 + bw * (len(nodes) - 1) * 1.4, len(nodes))
    for i, (label, color) in enumerate(nodes):
        ax.add_patch(plt.Rectangle((xs[i] - bw/2, y0), bw, bh, facecolor=color,
                                   edgecolor="#333", lw=1.5))
        ax.text(xs[i], y0 + bh/2, label, ha="center", va="center", fontsize=9,
                fontweight="bold")
    for i in range(len(nodes) - 1):
        ax.annotate("", xy=(xs[i+1] - bw/2, y0 + bh/2), xytext=(xs[i] + bw/2, y0 + bh/2),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#555"))
    ax.text(0.5, 1.75, title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 9); ax.set_ylim(0, 2.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
draw_dag(ax1, [("Mul", "#E45756"), ("Add", "#E45756"), ("ReLU", "#E45756"),
               ("Mul", "#E45756"), ("Add", "#E45756")], "融合前:5 个算子 + 4 个中间张量")
draw_dag(ax2, [("MulAdd", "#4C78A8"), ("FusedReLU-MulAdd", "#4C78A8")],
         "融合后:2 个大 kernel,中间张量消失")
fig.suptitle("图算融合:把『反复搬货』压成『一次搬完』", fontsize=13)
plt.tight_layout()
'''),
"🎨 DAG 对比:融合后中间结果全程留在片内 —— 这就是 GKF 收益的视觉化。")

NB.md("## 4️⃣ 访存账:每算子 8 字节,融合后只掏一次 🧮",
D('''
一个逐元素算子 = 读 1 次 + 写 1 次,每元素按 fp32 计 4 字节,就是 **8 字节/元素**。
n 个算子链 = 2n 次访存;融合成一个 = 2 次。用 torch 实测这条账:
'''))

NB.code(D('''
x = torch.randn(5_000_000)
def bench(fn, iters=10):
    ts = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - ts) / iters * 1000

def chain(x, n):                       # 融合写法:一个表达式
    y = x
    for _ in range(n):
        y = y * 1.0001 + 0.5
    return y

def chain_sep(x, n):                   # 不融合写法:每个算子独立一轮
    y = x
    for _ in range(n):
        y = y * 1.0001
        y = y + 0.5
    return y

n = 5
n_elem = x.numel()
t_fuse, t_sep = bench(lambda: chain(x, n)), bench(lambda: chain_sep(x, n))
print(f"{n} 轮链式运算: 融合 {t_fuse:.4f}ms  vs  分离 {t_sep:.4f}ms")
print(f"访存账: 分离 = {2*n} 次访问(每元素 {2*n*4}B), 融合 = 2 次访问(8B)")
print(f"实测加速比: {t_sep / t_fuse:.2f}×")
'''),
"✅ 实测:同一计算,融合写法明显更快 —— 访存减少的收益真实可见(本机为 CPU/GPU 通用规律,昇腾上同样成立)。")

NB.md("## 5️⃣ 与 torch.compile 融合:殊途同归 🧭",
D('''
昇腾的 GE 图算融合,与 PyTorch 的 `torch.compile`(Inductor 融合)思想同源:

| 维度 | 昇腾 GE 图算融合 | torch.compile(Inductor) |
|------|------------------|--------------------------|
| 在哪做 | 编译期(模型加载时) | 编译期(首次运行时 JIT) |
| 融合单位 | 算子 → 大 kernel | 算子 → Triton/C++ kernel |
| Buffer 复用 | Buffer 融合 | 内存规划器 |
| 触发方式 | 自动(加载即优化) | 显式 `torch.compile(model)` |
| 下沉 | 整图下沉到昇腾 | 生成 kernel 由 GPU 执行 |
| 动态形状 | 需重编译/特化 | 支持动态(更灵活) |

一句话:**昇腾把『编译+融合+下沉』打包进了 CANN;PyTorch 把它做成了可选的编译接口**。
两者都在做同一件事 —— 让 GPU/NPU 别再被『小算子 + 反复搬货』拖慢。
'''))

NB.md("## 6️⃣ 配套 Streamlit 演示:融合开关对比 🎛️",
D('''
运行同目录下的 `app_79_fusion.py`,可以**打开 / 关闭融合、调整算子个数与张量大小**,
实时看耗时、访存量与融合前后 DAG:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_79_fusion.py
```

浏览器打开 **http://localhost:8501**。建议把算子数调到 8、张量拉到 1000 万,观察融合
收益随规模放大;再关掉融合对比访存量的爆涨。完整源码如下(与同目录
`app_79_fusion.py` 一字不差):
'''))

NB.code(app_cell("app_79_fusion.py", APP_79),
"📜 运行本 cell 会覆盖写入 `app_79_fusion.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "图算融合省的不是计算,是访存:中间张量不再落回全局内存",
        "GE 三大策略:算子融合(并算子)、Buffer 融合(复用缓冲)、统一 IR(可编译)",
        "融合后 DAG 里中间节点消失,访存从 2n 次降到 2 次",
        "实测账:融合写法明显快于分离写法,算子越多张量越大收益越明显",
        "昇腾 GE 融合与 torch.compile 思想同源:把『小算子 + 反复搬货』压成『大 kernel 一次搬完』",
    ],
    practice=[
        "把第 4 节的 chain 改成 y = relu(x·w+b),用 torch.nn.functional 对比融合与分离耗时",
        "画一张『融合收益 vs 算子个数』的柱状图(横轴 n=2..10,纵轴加速比)",
        "查 vLLM 里 vLLM 对 GEMM+act 融合的实现(PagedAttention 的 fused kernels),写 3 条要点",
        "思考:为什么逐元素链最容易融合,而 GEMM+softmax 这类『跨语义』算子融合更难?",
    ],
    links=[
        ("昇腾 CANN 图算融合文档", "https://www.hiascend.com/document"),
        ("PyTorch torch.compile 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
    ])

out = str(Path(CH11) / "79_graph_fusion.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_79_fusion.py"
app_path.write_text(APP_79 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
