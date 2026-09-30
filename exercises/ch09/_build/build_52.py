# -*- coding: utf-8 -*-
"""生成 52_triton_vectoradd.ipynb 与 app_52_triton_vecadd.py"""
from pathlib import Path
from triton_helpers import D, new_nb, chapter_cover, wrapup, CH09
from triton_helpers import ENV_HEADER, PLT_STYLE, BENCH, ADD_KERNEL

APP_FILE = "app_52_triton_vecadd.py"

APP_52 = D('''
# -*- coding: utf-8 -*-
# app_52_triton_vecadd.py — 向量加法:tile 大小滑杆看性能 ➕
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch, triton, triton.language as tl

st.set_page_config(page_title="➕ 52 · Triton 向量加法", layout="wide")
st.title("➕ 第 52 课 · 向量加法:grid / block / pid 与性能")

st.markdown("""
向量加法 `y = x + a` 是最简单的 Triton kernel,却足以讲清 **grid(几个 program)/ block(一块多大)/
pid(我是第几块)** 三个核心概念。GPU kernel 是**访存受限**的:瓶颈在读写显存的带宽,而不是算力。
下方拖动 **tile 大小(BLOCK)** 与 **向量长度**,实时测 Triton 内核的带宽,并与 torch 原生 GPU 加法对比。
""")

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def bench(fn, *args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

@st.cache_data(show_spinner=False)
def bench_block(n, block):
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    def tfn():
        add_kernel[(triton.cdiv(n, block),)](x, y, out, n, BLOCK_SIZE=block)
    t0 = time.perf_counter(); tfn(); torch.cuda.synchronize()  # 触发编译
    t = bench(tfn, warmup=5, iters=30)
    def gfn(): torch.add(x, y, out=out)
    tg = bench(gfn, warmup=5, iters=30)
    bd = n * 4 * 2 / (t / 1000) / 1e9
    bdg = n * 4 * 2 / (tg / 1000) / 1e9
    return t, tg, bd, bdg

with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("向量长度 n(元素数)", 1_000_000, 64_000_000, 16_000_000, 1_000_000)
    block = st.select_slider("tile 大小 BLOCK", options=[32, 64, 128, 256, 512, 1024, 2048, 4096], value=1024)
    iters = st.slider("计时迭代次数", 10, 100, 30, 10)
    st.caption("BLOCK 越大,每个 program 一次处理的元素越多;向量长度需能被 BLOCK 近似整除才有意义。")

t_tri, t_torch, bd_tri, bd_torch = bench_block(n, block)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Triton 带宽", f"{bd_tri:.0f} GB/s")
c2.metric("torch GPU 带宽", f"{bd_torch:.0f} GB/s")
c3.metric("Triton 耗时", f"{t_tri:.3f} ms")
c4.metric("torch 耗时", f"{t_torch:.3f} ms")
st.caption(f"n={n:,} 元素 | BLOCK={block} | 理论需读写 {n*8/1e6:.0f} MB(读 x,y + 写 out)")

# ---------- 性能随 BLOCK 变化曲线 ----------
blocks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
rows = []
for b in blocks:
    tt, tg, bd, bdg = bench_block(n, b)
    rows.append((b, bd, bdg))
bs = [r[0] for r in rows]; bds = [r[1] for r in rows]; bdgs = [r[2] for r in rows]
fig = go.Figure()
fig.add_trace(go.Scatter(x=bs, y=bds, mode="lines+markers", name="Triton", line=dict(color="#4C78A8", width=3)))
fig.add_trace(go.Scatter(x=bs, y=bdgs, mode="lines+markers", name="torch GPU", line=dict(color="#E45756", width=3, dash="dot")))
fig.add_vline(x=block, line_dash="dash", line_color="gray")
fig.update_layout(xaxis=dict(type="log", title="BLOCK(tile 大小)"),
                  yaxis_title="带宽 GB/s", height=380, margin=dict(l=10, r=10, t=40, b=10),
                  title="带宽随 BLOCK 变化(虚线=当前选择)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:GPU 加法是访存受限的,带宽趋近于显存的峰值(HBM)。BLOCK 太小时,块数过多、
> 启动/调度开销占比大,带宽上不去;BLOCK 足够大后,曲线进入平台期——再大也不会有明显提升。
> 这解释了为什么“调 tile 大小”是 Triton 里最常见的性能旋钮之一。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 9 章 · 第 52 课配套演示")
''')

NB = new_nb("第 52 课 · 第一个 Triton kernel:向量加法",
            subtitle="用三行 kernel 讲清 grid / block / pid,并在 RTX 5060 上实测 CPU / torch / Triton 三种加法",
            emoji="➕")

chapter_cover(NB,
    objectives=[
        "写出并运行第一个 Triton kernel:向量加法",
        "理解 grid、block、pid 三者如何决定数据切分",
        "看懂 @triton.jit 的编译流程与 warmup 的必要性",
        "在 GPU 上实测 CPU / torch / Triton 三种加法的耗时与带宽",
        "掌握访存受限(memory-bound)的概念",
    ],
    toc=[
        ("直觉:快递分拣员与流水线", "向量加法为什么值得单独讲一课"),
        ("第一个 kernel:三行代码", "program_id + tl.arange + tl.load/store"),
        ("grid / block / pid:数据怎么切", "用一张图看懂切分与每个 program 的职责"),
        ("编译流程与 warmup", "为什么先跑一次再计时"),
        ("实测对比:CPU / torch / Triton", "RTX 5060 上三种加法的耗时与带宽"),
        ("性能曲线:带宽随 BLOCK 变化", "matplotlib + pyecharts 双图"),
        ("配套 Streamlit 演示", "app_52_triton_vecadd.py:拖 tile 大小看带宽"),
    ],
    links=[
        ("Triton 向量加法官方教程", "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html"),
        ("triton.language API", "https://triton-lang.org/main/python-api/triton.language.html"),
        ("PyTorch CUDA semantics", "https://pytorch.org/docs/stable/notes/cuda.html"),
    ])

NB.md("## 1️⃣ 直觉:快递分拣员与流水线 📦",
D('''
“向量加法 `z = x + y`”听着简单,但它是理解一切 GPU kernel 的**最小样本**:它有输入(x、y)、
有计算(相加)、有输出(z),而 GPU 的并行就在于**把一个大向量切成很多小段,交给很多工人同时加**。

把 GPU 想成一家快递分拣中心:一件大包裹(整个向量)太重,一个人搬不动,于是切成很多小块,
每个分拣员(program)抱走自己那一块,独立完成“相加”这件事,再放回货架。

关键问题是:**怎么切、每个分拣员拿哪一块**?这就是本课的主角——`grid`、`block`、`pid`。
'''))

NB.code(ENV_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed、确认 triton / torch / CUDA 就绪。")

NB.md("## 2️⃣ 第一个 kernel:三行核心 🧱",
D('''
Triton kernel 就是一个用 `@triton.jit` 装饰的 Python 函数。内部只能用 `tl` 的操作和“张量”,不能
print、不能随便用 Python 对象。向量加法内核的核心就三件事:算自己管哪些下标 → 读 → 加 → 写。
'''))

NB.code(ADD_KERNEL, "**逐行读**:`pid` 是第几块;`offsets` 是这块负责的所有下标;`mask` 防止越界;`tl.load` 读一块、`+` 相加、`tl.store` 写回。")

NB.code(D('''
n = 1 << 20                       # 1,048,576 个元素
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)

BLOCK = 1024
grid = (triton.cdiv(n, BLOCK),)   # 需要 n/BLOCK 个 program(向上取整)
add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)   # 语法:kernel[grid](参数...)
torch.cuda.synchronize()

err = (out - (x + y)).abs().max().item()
print(f"grid = {grid[0]} 个 program, BLOCK = {BLOCK}")
print(f"最大误差 = {err:.2e}  (应为 0 或浮点级小量)")
'''), "🎯 `kernel[grid](args)` 是 Triton 的调用语法:`grid` 决定启动多少个 program,其余是参数。")

NB.md("## 3️⃣ grid / block / pid:数据怎么切 🗺️",
D('''
三个概念一图说清:

- **grid**:一次启动的所有 program 总数,这里 `grid = (n // BLOCK,)`;
- **block(BLOCK_SIZE)**:每个 program 一次处理的元素个数;
- **pid(program_id)**:当前这个 program 是第几号,决定它管数据里的哪一段。

一个 program 内部的 `tl.arange(0, BLOCK)` 会在该 program 内并行(编译器自动铺开成多个线程/warp)。
所以“**外层 program 之间并行、内层由编译器铺开**”是 Triton 的双层并行心智模型。
'''))

NB.code(PLT_STYLE, "🎨 先套用统一样式头(浅色网格、大图、高 DPI),下面的示意图都用它。")

NB.code(D('''
# 画“grid / block / pid 切分数据”的示意图
n, BLOCK = 16, 4
fig, ax = plt.subplots(figsize=(10, 3.2))
for pid in range(n // BLOCK):
    x0 = pid * BLOCK
    for j in range(BLOCK):
        ax.add_patch(plt.Rectangle((x0 + j, 0.15), 0.85, 0.7,
                                   fc=f"C{pid}", alpha=0.55, ec="black", lw=0.5))
    ax.text(x0 + BLOCK / 2, 1.05, f"program {pid}", ha="center", va="bottom",
            fontsize=11, color=f"C{pid}", fontweight="bold")
    ax.annotate("", xy=(x0 + BLOCK, 0.5), xytext=(x0, 0.5),
                arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlim(-0.5, n); ax.set_ylim(0, 1.5); ax.axis("off")
ax.set_title(f"n={n} 个元素, BLOCK={BLOCK}, grid={n // BLOCK} 个 program:每个 program 管 BLOCK 个连续元素", fontsize=11)
plt.tight_layout(); plt.show()
'''), "📊 同色=同一个 program 负责。`pid * BLOCK + arange(0, BLOCK)` 正是这段连续下标的来源。")

NB.md("## 4️⃣ 编译流程与 warmup ⚙️",
D('''
`@triton.jit` 不是普通的函数调用:首次调用时,它会根据**参数类型 + BLOCK 值 + GPU 架构(sm_120)**
把源码编译成 GPU 机器码(几秒),然后按 key 缓存;之后同参数调用直接复用缓存(毫秒级)。

所以计时时必须**先 warmup 一次**(触发编译、预热缓存),再用 `torch.cuda.synchronize()` 同步后计时,
否则第一次的编译时间会污染你的基准测试。下面就把这套流程写成一个可复用的 `bench` 函数。
'''))

NB.code(BENCH, "**bench 的做法**:先 warmup 若干次,同步,再跑 iters 次取平均。GPU kernel 是异步的,`torch.cuda.synchronize()` 是计时准确的必要条件。")

NB.md("## 5️⃣ 实测对比:CPU / torch / Triton ⚖️",
D('''
在 RTX 5060 上,对比四种做法在不同向量长度下的耗时:

- **CPU**:`torch.add` 跑在 CPU(16 线程);
- **torch GPU**:`torch.add` 跑在 GPU(cuBLAS/自家 kernel);
- **Triton**:我们自己写的 `add_kernel`。

GPU 加法是**访存受限(memory-bound)**的:瓶颈不是算加法(太快了),而是**读 x、读 y、写 z** 的
显存带宽。所以用“带宽(GB/s)”来衡量最公平。
'''))

NB.code(D('''
sizes = [1 << 20, 1 << 22, 1 << 24, 1 << 26]   # 1M / 4M / 16M / 64M
rows = []
for n in sizes:
    xg = torch.randn(n, device="cuda"); yg = torch.randn(n, device="cuda"); og = torch.empty_like(xg)
    xc, yc, oc = xg.cpu(), yg.cpu(), torch.empty_like(xg.cpu())
    t_cpu = bench(lambda: torch.add(xc, yc, out=oc), warmup=2, iters=5)
    t_torch = bench(lambda: torch.add(xg, yg, out=og), warmup=5, iters=30)
    def tri():
        add_kernel[(triton.cdiv(n, 1024),)](xg, yg, og, n, BLOCK_SIZE=1024)
    t_tri = bench(tri, warmup=5, iters=30)
    bd = lambda ms: n * 4 * 2 / (ms / 1000) / 1e9   # 读 x,y + 写 out
    rows.append((n, t_cpu, t_torch, t_tri, bd(t_torch), bd(t_tri)))
    print(f"n={n:9d}: CPU {t_cpu:8.3f}ms | torch {t_torch:7.3f}ms {bd(t_torch):5.1f}GB/s | triton {t_tri:7.3f}ms {bd(t_tri):5.1f}GB/s")
'''), "🚀 数据量越大,CPU 与 GPU 的差距越明显;而 torch 与 Triton 都逼近显存带宽的峰值。")

NB.md("## 6️⃣ 性能曲线:带宽随 BLOCK 变化 📊",
D('''
上面用了固定的 BLOCK=1024。现在固定一个较大向量长度,把 **BLOCK 从 32 扫到 4096**,看 Triton 的
带宽如何随 tile 大小变化。BLOCK 太小,启动/调度开销占比大;足够大后进入平台期。
'''))

NB.code(D('''
n = 1 << 24   # 16M 元素,足够大,能看出趋势
blocks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
xg = torch.randn(n, device="cuda"); yg = torch.randn(n, device="cuda"); og = torch.empty_like(xg)
bd_list, tg_list = [], []
for b in blocks:
    def tri():
        add_kernel[(triton.cdiv(n, b),)](xg, yg, og, n, BLOCK_SIZE=b)
    t_tri = bench(tri, warmup=3, iters=20)
    tg = bench(lambda: torch.add(xg, yg, out=og), warmup=3, iters=20)
    bd_list.append(n * 8 / (t_tri / 1000) / 1e9)
    tg_list.append(n * 8 / (tg / 1000) / 1e9)
    print(f"BLOCK={b:5d}: triton {t_tri:6.3f}ms {bd_list[-1]:5.1f}GB/s | torch {tg:6.3f}ms {tg_list[-1]:5.1f}GB/s")
'''), "🔍 记录每个 BLOCK 的 Triton 带宽,下面画成曲线。")

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(blocks, bd_list, "o-", color="#4C78A8", lw=2.5, label="Triton")
ax.axhline(np.mean(tg_list), color="#E45756", ls="--", lw=2, label="torch GPU(平台期参考)")
ax.set_xscale("log", base=2)
ax.set_xticks(blocks); ax.set_xticklabels([str(b) for b in blocks])
ax.set_xlabel("BLOCK(tile 大小)"); ax.set_ylabel("带宽 GB/s")
ax.set_title("向量加法带宽随 BLOCK 变化(RTX 5060, n=16M)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
'''), "📊 BLOCK 太小时带宽低,BLOCK≥512 后进入平台期——这就是“调 tile 大小”的直观依据。")

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

bar = (Bar()
       .add_xaxis([f"BLOCK={b}" for b in blocks])
       .add_yaxis("Triton 带宽(GB/s)", [round(v, 1) for v in bd_list], color="#4C78A8")
       .set_global_opts(title_opts=opts.TitleOpts(title="不同 tile 大小下的 Triton 带宽"),
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                        yaxis_opts=opts.AxisOpts(name="GB/s")))
bar.render_notebook()
'''), "📊 pyecharts 交互柱状图:悬停看每个 BLOCK 的带宽,平台期的特征一目了然。")

NB.md("## 7️⃣ 配套 Streamlit 演示:拖 tile 大小看带宽 🎛️",
D('''
运行同目录下的 `app_52_triton_vecadd.py`,拖动 **向量长度** 与 **tile 大小(BLOCK)**,实时在 GPU 上
跑 Triton 加法内核、显示带宽,并与 torch 原生 GPU 加法对比;下方还画了“带宽随 BLOCK 变化”的完整曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_52_triton_vecadd.py
```

浏览器打开 **http://localhost:8501**。建议把 BLOCK 从 32 一路拖到 4096,看带宽先爬升、再进入平台期。
完整源码如下(与同目录 `app_52_triton_vecadd.py` 一字不差):
'''))

NB.code(f"%%writefile {APP_FILE}\n" + APP_52, "📜 这就是 app_52_triton_vecadd.py 的完整源码,用 st.cache_data 缓存不同 BLOCK 的基准结果,拖拽丝滑。")

wrapup(NB,
    summary=[
        "Triton kernel = @triton.jit 装饰的函数,内部只用 tl 原语,不能 print / 用任意 Python 对象",
        "grid 决定启动多少 program;BLOCK 决定每个 program 处理多少元素;pid 决定管哪一段",
        "kernel[grid](args) 是调用语法,首次调用触发编译,之后命中缓存",
        "GPU 加法是访存受限的:瓶颈在显存带宽,用 GB/s 衡量最公平",
        "BLOCK 太小调度开销大,足够大后进入带宽平台期——tile 大小是最常见性能旋钮",
    ],
    practice=[
        "把 BLOCK 从 1024 改成 333(非 2 的幂),观察是否报错,体会 constexpr 需为 2 的幂",
        "给 add_kernel 增加第三个向量 z,实现三向量相加 y = a + b + c",
        "用 torch.cuda.synchronize 前后计时对比,体会不同步导致的计时虚低",
        "把 BLOCK 扫到 8192,看带宽是否回落(寄存器/占用率下降),找出最佳点",
    ],
    links=[
        ("Triton 01-vector-add 教程", "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html"),
        ("PyTorch CUDA 语义", "https://pytorch.org/docs/stable/notes/cuda.html"),
    ])

NB.save(str(Path(CH09) / "52_triton_vectoradd.ipynb"))

app_path = Path(CH09) / APP_FILE
app_path.write_text(APP_52 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
