# -*- coding: utf-8 -*-
"""生成 51_triton_intro.ipynb 与 app_51_triton_intro.py"""
from pathlib import Path
from triton_helpers import D, new_nb, chapter_cover, wrapup, CH09
from triton_helpers import ENV_HEADER, PLT_STYLE, LIFECYCLE_KERNEL

APP_FILE = "app_51_triton_intro.py"

APP_51 = D('''
# -*- coding: utf-8 -*-
# app_51_triton_intro.py — Triton 是什么:与 CUDA 的概念对照 🐍
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🐍 51 · Triton 入门", layout="wide")
st.title("🐍 第 51 课 · Triton:GPU 编程新范式")

st.markdown("""
Triton 是一种**面向 GPU 的编程语言**。与 CUDA 需要你关心“线程、共享内存、同步”这些底层细节不同,
Triton 让你只关心**数据分块(tile)**:你说“把矩阵切成 128×128 的块,每块怎么算”,编译器帮你
把块里的并行、访存、同步全部搞定。下方用三个控件,直观对比 Triton 与 CUDA 的概念。
""")

concepts = {
    "tile 编程模型 (tl 块)": {
        "triton": "@triton.jit\\ndef k(x):\\n    offs = tl.arange(0, BLOCK)\\n    a = tl.load(x + offs)",
        "cuda": "int tid = blockIdx.x*blockDim.x + threadIdx.x;\\nfloat a = x[tid];",
        "说明": "Triton 以“块(block/tile)”为最小思考单位,写代码像在处理一个小矩阵;CUDA 则面向单个线程 (thread)。Triton 由编译器自动铺开线程。",
    },
    "program_id(块编号)": {
        "triton": "pid = tl.program_id(axis=0)",
        "cuda": "int pid = blockIdx.x;   // 每个 block",
        "说明": "告诉“这个块是第几个”。Triton 里每个 program 相当于 CUDA 的一个 block;块与块之间互相独立、可并行。",
    },
    "tl.arange(连续下标)": {
        "triton": "offs = tl.arange(0, BLOCK)",
        "cuda": "for(int i=tid; i<N; i+=stride) ...   // 手动算下标",
        "说明": "tl.arange(0, BLOCK) 一次性生成 0..BLOCK-1 的向量,是 Triton 最常用的“下标生成器”,把一维循环改写成向量化。",
    },
    "constexpr 编译期常量": {
        "triton": "def k(..., BLOCK: tl.constexpr):",
        "cuda": "#define BLOCK 128   // 模板参数",
        "说明": "标成 constexpr 的参数必须在编译期已知(如块大小),编译器据此展开、特化并优化;运行时参数则保持为变量。",
    },
    "num_warps(线程束数)": {
        "triton": "k[grid](..., num_warps=4)",
        "cuda": "blockDim.x = 128;  // 1 warp = 32 线程",
        "说明": "每个 program 内部用多少个 warp(32 线程为一束)去执行块内计算。调大通常提高块内并行度,但也增加寄存器压力。",
    },
    "tl.dot(块级矩阵乘)": {
        "triton": "acc = tl.dot(a, b, acc)",
        "cuda": "调用 tensor core / 手写分块矩阵乘",
        "说明": "Triton 把矩阵乘抽象成 tl.dot,自动调度张量核(Tensor Core);CUDA 需要手写或调 cuBLAS/mma 指令。",
    },
    "@triton.jit 装饰器": {
        "triton": "@triton.jit\\ndef k(...): ...",
        "cuda": "nvvcc 编译 .cu 源码 → cubin",
        "说明": "用 @triton.jit 装饰 Python 函数,它会被即时编译(JIT)成 GPU 机器码;第一次调用时编译并缓存,之后直接复用。",
    },
    "block 指针 (make_block_ptr)": {
        "triton": "p = tl.make_block_ptr(base, shape, strides, ...)",
        "cuda": "T* p = base + row*ld + col;   // 指针运算",
        "说明": "Triton 用“块指针”描述 HBM 上一块二维数据及其步长,配合 tl.load/store 一次性搬运整块;CUDA 用裸指针手动算偏移。",
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    sel = st.selectbox("选择一个 Triton 概念", list(concepts.keys()))
    grid_n = st.slider("grid 大小(program 个数)", 1, 32, 8, 1)
    model = st.radio("编程范式", ["SPMD(数据并行)", "线程中心(thread-centric)"])
    st.caption("grid = 一次启动的 program 总数,每个 program 独立处理自己那块数据。")

c = concepts[sel]
st.subheader("🔀 对照:" + sel)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**🐍 Triton**")
    st.code(c["triton"], language="python")
with col2:
    st.markdown("**🎯 CUDA**")
    st.code(c["cuda"], language="cpp")
st.markdown("**💡 解读**:" + c["说明"])

# ---------- grid / program 可视化 ----------
st.subheader("🧮 grid 上的 program 分布")
rows = int(grid_n ** 0.5)
cols = (grid_n + rows - 1) // rows
xs, ys, labels = [], [], []
for i in range(grid_n):
    xs.append(i % cols + 0.5)
    ys.append(i // cols + 0.5)
    labels.append(str(i))
fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers+text",
                           text=labels, textposition="middle center",
                           marker=dict(size=40, color="#4C78A8", line=dict(width=2, color="white"))))
fig.update_layout(xaxis=dict(range=[0, cols], showgrid=False, title="program 列"),
                  yaxis=dict(range=[0, rows], showgrid=False, title="program 行", autorange="reversed"),
                  height=300, margin=dict(l=10, r=10, t=30, b=10),
                  title=f"grid = {grid_n} 个 program(≈ CUDA 的 {grid_n} 个 block,块间并行)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:Triton 的核心心智模型是 **SPMD(单程序多数据)**——同一份 kernel 源码,被并行地在
> 很多个 program(block)上执行,每个 program 用 `tl.program_id` 知道自己管哪一块数据。
> 它把 CUDA 里“线程 ↔ 数据”的手动映射,升级成了“块 ↔ 数据”的声明式描述,复杂度交给编译器。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 9 章 · 第 51 课配套演示")
''')

NB = new_nb("第 51 课 · Triton 是什么",
            subtitle="GPU 编程新范式:像写 Python 一样写 GPU kernel,把 CUDA 的线程细节交给编译器",
            emoji="🐍")

chapter_cover(NB,
    objectives=[
        "理解 Triton 在 GPU 编程生态中的位置,与 CUDA 的本质区别(块级 vs 线程级)",
        "掌握 triton.language(tl)的核心概念:program_id、tl.arange、constexpr、tl.dot",
        "看懂 @triton.jit 的生命周期:源码 → 编译器 → 机器码 → 缓存复用",
        "跑通第一个真实的 Triton kernel,并在 GPU 上验证输出",
    ],
    toc=[
        ("直觉:工厂流水线与组装车间", "Triton 让 GPU 编程从“管线程”升级为“管数据块”"),
        ("Triton vs CUDA:两种心智模型", "块级编程(SPMD)与线程级编程(thread-centric)"),
        ("tl 核心概念一览", "program_id / tl.arange / constexpr / tl.dot / block 指针"),
        ("生命周期:@triton.jit 怎么干活", "从 Python 源码到 GPU 机器码再到缓存复用"),
        ("跑一个真实 kernel", "grid_demo:把每个 program 的编号写进显存"),
        ("plotly:program 分布可视化", "交互看 grid 上 program 如何铺开"),
        ("配套 Streamlit 演示", "app_51_triton_intro.py:逐个概念对照 Triton/CUDA"),
    ],
    links=[
        ("Triton 官方文档", "https://triton-lang.org/main/index.html"),
        ("Triton 官方教程(代码示例)", "https://triton-lang.org/main/getting-started/tutorials/index.html"),
        ("NVIDIA CUDA 编程指南", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"),
    ])

NB.md("## 1️⃣ 直觉:工厂流水线与组装车间 🏭",
D('''
传统 CUDA 编程,像让你**亲自管理一条流水线的每个工位**:每个线程(工人)手动算自己该站哪个位置、
该搬哪块料、什么时候等别人(同步)。数据一多,这种“线程中心”的写法就非常痛苦。

**Triton 换了个思路**:你只负责告诉工厂“把整批货切成若干大箱(块/tile),每箱怎么加工”,
至于每箱里有多少工人、怎么分工、怎么协作,全交给编译器去安排。你写的代码是 **SPMD(单程序多数据)**:
同一份加工说明书,在每个箱子上同时执行。

这就把 GPU 编程从“盯着线程算下标”升级成了“盯着数据块描述算法”——更接近数学、更贴近你要做的
矩阵/张量运算,这也是 vLLM、FlashAttention 等高性能 kernel 越来越爱用 Triton 的原因。
'''))

NB.code(ENV_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed、确认 triton / torch / CUDA 就绪。")

NB.md("## 2️⃣ Triton vs CUDA:两种心智模型 ⚖️",
D('''
两者最大的差别,是**思考的单位**不同:

- **CUDA**:以“线程(thread)”为单位。你写 `threadIdx`、`blockIdx`,手动把 1D 下标换算成 2D 行列,
  自己管理共享内存和同步。代码能压榨极致性能,但开发慢、易错。
- **Triton**:以“块(block/tile)”为单位。你写 `tl.program_id`、`tl.arange`,声明一块数据怎么读、
  怎么算、怎么写;线程的铺开、寄存器的分配、访存的重排都由编译器代劳。

一句话:CUDA 告诉你“每个线程做什么”,Triton 告诉你“每块数据怎么算”。
'''))

NB.code(PLT_STYLE, "🎨 先套用统一样式头(浅色网格、大图、高 DPI),下面的示意图都用它。")

NB.code(D('''
# 画一张“CUDA 线程中心 vs Triton 块中心”的对比示意图
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左:CUDA —— 网格/块/线程 三层
ax = axes[0]
ax.set_title("CUDA:网格 → 块 → 线程(thread-centric)")
for gx in range(3):
    for gy in range(2):
        ax.add_patch(plt.Rectangle((gx * 3.4, gy * 2.2), 3.0, 1.8, fc="#f6e8e8", ec="#E45756", lw=1.5))
        ax.text(gx * 3.4 + 1.5, gy * 2.2 + 0.9, f"block({gx},{gy})", ha="center", va="center", fontsize=8)
        for tx in range(2):
            for ty in range(2):
                ax.add_patch(plt.Rectangle((gx * 3.4 + 0.15 + tx * 1.4, gy * 2.2 + 0.15 + ty * 0.7),
                                           1.2, 0.5, fc="#F7D4D4", ec="#c0504d"))
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 4.6); ax.axis("off")
ax.text(5, 4.15, "每个小方格 = 一个线程,代码围绕线程转", ha="center", fontsize=9)

# 右:Triton —— 只有“program”这个概念
ax = axes[1]
ax.set_title("Triton:program(块)为中心,SPMD")
for gx in range(3):
    for gy in range(2):
        ax.add_patch(plt.Rectangle((gx * 3.4, gy * 2.2), 3.0, 1.8, fc="#e6eef6", ec="#4C78A8", lw=1.5))
        ax.text(gx * 3.4 + 1.5, gy * 2.2 + 0.9, f"program {gx + gy * 3}", ha="center", va="center", fontsize=8)
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 4.6); ax.axis("off")
ax.text(5, 4.15, "同一份 kernel 源码在每个 program 上执行,线程交给编译器", ha="center", fontsize=9)

plt.tight_layout(); plt.show()
'''), "📊 左边 CUDA 要手动管理 block/thread 两层并行;右边 Triton 只要描述 program,内部并行由编译器铺开。")

NB.md("## 3️⃣ tl 核心概念一览 🧰",
D('''
`import triton.language as tl` 提供了一套**张量级原语**,是写 kernel 的“积木”。最常用的五个:

| 概念 | 作用 | 类比 |
|------|------|------|
| `tl.program_id(axis)` | 拿到当前 program(块)的编号 | 我是第几个车间 |
| `tl.arange(0, N)` | 生成 0..N-1 的向量(下标生成器) | 把 1D 循环向量化 |
| `tl.load / tl.store` | 从 HBM 读/写一整块数据 | 一次搬一箱货 |
| `tl.constexpr` | 编译期常量(块大小等) | 出厂前就定死的参数 |
| `tl.dot` | 块级矩阵乘(张量核) | 自动调度 Tensor Core |

`BLOCK: tl.constexpr` 特别关键:它必须在编译期已知,编译器据此把 kernel 展开、特化、并决定
寄存器/共享内存布局。运行期才知道的值(如张量大小)则是普通参数。
'''))

NB.md("## 4️⃣ 生命周期:@triton.jit 怎么干活 ⚙️",
D('''
`@triton.jit` 装饰一个 Python 函数,它就被改造成一个 **JIT 编译器对象**。调用它并不立即执行 Python,
而是经历:

1. **首次调用**:按参数类型、`constexpr` 值、GPU 架构 `sm_120` 做**特化**,编译成 PTX/机器码
   (第一次较慢,几秒级);
2. **缓存**:编译产物按“key”缓存到磁盘,相同参数再次调用直接命中缓存(毫秒级);
3. **启动**:把编译好的 kernel 按 `grid` 大小在 GPU 上启动。

所以你总是**先 warmup 一次**(触发编译),再正式计时,否则第一次的编译时间会混进你的基准测试。
'''))

NB.md("## 5️⃣ 跑一个真实 kernel:grid_demo 🚀",
D('''
下面这个 kernel 没有做任何“数学运算”,它的作用是把每个元素的**所属 program 编号(pid)**
写进显存——用最直接的方式让你“看见” SPMD:哪些元素属于同一个 program,一目了然。
'''))

NB.code(LIFECYCLE_KERNEL, "`pid * 1.0 + lane * 1e-3`:整数部分=program 编号,小数部分=块内通道号,便于肉眼分辨。")

NB.code(D('''
# 直观展示生命周期:grid_demo 是一个 JITFunction 对象
print("JIT 缓存目录:", os.environ.get("TRITON_CACHE_DIR", "默认(系统临时目录)"))
print()
print("grid_demo 是一个 JITFunction 对象(可被调用):")
print(type(grid_demo))
'''), "🔍 JITFunction 封装了“源码 + 参数签名 + 编译产物”,第一次调用才真正编译。")

NB.code(D('''
n = 16
BLOCK = 4
out = torch.zeros(n, device="cuda")
grid = (triton.cdiv(n, BLOCK),)          # 需要 n/BLOCK 个 program
grid_demo[grid](out, n, BLOCK_SIZE=BLOCK)   # 首次调用触发编译
torch.cuda.synchronize()
print("grid =", grid, "| BLOCK =", BLOCK)
print("输出(整数=program 编号,小数=块内通道):")
print(out.cpu().numpy())
'''), "🎯 每 4 个元素(一块)的整数部分相同——它们由同一个 program 处理。`tl.cdiv(n, BLOCK)` 是向上取整的除法。")

NB.md("## 6️⃣ plotly:program 分布可视化 📊",
D('''
把上面 16 个元素画成网格,用颜色区分它们分别属于哪个 program。你会直观看到“块(program)”
把数据整齐地瓜分成几段——这就是 SPMD 的铺开方式。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

pid_map = out.cpu().numpy().astype(int)
fig = go.Figure(go.Heatmap(z=[pid_map], x=[f"元素 {i}" for i in range(n)],
                           y=["program 归属"], colorscale="Blues", text=[pid_map],
                           texttemplate="%{text}", showscale=False))
fig.update_layout(title="16 个元素被 4 个 program 瓜分(同色 = 同一块)",
                  height=220, margin=dict(l=10, r=10, t=50, b=10))
fig
'''), "🎨 相同 program 编号的元素在数据上连续成块——Triton 的块粒度由你设置的 BLOCK 决定。")

NB.md("## 7️⃣ 配套 Streamlit 演示:概念对照 🎛️",
D('''
运行同目录下的 `app_51_triton_intro.py`,下拉选择 **Triton 概念**(tile、program_id、tl.arange、
constexpr、num_warps、tl.dot、@jit、block 指针),逐条对照它的 **CUDA 等价写法**与解读;
还能拖动 grid 大小,实时看 program 在网格上的分布:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_51_triton_intro.py
```

浏览器打开 **http://localhost:8501**。完整源码如下(与同目录 `app_51_triton_intro.py` 一字不差):
'''))

NB.code(f"%%writefile {APP_FILE}\n" + APP_51, "📜 这就是 app_51_triton_intro.py 的完整源码,把本课讲到的 Triton 概念逐条与 CUDA 对照。")

wrapup(NB,
    summary=[
        "Triton 是面向 GPU 的高级语言:以数据块(tile)为单位编程(SPMD),线程细节交给编译器",
        "与 CUDA 的本质区别是思考单位:CUDA 管线程, Triton 管块",
        "tl 的核心原语:program_id、tl.arange、tl.load/store、constexpr、tl.dot",
        "@triton.jit 让函数变成 JIT 编译器:首次调用编译+缓存,之后毫秒级复用",
        "同色连续成块即同一 program 处理,这就是 SPMD 的铺开方式",
    ],
    practice=[
        "把 grid_demo 的 BLOCK 改成 8、2,观察每个 program 处理的元素个数如何变化",
        "给 grid_demo 加一行,把 pid*lane 也存下来,验证块内通道的编号",
        "用 triton.cdiv 解释:当 n 不能被 BLOCK 整除时,为什么还要向上取整",
        "查 triton 的磁盘缓存目录,看看编译产物到底存在哪里",
    ],
    links=[
        ("Triton 官方文档", "https://triton-lang.org/main/index.html"),
        ("Triton 教程:语言参考", "https://triton-lang.org/main/python-api/triton.language.html"),
    ])

NB.save(str(Path(CH09) / "51_triton_intro.ipynb"))

app_path = Path(CH09) / APP_FILE
app_path.write_text(APP_51 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
