# -*- coding: utf-8 -*-
"""生成 53_triton_tile_model.ipynb 与 app_53_triton_tile.py"""
from pathlib import Path
from triton_helpers import D, new_nb, chapter_cover, wrapup, CH09
from triton_helpers import ENV_HEADER, PLT_STYLE

APP_FILE = "app_53_triton_tile.py"

APP_53 = D('''
# -*- coding: utf-8 -*-
# app_53_triton_tile.py — tile 编程模型:矩阵分块可视化 🧩
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧩 53 · Triton tile 模型", layout="wide")
st.title("🧩 第 53 课 · tile 编程模型:把矩阵切成砖块")

st.markdown("""
Triton 的思考单位是 **tile(数据块)**。对矩阵运算,你会把它沿两个方向切成一块块 **BM×BN 的小瓷砖**,
每个 tile 交给一个 program(用 `tl.program_id(0/1)` 定位行列),块内再用 `tl.arange` 铺开成线程。
下方选择**矩阵大小**与**tile 形状**,实时看矩阵被分成几行几列的块、每块长什么样。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    M = st.slider("矩阵行数 M", 8, 1024, 128, 8)
    N = st.slider("矩阵列数 N", 8, 1024, 256, 8)
    BM = st.select_slider("tile 行数 BM", options=[16, 32, 64, 128], value=64)
    BN = st.select_slider("tile 列数 BN", options=[16, 32, 64, 128], value=64)
    show_id = st.checkbox("在每个 tile 上标注 program_id", value=True)
    st.caption("BM×BN 决定每块 tile 的大小;程序编号按行优先排列。")

gm = (M + BM - 1) // BM   # 行方向 program 数
gn = (N + BN - 1) // BN   # 列方向 program 数
grid = gm * gn

c1, c2, c3 = st.columns(3)
c1.metric("行方向 program 数", gm)
c2.metric("列方向 program 数", gn)
c3.metric("grid 总 program 数", grid)

# ---------- tile 布局矩阵 ----------
Z = np.zeros((M, N), dtype=int)
for i in range(gm):
    for j in range(gn):
        pid = i * gn + j                       # 行优先编号
        r0, r1 = i * BM, min((i + 1) * BM, M)
        c0, c1 = j * BN, min((j + 1) * BN, N)
        Z[r0:r1, c0:c1] = pid

fig = go.Figure(go.Heatmap(
    z=Z, x=[f"{j}" for j in range(N)], y=[f"{i}" for i in range(M)],
    colorscale="Blues", showscale=False,
    text=Z if show_id else np.zeros_like(Z),
    texttemplate="%{text}", textfont=dict(size=9),
    hovertemplate="program %{z}<extra>%{y} 行, %{x} 列</extra>"))
fig.update_layout(title=f"矩阵 {M}×{N} 按 tile {BM}×{BN} 切分 → grid {gm}×{gn} 个 program",
                  height=max(420, M * 3), margin=dict(l=10, r=10, t=50, b=10),
                  xaxis=dict(title="列 j", scaleanchor="y"), yaxis=dict(title="行 i", autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:每个 program 通过 `pid = tl.program_id(0) * gn + tl.program_id(1)`(或直接两个维度)找到
> 自己负责的那块 tile,再用 `offs = pid * BLOCK + tl.arange(0, BLOCK)` 生成块内下标。**tile 的大小
> (BM×BN) 和数量(grid)是一对跷跷板**:tile 越大、块越少;tile 越小、块越多、每块并行度越低。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 9 章 · 第 53 课配套演示")
''')

NB = new_nb("第 53 课 · tile 编程模型",
            subtitle="program_id + tl.arange + block 指针:把 CUDA 的线程世界抽象成一块块数据瓷砖",
            emoji="🧩")

chapter_cover(NB,
    objectives=[
        "理解 tile(块)是 Triton 的最小编程单位,以及它与数据的关系",
        "掌握 tl.program_id 与 tl.arange 如何生成 tile 内的下标",
        "看懂二维 tile 与 reshape:BM×BN 块如何铺在 M×N 矩阵上",
        "建立 Triton 块模型与 CUDA 线程模型的映射关系",
        "认识 block 指针(make_block_ptr)及其在 GEMM/FlashAttention 中的作用",
    ],
    toc=[
        ("直觉:拼图与地毯", "把大矩阵切成一块块瓷砖,每个工人铺一块"),
        ("一维 tile:program_id + tl.arange", "下标生成器如何定位一段数据"),
        ("二维 tile 与 reshape", "BM×BN 的块如何在 M×N 矩阵上铺开"),
        ("与 CUDA 线程模型的映射", "program≈block,块内 arange≈thread 的展开"),
        ("block 指针:make_block_ptr", "描述 HBM 上一块二维数据的视图"),
        ("matplotlib:tile 布局示意图", "把分块画成棋盘,标注 program_id"),
        ("配套 Streamlit 演示", "app_53_triton_tile.py:拖 tile 形状看矩阵分块"),
    ],
    links=[
        ("Triton 语言参考", "https://triton-lang.org/main/python-api/triton.language.html"),
        ("Triton 矩阵乘法教程(二维 tile)", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
        ("NVIDIA 张量核与 tile 编程", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma"),
    ])

NB.md("## 1️⃣ 直觉:拼图与地毯 🧩",
D('''
想象你负责铺一块巨大的地毯,但只有一块块小瓷砖,而且人手很多。聪明做法是:**把地毯按网格切成
很多小块,每人负责一块**,大家互不打扰地同时铺。

这就是 Triton 的 **tile(块)编程模型**:数据(矩阵/张量)被切成若干块 tile,每个 program 负责
处理一块。你只需要描述“块怎么切、每块怎么算”,而不用管块内部具体是哪个线程在算——那是编译器的事。

对矩阵运算尤其自然:一块 **BM×BN** 的 tile,正是矩阵乘法里一个输出小块、或 FlashAttention 里
一个注意力小块。
'''))

NB.code(ENV_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed、确认 triton / torch / CUDA 就绪。")

NB.md("## 2️⃣ 一维 tile:program_id + tl.arange 🎯",
D('''
先看最简单的 1D tile。核心两行:

- `pid = tl.program_id(axis=0)`:我是第几个 program;
- `offsets = pid * BLOCK + tl.arange(0, BLOCK)`:我负责哪些下标。

`tl.arange(0, BLOCK)` 一次性生成 `[0, 1, ..., BLOCK-1]`,配合 `pid * BLOCK` 平移到本块的位置。
这比 CUDA 里手写 `tid` 循环直观得多——**循环被向量化成了 arange**。
'''))

NB.code(D('''
# 用一个小 kernel 验证 2D tile 的坐标与 program_id 的关系
@triton.jit
def tile2d(out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)                       # 行方向第几块
    pid_n = tl.program_id(1)                       # 列方向第几块
    offs_m = pid_m * BM + tl.arange(0, BM)         # 块内行下标
    offs_n = pid_n * BN + tl.arange(0, BN)         # 块内列下标
    # 把 (block 行, block 列, 块内行, 块内列) 编码成一个 float 存起来,方便肉眼读
    code = pid_m * 100.0 + pid_n * 10.0 + offs_m[:, None] * 0.01 + offs_n[None, :] * 0.0001
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], code)

M, N, BM, BN = 8, 8, 4, 4
out = torch.zeros(M, N, device="cuda")
tile2d[(M // BM, N // BN)](out, M, N, BM=BM, BN=BN)   # grid = (行块数, 列块数)
torch.cuda.synchronize()
print("整数值 = pid_m*100 + pid_n*10; 小数 = (块内行, 块内列)")
print(out.cpu().numpy())
'''), "🎯 grid 是二维 `(M//BM, N//BN)`,`tl.program_id(0)` 管行、`program_id(1)` 管列——二维 tile 就是这样定位的。")

NB.md("## 3️⃣ 二维 tile 与 reshape 📐",
D('''
把上面输出取整,你会看到它其实是个**分块棋盘**:每个 4×4 的小方块内整数值相同(同属一个 program)。
这个棋盘,正是矩阵按 tile 切分的真实样子:

- 每个 tile = **BM×BN** 小块;
- 行方向有 `M/BM` 块、列方向有 `N/BN` 块;
- program_id `(pid_m, pid_n)` 就是 tile 的行列坐标。

理解“**矩阵 → 二维 tile 网格**”这一层映射,是看懂 GEMM(下一课)和 FlashAttention(55 课)的地基。
'''))

NB.md("## 4️⃣ 与 CUDA 线程模型的映射 🗺️",
D('''
Triton 的 tile 和 CUDA 的线程模型有一一对应关系,把它们对照起来,你会发现 Triton 只是**把 CUDA
的“网格化”步骤替你做了**:

| Triton 概念 | 对应 CUDA 概念 | 作用 |
|------------|---------------|------|
| program / program_id | block / blockIdx | 块级并行单元,块间独立 |
| grid | grid | 一次启动的所有块 |
| tl.arange(0, BLOCK) | threadIdx + 手动换算 | 生成块内坐标 |
| 块内并行(编译器铺开) | block 内线程 | 一块数据内部的并行 |
| constexpr(BLOCK) | 模板/宏常量 | 决定块内布局 |

关键洞察:**CUDA 让你同时管 block 和 thread 两层;Triton 只让你管 program(块)这一层**,块内
thread 由编译器根据 `tl.arange` 的形状自动铺开。
'''))

NB.code(PLT_STYLE, "🎨 先套用统一样式头(浅色网格、大图、高 DPI),下面的示意图都用它。")

NB.code(D('''
# 画“tile → 二维 program 网格 → CUDA block/thread”的映射示意图
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# 1) 原始矩阵
ax = axes[0]; ax.set_title("① 数据:M×N 矩阵")
ax.add_patch(plt.Rectangle((0, 0), 4, 4, fc="#EDF1F7", ec="#4C78A8", lw=2))
ax.text(2, 2, "M 行 × N 列", ha="center", va="center")
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5); ax.axis("off")

# 2) tile 网格
ax = axes[1]; ax.set_title("② 切成 BM×BN 的 tile")
for i in range(2):
    for j in range(3):
        ax.add_patch(plt.Rectangle((j * 1.3, i * 1.3), 1.2, 1.2, fc="C0", alpha=0.35, ec="white"))
        ax.text(j * 1.3 + 0.6, i * 1.3 + 0.6, f"tile\\n({i},{j})", ha="center", va="center", fontsize=8)
ax.set_xlim(-0.3, 4.0); ax.set_ylim(-0.3, 2.8); ax.axis("off")

# 3) program_id 网格(≈ CUDA block)
ax = axes[2]; ax.set_title("③ program_id(0/1) ≈ CUDA blockIdx")
for i in range(2):
    for j in range(3):
        ax.add_patch(plt.Rectangle((j * 1.3, i * 1.3), 1.2, 1.2, fc="C1", alpha=0.4, ec="white"))
        ax.text(j * 1.3 + 0.6, i * 1.3 + 0.6, f"pid=({i},{j})", ha="center", va="center", fontsize=8)
ax.set_xlim(-0.3, 4.0); ax.set_ylim(-0.3, 2.8); ax.axis("off")

plt.tight_layout(); plt.show()
'''), "🎨 每个 tile 对应一个 program,program_id 就是它的网格坐标——和 CUDA 的 blockIdx 是同一个东西。")

NB.md("## 5️⃣ block 指针:make_block_ptr 🔗",
D('''
除了用“指针 + 下标”逐元素访问,Triton 还提供 **block 指针**(`tl.make_block_ptr`):它一次性描述
“一块二维数据的起始地址、总形状、步长、块形状、内存顺序”,配合 `tl.load / tl.store` 一次搬运整块,
用 `tl.advance` 前进到下一块。这在 GEMM 和 FlashAttention 里是标配写法——把“访问一整块数据”
抽象成一条语句。
'''))

NB.md("## 6️⃣ matplotlib:tile 布局示意图 📊",
D('''
用热力图把“矩阵被切成几行几列 tile、每块标上 program_id”画出来,棋盘结构一目了然。下面的
布局用 `tile2d` 的真实输出生成。
'''))

NB.code(D('''
Z = out.cpu().numpy().astype(int)   # 取整:pid_m*100 + pid_n*10
# 归一化成 0..nprog-1 便于着色
vals = sorted(set(Z.ravel()))
lookup = {v: i for i, v in enumerate(vals)}
col = np.vectorize(lookup.get)(Z)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(col, cmap="Blues", vmin=0, vmax=max(len(vals) - 1, 1))
for i in range(M):
    for j in range(N):
        ax.text(j, i, f"{Z[i, j]}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(N)); ax.set_yticks(range(M))
ax.set_xlabel("列"); ax.set_ylabel("行")
ax.set_title(f"8×8 矩阵按 {BM}×{BN} tile 切分:格内=pid_m*100+pid_n*10")
ax.grid(False); plt.tight_layout(); plt.show()
'''), "📊 同色区域 = 同一个 program 负责的 tile。`pid_m` 管行、`pid_n` 管列,格内整数值一眼可读。")

NB.md("## 7️⃣ 配套 Streamlit 演示:拖 tile 形状看分块 🎛️",
D('''
运行同目录下的 `app_53_triton_tile.py`,拖动 **矩阵大小 M×N** 与 **tile 形状 BM×BN**,实时看矩阵
被切成几行几列、每个 tile 标着哪个 program_id(可关闭标注):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_53_triton_tile.py
```

浏览器打开 **http://localhost:8501**。建议把 BM、BN 从 16 拖到 128,观察 tile 变大后块数变少、
反之块数变多。完整源码如下(与同目录 `app_53_triton_tile.py` 一字不差):
'''))

NB.code(f"%%writefile {APP_FILE}\n" + APP_53, "📜 这就是 app_53_triton_tile.py 的完整源码,纯 numpy 计算分块布局,拖动即可看棋盘结构。")

wrapup(NB,
    summary=[
        "tile 是 Triton 的最小编程单位:数据被切成 BM×BN 的块,每个 program 管一块",
        "tl.program_id(0/1) 定位块的行列;tl.arange(0, BLOCK) 生成块内下标,把循环向量化",
        "二维 tile 铺在 M×N 矩阵上,形成 (M/BM)×(N/BN) 的 program 网格",
        "program≈CUDA block,块内 arange 展开≈CUDA thread——Triton 替你管了第二层",
        "block 指针(make_block_ptr)一次描述整块数据的访问,是 GEMM/FlashAttention 的标配",
    ],
    practice=[
        "把 tile2d 的 BM、BN 改成非整除值(如 3、6),观察 grid 与掩码是否需要处理边界",
        "修改 tile2d,让编码改成 pid_m*1000 + pid_n*100 + offs_m*10 + offs_n,重画棋盘",
        "用 make_block_ptr 改写 tile2d,体会与“指针+下标”写法的区别",
        "思考:如果 M 不能被 BM 整除,程序编号和边界 tile 该如何处理(掩码 other=0)",
    ],
    links=[
        ("Triton 语言参考", "https://triton-lang.org/main/python-api/triton.language.html"),
        ("Triton 03-matrix-multiplication 教程", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
    ])

NB.save(str(Path(CH09) / "53_triton_tile_model.ipynb"))

app_path = Path(CH09) / APP_FILE
app_path.write_text(APP_53 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
