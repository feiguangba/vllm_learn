# -*- coding: utf-8 -*-
"""生成 56_triton_compiler.ipynb 与 app_56_triton_compiler.py"""
from helpers import D, CH09, CHAPTER, TRITON_HEADER, BENCH, VEC_ADD, GEMM, new_nb
from pathlib import Path
from nb_builder import chapter_cover, wrapup

APP_56 = D('''
# -*- coding: utf-8 -*-
# app_56_triton_compiler.py — 编译流程概念图 + 真实编译产物查看器 ⚙️
import os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_PTXAS = r"D:\\CUDA\\v13.3\\bin\\ptxas.exe"
if os.path.exists(_PTXAS):
    os.environ.setdefault("TRITON_PTXAS_PATH", _PTXAS)
import streamlit as st
import plotly.graph_objects as go
import torch
import triton
import triton.language as tl

st.set_page_config(page_title="Triton 编译器原理 ⚙️", layout="wide")
st.title("⚙️ 第 56 课 · Triton 编译器原理:从 Python 到 PTX")

st.markdown("""
Triton 不是解释执行你的 kernel,而是把它**逐级翻译、逐级优化**成 GPU 机器码。
下方可以调整 `BLOCK / num_warps / num_stages`,每动一次都会**真实地重新编译**一次
triton 矩阵乘 kernel,并展示各中间产物的规模与片段 —— 编译器流水线一目了然。
""")

# ---------------------------------------------------------------- 编译用的 kernel(与 notebook 一致)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

STAGES = [
    ("Python 源码", "你写的 @triton.jit 函数\\n(Python + tl.* 操作)"),
    ("AST", "语法树\\n函数被拆成节点"),
    ("TTIR", "Triton IR\\n循环拍平,只剩基本运算"),
    ("TTGIR", "Triton GPU IR\\ntile 布局 + 共享内存 + 调度"),
    ("LLVM IR", "寄存器分配\\n指令选择"),
    ("PTX", "NVIDIA 虚拟指令集\\n经 ptxas → SASS"),
]

# ---------------------------------------------------------------- 编译缓存
def get_compiled(BLOCK, nw, ns):
    key = (BLOCK, nw, ns)
    if key in st.session_state:
        return st.session_state[key]
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = torch.empty(256, 256, device="cuda", dtype=torch.float32)
    t0 = time.perf_counter()
    kern = matmul_kernel.warmup(a, b, c, 256, 256, 256,
                                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1),
                                BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=32,
                                num_warps=nw, num_stages=ns, grid=(1,))
    dt = (time.perf_counter() - t0) * 1000
    st.session_state[key] = (kern, dt)
    return st.session_state[key]

# ---------------------------------------------------------------- 侧边栏参数
with st.sidebar:
    st.header("🎛️ 编译参数")
    BLOCK = st.slider("BLOCK(输出 tile 边长)", 32, 128, 64, 16)
    num_warps = st.slider("num_warps(线程束数)", 1, 8, 4, 1)
    num_stages = st.slider("num_stages(流水线深度)", 1, 6, 3, 1)
    stage = st.selectbox("📄 查看哪一级产物", ["TTIR", "TTGIR", "LLVM IR", "PTX"])
    st.caption("滑杆变化即触发重新编译;TTGIR 会显示 num-warps / shared memory 等调度信息。")

kern, dt = get_compiled(BLOCK, num_warps, num_stages)
asm = kern.asm
stage_keys = {"TTIR": "ttir", "TTGIR": "ttgir", "LLVM IR": "llir", "PTX": "ptx"}
sizes = {s: len(asm[k]) for s, k in stage_keys.items()}
ptx_lines = asm["ptx"].count("\\n")

# ---------------------------------------------------------------- 指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("编译耗时", f"{dt:.0f} ms")
c2.metric("TTIR", f"{sizes['TTIR']} 字符")
c3.metric("TTGIR", f"{sizes['TTGIR']} 字符")
c4.metric("PTX", f"{ptx_lines} 行")
c5.metric("num_warps", num_warps)

# ---------------------------------------------------------------- 编译流水线概念图(plotly)
st.subheader("🧭 编译流水线(概念图)")
labels = [s[0] for s in STAGES]
x_pos = list(range(len(STAGES)))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_pos, y=[1] * len(STAGES), mode="markers+text", text=labels,
    textposition="bottom center", textfont=dict(size=13),
    marker=dict(size=34, color=["#72B7B2", "#4C78A8", "#F28E2B", "#E45756", "#76B7B2", "#59A14F"])))
for i in range(len(STAGES) - 1):
    fig.add_annotation(x=(x_pos[i] + x_pos[i + 1]) / 2, y=1.0, text="➜", showarrow=False,
                       font=dict(size=20))
for i, (t, d) in enumerate(STAGES):
    fig.add_annotation(x=x_pos[i], y=0.72, text=d, showarrow=False, font=dict(size=12))
fig.update_yaxes(range=[0.4, 1.4], visible=False)
fig.update_xaxes(visible=False)
fig.update_layout(height=260, title="Python 源码 一路“翻译 + 优化”到 PTX / SASS",
                  margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("当前这个 kernel 已经编译到最后一站;你可以在侧边栏切换查看每一站的产物文本。")

# ---------------------------------------------------------------- 各阶段规模柱状图(plotly)
fig2 = go.Figure(go.Bar(x=list(sizes.keys()), y=list(sizes.values()),
                        marker_color="#4C78A8",
                        text=[f"{v // 1000:.1f} KB" for v in sizes.values()],
                        textposition="outside"))
fig2.update_layout(title="各编译阶段的产物规模(TTIR 最精简,PTX 最庞大)",
                   xaxis_title="编译阶段", yaxis_title="字符数", height=360,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- 产物片段
st.subheader(f"📄 {stage} 片段(前 30 行)")
lines = asm[stage_keys[stage]].splitlines()
st.code("\\n".join(lines[:30]), language="text")
if stage == "TTGIR":
    hit = [ln for ln in lines if "num-warps" in ln or "local_alloc" in ln][:4]
    if hit:
        st.markdown("**🔍 关键调度信息**")
        st.code("\\n".join(hit), language="text")
        st.caption("num-warps:该 kernel 编译成了几个线程束;local_alloc:在共享内存里分配了 tile 缓冲。")

st.markdown("""
> 💡 **结论**:同一个 Python 函数,编译器在不同参数下会编出**不同的 TTGIR 调度与 PTX**。
> tile 编程之所以高效,是因为编译器能基于 tile 做寄存器分配、共享内存复用与软件流水线。
""")
''')

NB = new_nb("第 56 课 · Triton 编译器原理",
            subtitle="从 Python 一路读到 PTX:看懂 triton 把 tile 程序“编译”成 GPU 机器码的完整流水线",
            emoji="⚙️")

chapter_cover(NB,
    objectives=[
        "理解 triton 编译管线的完整链路:Python AST → TTIR → TTGIR → LLVM IR → PTX → SASS",
        "亲手编译一个 kernel,读它生成的 TTIR / TTGIR / PTX 文本",
        "理解 tile(分块)编程为什么能让编译器高效地做寄存器分配与流水线调度",
        "用 num_warps / num_stages 观察同一段源码如何编出不同的调度",
    ],
    toc=[
        ("直觉:一条“编译器流水线工厂”", "源码要过六道工序,每一站都换成更接近硬件的语言"),
        ("第一站:Python 源码与 AST", "@triton.jit 先“读题”,把函数拆成语法树"),
        ("第二站:TTIR", "循环被拍平,只剩逐元素的 tl 运算"),
        ("第三站:TTGIR", "tile 布局、共享内存、线程调度在这里登场"),
        ("第四、五站:LLVM IR 与 PTX", "寄存器分配与指令选择,ptxas 最后落成 SASS"),
        ("为什么 tile 编程让编译器高效调度", "给编译器一块“大砖”,它才能帮你搬得快"),
        ("动手实验:改 num_warps 看调度变化", "同一段源码,不同参数编出不同 TTGIR/PTX"),
        ("配套 Streamlit 演示", "app_56_triton_compiler.py:调参实时看编译产物"),
    ],
    links=[
        ("Triton 官方仓库(triton-lang)", "https://github.com/triton-lang/triton"),
        ("Triton 编译器架构文档", "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html"),
        ("MLIR 官方文档(TTIR/TTGIR 的底座)", "https://mlir.llvm.org/"),
        ("NVIDIA PTX ISA 文档", "https://docs.nvidia.com/cuda/parallel-thread-execution/"),
    ])

NB.md("## 1️⃣ 直觉:一条“编译器流水线工厂” 🏭",
D('''
上一课我们把注意力 kernel 写了出来,它跑得飞快——但你写的明明是**没有写任何线程、没有写任何
寄存器分配**的 Python 代码,GPU 到底怎么执行它的?答案藏在编译器里。

想象你在一家工厂下了一张“积木玩具”的订单:你只画了**设计图**(Python + `tl.*` 算子),
工厂里一条流水线把设计图逐级“翻译 + 优化”成机器能直接执行的语言。每一站都换一种更接近硬件的语言:

| 站 | 语言 | 这一站在干什么 |
|----|------|----------------|
| 1 | Python 源码 | 你写的 `@triton.jit` 函数 |
| 2 | AST | 把函数解析成语法树(还没接触 GPU) |
| 3 | TTIR(Triton IR) | 循环拍平,只剩标量运算图 |
| 4 | TTGIR(Triton GPU IR) | **tile 布局 + 共享内存 + 线程调度** |
| 5 | LLVM IR | 寄存器分配、指令选择、向量化 |
| 6 | PTX → SASS | NVIDIA 虚拟指令集,最终落成硬件机器码 |

“tile 编程”之所以高效,秘密全在第 4 站:你只描述“把这块 tile 算完”,而**怎么把 tile 摊到
线程、怎么把数据放进共享内存、怎么流水线重叠访存与计算**,都是编译器在第 4~6 站自动完成的。
下面我们亲手把编译器的每一站“揭开”看看。
'''))

NB.code(TRITON_HEADER, "✅ 环境自检:统一设置 KMP 防护、指定可用的系统 ptxas(内嵌 ptxas 在本机报“内存分配失败”),并确认 GPU 与 triton 可用。")

NB.md("## 2️⃣ 第一站:Python 源码与 AST 📜",
D('''
`@triton.jit` 装饰器做的第一件事是**“读题”**:用 `inspect` 拿到函数的源码,再交给 Python 标准库
`ast` 解析成一棵**语法树(AST)**。这一站完全不知道 GPU 的存在——它只是一棵普通的 Python 语法树。

我们用一段文本模拟 triton 的“读题”过程:把向量加法的源码字符串解析成 AST,并打印出
`Module → FunctionDef → arguments → ...` 的结构。真正运行 kernel 时,triton 会对这棵树做
后续的逐级编译。
'''))

NB.code(D('''
import inspect, ast

VEC_SRC = """
@triton.jit
def vec_add_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))
"""
tree = ast.parse(VEC_SRC)
print("AST 根节点类型:", type(tree).__name__)
print("顶层语句个数   :", len(tree.body))
fn = tree.body[-1]
print("函数名         :", fn.name)
print("参数名         :", [a.arg for a in fn.args.args])
for node in fn.body[:3]:
    print("  语句 ->", type(node).__name__)
'''), "🔍 你会看到 AST 只是一棵“语法树”:函数、参数、语句都还是抽象节点,没有任何 GPU 相关概念。")

NB.md("## 3️⃣ 第二站:TTIR——循环被拍平 📐",
D('''
AST 很快被翻译成 **TTIR(Triton IR)**,这是一种 MLIR 方言。在这一站,你对整个 tile 的意图被
“拍平”成**基本运算图**:`tl.program_id` 变成 `tt.get_program_id`,`+` 变成 `tt.addf`,
`tl.load/store` 变成 `tt.load/tt.store`。

把编译产物打印出来看,我们用 `kernel.warmup` 触发一次编译(只编译、不运行),然后从
`compiled.asm` 字典里取出各阶段文本。`asm` 的键正是流水线的各站:`ttir / ttgir / llir / ptx / cubin`。
'''))

NB.code(VEC_ADD)

NB.code(D('''
# 编译一次,拿到五个阶段的产物
x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
o = torch.empty_like(x)
compiled = vec_add_kernel.warmup(x, y, o, BLOCK=256, num_warps=4, num_stages=3, grid=(1,))

print("asm 阶段键:", list(compiled.asm.keys()))
print()
print("====== TTIR(前 18 行)======")
ttir = compiled.asm["ttir"]
print("\\n".join(ttir.splitlines()[:18]))
'''), "🎯 关键观察:TTIR 里你已经看不到 Python 的 `for` 循环——它被展开成**扁平的基本块**;`tl.arange` 变成 `tt.make_range`,`program_id` 变成显式节点。")

NB.md("## 4️⃣ 第三站:TTGIR——tile 布局与调度登场 🎛️",
D('''
接下来是最精彩的一站:**TTGIR(Triton GPU IR)**。它才真正“懂 GPU”:编译器在这里决定

- 每个 tile 的数据**放哪个线程、哪个寄存器**(`ttg.blocked` 布局);
- 中间结果**要不要搬进共享内存**(`ttg.local_alloc`,注意这是 SMEM);
- 一共编译成**几个线程束**(`ttg.num-warps`),K 循环要不要流水线(`ttg.pipelined`)。

这就是“你只管 tile,调度交给编译器”的具体体现。打印 TTGIR,并 grep 出关键调度行:
'''))

NB.code(D('''
ttgir = compiled.asm["ttgir"]
print("====== TTGIR(前 14 行)======")
print("\\n".join(ttgir.splitlines()[:14]))
print()
print("====== 关键调度信息 ======")
for ln in ttgir.splitlines():
    if any(k in ln for k in ("num-warps", "local_alloc", "pipelined", "blocked")):
        print(ln[:150])
'''), "🔍 这些 `ttg.blocked<{sizePerThread=..., threadsPerWarp=..., warpsPerCTA=...}>` 就是编译器给你的“摊砖方案”:每个线程拿几块数据、几个线程一个 warp、几个 warp 一个 CTA。")

NB.md("## 5️⃣ 第四、五站:LLVM IR 与 PTX 💻",
D('''
TTGIR 再往下是 **LLVM IR**:编译器在这里做寄存器分配、指令选择与向量化;最终通过 NVIDIA 后端
生成 **PTX**(NVIDIA 的虚拟指令集),再由驱动里的 `ptxas` 编成 SASS(cubin)机器码。

PTX 是可以“读懂”的——它和 SASS 一样是显式指令,只是平台无关。打印 LLVM IR 与 PTX 片段,
再数一数这个简单的 kernel 编出了多少行 PTX:
'''))

NB.code(D('''
print("====== LLVM IR(前 12 行)======")
print("\\n".join(compiled.asm["llir"].splitlines()[:12]))
print()
print("====== PTX(前 14 行)======")
ptx = compiled.asm["ptx"]
print("\\n".join(ptx.splitlines()[:14]))
print()
print(f"PTX 总行数: {ptx.count(chr(10))} 行 | cubin 大小: {len(compiled.asm['cubin'])} 字节")
'''), "🎯 你会看到 PTX 里出现 `ld.global`, `add.f32`, `st.global`——数据从全局内存读进来、算出结果、写回去,这就是 GPU 的“三幕剧”。")

NB.md("## 6️⃣ 为什么 tile 编程让编译器高效调度 🧱",
D('''
回到本课的核心问题:**为什么“我描述 tile、编译器生成调度”比“我手写每个线程”更高效?**

想象你在教一只机械臂搬砖。方案 A:你逐块告诉它“左移 3 格、抓起、右移 5 格、放下……”;
方案 B:你说“把这面 8×8 的墙,搬到那个位置”。方案 B 让机械臂自己规划**最优路线**(复用、
顺路、并行),因为你给的是一块**可以被推理的大单元**。

tile 就是编译器能推理的“大砖头”:

- **寄存器分配**:整块 tile 的每个元素该住在哪个寄存器,编译器能统一规划,避免你手写时到处 spill;
- **共享内存复用**:`tl.dot` 的两个输入可以放在 SMEM,一块数据被多次复用(算术强度↑);
- **软件流水线**:K 方向的循环里,`num_stages` 让编译器把“下一块数据的加载”和“当前块的计算”重叠,
  访存与算力互不等待。

下面一张图总结数据流动:全局内存(HBM)→ 共享内存(SRAM)→ 寄存器 → 写回。tile 是每一层之间
搬运的最小单位,搬运路径由编译器规划:
'''))

NB.code(D('''
# 数据流动示意:tile 是各层之间搬运的最小单位
fig, ax = plt.subplots(figsize=(9, 4))
layers = [
    (0.0, "HBM(全局内存)", "几百 GB/s · 带宽最贵", "#E45756"),
    (1.0, "SMEM(共享内存)", "几十 TB/s · 块内共享", "#F28E2B"),
    (2.0, "寄存器 / 张量核", "几百 TB/s · 计算发生地", "#4C78A8"),
]
for i, (y, name, note, color) in enumerate(layers):
    ax.add_patch(plt.Rectangle((0.05, y), 0.9, 0.55, color=color, alpha=0.75, ec="k"))
    ax.text(0.5, y + 0.27, f"{name}\\n{note}", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.annotate("tile 数据搬运 ↑", xy=(0.15, y + 0.56), xytext=(0.85, y + 0.56),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"), fontsize=10, color="#333")
ax.set_xlim(0, 1); ax.set_ylim(-0.2, 2.9); ax.axis("off")
ax.set_title("编译器的搬运路线:tile 是每层之间的最小搬运单位", fontsize=12)
plt.tight_layout()
'''), "🎨 图里三层的带宽逐层提升两个数量级,所以“把数据搬进寄存器附近复用”就是 kernel 快慢的分水岭——tile 让编译器能自动设计这条路线。")

NB.md("## 7️⃣ 动手实验:改 num_warps,看调度变化 🧪",
D('''
最后做个小实验:同一段 GEMM 源码,只改 `num_warps`(1 vs 8),重新编译,对比 TTGIR 元数据与
PTX 规模。你会看到**调度真的变了**:`ttg.num-warps` 不同、`ttg.blocked` 的线程排布不同、
PTX 行数也不同。这就是“同样的 Python,不同的机器码”。
'''))

NB.code(GEMM)

NB.code(D('''
a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
c = torch.empty(128, 128, device="cuda", dtype=torch.float32)

for nw in (1, 8):
    kk = matmul_kernel.warmup(a, b, c, 128, 128, 128,
                              a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                              num_warps=nw, num_stages=3, grid=(1,))
    meta = [ln for ln in kk.asm["ttgir"].splitlines() if "num-warps" in ln][0].strip()
    print(f"num_warps={nw}: {meta[:80]}")
    print(f"             PTX 行数 = {kk.asm['ptx'].count(chr(10))}")
'''), "🎯 观察:num_warps=1 时编译器仍会自己决定用几个线程束——你给的是“上限/提示”,真正的分配由 TTGIR 层完成;越复杂的 kernel,这个差异越明显。")

NB.md("## 8️⃣ 配套 Streamlit 演示:调参实时看编译产物 🎛️",
D('''
运行同目录下的 `app_56_triton_compiler.py`,拖动 **BLOCK / num_warps / num_stages** 三个滑杆,
页面会**真实地重新编译**矩阵乘 kernel,并展示编译耗时、TTIR/TTGIR/LLVM/PTX 各阶段规模
与文本片段。编译流水线的概念图也一并在页面上呈现:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_56_triton_compiler.py
```

浏览器打开 **http://localhost:8501**。建议把 num_warps 从 1 拖到 8,观察 TTGIR 里
`num-warps` 与共享内存分配的变化。完整源码如下(与同目录 `app_56_triton_compiler.py` 一字不差):
'''))

NB.code("%%writefile app_56_triton_compiler.py\n" + APP_56, "📜 app_56_triton_compiler.py 完整源码:notebook 与 app 共用同一个 `matmul_kernel`,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "triton 编译管线:Python AST → TTIR → TTGIR → LLVM IR → PTX → SASS,每一站换一种更接近硬件的语言",
        "TTIR 把循环拍平成基本运算图;TTGIR 才是“懂 GPU”的地方:tile 布局、共享内存、num_warps、流水线都在此决定",
        "读产物用 `kernel.warmup(...)` + `compiled.asm['ttir'/'ttgir'/'llir'/'ptx'/'cubin']`",
        "tile 编程让编译器能做寄存器分配、SMEM 复用与软件流水线——这就是 triton 高生产力的来源",
        "同样的源码,改 num_warps / num_stages 会编出不同的 TTGIR 与 PTX,这就是下一课调优的物理基础",
    ],
    practice=[
        "把 vector add 的 BLOCK 从 256 改成 1024 重新编译,对比 TTGIR 里 sizePerThread 的变化",
        "给 vector add 加一层 mask(offs < N),再看 TTIR 里多了哪些节点",
        "数一数 matmul kernel 的 PTX 行数,并找到 `mma` / `ldmatrix` 指令(张量核证据)",
        "用 triton.compile 接口(`triton.compiler.compile`)直接编译一段手写 TTIR 字符串,体验 MLIR 层面",
    ],
    links=[
        ("Triton 官方教程:01-vector-add", "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html"),
        ("Triton 仓库(编译器源码在 python/src/triton)", "https://github.com/triton-lang/triton"),
        ("MLIR 方言(TTGIR 的语法基础)", "https://mlir.llvm.org/docs/LangRef/"),
    ])

NB.save(str(Path(CH09) / "56_triton_compiler.ipynb"))

app_path = Path(CH09) / "app_56_triton_compiler.py"
app_path.write_text(APP_56 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
