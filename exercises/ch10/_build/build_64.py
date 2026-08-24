# -*- coding: utf-8 -*-
"""生成 64_mlir_intro.ipynb 与 app_64_mlir.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH10, app_cell, finalize
from pathlib import Path

APP_64 = D('''
# -*- coding: utf-8 -*-
# app_64_mlir.py — MLIR 与中间表示:Dialect / Pass 概念浏览 🏗️
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="MLIR 与中间表示 🏗️", layout="wide")
st.title("🏗️ 第 64 课 · MLIR 与中间表示")

st.markdown("""
MLIR(Multi-Level Intermediate Representation)是 Google 推出的一套**编译器基础设施**。它的核心
思想是:**没有一种 IR 能同时『好优化』和『好执行』** —— 所以它造了一堆能相互转换的 **Dialect(方言)**,
每级方言贴近不同的抽象层次,用 **Pass 管线**逐级降低。下方挑选一个方言看它负责什么,再选几个 pass
拼一条管线,看看"降级"是怎么一步步发生的。
""")

DIALECTS = {
    "tensor": {
        "层次": "高层(语义)",
        "作用": "描述『整个张量』级别的操作,如 tensor.collapse_shape / tensor.extract_slice,不做循环展开",
        "类比": "菜谱里写的『把土豆切丁』 —— 只说做什么,不说怎么切",
        "颜色": "#1f77b4",
    },
    "linalg": {
        "层次": "中高层(结构化)",
        "作用": "以结构化形式描述『逐元素 / 归约 / 矩阵乘』等算子,保留循环结构供后续优化(tiling/fusion)",
        "类比": "菜谱写的『用刀把每个土豆切成 1cm 见方』 —— 已经暗示了循环结构",
        "颜色": "#2ca02c",
    },
    "scf": {
        "层次": "中层(控制流)",
        "作用": "结构化控制流:scf.for / scf.if,负责循环与分支的显式表达",
        "类比": "『重复 10 次:切一个土豆』 —— 循环写出来了",
        "颜色": "#d62728",
    },
    "arith": {
        "层次": "中低层(算术)",
        "作用": "标量算术:arith.addf / arith.mulf,单元素运算",
        "类比": "『把这一小块加那一小块』 —— 从张量降到标量",
        "颜色": "#9467bd",
    },
    "vector": {
        "层次": "低层(向量)",
        "作用": "向量/SIMD 级操作,vector.contract / vector.transfer,贴近 CPU 向量化与 GPU 并行",
        "类比": "『一次处理 8 个土豆』 —— 开始为硬件并行做准备",
        "颜色": "#ff7f0e",
    },
    "llvm": {
        "层次": "最低层(机器)",
        "作用": "LLVM IR Dialect,交给 LLVM 做寄存器分配与指令选择,最终生成机器码",
        "类比": "『用机器指令切土豆』 —— 完全贴近硬件",
        "颜色": "#8c564b",
    },
}

PASSES = {
    "canonicalize": "把算子化简成规范形式(x+0→x、合并嵌套算子),减小后续工作",
    "CSE(公共子表达式消除)": "删掉重复计算:同一个子表达式只算一次",
    "loop-fusion": "合并相邻循环,减少循环开销与中间缓冲",
    "tiling": "把大循环切成小块(tile),提高缓存/寄存器局部性",
    "vectorize": "把标量循环向量化(一次处理多个元素),利用 SIMD 或 GPU 并行",
    "lower-to-llvm": "把高层次方言『降级(lower)』成 LLVM Dialect",
}

st.sidebar.header("🎛️ 参数")
dialect = st.sidebar.selectbox("选择一个 Dialect(方言)", list(DIALECTS.keys()))
show_all_passes = st.sidebar.checkbox("显示完整 pass 管线", value=True)
st.sidebar.caption("MLIR 的优雅之处:每条 pass 只做一件小事,由框架按顺序串成管线。")

info = DIALECTS[dialect]
st.markdown(f"### 当前 Dialect:**{dialect}**  ({info['层次']})")
c1, c2 = st.columns([1, 1])
c1.info(f"**作用**:{info['作用']}")
c2.success(f"**类比**:{info['类比']}")

# ---------------- 抽象层级图 ----------------
st.subheader("⛰️ 多级 IR 的抽象金字塔")
levels = list(DIALECTS.keys())
yvals = list(range(len(levels), 0, -1))[::-1]
fig = go.Figure(go.Bar(x=[1] * len(levels), y=yvals,
                       orientation="v",
                       marker_color=[DIALECTS[k]["颜色"] for k in levels],
                       text=levels, textposition="inside",
                       hovertemplate="%{text}<extra></extra>"))
fig.update_layout(title="从高层语义(down)到低层机器:每级方言一个抽象层次",
                  xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                  height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# 标注当前方言在金字塔的位置
idx = levels.index(dialect)
st.markdown(f"**当前选中**:{dialect}(第 {len(levels) - idx} 层,从底往上是 {idx + 1} 层)")

# ---------------- pass 管线 ----------------
st.subheader("🔧 Pass 管线")
if show_all_passes:
    st.markdown("一条典型的『高到低』pass 管线(可理解为从左到右逐步降级):")
    plist = ["tensor", "canonicalize", "linalg", "tiling", "vectorize", "lower-to-llvm", "llvm"]
    fig2 = go.Figure()
    for i, p in enumerate(plist):
        fig2.add_annotation(x=i, y=0, text=p, showarrow=False,
                            font=dict(size=11, color="white"),
                            bgcolor="#c0392b" if p in PASSES else "#2c3e50",
                            borderpad=6)
        if i < len(plist) - 1:
            fig2.add_annotation(x=i + 0.5, y=0, text="→", showarrow=False, font=dict(size=16))
    fig2.update_xaxes(range=[-0.5, len(plist) - 0.5], showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    fig2.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📚 pass 词条")
for name, desc in PASSES.items():
    st.markdown(f"- **{name}**:{desc}")

st.caption("💡 记不住方言没关系,抓住一条主线即可:『每级方言降一点抽象,每条 pass 做一件小事,串起来就是编译』。")
''')

NB = new_nb("第 64 课 · MLIR 与中间表示:一座『方言之城』",
            subtitle="多级 IR 思想 · Dialect · Pass 管线 —— 与 Triton TTGIR 的关联",
            emoji="🏗️")

chapter_cover(NB,
    objectives=[
        "理解多级 IR 思想:没有一种 IR 能同时『好优化』与『好执行』,所以要分层",
        "掌握 Dialect(方言)概念:每级方言代表一个抽象层次,可相互转换",
        "理解 Pass 管线:每条 pass 做一件小事,按序串起来完成『降级』",
        "用文字 + matplotlib 画出多级 IR、方言、pass 管线的全景",
        "真跑 Triton:编译一个 kernel 并 dump 它的 TTGIR,看到『编译器内部 IR』长什么样",
        "跑通配套 App:浏览 Dialect / Pass 概念",
    ],
    toc=[
        ("直觉:多语种翻译团", "从『人话』到『机器话』要经过好几层翻译,每层一种方言"),
        ("多级 IR 思想", "高层好优化、低层好执行 —— 分级就是兼顾两者"),
        ("Dialect:一座方言之城", "tensor / linalg / scf / arith / vector / llvm 各管一段"),
        ("Pass 管线:串起降级", "canonicalize / CSE / tiling / vectorize / lower 一条线"),
        ("关联 Triton TTGIR", "真跑 Triton,dump TTGIR,看编译器内部 IR 的实物"),
        ("配套 App", "app_64_mlir.py:Dialect / Pass 概念浏览"),
    ],
    links=[
        ("MLIR 论文(2020)", "https://arxiv.org/abs/2002.11054"),
        ("MLIR 官方文档", "https://mlir.llvm.org/"),
        ("Triton 编译器(GPU Dialect)", "https://openai.com/index/triton/"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:多语种翻译团 🌍",
D('''
把一次编译想象成把一个故事从"中文"翻译成"机器能懂的 0/1"。直接翻会丢信息、难优化;
所以编译器像一支**多语种翻译团**,每层换一种"方言",一层层往下翻:

- 第一层(高层方言):贴近语义,好做**融合、化简**等大手术;
- 中间层(中层方言):循环、块(tile)都显式写出来,好做**调度**;
- 最后层(低层方言):贴近硬件指令,好做**代码生成**。

MLIR 的精髓正是:**把"每个抽象层次"本身做成一等公民 —— 一个 Dialect(方言)**。于是你可以
在同一个编译器里无缝混用不同抽象层次,逐级"降低(lower)"。这就像一本多语种对照词典,
随时能把任何一句话翻成下一层的语言。
'''))

NB.md("## 2️⃣ 多级 IR 思想:没有银弹 🎯",
D('''
为什么不能只用一个 IR?因为**没有一种表示能同时满足两个相反目标**:

- **高层 IR**(如 FX/ATen、tensor、linalg):张量级操作,一眼看懂、容易做算子融合,但**离硬件太远**,
  没法直接生成高效代码;
- **低层 IR**(如 LLVM、机器指令):贴着硬件,能生成代码,但**太琐碎**,看不出"这是一个矩阵乘"、
  没法做张量层优化。

所以业界共识:**分层表示**。高层 IR 负责"看得懂 + 做全局优化",逐级降级到低层 IR 负责"跑得快 + 出代码"。
下面画这张多级 IR 的层叠图。
'''))

NB.code(D('''
import matplotlib.pyplot as plt
layers = [
    ("高层 IR", "tensor / ATen / StableHLO", "张量级操作,好做融合/化简", "#dbe7f4", "#1f4e79"),
    ("中层 IR", "linalg / scf / arith", "循环与块显式化,好做调度/tiling", "#fff2cc", "#7f6000"),
    ("低层 IR", "vector / LLVM / PTX", "向量/指令级,好做代码生成", "#fce5cd", "#a64d17"),
]
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")
y = 3
for name, irs, role, fill, edge in layers:
    ax.add_patch(plt.Rectangle((0.2, y), 7.6, 0.85, facecolor=fill, edgecolor=edge, lw=2))
    ax.text(0.5, y + 0.42, f"{name}: {irs}", fontsize=11, fontweight="bold", va="center", color=edge)
    ax.text(7.5, y + 0.42, role, fontsize=9, ha="right", va="center", color="#333")
    if y > 0.8:
        ax.annotate("", xy=(4, y - 0.08), xytext=(4, y + 0.85),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#888"))
        ax.text(4.15, y + 0.38, "lower(降级)", fontsize=8, va="center", color="#888")
    y -= 1.1
ax.set_xlim(0, 8); ax.set_ylim(0, 4.2)
ax.set_title("多级 IR:逐级降低抽象,兼顾『好优化』与『好执行』", fontsize=13)
plt.tight_layout()
'''),
"🎨 每下降一级,抽象降低一层,离硬件更近一步 —— 这就是『多级 IR』的全部要义。")

NB.md("## 3️⃣ Dialect:一座方言之城 🏙️",
D('''
MLIR 里的 **Dialect(方言)** 是"一组算子 + 一组类型"的集合,每个方言代表一个抽象层次。
用 MLIR 内置方言举例:

| Dialect | 层次 | 作用 | 生活类比 |
|---|---|---|---|
| `tensor` | 高层 | 整个张量级的操作(reshape/slice) | 『把土豆切丁』 |
| `linalg` | 中高层 | 结构化算子(matmul/add),保留循环 | 『用刀把每个土豆切成 1cm 见方』 |
| `scf` | 中层 | 结构化控制流(for/if) | 『重复 10 次:切一个土豆』 |
| `arith` | 中低层 | 标量算术(addf/mulf) | 『把这一小块加那一小块』 |
| `vector` | 低层 | SIMD/向量级操作 | 『一次处理 8 个土豆』 |
| `llvm` | 最低 | 交给 LLVM 生成机器码 | 『用机器指令切土豆』 |

下面打印一段"伪 MLIR"文本,演示同一个矩阵乘如何从 `linalg` 一路降到 `arith`:
'''))

NB.code(D('''
mlir_snippet = """\\
// ---- 高层:linalg 方言描述一个矩阵乘 ----
%0 = linalg.matmul ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>)
                         outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>

// ---- 中层:scf 方言把循环写出来(tiling 之后)----
scf.for %i = 0 to 8 step 2 {
  scf.for %j = 0 to 8 step 2 {
    scf.for %k = 0 to 8 step 2 {
      %ai = tensor.extract_slice %a[%i, %k] sizes[2, 2]
      %bj = tensor.extract_slice %b[%k, %j] sizes[2, 2]
      %ci = tensor.extract_slice %c[%i, %j] sizes[2, 2]
      %c1 = linalg.matmul ins(%ai, %bj) outs(%ci)
      // 继续写回 %c ...
    }
  }
}

// ---- 低层:arith 方言 + 向量化 ----
%v = arith.addf %x, %y : vector<8xf32>
"""
print(mlir_snippet)
'''),
"🏷️ 同一个运算,在不同方言里表达从『整块张量』变到『循环+小块』再变到『向量』 —— 抽象逐级降低。")

NB.md("## 4️⃣ Pass 管线:串起一次降级 ⚙️",
D('''
**Pass** 是"对 IR 做的一次独立变换"。MLIR 哲学:**每条 pass 只做一件小事,由框架按顺序串成管线**。
一条典型的高→低管线:

`canonicalize` → `CSE` → `tiling` → `vectorize` → `lower-to-llvm`

- `canonicalize`:化简(x+0→x、合并嵌套算子);
- `CSE`:删除重复子表达式;
- `tiling`:把大循环切块,提高局部性;
- `vectorize`:向量化,为 SIMD/GPU 并行铺路;
- `lower-to-llvm`:最终降级到 LLVM。

下面画一条 pass 管线,并标注每一步改变了什么:
'''))

NB.code(D('''
passes = [
    ("canonicalize", "化简算子", "x+0 → x"),
    ("CSE", "删重复计算", "相同子式算一次"),
    ("tiling", "循环切块", "大循环 → 小块"),
    ("vectorize", "向量化", "标量 → SIMD/向量"),
    ("lower-llvm", "降级 LLVM", "生成机器码"),
]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
x = 0.3
for i, (name, what, effect) in enumerate(passes):
    ax.add_patch(plt.Rectangle((x, 0.6), 1.5, 1.0, facecolor="#eaf2f8", edgecolor="#1f4e79", lw=1.6))
    ax.text(x + 0.75, 1.55, name, ha="center", fontsize=10, fontweight="bold", color="#1f4e79")
    ax.text(x + 0.75, 1.05, what + "\\n" + effect, ha="center", va="center", fontsize=7.5, color="#333")
    if i < len(passes) - 1:
        ax.annotate("", xy=(x + 1.65, 1.1), xytext=(x + 1.55, 1.1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
    x += 1.8
ax.set_xlim(0, x); ax.set_ylim(0, 2.3)
ax.set_title("一条 pass 管线:每条 pass 做一件小事,按序降级", fontsize=12)
plt.tight_layout()
'''),
"🎨 Pass 管线图:从左到右,每条 pass 把 IR 向『更接近硬件』推进一步。")

NB.md("## 5️⃣ 关联 Triton TTGIR:看到编译器内部的 IR 🔬",
D('''
Triton 编译器内部同样用多级 IR,其中 **TTGIR(TritonGPU IR)** 是关键的"方言层":它把 Triton
的 tile 式写法映射到 GPU 的线程/块/共享内存结构 —— 这就是 MLIR 思想在 GPU 编译器的落地。
本机装了 Triton,我们真的编译一个 kernel 并 dump 出它的 TTGIR:
'''))

NB.code(D('''
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(a_ptr + offs) + tl.load(b_ptr + offs))

N = 2048
a = torch.randn(N, device="cuda"); b = torch.randn(N, device="cuda"); o = torch.empty_like(a)
add_kernel[(triton.cdiv(N, 512),)](a, b, o, N, BLOCK=512)
print("kernel 运行结果一致:", torch.allclose(a + b, o))
'''),
"✅ 先真跑一个 Triton kernel(加和),接下来 dump 它编译出的 TTGIR。")

NB.code(D('''
# 用 triton.compile 编译并把 TTGIR 文本 dump 出来
from triton.compiler.compiler import ASTSource
sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32", "BLOCK": "constexpr[i32]"}
src = ASTSource(add_kernel, signature=sig, constexprs={"BLOCK": 512})
kk = triton.compile(src, target=triton.backends.compiler.GPUTarget("cuda", 90, 32))
ttgir = kk.asm.get("ttgir", "")
print("TTGIR 长度:", len(ttgir))
print("---- TTGIR 开头 ----")
print(ttgir[:700])
'''),
"🔍 这就是『编译器内部 IR』的实物:你能看到 `ttg.blocked`、`ttg.num-warps` 这些 GPU 调度方言属性 —— 正是 MLIR 多级 IR 思想在 Triton 里的体现。")

NB.md("## 6️⃣ 配套 App:Dialect / Pass 概念浏览 🎛️",
D('''
同目录的 `app_64_mlir.py` 把概念做成交互:选一个 **Dialect** 看它的作用与生活类比、看它在
抽象金字塔里的位置;勾选 **Pass 管线** 看一条典型的高→低降级线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_64_mlir.py
```

浏览器打开 **http://localhost:8501**。建议从 `tensor` 一路点到 `llvm`,体会"每级降一点抽象"。
完整源码如下:
'''))

NB.code(app_cell("app_64_mlir.py", APP_64),
"📜 运行本 cell 覆盖写入 `app_64_mlir.py`,保证 notebook 与 app 一致。")

wrapup(NB,
    summary=[
        "多级 IR 思想:高层好优化、低层好执行,没有单一 IR 能两头兼顾,所以要分层",
        "Dialect = 一组算子 + 类型 = 一个抽象层次;tensor/linalg/scf/arith/vector/llvm 各管一段",
        "Pass 管线 = 每条 pass 做一件小事,按序串起来完成『降级』",
        "Triton 内部也有多级 IR:TTGIR 把 tile 式写法映射到 GPU 线程/块结构,是 MLIR 思想的落地",
        "真跑 Triton 并 dump TTGIR,让『编译器内部 IR』从概念变成可见的文本",
    ],
    practice=[
        "把 TTGIR 完整打印出来,找出 `ttg.blocked` 里的 sizePerThread / warpsPerCTA 等调度属性",
        "自己写一个更复杂的 Triton kernel(如逐元素 + 归约),看它的 TTGIR 结构如何变化",
        "把『抽象金字塔』图扩展成 5 层,标注每层之间由哪个 pass 完成降级",
        "查一查 torch-mlir 如何把 PyTorch 模型导入 MLIR 的 tensor/linalg 方言,画一条完整链路",
    ],
    links=[
        ("MLIR 论文", "https://arxiv.org/abs/2002.11054"),
        ("MLIR 官方文档", "https://mlir.llvm.org/"),
        ("Triton 编译器", "https://openai.com/index/triton/"),
        ("torch-mlir 项目", "https://github.com/llvm/torch-mlir"),
    ])

out = str(Path(CH10) / "64_mlir_intro.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH10) / "app_64_mlir.py"
app_path.write_text(APP_64 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
