# -*- coding: utf-8 -*-
"""生成 60_triton_oplib.ipynb 与 app_60_triton_oplib.py"""
from helpers import D, CH09, TRITON_HEADER, BENCH, OPLIB_KERNELS, new_nb
from pathlib import Path
from nb_builder import chapter_cover, wrapup

# app 源码直接从同目录文件读取,保证 notebook %%writefile 与磁盘文件一字不差
APP_60_SRC = Path(__file__).parent.parent / "app_60_triton_oplib.py"
APP_60 = APP_60_SRC.read_text(encoding="utf-8").strip()
APP_CELL = "%%writefile app_60_triton_oplib.py\n" + APP_60

NB = new_nb("第 60 课 · 用 Triton 写 mini 算子库",
            subtitle="LayerNorm / GELU / Softmax 的 triton 实现、封装与性能对比,以及 Triton 生态全景",
            emoji="📦")

chapter_cover(NB,
    objectives=[
        "用 triton 实现 LayerNorm / GELU / Softmax 三个经典算子",
        "把 kernel 封装成可复用的 Python API(像 vLLM 的 ops 目录那样)",
        "与 torch 原生算子做性能对比,理解 triton 的适用场景与边界",
        "认识 Triton 生态:triton-lang / triton-mosaic / OpenAI 与 CUDA/CUTLASS 的关系",
    ],
    toc=[
        ("直觉:算子库是“积木箱”", "写一次 kernel,封装成函数,处处调用"),
        ("LayerNorm:一行一次统计", "mean / var / rstd 一次扫完,乘 γ 加 β"),
        ("GELU:精确 vs 近似", "erf 版精确,tanh 版更快,逐元素无数据依赖"),
        ("Softmax:减 max 的威力", "数值稳定一行搞定,数据沿行内复用"),
        ("封装:一个 mini 算子库的 API 设计", "注册表 + 统一签名 + 后端可替换"),
        ("性能对比:bar chart 见真章", "访存受限算子上,triton vs torch 各说各话"),
        ("融合的艺术:layer_norm + gelu", "一个 kernel 一次读写,vs 两个 kernel 两次读写"),
        ("Triton 生态全景", "triton-lang / triton-mosaic / OpenAI 生态与 CUTLASS 的关系"),
        ("配套 Streamlit 演示", "app_60_triton_oplib.py:算子选择器对比"),
    ],
    links=[
        ("Triton 官方教程:矩阵乘(进阶)", "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html"),
        ("triton-mosaic(Intel 维护的 Triton fork)", "https://github.com/intel/triton-mosaic"),
        ("OpenAI Triton 介绍博客", "https://openai.com/research/triton"),
        ("CUTLASS(NVIDIA 手写 CUDA 模板库)", "https://github.com/NVIDIA/cutlass"),
    ])

NB.md("## 1️⃣ 直觉:算子库是“积木箱” 🧰",
D('''
把注意力 kernel、GEMM、归一化算子看成**积木**:写一次、封装好、到处复用——这就是**算子库
(oplib)**。vLLM 自己也有一个 ops 目录(`vllm/v1/attention/ops/`),里面全是 triton 写的
小 kernel(`triton_prefill_attention.py`、`triton_reshape_and_cache_flash.py`……),
按需拼装成完整的注意力后端。

本课我们用 Triton 实现三个“积木”:**LayerNorm、GELU、Softmax**,把它们封装成统一接口,
与 torch 原生实现做性能对比,最后聊聊 Triton 生态。你会发现一个反直觉的结论:
**对简单的访存受限算子,triton 未必比 torch 快**——triton 的真正价值在于“融合”和“定制”。
'''))

NB.code(TRITON_HEADER, "✅ 环境自检:本课三个算子都要真实编译运行。")

NB.md("## 2️⃣ LayerNorm:一行一次统计 📐",
D('''
LayerNorm 对每一行做归一化:

$$y_i = \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}\\cdot\\gamma_i + \\beta_i,
\\quad \\mu=\\frac{1}{N}\\sum_j x_j,\\ \\sigma^2=\\frac{1}{N}\\sum_j(x_j-\\mu)^2$$

它天然适合“一行一个 program”:整行读进来,块内两次归约(mean、var),最后乘 γ 加 β。
注意 mask 行的处理:读入时把无效列置 0,`tl.where` 后再参与统计,避免污染 mean/var。
'''))

NB.code(OPLIB_KERNELS, "📐 三个 kernel 一并给出:LayerNorm / GELU / Softmax。每个都是“小、独立、可复用”的积木。")

NB.code(D('''
M, N = 512, 256
x = torch.randn(M, N, device="cuda")
w = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")

y_tri = layer_norm_triton(x, w, b)
y_ref = torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5)
print("LayerNorm max err =", (y_tri - y_ref).abs().max().item())
'''), "✅ 与 torch 的 LayerNorm 对拍,误差应 < 1e-5(fp32)。")

NB.md("## 3️⃣ GELU:精确 vs 近似 📈",
D('''
GELU(Gaussian Error Linear Unit)是 Transformer 里常用的激活:

$$\\text{GELU}(x) = 0.5x\\Big(1 + \\text{erf}\\big(x/\\sqrt{2}\\big)\\Big)$$

- **精确版**用 `tl.math.erf`(我们实现的就是它),一步到位;
- **近似版**(`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`)不用 erf,某些平台上更快。

GELU 是逐元素算子,**没有任何跨元素依赖**,是 triton 最容易写的一类——但也意味着
它几乎纯访存受限。对拍一下:
'''))

NB.code(D('''
x = torch.randn(M * N, device="cuda")
y_tri = gelu_triton(x)
y_ref = torch.nn.functional.gelu(x)
print("GELU max err =", (y_tri - y_ref).abs().max().item())
'''), "✅ 误差 ~1e-7(fp32 内 erf 实现略有差异)。")

NB.md("## 4️⃣ Softmax:减 max 的威力 🌊",
D('''
Softmax 每行是“跨元素”运算:先减 max 保证 `exp` 不溢出,再求和归一。triton 一行内用
两次归约(`tl.max`、`tl.sum`)完成。注意第 43 课讲的 online softmax 是“流式”版本,
为了 FlashAttention 不落盘 S/P;这里行内一次读入,普通两遍版就够了。
'''))

NB.code(D('''
x = torch.randn(M, N, device="cuda")
y_tri = softmax_triton(x)
y_ref = torch.softmax(x, dim=-1)
print("Softmax max err =", (y_tri - y_ref).abs().max().item())
'''), "✅ 误差 ~1e-8,数值稳定(先减 max)。")

NB.md("## 5️⃣ 封装:一个 mini 算子库的 API 设计 🗂️",
D('''
三个 kernel 都裸露着指针参数——直接给用户用太丑了。真正的算子库要解决两件事:

1. **统一签名**:`fn(x, ...) -> y`,内部负责 dtype/device/连续性/铺 grid;
2. **后端可替换**:同一个名字,可以指 triton 版,也可以指 torch 版,接口不变。

一个迷你注册表长这样(和 vLLM 的 ops 目录思路一致):
'''))

NB.code(D('''
# 统一后端接口:注册表 + 可切换实现
OPS = {
    "layernorm": lambda x, w, b: layer_norm_triton(x, w, b),
    "gelu":      lambda x: gelu_triton(x),
    "softmax":   lambda x: softmax_triton(x),
}
OPS_TORCH = {
    "layernorm": lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, 1e-5),
    "gelu":      lambda x: torch.nn.functional.gelu(x),
    "softmax":   lambda x: torch.softmax(x, dim=-1),
}

# 调用方只看接口,不关心实现
def run_op(name, backend="triton", *args):
    table = OPS if backend == "triton" else OPS_TORCH
    return table[name](*args)

print("triton layernorm:", run_op("layernorm", "triton", x, w, b).shape)
print("torch   softmax :", run_op("softmax", "torch", x).shape)
print("triton  gelu    :", run_op("gelu", "triton", x).shape)
'''), "🗂️ 换 `backend` 一个词,实现就换了——这就是“算子库”的接口契约。")

NB.md("## 6️⃣ 性能对比:bar chart 见真章 📊",
D('''
现在把三个算子、两套实现、三种规模摆在一起对比。用第 57 课的 `bench`(warmup + synchronize
+ 多次平均)计时,画成 seaborn 分组柱状图:
'''))

NB.code(BENCH, "⏱️ 复用第 57 课的计时函数。")

NB.code(D('''
sizes = [256, 1024, 4096]
rows_n = 256
results = []
for N in sizes:
    xx = torch.randn(rows_n, N, device="cuda")
    ww = torch.randn(N, device="cuda"); bb = torch.randn(N, device="cuda")
    for name, f_tri, f_tor in [
        ("LayerNorm",
         lambda: layer_norm_triton(xx, ww, bb),
         lambda: torch.nn.functional.layer_norm(xx, (N,), ww, bb, 1e-5)),
        ("GELU",
         lambda: gelu_triton(xx),
         lambda: torch.nn.functional.gelu(xx)),
        ("Softmax",
         lambda: softmax_triton(xx),
         lambda: torch.softmax(xx, dim=-1)),
    ]:
        results.append((name, N, "triton", bench(f_tri)))
        results.append((name, N, "torch", bench(f_tor)))

import pandas as pd
df = pd.DataFrame(results, columns=["算子", "N", "实现", "耗时(ms)"])
print(df.pivot_table(index="算子", columns="实现", aggfunc="mean").round(4).to_string())
'''), "⏱️ 先看数字:同一算子、同一规模下 triton 与 torch 的耗时是否接近。")

NB.code(D('''
fig, ax = plt.subplots(figsize=(10, 4.2))
sns.barplot(data=df, x="N", y="耗时(ms)", hue="实现", ax=ax, palette=["#4C78A8", "#E45756"])
ax.set_title("三个访存受限算子:triton vs torch(rows=256)")
plt.tight_layout()
'''), "🎨 结论:**同一量级**。这类算子是 memory-bound,torch 的元素级 kernel 已经写得很好;triton 不输太多,但也很难赢——这正是它“不擅长”的战场。")

NB.md("## 7️⃣ 融合的艺术:layer_norm + gelu 🔥",
D('''
那 triton 的价值在哪?**融合(fusion)**。把两个串行算子合并进一个 kernel:数据**只读一次、只写一次**,
中间的均值/方差/激活都在寄存器里流转,不落全局内存。对一个 256×4096 的张量,省下的是
**两次全量读写**(≈ 8MB×2 的 HBM 流量)。

我们写一个 `fused_layer_norm_gelu`:读入一行 → LayerNorm → GELU → 写回。
'''))

NB.code(D('''
@triton.jit
def fused_ln_gelu_fwd(x_ptr, y_ptr, w_ptr, b_ptr, eps, M, N, row_stride,
                      BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    x = (xc * (1.0 / tl.sqrt(var + eps))) * tl.load(w_ptr + offs, mask=mask, other=0.0) \
        + tl.load(b_ptr + offs, mask=mask, other=0.0)
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(y_ptr + row * row_stride + offs, x, mask=mask)


def fused_ln_gelu(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    fused_ln_gelu_fwd[(M,)](x, y, w, b, eps, M, N, N, BLOCK_N=triton.next_power_of_2(N))
    return y

x = torch.randn(512, 8192, device="cuda")
w = torch.randn(8192, device="cuda"); b = torch.randn(8192, device="cuda")
y_fused = fused_ln_gelu(x, w, b)
y_ref = torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, (8192,), w, b, 1e-5))
print("融合版 max err =", (y_fused - y_ref).abs().max().item())

t_fused = bench(lambda: fused_ln_gelu(x, w, b))
t_two = bench(lambda: torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, (8192,), w, b, 1e-5)))
print(f"融合版    : {t_fused:.4f} ms")
print(f"串行两算子: {t_two:.4f} ms")
print(f"融合加速比: {t_two / t_fused:.2f}x")
'''), "🔥 融合版因为省了两次 HBM 往返,规模越大收益越明显(本机 512×8192 通常 2x 上下,256×4096 也有 1.3x+)。这才是 triton 的杀手锏场景。")

NB.md("## 8️⃣ Triton 生态全景 🌍",
D('''
最后把 Triton 放进它所在的生态看:

| 名字 | 谁维护 | 是什么 | 与我们的关系 |
|------|--------|--------|--------------|
| **triton-lang** | OpenAI / 社区 | 本课用的编译器,Python→GPU 的自动调度 | 主战场 |
| **triton-mosaic** | Intel | triton 的 Intel 分支,面向 XPU/Arc GPU | 多硬件适配 |
| **OpenAI Triton 内核库** | OpenAI | 官方维护的常用 kernel 集合 | 参考实现 |
| **CUTLASS / cuBLAS** | NVIDIA | 手写 CUDA 模板库 / 闭源库 | 性能上限标杆 |
| **TVM / MLIR** | Apache / LLVM | 通用编译器框架 | Triton 的底层伙伴 |

生态一句话:**Triton 站在 MLIR 之上,服务 GPU 加速计算;和手写 CUDA 相比它牺牲一点峰值,
换取“会 Python 就能写 kernel”的生产力**。vLLM 正是看重这一点,才把注意力核心用 Triton 实现。
'''))

NB.code(D('''
# 生态关系示意图
fig, ax = plt.subplots(figsize=(9, 3.6))
boxes = [
    (0.02, "triton-lang\\n(OpenAI 核心)", "#4C78A8"),
    (0.26, "triton-mosaic\\n(Intel 分支)", "#72B7B2"),
    (0.50, "OpenAI 内核库\\n(参考 kernel 集)", "#F28E2B"),
    (0.74, "CUTLASS / cuBLAS\\n(NVIDIA 手写)", "#E45756"),
]
for x, label, color in boxes:
    ax.add_patch(plt.Rectangle((x, 0.25), 0.2, 0.5, color=color, alpha=0.8, ec="k"))
    ax.text(x + 0.1, 0.5, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
ax.text(0.5, 0.92, "都站在 MLIR / LLVM 之上,目标都是“更快地算矩阵”", ha="center", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.axis("off")
plt.tight_layout()
'''), "🎨 生态图:同一个目标(高性能 GPU 计算),不同的取舍。vLLM 选了最“可维护”的一条。")

NB.md("## 9️⃣ 配套 Streamlit 演示:算子选择器对比 🎛️",
D('''
运行同目录下的 `app_60_triton_oplib.py`,**选择算子、拖动规模、切换测量对象**,即可看到
triton vs torch 的耗时柱状图、吞吐与对拍误差,以及“扫规模”曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_60_triton_oplib.py
```

浏览器打开 **http://localhost:8501**。完整源码如下(与同目录 `app_60_triton_oplib.py` 一字不差):
'''))

NB.code(APP_CELL, "📜 app_60_triton_oplib.py 完整源码:notebook 与 app 共用同一套 triton 算子实现。")

wrapup(NB,
    summary=[
        "三个经典算子都能用 triton 写:LayerNorm(行内归约)、GELU(逐元素 erf)、Softmax(减 max 归一)",
        "封装成统一接口(OPS 注册表 + backend 参数)后,调用方完全不用关心实现细节",
        "对 LayerNorm/GELU/Softmax 这类 memory-bound 算子,triton 与 torch 同量级——不必硬刚",
        "triton 的杀手锏是融合:LayerNorm+GELU 一次读写,省两次 HBM 往返,规模越大越快(本机实测 1.3x~4x)",
        "生态上:trition-lang(核心)、triton-mosaic(Intel)、OpenAI 内核库、CUTLASS/cuBLAS(标杆)各司其职",
    ],
    practice=[
        "给 OPS 注册表加一个 fused_ln_gelu,并验证它与 run_op('gelu', 'layernorm') 数值一致",
        "把 fused_ln_gelu 的 BLOCK_N 换成 2 的幂 + mask 兜底,支持任意 N(非 2 的幂)",
        "用第 57 课的 matmul 思路,把 Softmax 推广成“多行多 program”的版本,对比单 program 性能",
        "调研一个真实 vLLM triton kernel(triton_reshape_and_cache_flash.py),说出它的输入/输出签名",
    ],
    links=[
        ("OpenAI Triton 官方教程(10 个实战 kernel)", "https://triton-lang.org/main/getting-started/tutorials/index.html"),
        ("triton-mosaic(Intel)", "https://github.com/intel/triton-mosaic"),
        ("vLLM 的 triton ops 目录", "https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/ops"),
    ])

NB.save(str(Path(CH09) / "60_triton_oplib.ipynb"))

app_path = Path(CH09) / "app_60_triton_oplib.py"
app_path.write_text(APP_60 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
