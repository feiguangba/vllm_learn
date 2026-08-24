# -*- coding: utf-8 -*-
"""生成 26_torch_compile.ipynb(app_26_compile.py 已存在,%%writefile 覆盖写入相同内容)"""
from helpers import D, MINI_GPT, chapter_cover, wrapup, new_nb, CH04, app_src, finalize


def app_cell(name):
    return "%%writefile " + name + "\n" + app_src(name)


NB = new_nb("第 26 课 · torch.compile:把一串小算子冻成一颗大 kernel",
            subtitle="eager 每个 aten 算子各开一个 kernel;compile 把相邻算子融合 —— 少开火、少搬运",
            emoji="🧊")

chapter_cover(NB,
    objectives=[
        "理解 eager 模式的痛点:每个算子一次派发、一个 kernel、中间结果落回显存",
        "认识 torch.compile 两层架构:Dynamo 追踪(FX 图)+ Inductor 生成融合 kernel",
        "用 TorchDispatchMode 实测一块迷你 MLP 的算子个数,看融合能砍掉多少",
        "在本机真跑 torch.compile:Windows 缺 MSVC 时 Inductor 会怎样,aot_eager 又是什么",
        "掌握 vLLM 的 CompilationMode:NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE / VLLM_COMPILE",
        "跑通配套 App,交互查看 kernel 削减率与融合示意图",
    ],
    toc=[
        ("直觉:一锅炖", "eager 是每道工序单独开火,compile 把能合并的工序并成一锅"),
        ("两层架构:Dynamo + Inductor", "追踪 Python 得到 FX 图 → 调度融合 → 生成 Triton/C++ kernel"),
        ("实测:eager 到底派发多少算子", "TorchDispatchMode 当计数器,MLP 块 9 个算子现形"),
        ("本机真跑 compile", "Inductor 缺 MSVC 报错的现场教学;aot_eager 追踪成功但无融合"),
        ("融合收益模型与可视化", "kernel 数对比 + 时间估算,pyecharts/plotly 双图"),
        ("vLLM CompilationMode", "四档编译模式的取舍,与 CUDA Graph 双管齐下"),
        ("保存数据给 App", "生成 compile_measure_26.json"),
        ("配套 App:🧊 compile 与融合", "streamlit 交互演示"),
    ],
    links=[
        ("PyTorch torch.compiler 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("torch.compile 教程", "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.md("## 1. 直觉:一锅炖 🍲",
D('''
想象做一道菜要 9 道工序:切葱、热油、下葱、下蛋、翻炒、加盐、装盘……(eager 模式)每道工序
都**单独开一次火**:开火有固定成本(第 24 课的 launch 开销),每道工序的半成品还要**先端回冰箱
再取出来**(中间张量写回显存再读出来)。

`torch.compile` 像一位大厨 🧑‍🍳:**先看完整个菜谱(Dynamo 追踪整段计算),重排并合并工序
(Inductor 调度)**,把「下蛋→翻炒→加盐」这类相邻工序**一锅炖成一道** —— 这就是
**kernel 融合(fusion)**。收益有三:

1. **kernel 数变少** → 启动开销变少(第 24 课的手续费);
2. **中间结果留在寄存器/缓存** → 不再反复写显存读显存(访存往往是小算子的真瓶颈);
3. **有机会做全局优化**:融合后的代码可向量化、可重排。

vLLM 里 decode 阶段张量小、算子碎 —— 与 CUDA Graph 一样,正是 compile 的主场。
'''))

NB.md("## 2. 两层架构:Dynamo + Inductor 🏗️",
D('''
`torch.compile(model)` 背后是两层流水线:

| 层 | 组件 | 干什么 |
|---|---|---|
| 第 1 层 | **TorchDynamo** | 捕获 Python 字节码,追踪成 **FX Graph**(算子级计算图);带 guard,输入性质变了就重新追踪 |
| 第 2 层 | **TorchInductor** | 拿 FX 图做**后端编译**:算子调度、融合分组,生成 **Triton kernel**(GPU)或 C++ 代码(CPU) |

关键理解:**融合发生在 Inductor**。只走第 1 层(backend="eager"/"aot_eager")只能得到图,
不会变快 —— 但它让 vLLM 这类下游框架能拿到图自己后处理(下一节的 vLLM_COMPILE 就这么玩)。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch OMP 库冲突(Windows)
import time, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
torch.manual_seed(0)
print("torch =", torch.__version__, "| 本课在 CPU 上实验:dispatch 计数与计时都是真跑,融合收益用模型估算")
'''),
"🧊 说明:本机 Windows 无 MSVC 编译器,Inductor 的 CPU 后端无法生成代码 —— 我们把它变成一堂「现场教学课」。")

NB.code(MINI_GPT,
"🏗️ 迷你 GPT(与第 22 课同一台)。本课解剖它的 MLP 块与注意力块,数一数 eager 的算子。")

NB.md("## 3. 实测:eager 到底派发多少算子 🔬",
D('''
`TorchDispatchMode` 是 torch 的「算子拦截器」:任何经过派发层的 aten 调用都会先路过它 ——
拿它当**计数器**,eager 的隐藏成本立刻现形。
'''))

NB.code(D('''
from torch.utils._python_dispatch import TorchDispatchMode

class OpCounter(TorchDispatchMode):
    """派发层计数器:数一数一次前向到底派发了多少个 aten 算子"""
    def __init__(self):
        super().__init__()
        self.count = 0
        self.names = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        self.names.append(str(func).replace("aten.", ""))
        return func(*args, **(kwargs or {}))


model = MiniGPT(vocab=5000, hidden=256, n_layers=2, n_heads=4).eval()
attn, mlp = model.layers[0]
T, H = 8, 256
x = torch.randn(T, H)

with torch.no_grad(), OpCounter() as c_mlp:
    mlp(x)
with torch.no_grad(), OpCounter() as c_attn:
    attn(x)
ids, pos = torch.randint(0, 5000, (T,)), torch.arange(T)
with torch.no_grad(), OpCounter() as c_all:
    model(ids, pos)

print(f"MLP 块一次前向   : {c_mlp.count} 个算子  {c_mlp.names}")
print(f"注意力块一次前向 : {c_attn.count} 个算子  {c_attn.names[:8]} ...")
print(f"整模型一次前向   : {c_all.count} 个算子(2 层 = 2×(MLP+注意力)+ embedding/lm_head 等)")
'''),
"🔬 一层就几十个算子;真实模型几十层、几千个算子 —— eager 的「每算子一开火」在大模型上是重税。")

NB.md("## 4. 本机真跑 compile:两种命运 ⚠️",
D('''
现在真的调用 `torch.compile`。本机的命运我们先看默认后端(Inductor),再看 `aot_eager`:
'''))

NB.code(D('''
# 尝试 1:默认 Inductor 后端(GPU 上生成 Triton kernel,CPU 上生成 C++ 代码)
try:
    mlp_compiled = torch.compile(mlp)
    with torch.no_grad():
        mlp_compiled(x)
    print("✅ Inductor 编译成功(本机有 C++ 工具链)")
    inductor_ok = True
except Exception as e:
    inductor_ok = False
    print("⚠️ Inductor 编译失败 —— 现场教学:")
    print("   错误类型:", type(e).__name__)
    print("   错误信息:", str(e).splitlines()[0][:100])
    print("   原因:CPU 后端要现场编译生成的 C++ 代码,需要 MSVC 编译器 cl.exe,本机没装;")
    print("   GPU 后端则需要 Triton(Windows 支持有限)。Linux + GPU 环境两者都开箱即用。")
'''),
"⚠️ 这不是 torch 的 bug,而是 Windows 的工具链现实 —— 也是很多教程在 Windows 上翻车的地方。")

NB.code(D('''
# 尝试 2:aot_eager 后端 —— 只走 Dynamo/AOTAutograd 追踪,不做 Inductor 代码生成
t0 = time.perf_counter()
mlp_aot = torch.compile(mlp, backend="aot_eager")
with torch.no_grad():
    out_aot = mlp_aot(x)
trace_s = time.perf_counter() - t0

with torch.no_grad(), OpCounter() as c_aot:
    mlp_aot(x)

def ms_per_call(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3

with torch.no_grad():
    e_ms, a_ms = ms_per_call(lambda: mlp(x)), ms_per_call(lambda: mlp_aot(x))

print(f"aot_eager 追踪+首跑耗时 {trace_s:.2f} s(一次性)")
print(f"算子数:eager {c_mlp.count} 个 vs aot_eager {c_aot.count} 个 —— 追踪不融合,数量不变")
print(f"耗时  :eager {e_ms:.4f} ms vs aot_eager {a_ms:.4f} ms(比值 {e_ms/a_ms:.2f}x,≈1)")
print("✅ 结论:Dynamo 拿到了图,但没有 Inductor 就没有融合提速 —— 这正是 vLLM 的 DYNAMO_TRACE_ONCE 档的定位")
'''),
"🔍 aot_eager 的价值不在速度,而在「把图交出来」:vLLM 用它导出计算图,自己做算子级后处理。")

NB.md("## 5. 融合收益模型与可视化 📊",
D('''
没有 Inductor,融合收益算不出来吗?可以**模型估算**(思路与第 24 课一致):

$$t_{\\text{compiled}} \\approx t_{\\text{eager}} - (n_{\\text{eager}} - n_{\\text{compiled}}) \\times t_l$$

其中 $t_l$ 是每次派发/启动的手续费(本机实测),kernel 数按典型融合规则估算:
**矩阵乘保持独立,相邻的逐元素算子(SiLU·mul·add、norm 相关)各融合成一个**。
在 Linux + GPU 上跑同一实验即可得到真实数字(融合段还能省显存往返,实际收益更高)。
'''))

NB.code(D('''
def us_per_call(fn, n=4000, warm=400):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

tiny, bigT = torch.ones(16), torch.ones(16000)
launch_us = max(us_per_call(lambda: tiny.add_(1.0)) - 16 * us_per_call(lambda: bigT.add_(1.0)) / 16000, 0.0)
print(f"本机每次派发手续费 t_l ≈ {launch_us:.2f} µs")


def estimate_fused(names, eager_ms):
    """典型融合规则:matmul/SDPA 等重算子各占 1 个 kernel;相邻的轻量算子串各融成 1 个 kernel"""
    matmul_like = ("mm", "bmm", "addmm", "matmul", "baddbmm", "scaled_dot_product")
    kernels, run = 0, 0
    for n in names:
        if any(m in n for m in matmul_like):
            kernels += 1
            run = 0
        else:
            run += 1
            if run == 1:
                kernels += 1
    compiled_ms = max(eager_ms - (len(names) - kernels) * launch_us / 1e3, eager_ms * 0.3)
    return kernels, compiled_ms

with torch.no_grad():
    attn_ms = ms_per_call(lambda: attn(x))

groups = []
for label, cnt, ms in [("门控 MLP 块(norm+gate/up+SiLU 门控+down+残差)", c_mlp, e_ms),
                       ("LayerNorm+线性(QKV 投影)", OpCounter(), None),
                       ("注意力块(norm+QKV+分头+SDPA+输出投影)", c_attn, attn_ms)]:
    if ms is None:
        y = attn.norm(x)
        with torch.no_grad(), OpCounter() as c_qkv:
            attn.qkv(y)
        cnt = c_qkv
        ms = ms_per_call(lambda: attn.qkv(y))
    k_c, t_c = estimate_fused(cnt.names, ms)
    groups.append(dict(op=label, eager_kernels=cnt.count, compiled_kernels=k_c,
                       eager_ms=round(ms, 4), compiled_ms=round(t_c, 4)))

gdf = pd.DataFrame(groups)
gdf["kernel 削减率"] = (1 - gdf["compiled_kernels"] / gdf["eager_kernels"]).map(lambda v: f"{v*100:.0f}%")
gdf["估算加速比"] = (gdf["eager_ms"] / gdf["compiled_ms"]).map(lambda v: f"{v:.2f}x")
gdf
'''),
"📊 削减率 30~60% 是典型量级;GPU 上融合还省显存往返,实测加速常更高。诚实标注:compiled 侧为模型估算。")

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

xlabels = ["门控 MLP 块", "LayerNorm+线性", "注意力块"]
bar = (
    Bar()
    .add_xaxis(xlabels)
    .add_yaxis("eager kernel 数", [int(v) for v in gdf["eager_kernels"]], label_opts=opts.LabelOpts(position="inside"))
    .add_yaxis("融合后(估算)", [int(v) for v in gdf["compiled_kernels"]], label_opts=opts.LabelOpts(position="inside"))
    .set_global_opts(title_opts=opts.TitleOpts(title="kernel 融合:算子数对比(eager 实测 / 融合估算)"),
                     yaxis_opts=opts.AxisOpts(name="kernel 数"))
)
bar.render_notebook()
'''),
"🧊 被冻掉的每一个 kernel,都是一次不再支付的「开火手续费」和一次不再发生的显存往返。")

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.express as px

tdf = gdf.melt(id_vars=["op"], value_vars=["eager_ms", "compiled_ms"],
               var_name="方案", value_name="单步耗时(ms)")
tdf["方案"] = tdf["方案"].map({"eager_ms": "eager", "compiled_ms": "compile(估算)"})
fig = px.bar(tdf, x="op", y="单步耗时(ms)", color="方案", barmode="group",
             color_discrete_map={"eager": "#c0392b", "compile(估算)": "#27ae60"})
fig.update_layout(title="单步耗时对比(本机 CPU 实测 eager / 融合收益模型估算)",
                  height=380, xaxis_title="")
fig
'''),
"📈 收益来源 = 少开的火 × 手续费 + 省掉的显存往返(后者 GPU 上往往更大)。")

NB.code(D('''
# 保存给配套 App(它会自动加载 compile_measure_26.json;启动时显示「模型估算」的诚实说明)
payload = {"ops": [dict(op=g["op"], eager_kernels=int(g["eager_kernels"]),
                        compiled_kernels=int(g["compiled_kernels"]),
                        eager_ms=float(g["eager_ms"]), compiled_ms=float(g["compiled_ms"]))
                   for _, g in gdf.iterrows()]}
with open("compile_measure_26.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("已保存 compile_measure_26.json")
'''),
"💾 eager 侧(kernel 数、耗时)为本机实测;compiled 侧为融合规则估算 —— 文件里两组数字同源同义。")

NB.md("## 6. vLLM 的 CompilationMode 🏷️",
D('''
vLLM 把「编译到什么程度」做成四档配置(`vllm/config/compilation.py` 的 `CompilationMode`,配合
`-O0`~`-O3` 启动参数):

| 模式 | 干什么 | 适合谁 |
|---|---|---|
| `NONE`(-O0) | 完全 eager,不编译 | 调试;不想付任何编译启动时间 |
| `STOCK_TORCH_COMPILE`(-O1) | 直接用原版 `torch.compile`(Dynamo+Inductor 全套) | 想开箱获益、模型未被 vLLM 特判 |
| `DYNAMO_TRACE_ONCE` | 只用 Dynamo 追踪一次导出 FX 图,**不做** Inductor 编译 | vLLM 自己拿图做后处理(配 CUDA Graph 用,避免重复追踪开销) |
| `VLLM_COMPILE`(-O3) | vLLM 自研编译器:按 attention 算子把图**分段(piecewise)**,段间自定义算子保持不变,段内融合优化 | 追求极致性能的正式服务 |

配合关系:`VLLM_COMPILE` 把**逐元素段**融合成大 kernel,再把这些段包进 **CUDA Graph**(第 25 课)
—— 融合减少 kernel 数,CUDA Graph 抹掉剩余 kernel 的启动开销,**双管齐下**。

⚠️ 诚实说明:本练习环境的 `vendor/vllm` 目录尚未就绪,以上基于 vLLM 公开源码与文档;
配好 vendor 后可打开 `vllm/config/compilation.py` 核对各档定义。
'''))

NB.md("## 7. 配套 App:🧊 torch.compile 与 Kernel 融合 🎛️",
D('''
同目录的 `app_26_compile.py` 把对比做成交互演示:选择算子组合、切换「kernel 数量 / 单步耗时」指标,
勾选「融合示意图」还能看 eager 的 kernel 长龙如何被并成几截;启动时自动加载 `compile_measure_26.json`。

**运行方法**(在 `ch04` 目录执行):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_26_compile.py
```

浏览器打开 **http://localhost:8501**(也可加 `--server.port 8626` 换端口)。
下面这个 cell 会把 app 源码原样写入 `app_26_compile.py`:
'''))

NB.code(app_cell("app_26_compile.py"),
"📜 运行后覆盖写入相同内容,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "eager 痛点:每个 aten 算子一次派发一个 kernel,中间结果反复进出显存",
        "torch.compile 两层:Dynamo 追踪出 FX 图,Inductor 调度融合并生成 Triton/C++ kernel",
        "TorchDispatchMode 实测:一块门控 MLP 9 个算子、整模型前向上百个 —— eager 重税现形",
        "Windows 无 MSVC 时 Inductor 编译失败;aot_eager 可追踪但无融合(≈ vLLM 的 DYNAMO_TRACE_ONCE 定位)",
        "vLLM CompilationMode 四档:NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE / VLLM_COMPILE,与 CUDA Graph 双管齐下",
    ],
    practice=[
        "把 OpCounter 套到整模型前向,统计最多的 3 个算子是什么,想想哪些能融合",
        "修改 estimate_fused 的融合规则(比如把 LayerNorm 也并进相邻 matmul 的 epilogue),看削减率上限",
        "在 Linux + GPU 机器上重跑第 4 节:Inductor 成功后对比真实 kernel 数与耗时,和估算差多少",
        "解释:为什么 vLLM 的 piecewise 编译要把 attention 算子留在段间不动?(提示:PagedAttention 是自定义 CUDA 算子)",
    ],
    links=[
        ("PyTorch torch.compiler 文档", "https://pytorch.org/docs/stable/torch.compiler.html"),
        ("torch.compile 教程", "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html"),
        ("Triton 编译器", "https://openai.com/index/triton/"),
    ])

from pathlib import Path
out = str(Path(CH04) / "26_torch_compile.ipynb")
NB.save(out)
finalize(out)
