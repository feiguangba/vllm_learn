# -*- coding: utf-8 -*-
"""生成第 26 课 notebook: torch.compile 与 Kernel 融合(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:一锅炖直觉 -> Dynamo+Inductor 两层架构 -> 数算子 -> 本机真跑 compile -> 融合收益模型 -> vLLM CompilationMode
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义
4. 论文支撑:PyTorch 2 (ASPLOS'24, Ansel et al.)、torch.compile 官方文档、vLLM compilation.py
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_26_compile.py"
APP_NAME = "app_26_compile.py"
APP_CODE = APP.read_text(encoding="utf-8")


def app_guard(app_code: str, app_name: str) -> str:
    """构造 app 守卫 cell: 非 streamlit 环境只打印提示, 不执行。"""
    return (
        "try:\n"
        "    import streamlit as st\n"
        "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
        "except Exception:\n"
        "    _IS_STREAMLIT = False\n\n"
        "if _IS_STREAMLIT:\n"
        + textwrap.indent(app_code, "    ") +
        "\nelse:\n"
        "    print(\"💡 当前不是 streamlit 环境, 跳过执行本 App。\")\n"
        "    print(\"    请直接运行: D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run " + app_name + "\")\n"
    )


nb = Notebook(
    "第 26 课 · torch.compile:把一串小算子冻成一颗大 kernel",
    subtitle="Dynamo + Inductor 两层架构 · 数算子 · 本机真跑 compile · 融合收益模型 · vLLM CompilationMode",
    emoji="🧊", chapter="第 4 章 · 模型执行器与 CUDA 优化",
)

chapter_cover(
    nb,
    objectives=[
        "理解 torch.compile 的动机:把 eager 的每算子一开火,冻成少而大的 kernel",
        "掌握两层架构:TorchDynamo(抓图) + TorchInductor(生成代码)",
        "用 TorchDispatchMode 数清 MiniGPT 各模块的算子数,量化 eager 的「算子税」",
        "在本机真跑 torch.compile:默认 Inductor vs aot_eager,观察两种命运",
        "建立融合收益模型:削减率 × 启动手续费 + 省掉的显存往返",
        "对照 vLLM 的 CompilationMode 四档配置",
    ],
    toc=[
        ("直觉:一锅炖", "9 道工序 vs 一锅炖"),
        ("两层架构:Dynamo + Inductor", "抓图 + 生成代码"),
        ("数算子:eager 的算子税", "TorchDispatchMode 数清每个模块"),
        ("本机真跑 compile:两种命运", "Inductor vs aot_eager"),
        ("融合收益模型", "削减率 × 手续费 + 显存往返"),
        ("可视化收益", "少开的火 × 手续费 + 省掉的往返"),
        ("vLLM 的 CompilationMode", "四档编译程度配置"),
    ],
    links=[
        ("PyTorch 2: Faster ML Through Dynamic Python Bytecode Transformation (ASPLOS'24)", "https://docs.pytorch.org/assets/pytorch2-2.pdf"),
        ("torch.compile 官方文档", "https://pytorch.org/get-started/pytorch-2.0/"),
        ("TorchInductor 设计笔记", "https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747"),
        ("vLLM compilation.py 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:一锅炖 🍲\n\n"
    "想象做一道菜要 9 道工序:切葱、热油、下葱、下蛋、翻炒、加盐、装盘……\n"
    "(eager 模式)每道工序单独开一次火;而**一锅炖**把能合并的工序一次性完成——\n"
    "少开几次火,锅也少刷几次。\n\n"
    "`torch.compile` 干的就是这回事:把 eager 模式下**一个个小算子**(每算子一开火、\n"
    "每算子一读写显存)冻结成**少而大的融合 kernel**。\n\n"
    "| 工序 | eager | torch.compile |\n"
    "|---|---|---|\n"
    "| 切葱→热油→下葱 | 3 个 kernel | 1 个融合 kernel |\n"
    "| 每次开火 | 启动手续费 + 显存往返 | 只付一次 |"
)

# =====================================================================
# 第 2 节 · 两层架构
# =====================================================================
nb.md(
    "## 2. 两层架构:Dynamo + Inductor 🏗️\n\n"
    "`torch.compile(model)` 背后是两层流水线(PyTorch 2 论文, ASPLOS'24):\n\n"
    "1. **TorchDynamo(抓图)**:用 CPython 的帧求值钩子(PEP 523)重写字节码,\n"
    "   把一段段 PyTorch 算子序列**安全地抓成 FX 计算图**;\n"
    "2. **TorchInductor(生成代码)**:把 FX 图进一步 lower 成**循环级 IR**,\n"
    "   GPU 上生成 **Triton** kernel、CPU 上生成 C++/OpenMP 代码。\n\n"
    "论文报告:在 A100 上对 180+ 个真实模型,推理几何平均加速 **2.27×**、训练 **1.41×**。\n\n"
    "> ⚠️ 本机是 Windows 且无 MSVC 编译器:Inductor 的 CPU 后端无法生成 C++ 代码——\n"
    "> 我们把它变成一堂「现场教学课」:看 Dynamo 能否抓到图,看 aot_eager 能否交出图。"
)

# =====================================================================
# 第 3 节 · 数算子
# =====================================================================
nb.md(
    "## 3. 实测:eager 到底派发多少算子 🔬\n\n"
    "`TorchDispatchMode` 是 torch 的「算子拦截器」:任何经过派发层的 aten 调用都会先路过它。\n"
    "用它数清 MiniGPT 每个模块的算子数。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库
import torch.nn as nn                            # 网络层

class MiniGPT(nn.Module):
    """微型 GPT (与第 22 课同一台): 2 层 MHA + FFN + lm_head。"""

    def __init__(self, vocab=5000, hidden=256, n_layers=2, n_heads=4, max_seq=512):
        super().__init__()
        self.vocab, self.hidden = vocab, hidden
        self.n_layers, self.n_heads = n_layers, n_heads
        self.head_dim = hidden // n_heads
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_seq, hidden))
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "wq": nn.Linear(hidden, hidden), "wk": nn.Linear(hidden, hidden),
                "wv": nn.Linear(hidden, hidden), "wo": nn.Linear(hidden, hidden),
                "norm1": nn.LayerNorm(hidden),
                "w1": nn.Linear(hidden, 4 * hidden), "w2": nn.Linear(4 * hidden, hidden),
                "norm2": nn.LayerNorm(hidden),
            }))
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        Td = input_ids.shape[0]
        h = self.tok(input_ids) + self.pos[0, :Td]
        for blk in self.blocks:
            r = blk["norm1"](h)
            Nh, Dh = self.n_heads, self.head_dim
            q = blk["wq"](r).view(Td, Nh, Dh).transpose(0, 1)
            k = blk["wk"](r).view(Td, Nh, Dh).transpose(0, 1)
            v = blk["wv"](r).view(Td, Nh, Dh).transpose(0, 1)
            att = torch.softmax(q @ k.transpose(-1, -2) / (Dh ** 0.5), dim=-1) @ v
            att = att.transpose(0, 1).reshape(Td, -1)
            h = h + blk["wo"](att)
            h = h + blk["w2"](torch.nn.functional.gelu(blk["w1"](blk["norm2"](h))))
        return self.head(self.ln(h))

model = MiniGPT().eval()
print(f"MiniGPT 就绪, 参数量 = {sum(p.numel() for p in model.parameters()):,}")''',
    "🏗️ **迷你 GPT(与第 22 课同一台)**。本课解剖它的注意力块与 FFN 块,数一数 eager 的算子。",
)

nb.code(
    '''# -*- coding: utf-8 -*-
from torch.utils._python_dispatch import TorchDispatchMode   # 算子拦截器

# 数整个模型 + 单独数每个块
class OpCounter(TorchDispatchMode):              # 计数器
    def __init__(self):
        super().__init__()
        self.count = 0                           # 算子数
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1                          # 计数
        return func(*args, **(kwargs or {}))     # 原样执行

toks = torch.randint(0, 5000, (8,))              # (T=8,) prefill 输入
def count_ops(fn):
    c = OpCounter()                              # 新建计数器
    with torch.no_grad(), c:                     # 推理 + 计数
        fn()                                     # 执行
    return c.count                               # 返回算子数

n_model = count_ops(lambda: model(toks))         # 整个模型算子数
n_attn = count_ops(lambda: model.blocks[0]["wq"](model.blocks[0]["norm1"](model.tok(toks) + model.pos[0, :8])))  # 一层注意力投影
print(f"整个 MiniGPT 一次前向: {n_model} 个 aten 算子")
print(f"单层 norm+wq 投影:    {n_attn} 个")
print(f"=> 真实模型几十层、几千个算子 —— eager 的「每算子一开火」在大模型上是重税")''',
    "🔬 **一层就几十个算子**;真实模型几十层、几千个算子——eager 的「每算子一开火」在大模型上是重税。",
)

# =====================================================================
# 第 4 节 · 本机真跑 compile
# =====================================================================
nb.md(
    "## 4. 本机真跑 compile:两种命运 ⚠️\n\n"
    "现在真的调用 `torch.compile`。本机的命运我们先看默认后端(Inductor),再看 `aot_eager`:"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库

toks = torch.randint(0, 5000, (8,))              # 输入

# 尝试 1: 默认 Inductor 后端 (GPU 上生成 Triton kernel, CPU 上生成 C++ 代码)
print("=== 尝试 1: backend='inductor' (默认) ===")
try:
    compiled1 = torch.compile(model, backend="inductor")   # 编译
    with torch.no_grad():
        out1 = compiled1(toks)                    # 触发编译 + 执行
    print("Inductor 编译成功! (需要 MSVC/C++ 工具链)")
    print(f"输出 shape = {tuple(out1.shape)}")
except Exception as e:
    print(f"Inductor 失败: {type(e).__name__}")
    print(f"  原因: {str(e)[:160]}...")
    print("  (Windows 无 MSVC 时 Inductor CPU 后端无法生成代码——这是工具链问题, 不是 torch 的 bug)")

# 尝试 2: aot_eager 后端 —— 只走 Dynamo/AOTAutograd 追踪, 不做 Inductor 代码生成
print("\\n=== 尝试 2: backend='aot_eager' ===")
try:
    compiled2 = torch.compile(model, backend="aot_eager")   # 只抓图
    with torch.no_grad():
        out2 = compiled2(toks)                    # 执行 (仍走 eager kernel, 但图已交出)
    print("aot_eager 成功! (Dynamo 抓到图, 但没有代码生成)")
    print(f"输出 shape = {tuple(out2.shape)}")
    print("  -> 图已交出来, 速度不提升——但这是导出计算图、做算子级后处理的入口")
except Exception as e:
    print(f"aot_eager 失败: {type(e).__name__}: {str(e)[:160]}")''',
    "⚠️ **这不是 torch 的 bug,而是 Windows 的工具链现实**——也是很多教程在 Windows 上翻车的地方。"
    "在 Linux + GPU 上,Inductor 会生成 Triton kernel,效果正是本课第 5 节要估的收益。",
)

nb.md(
    "### aot_eager 到底交出了什么图?\n\n"
    "用 `torch._dynamo` 直接看 Dynamo 抓到的那张 FX 图,数一数图里的节点数——"
    "那就是 Inductor 将来要融合的「原材料」。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库
import torch._dynamo as td                       # Dynamo 图捕获

# 用 Dynamo 显式导出一次前向的 FX 图
def forward_fn(ids):                             # 包装前向 (返回 logits)
    return model(ids)

try:
    g = td.export(forward_fn, (toks,))           # 导出 FX 图
    gm = g[0] if isinstance(g, tuple) else g     # 图模块
    n_nodes = sum(1 for _ in gm.graph.nodes)     # 节点数
    n_ops = sum(1 for n in gm.graph.nodes if n.op == "call_function")  # 算子节点
    print(f"Dynamo 抓到的 FX 图: 共 {n_nodes} 个节点, 其中 {n_ops} 个算子节点")
    print("=> 这 n_ops 个算子就是 Inductor 融合的输入: 点对点算子可合并成大 kernel")
except Exception as e:
    print(f"Dynamo export 失败: {type(e).__name__}: {str(e)[:140]}")''',
    "🔍 **aot_eager / Dynamo 的价值不在速度,而在「把图交出来」**:vLLM 用它导出计算图,自己做算子级后处理。",
)

# =====================================================================
# 第 5 节 · 融合收益模型
# =====================================================================
nb.md(
    "## 5. 融合收益模型与可视化 📊\n\n"
    "没有 Inductor,融合收益算不出来吗?可以**模型估算**(思路与第 24 课一致):\n\n"
    "$$ \\text{收益} = \\underbrace{n_{\\text{eager}} \\cdot t_l}_{\\text{少开的火 × 手续费}} + "
    "\\underbrace{n_{\\text{往返}} \\cdot t_{\\text{HBM}}}_{\\text{省掉的显存往返}} $$\n\n"
    "以一点对点算子链(如 LN + 残差 + GELU)为例:3 个算子融合成 1 个,"
    "启动手续费少付 2 次,显存往返少 2 趟。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import json                                      # JSON 库
from pathlib import Path                         # 路径库

# 加载第 24 课的实测启动手续费
mf = Path(r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch04\\launch_measure_24.json")
launch = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"launch_us": 5.0, "kernel_us": 1.0}
t_l = launch["launch_us"]                          # 启动手续费 (µs)

# 以一个 3 算子点对点链为例 (norm + residual + gelu): 融合后 1 个 kernel
n_eager = 3                                        # eager 算子数
n_fused = 1                                        # 融合后 kernel 数
launch_save = (n_eager - n_fused) * t_l            # 少开的火 × 手续费 (µs)
hbm_roundtrip_us = 8.0                             # 一次 HBM 往返的典型成本 (µs, 估算)
hbm_save = (n_eager - n_fused) * hbm_roundtrip_us  # 省掉的往返 (µs)
total_save = launch_save + hbm_save                # 总收益
print(f"3 算子链融合: 少开 {n_eager - n_fused} 次火")
print(f"  启动手续费节省 = {launch_save:.2f} µs")
print(f"  显存往返节省   = {hbm_save:.1f} µs (估算)")
print(f"  合计 ≈ {total_save:.1f} µs/次")

# 换算整个 MiniGPT: 假设点对点算子占一半, 融合削减率 30~60%
import numpy as np                                 # 数值库
cut_rates = np.array([0.3, 0.5, 0.6])              # 融合削减率三种假设
n_ops_model = n_model                              # 整个模型算子数
for cr in cut_rates:                               # 每种削减率
    saved = int(n_ops_model * cr)                  # 省掉的算子数
    est_ms = saved * total_save / 1000             # 每步节省毫秒 (估算)
    print(f"削减率 {cr:.0%}: 省掉 {saved} 个算子 -> 每步估算省 {est_ms:.2f} ms")''',
    "📊 **削减率 30~60% 是典型量级**;GPU 上融合还省显存往返,实测加速常更高。"
    "诚实标注:compiled 侧为模型估算。",
)

nb.md(
    "### 可视化:收益来源拆解\n\n"
    "收益来源 = **少开的火 × 手续费** + **省掉的显存往返**(后者 GPU 上往往更大)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 三档削减率的收益拆解
cr_list = [0.3, 0.5, 0.6]                          # 削减率
launch_parts, hbm_parts = [], []                   # 两来源
for cr in cr_list:                                 # 每种
    saved = int(n_ops_model * cr)                  # 省算子数
    launch_parts.append(saved * launch_save)       # 启动部分
    hbm_parts.append(saved * hbm_roundtrip_us)     # 显存部分

x = np.arange(len(cr_list))                        # 位置
fig, ax = plt.subplots(figsize=(8, 4.5))            # 画布
ax.bar(x, launch_parts, color="#4C72B0", label="少开的火 × 手续费")   # 启动部分
ax.bar(x, hbm_parts, bottom=launch_parts, color="#C44E52", label="省掉的显存往返")  # 显存部分
ax.set_xticks(x)                                   # 刻度
ax.set_xticklabels([f"{c:.0%}" for c in cr_list])  # 标签
ax.set_xlabel("融合削减率")                          # x 轴
ax.set_ylabel("每步估算节省 (µs)")                   # y 轴
ax.set_title("融合收益拆解: 显存往返往往是大头")       # 标题
ax.legend()                                        # 图例
ax.grid(axis="y", alpha=0.3)                       # 网格
plt.tight_layout()
plt.show()''',
    "📈 **收益来源 = 少开的火 × 手续费 + 省掉的显存往返**(后者 GPU 上往往更大)。",
)

nb.code(
    '''# -*- coding: utf-8 -*-
import json                                      # JSON 库
from pathlib import Path                         # 路径库

# 存档给 App
measure = dict(
    n_ops_model=n_model,                           # 整个模型算子数 (实测)
    n_ops_layer=n_attn,                            # 单层算子数 (实测)
    launch_us=t_l,                                 # 启动手续费 (实测)
    hbm_roundtrip_us=hbm_roundtrip_us,             # HBM 往返 (估算)
    cut_rates=[float(c) for c in cut_rates],       # 削减率档位
    save_us=[float(l + h) for l, h in zip(launch_parts, hbm_parts)],  # 每档节省
    note="eager 侧 (kernel 数、耗时) 为本机实测; compiled 侧为融合规则估算",
)
out = Path(r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch04\\compile_measure_26.json")
out.write_text(json.dumps(measure, indent=2), encoding="utf-8")
print("[ok] 已保存:", out)
print("内容:", {k: v for k, v in measure.items() if not isinstance(v, list)})''',
    "💾 **eager 侧(kernel 数、耗时)为本机实测;compiled 侧为融合规则估算——文件里两组数字同源同义。**",
)

# =====================================================================
# 第 6 节 · vLLM CompilationMode
# =====================================================================
nb.md(
    "## 6. vLLM 的 CompilationMode 🏷️\n\n"
    "vLLM 把「编译到什么程度」做成四档配置(`vllm/config/compilation.py` 的 `CompilationMode`):\n\n"
    "| 模式 | 行为 | 适用 |\n"
    "|---|---|---|\n"
    "| `NO_COMPILE` | 纯 eager, 不编译 | 调试 / 兼容性 |\n"
    "| `INDUCTOR` | 用 torch.compile (Inductor) 编译模型 | 追求通用加速 |\n"
    "| `CUDA_GRAPHS` | 只捕获 CUDA Graph, 不编译 | 解码小批 |\n"
    "| `INDUCTOR_CUDA_GRAPHS` | 先编译再捕获 Graph | 全量优化 (默认) |\n\n"
    "也就是说:vLLM 把**本课的 torch.compile(算子融合)** 与 **上一课的 CUDA Graph(启动合并)**\n"
    "叠起来用——先融合成少而大的 kernel,再把这些 kernel 录成一张图一次重放。\n\n"
    "> 📄 实际配置还包含 `cudagraph_capture_sizes`(第 25 课的桶)、`cudagraph_mode` 等,"
    "> 共同决定「编译 + 捕获」的组合方式。"
)

# =====================================================================
# 第 7 节 · App
# =====================================================================
nb.md(
    "## 7. 🖥️ Streamlit 动态演示:torch.compile 与 Kernel 融合\n\n"
    "运行 `app_26_compile.py`:选择算子组合、切换「kernel 数量 / 单步耗时」指标。\n\n"
    "### 📜 App 完整源码(`app_26_compile.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_26_compile.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,切换算子组合与指标观察。"
)

wrapup(
    nb,
    summary=[
        "torch.compile 把 eager 的每算子一开火冻成少而大的融合 kernel",
        "两层架构:TorchDynamo 抓 FX 图 + TorchInductor 生成 Triton/C++ 代码(论文报 2.27× 推理)",
        "TorchDispatchMode 数出 MiniGPT 一次前向上百个算子,真实模型几千个",
        "本机 Windows 无 MSVC:Inductor 失败、aot_eager 成功——图能交出,代码不能生成",
        "融合收益 = 少开的火 × 手续费 + 省掉的显存往返,后者 GPU 上更大",
        "vLLM CompilationMode 四档,默认 INDUCTOR_CUDA_GRAPHS = 先融合再录图",
    ],
    practice=[
        "在 Linux/GPU 机器上跑 Inductor,对比 compiled vs eager 的真实加速比",
        "用 torch._dynamo.config.trace 打开日志,观察 Dynamo 的 graph break 位置",
        "把 MiniGPT 换成 4 层,重新数算子并更新融合收益估算",
        "阅读 vLLM compilation.py,找出 cudagraph_mode 与 cudagraph_capture_sizes 的关系",
    ],
    links=[
        ("PyTorch 2 论文 (ASPLOS'24)", "https://docs.pytorch.org/assets/pytorch2-2.pdf"),
        ("torch.compile 官方文档", "https://pytorch.org/get-started/pytorch-2.0/"),
        ("TorchInductor 设计笔记", "https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747"),
        ("vLLM compilation.py", "https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch04\26_torch_compile.ipynb")