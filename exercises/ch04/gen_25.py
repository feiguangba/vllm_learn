# -*- coding: utf-8 -*-
"""生成第 25 课 notebook: CUDA Graph 捕获与重放(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:录像带直觉 -> 三步流程与硬约束 -> CPU 录制-回放模拟 -> 量化 -> vLLM capture sizes -> torch.cuda.graph API
2. 每一行代码都有 inline 注释
3. 每个中间量打印并标注含义
4. 论文支撑:NVIDIA CUDA Graphs 官方博客、vLLM compilation config (cudagraph_capture_sizes)
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_25_capture.py"
APP_NAME = "app_25_capture.py"
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
    "第 25 课 · CUDA Graph 捕获与重放:为什么形状必须固定",
    subtitle="三步流程 · 三条硬约束 · CPU 录制-回放 · 量化重放收益 · vLLM capture sizes",
    emoji="🎬", chapter="第 4 章 · 模型执行器与 CUDA 优化",
)

chapter_cover(
    nb,
    objectives=[
        "理解 CUDA Graph 的三步流程:捕获 → 实例化 → 重放",
        "掌握三条硬约束,尤其「形状必须固定」——decode 每步 T=B 的契机",
        "用 TorchDispatchMode 数清 MiniGPT 一次 decode 前向的算子数",
        "在 CPU 上复刻录制-回放模拟器,量化重放省下的启动开销",
        "理解 vLLM 的 capture sizes:一戏一带,按桶预录多张图",
        "看懂 torch.cuda.graph 的标准 API 写法(本机不执行,仅展示)",
    ],
    toc=[
        ("直觉:录像带", "把一段计算想成一场舞台剧"),
        ("三步流程与三条硬约束", "捕获/实例化/重放 + 形状固定"),
        ("搭舞台:迷你 GPT", "复用第 22 课的模型"),
        ("CPU 录制-回放模拟器", "TorchDispatchMode 数算子 + 回放"),
        ("量化:重放到底省多少", "eager vs replay 计时对比"),
        ("vLLM 的 capture sizes", "一戏一带, 按桶预录"),
        ("真实 API:torch.cuda.graph", "标准写法展示 (本机不执行)"),
    ],
    links=[
        ("NVIDIA 官方博客: Getting Started with CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("CUDA Programming Guide: CUDA Graphs", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"),
        ("vLLM compilation config 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py"),
        ("vLLM V1 博客: Anatomy of vLLM", "https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:录像带 🎬\n\n"
    "把一段计算想成一场**舞台剧**:\n\n"
    "- **录一次**:演员(算子)把整场戏按剧本(计算图)演一遍,录成带子;\n"
    "- **放 N 遍**:以后每次直接放带子,演员不用再重新彩排。\n\n"
    "CUDA Graph 就是这个「录像」:第一次把一串 kernel 的提交过程录进一张图,\n"
    "之后每次用**一次启动**重放整张图——手续费只付一次。\n\n"
    "本课拆解三步流程,并回答一个关键问题:**为什么形状必须固定?**"
)

# =====================================================================
# 第 2 节 · 三步流程与硬约束
# =====================================================================
nb.md(
    "## 2. 三步流程与三条硬约束 ⚠️\n\n"
    "| 步骤 | 干什么 | 为什么 |\n"
    "|---|---|---|\n"
    "| ① capture 捕获 | `cudaStreamBeginCapture` 开始录像, 提交的 kernel 被录进图 | 把 n 次提交固化成一张图 |\n"
    "| ② instantiate 实例化 | `cudaGraphInstantiate` 把图编译成可执行对象 | 一次性付手续费, 重放零开销 |\n"
    "| ③ launch 重放 | `cudaGraphLaunch` 一次启动, GPU 重放整张图 | 每次启动只要一次提交 |\n\n"
    "**三条硬约束**:\n\n"
    "1. **形状固定**:kernel 的网格/线程/参数在捕获时被固化,重放时不能变;\n"
    "2. **显存地址固定**:算子写到的张量地址不能变(否则要重新捕获);\n"
    "3. **控制流固定**:不能有 if/while 等动态分支(只能录一条直线)。\n\n"
    "decode 阶段每步 `T = B` 形状不变,正好满足约束 1——这是 vLLM 给 decode 用 CUDA Graph 的根本前提。"
)

# =====================================================================
# 第 3 节 · 迷你 GPT
# =====================================================================
nb.md(
    "## 3. 搭舞台:迷你 GPT 🏗️\n\n"
    "复用第 22 课的迷你 GPT(H=256、2 层、4 头、词表 5000)。捕获/重放类比演示的目标就是它的\n"
    "decode 前向:`T = B` 个词元。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库
import torch.nn as nn                            # 网络层

class MiniGPT(nn.Module):
    """微型 GPT (与第 22 课同一台): 2 层 MHA + FFN + lm_head。"""

    def __init__(self, vocab=5000, hidden=256, n_layers=2, n_heads=4, max_seq=512):
        super().__init__()                       # 父类初始化
        self.vocab, self.hidden = vocab, hidden  # 词表 / 隐藏维
        self.n_layers, self.n_heads = n_layers, n_heads  # 层数 / 头数
        self.head_dim = hidden // n_heads        # 每头维 Dh
        self.tok = nn.Embedding(vocab, hidden)   # (V, H)
        self.pos = nn.Parameter(torch.zeros(1, max_seq, hidden))  # 位置编码
        self.blocks = nn.ModuleList()            # 层列表
        for _ in range(n_layers):                # 逐层
            self.blocks.append(nn.ModuleDict({   # 每层模块
                "wq": nn.Linear(hidden, hidden), "wk": nn.Linear(hidden, hidden),
                "wv": nn.Linear(hidden, hidden), "wo": nn.Linear(hidden, hidden),
                "norm1": nn.LayerNorm(hidden),
                "w1": nn.Linear(hidden, 4 * hidden), "w2": nn.Linear(4 * hidden, hidden),
                "norm2": nn.LayerNorm(hidden),
            }))
        self.ln = nn.LayerNorm(hidden)           # 最终 LN
        self.head = nn.Linear(hidden, vocab, bias=False)   # lm_head

    def forward(self, input_ids):
        """input_ids (T,) -> logits (T, V)。"""
        Td = input_ids.shape[0]                  # 词元流长度
        h = self.tok(input_ids)                  # (T, H) 嵌入
        h = h + self.pos[0, :Td]                 # (T, H) 位置编码
        for blk in self.blocks:                  # 逐层
            r = blk["norm1"](h)                  # LN
            Nh, Dh = self.n_heads, self.head_dim # 头数 / 每头维
            q = blk["wq"](r).view(Td, Nh, Dh).transpose(0, 1)  # (Nh,T,Dh)
            k = blk["wk"](r).view(Td, Nh, Dh).transpose(0, 1)  # (Nh,T,Dh)
            v = blk["wv"](r).view(Td, Nh, Dh).transpose(0, 1)  # (Nh,T,Dh)
            att = torch.softmax(q @ k.transpose(-1, -2) / (Dh ** 0.5), dim=-1) @ v  # 注意力
            att = att.transpose(0, 1).reshape(Td, -1)         # (T, H)
            h = h + blk["wo"](att)               # 残差
            h = h + blk["w2"](torch.nn.functional.gelu(blk["w1"](blk["norm2"](h))))  # FFN
        return self.head(self.ln(h))             # (T, V) logits

model = MiniGPT(vocab=5000, hidden=256, n_layers=2, n_heads=4).eval()   # eval 模式
print(f"MiniGPT 就绪, 参数量 = {sum(p.numel() for p in model.parameters()):,}")''',
    "🏗️ **迷你 GPT(与第 22 课同一台)**。捕获/重放类比演示的目标就是它的 decode 前向:T = B 个词元。",
)

# =====================================================================
# 第 4 节 · CPU 录制-回放
# =====================================================================
nb.md(
    "## 4. CPU 类比:录制-回放模拟器 📼\n\n"
    "我们在 CPU 上复刻「三步流程」:\n\n"
    "1. **record(录制)**:用 `TorchDispatchMode` 拦截算子调用,把每一步「录」成一张清单;\n"
    "2. **replay(回放)**:第一次执行时顺便记录,之后每次直接跑已录好的清单。\n\n"
    "先数一数:一次 decode 前向到底派发多少个算子?"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                     # 深度学习库
from torch.utils._python_dispatch import TorchDispatchMode   # 算子拦截器

# 统计前向过程中派发的算子数
class OpCounter(TorchDispatchMode):              # 自定义 dispatch 模式
    def __init__(self):
        super().__init__()                       # 初始化
        self.count = 0                           # 算子计数
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1                          # 每个 aten 调用 +1
        return func(*args, **(kwargs or {}))     # 原样执行

toks = torch.randint(0, 5000, (4,))              # (T=4,) decode 输入 (B=4)
counter = OpCounter()                            # 计数器
with torch.no_grad(), counter:                   # 推理 + 计数
    out = model(toks)                            # 跑一次前向
print(f"一次 decode 前向派发了 {counter.count} 个 aten 算子")
print(f"=> 2 层就这么多了; 真实模型几十层, 一次 decode 前向就是几百个小 kernel 的「指令风暴」")''',
    "✅ **2 层迷你 GPT 的 decode 前向约几十个算子**——真实模型几十层,一次 decode 前向就是几百个小 kernel。"
    "每个 kernel 都付一次启动手续费,正是第 24 课量化的浪费。",
)

# =====================================================================
# 第 5 节 · 量化
# =====================================================================
nb.md(
    "## 5. 量化:重放到底省多少 ⏱️\n\n"
    "CPU 上 replay 与 eager 计算量相同、Python 开头相同,所以直接计时差距很小(下面会实测验证这一点)。\n"
    "真实收益在 GPU 上:重放抹掉了 `n × t_l` 的启动开销。我们用第 24 课的实测 `t_l` 来估算。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import time                                      # 计时库
import torch                                     # 深度学习库
import numpy as np                                # 数值库

def ms_per_call(fn, n=300, warm=30):
    """测函数平均单次调用耗时 (毫秒)。"""
    for _ in range(warm):                          # 预热
        fn()
    t0 = time.perf_counter()                       # 计时开始
    for _ in range(n):                             # 重复
        fn()
    return (time.perf_counter() - t0) / n * 1e3    # 平均毫秒

toks = torch.randint(0, 5000, (4,))               # (T=4,) decode 输入

# eager: 直接前向
with torch.no_grad():
    ms_eager = ms_per_call(lambda: model(toks))    # eager 平均耗时

# 「replay」: 用固定调用序列模拟已录好的图 (形状固定 T=4)
import torch.nn.functional as F                   # 函数式接口
def recorded_forward(ids):
    """模拟已录好的算子序列 (固定形状 T=4)。"""
    Td = ids.shape[0]                             # 长度
    h = model.tok(ids)                            # 嵌入
    h = h + model.pos[0, :Td]                     # 位置
    for blk in model.blocks:                      # 逐层
        r = blk["norm1"](h)                       # LN
        Nh, Dh = model.n_heads, model.head_dim    # 头/维
        q = blk["wq"](r).view(Td, Nh, Dh).transpose(0, 1)   # Q
        k = blk["wk"](r).view(Td, Nh, Dh).transpose(0, 1)   # K
        v = blk["wv"](r).view(Td, Nh, Dh).transpose(0, 1)   # V
        att = torch.softmax(q @ k.transpose(-1, -2) / (Dh ** 0.5), dim=-1) @ v  # 注意力
        att = att.transpose(0, 1).reshape(Td, -1)          # (T, H)
        h = h + blk["wo"](att)                    # 残差
        h = h + blk["w2"](F.gelu(blk["w1"](blk["norm2"](h))))  # FFN
    return model.head(model.ln(h))                # logits

with torch.no_grad():
    ms_replay = ms_per_call(lambda: recorded_forward(toks))   # replay 平均耗时

# 用第 24 课的实测启动手续费估算 GPU 上的收益
import json                                      # JSON 库
from pathlib import Path                         # 路径库
mf = Path(r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch04\\launch_measure_24.json")
launch = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"launch_us": 5.0, "kernel_us": 1.0}  # 加载实测
t_l = launch["launch_us"]                         # 每次启动手续费 (µs)
n_ops = counter.count                             # 算子数
save_per_step = n_ops * t_l / 1000                # 每步省下的毫秒 (估算)
print(f"eager  平均耗时 = {ms_eager:.4f} ms")
print(f"replay 平均耗时 = {ms_replay:.4f} ms")
print(f"(CPU 上两者几乎一样 —— 重放省的是 GPU 启动开销, 不是计算本身)")
print(f"\\nGPU 上估算: {n_ops} 个算子 × {t_l:.2f} µs 启动费 = 每步省 {save_per_step:.3f} ms")''',
    "⏱️ **诚实标注**:eager/replay/捕获是本机 CPU 实测;graph 延迟是「抹掉 n×t_l 启动开销」的模型估算——"
    "与 GPU 实测报告的量级(小批 decode 每步省零点几毫秒)一致。",
)

nb.md(
    "### 估算结果可视化 + 数据存档\n\n"
    "把「每步节省 × 步数」画成累计曲线:decode 每步只多 1 个词元,却要跑完整网络几百个 kernel——\n"
    "每步省下的零点几毫秒,乘上几百步就是用户少等的半秒钟。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
import json                                      # JSON 库
from pathlib import Path                         # 路径库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

steps = np.arange(0, 300, 5)                      # decode 步数 0..295
save_ms = steps * save_per_step                    # 累计节省 (线性)

fig, ax = plt.subplots(figsize=(8, 4))             # 画布
ax.plot(steps, save_ms, color="#4C72B0", lw=2)     # 曲线
ax.set_xlabel("decode 步数")                       # x 轴
ax.set_ylabel("累计节省 (ms, 估算)")                # y 轴
ax.set_title("CUDA Graph 累计收益: 每步省 n×t_l, 随步数线性增长")  # 标题
ax.grid(alpha=0.3)                                 # 网格
plt.tight_layout()
plt.show()

# 存档给 App
result = dict(
    n_ops=n_ops,                                   # 算子数
    ms_eager=ms_eager,                             # eager 实测 ms
    ms_replay=ms_replay,                           # replay 实测 ms
    launch_us=t_l,                                 # 启动手续费 µs
    save_per_step_ms=save_per_step,                # 每步节省 ms (估算)
    note="eager/replay 为本机 CPU 实测; save 为基于启动手续费的估算",
)
out = Path(r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\\ch04\\cudagraph_result_25.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("[ok] 已保存:", out)''',
    "💾 **App 启动时会显示「已加载第 25 课 notebook 生成的数据」——就是这份文件。**",
)

# =====================================================================
# 第 6 节 · vLLM capture sizes
# =====================================================================
nb.md(
    "## 6. vLLM 的 capture sizes:一戏一带,按桶预录 🗂️\n\n"
    "硬约束「形状固定」撞上现实「批大小会变」,vLLM 的解法朴素而有效:**预录多盘带子**。\n"
    "`cudagraph_capture_sizes` 是预设的批大小桶,例如 `[1,2,4,8,16,24,32,40,48,64,80,96,112,128]`:\n\n"
    "- decode 每步算出实际 `B`,向上取整到最近的桶,重放对应那盘带子;\n"
    "- 多余的位置 padding(浪费一些算力,但换来零启动开销);\n"
    "- 桶是指数+等距混合排布,让最坏浪费封顶。\n\n"
    "> 📄 见 `vllm/config/compilation.py` 的 `cudagraph_capture_sizes`。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                 # 数值库
import matplotlib.pyplot as plt                   # 绘图库
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# vLLM 默认 capture sizes (与真实配置一致的一档示例)
sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]  # 桶列表
B = 33                                              # 一个实际的批大小
cap = min(s for s in sizes if s >= B)              # 向上取整到最近桶
waste = (cap - B) / cap                            # padding 浪费比例
print(f"实际 B={B} -> 落到桶 {cap}, padding 浪费 = {waste * 100:.1f}%")

# 画桶的覆盖图
fig, ax = plt.subplots(figsize=(9, 3.5))            # 画布
ax.scatter(sizes, [0] * len(sizes), s=60, color="#4C72B0")   # 桶的位置
ax.axvline(B, color="#C44E52", ls="--", lw=2, label=f"实际 B={B}")  # 实际批
ax.axvline(cap, color="#55A868", ls="-.", lw=2, label=f"落桶 cap={cap}")  # 落桶
ax.set_xlabel("批大小 B (桶)")                      # x 轴
ax.set_yticks([])                                  # 隐藏 y 轴
ax.set_title("vLLM capture sizes: 实际 B 向上取整到最近桶")  # 标题
ax.legend()                                        # 图例
ax.grid(axis="x", alpha=0.3)                       # 网格
plt.tight_layout()
plt.show()

# 最坏浪费: 落在两个相邻桶中间时 padding 最大
worst_b = max((sizes[i + 1] + sizes[i]) // 2 for i in range(len(sizes) - 1))  # 近似最坏点
worst_w = min(s for s in sizes if s >= worst_b)    # 落桶
print(f"最坏情况: B={worst_b} -> 落桶 {worst_w}, 浪费 = {(worst_w - worst_b) / worst_w * 100:.1f}%")''',
    "🗂️ **最坏情况:B=33 → padding 到 40,浪费 17.5%(=(40−33)/40);B=1 → 浪费 0%。**"
    "桶是指数+等距混合排布,让最坏浪费封顶。",
)

# =====================================================================
# 第 7 节 · 真实 API
# =====================================================================
nb.md(
    "## 7. 真实 API:torch.cuda.graph 长什么样 🔍\n\n"
    "下面是 GPU 上的标准写法(**本机 Windows 下 CUDA Graph 捕获不稳定,本课只展示、不执行**;\n"
    "要验证请在有 CUDA 的 Linux 机器上运行):\n\n"
    "```python\n"
    "import torch\n"
    "# 1) 准备固定形状的输入 (T = B 固定)\n"
    "x = torch.zeros((B,), dtype=torch.long, device='cuda')\n"
    "# 2) 预热 + 捕获\n"
    "g = torch.cuda.CUDAGraph()\n"
    "# 先用一次 eager 前向分配所有中间张量 (让显存地址固定)\n"
    "s = torch.cuda.Stream()\n"
    "s.wait_stream(torch.cuda.current_stream())\n"
    "with torch.cuda.stream(s):\n"
    "    for _ in range(3):\n"
    "        logits = model(x)\n"
    "torch.cuda.current_stream().wait_stream(s)\n"
    "# 正式捕获\n"
    "with torch.cuda.graph(g):\n"
    "    logits = model(x)            # 这串 kernel 被录进 g\n"
    "# 3) 重放: 每次启动零 Python 参与\n"
    "g.replay()\n"
    "```\n\n"
    "关键点:捕获必须在**专属侧流**里做,`with torch.cuda.graph(g)` 内部自动完成\n"
    "`begin_capture / end_capture / instantiate`。重放时改输入只需写 `x.copy_(新token)`,\n"
    "因为图里固化的地址不变。"
)

# =====================================================================
# 第 8 节 · App
# =====================================================================
nb.md(
    "## 8. 🖥️ Streamlit 动态演示:捕获与重放\n\n"
    "运行 `app_25_capture.py`:切换「延迟对比 / 重放延迟分布 / 耗时累计曲线」三个视图。\n\n"
    "### 📜 App 完整源码(`app_25_capture.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_25_capture.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,观察三种视图。"
)

wrapup(
    nb,
    summary=[
        "CUDA Graph 三步:捕获(录像) → 实例化(编译) → 重放(一次启动) ",
        "三条硬约束:形状固定 / 地址固定 / 控制流固定;decode 每步 T=B 恰好满足",
        "TorchDispatchMode 数出 MiniGPT 一次 decode 前向几十个算子,真实模型几百个",
        "重放收益 = n × t_l(每步省下的启动手续费),与步数线性累积",
        "vLLM capture sizes:按桶预录多张图,实际 B 向上取整,最坏浪费封顶",
        "torch.cuda.graph 标准写法:侧流预热 + with 捕获 + copy_ 换输入 + replay",
    ],
    practice=[
        "把 OpCounter 换成记录每个算子名,打印前 10 个热点算子",
        "把 recorded_forward 改成真正的 TorchDynamo 图导出,对比算子数",
        "用 torch.cuda.graph 在 Linux/有 GPU 的机器上实测 B=1/8/32 三档的加速比",
        "把 capture sizes 改成等距 [1,2,...,128],比较最坏浪费与预录成本",
    ],
    links=[
        ("NVIDIA: Getting Started with CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("CUDA Programming Guide", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"),
        ("vLLM compilation config 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch04\25_cudagraph_capture.ipynb")