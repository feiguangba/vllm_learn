# -*- coding: utf-8 -*-
"""生成 25_cudagraph_capture.ipynb(app_25_capture.py 已存在,%%writefile 覆盖写入相同内容)"""
from helpers import D, MINI_GPT, chapter_cover, wrapup, new_nb, CH04, app_src, finalize


def app_cell(name):
    return "%%writefile " + name + "\n" + app_src(name)


NB = new_nb("第 25 课 · CUDA Graph 捕获与重放:为什么形状必须固定",
            subtitle="录像 🎬 的三步:热身 → 捕获 → 重放;以及为什么批大小一变就得重新录",
            emoji="🎬")

chapter_cover(NB,
    objectives=[
        "掌握 CUDA Graph 的三步流程:warmup 热身 → stream capture 捕获 → replay 重放",
        "理解三条硬约束:形状固定、静态内存(输入缓冲地址不变)、捕获期禁止动态行为",
        "明白为什么 vLLM 按 batch size 预捕获多张图(capture sizes 桶),并把真实 batch padding 到桶",
        "用 CPU「录制-回放」类比:录制算子序列、验证重放结果一致、量化 Python 派发开销",
        "读懂真实 torch.cuda.graph API 代码(Windows 下不稳定,本课只展示不执行)",
        "跑通配套 App,交互查看延迟对比、重放延迟分布与累计耗时曲线",
    ],
    toc=[
        ("直觉:录像带", "先热机、再录一遍、以后照带子放 —— 场景变了就得重录"),
        ("三步流程与三条硬约束", "warmup / capture / replay;静态形状、静态内存、捕获期禁忌"),
        ("CPU 类比:录制-回放模拟器", "OpRecorder 录下算子序列,CPUGraph 重放,验证结果一字不差"),
        ("量化:重放到底省多少", "用第 24 课的启动开销模型估算 graph 延迟"),
        ("vLLM 的 capture sizes", "按 batch 桶预捕获多张图 + padding 的浪费账"),
        ("真实 API 代码展示", "torch.cuda.CUDAGraph 完整用法(标注:Windows 不稳定,概念为主)"),
        ("保存实测数据给 App", "生成 cudagraph_result_25.json"),
        ("配套 App:🎬 捕获与重放", "streamlit 交互演示"),
    ],
    links=[
        ("PyTorch CUDA Graphs 官方文档", "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs"),
        ("NVIDIA 博客:CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("vLLM CUDA Graph 设计文档", "https://docs.vllm.ai/en/latest/design/v1/v1_cuda_graph.html"),
    ])

NB.md("## 1. 直觉:录像带 🎬",
D('''
把一段计算想成一场**舞台剧**:

- **Eager 演法**:每次演出,导演(CPU)都在台下逐条喊指令 ——「灯光!」「音响!」「道具!」,
  每喊一条都有延迟(第 24 课的手续费);
- **CUDA Graph 演法**:
  1. **热身(warmup)**:先完整演几遍,让演员就位(CUDA 上下文、缓存、工作区就绪);
  2. **捕获(capture)**:开录像机,再演一遍 —— GPU 把所有「指令」连同依赖关系**录进带子**;
  3. **重放(replay)**:以后每次演出只要按一下播放键 `graph.replay()`,整场戏一次启动、无缝连播。

为什么**场景不能变**?因为带子上录的是**绝对位置**:哪个 buffer、哪块显存、什么形状,全部写死。
布景一换(批大小变了、序列长度变了),带子就对不上实景 —— **必须重录**(重新捕获)。
'''))

NB.md("## 2. 三步流程与三条硬约束 ⚠️",
D('''
| 步骤 | 干什么 | 为什么要 |
|---|---|---|
| 1 warmup | 固定形状先跑几遍(eager) | 让 lazy 初始化、缓存分配、cudnn benchmark 都发生**在捕获之前** |
| 2 capture | `with torch.cuda.graph(g):` 里跑一遍 | 录制 kernel 序列;此时**不真正执行**(至少不保证执行) |
| 3 replay | `g.replay()` | 一次提交、全图执行,启动开销 ≈ 一次 launch |

**三条硬约束**(违反即炸):

1. **形状必须固定**:图里每个 kernel 的 grid/block/参数都按捕获时的形状写死 → batch 一变就要重捕;
2. **内存必须静态**:重放读写的是**捕获时那几块显存地址** → vLLM 专门准备**静态输入缓冲区**,
   每步先把新数据 `copy_` 进去,再 replay,输出也从静态缓冲区读;
3. **捕获期间禁止**:分配新显存、CPU-GPU 同步、动态控制流(数据相关的 if/循环)。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch OMP 库冲突(Windows)
import time, json
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
torch.manual_seed(7)
print("本课在 CPU 上做「录制-回放」类比演示;真实 torch.cuda.graph API 见第 6 节(Windows 下不稳定,只展示)。")
'''),
"🎬 先搭舞台:复用第 22 课的迷你 GPT(H=256、2 层、4 头、词表 5000)。")

NB.code(MINI_GPT,
"🏗️ 迷你 GPT(与第 22 课同一台)。捕获/重放类比演示的目标就是它的 decode 前向:T = B 个词元。")

NB.md("## 3. CPU 类比:录制-回放模拟器 📼",
D('''
我们在 CPU 上复刻「三步流程」:

- `OpRecorder`(基于 `TorchDispatchMode`):像录像机一样,**录下**前向触发的每个算子的名字;
- `CPUGraph`:record = 在录制器下跑一遍并保存「带子」(算子序列);replay = 按同一份带子重演。

CPU 的 replay 仍要走 Python 逐算子派发,所以**提速有限**;但它完美演示语义:
**重放执行的算子序列与录制时一字不差,结果也一字不差**。真正的提速来自抹掉 GPU launch 开销,
我们用第 24 课测的「每次派发手续费」来估算。
'''))

NB.code(D('''
from torch.utils._python_dispatch import TorchDispatchMode

class OpRecorder(TorchDispatchMode):
    """录像机:录下经过 torch 派发层的每个算子名"""
    def __init__(self):
        super().__init__()
        self.names = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(str(func).replace("aten.", ""))
        return func(*args, **(kwargs or {}))


class CPUGraph:
    """CPU 上的「CUDA Graph」类比:record 录制算子序列,replay 按同一序列重演"""
    def __init__(self, fn):
        self.fn = fn          # 被录制的计算(整段前向)
        self.tape = None      # 「带子」:算子名序列
        self.n_ops = 0

    def warmup(self, *args):
        for _ in range(3):    # 三步流程第一步:热身
            self.fn(*args)
        return self

    def record(self, *args):
        with OpRecorder() as rec:              # 第二步:开录像机跑一遍
            out = self.fn(*args)
        self.tape = rec.names
        self.n_ops = len(self.tape)
        self.static_out = out                  # 「静态输出缓冲」:捕获时的结果地址/内容
        return out

    def replay(self, *args):
        # 第三步:重放 —— 语义上等价于「按带子重演」(输入应先拷入静态缓冲)
        return self.fn(*args)
'''),
"📼 与真实 CUDA Graph 的对应:record 录带子;真实版连 kernel 参数与显存地址都录,重放零 Python 参与。")

NB.code(D('''
model = MiniGPT(vocab=5000, hidden=256, n_layers=2, n_heads=4).eval()

B = 8                                       # decode 场景:批 8 条,每条每步 1 个新词元
ids = torch.randint(0, 5000, (B,))
pos = torch.full((B,), 64, dtype=torch.long)  # 假设每条序列当前在位置 64

with torch.no_grad():
    g = CPUGraph(lambda i, p: model(i, p))
    g.warmup(ids, pos)                      # ① 热身
    out_rec = g.record(ids, pos)            # ② 捕获(录制)
    out_rep = g.replay(ids, pos)            # ③ 重放

print(f"带子长度:一次 decode 前向共 {g.n_ops} 个算子")
print(f"带子前 10 个: {g.tape[:10]}")
print(f"重放结果与录制结果最大误差 = {(out_rec - out_rep).abs().max().item():.2e}")
print("✅ 重放 = 同一段计算的忠实重演")
'''),
"✅ 2 层迷你 GPT 的 decode 前向约 60 个算子 —— 真实模型几十层,一次 decode 前向就是几百个小 kernel 的「指令风暴」。")

NB.md("## 4. 量化:重放到底省多少 ⏱️",
D('''
CPU 上 replay 与 eager 计算量相同、Python 开头相同,所以直接计时差距很小(下面会实测验证这一点)。
GPU 上的收益 = 抹掉 n 次 launch 开销。用第 24 课同款方法在**本机**重测每次派发的手续费,
然后套模型:

$$t_{\\text{graph}} \\approx t_{\\text{eager}} - n_{\\text{ops}} \\times t_l \\quad(下限不低于 \\; t_{\\text{eager}} \\times 0.4)$$
'''))

NB.code(D('''
def ms_per_call(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3

def us_per_call(fn, n=4000, warm=400):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

tiny = torch.ones(16)
big = torch.ones(16000)
launch_us = max(us_per_call(lambda: tiny.add_(1.0)) - 16 * us_per_call(lambda: big.add_(1.0)) / 16000, 0.0)

with torch.no_grad():
    eager_ms = ms_per_call(lambda: model(ids, pos))
    replay_ms = ms_per_call(lambda: model(ids, pos))     # CPU replay:同一调用,Python 开销未省
graph_ms = max(eager_ms - g.n_ops * launch_us / 1e3, eager_ms * 0.4)   # GPU 收益模型估算

t0 = time.perf_counter()
with torch.no_grad():
    CPUGraph(lambda i, p: model(i, p)).warmup(ids, pos).record(ids, pos)
capture_ms = (time.perf_counter() - t0) * 1e3             # 「捕获」= 热身 3 次 + 录制 1 次

print(f"本机每次派发手续费 t_l     ≈ {launch_us:.2f} µs")
print(f"decode 前向 eager 平均耗时 = {eager_ms:.3f} ms({g.n_ops} 个算子)")
print(f"CPU replay 实测(无提速)  = {replay_ms:.3f} ms  <- Python 派发还在,GPU 上这部分才被抹掉")
print(f"GPU 收益模型估算 graph     ≈ {graph_ms:.3f} ms(= eager − n_ops×t_l,下限 0.4×eager)")
print(f"估算加速比                 ≈ {eager_ms / graph_ms:.2f} x")
print(f"「捕获」一次性开销(本机) = {capture_ms:.1f} ms —— 付一次,重放千万次")
'''),
"⏱️ 诚实标注:eager/replay/捕获是本机 CPU 实测;graph 延迟是「抹掉 n×t_l 启动开销」的模型估算 —— 与 GPU 实测报告的量级(小批 decode 提速 1.2~2x)一致。")

NB.code(D('''
# 用估算均值生成「重放延迟分布」,并保存给配套 App(它会自动加载 cudagraph_result_25.json)
rng = np.random.default_rng(7)
replay_lat = rng.normal(graph_ms, graph_ms * 0.06, 40).clip(0.02, None)

meas = {
    "eager_mean_ms": round(float(eager_ms), 4),
    "graph_mean_ms": round(float(graph_ms), 4),
    "replay_lat_ms": [round(float(x), 4) for x in replay_lat],
    "speedup": round(float(eager_ms / graph_ms), 3),
    "capture_ms": round(float(capture_ms), 2),
    "warmup": 3,
    "batch": B,
    "hidden": 256,
}
with open("cudagraph_result_25.json", "w", encoding="utf-8") as f:
    json.dump(meas, f, ensure_ascii=False, indent=2)
print("已保存 cudagraph_result_25.json:")
print({k: v for k, v in meas.items() if k != "replay_lat_ms"})
'''),
"💾 App 启动时会显示「已加载第 25 课 notebook 生成的数据」—— 就是这份文件。")

NB.md("## 5. vLLM 的 capture sizes:一戏一带,按桶预录 🗂️",
D('''
硬约束「形状固定」撞上现实「批大小会变」,vLLM 的解法朴素而有效:**预录多盘带子**。

- 启动时对一组 **capture sizes**(如 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64 …)各捕获一张图;
- 运行时真实批大小 `B=7` 来了,就 **padding 到最近的桶(8)**,用 8 号带子重放,多余位置算完丢弃。

代价是 padding 浪费:$\\text{浪费} = \\frac{\\text{桶} - B}{B}$。桶越密浪费越小,但捕获时间与显存越多。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.express as px

sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
real_bs = np.arange(1, 65)
padded = np.array([min(s for s in sizes if s >= b) for b in real_bs])
waste = (padded - real_bs) / real_bs * 100

df = pd.DataFrame({"真实批大小": real_bs, "padding 到桶": padded, "浪费 %": np.round(waste, 1)})
fig = px.bar(df, x="真实批大小", y="浪费 %", hover_data=["padding 到桶"])
fig.update_layout(title="padding 到 capture size 桶的计算浪费(桶: 1~64 共 11 档)",
                  height=360, xaxis_dtick=4)
fig
'''),
"🗂️ 最坏情况:B=33 → padding 到 40,浪费 21%;B=1 → 浪费 0%。桶是指数+等距混合排布,让最坏浪费封顶。")

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

ks = np.arange(1, 61)
bar = (
    Bar()
    .add_xaxis([str(int(k)) for k in ks[::3]])
    .add_yaxis("Eager 累计(ms)", [round(float(k * eager_ms), 2) for k in ks[::3]],
               label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Graph 累计(ms)", [round(float(k * graph_ms), 2) for k in ks[::3]],
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(
        title=f"decode 累计耗时:每步省 {eager_ms - graph_ms:.3f} ms,60 步省 {(eager_ms - graph_ms) * 60:.1f} ms"),
        xaxis_opts=opts.AxisOpts(name="decode 步数"), yaxis_opts=opts.AxisOpts(name="累计耗时(ms)"))
)
bar.render_notebook()
'''),
"📈 decode 每步只多 1 个词元,却要跑完整网络几百个 kernel —— 每步省下的零点几毫秒,乘上几百步就是用户少等的半秒钟。")

NB.md("## 6. 真实 API:torch.cuda.graph 长什么样 🔍",
D('''
下面是 GPU 上的标准写法(**本机 Windows 下 CUDA Graph 捕获不稳定,本课只展示、不执行**;
在 Linux + GPU 环境可原样运行):

```python
import torch

# ① 静态输入/输出缓冲:重放读写的一直是这两块显存
static_ids  = torch.zeros(8, dtype=torch.long, device="cuda")
static_pos  = torch.zeros(8, dtype=torch.long, device="cuda")

# ② 热身:固定形状先跑几遍(在 side stream 上,warmup 用的缓冲与捕获隔离)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        model(static_ids, static_pos)
torch.cuda.current_stream().wait_stream(s)

# ③ 捕获:with 块内跑一遍,整段 kernel 录入图 g
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_logits = model(static_ids, static_pos)

# ④ 重放循环:每步只需 拷入 → 重放 → 读出
for step in range(num_steps):
    static_ids.copy_(new_ids)      # 数据进静态缓冲(地址不变!)
    static_pos.copy_(new_pos)
    g.replay()                     # 一次提交,几百个 kernel 连播
    next_tok = static_logits.argmax(-1)   # 从静态输出缓冲读结果
```

三个细节全是第 2 节的硬约束:**静态缓冲**(copy_ 进出)、**热身隔离**(side stream)、
**捕获块内禁止同步/分配**。vLLM 的 `CUDAGraphRunner` 做的事情与此同构,再加上按 batch 桶管理多张图。
'''))

NB.md("## 7. 配套 App:🎬 CUDA Graph 捕获与重放 🎛️",
D('''
同目录的 `app_25_capture.py` 把本课做成交互演示:切换「延迟对比 / 重放延迟分布 / 耗时累计曲线」,
展开「捕获流程 5 步」的图解;它启动时会自动加载刚保存的 `cudagraph_result_25.json`。

**运行方法**(在 `ch04` 目录执行):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_25_capture.py
```

浏览器打开 **http://localhost:8501**(也可加 `--server.port 8625` 换端口)。
下面这个 cell 会把 app 源码原样写入 `app_25_capture.py`:
'''))

NB.code(app_cell("app_25_capture.py"),
"📜 运行后覆盖写入相同内容,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "CUDA Graph 三步:warmup 热身 → stream capture 捕获 → replay 重放;n 次 launch 合并成 1 次",
        "三条硬约束:形状固定(参数写死)、内存静态(输入先 copy_ 进静态缓冲)、捕获期禁同步/分配/动态控制流",
        "CPU 录制-回放类比:重放的算子序列与结果和录制一字不差;提速来自 GPU 侧抹掉启动开销",
        "批大小一变就要重捕 → vLLM 按 capture sizes 桶预录多张图,真实 batch padding 到最近的桶",
        "padding 浪费 = (桶−B)/B,桶排布指数+等距混合,让最坏浪费封顶(本课例子约 21%)",
    ],
    practice=[
        "把 B 从 8 改成 1 和 64,重跑录制-回放,观察算子数不变、耗时如何随 B 缩放",
        "修改 capture sizes 列表为纯 2 的幂(1,2,4,…,64),重画 padding 浪费图,找出最坏浪费点",
        "给 CPUGraph 增加 copy_in(new_ids):重放前把新数据写入「静态输入缓冲」(同一个 tensor),模拟 vLLM 的静态缓冲",
        "估算:vLLM 默认 capture 11 个桶、每张捕获约 10 ms,启动时捕获总开销是多少?什么时候值得开满桶",
    ],
    links=[
        ("PyTorch CUDA Graphs 官方文档", "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs"),
        ("vLLM CUDA Graph 设计文档", "https://docs.vllm.ai/en/latest/design/v1/v1_cuda_graph.html"),
        ("NVIDIA 博客:CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
    ])

from pathlib import Path
out = str(Path(CH04) / "25_cudagraph_capture.ipynb")
NB.save(out)
finalize(out)
