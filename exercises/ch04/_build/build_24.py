# -*- coding: utf-8 -*-
"""生成 24_cuda_graph_principle.ipynb(app_24_cuda_graph.py 已存在,%%writefile 覆盖写入相同内容)"""
from helpers import D, chapter_cover, wrapup, new_nb, CH04, app_src, finalize


def app_cell(name):
    return "%%writefile " + name + "\n" + app_src(name)


NB = new_nb("第 24 课 · kernel 启动开销:CPU 为什么会成为 GPU 的瓶颈",
            subtitle="GPU 跑一个小 kernel 只要几微秒,CPU 提交一个 kernel 也要几微秒 —— 旗鼓相当时,卡的是 CPU",
            emoji="📸")

chapter_cover(NB,
    objectives=[
        "理解 GPU 的异步执行模型:CPU 负责「提交」kernel,GPU 负责「执行」kernel",
        "认识 launch overhead:每次提交约几微秒的固定手续费,与 kernel 大小无关",
        "用 torch CPU 计时类比:1000 次「逐次提交」vs 1 次「一次成批提交」,量化手续费",
        "推导「启动间隙占比」公式,理解为什么 kernel 越碎、CPU 越容易成为瓶颈",
        "看懂 CUDA Graph 的承诺:一串 kernel 一次提交,抹掉中间所有间隙",
        "跑通配套 App,交互式调整 kernel 数 / 启动开销 / 执行时间,观察收益曲线",
    ],
    toc=[
        ("直觉:每单外卖的手续费", "点 1000 单各付一次手续费 vs 拍一张菜谱照单全炒"),
        ("GPU 异步执行模型", "提交队列、stream、几微秒的 launch overhead"),
        ("CPU 类比实测:手续费到底多少", "tiny op 逐次调用 vs 同样工作量一次成批调用"),
        ("公式:启动间隙占比", "gap = n×launch / (n×(launch+exec)),小 kernel 场景轻松过半"),
        ("可视化:收益随 kernel 数增长", "pyecharts 折线 + plotly 柱状图"),
        ("CUDA Graph 原理与 vLLM", "捕获 → 重放;vLLM 用它救 decode 阶段的小 kernel 风暴"),
        ("保存实测数据给 App", "生成 launch_measure_24.json"),
        ("配套 App:📸 CPU-GPU 启动间隙", "streamlit 交互仿真"),
    ],
    links=[
        ("NVIDIA 博客:CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("CUDA 编程指南", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.md("## 1. 直觉:每单外卖的手续费 🛵",
D('''
想象你点外卖 🛵:每下一单,平台都要收 **2 元固定手续费**,不管你点的是一份小菜还是十人套餐。

- **Eager 模式**:想吃 1000 道菜,下 1000 单 —— 手续费付 1000 次;
- **CUDA Graph 模式**:先花点时间把 1000 道菜**拍成一张菜谱**(捕获),之后每次照着菜谱喊一声
  「照单全炒」(重放)—— 手续费只付 1 次。

GPU 世界里的「手续费」就是 **kernel launch overhead**:CPU 通过驱动把一个 kernel 提交到 GPU,
每次大约 **2~10 微秒**,与 kernel 本身干活多少无关。大 kernel 干活几百微秒,手续费无所谓;
但小 kernel 干活只有几微秒 —— **手续费和菜钱一样贵**,此时瓶颈从 GPU 挪到了 CPU:
CPU 拼命提交,GPU 干等下一单,这就是「CPU-bound」。
'''))

NB.md("## 2. GPU 异步执行模型 ⚙️",
D('''
关键事实三条:

1. **异步**:CPU 调用 `torch.add(x, y)` 后**不等 GPU 算完**,把 kernel 丢进队列就返回;
2. **队列(stream)**:GPU 按提交顺序执行 kernel;只要 GPU 手里的活没干完,CPU 可以一路超前提交;
3. **但提交本身有成本**:每次提交要经过 Python → PyTorch 派发 → CUDA 驱动,GPU 侧还有 kernel
   启动的固定开销。**每 kernel 合计约几微秒**。

$$\\text{Eager 总时间} \\approx n \\times (t_{\\text{launch}} + t_{\\text{exec}}),\\qquad t_{\\text{launch}} \\approx 2\\!\\sim\\!10\\,\\mu s$$

decode 阶段正是重灾区:批小、每步只有 `T=B` 个词元,网络里的 kernel 又小又多
(第 22 课数过,一层注意力+MLP 就有十来个算子)。**kernel 越碎越勤,手续费越伤**。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch OMP 库冲突(Windows)
import time, json
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

print("本课在 CPU 上做「类比实测」:CPU 上每次调用一个 torch 算子,同样有 Python 派发 + 调度的固定开销,")
print("它与 GPU 的 launch overhead 结构同源 —— 都是「每单手续费」。真实 GPU 数字见文末说明。")
'''),
"⚙️ 类比实验设计:同一个「小 kernel」= 对 16 个 float 做一次加法;「提交手续费」= 调用一次的固定成本。")

NB.md("## 3. CPU 类比实测:手续费到底多少 🔬",
D('''
实验设计非常干净:

- **Eager 类比**:循环 1000 次,每次 `tiny.add_(1.0)` 处理 16 个元素 —— 1000 次「提交」,每次 16 个元素的活;
- **Graph 类比**:一次 `big.add_(1.0)` 处理 16000 个元素 —— **同样的总工作量(1000×16),只提交 1 次**。

如果手续费真的存在,第二种应该快一个数量级以上。
'''))

NB.code(D('''
def us_per_call(fn, n=4000, warm=400):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6     # 平均每次调用(µs)

WORK = 1000                                   # 「1000 个小 kernel」
tiny = torch.ones(16)                         # 一个小 kernel 的工作量:16 个元素
big = torch.ones(16 * WORK)                   # 全部工作摊平:16000 个元素一次做

eager_us = us_per_call(lambda: tiny.add_(1.0))    # 逐次提交:每 16 元素一次调用
batch_us = us_per_call(lambda: big.add_(1.0))     # 一次成批:16000 元素一次调用

per_el_us = batch_us / (16 * WORK)                # 成批时每个元素的「纯计算」单价
kernel_us = 16 * per_el_us                        # 一个小 kernel 的纯计算时间(µs)
launch_us = max(eager_us - kernel_us, 0.0)        # 每次调用的固定手续费(µs)

print(f"逐次提交(16 元素/次)  : {eager_us:8.2f} µs/次")
print(f"一次成批(16000 元素)  : {batch_us:8.2f} µs/次")
print(f"  -> 摊到每个元素      : {per_el_us*1000:8.2f} ns/元素")
print(f"  -> 小 kernel 纯计算  : {kernel_us:8.3f} µs")
print(f"  -> 每次提交的手续费  : {launch_us:8.2f} µs  (= {eager_us:.2f} − {kernel_us:.3f})")
print(f"  -> 手续费占比        : {launch_us / eager_us * 100:5.1f} %")
'''),
"🔬 CPU 上的派发手续费大约几微秒 —— 和 GPU launch overhead 同一量级。这就是为什么它能当类比。")

NB.code(D('''
# 正式对决:完成「1000 份小工作」,逐次提交 vs 一次成批
t0 = time.perf_counter()
for _ in range(WORK):
    tiny.add_(1.0)
eager_1000_ms = (time.perf_counter() - t0) * 1e3          # 1000 次提交

t0 = time.perf_counter()
for _ in range(50):                                        # 重复 50 次取平均更稳
    big.add_(1.0)
graph_1000_ms = (time.perf_counter() - t0) / 50 * 1e3     # 1 次提交干完同样的活

# 「捕获开销」类比:把 1000 份小工作拼成一张大菜谱(一次性成本)
parts = [torch.ones(16) for _ in range(WORK)]
t0 = time.perf_counter()
torch.cat(parts)
capture_ms = (time.perf_counter() - t0) * 1e3

speedup = eager_1000_ms / max(graph_1000_ms, 1e-12)
print(f"逐次提交 1000 份工作: {eager_1000_ms:8.3f} ms")
print(f"一次成批(图重放类比): {graph_1000_ms:8.3f} ms")
print(f"「捕获」菜谱开销(一次): {capture_ms:8.3f} ms")
print(f"加速比: {speedup:8.1f} x")
'''),
"✅ 同样的总计算量,只是把 1000 次提交合并成 1 次,就快了一个数量级 —— 手续费实锤。")

NB.md("## 4. 公式:启动间隙占比 📐",
D('''
把上面的数字代入通用公式(n 个 kernel,每次提交 $t_l$ µs、执行 $t_e$ µs):

$$\\text{gap\\%} = \\frac{n \\cdot t_l}{n \\cdot (t_l + t_e)} = \\frac{t_l}{t_l + t_e}$$

- 大 kernel($t_e \\gg t_l$):gap ≈ 0,不用折腾;
- 小 kernel($t_e \\approx t_l$):gap ≈ 50%,**一半时间 GPU 在等 CPU 提交** —— CUDA Graph 的主场。

用我们刚测的数字算一算,再看它随 kernel 数如何线性放大。
'''))

NB.code(D('''
gap_pct = launch_us / (launch_us + kernel_us) * 100
print(f"本机类比:t_l = {launch_us:.2f} µs, t_e = {kernel_us:.3f} µs")
print(f"启动间隙占比 = t_l / (t_l + t_e) = {gap_pct:.1f} %")
print("即:这种小 kernel 循环里,约一半时间 GPU(类比中为执行单元)在等提交。")
'''),
"📐 数字对上了:小 kernel 场景的手续费占比轻松过半 —— 这正是 vLLM decode 阶段的日常。")

NB.md("## 5. 可视化:收益随 kernel 数增长 📊",
D('''
kernel 越多,省下的手续费越多:**节省总量 = n × t_l**,与 n 成正比。
先看单点对比柱状图(plotly),再看累计曲线(pyecharts)。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.express as px

bar_df = pd.DataFrame({
    "方案": ["Eager:逐次提交 1000 份", "一次成批(图重放类比)"],
    "耗时(ms)": [eager_1000_ms, graph_1000_ms],
})
fig = px.bar(bar_df, x="方案", y="耗时(ms)", text="耗时(ms)", color="方案",
             color_discrete_sequence=["#e74c3c", "#27ae60"])
fig.update_layout(title="完成同样 1000 份小工作:逐次提交 vs 一次成批(本机 CPU 类比实测)",
                  height=380, showlegend=False)
fig
'''),
"📊 红柱里大部分是「手续费」,绿柱几乎全是干货 —— CUDA Graph 的全部秘密就在这一红一绿之间。")

NB.code(D('''
from pyecharts.charts import Line
from pyecharts import options as opts

ks = np.logspace(1, 5, 40).astype(int)
eager_curve = ks * (launch_us + kernel_us) / 1e3     # ms
graph_curve = ks * kernel_us / 1e3 + capture_ms * 0  # 捕获开销一次摊销,曲线只画纯执行

pl = (
    Line()
    .add_xaxis([str(int(k)) for k in ks])
    .add_yaxis("Eager 累计(ms)", [round(float(v), 2) for v in eager_curve],
               label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False)
    .add_yaxis("CUDA Graph 累计(ms)", [round(float(v), 2) for v in graph_curve],
               label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False)
    .set_global_opts(title_opts=opts.TitleOpts(title="启动间隙随 kernel 数线性累积(对数轴)"),
                     xaxis_opts=opts.AxisOpts(type_="log", name="kernel 数"),
                     yaxis_opts=opts.AxisOpts(type_="log", name="累计耗时(ms)"))
)
pl.render_notebook()
'''),
"📈 两条线的差 = n × t_l,随 kernel 数线性拉大。一条 decode 生成几百步,每步几百个 kernel —— 差距乘起来就是用户可感知的延迟。")

NB.md("## 6. CUDA Graph 原理与 vLLM 🚀",
D('''
[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 的思路与本课类比一一对应:

| 本课类比 | CUDA Graph |
|---|---|
| 拍菜谱 | **捕获(capture)**:把一串 kernel 的启动参数与依赖关系录成一张图 |
| 照单全炒 | **重放(replay)**:一次 `graph.replay()` 提交整张图,全图执行 |
| 手续费付一次 | n 次 launch 合并成 1 次,启动间隙 ≈ 0 |

vLLM 为什么特别需要它?**decode 阶段每步张量极小**(T = 批大小),网络却要完整跑一遍,
几百个又小又碎的 kernel —— 启动开销占比最高可达一半。vLLM 因此对 decode 前向做
CUDA Graph 捕获(下一课详细讲捕获流程与「形状必须固定」的约束)。

⚠️ 关于真实 GPU 数字:NVIDIA 官方与实测报告的 launch overhead 约 **2~10 µs/kernel**,
一次整图 replay 的提交开销约 **几十 µs**(与图中 kernel 数几乎无关)——结构与本课 CPU 类比完全一致。
本练习在 Windows + CPU 环境执行,故用 CPU 派发开销做同构演示。
'''))

NB.code(D('''
# 把本机类比实测保存下来,供配套 App 读取(App 会自动加载同目录的 launch_measure_24.json)
meas = {
    "launch_overhead_us": round(float(launch_us), 3),
    "gpu_kernel_us": round(float(max(kernel_us, 0.05)), 3),
    "eager_1000_ms": round(float(eager_1000_ms), 3),
    "graph_1000_ms": round(float(graph_1000_ms), 4),
    "capture_overhead_ms": round(float(capture_ms), 4),
}
with open("launch_measure_24.json", "w", encoding="utf-8") as f:
    json.dump(meas, f, ensure_ascii=False, indent=2)
print("已保存 launch_measure_24.json:", meas)
'''),
"💾 App 启动时会显示「已加载本机 CPU 类比实测数据」—— 就是这份文件。")

NB.md("## 7. 配套 App:📸 CPU-GPU 启动间隙与 CUDA Graph 🎛️",
D('''
同目录的 `app_24_cuda_graph.py` 把本课的模型做成了**交互仿真台**:
拖动 kernel 数、单次启动开销、单 kernel 执行时间,切换「总耗时对比 / 间隙占比分解 / 间隙随规模增长」
三种视图,还可以打开 Y 轴对数坐标。它启动时会自动加载刚才保存的 `launch_measure_24.json`。

**运行方法**(在 `ch04` 目录执行):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_24_cuda_graph.py
```

浏览器打开 **http://localhost:8501**(也可加 `--server.port 8624` 换端口)。
下面这个 cell 会把 app 源码原样写入 `app_24_cuda_graph.py`:
'''))

NB.code(app_cell("app_24_cuda_graph.py"),
"📜 运行后覆盖写入相同内容,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "GPU 异步执行:CPU 提交 kernel(每次约 2~10 µs 手续费),GPU 按队列执行",
        "Eager 总时间 ≈ n×(launch+exec);小 kernel 时 launch 占比轻松过半,CPU 成为瓶颈",
        "CPU 类比实测:1000×16 元素逐次提交 vs 一次成批提交,同样的计算量差一个数量级",
        "启动间隙占比 = t_l/(t_l+t_e),节省总量 = n×t_l 随 kernel 数线性增长",
        "CUDA Graph = 捕获(录菜谱)+ 重放(照单全炒),n 次 launch 合并成 1 次;vLLM 用它救 decode",
    ],
    practice=[
        "把 WORK 从 1000 改成 100 与 10000,观察加速比如何变化并解释",
        "把 tiny 的元素数从 16 改成 4096(kernel 变大),看手续费占比如何暴跌",
        "在公式 gap% = t_l/(t_l+t_e) 中代入 t_e=200µs 的大 kernel,算出为什么大 kernel 不需要 CUDA Graph",
        "给 App 的「间隙随规模增长」视图换算:一条 500 步、每步 300 kernel 的 decode,总共省下多少毫秒",
    ],
    links=[
        ("NVIDIA 博客:CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("CUDA 编程指南", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"),
        ("PyTorch CUDA Graphs 文档", "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs"),
    ])

from pathlib import Path
out = str(Path(CH04) / "24_cuda_graph_principle.ipynb")
NB.save(out)
finalize(out)
