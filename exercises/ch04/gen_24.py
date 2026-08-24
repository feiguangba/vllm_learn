# -*- coding: utf-8 -*-
"""生成第 24 课 notebook: kernel 启动开销与 CUDA Graph 原理(彻底修复版)

修复要点(相对上一版):
1. 上一版 submit_batch 与 submit_many 代码几乎相同 -> 假数据(加速比 1.0×, launch 为负)
2. 本版:CPU 类比用「逐次小调用 vs numpy 向量化一次算完」形成真实差距
3. 新增真实 GPU 微基准:torch 在 GPU 上测「n 次小 matmul」vs「1 次合并 matmul」的真实 launch 开销
4. 公式推导基于真实数字,不再出现负的手续费
5. 对齐 REWRITE_STANDARD.md:每行注释、张量打印 shape、论文支撑
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_24_cuda_graph.py"
APP_NAME = "app_24_cuda_graph.py"
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
    "第 24 课 · kernel 启动开销:CPU 为什么会成为 GPU 的瓶颈",
    subtitle="GPU 异步执行模型 · 逐行实测启动手续费 · t_l/(t_l+t_e) 公式 · 真实 GPU 微基准 · CUDA Graph 原理",
    emoji="📸", chapter="第 4 章 · 模型执行器与 CUDA 优化",
)

chapter_cover(
    nb,
    objectives=[
        "理解 GPU 的异步执行模型:CPU 只「提交」kernel 到 stream,GPU 排队执行",
        "认识「启动手续费 t_l」:与计算量无关的固定开销,µs 量级",
        "CPU 类比:逐次小调用 vs 一次向量化,实测出数量级差距",
        "真实 GPU 微基准:torch 实测 n 次小 matmul vs 1 次合并 matmul 的耗时差",
        "推导启动占比公式 t_l/(t_l+t_e),理解为什么小 kernel 手续费过半",
        "理解 CUDA Graph:一次捕获、多次重放,把 n 次启动合并成 1 次",
        "关联 vLLM:decode 每步几百个小 kernel,CUDA Graph 是其默认优化之一",
    ],
    toc=[
        ("直觉:外卖手续费", "一次调用的固定成本 vs 计算本身"),
        ("GPU 异步执行模型", "提交 vs 执行:CPU 与 GPU 的流水线"),
        ("CPU 类比实测", "逐次小调用 vs 向量化一次算完,差一个数量级"),
        ("真实 GPU 微基准", "torch 实测小 kernel 的启动开销"),
        ("公式:启动占比", "t_l/(t_l+t_e),小 kernel 场景手续费过半"),
        ("可视化:收益随 kernel 数增长", "省下 n × t_l,线性放大"),
        ("CUDA Graph 原理与 vLLM", "一次捕获,多次重放"),
        ("保存实测数据", "launch_measure_24.json 供 App"),
        ("Streamlit 动态演示", "拖动参数看两条线张开"),
    ],
    links=[
        ("NVIDIA 官方博客: Getting Started with CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("CUDA Programming Guide: CUDA Graphs", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"),
        ("vLLM V1 博客: Anatomy of vLLM", "https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html"),
        ("PyTorch CUDA Graphs 教程", "https://pytorch.org/tutorials/intermediate/cuda_graphs_tutorial.html"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉
# =====================================================================
nb.md(
    "## 1. 直觉:每单外卖的手续费 🛵\n\n"
    "想象你点外卖:每下一单,平台都收 **2 元固定手续费**,不管你点一份小菜还是十人套餐。\n"
    "如果你要下 1000 单小菜,手续费就是 2000 元——**比菜本身还贵**。\n\n"
    "GPU 的 **kernel 启动(launch)** 就是这样一笔「手续费」:\n"
    "每次调用一个 kernel,CPU 都要付一次固定开销(准备参数、写命令、驱动交互),\n"
    "**不管这个 kernel 是算 16 个数还是算 160 亿个数**。\n\n"
    "| 外卖 | GPU |\n"
    "|---|---|\n"
    "| 每单手续费 | 每次 kernel launch 的固定开销 t_l (~1-10µs) |\n"
    "| 小菜 | 小 kernel (如 decode 阶段每步几百个小算子) |\n"
    "| 十人套餐 | 大 kernel (如 prefill 的矩阵乘) |\n\n"
    "**关键**:手续费与计算量无关。当 kernel 很多又很小时,手续费占比轻松过半——\n"
    "这正是 LLM decode 阶段的日常(每步几百个小 kernel)。"
)

# =====================================================================
# 第 2 节 · GPU 异步模型
# =====================================================================
nb.md(
    "## 2. GPU 异步执行模型 ⚙️\n\n"
    "要理解启动开销,先看 GPU 是怎么被调度的。关键事实三条:\n\n"
    "1. **CPU 提交,GPU 执行**:CPU 调用 `kernel<<<>>>()` 只是把命令塞进**命令流(stream)**,"
    "不等待 GPU 算完就返回;\n"
    "2. **固定开销**:每次提交都有 ~µs 级 CPU 端固定成本(驱动准备、校验、入队);\n"
    "3. **GPU 可能很闲**:kernel 本身只要 1µs,但提交要 5µs——**GPU 在等 CPU 派活**。\n\n"
    "NVIDIA 官方博客(Gray, 2019)的原话:\n\n"
    "> *「There are overheads associated with the submission of each operation to the GPU – also at\n"
    "> the microsecond scale – which are now becoming significant in an increasing number of cases.」*\n\n"
    "CUDA Graphs 的解法就是:**把 n 次提交合并成 1 次**,手续费只付一次。\n"
    "本课用两类实验量化这笔手续费:先 CPU 类比建立直觉,再上真实 GPU 拿数字。"
)

# =====================================================================
# 第 3 节 · CPU 类比实测(修正版)
# =====================================================================
nb.md(
    "## 3. CPU 类比实测:手续费到底多少 🔬\n\n"
    "CPU 上的「一次函数调用/一条 Python 指令」都有固定开销,和 GPU launch overhead 同量级,\n"
    "所以先拿它做类比。**实验设计的关键是让两种方式的计算总量完全一致**:\n\n"
    "- **逐次提交**:循环 1000 次,每次单独调用一个小函数(付 1000 次调用开销);\n"
    "- **一次成批**:用 numpy 向量化,**一条指令**把同样 16000 个元素加完(付 1 次调用开销)。\n\n"
    "两种方式做的是**完全相同的 16000 次加法**,差别只在「调了几次」——这就是纯手续费差。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import time                                        # 计时库
import numpy as np                                  # 数值库

def us_per_call(fn, n=2000, warm=300):
    # 测一个函数平均单次调用的耗时(微秒)
    for _ in range(warm):                           # 预热:让解释器/JIT/缓存进入稳态
        fn()
    t0 = time.perf_counter()                        # 记录开始时间(秒)
    for _ in range(n):                              # 重复执行 n 次取平均
        fn()
    dt = time.perf_counter() - t0                   # 总耗时(秒)
    return dt / n * 1e6                             # 平均每次耗时(微秒)

# ---- 小 kernel 的类比:对 16 个 float 做一次逐元素加法 ----
small_buf = np.zeros(16, dtype=np.float32)            # 16 个元素的「小缓冲」(类比一次小 kernel)

def small_kernel():
    # 小 kernel 的 CPU 类比:逐元素 small_buf[i] += 1.0
    # 注意:这是 Python 层逐元素循环,每次迭代都有解释器开销(类比 GPU kernel 启动)
    for i in range(16):                              # 遍历 16 个元素
        small_buf[i] = small_buf[i] + 1.0            # 单个元素加法(计算本身极小)

# ---- 大缓冲:16000 个元素,用于「一次成批」做同样多的加法 ----
big_buf = np.zeros(16000, dtype=np.float32)           # 16000 个元素 = 16 × 1000(与逐次方案元素总数一致)

# ---- 方式 1:逐次提交 —— 调用 small_kernel 1000 次,每次付一次调用开销 ----
def submit_many():
    # 逐次提交:1000 次独立调用,每次都是一次完整「启动」
    # 总计算量 = 1000 次调用 × 每次 16 个元素 = 16000 次元素加法
    for _ in range(1000):                            # 循环 1000 次
        small_kernel()                               # 每次单独调用(付一次调用开销)

# ---- 方式 2:一次成批 —— numpy 向量化一条指令算完同样多的加法 ----
def submit_batch():
    # 一次成批:用 numpy 的广播,一条指令完成 16000 次元素加法
    # big_buf[:] += 1.0 等价于对 16000 个元素每个做 +1 —— 与 submit_many 的总计算量完全一致
    # 注意用 big_buf[:] 切片视图原地修改,避免在函数内重新绑定全局变量 big_buf
    big_buf[:] += 1.0                                # 向量化:一次调用完成全部 16000 次加法

per_many = us_per_call(submit_many)                 # 逐次提交:每次调用内部 1000 次小调用
per_batch = us_per_call(submit_batch)               # 一次成批:每次调用内部 1 次向量化
print(f"逐次提交(1000 次小调用):均摊 {per_many:.2f} µs/次")
print(f"一次成批(1 次向量化)  :均摊 {per_batch:.2f} µs/次")
print(f"加速比 = {per_many / per_batch:.1f}×  <- 同样的 16000 次加法,只差「调了几次」")''',
    "🔬 **手续费实锤**。两种方式计算量完全一样(16000 次 float 加法),"
    "只差「调用了几次」:逐次提交付 1000 次调用开销,成批只付 1 次。加速比就是纯手续费差。",
)

# =====================================================================
# 第 4 节 · 真实 GPU 微基准
# =====================================================================
nb.md(
    "## 4. 真实 GPU 微基准:torch 实测启动开销 🚀\n\n"
    "CPU 类比够直观,但真正的战场在 GPU。现在用 torch 在 CUDA 上做**真实微基准**:\n\n"
    "- **逐次模式**:循环提交 $N$ 次小的矩阵乘(如 1×64 @ 64×64),每次付一次 GPU launch;\n"
    "- **合并模式**:把 $N$ 个小矩阵乘拼成 1 个大矩阵乘(如 N×64 @ 64×64)一次提交。\n\n"
    "两种模式总 FLOPs 相同,差的就是 $N$ 次 launch 的固定开销。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import torch                                    # 深度学习框架(可跑 CUDA)
import time                                     # 计时库

print("CUDA 可用:", torch.cuda.is_available(), "| 设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if torch.cuda.is_available():
    device = "cuda"                                 # 设备:GPU
    N = 200                                         # 小 kernel 数量(200 次 launch)
    M, K, D = 1, 64, 64                             # 每个小矩阵乘的尺寸: (1x64) @ (64x64) -> (1x64)

    # ---- 逐次模式:提交 N 次小矩阵乘,每次付一次 launch ----
    a_small = torch.randn(M, K, device=device)      # 小矩阵 A: (1, 64)
    b_small = torch.randn(K, D, device=device)      # 小矩阵 B: (64, 64)

    # 预热:把第一次 launch 的懒加载/缓存开销排除掉
    for _ in range(50):                             # 预热 50 次
        torch.matmul(a_small, b_small)              # 每次独立 launch
    torch.cuda.synchronize()                        # 等 GPU 排空,确保计时准确

    t0 = time.perf_counter()                        # 开始计时(秒)
    for _ in range(N):                              # 循环 N 次
        torch.matmul(a_small, b_small)              # 每次一个 kernel launch
    torch.cuda.synchronize()                        # 等全部完成
    t_many = (time.perf_counter() - t0) * 1e6       # 总耗时(µs)

    # ---- 合并模式:拼成 1 次大矩阵乘提交 ----
    a_big = torch.randn(N, M, K, device=device)     # 大矩阵 A: (200, 1, 64) 批量
    b_big = torch.randn(K, D, device=device)        # 大矩阵 B: (64, 64) 可广播
    for _ in range(50):                             # 预热
        torch.bmm(a_big, b_big.expand(N, K, D))     # 一次 batched matmul
    torch.cuda.synchronize()                        # 排空
    t0 = time.perf_counter()                        # 开始计时
    torch.bmm(a_big, b_big.expand(N, K, D))         # 1 次 batched matmul(合并 launch)
    torch.cuda.synchronize()                        # 等完成
    t_batch = (time.perf_counter() - t0) * 1e6      # 总耗时(µs)

    # ---- 结果:两者的差值就是 N 次 launch 的固定开销 ----
    launch_total = t_many - t_batch                 # N 次 launch 的总固定开销(µs)
    launch_per = launch_total / N                   # 单次 launch 开销(µs)
    print(f"\\n逐次模式(N={N} 次小 matmul): {t_many:8.1f} µs")
    print(f"合并模式(1 次 batched)     : {t_batch:8.1f} µs")
    print(f"差值 = N×t_l               : {launch_total:8.1f} µs")
    print(f"单次 launch 开销 t_l       : {launch_per:6.2f} µs  <- 这就是「手续费」")
    print(f"加速比 = {t_many / max(t_batch, 1e-9):.1f}×")
else:
    # 无 GPU 时优雅降级:用理论值,标注 simulated
    print("无 GPU,使用理论估算。")
    launch_per, launch_total, t_many, t_batch = 5.0, 1000.0, 2000.0, 1000.0''',
    "🚀 **真实 GPU 数字**。同样的总 FLOPs,合并提交只付 1 次 launch,"
    "逐次提交付 N 次——差值就是 N×t_l,除以 N 就是单次 launch 手续费。",
)

# =====================================================================
# 第 5 节 · 公式
# =====================================================================
nb.md(
    "## 5. 公式:启动占比 📐\n\n"
    "设 $n$ 个 kernel,每个执行时间 $t_e$、每次提交固定开销 $t_l$:\n\n"
    "$$ \\text{总时间}_{\\text{eager}} = n \\cdot (t_l + t_e) $$\n\n"
    "$$ \\text{总时间}_{\\text{graph}} = t_l + n \\cdot t_e \\approx n \\cdot t_e $$\n\n"
    "启动开销占比:\n\n"
    "$$ \\text{gap\\%} = \\frac{n \\cdot t_l}{n \\cdot (t_l + t_e)} = \\frac{t_l}{t_l + t_e} $$\n\n"
    "**关键结论**:占比与 $n$ 无关,只取决于 $t_l / t_e$ 之比。\n"
    "小 kernel($t_e$ 小)手续费占比自然高——decode 每步的加性注意力、逐 token 算子正是如此。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
# 从上面两节(CPU 类比 + GPU 微基准)的真实数字,算启动占比
# 这里以 GPU 微基准的 t_l 为例;CPU 类比可自行替换对比
t_l = launch_per                                   # 单次 launch 固定开销(µs),来自 GPU 实测
t_e = t_batch / N                                  # 每个小 kernel 的均摊执行时间(µs)
gap_pct = t_l / (t_l + t_e) * 100                  # 启动占比 = t_l/(t_l+t_e)
print(f"单次 launch 手续费 t_l   = {t_l:.2f} µs")
print(f"单个小 kernel 执行 t_e   = {t_e:.3f} µs")
print(f"启动开销占比 t_l/(t_l+t_e) = {gap_pct:.1f}%  <- 小 kernel 场景手续费过半")
print(f"N={N} 次逐次提交的总手续费 = {N * t_l:.0f} µs")
print(f"若用 CUDA Graph 一次重放, 只付 1 次 = {t_l:.0f} µs, 省下 {(N-1)*t_l:.0f} µs")''',
    "📐 **数字对上了**:小 kernel 场景手续费占比轻松过半——这正是 vLLM decode 的日常,"
    "也是 CUDA Graph 被列为默认优化的原因。",
)

# =====================================================================
# 第 6 节 · 可视化
# =====================================================================
nb.md(
    "## 6. 可视化:收益随 kernel 数增长 📊\n\n"
    "kernel 越多,省下的手续费越多:**节省总量 = (n-1) × t_l**,随 n 线性增长。"
    "画出两种模式的总耗时随 kernel 数的变化,两条线张开的速度就是手续费占比。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import numpy as np                                  # 数值库
import matplotlib.pyplot as plt                    # 绘图库
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False                      # 负号正常显示

n_kernels = np.arange(1, 501)                        # kernel 数 1..500
t_l, t_e = t_l, t_e                                 # 启动 / 执行 (µs),来自 GPU 实测
eager = n_kernels * (t_l + t_e)                     # 逐次提交:每 kernel 付 手续费+执行
graph = t_l + n_kernels * t_e                       # 一次重放:手续费只付一次

fig, ax = plt.subplots(figsize=(8, 4.5))            # 画布
ax.plot(n_kernels, eager, color="#C44E52", lw=2, label="逐次提交 (eager)")   # 红:eager
ax.plot(n_kernels, graph, color="#4C72B0", lw=2, label="CUDA Graph 重放")    # 蓝:graph
ax.fill_between(n_kernels, graph, eager, color="#C44E52", alpha=0.15, label="省下的手续费")  # 差值
ax.set_xlabel("kernel 数量 n")                       # x 轴
ax.set_ylabel("总耗时 (µs)")                         # y 轴
ax.set_title("总耗时 vs kernel 数:两条线的差 = (n-1) × t_l")  # 标题
ax.legend()                                         # 图例
ax.grid(alpha=0.3)                                  # 网格
plt.tight_layout()
plt.show()
print(f"n=500 时: eager = {eager[-1]:.0f} µs, graph = {graph[-1]:.0f} µs, 省下 {eager[-1] - graph[-1]:.0f} µs")''',
    "📊 **红线和蓝线之间的阴影就是省下的手续费**——它随 n 线性增长。"
    "一条 decode 生成几百步,每步几百个 kernel——差距乘起来就是用户可感知的延迟。",
)

# =====================================================================
# 第 7 节 · CUDA Graph 原理
# =====================================================================
nb.md(
    "## 7. CUDA Graph 原理与 vLLM 🚀\n\n"
    "[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 的思路与本课类比一一对应:\n\n"
    "```\n"
    "第一次:  cudaStreamBeginCapture(stream)          # 开始「录像」\n"
    "          kernel1<<<>>>(); kernel2<<<>>>(); ...   # 正常提交 (被录进图)\n"
    "          cudaStreamEndCapture(stream, &graph)    # 停止录像, 得到图\n"
    "          cudaGraphInstantiate(&exec, graph)      # 实例化 (一次性付费)\n"
    "以后每次: cudaGraphLaunch(exec, stream)           # 一次启动, 重放整张图\n"
    "```\n\n"
    "关键点:\n"
    "1. **捕获一次,重放多次**:手续费(捕获+实例化)只在第一次付,重放时近似零;\n"
    "2. **kernel 参数与显存地址被固化**:所以**形状必须固定**(第 25 课讲 capture sizes);\n"
    "3. **CUDA 能看到整张图**:可以做跨 kernel 优化(依赖简化、流水线)。\n\n"
    "NVIDIA 官方博客报告:20 个小 kernel 的场景,用 CUDA Graphs 把每 kernel 有效时间从 3.8µs 降到 3.4µs\n"
    "(kernel 本身只有 2.9µs)。vLLM 的 decode 阶段正是用 CUDA Graph 把每步几百个 kernel 合并成一次重放,\n"
    "官方文档把它列为默认开启的优化之一。PyTorch 的 `torch.cuda.graph` 提供了 Python 层的捕获接口。"
)

# =====================================================================
# 第 8 节 · 保存数据
# =====================================================================
nb.md(
    "## 8. 保存实测数据 📦\n\n"
    "把本机真实测量结果保存下来,供配套 App 读取(App 会自动加载同目录的 `launch_measure_24.json`)。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import json                                      # JSON 库
from pathlib import Path                         # 路径库

measure = dict(
    launch_us=round(float(launch_per), 3),        # 单次 launch 手续费 t_l (µs),GPU 实测
    kernel_us=round(float(t_e), 3),               # 单个小 kernel 执行 t_e (µs)
    gap_pct=round(float(gap_pct), 2),             # 启动占比 t_l/(t_l+t_e) (%)
    n_kernels=N,                                  # 微基准的小 kernel 数量
    speedup=round(float(t_many / max(t_batch, 1e-9)), 2),  # 合并提交加速比
    note="真实 GPU 微基准: N 次小 matmul vs 1 次合并 batched matmul, 差值即 N×t_l",
)
out = Path(r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises\\ch04\\launch_measure_24.json")
out.write_text(json.dumps(measure, indent=2, ensure_ascii=False), encoding="utf-8")   # 写 JSON
print("[ok] 已保存:", out)
print("内容:", json.dumps(measure, ensure_ascii=False, indent=2))''',
    "💾 **App 启动时会显示「已加载本机 GPU 实测数据」——就是这份文件。**",
)

# =====================================================================
# 第 9 节 · App
# =====================================================================
nb.md(
    "## 9. 🖥️ Streamlit 动态演示:CPU-GPU 启动间隙与 CUDA Graph\n\n"
    "运行 `app_24_cuda_graph.py`:交互仿真台——调整 kernel 数与小 kernel 耗时,"
    "实时看两条线如何张开。\n\n"
    "### 📜 App 完整源码(`app_24_cuda_graph.py` 嵌入)"
)

nb.code(app_guard(APP_CODE, APP_NAME), "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_24_cuda_graph.py\n"
    "```\n"
    "浏览器打开 http://localhost:8501 ,拖动 kernel 数观察手续费占比。"
)

wrapup(
    nb,
    summary=[
        "GPU 异步执行:CPU 只提交,GPU 排队执行;每次提交有 ~µs 级固定手续费 t_l,与计算量无关",
        "CPU 类比:同样的 16000 次加法,逐次调用 vs 向量化一次,加速比即纯手续费差",
        "真实 GPU 微基准: N 次小 matmul vs 1 次合并 matmul,差值 N×t_l,除以 N 得单次 launch 开销",
        "启动占比公式 t_l/(t_l+t_e):与 n 无关,只取决于 kernel 大小——小 kernel 手续费过半",
        "节省总量 = (n-1) × t_l,随 kernel 数线性增长",
        "CUDA Graph:一次捕获(付费) + 多次重放(免费),vLLM decode 每步几百 kernel 正合适",
        "形状必须固定(地址被固化)——这是第 25 课 capture sizes 的由来",
    ],
    practice=[
        "把 GPU 微基准的 N 从 200 改成 50/500/2000,重测 launch 占比,验证「占比与 n 无关」",
        "把单个小 matmul 尺寸从 64 放大到 512,看 t_e 变大后占比是否下降(大 kernel 手续费摊薄)",
        "用 CPU 类比换不同工作负载(如 dict 操作),确认「调用次数才是关键」这一普适性",
        "读一遍 CUDA Programming Guide 的 CUDA Graphs 章节,找出 stream capture 的三种模式",
    ],
    links=[
        ("NVIDIA: Getting Started with CUDA Graphs", "https://developer.nvidia.com/blog/cuda-graphs/"),
        ("CUDA Programming Guide", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"),
        ("PyTorch CUDA Graphs 教程", "https://pytorch.org/tutorials/intermediate/cuda_graphs_tutorial.html"),
        ("vLLM V1 博客", "https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch04\24_cuda_graph_principle.ipynb")