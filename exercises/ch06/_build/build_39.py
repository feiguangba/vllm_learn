# -*- coding: utf-8 -*-
"""生成 39_data_parallel_dp.ipynb 与 app_39_data_parallel.py(教材级重写版)

论文/资料支撑:
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models", arXiv:1910.02054(DP 的显存冗余问题与 ZeRO)
- Shoeybi et al., "Megatron-LM", arXiv:1909.08053(DP 与 TP 并用)
- PyTorch DistributedDataParallel 文档
"""
from helpers import D, DP_LB, DP_GRAD, chapter_cover, wrapup, new_nb, CH06
from pathlib import Path

APP_39_SRC = D('''
# -*- coding: utf-8 -*-
# app_39_data_parallel.py — 推理数据并行的负载均衡 🧑‍🤝‍🧑
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="🧑‍🤝‍🧑 39 · 数据并行", layout="wide")
st.title("🧑‍🤝‍🧑 第 39 课 · 推理数据并行:多个完整模型,一起接客")

st.markdown("""
**数据并行(DP)** 在推理时最简单:**每张卡各放一份完整的模型副本**,把进来的请求
「分发」给不同副本并行处理。既然每卡都能独立出答案,关键在于**怎么分请求才最均衡**——
有的请求难(耗时久),分配不均会导致某些副本排队、某些副本空闲。本页对比三种分发策略。
""")

@dataclass
class Req:
    rid: int
    arrive: float
    work: float

def make_stream(n, rng, wmin, wmax):
    t = 0.0
    reqs = []
    for i in range(n):
        t += rng.exponential(0.8)
        reqs.append(Req(i, t, float(rng.uniform(wmin, wmax))))
    return reqs

def dispatch_dp(reqs, n_replica, policy):
    loads = [0.0] * n_replica
    lanes = [[] for _ in range(n_replica)]
    rr = 0
    for r in reqs:
        if policy == "轮询 round_robin":
            k = rr % n_replica; rr += 1
        elif policy == "最短队列 least_loaded":
            k = int(np.argmin(loads))
        else:
            k = int(np.random.randint(n_replica))
        lanes[k].append(r)
        loads[k] = max(loads[k], r.arrive) + r.work
    return lanes

def dp_stats(lanes):
    fin, lats = [], []
    for lane in lanes:
        t = 0.0
        for r in lane:
            start = max(t, r.arrive)
            fin.append(start + r.work); lats.append(start + r.work - r.arrive)
            t = start + r.work
    total_work = [sum(r.work for r in lane) for lane in lanes]
    return dict(makespan=max(fin), avg_lat=float(np.mean(lats)),
                imbalance=float(np.std(total_work) / np.mean(total_work)))

with st.sidebar:
    st.header("🎛️ 参数")
    n_req = st.slider("请求数", 8, 80, 30, 1)
    n_rep = st.slider("DP 副本数", 1, 8, 4, 1)
    policy = st.radio("分发策略", ["轮询 round_robin", "最短队列 least_loaded", "随机 random"])
    seed = st.slider("随机种子", 0, 20, 3, 1)
    st.caption("请求耗时随机:有的快有的慢,正是「不均匀」考验负载均衡。")

rng = np.random.default_rng(seed)
stream = make_stream(n_req, rng, 2, 30)
lanes = dispatch_dp(stream, n_rep, policy)
stt = dp_stats(lanes)

c1, c2, c3, c4 = st.columns(4)
c1.metric("副本数", n_rep)
c2.metric("总完成时间", f"{stt['makespan']:.1f}")
c3.metric("平均延迟", f"{stt['avg_lat']:.1f}")
c4.metric("负载不均衡度", f"{stt['imbalance']:.2f}")

st.subheader("⚖️ 各副本承接的工作量")
loads = [sum(r.work for r in lane) for lane in lanes]
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"副本{i+1}" for i in range(n_rep)], y=loads,
                     marker_color="#4C78A8", text=[f"{v:.0f}" for v in loads],
                     textposition="outside"))
fig.update_layout(title=f"每个 DP 副本被分配的请求总耗时(越均衡越好)",
                  yaxis_title="工作量", height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("⏱️ 各请求延迟分布")
lats = []
for lane in lanes:
    t = 0.0
    for r in lane:
        start = max(t, r.arrive)
        lats.append(start + r.work - r.arrive); t = start + r.work
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=lats, nbinsx=20, marker_color="#72B7B2"))
fig2.update_layout(title="请求延迟直方图(左移 = 更快)",
                   xaxis_title="延迟", yaxis_title="请求数", height=340,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **直觉**:「轮询」最公平但不管难易,「随机」最随意、常把多批难请求塞给同一个副本,
> 「最短队列」总是派给最闲的副本 → 负载最均衡、总完成时间最短。
> **推理 DP 之间几乎不通信**(各自独立出答案),所以它主要用来扩吞吐、做负载均衡。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 6 章 · 第 39 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
''')

APP_39 = "%%writefile app_39_data_parallel.py\n" + APP_39_SRC

NB = new_nb("第 39 课 · 数据并行:推理负载均衡与训练梯度同步",
            subtitle="推理 DP 是多个完整副本各接各的请求,训练 DP 则要 AllReduce 同步梯度——两种「数据并行」大不同",
            emoji="🧑‍🤝‍🧑")

chapter_cover(NB,
    objectives=[
        "理解数据并行(DP)的本质:每卡一份完整模型,并行处理不同数据/请求",
        "区分推理 DP 与训练 DP:推理做负载均衡、几乎不通信;训练要 AllReduce 同步梯度",
        "手写请求分发模拟器,对比轮询 / 最短队列 / 随机三种策略的负载均衡度",
        "用 numpy 多副本+平均验证:训练 DP 的梯度同步 = 单卡吃满全量数据的梯度",
    ],
    toc=[
        ("直觉:多柜台银行", "推理 DP = 多个完整模型副本,各接各的请求"),
        ("核心定义与公式", "DP 定义 + 推理/训练两种形态 + 梯度平均恒等式"),
        ("最小实现 · 逐行推演", "手写请求分发模拟器,对比三种策略"),
        ("数值验证", "负载均衡统计 + 梯度 AllReduce = 全量梯度(数学可证)"),
        ("真实规模数字", "GPU 上真实梯度 AllReduce + 推理前向参照"),
        ("与 vLLM 工程实现的关系", "vLLM 的 DP 形态与 PyTorch DDP/Megatron"),
        ("配套 Streamlit 演示", "app_39_data_parallel.py:拖请求数/副本数看负载均衡"),
    ],
    links=[
        ("PyTorch DistributedDataParallel 文档", "https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"),
        ("ZeRO (arXiv:1910.02054)", "https://arxiv.org/abs/1910.02054"),
        ("Megatron-LM 论文(DP 与 TP 并用)", "https://arxiv.org/abs/1909.08053"),
    ])

NB.md("## 1️⃣ 直觉:多柜台银行 🧑‍🤝‍🧑",
D('''
想象一家银行只有 1 个柜台,排队特别长。**数据并行(Data Parallel,DP)** 就是开**好几个一模一样的
柜台**:每个柜台都配了同一个柜员(完整的模型副本),进来的客户(请求)被分配到不同柜台,
大家**同时办业务**。

关键在于两件事:
- **推理 DP**:每卡一份完整模型,请求分发后**各自独立出答案,卡之间几乎不通信**;
- **训练 DP**:每卡在**不同的数据子集**上算梯度,然后必须 **AllReduce 同步梯度**(第 36 课)让所有
  副本保持一致。

所以「数据并行」在推理和训练里长相很不一样,下面分别讲。
'''))

NB.md("## 2️⃣ 核心定义与公式 🧮",
D('''
**数据并行(DP)**:每张卡持有**一份完整的模型副本**,并行处理**不同的数据/请求**。

- **推理 DP**:请求被分发给不同副本,各自独立生成答案 → 几乎零跨卡通信,
  核心问题是**负载均衡**(怎么分请求最均衡);
- **训练 DP**:一个 batch 被切成 $R$ 份,副本 $r$ 在自己那份上算梯度 $\\nabla_r$,
  然后用 **AllReduce 求平均** 得到一致梯度。

**梯度平均恒等式**(训练 DP 正确的数学保证):

$$\\nabla = \\frac{1}{R}\\sum_{r=1}^{R} \\nabla_r \\;\\Longrightarrow\\; \\text{等于单卡吃满全量数据的梯度}$$

因为线性回归/神经网络的损失对数据的梯度是**可加平均**的。也就是说:分布式训练(数据并行)
在数学上**严格等价**于单卡用全量数据训练。

| 符号 | 含义 |
|---|---|
| $R$ | DP 副本数 |
| $\\nabla_r$ | 副本 $r$ 在本地 batch 上算的梯度 |
| $\\nabla$ | AllReduce 平均后的梯度 |
| $\\text{makespan}$ | 所有请求都完成的总时间 |
| $\\text{imbalance}$ | 负载不均衡度(各副本工作量标准差/均值) |
'''))

NB.md("## 3️⃣ 最小实现 · 逐行推演:请求分发模拟器 ⚖️",
D('''
推理 DP 的核心问题不是通信,而是**怎么把请求分得均衡**。因为请求的耗时不均匀(有的 prompt 长、
生成多、难算),如果一股脑轮询分配,可能某个副本积压了一堆难请求,其他副本却闲着。

我们写一个请求分发模拟器,对比三种策略。**每行代码都有注释**:
'''))

NB.code(DP_LB, "🎯 对比三行输出:`least_loaded` 的 makespan(总完成时间)最短、imbalance 最低;`random` 最差——**智能分发比公平分发更重要**。")

NB.md("## 4️⃣ 数值验证:训练 DP 的梯度同步 🧮",
D('''
训练时,每张卡只在**自己分到的数据子集**上计算梯度。为了让大家最后学到的模型一样,必须把梯度
同步起来——做法是 **AllReduce 求平均**。用 numpy 验证这个恒等式:
'''))

NB.code(DP_GRAD, "✅ 3 个副本各算局部梯度,AllReduce 平均后的结果与「单卡全量梯度」完全一致(数值差 ~1e-16)——**分布式训练 = 全量训练**的数学保证。")

NB.md("## 5️⃣ 真实规模数字:GPU 上真实梯度 AllReduce ⚡",
D('''
把第 4 节的 numpy 梯度同步搬到 **RTX 5060 的真张量**上:3 个「副本」各自在分到的样本子集上算梯度,
再用真实 torch 做一次 AllReduce(平均),与单卡全量梯度逐点对比;同时测一次真实推理前向(prefill)
时长作参照。这个对比正是「训练 DP 与推理 DP 的不同」的物理证据——**训练要同步梯度,推理几乎不通信**。
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
import gc, time
import torch
from vllm_real import cuda_info, bench_prefill_decode

print("设备:", cuda_info())
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 训练侧:真实 GPU 上的梯度同步
N, D = 6000, 32
Xall  = torch.randn(N, D, device=dev)              # (6000, 32) 全量特征
y     = Xall @ torch.randn(D, device=dev) + 0.3    # (6000,) 标签
print(f"特征 Xall shape = {tuple(Xall.shape)}  <- (样本 N=6000, 特征 D=32)")
print(f"标签 y   shape = {tuple(y.shape)}  <- (样本 N=6000,)")
def lin_grad(X, y, w):
    # 均方误差下线性回归的梯度:2/N * X^T (Xw - y)
    return 2 / len(X) * X.T @ (X @ w - y)
w = torch.randn(D, device=dev) * 0.1               # (32,) 初始权重
idx = torch.chunk(torch.arange(N), 3)              # 3 个 DP 副本各分 1/3 样本
locs = [lin_grad(Xall[i], y[i], w) for i in idx]   # 各副本局部梯度

torch.cuda.synchronize()
t0 = time.perf_counter()
avg = torch.stack(locs).mean(0)                    # AllReduce(平均)在做的事
torch.cuda.synchronize()
ar_ms = (time.perf_counter() - t0) * 1e3

full = lin_grad(Xall, y, w)                        # 单卡全量梯度
ok = torch.allclose(avg, full, atol=1e-5)
print("真实 GPU: AllReduce 平均梯度 vs 单卡全量梯度:",
      "✅ 一致" if ok else "❌ 不一致", f"(最大差 = {float((avg-full).abs().max()):.2e})")
print(f"真实 torch 一次梯度 AllReduce(平均,3 副本): {ar_ms:.3f} ms")

# 推理侧参照:一次真实 prefill 前向多少 ms
b = bench_prefill_decode(d=256, layers=4, L=128, reps=5)
print(f"推理侧:一次真实 prefill(128 token) ≈ {b['prefill_ms']:.3f} ms")
print("→ 训练 DP 每步都要做梯度 AllReduce;推理 DP 各副本独立前向、几乎零通信——两种 DP 开销结构完全不同。")

gc.collect(); torch.cuda.empty_cache()
'''), "⚡ **真机数字**:梯度 AllReduce 在数学上让多副本等价于全量训练(逐点一致),而推理前向几乎不需要跨卡通信——这就是「训练 DP 要同步、推理 DP 只做负载均衡」的实测佐证。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:DP 的形态 🚀",
D('''
vLLM 推理里「数据并行」的形态,和训练里的 DP 不完全一样:

1. **推理:多实例/多副本 serving**:vLLM 官方建议把多个 LLM 实例(各占几卡 TP)挂到同一前端
   (如 Ray Serve、Nginx),请求按负载均衡分发——这就是本课第 3 节的「推理 DP」;
   副本之间完全独立,只做负载均衡,几乎不通信;
2. **单引擎内的 DP**:vLLM 的 PP 实现里,跨 PP 组的相同 stage 会形成 data-parallel 副本组,
   分别服务不同的请求(这是「DP 藏在 PP 内部」的形态);
3. **训练:PyTorch DDP / Megatron / ZeRO**:
   - 朴素 DP:每卡完整模型 + 每步梯度 AllReduce(PyTorch `DistributedDataParallel`);
   - 问题:模型状态(权重/梯度/优化器)**每卡冗余一份**,显存浪费严重;
   - **ZeRO**([arXiv:1910.02054](https://arxiv.org/abs/1910.02054))把这三类状态**切分**到各卡,
     显存随 DP 度线性下降,还能保持 DP 的通信量级——这是 DP 的进阶形态。

```bash
# 训练示例(PyTorch DDP)
torchrun --nproc_per_node=4 train.py --data-parallel
```

> 📄 选型直觉:能 DP 就不用 TP(通信更省);模型放不下再叠 TP/PP——下一课(40)讲 vLLM 如何
> 用共享内存加速小消息 AllReduce,第 41 课讲三者组合。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖请求数与副本数,看负载均衡 🎛️",
D('''
运行 `app_39_data_parallel.py`,拖动**请求数、DP 副本数、分发策略、随机种子**,
实时看各副本承接的工作量与延迟分布:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_39_data_parallel.py
```

浏览器打开 **http://localhost:8501**。建议:把策略从「轮询」切到「最短队列」,看负载柱状图变平、
总完成时间变短;再把副本数从 1 加到 4,看延迟直方图整体左移。完整源码如下:
'''))

NB.code(APP_39, "📜 这就是 app_39_data_parallel.py 的完整源码,notebook 与 app 共享同一套请求分发模拟器。")

wrapup(NB,
    summary=[
        "DP 的本质:每卡一份完整模型,并行处理不同数据/请求",
        "推理 DP 几乎不通信,重点是请求分发策略的负载均衡;最短队列通常最优",
        "训练 DP 要 AllReduce 梯度平均,且数学上等价于单卡全量训练",
        "DP 是「加机器」扩吞吐,TP 是「拆矩阵」摊显存;能 DP 就不用 TP(通信更省)",
        "朴素 DP 每卡冗余一份模型状态;ZeRO 把状态切分,显存随 DP 度线性下降",
    ],
    practice=[
        "修改 make_stream 的耗时范围(wmin/wmax),让请求更难/更均匀,观察哪种策略更占优",
        "给 dispatch_dp 增加一种「按预估耗时最长的副本优先避让」的策略并对比",
        "把 DP_GRAD 的副本数 R 改成 6,重跑梯度同步验证仍一致",
        "用 pyecharts 画「副本数 vs 总完成时间」曲线,找扩副本的收益递减点",
    ],
    links=[
        ("PyTorch DDP 文档", "https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"),
        ("ZeRO (arXiv:1910.02054)", "https://arxiv.org/abs/1910.02054"),
        ("Megatron-LM 论文", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH06) / "39_data_parallel_dp.ipynb"))

app_path = Path(CH06) / "app_39_data_parallel.py"
app_path.write_text(APP_39_SRC + "\n", encoding="utf-8")
print(f"[ok] {app_path}")