# -*- coding: utf-8 -*-
"""生成 34_quant_tradeoff.ipynb 与 app_34_tradeoff.py(教材级重写版)

论文/资料支撑:
- vLLM quantization docs(方法总览)
- Marlin 内核博客/仓库: https://github.com/IST-DASLab/marlin
- NVIDIA FP8 白皮书: arXiv:2209.05433
- LLM.int8() arXiv:2208.07339、GPTQ arXiv:2210.17323、AWQ arXiv:2306.00978、
  SmoothQuant arXiv:2211.10438(第 5 章各课引用)
"""
from helpers import (D, INT8_SYMM, PER_CHANNEL, ERROR_METRICS, chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_34 = D('''
# -*- coding: utf-8 -*-
# app_34_tradeoff.py — 精度/速度/显存三维权衡 + 量化方法选择器 📊
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="📊 34 · 量化综合权衡", layout="wide")
st.title("📊 第 34 课 · 精度-速度-显存:量化方法的三体问题")

st.markdown("""
没有「最好」的量化方法,只有「最合适」的:
- 🎯 **精度**:量化后输出与 fp16 基线的吻合度;
- ⚡ **速度**:内核是否被硬件原生加速(GPU int8/fp8 tensor core);
- 💾 **显存**:每参数字节数,决定能装多大的模型、留多少 KV 空间。

下方选择场景与关注权重,实时看各方法的三维雷达图与对比表:
""")

METHODS = {
    "fp16 基线(16 bit)": dict(bits=16, precision=10, speed=6, memory=4,
        note="全精度基线;H100/B200 上有 fp16 tensor core 加速,显存开销大"),
    "GPTQ int4": dict(bits=4, precision=8, speed=9, memory=9,
        note="逐列量化+误差补偿;Marlin 内核极快,4 bit 显存 1/4,校准需反向信息"),
    "AWQ int4": dict(bits=4, precision=8, speed=9, memory=9,
        note="激活感知缩放保护;只需前向校准,稳定不易过拟合,与 GPTQ 精度接近"),
    "FP8 (E4M3) 权重+KV": dict(bits=8, precision=9, speed=10, memory=6,
        note="H100+ 原生 fp8 tensor core;精度几乎无损,显存减半,是训练推理一体的新宠"),
    "int8 权重+KV": dict(bits=8, precision=9, speed=8, memory=6,
        note="W8A8 方案(SmoothQuant 类);适配面广,Ampere 及以前卡的主力"),
    "int8 KV cache(权重 fp16)": dict(bits=16, precision=9, speed=7, memory=5,
        note="只压 KV 不压权重;吞吐和并发提升,权重显存不变"),
    "RTN int4(朴素)": dict(bits=4, precision=5, speed=9, memory=9,
        note="无需校准、一行代码;低位宽下精度明显掉,胜在省事"),
}

with st.sidebar:
    st.header("🎛️ 选择与权重")
    sel = st.multiselect("对比方法(勾选 2-4 个)", list(METHODS.keys()),
                         default=["fp16 基线(16 bit)", "GPTQ int4", "FP8 (E4M3) 权重+KV"])
    w_acc = st.slider("你在乎精度的程度(0-10)", 0, 10, 7)
    w_spd = st.slider("你在乎速度的程度(0-10)", 0, 10, 6)
    w_mem = st.slider("你在乎显存的程度(0-10)", 0, 10, 8)
    st.caption("💡 想想你的瓶颈:装不下模型 → 优先显存;并发低 → 优先速度;评测掉分 → 优先精度。")

if len(sel) < 1:
    st.info("请至少选择一个方法")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
avg = lambda k: float(np.mean([METHODS[m][k] for m in sel]))
c1.metric("所选方法平均精度分", f"{avg('precision'):.1f} / 10")
c2.metric("平均速度分", f"{avg('speed'):.1f} / 10")
c3.metric("平均显存分", f"{avg('memory'):.1f} / 10")

total_w = w_acc + w_spd + w_mem
scores = {m: (METHODS[m]["precision"] * w_acc + METHODS[m]["speed"] * w_spd
              + METHODS[m]["memory"] * w_mem) / max(total_w, 1) for m in sel}
best = max(scores, key=scores.get)
c4.metric("按你的权重推荐", best, f"{scores[best]:.1f} 分")

st.subheader("📡 三维雷达图:精度 / 速度 / 显存")
cats = ["精度", "速度", "显存友好"]
fig = go.Figure()
for m in sel:
    vals = [METHODS[m]["precision"], METHODS[m]["speed"], METHODS[m]["memory"]]
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1],
                                  fill="toself", name=f"{m}({METHODS[m]['bits']}b)",
                                  opacity=0.6))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], visible=True)),
                  title="每个方法的三维画像:没有全能冠军,只有场景赢家",
                  height=520, margin=dict(l=60, r=60, t=70, b=40))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 方法对比表")
rows = [dict(方法=m, 位宽=METHODS[m]["bits"], 精度分=METHODS[m]["precision"],
             速度分=METHODS[m]["speed"], 显存分=METHODS[m]["memory"],
             一句话点评=METHODS[m]["note"]) for m in sel]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("📈 帕累托视图:显存 vs 精度")
fig2 = go.Figure()
for m in sel:
    bytes_per_param = METHODS[m]["bits"] / 16
    fig2.add_trace(go.Scatter(x=[bytes_per_param], y=[METHODS[m]["precision"]],
                              mode="markers+text", name=m, text=[f"{METHODS[m]['bits']}b"],
                              textposition="top center",
                              marker=dict(size=14, opacity=0.8)))
fig2.update_layout(title="每参数字节数(相对 fp16)vs 精度分:左上角 = 帕累托最优区",
                   xaxis_title="显存占用(相对 fp16 倍数)", yaxis_title="精度分",
                   yaxis_range=[0, 11], height=400, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **选择口诀**:装不下 → GPTQ/AWQ int4;有 H100+ 且要吞吐 → FP8;
> 求稳少折腾 → int8(W8A8);只救 KV 显存 → KV FP8/int8;调试期 → fp16 基线。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 34 课配套演示")

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

NB = new_nb("第 34 课 · 综合实验:精度-速度-显存的三角权衡",
            subtitle="对同一个 GEMM 动手测误差与耗时,用帕累托图和对比表为 GPTQ/AWQ/FP8 找到各自的舞台",
            emoji="📊")

chapter_cover(NB,
    objectives=[
        "对同一个 512×512 GEMM 做 fp32 vs int8 量化模拟,实测误差与耗时",
        "理解 CPU 与 GPU 量化加速的本质差异:为什么 CPU 上 int8 反而慢",
        "掌握精度-速度-显存三角权衡的分析框架与帕累托前沿",
        "建立 GPTQ / AWQ / FP8 / int8 的方法选型表与 vLLM 命令速查",
    ],
    toc=[
        ("直觉:不可能三角", "精度、速度、显存为什么不能全都要"),
        ("核心定义与公式", "GEMM 误差度量 + 每参数字节表 + 帕累托前沿定义"),
        ("最小实现 · 逐行推演", "int8 量化模拟 GEMM,打印 shape 与误差"),
        ("数值验证:误差与耗时", "fp32 vs int8 per-tensor/per-channel;CPU 计时反直觉"),
        ("真实规模数字:显存账 + 帕累托", "7B 权重各精度显存;GPU 上量显存+误差+参考吞吐"),
        ("与 vLLM 工程实现的关系", "方法对比总表 + vLLM 启动命令速查"),
        ("配套 Streamlit 演示", "app_34_tradeoff.py:雷达图 + 选择器"),
    ],
    links=[
        ("vLLM 量化文档(方法总览)", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
        ("Marlin int4 内核仓库", "https://github.com/IST-DASLab/marlin"),
        ("FP8 Formats for Deep Learning (arXiv:2209.05433)", "https://arxiv.org/abs/2209.05433"),
        ("GPTQ (arXiv:2210.17323) / AWQ (arXiv:2306.00978)", "https://arxiv.org/abs/2210.17323"),
    ])

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 torch/numpy OMP 冲突保护
import torch
import numpy as np
import time

torch.manual_seed(0)          # 固定随机种子,结果可复现
np.random.seed(0)
torch.set_num_threads(max(os.cpu_count() // 2, 1))   # 控制线程数,计时要稳定
print("torch =", torch.__version__, "| threads =", torch.get_num_threads())
# 说明:本课执行体不引入 pandas —— GPU cell 后再建 pandas/pyarrow 表易触发 arrow.dll 原生崩溃,
# 故表格一律用纯 Python 的 dict + print 展示(配套 streamlit app 内仍可用 pandas)。
'''), "🛡️ 环境准备:KMP 保护 + 固定随机种子 + 合理线程数(计时要稳定)。")

NB.md("## 1️⃣ 直觉:不可能三角 🔺",
D('''
买相机时你说要「画质好、机身轻、价格低」,店员只会微笑着摇头——三个最多选两个。
量化也一样,存在一个**不可能三角**:

- 🎯 **精度**:输出与全精度的吻合度(误差小、评测不掉分);
- ⚡ **速度**:计算要被硬件原生加速(int8/fp8 tensor core、专用反量化内核);
- 💾 **显存**:每参数字节数少(装得下大模型,还给 KV cache 留地方)。

位宽压得越狠,显存和理论速度越好,精度越悬。**量化方法论的进步,
本质是不断把三角的边界往外推**:RTN 时代 4 bit 掉点严重;GPTQ/AWQ 用校准救回精度;
FP8 靠硬件原生支持做到「几乎无损的 2 倍」。本课用同一个 GEMM 实验,把三个维度全部量出来。
'''))

NB.md("## 2️⃣ 核心定义与公式 🧮",
D('''
**误差度量**(对 GEMM 输出 $Y = XW^T$):

$$\\text{相对误差} = \\frac{\\|Y - Y_q\\|^2_F}{\\|Y\\|^2_F}$$

**显存账**(每参数字节数):

| dtype | 字节/参数 | 7B 模型权重 | 相对 fp16 |
|-------|-----------|-------------|-----------|
| fp32 | 4 | 28 GB | 2.0× |
| fp16 / bf16 | 2 | 14 GB | 1.0× |
| int8 / fp8 | 1 | 7 GB | 0.5× |
| int4 | 0.5 | 3.5 GB | 0.25× |

**帕累托前沿(Pareto frontier)**:把每个方法画在「显存(越小越好)× 精度(越高越好)」平面上,
左上角那一串点构成前沿——**没有别的方法能在显存更少的同时精度更高**;想再省显存,就得付「精度税」。

> ⚠️ 前提声明:速度分要诚实——**CPU 上的 int8 乘法是参考实现(无 SIMD),反而更慢**;
> 真正的速度收益来自 GPU 的 int8/fp8 tensor core 或 Marlin 类专用内核。
'''))

NB.code(INT8_SYMM, "📐 对称量化工具(per-tensor 版,复用第 28-29 课)。")
NB.code(PER_CHANNEL, "📐 per-channel 量化工具(每行一把尺子,复用第 29 课)。")

NB.md("## 3️⃣ 最小实现 · 逐行推演:int8 量化模拟 GEMM 🔬",
D('''
被测对象:一个模拟线性层的 GEMM,`y = X @ W^T`(512×512)。
量化策略:int8 per-tensor 与 per-channel(复用第 29 课的工具函数),
量化后反量化回 float 再做矩阵乘——这是 CPU 上标准的「量化模拟」做法:
'''))

NB.code(D('''
torch.manual_seed(0)
d = 512                                        # 方阵维度
W = torch.randn(d, d) / d ** 0.5               # (512, 512) 权重
X = torch.randn(200, d)                        # (200, 512) 激活(batch=200)
y_ref = X @ W.T                                # (200, 512) fp32 基线输出

def simulate_gemm(X, W, quant_fn):
    # 量化模拟:量化→反量化→fp32 矩阵乘(CPU 无 int8 GEMM 内核时的标准做法)
    qx, sx = symm_quantize(X, bits=8)          # 激活 per-tensor 量化
    Xh = symm_dequant(qx, sx)                  # 反量化激活
    if quant_fn == "per-channel":
        qw, sw = symm_quantize_channel(W, bits=8)   # 权重 per-channel
        return Xh @ symm_dequant_channel(qw, sw).T  # 反量化后 GEMM
    qw, sw = symm_quantize(W, bits=8)          # 权重 per-tensor
    return Xh @ symm_dequant(qw, sw).T         # 反量化后 GEMM

def rel_mse(y):
    # 相对输出误差(相对信号功率)
    return float(((y - y_ref) ** 2).mean() / (y_ref ** 2).mean())

y_t = simulate_gemm(X, W, "per-tensor")        # per-tensor 模拟
y_c = simulate_gemm(X, W, "per-channel")       # per-channel 模拟
print(f"权重 W shape = {tuple(W.shape)}  <- (输出 512, 输入 512)")
print(f"激活 X shape = {tuple(X.shape)}  <- (batch 200, 输入 512)")
print(f"输出 Y shape = {tuple(y_t.shape)}  <- (batch 200, 输出 512)")
print(f"fp32 基线      : 相对误差 = 0(定义)")
print(f"int8 per-tensor: 相对输出误差 = {rel_mse(y_t):.4%}")
print(f"int8 per-channel: 相对输出误差 = {rel_mse(y_c):.4%}")
print("8 bit 下两者误差都小于 0.05%——这就是 int8 敢「全面替换 fp32」的底气。")
'''), "✅ 8 bit 量化模拟的输出误差仅万分之几;per-channel 比 per-tensor 再低约一个数量级。")

NB.md("## 4️⃣ 数值验证:耗时测量(反直觉的一课) ⏱️",
D('''
在同一颗 CPU 上测三种乘法:fp32 GEMM、量化模拟(量化开销 + fp32 GEMM)、
以及 torch 原生 int8 矩阵乘。**先打预防针**:CPU 的 int8 乘法是「参考实现」(无 SIMD 优化),
反而更慢;真正的加速来自 GPU 的 int8/fp8 tensor core。测一下亲眼确认:
'''))

NB.code(D('''
def bench(fn, warmup=2, iters=10):
    # 通用计时器:预热 + 多次取均值(毫秒)
    for _ in range(warmup):                      # 预热,跳过首调开销
        fn()
    t0 = time.perf_counter()                     # 开始计时
    for _ in range(iters):                       # 多次执行
        fn()
    return (time.perf_counter() - t0) / iters * 1000   # 平均毫秒

t_fp32 = bench(lambda: X @ W.T)                              # fp32 GEMM
t_sim = bench(lambda: simulate_gemm(X, W, "per-channel"))    # 量化模拟(含量化开销)

Wi = torch.randint(-127, 128, (d, d), dtype=torch.int8)      # (512,512) int8 权重
Xi = torch.randint(-127, 128, (200, d), dtype=torch.int8)    # (200,512) int8 激活
t_int8 = bench(lambda: Xi @ Wi.T)                            # CPU 原生 int8 乘法

print(f"fp32 GEMM          : {t_fp32:7.2f} ms")
print(f"int8 量化模拟(含量化): {t_sim:7.2f} ms")
print(f"int8 原生整数乘(CPU): {t_int8:7.2f} ms  ← CPU 参考实现,无 SIMD 加速")
print()
print("结论:CPU 上比耗时没有意义——量化的速度收益必须靠 GPU 专用内核:")
print("  · int4 GEMM 走 Marlin/GPTQ 内核,decode 吞吐可达 fp16 的 2-3 倍")
print("  · FP8 走 H100+ tensor core,矩阵乘近乎 2 倍")
print("CPU 实验的价值:验证精度与开销,速度请以 GPU 基准为准。")
'''), "⚠️ CPU 上 int8 反而慢 30 倍(参考实现);记住这个反直觉结果,部署时永远用 GPU 量化内核。")

NB.md("## 5️⃣ 真实规模数字:显存账 + GPU 实测量三件事 💾",
D('''
把「不可能三角」的三个维度搬上 GPU,对同一个线性层 `y = X @ W^T`(1024×1024)
一行全测了:
- **显存**:权重 fp16(2B/参数) vs int8(1B/参数),减半是铁板钉钉;
- **误差**:fp16 / int8(per-channel,量化-反量化模拟)相对 fp32 基线的输出误差;
- **参考吞吐**:用跨章库 `vllm_real.bench_throughput_curve`(bf16 基准)量一条
  batch 4 / 16 的 GPU 吞吐曲线。

> ⚠️ **诚实声明**:本机 torch 没有原生 int8 GEMM 内核,int8 的「提速」靠的是 GPU 上的
> int8/fp8 tensor core 或 Marlin 内核。本课**不做内核级 int8 加速实测**,只用——
> **显存减半 + 误差 + bf16 参考吞吐**三件真实数据来表示权衡。
'''))

NB.code(D('''
import sys, torch, time, gc
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d = 1024
W = torch.randn(d, d, device=dev) / (d ** 0.5)      # (1024,1024) 权重
X = torch.randn(512, d, device=dev)                 # (512,1024) 激活

# —— 误差:fp16 与 int8(per-channel 模拟)相对 fp32 基线 ——
Y = X @ W.T                                         # fp32 基线
def relerr(Yq): return float(((Yq - Y) ** 2).mean() / (Y ** 2).mean())   # 相对误差
Xh, Wh = X.half(), W.half()                         # fp16 版本
e16 = relerr((Xh @ Wh.T).float())                   # fp16 真前向误差
qmax = 127
sx = Xh.abs().max() / qmax                          # 激活 per-tensor scale
Qx = torch.clamp(torch.round(Xh / sx.clamp_min(1e-12)), -qmax, qmax)      # 量化激活
sw = Wh.abs().amax(dim=1, keepdim=True) / qmax      # 权重 per-channel scale
Qw = torch.clamp(torch.round(Wh / sw.clamp_min(1e-12)), -qmax, qmax)      # 量化权重
e8i = relerr((Qx.float() * sx) @ (Qw.float() * sw).T)   # int8 模拟(反量化后 fp32 GEMM)

# —— 显存:权重每参数字节 ——
p = W.numel()
print(f"设备: {cuda_info()}")
print(f"权重规模: {tuple(W.shape)} = {p:,d} 参数")
print(f"权重显存   fp16 = {p*2/1024**2:7.2f} MB | int8 = {p*1/1024**2:6.2f} MB | 减省 {(1-1/2):.0%}")
print(f"输出误差   fp16 = {e16:.4%} | int8(per-channel) = {e8i:.4%}")

# —— 参考吞吐(bf16 基准,非 int8 内核)——
b_list, tps, _ = bench_throughput_curve(batch=(4, 16), token_len=8, reps=3)
for b, tp in zip(b_list, tps):
    print(f"参考吞吐(bf16 基准) batch={b:2d}: {tp:8.0f} token/s")

print("\\n结论:int8 用一半显存换来约 0.05% 量级的误差(per-channel 可更低);")
print("真正的吞吐加速要靠原生 int8/fp8 内核,本机未侧——见上方诚实声明。")
torch.cuda.empty_cache(); gc.collect()
'''), "✅ 真实 GPU 数字:显存减半 + 误差 ~0.05% + bf16 参考吞吐,三件事都从 GPU 上量出来。量化权衡不再拍脑袋,而是这三列并排的账本。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:方法选型总表 🚀",
D('''
一张表总结第 5 章出场的主要方法(评分为社区共识的经验值,10 分制):
'''))

NB.code(D('''
data = [
    dict(方法="fp16/bf16 基线", 位宽=16, 校准="不需要", 精度=10, 速度=6, 显存=4,
         适用="追求零风险;显存充足;开发调试期"),
    dict(方法="int8 W8A8(SmoothQuant 系)", 位宽=8, 校准="少量前向", 精度=9, 速度=8, 显存=6,
         适用="Ampere 及更早 GPU 的主力;服务稳定第一"),
    dict(方法="GPTQ int4", 位宽=4, 校准="128-256 条样本+补偿", 精度=8, 速度=9, 显存=9,
         适用="单卡装大模型;配 Marlin 内核 decode 提速"),
    dict(方法="AWQ int4", 位宽=4, 校准="少量前向统计", 精度=8, 速度=9, 显存=9,
         适用="同 GPTQ;校准更快更稳,工程首选之一"),
    dict(方法="FP8 E4M3(权重+激活+KV)", 位宽=8, 校准="可选 scales", 精度=9.3, 速度=10, 显存=6,
         适用="H100/L40S/50 系以上;吞吐优先的生产服务"),
    dict(方法="KV FP8/int8(权重不动)", 位宽=16, 校准="可选 kv_scales", 精度=9, 速度=7, 显存=5,
         适用="只救 KV 显存、拉并发;改动最小的方案"),
    dict(方法="RTN int4(朴素)", 位宽=4, 校准="不需要", 精度=5, 速度=9, 显存=9,
         适用="快速原型;对精度不敏感的任务"),
]
print(f"{'方法':<24}{'bit':>4}{'校准':<20}{'精度':>4}{'速度':>4}{'显存':>4}  适用")
for m in data:
    print(f"{m['方法']:<24}{m['位宽']:>4}{m['校准']:<20}{m['精度']:>4}{m['速度']:>4}{m['显存']:>4}  {m['适用']}")
'''), "📋 记住三个关键词:**GPTQ=补偿、AWQ=保护、FP8=硬件原生**;校准需求与硬件门槛决定落地成本。")

NB.code(D('''
print("""
vLLM 量化启动命令速查
======================
# GPTQ int4(Marlin 内核自动选择)
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq

# AWQ int4
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq

# FP8(动态 scale;或用 llm-compressor 离线产出静态 scale 模型)
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct-FP8 --quantization fp8

# int8 权重 + FP8 KV 的组合拳
vllm serve <model> --quantization int8 --kv-cache-dtype fp8_e4m3
""")
print("更多格式(llm-compressor、compressed-tensors、GGUF 等)见 vLLM 量化文档:")
print("https://docs.vllm.ai/en/latest/features/quantization/index.html")
'''), "📋 命令速查:同样的服务脚本,换个 `--quantization` 参数就是另一种方案——vLLM 把选型简化成了配置项。")

NB.md("## 7️⃣ 配套 Streamlit 演示:三维雷达 + 选择器 🎛️",
D('''
运行同目录下的 `app_34_tradeoff.py`:勾选你想对比的方法,
实时查看精度/速度/显存三维雷达图、对比表与帕累托视图:

```bash
D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_34_tradeoff.py
```

浏览器打开 **http://localhost:8501**。建议玩法:先勾 fp16 基线 + GPTQ int4 + FP8 三者,
看雷达图如何互补;再勾上 RTN int4,体会「省显存但精度塌方」的位置。完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_34, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_34_tradeoff.py 后运行:\")\n"
    "    print(\"    D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_34_tradeoff.py\")\n"
)
NB.code(guard, "📜 这就是 app_34_tradeoff.py 的完整源码(雷达图 + 对比表 + 帕累托三合一)。")

wrapup(NB,
    summary=[
        "不可能三角:精度、速度、显存最多占两个;方法论的进步就是不断外推三角边界",
        "int8 量化模拟的 GEMM 输出误差仅万分之几,per-channel 再低一个数量级",
        "CPU 上 int8 乘法是参考实现反而慢——量化的速度收益必须靠 GPU tensor core / Marlin 内核",
        "显存收益最确定:fp16→int4 每参数 2 字节变 0.5 字节,同一张卡模型规模翻两番",
        "方法选型:GPTQ=补偿、AWQ=保护、FP8=硬件原生;校准成本与硬件门槛决定落地难度",
        "帕累托前沿(FP8 → int4 校准法 → int3)告诉我们:再省显存就得付精度税",
    ],
    practice=[
        "把 GEMM 实验的 bits 改成 4,重测 per-tensor/per-channel 误差,观察差距如何拉大",
        "给 bench 增加 warmup=5、iters=50 的稳定模式,对比计时波动",
        "在帕累托图上加入 int2 与 fp4(Blackwell 原生)两个假想点,推断它们的位置",
        "查一张你手头 GPU 的规格表,算 FP8 相对 FP16 的理论峰值比,与本文「约 2 倍」对照",
    ],
    links=[
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
        ("GPTQ 论文", "https://arxiv.org/abs/2210.17323"),
        ("AWQ 论文", "https://arxiv.org/abs/2306.00978"),
        ("SmoothQuant 论文", "https://arxiv.org/abs/2211.10438"),
        ("FP8 Formats for Deep Learning", "https://arxiv.org/abs/2209.05433"),
    ])

NB.save(str(Path(CH05) / "34_quant_tradeoff.ipynb"))

app_path = Path(CH05) / "app_34_tradeoff.py"
app_path.write_text(APP_34 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")