# -*- coding: utf-8 -*-
"""生成 28_quantization_basics.ipynb 与 app_28_quant_basics.py(教材级重写版)

论文/资料支撑:
- Nagel et al., "A White Paper on Neural Network Quantization", arXiv:2106.08295
- HF Transformers 量化概念指南: https://huggingface.co/docs/transformers/quantization/concept_guide
- PyTorch 量化实践博客: https://pytorch.org/blog/quantization-in-practice/
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference", arXiv:1712.05877
"""
from helpers import (D, INT8_SYMM, ASYMM_QUANT, chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_28 = D('''
# -*- coding: utf-8 -*-
# app_28_quant_basics.py — 量化基础:位宽与量化误差交互演示 📉
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📉 28 · 量化基础", layout="wide")
st.title("📉 第 28 课 · 量化基础:位宽越小,格子越粗,误差越大")

st.markdown("""
把一段光滑的波形塞进 **$2^b$ 个离散格子**里,就是量化。格子越少(位宽 $b$ 越小),
每个格子越宽(scale 越大),舍入误差越大。下方拖一拖滑块,直观感受「精度」是怎么被位宽吃掉的:
- 🧮 **对称量化**:正负各一半,zero-point 固定为 0;
- ⚖️ **非对称量化**:给分布加上零点偏移,适合全为正的激活。
""")

def quantize(x, bits, mode):
    # 统一量化函数:按模式返回 (量化整数, 重建值, scale, 级别数)
    x = np.asarray(x, dtype=np.float64)
    if mode == "对称(symmetric)":
        qmax = 2 ** (bits - 1) - 1          # int 最大正值:2^(b-1)-1
        scale = max(np.max(np.abs(x)) / qmax, 1e-12)   # scale = max|x| / Qmax
        q = np.round(x / scale).clip(-qmax, qmax)      # 量化:round(x/scale)
        x_hat = q * scale                              # 反量化:q * scale
        levels = 2 * qmax + 1                          # 对称可表示级别数
    else:
        qmin, qmax = 0, 2 ** bits - 1        # 非对称用 uint 范围 [0, 2^b-1]
        rmin, rmax = x.min(), x.max()        # 数据的真实范围
        scale = max((rmax - rmin) / (qmax - qmin), 1e-12)  # scale 由范围决定
        zp = np.round(qmin - rmin / scale).clip(qmin, qmax) # zero-point:浮点0 落在哪个整数
        q = np.round(x / scale + zp).clip(qmin, qmax)      # 量化:round(x/scale + zp)
        x_hat = (q - zp) * scale                          # 反量化:(q - zp) * scale
        levels = 2 ** bits                                # 非对称级别数 = 2^b
    return q, x_hat, scale, levels

with st.sidebar:
    st.header("🎛️ 参数")
    bits = st.slider("量化位宽 b(bit)", 2, 16, 8, 1)
    n = st.slider("采样点数量", 32, 512, 128, 8)
    mode = st.radio("量化方式", ["对称(symmetric)", "非对称(asymmetric)"])
    outlier = st.slider("离群值幅度(越大分布越尖)", 0.0, 5.0, 1.0, 0.1,
                        help="给波形叠加一个尖峰,观察离群值如何拉大 scale、恶化整体精度")
    st.caption("离群值是量化的大敌:一个极端值会撑大 scale,让所有格子一起变粗。")

x = np.linspace(-1, 1, n)                          # 均匀采样点作为横轴
signal = np.cos(2 * np.pi * x) + outlier * np.exp(-((x - 0.5) ** 2) / 0.002)  # 波形信号
q, x_hat, scale, levels = quantize(signal, bits, mode)   # 量化整段信号

err = x_hat - signal                             # 量化误差
mse = float(np.mean(err ** 2))                   # 均方误差
maxerr = float(np.max(np.abs(err)))              # 最大绝对误差

c1, c2, c3, c4 = st.columns(4)
c1.metric("量化步长 scale", f"{scale:.5f}")
c2.metric("可表示级别数", f"{levels}")
c3.metric("均方误差 MSE", f"{mse:.3e}")
c4.metric("最大绝对误差", f"{maxerr:.4f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=signal, mode="lines", name="原始浮点信号",
                         line=dict(width=2, color="#4C78A8")))
fig.add_trace(go.Scatter(x=x, y=x_hat, mode="markers", name="量化重建信号",
                         marker=dict(size=5, color="#F2C14E", symbol="diamond")))
fig.add_trace(go.Scatter(x=x, y=err, mode="lines", name="量化误差",
                         line=dict(width=1, dash="dot", color="#E45756"),
                         yaxis="y2"))
fig.update_layout(
    title=f"位宽 {bits} bit · 量化前后对比(误差放大在右轴)",
    xaxis_title="x", height=460,
    yaxis=dict(title="信号值"), yaxis2=dict(title="误差", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("💡 观察:把位宽从 8 拖到 2,黄色重建点明显变稀疏、误差曲线剧烈抖动;"
           "把离群值调大,即使位宽不变,误差也会整体放大——因为一个尖峰撑大了 scale。")

st.subheader("📊 位宽扫描:MSE 随位宽指数下降")
sweep = []
for b in range(2, 17):                              # 扫描位宽
    _, xh, s, lv = quantize(signal, b, mode)        # 重新量化
    sweep.append(dict(bits=b, mse=float(np.mean((xh - signal) ** 2)), levels=lv, scale=s))
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=[s["bits"] for s in sweep], y=[s["mse"] for s in sweep],
                      marker_color="#72B7B2", name="MSE"))
fig2.update_layout(title="位宽 b vs 均方误差 MSE(注意纵轴对数)",
                   xaxis_title="位宽 b(bit)", yaxis_title="MSE", height=360,
                   yaxis_type="log", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **直觉**:量化就像用一把有刻度的尺子去量长度——刻度越粗,量得越粗。
> 对称量化把 0 放在中间(正负各一半刻度),非对称量化则给零点加个偏移,
> 让有限的格子尽可能盖住数据的真实分布范围。
> 参考:[A White Paper on Neural Network Quantization (arXiv:2106.08295)](https://arxiv.org/abs/2106.08295)。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 28 课配套演示")

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

NB = new_nb("第 28 课 · 量化基础:位宽、scale 与 zero-point",
            subtitle="为什么模型「瘦身」会丢精度?从 int8/fp16/bf16 三种数制出发,搞懂对称与非对称量化的一对核心参数",
            emoji="📉")

chapter_cover(NB,
    objectives=[
        "认识三种常见数制:int8、fp16、bf16 的位宽、动态范围与精度差异",
        "建立量化公式的符号表:位宽 b、scale s、zero-point z、量化级别 Qmax 各自是什么",
        "手写对称量化:理解 scale = max|x| / (2^(b-1) − 1) 的来源",
        "手写非对称量化:理解 zero-point 为什么能让浮点 0 精确映射",
        "在真实模型规模上算账:7B 权重 fp16 → int8 省多少显存",
    ],
    toc=[
        ("直觉:给数值「减肥」", "用尺子刻度比喻位宽,引出量化 = 有损压缩"),
        ("核心定义与公式", "量化/反量化映射 + 符号表 + 三种数制对比"),
        ("最小实现 · 逐行推演", "torch 手写对称量化,小张量打印 shape 与中间量"),
        ("数值验证:对称 vs 非对称", "同一批数据两种量化的误差上界验证"),
        ("真实规模数字", "7B/8B 模型权重 fp16→int8 的显存账与 KV 账"),
        ("与 vLLM 工程实现的关系", "vLLM 里 scale/zero-point 怎么存、怎么选"),
        ("配套 Streamlit 演示", "app_28_quant_basics.py:拖动滑块看误差"),
    ],
    links=[
        ("A White Paper on Neural Network Quantization (arXiv:2106.08295)", "https://arxiv.org/abs/2106.08295"),
        ("Jacob et al., Integer-Arithmetic-Only Inference (arXiv:1712.05877)", "https://arxiv.org/abs/1712.05877"),
        ("HF Transformers 量化概念指南", "https://huggingface.co/docs/transformers/quantization/concept_guide"),
        ("PyTorch 量化实践博客", "https://pytorch.org/blog/quantization-in-practice/"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.md("## 1️⃣ 直觉:给数值「减肥」 📏",
D('''
想象你有一把**毫米刻度的尺子**,能量出 1.234 mm 这样的精度;现在换一把**厘米刻度的尺子**,
能量出的只有 1 cm、2 cm…… 任何 1.4 cm 的长度都只能「约等于」1 cm 或 2 cm。
**刻度越粗,量得越粗**——这就是量化。

神经网络里存储参数用的是浮点数(fp16/bf16/fp32),每个数占用 16 或 32 位(bit)。
量化(quantization)就是换一把「更粗的尺子」:用更少的 bit(比如 8 bit 的 int8)去近似原来的数。
好处是显存和计算量直接减半甚至减到 1/4,代价是引入**舍入误差**。

> 📄 这是 PTQ(Post-Training Quantization,训练后量化)的核心设定:模型已经训练好,
> 我们只做「有损压缩」。系统性的数学框架见
> [Nagel et al., *A White Paper on Neural Network Quantization*, 2021](https://arxiv.org/abs/2106.08295)。

为什么大模型特别需要量化?因为参数量动辄 70 亿、上千亿,而显存是稀缺资源——
**显存 = 参数量 × 每参数字节数**,把字节数从 2(fp16)降到 1(int8),同一张卡就能装下两倍的模型。
'''))

NB.md("## 2️⃣ 核心定义与公式:先把符号表建起来 🧮",
D('''
量化的本质是一对映射:

$$x \\xrightarrow{\\text{量化}} q = \\mathrm{round}\\left(\\frac{x}{s} + z\\right), \\qquad
\\hat{x} \\xleftarrow{\\text{反量化}} (q - z) \\cdot s$$

| 符号 | 含义 | 维度/取值 |
|---|---|---|
| $x$ | 原始浮点值(权重或激活) | 标量,实际是一整个张量 $(d_{out}, d_{in})$ |
| $q$ | 量化后的整数值 | 标量,整数,范围 $[q_{min}, q_{max}]$ |
| $\\hat{x}$ | 反量化重建值,近似 $x$ | 与 $x$ 同形状 |
| $b$ | 位宽(bit-width) | int8 → $b{=}8$;int4 → $b{=}4$ |
| $s$ | 量化步长 scale | 正浮点数,「尺子一格有多宽」 |
| $z$ | 零点 zero-point | 整数,浮点 0 精确映射到的整数 |
| $q_{min}, q_{max}$ | 整数范围 | 对称 int8: $[-127,127]$;非对称 uint8: $[0,255]$ |

两种流派:

- **对称量化**:$z = 0$ 固定,浮点 0 → 整数 0。整数用**有符号** $[-127, 127]$。
  $s = \\max|x| / (2^{b-1} - 1)$。公式简单、硬件快,适合关于 0 对称的权重;
- **非对称量化**:$z \\ne 0$,让 $[r_{min}, r_{max}]$ 用满全部 $2^b$ 个格子。整数用**无符号** $[0, 2^b-1]$。
  $s = (r_{max} - r_{min})/(2^b - 1)$,适合 ReLU 激活等全正分布。

**符号表的直觉**:$b$ 决定「有几格」,$s$ 决定「一格多宽」,$z$ 决定「格子从哪开始」。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 torch/numpy 的 OMP 冲突保护
import torch

torch.manual_seed(0)          # 固定随机种子,保证每次运行结果一致

# --------------------------------------------------------------------------
# 三种数制的关键参数:同一个 bit 数,指数/尾数分法不同,范围与精度就不同
# --------------------------------------------------------------------------
def fmt_info(dtype, bits, kind):
    # 打印一个 dtype 的关键参数:int 看范围,浮点看范围+精度 eps
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        info = torch.iinfo(dtype)              # int 类型的元信息
        return f"{kind:6s} | {bits:3d} bit | 范围 = [{info.min}, {info.max}] | 均匀刻度间距 = 1"
    info = torch.finfo(dtype)                  # 浮点类型的元信息
    return f"{kind:6s} | {bits:3d} bit | 范围 ≈ [{info.min:.3g}, {info.max:.3g}] | 精度 eps = {info.eps:.3g}"

print(fmt_info(torch.int8,      8,  "int8"))   # int8: 范围小但均匀
print(fmt_info(torch.float16,  16,  "fp16"))   # fp16: 1符号+5指数+10尾数
print(fmt_info(torch.bfloat16, 16,  "bf16"))   # bf16: 1符号+8指数+7尾数
print(fmt_info(torch.float32,  32,  "fp32"))   # fp32: 1符号+8指数+23尾数

print("\\nfp16 最大  =", torch.finfo(torch.float16).max)     # 65504
print("bf16 最大  =", torch.finfo(torch.bfloat16).max)      # 3.39e38
print("fp32 最大  =", torch.finfo(torch.float32).max)       # 3.40e38
print("\\n→ bf16 的指数位与 fp32 相同(8 位),所以范围几乎一样;")
print("  fp16 尾数 10 位 > bf16 尾数 7 位,所以 fp16 精度更高但范围小得多。")
'''), "✅ 关键结论:bf16「范围像 fp32、精度像 fp16 减半」,所以训练常用 bf16 防溢出;推理量化常用 int8 省显存。")

NB.md("## 3️⃣ 最小实现 · 逐行推演:手写对称量化 🔬",
D('''
先把对称量化的工具函数写出来。**每一行都有注释**,并用一个小张量把每一步的 shape 打出来——
读者能心算验证:比如取 `x = [-1.27, -0.5, 0.0, 0.3, 0.99, 1.27]`,位宽 8。
'''))

NB.code(INT8_SYMM, "📐 **对称量化工具**(`symm_quantize` / `symm_dequant`),本课与第 29-30 课共用。")

NB.code(D('''
# --------------------------------------------------------------------------
# 构造一个 1 维小张量,把公式里的每个量都打印出来
# --------------------------------------------------------------------------
x = torch.tensor([-1.27, -0.5, 0.0, 0.3, 0.99, 1.27])   # (6,) 6 个标量,故意含最大值
print(f"输入张量 x   shape = {tuple(x.shape)}  <- 6 个浮点权重")

q, s = symm_quantize(x, bits=8)      # 量化:返回 (整数 q, 步长 scale)
x_hat = symm_dequant(q, s)           # 反量化:整数 × scale 还原

print(f"量化整数 q   shape = {tuple(q.shape)}  <- 与 x 同形状,值域 [-127,127]")
print(f"步长 scale   shape = {tuple(s.shape)}  <- 标量,整条张量共用一把尺子")
print(f"重建值 x_hat shape = {tuple(x_hat.shape)}  <- 与 x 同形状,≈ 原值")

# 逐元素核对公式 x_hat = q * s
print("\\n原始值 x   :", [f"{v:.3f}" for v in x.tolist()])
print("量化整数 q :", q.tolist())
print("重建值 x_hat:", [f"{v:.3f}" for v in x_hat.tolist()])
print(f"scale = {float(s):.5f}  (即 max|x|=1.27 / 127)")
print(f"0 映射到 {q[2].item()} (浮点0→整数0,零点不动)")
print(f"1.27 映射到 {q[5].item()} (= Qmax=127)")
print(f"最大绝对误差 = {(x - x_hat).abs().max():.4f}  <= s/2 = {float(s)/2:.4f}")
'''), "✅ 逐行推演要点:0 精确映射到 0、最大值映射到 127,最大误差不超过半个格子 s/2——对称量化的全部秘密就在这几行。")

NB.md("## 4️⃣ 数值验证:对称 vs 非对称,误差上界与格子利用率 🥊",
D('''
量化再反量化,$\\hat{x}$ 与 $x$ 的差就是量化误差。均匀量化的误差上界是**半个格子** $s/2$。

对**有偏分布**(比如全为正的 ReLU 激活),对称量化会把一半刻度浪费在负半轴,
等于**白白丢掉 1 bit 精度**。此时非对称量化用 zero-point 把格子用满。用数值验证这个直觉:
'''))

NB.code(ASYMM_QUANT, "📐 **非对称量化工具**(`asymm_quantize` / `asymm_dequant`),zero-point 是关键角色。")

NB.code(D('''
torch.manual_seed(7)
x_sym = torch.randn(1000) * 0.8           # (1000,) 零对称分布(模拟权重)
x_pos = torch.rand(1000) * 6.0            # (1000,) 全正分布(模拟 ReLU 激活)

# --- 对称量化:对零对称分布 ---
q1, s1 = symm_quantize(x_sym, bits=8)              # 量化
err1 = (x_sym - symm_dequant(q1, s1)).abs().max()  # 最大绝对误差
print(f"[对称·零对称数据]   scale={float(s1):.5f}  格子数=255  最大误差={float(err1):.5f}  ≤ s/2={float(s1)/2:.5f}")

# --- 非对称量化:对全正分布 ---
q2, s2, z2 = asymm_quantize(x_pos, bits=8)         # 量化,含 zero-point
err2 = (x_pos - asymm_dequant(q2, s2, z2)).abs().max()
print(f"[非对称·全正数据] scale={float(s2):.5f}  zero_point={int(z2)}  格子数=256  最大误差={float(err2):.5f}  ≤ s/2={float(s2)/2:.5f}")

# --- 对照组:全正数据硬用对称量化 ---
q3, s3 = symm_quantize(x_pos, bits=8)              # 错误搭配
err3 = (x_pos - symm_dequant(q3, s3)).abs().max()
print(f"[对称·全正数据]  scale={float(s3):.5f}  最大误差={float(err3):.5f}  → 误差是非对称的 {float(err3/err2):.1f} 倍!")
print("\\n原因:全正数据用对称,scale 由 max|x|=6 决定,负半边格子全浪费,正半边只有一半刻度可用。")
'''), "✅ 分布越「偏」,对称量化越吃亏——这就是工程上「权重用对称、激活用非对称」的原因(第 29 课展开)。")

NB.md("## 5️⃣ 真实规模数字:7B 模型量化省多少显存 💾",
D('''
公式层面搞定后,看真实模型。设模型参数量为 $P$(如 LLaMA-3-8B 为 $P{=}8\\times10^9$):

$$\\text{权重显存} = P \\times \\text{每参数字节数}$$

| 精度 | 每参数字节 | LLaMA-3-8B 权重 | 相对 fp16 |
|---|---|---|---|
| fp32 | 4 | 32 GB | 2.0× |
| fp16 / bf16 | 2 | 16 GB | 1.0× |
| int8 / fp8 | 1 | 8 GB | 0.5× |
| int4 | 0.5 | 4 GB | 0.25× |

省下的显存干什么?**装 KV cache、开更大并发、上更长上下文**——服务吞吐的三大燃料。
用代码把这张表算出来,再实测一张 GPU 权重张量的量化误差:
'''))

NB.code(D('''
# --------------------------------------------------------------------------
# 显存账:不同精度的每参数字节数 → 8B 模型权重显存
# --------------------------------------------------------------------------
print(f"{'格式':<12}{'字节/参数':>10}{'8B 权重显存':>12}{'相对fp16':>10}")
for name, nbytes in [("fp32", 4), ("fp16/bf16", 2), ("int8/fp8", 1), ("int4", 0.5)]:
    gb = 8e9 * nbytes / 1024 ** 3                 # 8B 参数 × 每参数字节 → GB
    print(f"{name:<12}{nbytes:>10}{gb:>10.1f} GB{nbytes / 2:>11.2f}x")

# --------------------------------------------------------------------------
# 真机验证:一张 (1024, 1024) 的 fp16 权重,量化成 int8 后量出误差
# --------------------------------------------------------------------------
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("\\n设备:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")

torch.manual_seed(0)
N = 1024
W = (torch.randn(N, N, device=dev) / (N ** 0.5)).half()   # 模拟一层 Linear 的 fp16 权重,shape (1024,1024)

qmax = 127                                            # int8 最大正值
s = (W.float().abs().max() / qmax).item()             # per-tensor scale
Q = torch.clamp(torch.round(W.float() / s), -qmax, qmax).to(torch.int8)  # 量化
W_hat = Q.float() * s                                 # 反量化

mse_v = float(((W.float() - W_hat) ** 2).mean())      # 均方误差
cos_v = float(torch.dot(W.float().flatten(), W_hat.flatten()) /
              (W.float().norm() * W_hat.norm() + 1e-12))   # 余弦相似度

print(f"权重规模: {W.shape} = {W.numel():,d} 元素  <- (输出维度, 输入维度)")
print(f"缩放 scale = {s:.6f}  (max|x| / 127)")
print(f"fp16 显存 = {W.numel()*2/1024**2:7.2f} MB | int8 显存 = {Q.numel()*1/1024**2:6.2f} MB")
print(f"显存减半   = 每参数 2B → 1B,省下 {W.numel():,d} 字节")
print(f"量化-反量化: MSE = {mse_v:.3e} | 相对信号功率 = {mse_v/float((W.float()**2).mean()):.4%}")
print(f"            余弦相似度 = {cos_v:.6f} (越接近 1 越好)")

import gc
torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
'''), "✅ 真实 GPU 数字:每参数从 2 字节降到 1 字节(显存减半铁板钉钉),代价是 MSE / 余弦反映出的精度损失。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:scale/zero-point 在哪存 🚀",
D('''
vLLM 并不自己发明量化算法,而是**接入社区量化好格式的模型**,推理时用专门内核加载。关键事实:

1. **离线量化**:模型用 GPTQ / AWQ / FP8 等工具(如 [llm-compressor](https://github.com/vllm-project/llm-compressor))提前量好,
   权重存成 int4/int8 + **scale(和 zero-point)元数据**,vLLM 启动时解析;
2. **格式即约定**:`config.json` 里的 `quantization_config` 字段告诉 vLLM 用哪个
   `QuantizationMethod`(见 `vllm/model_executor/layers/quantization/` 下的
   `gptq.py`、`awq.py`、`fp8.py`、`smoothquant.py` 等);
3. **内核反量化**:Marlin / CUTLASS / Triton 内核在矩阵乘**内部**做「取整数→乘 scale→累加」,
   而不是先把权重反量化回 fp16 再做全精度 GEMM——这正是量化能加速的原因;
4. **工程默认组合**:权重 per-channel 对称 + 激活 per-tensor 非对称,与第 29 课的结论一致。

```bash
# vLLM 加载量化模型的两种方式
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq
vllm serve casperhansen/llama-3-8b-instruct-awq --quantization awq
```

> 📄 量化格式的系统梳理见 HF 官方指南
> [Quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide);
> vLLM 支持的格式见 [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/index.html)。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖一拖,感受尺子变粗 🎛️",
D('''
光看静态图不过瘾?运行同目录下的 `app_28_quant_basics.py`,可以拖动滑块实时改变位宽、采样点、
量化方式与离群值幅度,量化前后对比图和位宽扫描图随之刷新:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_28_quant_basics.py
```

浏览器打开 **http://localhost:8501**。建议:把位宽从 8 一路拖到 2,看黄色重建点如何退化成台阶;
再把「离群值幅度」调大,观察一个尖峰如何撑大 scale、让整体误差一起恶化。完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_28, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_28_quant_basics.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_28_quant_basics.py\")\n"
)
NB.code(guard, "📜 这就是 app_28_quant_basics.py 的完整源码,notebook 与 app 共享同一套量化函数,保证演示与讲解完全一致。")

wrapup(NB,
    summary=[
        "量化 = 用更少的 bit 近似原值,本质是「换一把更粗的尺子」,属于有损压缩(PTQ)",
        "量化公式三件套:位宽 b 决定格子数、scale s 决定格子宽度、zero-point z 决定格子起点",
        "int8 范围小但均匀;fp16 精度高但范围小;bf16 范围大但精度低——三者指数/尾数分配不同",
        "对称量化:scale = max|x|/(2^(b-1)−1),零点固定 0,误差上界 s/2,适合零对称权重",
        "非对称量化:多一个 zero-point 让浮点 0 精确映射、格子用满,适合有偏分布",
        "8B 模型 fp16→int8:权重显存 16GB→8GB 减半,误差仅万分之几量级",
    ],
    practice=[
        "把 symm_quantize 的 bits 改成 4,量化 [−1,1] 上的 1000 个均匀采样点,统计误差分布直方图",
        "推导:为什么对称量化的最大量化误差上界是 s/2?(提示:考虑 round 的舍入规则)",
        "给 asymm_quantize 传入一个全负的向量(如 ReLU 前的激活),观察 zero-point 会落在哪里",
        "对比 int8 量化同一个数 0.5 在「对称」与「非对称」下的重建值,解释为何非对称更准",
    ],
    links=[
        ("A White Paper on Neural Network Quantization", "https://arxiv.org/abs/2106.08295"),
        ("Jacob et al., Integer-Arithmetic-Only Inference", "https://arxiv.org/abs/1712.05877"),
        ("PyTorch 量化基础教程", "https://pytorch.org/blog/quantization-in-practice/"),
        ("HF Quantization concepts", "https://huggingface.co/docs/transformers/quantization/concept_guide"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH05) / "28_quantization_basics.ipynb"))

app_path = Path(CH05) / "app_28_quant_basics.py"
app_path.write_text(APP_28 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")