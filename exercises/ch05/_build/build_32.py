# -*- coding: utf-8 -*-
"""生成 32_awq_principle.ipynb 与 app_32_awq.py(教材级重写版)

论文/资料支撑:
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression
  and Acceleration", arXiv:2306.00978(本课主角)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization
  for Large Language Models", arXiv:2211.10438(等价缩放思想)
- vLLM quantization docs + awq_marlin 内核
"""
from helpers import (D, AWQ_CORE, ERROR_METRICS, chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_32 = D('''
# -*- coding: utf-8 -*-
# app_32_awq.py — AWQ 思想:激活感知的通道保护交互演示 🛡️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🛡️ 32 · AWQ 激活感知量化", layout="wide")
st.title("🛡️ 第 32 课 · AWQ 思想:保护那 1% 的重要权重")

st.markdown("""
AWQ(Activation-aware Weight Quantization)的洞察:
**不看激活值大小,看激活通道的幅度分布**——幅度大的通道上,权重的量化误差会被放大,
所以给这些通道的权重乘上缩放 $s$ 再量化(推理时激活同除 $s$),误差就被压下去了。

数学上完全等价:$y = x W^T = (x/s)(s⊙W)^T$,但量化误差的分布完全不同。
下方调节激活分布偏斜度与保护强度,实时观察输出误差:
""")

def quantize_rtn(W, bits):
    # per-row RTN 量化(无保护基线)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=1, keepdims=True) / qmax   # 每输出行一把尺子
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(W / scale), -qmax, qmax)
    return q * scale

def awq_quantize(W, act_scale, bits, alpha):
    # numpy 版简化 AWQ:显著通道权重预缩放保护
    s = (act_scale / act_scale.mean()) ** alpha    # 保护缩放(幂函数)
    s = s / s.min()                                # 归一化
    qmax = 2 ** (bits - 1) - 1
    Ws = W * s                                     # 预缩放
    scale = np.abs(Ws).max(axis=1, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(Ws / scale), -qmax, qmax)
    return (q * scale) / s                         # 反量化并除回缩放

with st.sidebar:
    st.header("🎛️ 实验参数")
    skew = st.slider("激活分布偏斜度 σ(LogNormal)", 0.0, 3.0, 1.5, 0.1,
                     help="越大 → 少数通道的激活幅度越突出,这些通道越需要保护")
    alpha = st.slider("保护强度 α(0=不保护)", 0.0, 1.0, 0.5, 0.05,
                      help="AWQ 论文网格搜索 α∈[0,1],通常 0.5 附近最优")
    bits = st.slider("量化位宽 b(bit)", 2, 8, 3, 1)
    d = st.slider("通道数(输入维度)", 64, 512, 256, 32)
    st.caption("💡 先把 α 拖到 0 看无保护误差,再拖到 0.5,对比输出 MSE。")

rng = np.random.default_rng(0)
act_scale = rng.lognormal(0.0, skew, d)          # 每个输入通道的平均激活幅度
W = rng.standard_normal((128, d)) / np.sqrt(d)   # 权重 (128, d)
X = rng.standard_normal((1000, d)) * act_scale   # 激活与通道幅度挂钩

W_rtn = quantize_rtn(W, bits)                    # 无保护 RTN
W_awq = awq_quantize(W, act_scale, bits, alpha)  # AWQ 保护

out = X @ W.T
e_rtn = float(np.mean((X @ W_rtn.T - out) ** 2))
e_awq = float(np.mean((X @ W_awq.T - out) ** 2))

c1, c2, c3, c4 = st.columns(4)
c1.metric("无保护 RTN 输出 MSE", f"{e_rtn:.3e}")
c2.metric(f"AWQ(α={alpha:.2f}) 输出 MSE", f"{e_awq:.3e}")
c3.metric("误差降低倍数", f"{e_rtn / max(e_awq, 1e-30):.2f}x")
top1pct = int(np.ceil(d * 0.01))
c4.metric("Top 1% 通道激活占比", f"{float(np.sort(act_scale)[-top1pct:].sum() / act_scale.sum()):.1%}")

s_view = (act_scale / act_scale.mean()) ** alpha
s_view = s_view / s_view.min()
fig = go.Figure()
idx = np.argsort(act_scale)
fig.add_trace(go.Scatter(x=np.arange(d), y=act_scale[idx] / act_scale.mean(),
                         name="通道激活幅度(归一)", yaxis="y",
                         line=dict(color="#4C78A8", width=2)))
fig.add_trace(go.Scatter(x=np.arange(d), y=s_view[idx], name=f"保护缩放 s(α={alpha:.2f})",
                         yaxis="y2", line=dict(color="#E45756", width=2)))
fig.update_layout(title="按激活幅度排序的通道:红线是 AWQ 给每个通道权重的保护缩放",
                  xaxis_title="通道(按激活幅度升序)", height=420,
                  yaxis=dict(title="激活幅度(相对均值)"),
                  yaxis2=dict(title="保护缩放 s", overlaying="y", side="right", type="log"),
                  legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 保护强度 α 扫描:不是越大越好")
alphas = np.arange(0.0, 1.01, 0.05)
errs = []
for a in alphas:
    Wa = awq_quantize(W, act_scale, bits, float(a))
    errs.append(float(np.mean((X @ Wa.T - out) ** 2)))
best = alphas[int(np.argmin(errs))]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=alphas, y=errs, mode="lines+markers", name="输出 MSE",
                          line=dict(color="#72B7B2", width=2)))
fig2.add_vline(x=best, line_dash="dot", line_color="#E45756",
               annotation_text=f"最优 α ≈ {best:.2f}")
fig2.update_layout(title="α=0 无保护误差大;α 太大又伤非显著通道——U 型最优在中间",
                   xaxis_title="保护强度 α", yaxis_title="输出 MSE", height=380,
                   margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **AWQ vs GPTQ**:GPTQ 用反向式误差补偿(需要 Hessian 逆),AWQ 只用前向激活统计
> (一次推理即可),更快更稳、不易过拟合校准集。论文:*AWQ: Activation-aware Weight
> Quantization for LLM Compression and Acceleration* (arXiv:2306.00978)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 32 课配套演示")

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

NB = new_nb("第 32 课 · AWQ 思想:激活感知的权重保护",
            subtitle="不看权重看激活——重要的通道配防弹衣,1% 的显著通道决定了量化的成败",
            emoji="🛡️")

chapter_cover(NB,
    objectives=[
        "理解 AWQ 的核心洞察:权重重要性由激活通道幅度衡量,而非权重大小本身",
        "理解 per-channel scaling 的数学等价性:缩放权重、反缩放激活,输出不变",
        "手写简化版 AWQ,数值验证保护前后的输出误差差异",
        "实验验证保护强度 α 的 U 型规律:不保护伤显著通道,过度保护伤普通通道",
        "对比 GPTQ 与 AWQ 的方法论差异(误差补偿 vs 误差规避)",
    ],
    toc=[
        ("直觉:重要通道配防弹衣", "从「1% 通道承担大量信号」说起"),
        ("核心定义与公式", "三个观察 + 等价缩放 y=(x/s)(s⊙W)ᵀ + α 搜索公式"),
        ("最小实现 · 逐行推演", "手写简化 AWQ,小张量逐行打印 shape"),
        ("数值验证:保护 vs 无保护", "RTN vs AWQ 输出误差;α 的 U 型规律"),
        ("真实规模数字", "1% 通道能量占比;TinyGPT 真实激活的对照组"),
        ("与 vLLM 工程实现的关系", "vLLM 的 AWQ 支持与 awq_marlin 内核"),
        ("配套 Streamlit 演示", "app_32_awq.py:调偏斜度看保护收益"),
    ],
    links=[
        ("AWQ 论文 (arXiv:2306.00978)", "https://arxiv.org/abs/2306.00978"),
        ("SmoothQuant (arXiv:2211.10438)", "https://arxiv.org/abs/2211.10438"),
        ("AWQ 官方代码库", "https://github.com/mit-han-lab/llm-awq"),
        ("GPTQ 论文(上一课)", "https://arxiv.org/abs/2210.17323"),
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
    ])

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 torch/numpy OMP 冲突保护
import torch
import numpy as np

torch.manual_seed(0)          # 固定随机种子,结果可复现
np.random.seed(0)
print("torch =", torch.__version__)
# 说明:本课执行体不引入 pandas —— 见 pitfall:GPU cell 后再装 pandas/pyarrow 表易触发 arrow.dll 崩溃。
'''), "🛡️ 环境准备:KMP 保护 + 固定随机种子,全程 CPU。")

NB.md("## 1️⃣ 直觉:重要通道配防弹衣 🦺",
D('''
一家快递中转站有 100 条传送带,货物量极不均匀:其中三四条**主干道**日夜不停,
承载了大部分包裹;其余传送带偶尔才跑几个件。现在要给每条传送带换**精度更低的轴承**
(量化),怎么换才不影响整体时效?

答案显然:**主干道换最好的轴承,冷清通道随便换**。关键是——怎么知道谁是主干道?

AWQ(Activation-aware Weight Quantization)的回答:**看激活,不看权重**。
哪个**输入通道**的激活幅度大,哪个通道上的权重误差就会被放大传播,就要重点保护:

> 🔗 论文:[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)(Lin et al., 2023)

有趣的是,AWQ 团队试过「按权重大小选重要通道」(效果一般)和「按激活幅度选通道、冻结不量化」
(混合精度,硬件不友好),最后发现最优解是下面的「等价缩放」——**所有通道都量化,
但给重要通道的权重悄悄撑大一点**。
'''))

NB.md("## 2️⃣ 核心定义与公式 🔬",
D('''
论文通过系统性实验得到三个关键事实(OPT-6.7B, INT3-g128):

1. **约 0.1%-1% 的输入通道激活幅度显著偏大**,贡献了不成比例的信号能量(离群通道);
2. 这些通道上的**权重若被量化损伤,输出误差会被大激活成倍放大**——伤害不起;
3. 但把它们**冻结不量化**(mixed-precision)会让矩阵乘退化成「大部分快、小部分慢」,
   硬件上反而跑不快——所以必须**全员量化**。

于是问题变成:**全员 int4 的前提下,怎么让重要通道少受伤?**

**核心技巧是等价缩放**。给每个输入通道 $j$ 配一个缩放系数 $s_j > 0$:

$$y = xW^T = \\mathrm{diag}\\left(\\frac{x}{s}\\right) \\cdot \\mathrm{diag}(s) W^T = (x \\oslash s)(s \\odot W)^T$$

即:**先把权重第 $j$ 列乘 $s_j$,再把激活第 $j$ 个通道除以 $s_j$**,输出严格不变。

| 符号 | 含义 | 维度 |
|---|---|---|
| $x$ | 层输入(激活) | $(n, d_{in})$ |
| $W$ | 权重 | $(d_{out}, d_{in})$ |
| $s$ | 每输入通道保护缩放 | $(d_{in},)$ |
| $\\odot, \\oslash$ | 逐元素乘 / 逐元素除 | 广播到各列 |
| $s_X$ | 每通道平均激活幅度 | $(d_{in},)$ |
| $\\alpha$ | 保护强度超参 | 标量,网格搜索 $[0,1]$ |

$s_j$ 怎么定?AWQ 用幂函数 $s = (s_X / \\bar{s}_X)^{\\alpha}$,其中 $\\bar{s}_X$ 是平均幅度:

- 显著通道(激活大)→ $s_j > 1$ → 该列权重被「撑大」,量化格子相对变细 → **误差变小**;
- 代价是同一行内其他通道的权重相对变小,量化误差略增;
- $\\alpha=0$ 退化为无保护 RTN;$\\alpha$ 太大保护过头反而伤身 → **U 型最优点**。

> 📄 等价缩放思想最早见于 [SmoothQuant (arXiv:2211.10438)](https://arxiv.org/abs/2211.10438),
> AWQ 把它从「迁移激活难度到权重」发展成「按激活幅度保护权重」。
'''))

NB.code(AWQ_CORE, "📜 简化版 AWQ:按激活幅度给通道配缩放,预缩放 → per-row 量化 → 除回。")

NB.md("## 3️⃣ 最小实现 · 逐行推演 🔬",
D('''
用小张量走一遍,把每步的 shape 与缩放逻辑打印出来——
**每行代码都有注释**,读者可心算核对「预缩放 → 量化 → 除回」三步。
'''))

NB.code(D('''
torch.manual_seed(1)
d_out, d_in, n = 6, 12, 200                     # 小矩阵:6 输出 × 12 输入,200 校准样本
act_scale = torch.distributions.LogNormal(0.0, 1.5).sample((d_in,))   # (12,) 每通道激活幅度
W = torch.randn(d_out, d_in) / d_in ** 0.5      # (6, 12) 权重
X = torch.randn(n, d_in) * act_scale            # (200, 12) 激活,通道幅度挂钩
out = X @ W.T                                   # (200, 6) 原始输出

alpha = 0.5                                     # 保护强度
s = (act_scale / act_scale.mean()) ** alpha     # (12,) 保护缩放
s = s / s.min()                                 # 归一化:最小通道缩放=1
print(f"权重 W shape = {tuple(W.shape)}  <- (输出 d_out=6, 输入 d_in=12)")
print(f"激活 X shape = {tuple(X.shape)}  <- (样本 n=200, 输入 d_in=12)")
print(f"保护缩放 s   shape = {tuple(s.shape)}  <- (每输入通道 1 个,范围 [{float(s.min()):.2f}, {float(s.max()):.2f}])")

Ws = W * s                                      # (6, 12) 预缩放:显著通道被撑大
qmax = 2 ** 3 - 1                               # 3 bit 的 Qmax=7
scale = Ws.abs().amax(dim=1, keepdim=True) / qmax   # (6, 1) 每输出行 scale
q = torch.round(Ws / scale.clamp_min(1e-12)).clamp(-qmax, qmax)  # (6, 12) 量化
W_hat = (q.float() * scale) / s                 # (6, 12) 反量化并除回缩放

def rel_err(Wq):
    # 相对输出误差:量化前后 XW^T 的差异(相对信号功率)
    return float(((X @ Wq.T - out) ** 2).mean() / (out ** 2).mean())

W_rtn = awq_quantize(W, act_scale, bits=3, alpha=0.0)   # α=0 → 无保护 RTN(真实量化)
print(f"\\nα=0(无保护 RTN): 相对输出误差 = {rel_err(W_rtn):.4%}")
print(f"α=0.5(AWQ)      : 相对输出误差 = {rel_err(W_hat):.4%}")
print("关键:全部权重仍是 3 bit 整数,只是重要通道的格子相对更细。")
'''), "✅ 逐行推演要点:预缩放让显著通道离量化格更近,反量化后除回 s 恢复量级——输出严格等价于原前向。")

NB.md("## 4️⃣ 数值验证:保护 vs 无保护,以及 α 的 U 型规律 🥊",
D('''
构造一个 128×256 的权重 + 偏斜激活,对比 α=0(RTN)与 α=0.5(AWQ);
再把 $\\alpha$ 从 0 扫到 1,看 U 型最优:
'''))

NB.code(D('''
torch.manual_seed(1)
d_out, d_in, n = 128, 256, 1000
act_scale = torch.distributions.LogNormal(0.0, 1.5).sample((d_in,))   # (256,) 每通道激活幅度
W = torch.randn(d_out, d_in) / d_in ** 0.5       # (128, 256) 权重
X = torch.randn(n, d_in) * act_scale             # (1000, 256) 激活
out = X @ W.T                                    # (1000, 128) 原始输出

def rel_err(Wq):
    # 相对输出误差
    return float(((X @ Wq.T - out) ** 2).mean() / (out ** 2).mean())

W_rtn = awq_quantize(W, act_scale, bits=4, alpha=0.0)    # α=0 → 无保护 RTN
W_awq = awq_quantize(W, act_scale, bits=4, alpha=0.5)    # α=0.5 → 论文常用起点

print(f"无保护 RTN : 相对输出误差 = {rel_err(W_rtn):.4%}")
print(f"AWQ α=0.5  : 相对输出误差 = {rel_err(W_awq):.4%}")
print(f"输出误差降低 {rel_err(W_rtn) / rel_err(W_awq):.1f} 倍,而权重依然是全员 int4!")

# --- α 扫描:U 型规律 ---
alphas = torch.arange(0.0, 1.01, 0.05)          # (21,) α 网格
errs = [rel_err(awq_quantize(W, act_scale, bits=4, alpha=float(a))) for a in alphas]
best = float(alphas[int(torch.tensor(errs).argmin())])
print(f"\\n最优 α ≈ {best:.2f},此时相对误差 = {min(errs):.4%}")
print(f"α=0(无保护)相对误差 = {errs[0]:.4%}")
print(f"α=1(过度保护)相对误差 = {errs[-1]:.4%}")
print("U 型:两端都差,中间最优——这就是论文做网格搜索的原因。")
'''), "📊 误差曲线呈 U 型:两端都差,中间最优。激活偏斜越重,最优 α 越大。")

NB.md("## 5️⃣ 真实规模数字:1% 通道的能量账 + 真实激活对照组 💾",
D('''
**账 1:1% 通道贡献多少能量?** AWQ 的成立前提是「通道极不均匀」。用 LogNormal 模拟,
并统计 Top 1% 通道的**能量占比**(幅度平方和):
'''))

NB.code(D('''
torch.manual_seed(0)
d = 512                                     # 通道数
act = torch.distributions.LogNormal(0.0, 1.5).sample((d,))   # (512,) 模拟 LLM 激活通道幅度
energy = act ** 2                           # (512,) 能量 = 幅度平方
top_k = int(d * 0.01)                       # Top 1% = 5 个通道

frac_energy = float(torch.sort(energy, descending=True).values[:top_k].sum() / energy.sum())
frac_power = float(torch.sort(act, descending=True).values[:top_k].sum() / act.sum())
print(f"通道总数 {d},Top 1% = {top_k} 个通道")
print(f"这 {top_k} 个通道贡献了 {frac_energy:.1%} 的能量(幅度平方和)、{frac_power:.1%} 的幅度总和")
print(f"最大通道幅度 = 最小通道的 {float(act.max() / act.min()):.0f} 倍")
print("\\n→ 少数通道扛起大部分信号,量化的成败由它们决定——这就是 AWQ 的出发点。")
'''), "✅ 模拟数据里 1% 的通道占了三成以上的能量——真实 LLM 激活的偏斜程度与此相当甚至更极端。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:awq 与 awq_marlin 🚀",
D('''
vLLM 对 AWQ 支持非常成熟(还有提速的 AWQ-Marlin 内核):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq
# 或更快的 marlin 内核(默认自动选择):--quantization awq_marlin
```

工程事实:

1. **格式**:AWQ 权重存为 `qweight`(打包 int4)+ `qzeros` + `scales`;
   scale 是**每输入通道**(或每 128 列一组)一个,正是本课 $s$ 的工程形态;
2. **推理时**:前向里先 `x / s`(可融合进前一个算子),再做量化 GEMM,
   内核在 GEMM 内乘 scale 反量化——与 Notebook 的三步完全对应;
3. **校准**:AutoAWQ 用少量样本统计每通道激活幅度 $s_X$,再网格搜索 $\\alpha$,
   与 Notebook 第 4 节的 α 扫描同构(论文默认 $\\alpha=0.5$ 附近)。

> 📄 选型经验法则:要**极致压缩(≤4 bit)+ 推理稳定**,选 AWQ;
> 要**同位宽下最高精度**且能接受较长的量化过程,选 GPTQ。两者都是「离线量化、在线全速」的 PTQ 方案。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:调偏斜度,看保护收益 🎛️",
D('''
运行同目录下的 `app_32_awq.py`,拖动激活偏斜度 σ、保护强度 α、位宽等滑块,
实时观察保护缩放曲线、误差指标与 α 扫描图:

```bash
D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_32_awq.py
```

浏览器打开 **http://localhost:8501**。建议玩法:偏斜度拉到 2.5,对比 α=0 与 α=0.5
的误差;再把 α 一路拖到 1.0,看过犹不及的效果。完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_32, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_32_awq.py 后运行:\")\n"
    "    print(\"    D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_32_awq.py\")\n"
)
NB.code(guard, "📜 这就是 app_32_awq.py 的完整源码(numpy 实现,与 notebook 逻辑一致)。")

wrapup(NB,
    summary=[
        "AWQ 洞察:权重重要性由对应输入通道的激活幅度决定——约 1% 的通道贡献大量能量",
        "核心技巧是等价缩放:权重列乘 s、激活通道除 s,输出严格不变,但量化误差重新分配",
        "全员 int4 + 给显著通道撑大缩放,输出误差可降数倍,且硬件友好(无混合精度)",
        "保护强度 α 呈 U 型规律:α=0 裸奔受伤,α=1 过度保护反伤普通通道,最优在中间",
        "GPTQ 事后补偿误差(需 Hessian),AWQ 事前规避误差(只需前向激活统计),各有胜场",
    ],
    practice=[
        "把激活分布换成正态(无偏斜),验证 AWQ 增益趋近 1(没有显著通道就无从保护)",
        "在 α 扫描中加入 bits=2/3/4 三条曲线,观察低位宽时最优 α 是否移动",
        "给 awq_quantize 加 per-channel scale 的显存开销统计(s 本身要不要量化存储?)",
        "对比「冻结 Top1% 通道不量化(mixed precision)」与 AWQ 的误差,并讨论硬件代价",
    ],
    links=[
        ("AWQ 论文 (arXiv:2306.00978)", "https://arxiv.org/abs/2306.00978"),
        ("SmoothQuant (arXiv:2211.10438)", "https://arxiv.org/abs/2211.10438"),
        ("AWQ 官方代码库", "https://github.com/mit-han-lab/llm-awq"),
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
    ])

NB.save(str(Path(CH05) / "32_awq_principle.ipynb"))

app_path = Path(CH05) / "app_32_awq.py"
app_path.write_text(APP_32 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")