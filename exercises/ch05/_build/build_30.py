# -*- coding: utf-8 -*-
"""生成 30_quantization_error.ipynb 与 app_30_quant_error.py(教材级重写版)

论文/资料支撑:
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale",
  arXiv:2208.07339(离群值问题的经典研究)
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference", arXiv:1712.05877(clipping 讨论)
- Nagel et al., "A White Paper on Neural Network Quantization", arXiv:2106.08295
"""
from helpers import (D, INT8_SYMM, ERROR_METRICS, chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_30 = D('''
# -*- coding: utf-8 -*-
# app_30_quant_error.py — 量化误差分析:MSE/SNR 与校准方法交互演示 📏
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📏 30 · 量化误差分析", layout="wide")
st.title("📏 第 30 课 · 量化误差分析:离群值、重尾与校准的攻防战")

st.markdown("""
量化误差不只取决于位宽,更取决于**数据分布的形状**:
- 🐭 正态分布「温柔均匀」,量化轻松;
- 🦎 重尾分布(t 分布、离群值)会让 max 校准的 scale 被极端值撑爆;
- ✂️ **钳位(clipping)+ 分位数校准**:主动舍弃最极端的尾巴,换取全体更细的格子。

下方选择分布与校准方法,实时观察误差指标与 MSE-分位曲线:
""")

def quantize_clip(x, bits, clip_pct):
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    thr = np.percentile(np.abs(x), clip_pct) if clip_pct < 100 else np.abs(x).max()
    thr = max(thr, 1e-12)
    scale = thr / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale, scale

def metrics(x, xh):
    err = xh - x
    mse = float(np.mean(err ** 2))
    sig = float(np.mean(x ** 2))
    snr = 10.0 * np.log10(sig / max(mse, 1e-30))
    return mse, snr, float(np.max(np.abs(err)))

DISTS = {
    "正态 N(0,1)":        lambda rng, n, k: rng.normal(0, 1, n),
    "拉普拉斯(尖峰)":     lambda rng, n, k: rng.laplace(0, 1, n),
    "t 分布 df=3(重尾)":  lambda rng, n, k: rng.standard_t(3, n),
    "正态 + 离群值":       lambda rng, n, k: np.where(rng.random(n) < 0.01,
                                               rng.normal(0, 1, n) * (1.0 + k * 9.0),
                                               rng.normal(0, 1, n)),
}

with st.sidebar:
    st.header("🎛️ 实验参数")
    dist_name = st.selectbox("分布类型", list(DISTS.keys()))
    k = st.slider("离群值强度(仅'正态+离群值'生效)", 0.0, 3.0, 1.0, 0.1)
    bits = st.slider("量化位宽 b(bit)", 2, 10, 4, 1)
    clip = st.slider("校准分位数(%)", 90.0, 100.0, 99.9, 0.1,
                     help="100 = max 校准(不钳位);99.9 = 把最极端 0.1% 的值钳掉")
    n = st.slider("采样点数", 2000, 50000, 20000, 2000)

rng = np.random.default_rng(0)
x = DISTS[dist_name](rng, n, k)                      # 生成数据

xh_max, s_max = quantize_clip(x, bits, 100.0)       # max 校准
xh_pct, s_pct = quantize_clip(x, bits, clip)        # 分位校准

mse_max, snr_max, mae_max = metrics(x, xh_max)
mse_pct, snr_pct, mae_pct = metrics(x, xh_pct)

c1, c2, c3 = st.columns(3)
c1.metric("max 校准 MSE", f"{mse_max:.3e}", f"SNR {snr_max:.1f} dB")
c2.metric(f"{clip}% 校准 MSE", f"{mse_pct:.3e}", f"SNR {snr_pct:.1f} dB")
c3.metric("误差倍数(max/钳位)", f"{mse_max / max(mse_pct, 1e-30):.2f}x")

fig = go.Figure()
fig.add_trace(go.Histogram(x=x, name="原始分布", opacity=0.5, marker_color="#4C78A8", nbinsx=120))
fig.add_trace(go.Histogram(x=xh_max, name="max 校准重建", opacity=0.5, marker_color="#E45756", nbinsx=120))
fig.add_trace(go.Histogram(x=xh_pct, name=f"{clip}% 钳位重建", opacity=0.5, marker_color="#F2C14E", nbinsx=120))
fig.update_layout(barmode="overlay", title=f"{dist_name} · {bits} bit · 量化前后分布对比",
                  xaxis_title="数值", yaxis_title="频数", height=420,
                  legend=dict(orientation="h", y=1.14), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 校准分位数扫描:MSE 的 U 型曲线")
pcts = np.arange(80.0, 100.01, 0.5)
mses = []
for p in pcts:
    xh, _ = quantize_clip(x, bits, float(p))
    mses.append(metrics(x, xh)[0])
best = pcts[int(np.argmin(mses))]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=pcts, y=mses, mode="lines", name="MSE",
                          line=dict(color="#72B7B2", width=2)))
fig2.add_vline(x=best, line_dash="dot", line_color="#E45756",
               annotation_text=f"最优 ≈ {best:.1f}%")
fig2.add_vline(x=100.0, line_dash="dot", line_color="#999",
               annotation_text="max 校准")
fig2.update_layout(title="钳位分位 vs MSE:左边格子太粗(钳太狠),右边离群值撑爆 scale(不钳)",
                   xaxis_title="校准分位数(%)", yaxis_title="MSE", height=380,
                   yaxis_type="log", margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **校准(calibration)的本质**:找一个 clip 阈值 $c$,让「钳位误差 + 舍入误差」总和最小。
> 重尾越重,最优 $c$ 离 max 越远——这正是 GPTQ/AWQ 用几百条真实样本做校准集的原因:
> 用经验分布替你回答「哪些值可以牺牲」。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 5 章 · 第 30 课配套演示")

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

NB = new_nb("第 30 课 · 量化误差分析:MSE、SNR 与校准",
            subtitle="同样是 4 bit,正态分布轻松拿捏、重尾分布直接躺平——离群值钳位与 percentile 校准的攻防战",
            emoji="📏")

chapter_cover(NB,
    objectives=[
        "掌握量化误差的两大指标:MSE 与信噪比 SNR(分贝)",
        "理解分布形状如何决定量化难度:正态 vs 拉普拉斯 vs t 分布重尾",
        "搞懂离群值的破坏机制:一个极端值撑大 scale,全体格子变粗",
        "手写钳位(clipping)+ percentile 校准,实验对比 max 校准 vs 99.9% 分位校准",
    ],
    toc=[
        ("直觉:一间宿舍一个空调", "离群值如何绑架全体精度的比喻"),
        ("核心定义与公式", "MSE 与 SNR 的定义 + 符号表 + 6.02b dB 直觉"),
        ("最小实现 · 逐行推演", "手写量化误差工具箱,小张量逐行打印"),
        ("数值验证:位宽扫描与分布对决", "每少 1 bit 降 ~6 dB;四种分布同台竞技"),
        ("真实规模数字:离群值攻防", "GPU 上 200 万元素重尾权重,max vs 99.9% 钳位"),
        ("与 vLLM 工程实现的关系", "GPTQ/AWQ 为什么需要校准集、校准集怎么选"),
        ("配套 Streamlit 演示", "app_30_quant_error.py:实时调参看误差"),
    ],
    links=[
        ("LLM.int8():离群值与 8bit 矩阵乘 (arXiv:2208.07339)", "https://arxiv.org/abs/2208.07339"),
        ("Jacob et al., Integer-Arithmetic-Only Inference (arXiv:1712.05877)", "https://arxiv.org/abs/1712.05877"),
        ("A White Paper on Neural Network Quantization (arXiv:2106.08295)", "https://arxiv.org/abs/2106.08295"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 torch/numpy OMP 冲突保护
import torch
import numpy as np

torch.manual_seed(0)          # 固定随机种子,结果可复现
np.random.seed(0)
print("torch =", torch.__version__)
# 说明:本课执行体不引入 pandas —— GPU 量化 cell 后再建 pandas/pyarrow 表易触发 arrow.dll 原生崩溃,
# 故表格一律用纯 Python 的 dict + print 展示。
'''), "🛡️ 环境准备:KMP 保护 + 固定随机种子。")

NB.md("## 1️⃣ 直觉:一间宿舍一个空调 🎛️",
D('''
想象一间四人宿舍共用一台空调,温度档位(量化格子)是固定的。
如果三位同学觉得 24℃ 舒服,第四位非要 16℃——为了迁就这个「离群值」,
空调只能设成一个谁都难受的折中档。**一个极端样本,绑架了所有人的精度**。

量化的世界一模一样:对称量化用 $\\max|x|$ 定 scale,
分布里只要出现一个很大的离群值,scale 就被撑大,于是:
- 绝大多数「正常」值挤在少数几个格子里,舍入误差巨大;
- 那个离群值倒是量化得很准——但它只有一个啊!

LLM 的激活分布恰恰以**重尾、带离群值**著称(尤其 Transformer 的 outlier features,
见 [LLM.int8() 论文](https://arxiv.org/abs/2208.07339)),所以「怎么对付离群值」
是量化工程的核心战场。武器有两件:**钳位(clipping)** 与 **校准(calibration)**。
'''))

NB.md("## 2️⃣ 核心定义与公式:两把误差尺子 📐",
D('''
**均方误差 MSE**(量化误差的「平均火力」):

$$\\mathrm{MSE} = \\frac{1}{n}\\sum_i (x_i - \\hat{x}_i)^2$$

MSE 有量纲(是值的平方),不同量级的数据没法直接比。**信噪比 SNR** 把误差放到
信号功率的尺度上,再用分贝(dB)表达:

$$\\mathrm{SNR_{dB}} = 10 \\log_{10} \\frac{\\sum_i x_i^2}{\\sum_i (x_i - \\hat{x}_i)^2}$$

| 符号 | 含义 |
|---|---|
| $x_i$ | 第 $i$ 个原始浮点值 |
| $\\hat{x}_i$ | 第 $i$ 个量化重建值 |
| $n$ | 元素总数 |
| $\\sum_i x_i^2$ | 信号功率(分子) |
| $\\sum_i (x_i - \\hat{x}_i)^2$ | 噪声功率(分母) |

**直觉**:SNR = 40 dB 意味着误差能量只有信号的万分之一;每差 20 dB,误差差 100 倍。
**经典结论**:均匀量化下每少 1 bit,SNR 约降 $6.02b$ dB 中的 6 dB(误差能量 ×4)。

**钳位与校准**:把数据先钳位到 $[-c, c]$ 再量化,总误差 = **钳位误差 + 舍入误差**,
两者此消彼长,存在 U 型最优点;校准就是找这个最优 $c$。
'''))

NB.code(ERROR_METRICS, "📐 误差指标工具箱:`mse` / `max_abs_err` / `cosine_sim`,本课反复使用。")
NB.code(INT8_SYMM, "📐 还要用到对称量化工具(`symm_quantize` / `symm_dequant`),复用第 28 课版本。")

NB.md("## 3️⃣ 最小实现 · 逐行推演 🔬",
D('''
用一个小张量实现 SNR,并把每一步的 shape 打印出来——
**每行代码都有注释**,读者可心算验证。
'''))

NB.code(D('''
# --------------------------------------------------------------------------
# 实现 SNR 并验证「每少 1 bit 约降 6 dB」的经典结论
# --------------------------------------------------------------------------
def snr_db(x, x_hat):
    # 信噪比(分贝):信号功率 / 量化噪声功率(用均值口径,与第 5 节 metrics 一致)
    sig = float((x.double() ** 2).mean())              # 信号功率:值的平方平均(标量)
    noise = float(((x - x_hat).double() ** 2).mean())  # 噪声功率:误差平方平均(标量)
    return 10.0 * torch.log10(torch.tensor(sig / max(noise, 1e-30)))  # 10*log10(功率比)

torch.manual_seed(0)
x = torch.randn(10000)                       # (10000,) 一万个标准正态值,作为「信号」
print(f"信号张量 x shape = {tuple(x.shape)}  <- (n=10000,)")

for bits in [8, 6, 4, 3, 2]:
    q, s = symm_quantize(x, bits=bits)       # 量化到 bits 位
    xh = symm_dequant(q, s)                  # 反量化
    print(f"{bits} bit: MSE={mse(x, xh):.3e}  SNR={float(snr_db(x, xh)):5.1f} dB")

print("\\n观察:8→6→4→2 bit,每少 1 bit SNR 约降 6 dB(误差能量 ×4)——均匀量化的经典结论。")
'''), "✅ 规律:每少 1 bit,SNR 约降 6 dB(误差×4)——这是均匀量化的经典结论 6.02b dB 的体现。")

NB.md("## 4️⃣ 数值验证:正态 vs 重尾,量化难度对决 🥊",
D('''
同样 4 bit,四种分布同台竞技:**正态**(尾巴指数衰减)、**拉普拉斯**(尖峰厚尾)、
**t 分布 df=3**(幂律重尾)、**t 分布 df=1**(柯西级重尾)。尾巴越厚,极端值越多,
max 校准的 scale 越被撑大:
'''))

NB.code(D('''
torch.manual_seed(42)
n = 50000                                       # 每分布 5 万个样本
dists = {
    "正态":       torch.randn(n),                                    # 指数尾巴
    "拉普拉斯":   torch.distributions.Laplace(0, 1).sample((n,)),   # 尖峰厚尾
    "t(df=3)":    torch.distributions.StudentT(3).sample((n,)),     # 幂律重尾
    "t(df=1)柯西": torch.distributions.StudentT(1).sample((n,)),    # 极重尾
}
rows = []
for name, x in dists.items():
    q, s = symm_quantize(x, bits=4)             # 对称量化
    xh = symm_dequant(q, s)                     # 反量化
    rows.append(dict(分布=name, 最大绝对值=float(x.abs().max()),
                     scale=float(s), MSE=mse(x, xh), SNR_dB=float(snr_db(x, xh))))
print(f"{'分布':<12}{'最大绝对值':>10}{'scale':>10}{'MSE':>12}{'SNR(dB)':>9}")
for r in rows:
    print(f"{r['分布']:<12}{r['最大绝对值']:10.2f}{r['scale']:10.4f}{r['MSE']:12.3e}{r['SNR_dB']:9.1f}")
print("\\n读表:尾巴越重 → 最大绝对值越大 → scale 越粗 → SNR 越差。柯西分布的极值能把 scale 撑到正态的十倍以上。")
'''), "📊 尾巴越重,scale 越粗、SNR 越差——这就是「量化难度由分布形状决定」的证据。")

NB.code(D('''
# --------------------------------------------------------------------------
# 控制变量实验:0.1% / 1% / 10% 的离群值把 scale 撑大多少?
# --------------------------------------------------------------------------
torch.manual_seed(0)
n = 20000                                        # 2 万个样本
base = torch.randn(n)                            # (20000,) 标准正态基底
print(f"{'离群比例':>6s} {'离群强度':>8s} {'max|x|':>8s} {'scale':>10s} {'MSE':>10s} {'SNR(dB)':>8s}")
for frac, mag in [(0.0, 0), (0.001, 10), (0.01, 10), (0.01, 30), (0.1, 10)]:
    x = base.clone()                             # 复制基底,避免污染
    if frac > 0:
        m = int(n * frac)                        # 离群值个数
        idx = torch.randperm(n)[:m]              # 随机挑 m 个位置
        x[idx] = x[idx] * mag                    # 放大 mag 倍,造离群值
    q, s = symm_quantize(x, bits=4)              # 量化
    xh = symm_dequant(q, s)                      # 反量化
    print(f"{frac*100:5.1f}% {mag:8d} {float(x.abs().max()):8.2f} {float(s):10.4f} {mse(x, xh):10.3e} {float(snr_db(x, xh)):8.1f}")
print("\\n⚠️ 只要有 0.1% 的值放大 10 倍,scale 就被撑大约 10 倍,SNR 掉 10+ dB——99.9% 的正常值为 0.1% 的离群值陪葬。")
'''), "⚠️ 0.1% 离群值撑大 scale 一个数量级——离群值是量化精度的第一杀手。")

NB.md("## 5️⃣ 真实规模数字:GPU 上的离群值攻防 💾",
D('''
把离群值的破坏机制搬到一张**百万级元素的真实 GPU 权重张量**上:样本来自重尾 t 分布
(LLM 权重分布的常见画像),再人工注入 0.01% 的强离群值。对比两种校准:
**max 校准**(给离群值陪葬)与 **99.9% 分位钳位**(牺牲离群值、救活全体)。
同时量 MSE、SNR(dB) 与**余弦相似度**——Cosine 看的是方向漂移:
'''))

NB.code(D('''
import torch, gc
dev = "cuda" if torch.cuda.is_available() else "cpu"   # 优先 GPU
torch.manual_seed(0)
n = 2_000_000
w = torch.distributions.StudentT(3).sample((n,)).to(dev)      # 重尾权重 (2_000_000,)
w[torch.randint(0, n, (200,))] *= 20.0                        # 0.01% 强离群值

def metrics(x, xh):
    # 一揽子误差指标:返回 (MSE, SNR_dB, 余弦相似度)
    x, xh = x.float(), xh.float()
    mse = float(((x - xh) ** 2).mean())
    snr = float(10 * torch.log10(torch.tensor((x ** 2).mean() / max(mse, 1e-30))))
    cos = float(torch.dot(x.flatten(), xh.flatten()) / (x.norm() * xh.norm() + 1e-12))
    return mse, snr, cos

qmax = 127
# max 校准:离群值撑大 scale,全体格子变粗
s_max = (w.abs().max() / qmax).item()
Wh_max = torch.clamp(torch.round(w / max(s_max, 1e-12)), -qmax, qmax).float() * s_max
# 99.9% 分位钳位:把最极端 0.1% 尾巴钳掉,换全体更细的格子
c = float(torch.quantile(w.abs(), 0.999))        # 99.9% 分位阈值
s_pct = c / qmax
Wh_pct = torch.clamp(torch.round(w / max(s_pct, 1e-12)), -qmax, qmax).float() * s_pct

m_max, snr_max, cos_max = metrics(w, Wh_max)
m_pct, snr_pct, cos_pct = metrics(w, Wh_pct)

print(f"设备: {dev} | 权重 shape = {tuple(w.shape)}  <- (n={n:,} 个元素)")
print(f"max 校准    : scale={s_max:.4f} · MSE={m_max:.3e} · SNR={snr_max:6.1f} dB · 余弦={cos_max:.6f}")
print(f"99.9% 钳位  : scale={s_pct:.4f} · MSE={m_pct:.3e} · SNR={snr_pct:6.1f} dB · 余弦={cos_pct:.6f}")
print(f"→ 离群值把 scale 撑大 {s_max/max(s_pct,1e-12):.0f} 倍,SNR 损失 {snr_max-snr_pct:.1f} dB,余弦滑落 {cos_max-cos_pct:.5f}")

torch.cuda.synchronize()
gc.collect(); torch.cuda.empty_cache()
'''), "✅ 真实 GPU 数字:0.01% 的离群值就把 max 校准的 scale 撑大一个数量级——百分位钳位牺牲几百个离群值、救活两百万元素。这就是 GPTQ/AWQ 要用校准集挑「牺牲哪些值」的原因。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:校准集为什么重要 🚀",
D('''
校准(calibration)不是理论玩具,而是 GPTQ/AWQ 等后训练量化方法**绕不开的一步**:

1. **校准集 = 几行真实输入**:拿一小批文本(如 C4 语料的几百条),前向跑一遍模型,
   记录每层的激活分布;量化参数(scale/zero-point/clip 阈值)就由这批统计决定;
2. **max vs percentile 的选择**:对重尾分布,工程上常选 99.9%/99.99% 分位而非 max,
   这正是本课第 5 节的结论;vLLM 的 FP8 KV cache 也支持 `--calculate-kv-scales`
   用校准数据统计 scale,而不是无脑 max;
3. **校准集要贴近真实分布**:校准集若与推理数据分布不一致,分位数是「错的经验统计量」,
   量化的容忍度会被错误估计——所以社区发布的 GPTQ/AWQ 模型都用通用语料校准。

```bash
# vLLM 为 FP8 KV cache 校准 scale
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype fp8_e4m3 --calculate-kv-scales
```

> 📄 离群值破坏量化的系统研究见 [LLM.int8()](https://arxiv.org/abs/2208.07339):
> 少数几个激活通道的离群值让 8bit 矩阵乘全面失效,需要把这些通道单独保留高精度。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:调分布、选校准、看误差 🎛️",
D('''
运行同目录下的 `app_30_quant_error.py`,自由选择分布类型、离群值强度、位宽与校准分位,
误差指标、分布直方图与 MSE-分位扫描曲线全部实时刷新:

```bash
D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_30_quant_error.py
```

浏览器打开 **http://localhost:8501**。建议玩法:选「t 分布 df=3」,把校准分位从 100 拖到 99.5,
看「MSE-分位 U 型曲线」上的最优点如何移动。完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_30, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_30_quant_error.py 后运行:\")\n"
    "    print(\"    D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_30_quant_error.py\")\n"
)
NB.code(guard, "📜 这就是 app_30_quant_error.py 的完整源码,notebook 与 app 共享同一套校准逻辑。")

wrapup(NB,
    summary=[
        "误差两把尺:MSE 衡量平均火力;SNR(dB)把误差放到信号尺度,每少 1 bit 约降 6 dB",
        "分布形状决定量化难度:正态 < 拉普拉斯 < t 分布,尾巴越重越难量化",
        "离群值的破坏机制:0.1% 的极端值能把 scale 撑大一个数量级,全体格子陪葬",
        "钳位 + percentile 校准:总误差 = 钳位误差 + 舍入误差,存在 U 型最优分位",
        "校准需要真实数据:这正是 GPTQ/AWQ 要喂几百条校准样本的原因",
    ],
    practice=[
        "把第 4 节的 bits 改成 8,重画 U 型曲线,观察最优分位是否更靠近 100%",
        "给 symm_quantize_clip 增加 MSE 最优搜索(网格扫描),写成一个 auto_calibrate 函数",
        "用 torch.quantile 分别取 99.99/99.9/99.5 分位,统计各自被钳掉的样本比例",
        "思考:为什么校准集要与真实推理数据同分布?(提示:分位数是经验统计量)",
    ],
    links=[
        ("LLM.int8():离群值与 8bit 矩阵乘", "https://arxiv.org/abs/2208.07339"),
        ("Jacob et al., Integer-Arithmetic-Only Inference(clipping 讨论)", "https://arxiv.org/abs/1712.05877"),
        ("A White Paper on Neural Network Quantization", "https://arxiv.org/abs/2106.08295"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH05) / "30_quantization_error.ipynb"))

app_path = Path(CH05) / "app_30_quant_error.py"
app_path.write_text(APP_30 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")