# -*- coding: utf-8 -*-
"""生成 29_sym_asym_quant.ipynb 与 app_29_sym_asym.py(教材级重写版)

论文/资料支撑:
- Nagel et al., "A White Paper on Neural Network Quantization", arXiv:2106.08295
- HF Transformers quantization concept guide
- Intel Neural Compressor 量化文档(per-tensor vs per-channel 的严谨公式)
- The Neural Base: quantization granularity per-tensor vs per-channel
"""
from helpers import (D, INT8_SYMM, ASYMM_QUANT, PER_CHANNEL, ERROR_METRICS,
                     chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_29 = D('''
# -*- coding: utf-8 -*-
# app_29_sym_asym.py — 对称 vs 非对称量化:分布直方图实时对比 ⚖️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="⚖️ 29 · 对称 vs 非对称量化", layout="wide")
st.title("⚖️ 第 29 课 · 对称 vs 非对称量化:谁的格子用得更值?")

st.markdown("""
同一批数据、同样的位宽,**对称量化**把 0 钉在正中间(正负各半),
**非对称量化**给零点加一个 zero-point 偏移,让有限格子盖住数据的真实范围。
选择哪个,取决于分布长什么样。下面拖动参数,实时看量化前后的直方图对比:
- 📊 均值远离 0(比如 ReLU 激活全为正)→ 非对称明显占优;
- ⚖️ 分布关于 0 对称 → 两者几乎打平,对称更简单(不用存 zero-point)。
""")

def quantize(x, bits, mode):
    x = np.asarray(x, dtype=np.float64)
    if mode == "对称(symmetric)":
        qmax = 2 ** (bits - 1) - 1
        scale = max(np.max(np.abs(x)) / qmax, 1e-12)
        q = np.round(x / scale).clip(-qmax, qmax)
        x_hat = q * scale
        zp = 0
    else:
        qmax = 2 ** bits - 1
        rmin, rmax = x.min(), x.max()
        scale = max((rmax - rmin) / qmax, 1e-12)
        zp = int(np.clip(np.round(-rmin / scale), 0, qmax))
        q = np.round(x / scale + zp).clip(0, qmax)
        x_hat = (q - zp) * scale
    return x_hat, scale, zp

with st.sidebar:
    st.header("🎛️ 分布与量化参数")
    dist = st.selectbox("分布类型", ["正态(对称)", "全正 ReLU 型", "偏斜 LogNormal", "均匀(对称)"])
    mu = st.slider("分布均值 μ(位置)", -4.0, 4.0, 0.0, 0.1)
    sigma = st.slider("分布标准差 σ(散布)", 0.2, 3.0, 1.0, 0.1)
    n = st.slider("采样点数", 512, 20000, 8000, 512)
    bits = st.slider("量化位宽 b(bit)", 2, 10, 4, 1)
    st.caption("💡 提示:把 μ 拉到 +3(全正分布),再切换两种模式,看 MSE 差多少倍。")

rng = np.random.default_rng(42)
if dist == "正态(对称)":
    x = rng.normal(mu, sigma, n)                         # 零对称正态
elif dist == "全正 ReLU 型":
    x = np.maximum(rng.normal(mu, sigma, n), 0.0)       # ReLU 截断
elif dist == "偏斜 LogNormal":
    x = rng.lognormal(max(mu, 0.01), sigma, n)          # 右偏对数正态
else:
    x = rng.uniform(mu - sigma, mu + sigma, n)          # 均匀

results = {}
for mode in ["对称(symmetric)", "非对称(asymmetric)"]:
    xh, s, zp = quantize(x, bits, mode)
    results[mode] = dict(xh=xh, s=s, zp=zp, mse=float(np.mean((xh - x) ** 2)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("对称 MSE", f"{results['对称(symmetric)']['mse']:.3e}")
c2.metric("非对称 MSE", f"{results['非对称(asymmetric)']['mse']:.3e}")
ratio = results["对称(symmetric)"]["mse"] / max(results["非对称(asymmetric)"]["mse"], 1e-30)
c3.metric("对称/非对称 误差比", f"{ratio:.2f}x")
c4.metric("非对称 zero-point", f"{results['非对称(asymmetric)']['zp']}")

fig = go.Figure()
fig.add_trace(go.Histogram(x=x, name="原始分布", opacity=0.55,
                           marker_color="#4C78A8", nbinsx=80))
for mode, color in [("对称(symmetric)", "#E45756"), ("非对称(asymmetric)", "#F2C14E")]:
    fig.add_trace(go.Histogram(x=results[mode]["xh"], name="量化后·" + mode,
                               opacity=0.55, marker_color=color, nbinsx=80))
fig.update_layout(barmode="overlay", title=f"{dist} · μ={mu:.1f}, σ={sigma:.1f} · {bits} bit 量化前后直方图",
                  xaxis_title="数值", yaxis_title="频数(计数)", height=460,
                  legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 位宽扫描:两种量化方式的 MSE 对比")
sweep = []
for b in range(2, 11):
    for mode in ["对称(symmetric)", "非对称(asymmetric)"]:
        xh, _, _ = quantize(x, b, mode)
        sweep.append((b, mode, float(np.mean((xh - x) ** 2))))
fig2 = go.Figure()
for mode, color in [("对称(symmetric)", "#E45756"), ("非对称(asymmetric)", "#F2C14E")]:
    ys = [v for (b, m, v) in sweep if m == mode]
    bs = [b for (b, m, v) in sweep if m == mode]
    fig2.add_trace(go.Scatter(x=bs, y=ys, mode="lines+markers", name=mode,
                              line=dict(color=color, width=2)))
fig2.update_layout(title="位宽 b vs MSE(对数轴):分布越偏,非对称优势越大",
                   xaxis_title="位宽 b(bit)", yaxis_title="MSE", yaxis_type="log",
                   height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **经验法则**:权重(大致零对称)用对称量化;激活(ReLU 后全为正)用非对称量化。
> 这也是 PyTorch / vLLM 里最常见的 QConfig 组合:权重 per-channel 对称 + 激活 per-tensor 非对称。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 5 章 · 第 29 课配套演示")

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

NB = new_nb("第 29 课 · 对称 vs 非对称量化:per-tensor 与 per-channel",
            subtitle="同一把 int8 尺子,零点放哪、尺子配几把,误差天差地别——量化粒度的选择指南",
            emoji="⚖️")

chapter_cover(NB,
    objectives=[
        "彻底搞懂对称与非对称量化的 scale / zero-point 公式与适用场景",
        "用数值实验验证:对称适合零对称分布,非对称适合有偏分布",
        "理解量化粒度:per-tensor(一把尺子)vs per-channel(每通道一把)",
        "在真实 GPU 权重上量出 per-tensor vs per-channel 的 MSE/SNR 差距",
    ],
    toc=[
        ("直觉:温度计与体重秤", "两种零点设计的比喻,引出对称/非对称"),
        ("核心定义与公式", "两套映射公式 + 符号表 + 粒度定义"),
        ("最小实现 · 逐行推演", "手写对称/非对称/per-channel 量化,小张量打印 shape"),
        ("数值验证:误差上界与三种分布", "验证 s/2 上界;正态/全正/偏斜三种分布的对决"),
        ("真实规模数字", "行幅度差 100 倍的权重:per-channel 救场,GPU 真机 MSE/SNR"),
        ("与 vLLM 工程实现的关系", "QConfig 默认组合:权重 per-channel 对称 + 激活 per-tensor 非对称"),
        ("配套 Streamlit 演示", "app_29_sym_asym.py:拖滑块实时对比"),
    ],
    links=[
        ("A White Paper on Neural Network Quantization (arXiv:2106.08295)", "https://arxiv.org/abs/2106.08295"),
        ("PyTorch QConfig 文档", "https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig.QConfig.html"),
        ("Intel Neural Compressor 量化文档", "https://intel.github.io/neural-compressor/latest/docs/source/quantization.html"),
        ("The Neural Base: per-tensor vs per-channel", "https://theneuralbase.com/quantization-fundamentals/learn/beginner/quantization-granularity-per-tensor-vs-per-channel/"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 torch/numpy OMP 冲突保护
import torch
import numpy as np

torch.manual_seed(0)          # 固定随机种子,保证结果可复现
np.random.seed(0)
print("torch =", torch.__version__)
# 说明:本课执行体不引入 pandas —— GPU 量化 cell 后再建 pandas/pyarrow 表可能触发 arrow.dll 原生崩溃,
# 故表格一律用纯 Python 的 dict + print 展示(配套 streamlit app 内仍可用 pandas,不参与 notebook 执行)。
'''), "🛡️ 环境准备:KMP 保护 + 固定随机种子,保证每次运行结果一致。")

NB.md("## 1️⃣ 直觉:温度计与体重秤 🌡️",
D('''
上一课我们知道了量化就是「换一把更粗的尺子」。这一课要回答两个更细的问题:

**问题一:零点放在哪?**
- 🌡️ **温度计**是**对称**设计:0℃ 在中间,零上零下各占一半刻度,像 int8 的 $[-127, +127]$;
- ⚖️ **体重秤**是**非对称**设计:0 kg 在最左端,量程全是正数,像 uint8 的 $[0, 255]$。

如果你要量的东西全为正(比如 ReLU 之后的激活,取值 $[0, 6]$),却用温度计式的对称尺子,
那么 $[-6, 0)$ 这半边刻度**全部浪费**——等于白白丢掉 1 bit 精度!

**问题二:尺子配几把?**
- 🎒 **per-tensor**:整个张量共用一把尺子(一个 scale),省显存但粗糙;
- 🧰 **per-channel**:每个输出通道各配一把尺子(一组 scale),多花一点存储,误差显著更小。

> 📄 为什么 per-channel 对**权重**可行、对**激活**不可行?因为权重尺度可以离线折叠进矩阵乘,
> 而激活要按输入通道反缩放,硬件累加器难处理——详见
> [Intel Neural Compressor 量化文档](https://intel.github.io/neural-compressor/latest/docs/source/quantization.html)。
'''))

NB.md("## 2️⃣ 核心定义与公式 🧮",
D('''
**对称量化**(有符号整数,$q \\in [-Q_{max}, Q_{max}]$,int8 时 $Q_{max}=127$):

$$s = \\frac{\\max|x|}{2^{b-1}-1}, \\qquad q = \\mathrm{round}\\left(\\frac{x}{s}\\right), \\qquad \\hat{x} = q \\cdot s$$

零点固定为 0:浮点 0 → 整数 0,天然对齐。**非对称量化**(无符号整数,$q \\in [0, 2^b-1]$,uint8 时 255):

$$s = \\frac{r_{max} - r_{min}}{2^b - 1}, \\qquad z = \\mathrm{round}\\left(\\frac{-r_{min}}{s}\\right), \\qquad q = \\mathrm{round}\\left(\\frac{x}{s} + z\\right), \\qquad \\hat{x} = (q - z) \\cdot s$$

| 符号 | 含义 | 对称 | 非对称 |
|---|---|---|---|
| $b$ | 位宽 | 8 | 8 |
| $Q_{max}$ | 整数上界 | $2^{b-1}-1=127$ | $2^b-1=255$ |
| $s$ | scale | $\\max\\|x\\|/Q_{max}$ | $(r_{max}-r_{min})/(2^b-1)$ |
| $z$ | zero-point | 固定 0 | $\\mathrm{round}(-r_{min}/s)$,存为一个整数 |
| $\\hat{x}$ | 重建值 | $q\\cdot s$ | $(q-z)\\cdot s$ |

**量化粒度**(granularity):

- **per-tensor**:整张量 1 个 $s$、1 个 $z$;
- **per-channel**:对权重矩阵 $(d_{out}, d_{in})$ 沿**输出通道**(dim=0 的每一行)各配一个 $s$。
  权重矩阵往往「各行幅度差异巨大」,per-tensor 会让小幅度行被大幅度的行「陪葬」。

> 🔍 注意:非对称的反量化矩阵乘法里要多做一次减法 $q - z$,硬件上通常把它折叠进 bias,几乎零开销。
'''))

NB.code(INT8_SYMM, "📐 复用第 28 课的对称量化函数(`symm_quantize` / `symm_dequant`)。")
NB.code(ASYMM_QUANT, "📐 非对称量化函数(`asymm_quantize` / `asymm_dequant`),zero-point 是关键角色。")

NB.md("## 3️⃣ 最小实现 · 逐行推演 🔬",
D('''
先用一个小张量走一遍两种量化,把每个中间量的 shape 与含义打出来——
**每行代码都有注释**,读者可以心算核对。
'''))

NB.code(D('''
# --------------------------------------------------------------------------
# 构造一个「全正分布」的小张量:模拟 ReLU 之后的激活
# --------------------------------------------------------------------------
x_pos = torch.tensor([0.0, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0])  # (8,) 8 个正数,范围 [0,5]
print(f"输入张量 x_pos shape = {tuple(x_pos.shape)}  <- 8 个全正激活值")

# --- 对称量化(浪费负半边格子) ---
qs, ss = symm_quantize(x_pos, bits=8)          # 量化:q 在 [-127,127],但数据没有负数
err_s = float(((x_pos - symm_dequant(qs, ss)) ** 2).mean())   # 对称 MSE
print(f"\\n[对称]   scale = {float(ss):.5f}  <- 由 max|x|=5.0 决定,负半边格子全空")

# --- 非对称量化(用满全部格子) ---
qa, sa, zp = asymm_quantize(x_pos, bits=8)     # 量化:q 在 [0,255],zero-point 平移
err_a = float(((x_pos - asymm_dequant(qa, sa, zp)) ** 2).mean())  # 非对称 MSE
print(f"[非对称] scale = {float(sa):.5f}  zero_point = {int(zp)}  <- 用满 [0,255]")

print(f"\\n对称 scale / 非对称 scale = {float(ss)/float(sa):.2f}  (格子粗了多少倍)")
print(f"对称 MSE   = {err_s:.2e}")
print(f"非对称 MSE = {err_a:.2e}")
print(f"非对称比对称误差低 {err_s/err_a:.1f} 倍")
'''), "✅ 对全正分布,非对称量化的 scale 更细(格子更窄),误差明显更小——格子用在了刀刃上。")

NB.md("## 4️⃣ 数值验证:误差上界 + 三种分布的对决 🥊",
D('''
量化再反量化,$\\hat{x}$ 与 $x$ 的差就是量化误差,上界是半个格子 $s/2$。
然后来一场公平比武:三种典型分布(零对称正态、全正 ReLU 型、右偏 LogNormal),
两种量化方式,同样 4 bit(误差差异在低位宽下更刺眼),各算 MSE:
'''))

NB.code(D('''
torch.manual_seed(42)
n = 4096                                          # 每分布采样 4096 个点
dists = {
    "正态 N(0,1)":        torch.randn(n),                                   # 零对称
    "全正 ReLU":           torch.relu(torch.randn(n) * 0.5 + 1.0),          # 全正
    "右偏 LogNormal":      torch.distributions.LogNormal(0.0, 1.0).sample((n,)),  # 右偏
}
rows = []
for name, x in dists.items():
    qs, ss = symm_quantize(x, bits=4)             # 对称量化
    e_sym = float(((x - symm_dequant(qs, ss)) ** 2).mean())
    qa, sa, za = asymm_quantize(x, bits=4)        # 非对称量化
    e_asym = float(((x - asymm_dequant(qa, sa, za)) ** 2).mean())
    rows.append(dict(分布=name, 对称MSE=e_sym, 非对称MSE=e_asym,
                     对称scale=float(ss), 非对称scale=float(sa), 倍数=e_sym / e_asym))
print(f"{'分布':<14}{'对称MSE':>12}{'非对称MSE':>12}{'倍数':>8}")
for r in rows:
    print(f"{r['分布']:<14}{r['对称MSE']:12.4e}{r['非对称MSE']:12.4e}{r['倍数']:8.1f}")
print("\\n读表:分布越「偏」,倍数越大;零对称分布两者几乎打平(此时选更简单的对称)。")
'''), "📊 分布越偏,非对称的优势越大;零对称分布两者打平——选型取决于数据分布形状。")

NB.md("## 5️⃣ 真实规模数字:per-channel 救场的硬证据 💾",
D('''
per-tensor 只存 1 个 scale;per-channel 沿输出通道(dim=0)每行存 1 个 scale。
如果权重矩阵各行幅度差异很大(神经网络里很常见!),共用一把尺子会让小幅度行「陪葬」。

先用一个**行幅度差 100 倍**的手工矩阵做实验(快),再把它搬到 **GPU 上的重尾 t 分布权重**(慢但真实):
'''))

NB.code(PER_CHANNEL, "📐 per-channel 对称量化:每行(输出通道)各配一把尺子。")

NB.code(D('''
# --------------------------------------------------------------------------
# 实验 1:行幅度差 100 倍的矩阵(CPU,快速)
# --------------------------------------------------------------------------
torch.manual_seed(0)
d_out, d_in = 64, 128
base = torch.randn(d_out, d_in)                       # (64, 128) 基础权重
row_scale = torch.logspace(0, 2, d_out).unsqueeze(1)  # (64, 1) 行幅度从 1 到 100
W = base * row_scale                                  # (64, 128) 每行幅度差很大

q_t, s_t = symm_quantize(W, bits=4)                   # per-tensor:全矩阵一个 scale
e_t = float(((W - symm_dequant(q_t, s_t)) ** 2).mean())   # per-tensor MSE

q_c, s_c = symm_quantize_channel(W, bits=4)           # per-channel:每行一个 scale
e_c = float(((W - symm_dequant_channel(q_c, s_c)) ** 2).mean())  # per-channel MSE

print(f"per-tensor  : 1  个 scale={float(s_t):.4f}  MSE={e_t:.4e}")
print(f"per-channel : {d_out} 个 scale(均值 {float(s_c.mean()):.4f})  MSE={e_c:.4e}")
print(f"per-channel 误差降低 {e_t / e_c:.0f} 倍,额外代价只是每行存一个 fp16 scale({d_out} 个数)")
'''), "✅ 行幅度差 100 倍时,per-channel 每个输出行一把尺子,误差明显下降(4 bit 低位宽下约 4 倍)——低位宽格粗,优势还没到数量级,但已经让 MSE 显著变小;这就是权重量化默认 per-channel 的原因。")

NB.code(D('''
import torch, gc
dev = "cuda" if torch.cuda.is_available() else "cpu"   # 优先 GPU

torch.manual_seed(0)
d_out, d_in = 256, 512
W = torch.distributions.StudentT(3).sample((d_out, d_in)).to(dev) * 0.1   # 重尾 t 分布权重
row_scale = torch.logspace(0, 2, d_out, device=dev).unsqueeze(1)         # 各输出通道幅度不同
W = W * row_scale                                        # (256, 512) 更接近真机的权重画像

def snr_db(x, xh):
    # 信噪比(分贝):信号功率 / 噪声功率
    sig = float((x.double() ** 2).sum())
    noise = float(((x - xh).double() ** 2).sum())
    return float(10 * torch.log10(torch.tensor(sig / max(noise, 1e-30))))

qmax = 127
# per-tensor:整张矩阵一把尺子
s_t = (W.float().abs().max() / qmax).item()
Wh_t = torch.clamp(torch.round(W.float() / s_t), -qmax, qmax).float() * s_t
# per-channel:每个输出通道(dim=0 行)一把尺子
s_c = W.float().abs().amax(dim=1, keepdim=True) / qmax
Wh_c = torch.clamp(torch.round(W.float() / s_c), -qmax, qmax).float() * s_c

e_t, snr_t = float(((W.float() - Wh_t) ** 2).mean()), snr_db(W.float(), Wh_t)
e_c, snr_c = float(((W.float() - Wh_c) ** 2).mean()), snr_db(W.float(), Wh_c)

print(f"设备: {dev} | 权重 shape = {tuple(W.shape)}  <- (输出通道 {d_out}, 输入通道 {d_in})")
print(f"per-tensor : 1 把尺子  · MSE={e_t:.5e}  SNR={snr_t:6.1f} dB")
print(f"per-channel: {d_out} 把尺子 · MSE={e_c:.5e}  SNR={snr_c:6.1f} dB")
print(f"→ MSE 下降 {e_t/max(e_c,1e-30):.0f} 倍,SNR 提升 {snr_c-snr_t:+.1f} dB")
print(f"→ 额外存储极小:每通道 1 个 fp16 scale = {d_out*2/1024:.1f} KB,误差却降约一个数量级(本例 {e_t/max(e_c,1e-30):.0f} 倍)")

gc.collect(); torch.cuda.empty_cache()
'''), "✅ 真实 GPU 数字:重尾 + 逐通道幅度不均时,per-channel 用几百个尺子换来约一个数量级(本例约十几倍)的误差下降——这就是工程上权重量化默认 per-channel 的硬证据。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:QConfig 怎么配 🚀",
D('''
工程里的量化配置(QConfig)就是「给每个算子选:权重怎么量化、激活怎么量化」。

- **权重**:一律 **per-channel 对称**(int8 时代)。因为权重离线可知、各行幅度差异大,
  per-channel 误差显著更低(本课第 5 节已实测),且可折叠进 GEMM 累加;
- **激活**:一律 **per-tensor 非对称**(或动态 per-token)。因为激活运行期才知道范围,
  无法逐通道预先配尺子;非对称则让 ReLU 后的全正分布用满格子。

这一组合正是 [A White Paper (arXiv:2106.08295)](https://arxiv.org/abs/2106.08295) 与
[PyTorch 量化实践](https://pytorch.org/blog/quantization-in-practice/) 推荐的标准做法:

```
权重:  per-channel symmetric   (per output channel 一把 scale)
激活:  per-tensor asymmetric   (整张量一把 scale + zero-point)
```

> ⚠️ 但在 **int4/int8 权重量化**时代,vLLM 用的是 GPTQ/AWQ 的 **group-wise** 格式
> (每 128 个连续输入通道一组 scale)——比 per-channel 更细、误差更低,第 31-32 课展开。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖滑块,实时对决 🎛️",
D('''
运行同目录下的 `app_29_sym_asym.py`,可以自由切换分布类型、调节均值/方差/位宽,
直方图和 MSE 对比实时刷新。运行方法:

```bash
D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_29_sym_asym.py
```

浏览器打开 **http://localhost:8501**。建议玩法:选「全正 ReLU 型」分布,
把均值拉到 +3,然后在对称/非对称之间切换,观察右下角「对称/非对称误差比」能到多少倍。
完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_29, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_29_sym_asym.py 后运行:\")\n"
    "    print(\"    D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_29_sym_asym.py\")\n"
)
NB.code(guard, "📜 这就是 app_29_sym_asym.py 的完整源码,notebook 与 app 共享同一套量化逻辑。")

wrapup(NB,
    summary=[
        "对称量化:零点固定 0,公式简单,适合大致零对称的权重;负半边刻度对全正数据是浪费",
        "非对称量化:zero-point 让 [rmin, rmax] 用满全部 2^b 个格子,适合 ReLU 激活等有偏分布",
        "量化误差上界是半个格子 s/2;分布越偏,对称 vs 非对称的差距越大",
        "per-channel 给每个输出通道配一把尺子,行幅度差异大时误差可降数倍到约一个数量级",
        "工程默认组合:权重 per-channel 对称 + 激活 per-tensor 非对称",
        "int4 时代更细的粒度是 group-wise(每 128 列一组 scale),见第 31-32 课",
    ],
    practice=[
        "把第 4 节的 bits 改成 8,观察对称/非对称的差距如何缩小,并解释原因",
        "写一个 per-channel 非对称量化函数(把 asymm_quantize 推广到逐行),测试右偏矩阵",
        "构造一个「双峰」分布(两个高斯混合),判断哪种量化方式更适合并实验验证",
        "统计量化后不同取值的个数,验证它不超过 2^b",
    ],
    links=[
        ("A White Paper on Neural Network Quantization", "https://arxiv.org/abs/2106.08295"),
        ("PyTorch 量化配置 QConfig", "https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig.QConfig.html"),
        ("PyTorch 量化实践博客", "https://pytorch.org/blog/quantization-in-practice/"),
        ("Intel Neural Compressor 量化文档", "https://intel.github.io/neural-compressor/latest/docs/source/quantization.html"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH05) / "29_sym_asym_quant.ipynb"))

app_path = Path(CH05) / "app_29_sym_asym.py"
app_path.write_text(APP_29 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")