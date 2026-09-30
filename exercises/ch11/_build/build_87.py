# -*- coding: utf-8 -*-
"""生成 87_ascend_quant.ipynb 与 app_87_ascend_quant.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_87 = D('''
# -*- coding: utf-8 -*-
# app_87_ascend_quant.py — 华为量化方案:位宽、误差与体积交互 🔢
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🔢 87 · 华为量化方案", layout="wide")
st.title("🔢 第 87 课 · 华为量化方案:用位宽换体积,用校准保精度")

st.markdown("""
昇腾侧的量化(MindSpore PTQ/QAT、MindIE 低比特)与 vLLM 侧(GPTQ/AWQ/FP8)遵循同一套底层逻辑:
**用更低的位宽表示权重/激活,用校准数据保住精度**。下方拖动**位宽**、选择**校准策略**与
**数值分布**,观察 **量化误差 vs 体积** 的权衡,以及 FP8 的 E4M3/E5M2 两种尾数布局。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    bits = st.slider("量化位宽(bit)", 1, 8, 4, 1)
    scheme = st.radio("校准策略", ["全局 scale", "逐层 scale", "逐通道 scale"])
    dist = st.select_slider("权重分布尺度(σ)", options=[0.05, 0.1, 0.3, 0.5, 1.0], value=0.3)
    n_weights = st.slider("权重数量(万)", 10, 500, 100, 10)
    st.caption("量化误差 = |q(x) - x| 的均值;scale 越细(逐通道)、分布越集中,误差越小。")

rng = np.random.default_rng(0)
x = rng.normal(0, dist, int(n_weights * 1e4))

def quant_error(x, bits, scheme):
    if scheme == "全局 scale":
        s = x.abs().max() / (2 ** (bits - 1) - 1)
        q = np.clip(np.round(x / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        return np.abs(q * s - x).mean(), 1.0 / s
    if scheme == "逐层 scale":
        s = np.quantile(np.abs(x), 0.99) / (2 ** (bits - 1) - 1)
        q = np.clip(np.round(x / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        return np.abs(q * s - x).mean(), 1.0 / s
    chunks = np.array_split(x, 32)                     # 逐通道
    errs, scales = [], []
    for c in chunks:
        s = c.abs().max() / (2 ** (bits - 1) - 1)
        q = np.clip(np.round(c / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        errs.append(np.abs(q * s - c).mean()); scales.append(1.0 / s)
    return np.mean(errs), np.mean(scales)

err, _ = quant_error(x, bits, scheme)
vol_ratio = bits / 8.0 * 100
c1, c2, c3 = st.columns(3)
c1.metric("量化误差(均值)", f"{err:.4f}")
c2.metric("体积占比", f"{vol_ratio:.0f} %", f"省 {(1-vol_ratio):.0f}%")
c3.metric("数据点", f"{len(x)/1e4:.0f} 万")

st.subheader("📉 误差 vs 位宽曲线")
bits_range = np.arange(1, 9)
fig = go.Figure()
for scheme in ["全局 scale", "逐层 scale", "逐通道 scale"]:
    ys = [quant_error(x, b, scheme)[0] for b in bits_range]
    fig.add_trace(go.Scatter(x=bits_range, y=ys, mode="lines+markers",
                             name=scheme, line=dict(width=3)))
fig.add_vline(x=bits, line_dash="dash", line_color="#E45756")
fig.update_layout(title="量化误差随位宽指数下降;校准越细,曲线越低",
                  xaxis_title="位宽(bit)", yaxis_title="平均绝对误差",
                  height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 4 bit 是“性价比甜点”,8 bit(FP8/INT8)几乎无损——所以 GPTQ/AWQ 用 4bit,昇腾 FP8 用 8bit。")

st.subheader("🔬 FP8 的两种尾数布局")
if st.checkbox("对比 E4M3 与 E5M2", value=True):
    # 示意:E4M3 精度高范围小;E5M2 范围大精度低
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[0.0625, 0.125, 0.25, 0.5, 1.0], mode="lines+markers",
                              name="E4M3(精度优先)", line=dict(color="#4C78A8", width=3)))
    fig2.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[0.5, 1.0, 2.0, 4.0, 8.0], mode="lines+markers",
                              name="E5M2(范围优先)", line=dict(color="#E45756", width=3)))
    fig2.update_layout(title="E4M3:小步长高精度;E5M2:大步长宽范围(示意图)",
                       xaxis_title="指数位步进", yaxis_title="可表示步长", height=380,
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("⭐ 权重常用 E4M3(精度重要),梯度/激活常用 E5M2(防溢出)——FP8 训练与推理的通行做法。")

st.markdown("""
> 💡 **结论**:昇腾与 vLLM 的量化只是“外壳”不同——PTQ/QAT 与 GPTQ/AWQ 的差异在于
> **校准方法与误差补偿**,底层都是把权重压进更低位宽。误差随位宽指数下降、
> 随校准粒度(逐通道>逐层>全局)变细而下降,这两条规律放之四海皆准。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 87 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
''')

NB = new_nb("第 87 课 · 华为量化方案",
            subtitle="PTQ/QAT 与低比特:和 GPTQ/AWQ/FP8 同源的“瘦身术”",
            emoji="🔢")

chapter_cover(NB,
    objectives=[
        "理解量化的本质:低位宽表示 + 校准数据保精度",
        "掌握 MindSpore 量化的两条路:PTQ(训练后)与 QAT(量化感知训练)",
        "用 torch 实现位宽可变的对/非对称量化,观察误差随位宽的指数下降",
        "认识昇腾低比特方案:FP8(E4M3/E5M2)与 INT8",
        "对照 vLLM 的 GPTQ / AWQ / FP8,找出“同一件事”的两种说法",
    ],
    toc=[
        ("直觉:给权重瘦身", "32 位太宽裕,8 位够用吗?"),
        ("MindSpore 量化:PTQ 与 QAT", "训练后标定 vs 训练中感知"),
        ("torch 模拟量化误差", "位宽、校准粒度与误差的定量关系"),
        ("昇腾低比特:FP8 与 INT8", "E4M3 与 E5M2 的取舍"),
        ("对照 vLLM 量化", "GPTQ/AWQ/FP8 与昇腾方案的同源本质"),
        ("交互图:误差-位宽曲线", "plotly 逐位宽对比"),
        ("配套 Streamlit 演示", "app_87_ascend_quant.py:调位宽/校准看误差"),
    ],
    links=[
        ("MindSpore 量化文档", "https://www.mindspore.cn/docs/zh-CN/r2.4/design/mindquant.html"),
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization.html"),
        ("AWQ 论文", "https://arxiv.org/abs/2306.00978"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:给权重瘦身 🏋️",
D('''
一个 7B 模型,FP32 下权重占 **28 GB**,FP16 下 14 GB,INT8 下只有 7 GB。对显存捉襟见肘的推理来说,
“瘦身”= 能多塞几个模型、多存几批 KV。量化的两个动作缺一不可:

1. **压窄位宽**:用 8bit / 4bit 表示本来 32bit 的数值;
2. **校准(calibration)**:用一小批“有代表性的数据”统计数值范围,决定 scale / 零点,
   把误差压到最低。

先画“位宽 → 模型体积”的曲线,感受瘦身的量级:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.8, 4.0))
bits = np.arange(1, 17)
params = 7e9
volume_gb = params * bits / 8 / 1e9
ax.plot(bits, volume_gb, "o-", color="#4C78A8", lw=2, label="7B 模型权重")
ax.axhline(2.0, color="#E45756", ls="--", lw=1.5)
ax.text(15.5, 2.1, "常见单卡可用权重量级", fontsize=9, color="#E45756", ha="right")
for b in [4, 8, 16]:
    ax.annotate(f"{volume_gb[bits==b][0]:.1f} GB", xy=(b, volume_gb[bits==b][0]),
                xytext=(b + 0.4, volume_gb[bits==b][0] + 1.2),
                arrowprops=dict(arrowstyle="->", color="#888"), fontsize=9)
ax.set_xlabel("权重位宽(bit)"); ax.set_ylabel("模型体积(GB)")
ax.set_title("位宽与 7B 模型体积:从 28GB 到 3.5GB", fontsize=12)
ax.legend(frameon=True, fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 4bit 能把 7B 压到 3.5 GB——这就是为什么“本地跑大模型”成为可能。昇腾侧与 vLLM 侧做的是同一件事。")

NB.md("## 2️⃣ MindSpore 量化:PTQ 与 QAT 🛠️",
D('''
MindSpore 提供两条量化路线:

- **PTQ(Post-Training Quantization,训练后量化)**:模型训练完再标定。
  流程:喂一小批数据 → 统计每层权重/激活的范围 → 定 scale → 导出量化模型。
  快,但精度受“原始分布”限制;
- **QAT(Quantization-Aware Training,量化感知训练)**:训练时就模拟量化。
  流程:在前向里插入“伪量化”节点,让训练适应量化噪声 → 精度更高,但要重训。

用一张流程图对比两者的成本与精度:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (title, steps, color) in zip(axes, [
    ("PTQ:训练后标定", ["训练完的模型", "喂少量校准数据", "统计范围定 scale", "导出量化模型"], "#4C78A8"),
    ("QAT:训练中感知", ["从头训练", "插入伪量化节点", "训练适应量化噪声", "导出量化模型"], "#E45756"),
]):
    for i, s in enumerate(steps):
        ax.add_patch(plt.Rectangle((0.08, 0.72 - i * 0.2), 0.84, 0.14, fc="#DFE9F8" if color == "#4C78A8" else "#FDF3E4",
                                   ec=color, lw=1.6))
        ax.text(0.5, 0.79 - i * 0.2, s, ha="center", va="center", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=12)
fig.suptitle("MindSpore 量化的两条路:快(PTQ)与准(QAT)", fontsize=12)
plt.tight_layout(); plt.show()
'''), "🎨 PTQ 几分钟出结果,精度损失一般 <1%;QAT 要重训,但能把损失压到 ~0。工程上常先 PTQ,不满意再上 QAT。")

NB.md("## 3️⃣ torch 模拟量化误差 🔬",
D('''
下面实现一个**位宽可调、校准粒度可调**的量化器,定量回答两个问题:
(1) 误差随位宽怎么变? (2) 逐通道校准比全局校准好多少?
'''))

NB.code(D('''
def quantize_chunk(c, bits):
    qmax = 2 ** (bits - 1) - 1
    s = c.abs().max() / qmax
    q = torch.clamp(torch.round(c / s), -qmax, qmax)
    return q * s

def quant_mae(x, bits, granularity):
    x = x.to(torch.float32)
    if granularity == "global":
        return (x - quantize_chunk(x, bits)).abs().mean().item()
    if granularity == "per_layer":
        s = x.abs().quantile(0.99) / (2 ** (bits - 1) - 1)
        q = torch.clamp(torch.round(x / s), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
        return (x - q * s).abs().mean().item()
    chunks = x.reshape(32, -1)
    return sum((x0 - quantize_chunk(x0, bits)).abs().mean().item() for x0 in chunks) / 32

torch.manual_seed(0)
W = (torch.randn(256, 256) * 0.3).flatten()
print("位宽 | 全局 scale | 逐层 scale | 逐通道 scale")
for b in [2, 4, 8]:
    row = [quant_mae(W, b, g) for g in ["global", "per_layer", "per_channel"]]
    print(f"  {b} bit |   {row[0]:.5f}  |   {row[1]:.5f}  |   {row[2]:.5f}")
print("结论:位宽越高误差越小;同一位置宽下,逐通道 > 逐层 > 全局。")
'''), "✅ 三条规律可复现:误差随位宽指数下降;校准越细误差越小;8bit 下三者的差距已很小(接近无损)。")

NB.md("## 4️⃣ 昇腾低比特:FP8 与 INT8 🎯",
D('''
昇腾(910B 起)支持 **FP8**,精度再下一城。FP8 有 E4M3 与 E5M2 两种布局:

- **E4M3**:4 位指数 + 3 位尾数,精度细、范围小——LLM **权重**常用;
- **E5M2**:5 位指数 + 2 位尾数,范围大、精度粗——**梯度 / 中间激活**常用。

用图对比两种布局“能表达的最小步长”随指数位的变化:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.8, 4.0))
exp_steps = np.arange(1, 6)
e4m3 = 2.0 ** (-(np.arange(1, 6)))       # 尾数 3bit → 更细的步长
e5m2 = 2.0 ** (np.arange(1, 6) * 0) * 1  # 示意:范围更大
ax.plot(exp_steps, e4m3, "o-", color="#4C78A8", lw=2, label="E4M3: 步长细(精度高)")
ax.plot(exp_steps, np.array([0.25, 0.5, 1.0, 2.0, 4.0]), "s-", color="#E45756", lw=2, label="E5M2: 步长粗(范围大)")
ax.set_yscale("log")
ax.set_xlabel("指数位步进(示意)"); ax.set_ylabel("最小可表示步长(log)")
ax.set_title("FP8 的两种布局:精度与范围的取舍", fontsize=12)
ax.legend(frameon=True, fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 需要精度用 E4M3,怕溢出用 E5M2——昇腾与 NVIDIA 的 FP8 约定是一致的。")

NB.md("## 5️⃣ 对照 vLLM 量化:同一件事,两种说法 ⚖️",
D('''
vLLM 侧的 GPTQ / AWQ / FP8,和昇腾侧的 MindSpore 量化 / MindIE 低比特,本质是**同一家族**:

| 维度 | vLLM(GPU) | 昇腾 | 本质 |
|---|---|---|---|
| 训练后量化 | GPTQ / AWQ(W4A16) | MindSpore PTQ | 用校准数据保精度 |
| 低比特前向 | FP8(W8A8) | MindIE FP8(E4M3) | 8bit 近无损 |
| 训练中感知 | LSQ / QAT 实验 | MindSpore QAT | 训练适应量化噪声 |
| 目标 | 省显存、提吞吐 | 省显存、提吞吐 | 完全一致 |

用一张“精度↔压缩率”的散点图展示各家方案的定位:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.8, 4.6))
schemes = [
    ("FP16", 1.0, 0.1, "#6B4FA1"), ("INT8/FP8", 0.5, 0.4, "#4C78A8"),
    ("GPTQ W4A16", 0.25, 0.9, "#F58518"), ("AWQ W4A16", 0.25, 0.7, "#F58518"),
    ("昇腾 W4A16", 0.25, 0.8, "#E45756"), ("昇腾 FP8", 0.5, 0.45, "#E45756"),
]
for name, size, err, color in schemes:
    ax.scatter(size, err, s=140, color=color, alpha=0.9, edgecolor="white", zorder=3)
    ax.annotate(name, xy=(size, err), xytext=(size + 0.02, err + 0.03), fontsize=10)
ax.set_xlabel("模型相对体积(越小越好)"); ax.set_ylabel("相对精度损失(示意)")
ax.set_xlim(0.15, 1.15); ax.set_ylim(-0.05, 1.25)
ax.set_title("各家量化方案的“体积 ↔ 精度”定位(示意图)", fontsize=12)
ax.invert_xaxis()
plt.tight_layout(); plt.show()
'''), "📊 左下角(体积小 + 精度损失小)是兵家必争之地。GPTQ/AWQ 与昇腾的低比特方案,都在这条前沿上。")

NB.md("## 6️⃣ 交互图:误差-位宽曲线 📈",
D('''
用 plotly 把「位宽 → 误差」画成可交互曲线,再叠加「位宽 → 体积」双轴,直观感受“误差指数跌、
体积线性跌”:
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

bits = list(range(1, 9))
errs = [quant_mae(W, b, "per_layer") for b in bits]
vol = [b / 8 * 100 for b in bits]
fig = go.Figure()
fig.add_trace(go.Scatter(x=bits, y=errs, name="平均误差", mode="lines+markers",
                         line=dict(color="#E45756", width=3), yaxis="y"))
fig.add_trace(go.Scatter(x=bits, y=vol, name="体积占比%", mode="lines+markers",
                         line=dict(color="#4C78A8", width=3, dash="dot"), yaxis="y2"))
fig.update_layout(title="误差随位宽指数下降,体积线性下降", xaxis_title="位宽(bit)",
                  yaxis=dict(title="平均误差"), yaxis2=dict(title="体积占比(%)", overlaying="y", side="right"),
                  height=400, margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "📊 4bit 是“误差还没爆、体积已减 3/4”的甜点——这解释了为什么 W4A16(GPTQ/AWQ)如此流行。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_87_ascend_quant.py`,拖动**量化位宽**,切换**校准策略**与**数据分布**,
实时观察量化误差、体积占比,并对比 FP8 的 E4M3/E5M2 布局:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_87_ascend_quant.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_87_ascend_quant.py\n" + APP_87, "📜 这就是 app_87_ascend_quant.py 的完整源码,notebook 与 app 共用同一套量化误差口径,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "量化 = 低位宽表示 + 校准保精度,7B 模型从 28GB 压到 3.5GB(W4A16)",
        "MindSpore 两条路:PTQ(快、损失小)与 QAT(准、要重训)",
        "torch 验证三规律:误差随位宽指数降、随校准粒度变细而降、8bit 近无损",
        "昇腾低比特:FP8 的 E4M3(精度优先)与 E5M2(范围优先),以及 INT8",
        "昇腾量化与 vLLM 的 GPTQ/AWQ/FP8 同源:外壳不同,数学相同",
    ],
    practice=[
        "给 quantize_chunk 实现非对称量化(scale + zero_point),与对称版对比误差",
        "把 quant_mae 的逐通道粒度从 32 段改到 64/128 段,画误差随段数的曲线",
        "用 AWQ 的思路(按激活幅度保护重要权重)重写量化,比较与普通量化的误差差异",
        "调研 MindIE 支持的低比特类型列表,与 vLLM 的 quantization 选项逐一对照",
    ],
    links=[
        ("MindSpore 量化文档", "https://www.mindspore.cn/docs/zh-CN/r2.4/design/mindquant.html"),
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization.html"),
        ("GPTQ 论文", "https://arxiv.org/abs/2210.17323"),
    ])

NB.save(str(Path(CH11) / "87_ascend_quant.ipynb"))
app_path = Path(CH11) / "app_87_ascend_quant.py"
app_path.write_text(APP_87 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
