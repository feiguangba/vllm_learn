# -*- coding: utf-8 -*-
"""生成 97_ascend_sparsity.ipynb 与 app_97_ascend_sparsity.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_97 = D('''
# -*- coding: utf-8 -*-
# app_97_ascend_sparsity.py — 昇腾量化与稀疏化:参数交互实验 🔢
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾量化与稀疏化 🔢", layout="wide")
st.title("🔢 第 97 课 · 昇腾量化与稀疏化:瘦身提速交互实验")

st.markdown("""
**量化** = 用更少的位存权重(W8A8 / FP8),显存减半、计算变快;**稀疏化** = 只保留重要的
权重(2:4 结构化稀疏 / 剪枝),跳过零运算。两者都是"用一点精度换一大截性能"。
拖一拖参数,观察精度损失与加速比的此消彼长。
""")

st.sidebar.header("🎛️ 参数")
bits = st.sidebar.slider("量化位数", 4, 16, 8, 1)
sparse_ratio = st.sidebar.slider("稀疏比例(0-90%)", 0, 90, 50, 5)
mode = st.sidebar.radio("稀疏模式", ["非结构化剪枝", "2:4 结构化稀疏"])
weight_gb = st.sidebar.slider("权重规模(GB)", 1, 64, 14, 1)
st.sidebar.caption("量化位数越低越省,但误差越大;稀疏比例越高越快,但精度掉得越快 —— 鱼与熊掌。")

# ---- 量化误差模拟:误差 ~ 2^(-bits+1) ----
qerr = 2 ** -(bits - 1) * 0.02
mem = weight_gb * (bits / 16)                     # 位宽减半 → 权重减半
# ---- 稀疏加速模拟 ----
if mode == "2:4 结构化稀疏":
    eff = min(50, sparse_ratio) * 0.9             # 2:4 的硬件收益更稳定
else:
    eff = min(50, sparse_ratio) * 0.7             # 非结构化收益打折
speedup = 1 + eff / (100 - eff + 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("量化位数", f"{bits} bit")
c2.metric("相对精度损失", f"{qerr*100:.2f}%")
c3.metric("权重占用", f"{mem:.1f} GB(省 {100*(1-bits/16):.0f}%)")
c4.metric("稀疏加速(示意)", f"{speedup:.2f}×")

st.subheader("📉 量化误差 vs 位数")
b = np.arange(4, 17)
errs = 2 ** -(b - 1) * 0.02 * 100
fig = go.Figure()
fig.add_trace(go.Scatter(x=b, y=errs, mode="lines+markers",
                         line=dict(color="#c0392b", width=3), name="相对误差 %"))
fig.add_vline(x=bits, line_dash="dash", line_color="#888")
fig.update_layout(title="量化位数越低,误差指数级上升(对数感)", xaxis_title="位数(bits)",
                  yaxis_title="相对误差(%)", height=320,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧊 稀疏矩阵可视化(模拟)")
n = 16
rng = np.random.default_rng(0)
w = rng.normal(size=(n, n))
w[abs(w) < np.quantile(abs(w), 1 - sparse_ratio / 100)] = 0   # 剪掉最小的一批
fig2 = go.Figure(go.Heatmap(z=w, colorscale="RdBu", zmid=0,
                            showscale=False,
                            hovertemplate="值: %{z:.3f}<extra></extra>"))
fig2.update_layout(title=f"稀疏权重矩阵({mode},稀疏比例 {sparse_ratio}%)",
                   height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:量化管「存得少、算得快」,稀疏管「跳着算」。生产实践通常**先量化后稀疏**,
> 且用「精度回退」指标监控 —— 一旦准确率掉出红线,就回调一档。昇腾上 MindIE 已内置
> W8A8 与 KV 量化,稀疏则多见于剪枝后的推理模型。
""")
''')

NB = new_nb("第 97 课 · 昇腾量化与稀疏化:让模型瘦身提速",
            subtitle="W8A8 / INT8 / FP8 量化 · 2:4 结构化稀疏 · 与 GPU 量化工具链对照",
            emoji="🔢")

chapter_cover(NB,
    objectives=[
        "理解量化的本质:用更少的位存同样的信息,scale + round 两步",
        "掌握 W8A8 / INT8 / FP8 等主流量化格式与校准概念",
        "量化误差与位数的关系:为什么 8bit 是甜点位",
        "理解稀疏化:非结构化剪枝 vs 2:4 结构化稀疏",
        "对照 GPU(GPTQ/AWQ/FP8)与昇腾(MindIE W8A8/FP8)的量化工具链",
        "用 numpy/torch 模拟量化与稀疏,画误差曲线与稀疏热力图,配 App 交互",
    ],
    toc=[
        ("直觉:缩略图与速记", "量化 = 少存;稀疏 = 只记重要的"),
        ("量化数学:scale 与 round", "手写 INT8 量化,量误差"),
        ("位宽误差曲线", "从 4bit 到 16bit,误差怎么变"),
        ("昇腾量化工具链", "MindIE W8A8 / FP8 / 量化感知训练"),
        ("稀疏化:跳过零运算", "2:4 结构化 vs 非结构化剪枝"),
        ("GPU vs 昇腾对照", "GPTQ/AWQ ↔ W8A8/FP8 一张表"),
        ("配套 App", "app_97_ascend_sparsity.py:参数交互"),
    ],
    links=[
        ("MindIE 推理引擎", "https://www.hiascend.com/software/mindie"),
        ("昇腾量化文档", "https://www.hiascend.com/document"),
        ("FP8 在 LLM 中的实践", "https://arxiv.org/abs/2306.04803"),
        ("2:4 结构化稀疏(NVIDIA 论文)", "https://arxiv.org/abs/2104.08378"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:缩略图与速记 📸",
D('''
大模型权重动辄几十 GB,推理时又要快 —— 怎么破?两个古老的思路:

- **量化**:像把高清照片压成"缩略图" —— 用更少的比特(bits)存同一个数字。
  fp16 变 int8,显存直接减半;代价是精度(照片变糊一点点);
- **稀疏化**:像"速记只记关键词" —— 权重里很多值本来就接近于 0,把它们剪掉/跳过,
  运算量就变少了。

本课把这两件事讲透:先动手量化,再动手稀疏,最后对比昇腾与 GPU 的工具链差异。
'''))

NB.md("## 2️⃣ 量化数学:scale 与 round 🧮",
D('''
最朴素的对称量化只有两步:

$$\\hat{x} = \\text{round}\\left(\\frac{x}{s}\\right), \\qquad s = \\frac{\\max|x|}{Q_{\\max}}$$

- `s`(scale):把浮点范围压到整数范围的比例;
- `round`:取整(舍入误差是精度损失的主要来源);
- 反量化:`x ≈ round(x/s) * s`。

权重是 W8A8(权重 8bit、激活 8bit)、还是 FP8、还是 per-tensor/per-channel,差别只在
"scale 怎么选、误差怎么分布"。先手写一个 INT8 量化器,看看真实误差:
'''))

NB.code(D('''
def quantize_int8(x, bits=8):
    """对称量化:返回 (量化后的整数, scale)"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().item() / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q, scale

def dequantize(q, scale):
    return q * scale

torch.manual_seed(0)
w = torch.randn(512, 512) * 0.1
for bits in [4, 8, 12, 16]:
    q, s = quantize_int8(w, bits)
    wq = dequantize(q, s)
    rel_err = ((w - wq).abs().mean() / w.abs().mean()).item()
    print(f"{bits:2d}bit:平均相对误差 = {rel_err:.4f}   (占用 = fp16 的 {bits/16*100:.0f}%)")

q8, s8 = quantize_int8(w, 8)
wq8 = dequantize(q8, s8)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].imshow(w[:64, :64], cmap="RdBu", vmin=-0.4, vmax=0.4)
axes[0].set_title("原始 fp32 权重")
axes[1].imshow(wq8[:64, :64], cmap="RdBu", vmin=-0.4, vmax=0.4)
axes[1].set_title("INT8 反量化后(肉眼看几乎一样)")
plt.tight_layout()
'''),
"🧮 看数字:8bit 时平均相对误差已压到千分之一量级,肉眼热力图几乎无差别 —— 这就是 8bit 成为甜点位的原因。")

NB.md("## 3️⃣ 位宽误差曲线:8bit 为什么是甜点位 🍬",
D('''
误差随位宽**指数下降**:每加 1 bit,量化间隔减半。扫一遍 4→16 bit,画相对误差曲线,
观察拐点在哪:
'''))

NB.code(D('''
bits = np.arange(4, 17)
errs = []
for b in bits:
    q, s = quantize_int8(w, int(b))
    wq = dequantize(q, s)
    errs.append(((w - wq).abs().mean() / w.abs().mean()).item())

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(bits, np.array(errs) * 100, "o-", color="#c0392b", lw=2.2, ms=6)
ax.axvline(8, color="#27ae60", ls="--", lw=1.5)
idx8 = int(np.where(bits == 8)[0][0])
ax.text(8.1, errs[idx8] * 100, "8bit 甜点位", fontsize=9, color="#27ae60")
ax.set_yscale("log")
ax.set_xlabel("量化位数 (bits)"); ax.set_ylabel("平均相对误差 %(对数)")
ax.set_title("量化误差随位宽指数下降:4bit 糊、16bit 几乎无损")
plt.tight_layout()
'''),
"📈 误差曲线:8bit 之后误差下降开始「躺平」,而 16bit 权重占用翻倍 —— 在『误差与体积』的权衡里,8bit 是最划算的一档。")

NB.md("## 4️⃣ 昇腾量化工具链 🔧",
D('''
昇腾上做量化,主要靠这些家伙:

- **MindIE 内置 W8A8**:MindIE 推理引擎原生支持 W8A8(权重 INT8 + 激活 INT8)与 KV 量化,
  用 `--quantization W8A8` 之类参数即可打开,无需改模型代码;
- **FP8 支持**:新一代昇腾(910B/C 系列)支持 FP8 计算,配合 MindSpore 的 AMP 混合精度;
- **量化感知训练(QAT)**:MindSpore 提供 QAT 能力 —— 在训练时就"模拟量化噪声",
  让模型适应低比特(比事后 PTQ 更稳);
- **AMCT / MindSpore 量化工具**:把 PyTorch/MindSpore 模型转成可量化的图再导出。

一句话:**昇腾的量化路径 = MindIE(推理时量化)+ MindSpore(训练时 QAT)+ CANN(算子支持)**。
下面把"推理时 PTQ 与训练时 QAT"的精度差异画个对比:
'''))

NB.code(D('''
# PTQ 与 QAT 的精度对比(示意):QAT 在低比特时明显更稳
bits_ax = np.arange(4, 9)
ptq = 0.9 - 0.08 * (8 - bits_ax)          # PTQ:位宽越低掉越多
qat = 0.94 - 0.04 * (8 - bits_ax)         # QAT:模拟量化,更抗噪
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(bits_ax, ptq, "o-", color="#c0392b", lw=2.2, label="PTQ(事后量化)")
ax.plot(bits_ax, qat, "s-", color="#2e86c1", lw=2.2, label="QAT(量化感知训练)")
ax.set_xlabel("量化位数 (bits)"); ax.set_ylabel("相对准确率(示意)")
ax.set_title("低比特下 QAT 比 PTQ 更稳:训练时提前适应噪声")
ax.legend(); ax.grid(True, ls="--", alpha=0.5); plt.tight_layout()
'''),
"📊 PTQ vs QAT:8bit 差别不大,4-6bit 时 QAT 明显更稳 —— 要冲极限压缩,就得用训练时量化。")

NB.md("## 5️⃣ 稀疏化:跳过零运算 🕳️",
D('''
稀疏化分两种:

1. **非结构化剪枝**:把绝对值最小的权重直接置零 —— 灵活但"零"的位置是乱的,
   硬件难以加速(只能靠软件跳过);
2. **2:4 结构化稀疏**:每 4 个连续权重里强制留 2 个 —— 位置规律,硬件可以用
   "压缩存储 + 跳过零"的方式稳定提速(NVIDIA Ampere 之后,昇腾部分算力也支持类似思路)。

加速收益的直觉:**算一个 matmul 的 FLOPs = 2·M·N·K,稀疏掉一半,理论上 FLOPs 减半**。
但真实加速取决于硬件支不支持"跳过零",所以"理论 vs 实际"要分开看。动手模拟 2:4:
'''))

NB.code(D('''
def sparsity_24(w):
    """2:4 结构化稀疏:每 4 个中保留绝对值最大的 2 个,其余置零"""
    out = w.clone()
    m, k = out.shape
    for i in range(m):
        for j in range(0, k, 4):
            block = out[i, j:j+4]
            if block.numel() == 4:
                keep = torch.topk(block.abs(), 2).indices
                mask = torch.zeros(4, dtype=torch.bool); mask[keep] = True
                out[i, j:j+4] = block * mask
    return out

torch.manual_seed(3)
W = torch.randn(64, 128) * 0.1
Ws = sparsity_24(W)
real_ratio = (Ws == 0).float().mean().item()
print(f"2:4 稀疏后:实际零占比 = {real_ratio*100:.0f}% (应为 50%)")
print(f"理论 FLOPs 削减 = {real_ratio*100:.0f}% → 理论加速 ≤ {1/(1-real_ratio):.2f}×")

fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(Ws[:32, :], cmap="gray_r", aspect="auto")
ax.set_title("2:4 结构化稀疏后的权重(黑=零,每 4 列恰好 2 个黑)")
ax.set_xlabel("K 维(每 4 列一组)"); ax.set_ylabel("M 维")
plt.tight_layout()
'''),
"🕳️ 2:4 热力图:每 4 列恰好 2 个黑格 —— 位置规律,硬件好加速;非结构化剪枝则「到处是洞」,加速难。")

NB.md("## 6️⃣ 稀疏加速:理论 vs 实际 📉",
D('''
画一张"稀疏比例 → 理论加速"与"实际加速(打个折扣)"的对比曲线,提醒自己别被
理论值骗了:
'''))

NB.code(D('''
ratios = np.arange(0.1, 0.91, 0.05)
theoretical = 1 / (1 - ratios)                    # 理论:FLOPs 减多少快多少
practical = 1 + 0.6 * ratios / (1 - ratios)       # 实际:只有 60% 的效率(示意)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ratios * 100, theoretical, "o-", color="#c0392b", lw=2.2, label="理论加速(FLOPs 反比)")
ax.plot(ratios * 100, practical, "s-", color="#27ae60", lw=2.2, label="实际加速(硬件打折)")
ax.set_xlabel("稀疏比例 %"); ax.set_ylabel("加速比")
ax.set_title("稀疏加速:理论很美,实际要打折")
ax.legend(); ax.grid(True, ls="--", alpha=0.5); plt.tight_layout()
'''),
"📉 注意:90% 稀疏理论上 10×,实际可能只有 3-4× —— 因为零跳过也要开销。评估稀疏方案要看『实测』,别只看 FLOPs。")

NB.md("## 7️⃣ GPU vs 昇腾:量化工具链对照 ⚖️",
D('''
把两边的量化/稀疏手段摆在一起:

| 目标 | GPU(NVIDIA) | 昇腾(NPU) |
|---|---|---|
| 权重量化 | GPTQ / AWQ(4bit) | MindIE W8A8 / FP8 |
| 激活量化 | INT8 校准(Calibration) | W8A8 动态量化 |
| KV 量化 | fp8 kv cache(vLLM) | MindIE KV 量化 |
| 训练时量化 | QAT(如 tensorRT QAT) | MindSpore QAT |
| 结构化稀疏 | 2:4 稀疏(Tensor Core) | 部分算力支持 2:4 |
'''))

NB.code(D('''
rows = ["权重量化", "激活量化", "KV 量化", "训练时量化", "结构化稀疏"]
gpu = [5, 4, 4, 4, 4]
asc = [4, 4, 4, 4, 3]
df = pd.DataFrame({"GPU(GPTQ/AWQ/FP8)": gpu, "昇腾(W8A8/FP8/QAT)": asc}, index=rows)
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(df.T, annot=True, fmt="d", cmap="YlGnBu", linewidths=1, linecolor="white",
            cbar=False, ax=ax)
ax.set_title("量化/稀疏能力成熟度对比(5=最成熟)")
plt.tight_layout()
'''),
"📊 热力图:两边手段几乎一一对应 —— GPU 上会的(GPTQ/AWQ/FP8),昇腾上用 MindIE/MindSpore 都能找到等价物。")

NB.md("## 8️⃣ 配套 App:参数交互实验 🎛️",
D('''
运行同目录的 `app_97_ascend_sparsity.py`,**拖量化位数、稀疏比例,切稀疏模式**,
实时看精度损失、加速比与稀疏矩阵热力图:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_97_ascend_sparsity.py
```

浏览器打开 **http://localhost:8501**(也可 `--server.port 8697`)。完整源码如下:
'''))

NB.code(app_cell("app_97_ascend_sparsity.py", APP_97),
"📜 运行本 cell 会覆盖写入 `app_97_ascend_sparsity.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "量化本质 = scale + round,误差随位宽指数下降,8bit 是甜点位",
        "昇腾量化路径:MindIE W8A8/FP8(推理)+ MindSpore QAT(训练)+ CANN 算子支持",
        "稀疏化:非结构化剪枝灵活难加速,2:4 结构化稀疏位置规律易加速",
        "理论加速(FLOPs 反比)要打折扣看 —— 以实测为准",
        "GPU 与昇腾的量化/稀疏手段几乎一一对应,思维可平移",
    ],
    practice=[
        "给 quantize_int8 加 per-channel scale(每行一个 scale),对比误差变化",
        "把第 5 节 W 改成 4 的倍数不可整除的 K 值,观察 2:4 稀疏如何处理尾部",
        "调研 MindIE 的 W8A8 参数,写出在 vllm-ascend / MindIE 上启用的命令示例",
        "设计一个小实验:量化 + 稀疏叠加,观察误差是相加还是相乘",
    ],
    links=[
        ("MindIE 推理引擎", "https://www.hiascend.com/software/mindie"),
        ("昇腾量化开发文档", "https://www.hiascend.com/document"),
        ("FP8 论文", "https://arxiv.org/abs/2306.04803"),
        ("2:4 结构化稀疏论文", "https://arxiv.org/abs/2104.08378"),
    ])

out = str(Path(CH11) / "97_ascend_sparsity.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_97_ascend_sparsity.py"
app_path.write_text(APP_97 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
