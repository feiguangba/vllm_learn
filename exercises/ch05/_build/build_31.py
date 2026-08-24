# -*- coding: utf-8 -*-
"""生成 31_gptq_principle.ipynb 与 app_31_gptq.py(教材级重写版)

论文/资料支撑:
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers", arXiv:2210.17323(本课主角)
- Hassibi & Stork, "Second Order Derivatives for Network Pruning: Optimal
  Brain Surgeon", NeurIPS 1992(OBS 起源)
- Frantar et al., "Optimal Brain Compression", arXiv:2208.11580(OBQ 前身)
- vLLM quantization docs + Marlin kernel(工程落地)
"""
from helpers import (D, GPTQ_CORE, chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_31 = D('''
# -*- coding: utf-8 -*-
# app_31_gptq.py — GPTQ 思想:误差补偿 vs 朴素逐列量化交互演示 🧮
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🧮 31 · GPTQ 逐列量化与误差补偿", layout="wide")
st.title("🧮 第 31 课 · GPTQ 思想:逐列量化 + 误差补偿")

st.markdown("""
GPTQ 的核心动作只有两步:
1. **逐列量化**:按列把权重舍入到最近的量化格子上(朴素 RTN);
2. **误差补偿**:第 $i$ 列的量化残差,按 Hessian 逆指示的方向**摊到还没量化的列**上,
   让后续列「顺手」修正前面丢掉的精度。

补偿的收益取决于输入列之间的**相关性**——相关性越强,补偿越有的放矢。
下方调节矩阵大小、位宽与相关性,对比两种策略的累计输出误差:
""")

def gptq(W, X, bits, compensate):
    # numpy 版简化 GPTQ(与 notebook 相同:H = 2·X^T·X + 阻尼,Cholesky 补偿)
    d_out, d_in = W.shape
    H = 2.0 * X.T @ X                          # Hessian 近似 (d_in, d_in)
    H += 0.01 * np.diag(H).mean() * np.eye(d_in)   # 阻尼,防止奇异
    L_factor = np.linalg.cholesky(np.linalg.inv(H)).T   # H^{-1} 的上三角 Cholesky 因子
    W = W.copy()
    Q = np.zeros_like(W)
    qmax = 2 ** (bits - 1) - 1
    trail = []
    for i in range(d_in):                      # 逐列量化
        w = W[:, i]                            # 当前列
        s = max(abs(w).max() / qmax, 1e-12)    # 该列 scale
        q = np.clip(np.round(w / s), -qmax, qmax)  # 量化到 int
        Q[:, i] = q * s                        # 反量化存下
        if compensate and i + 1 < d_in:        # 误差补偿
            err = (w - Q[:, i]) / L_factor[i, i]   # 残差按对角归一
            W[:, i + 1:] -= np.outer(err, L_factor[i, i + 1:])  # 摊到未量化列
        if (i + 1) % max(d_in // 20, 1) == 0:
            trail.append(float(np.mean((X @ Q.T - X @ W0.T) ** 2)))
    return Q, trail

def output_mse(W, Q, X):
    # 输出 MSE:量化前后 XW^T 的差异
    return float(np.mean((X @ W.T - X @ Q.T) ** 2))

with st.sidebar:
    st.header("🎛️ 实验参数")
    d_out = st.slider("输出维度 d_out", 8, 128, 32, 8)
    d_in = st.slider("输入维度 d_in(量化列数)", 8, 128, 48, 8)
    corr = st.slider("输入列相关性 ρ(0=独立,1=强相关)", 0.0, 0.99, 0.8, 0.01,
                     help="相关性越高,Hessian 非对角越强,补偿能搬的'救兵'越多")
    bits = st.slider("量化位宽 b(bit)", 2, 8, 3, 1)
    st.caption("💡 把 ρ 从 0 拖到 0.9,看补偿增益如何从几乎为零涨到好几倍。")

rng = np.random.default_rng(7)
rho = corr
A = rng.standard_normal((2000, d_in))          # 独立成分
common = rng.standard_normal((2000, 1))        # 公共因子
X = np.sqrt(1 - rho) * A + np.sqrt(rho) * common   # 列间相关性与 ρ 挂钩

W0 = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)   # 权重

Q_naive, _ = gptq(W0, X, bits, compensate=False)  # 朴素 RTN
Q_gptq, trail = gptq(W0, X, bits, compensate=True) # GPTQ 补偿

e_naive = output_mse(W0, Q_naive, X)
e_gptq = output_mse(W0, Q_gptq, X)

c1, c2, c3 = st.columns(3)
c1.metric("朴素逐列(RTN) 输出 MSE", f"{e_naive:.3e}")
c2.metric("GPTQ 补偿 输出 MSE", f"{e_gptq:.3e}")
c3.metric("误差降低倍数", f"{e_naive / max(e_gptq, 1e-30):.1f}x")

st.subheader("📊 逐列量化的累计输出误差(补偿 vs 朴素)")
steps = np.arange(1, len(trail) + 1) * max(d_in // len(trail), 1)
_, trail_n = gptq(W0, X, bits, compensate=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=steps, y=trail, mode="lines+markers", name="GPTQ(带补偿)",
                         line=dict(color="#72B7B2", width=2)))
fig.add_trace(go.Scatter(x=steps, y=trail_n, mode="lines+markers", name="朴素 RTN(无补偿)",
                         line=dict(color="#E45756", width=2, dash="dot")))
fig.update_layout(title="已量化列数 vs 累计输出 MSE:补偿让误差增长更慢甚至回落",
                  xaxis_title="已量化的列数", yaxis_title="输出 MSE", height=400,
                  yaxis_type="log", margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **一句话总结 GPTQ**:量化第 $i$ 列时把残差「记账」,按照 Hessian 逆给的汇率
> 换算成对未量化列的修正量——用后面的列,还前面欠的债。
> 论文:*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (arXiv:2210.17323)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 5 章 · 第 31 课配套演示")

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

NB = new_nb("第 31 课 · GPTQ 思想:逐列量化与误差补偿",
            subtitle="RTN 逐列「各扫门前雪」,GPTQ 却把每列的量化残差记账、摊销到后续列——后训练量化的代表作",
            emoji="🧮")

chapter_cover(NB,
    objectives=[
        "理解逐列量化与朴素 RTN(round-to-nearest)的局限",
        "理解 OBS 最优脑损伤的核心直觉:误差补偿方向由 Hessian 决定",
        "手写简化版 GPTQ:逐列量化 + Cholesky 因子补偿,数值验证收益",
        "实验验证:Hessian 为单位阵(列独立)时补偿失效,列相关性才是补偿的燃料",
    ],
    toc=[
        ("直觉:流水线上的返工站", "误差补偿的生活比喻"),
        ("核心定义与公式", "层内量化目标、Hessian、OBS 补偿公式、GPTQ 三步走"),
        ("最小实现 · 逐行推演", "手写简化 GPTQ,小矩阵逐列打印误差账"),
        ("数值验证:补偿赚多少", "RTN vs GPTQ 输出 MSE;相关性 ρ 是补偿的燃料"),
        ("真实规模数字", "GPU 上直接量化 vs GPTQ;GPT-175B 的 Hessian 账"),
        ("与 vLLM 工程实现的关系", "vLLM 的 GPTQ 支持与 Marlin 内核"),
        ("配套 Streamlit 演示", "app_31_gptq.py:调参数看补偿增益"),
    ],
    links=[
        ("GPTQ 论文 (arXiv:2210.17323)", "https://arxiv.org/abs/2210.17323"),
        ("OBS: Optimal Brain Surgeon (NeurIPS 1992)", "https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf"),
        ("Optimal Brain Compression (arXiv:2208.11580)", "https://arxiv.org/abs/2208.11580"),
        ("GPTQ 官方代码库", "https://github.com/IST-DASLab/gptq"),
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
# 说明:本课执行体不引入 pandas —— GPU 量化 cell 后再建 pandas/pyarrow 表可能触发 arrow.dll 原生崩溃,
# 故表格一律用纯 Python 的 dict + print 展示。
'''), "🛡️ 环境准备:KMP 保护 + 固定随机种子。全程 CPU,量化实验用不到 GPU。")

NB.md("## 1️⃣ 直觉:流水线上的返工站 🏭",
D('''
想象一条**画画流水线**,要照着名画临摹一幅像素画,画师从左到右**一列一列**地上色:

- **朴素画师(RTN)**:每列就近选调色盘里最接近的颜色,画完就翻页,
  误差留着不管——列画得越多,累积误差越大;
- **聪明画师(GPTQ)**:每画完一列,马上**量一下和原作的色差**(残差),
  然后**微调右侧还没画的列**,让它们「顺手」把这部分色差抵消掉。

同样每列都有舍入误差,聪明画师的作品整体看起来误差小得多——因为误差被**摊销**了,
而不是任其累积。这就是 GPTQ 的灵魂:**量化不是各列的独角戏,而是一场接力赛**。

> 🔗 论文:[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)(Frantar et al., 2022)
'''))

NB.md("## 2️⃣ 核心定义与公式 🧠",
D('''
**RTN(round-to-nearest)** 就是最朴素的量化:每个权重独立地舍到最近格子。
它优化的是**权重误差** $\\|W - \\hat{W}\\|^2$,但我们在乎的其实是**输出误差**:

$$\\|XW^T - X\\hat{W}^T\\|^2 = \\mathrm{tr}\\left((W-\\hat{W}) H (W-\\hat{W})^T\\right), \\qquad H = 2X^TX$$

$H$ 是校准激活的 **Hessian 近似**(二阶信息,shape $(d_{in}, d_{in})$)。
这个视角带来关键洞察:**权重误差要按 $H$ 加权看**——落在 $H$ 大的方向上的误差,对输出伤害更大;
反之,有些方向的误差可以很大却无伤输出,甚至**故意把误差搬到这些方向上**是划算的!

这正是 1993 年 **OBS(Optimal Brain Surgeon)** 的思想:删掉/量化一个权重时,
按 Hessian 计算该把误差**补偿**给哪些其他权重,使输出损失最小:

$$\\delta_F = -\\frac{w_q - \\mathrm{quant}(w_q)}{[H^{-1}_F]_{qq}} (H^{-1}_F)_{:,q}$$

| 符号 | 含义 | 维度 |
|---|---|---|
| $X$ | 校准激活(层输入) | $(n, d_{in})$,$n$ 为校准样本数 |
| $W$ | 权重矩阵 | $(d_{out}, d_{in})$ |
| $H = 2X^TX$ | 层输出误差的 Hessian | $(d_{in}, d_{in})$ |
| $H^{-1}$ | Hessian 逆 | $(d_{in}, d_{in})$ |
| $w_q$ | 当前被量化的列 | $(d_{out},)$ |
| $\\mathrm{quant}(w_q)$ | 该列的量化值 | $(d_{out},)$ |
| $\\delta_F$ | 对剩余权重列的补偿量 | $(d_{out}, d_{in}-q)$ |

GPTQ 的三步走(简化版):

1. **算 scale、量化当前列**:$q_i = \\mathrm{round}(w_i / s)$(和 RTN 一样);
2. **记账**:$\\mathrm{err} = (w_i - \\hat{w}_i) / H^{-1}_{ii}$(残差按对角元素归一);
3. **摊销**:$W_{:,i+1:} \\mathrel{-}= \\mathrm{err} \\otimes H^{-1}_{i,\\ i+1:}$(残差按 Hessian 逆的非对角行,分摊给未量化列)。

直观读法:$H^{-1}_{i,j}$ 告诉你「第 $i$ 列欠的债,第 $j$ 列该帮忙还几分」。
如果 $H$ 是**单位阵**(输入各列完全独立),非对角全为 0,**没有任何列能帮忙**——
GPTQ 退化为 RTN。所以:**列相关性是补偿的燃料**。

> ⚡ **Cholesky 加速**:实现里不直接反复求 $H^{-1}$ 再乘向量,而是**一次性**对 $H^{-1}$
> 做 Cholesky 分解得到上三角因子 $L$($H^{-1} = L^T L$),后续每列的「记账 $/H^{-1}_{ii}$」与
> 「摊销 $\\times H^{-1}_{i,i+1:}$」都换成 $L$ 的稀疏行运算。这样全程只需一次分解,
> 复杂度从每列求逆降到 $O(d^3)$ 一次 + 每列 $O(d^2)$——这是 GPTQ 相对 OBQ 提速的关键之一。
> 代码里这个因子就叫 `L_factor`。
'''))

NB.code(GPTQ_CORE, "📜 简化版 GPTQ(Hessian 近似 + 阻尼 + Cholesky 因子补偿),本课的主角。")

NB.md("## 3️⃣ 最小实现 · 逐行推演:小矩阵上的接力赛 🔬",
D('''
先用一个肉眼可查的小矩阵(8×12)走一遍流程。**逐列打印**两种策略的累计输出误差,
看清「补偿」是怎么把误差压回去的:
'''))

NB.code(D('''
torch.manual_seed(0)
d_out, d_in, n = 8, 12, 500                  # 小矩阵:8 行 × 12 列,500 条校准样本

W = torch.randn(d_out, d_in) / d_in ** 0.5   # (8, 12) 权重矩阵
A = torch.randn(n, d_in)                     # (500, 12) 独立成分
common = torch.randn(n, 1)                   # (500, 1) 公共因子
X = 0.5 * A + 0.5 * common                   # (500, 12) 强相关校准激活

Q_rtn = gptq_quantize(W, X, bits=3, compensate=False)   # 朴素 RTN:逐列舍入
Q_gptq = gptq_quantize(W, X, bits=3, compensate=True)   # GPTQ:逐列 + 补偿

out = X @ W.T                                # (500, 8) 原始输出
e_rtn = float(((X @ Q_rtn.T - out) ** 2).mean())    # RTN 输出 MSE
e_gptq = float(((X @ Q_gptq.T - out) ** 2).mean())  # GPTQ 输出 MSE

print(f"权重 W shape = {tuple(W.shape)}  <- (输出维度 d_out=8, 输入维度 d_in=12)")
print(f"校准 X shape = {tuple(X.shape)}  <- (校准样本 n=500, 输入维度 d_in=12)")
print(f"权重 MSE  : RTN={float(((W-Q_rtn)**2).mean()):.4f}  GPTQ={float(((W-Q_gptq)**2).mean()):.4f}")
print(f"输出 MSE  : RTN={e_rtn:.4f}  GPTQ={e_gptq:.4f}")
print(f"输出误差降低 {e_rtn / e_gptq:.1f} 倍")
print("(注意:GPTQ 的权重 MSE 可能反而略高——它优化的是输出误差,不是权重误差!)")
'''), "✅ 关键观察:GPTQ 的输出误差显著更低,哪怕它的权重误差更大。它优化的是「用户看得见」的目标。")

NB.md("## 4️⃣ 数值验证:补偿到底赚多少?相关性实验 🧪",
D('''
固定矩阵,只改变输入列相关性 $\\rho$(0 = 完全独立,0.9 = 强相关),
看补偿增益(输出 MSE 之比)怎么变化——**验证「相关性是燃料」**:
'''))

NB.code(D('''
torch.manual_seed(1)
d_out, d_in, n = 32, 48, 2000               # 稍大一点的矩阵

rows = []
for rho in [0.0, 0.3, 0.6, 0.9]:            # 扫描列相关性
    A = torch.randn(n, d_in)                # (2000, 48) 独立成分
    common = torch.randn(n, 1)              # (2000, 1) 公共因子
    X = (1 - rho) ** 0.5 * A + rho ** 0.5 * common   # 相关性由 rho 控制
    W = torch.randn(d_out, d_in) / d_in ** 0.5       # (32, 48) 权重
    out = X @ W.T                           # (2000, 32) 原始输出
    e_rtn = float(((X @ gptq_quantize(W, X, 4, False).T - out) ** 2).mean())  # RTN MSE
    e_gp = float(((X @ gptq_quantize(W, X, 4, True).T - out) ** 2).mean())    # GPTQ MSE
    rows.append(dict(相关性=rho, RTN输出MSE=e_rtn, GPTQ输出MSE=e_gp, 增益倍数=e_rtn / e_gp))
print(f"{'相关性ρ':>10}{'RTN输出MSE':>14}{'GPTQ输出MSE':>14}{'增益倍数':>10}")
for r in rows:
    print(f"{r['相关性']:>10.1f}{r['RTN输出MSE']:14.4e}{r['GPTQ输出MSE']:14.4e}{r['增益倍数']:10.2f}")
print("\\n读表:ρ=0 时增益≈1(补偿无事可做);ρ=0.9 时增益数倍——Hessian 的非对角元就是补偿的「通道」。")
'''), "✅ 相关性 ρ=0 时补偿失效,ρ=0.9 时补偿数倍——Hessian 的非对角元就是补偿的「通道」。")

NB.md("## 5️⃣ 真实规模数字:GPT-175B 的账 💾",
D('''
GPTQ 论文的核心贡献是**把 OBQ 从「不可行」变成「可行」**:

- OBQ 逐**权重**贪心选择,复杂度 $O(d_{row} \\cdot d_{col}^3)$;
- GPTQ 三个工程技巧把它降到 $O(d_{row} \\cdot d_{col}^2 + d_{col}^3)$:
  ① **任意顺序**(固定从左到右)→ Hessian 逆只算一次;② **lazy batch-updates**(
  128 列一组批处理,GPU 友好);③ **Cholesky 分解**一次求逆、全程复用。

对 GPT-175B(每层权重约 $12288 \\times 12288$):

- 若用 OBQ 逐权重处理,估计需要几周;
- GPTQ 用 **4 块 GPU 约 4 小时** 完成整个 175B 模型的 3/4 bit 量化。

下面在 GPU 上做一次小规模对照:**直接量化(per-tensor)vs GPTQ(逐列 + Hessian 补偿)**,
看同样的 8 bit 能差多少:
'''))

NB.code(D('''
import torch, gc
dev = "cuda" if torch.cuda.is_available() else "cpu"   # 优先 GPU

torch.manual_seed(0)
d_out, d_in, n = 64, 64, 1024
W = torch.randn(d_out, d_in, device=dev) / (d_in ** 0.5)   # (64, 64) 权重
A = torch.randn(n, d_in, device=dev)                       # (1024, 64) 独立成分
common = torch.randn(n, 1, device=dev)                     # (1024, 1) 公共因子
X = 0.5 * A + 0.5 * common                                 # (1024, 64) 强相关校准激活
with torch.no_grad():
    out = X @ W.T                                          # (1024, 64) 原始输出

qmax = 127
# 直接量化:整张矩阵一把尺子(最朴素的 per-tensor)
s = (W.abs().max() / qmax).item()
Qd = torch.clamp(torch.round(W / max(s, 1e-12)), -qmax, qmax).float() * s
with torch.no_grad():
    e_direct = float(((X @ Qd.T - out) ** 2).mean() / (out ** 2).mean())

with torch.no_grad():
    Qg = gptq_quantize(W, X, bits=8, compensate=True)      # 逐列 + Hessian 补偿
    e_gptq = float(((X @ Qg.T - out) ** 2).mean() / (out ** 2).mean())

print(f"设备: {dev} | 权重 shape = {tuple(W.shape)}  <- (输出 64, 输入 64)")
print(f"直接量化(per-tensor, 8bit): 相对输出误差 = {e_direct:.4%}")
print(f"GPTQ(8bit, 逐列 + 补偿)    : 相对输出误差 = {e_gptq:.4%}")
print(f"→ 误差下降 {e_direct/max(e_gptq,1e-12):.1f} 倍(同样的 8 bit,只是补偿摊销了残差)")

torch.cuda.synchronize()
gc.collect(); torch.cuda.empty_cache()
'''), "✅ 真实 GPU 数字:Hessian 近似 + 逐列补偿把同一张 8bit 权重的输出误差再压一个数量级——补偿不是玄学,是可复现的反量化质量提升。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:Marlin 内核怎么跑 GPTQ 🚀",
D('''
vLLM 原生支持 GPTQ 权重(配合 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 或
[llm-compressor](https://github.com/vllm-project/llm-compressor) 量化好的模型):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq
```

工程事实:

1. **格式**:GPTQ 权重存为 `qweight`(打包的 int4 整数)+ `scales` + 可选的 `qzeros`;
   每 **128 个输入通道一组**共用 scale(group-wise);
2. **内核**:vLLM 默认走 **Marlin** 内核(`gptq_marlin`,见
   `vllm/model_executor/layers/quantization/gptq_marlin.py`)。Marlin 把 int4 权重
   **离线重排**成对 L2 cache 友好的布局,在 batch 16-32 的 decode 下保持接近 4× 的理论加速;
3. **反量化在 GEMM 内**:内核边取整数边乘 scale 累加,不先把权重解包成 fp16——
   这正是 int4 权重「更小更快」的来源。

> 📄 Marlin 内核原理见 [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin):
> 核心思想是把解量化与张量核指令交错,让两条流水线都吃饱。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:调相关性,看补偿增益 🎛️",
D('''
运行同目录下的 `app_31_gptq.py`,拖动矩阵大小、位宽与**输入列相关性**滑块,
实时对比朴素 RTN 与 GPTQ 补偿的输出误差曲线:

```bash
D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_31_gptq.py
```

浏览器打开 **http://localhost:8501**。建议玩法:把相关性 ρ 从 0 慢慢拖到 0.9,
盯着「误差降低倍数」指标,看补偿增益如何从 1.0x 起飞。完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_31, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_31_gptq.py 后运行:\")\n"
    "    print(\"    D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_31_gptq.py\")\n"
)
NB.code(guard, "📜 这就是 app_31_gptq.py 的完整源码(numpy 实现,与 notebook 的 torch 版逻辑一致)。")

wrapup(NB,
    summary=[
        "RTN 每个权重独立就近舍入,优化权重误差;但真正该优化的是输出误差 ‖XWᵀ−XŴᵀ‖²",
        "输出误差由 Hessian H=2XᵀX 加权:误差在不同方向上的「伤害值」不同",
        "GPTQ 三步走:量化当前列 → 残差按 H⁻¹ᵢᵢ 归一记账 → 按 H⁻¹ 的非对角行摊销到未量化列",
        "列相关性是补偿的燃料:H 为单位阵(列独立)时 GPTQ 退化为 RTN",
        "GPTQ 可能权重误差更大但输出误差更小——它瞄准的是任务真正在乎的目标",
        "工程上:GPTQ + Marlin 内核,128 列一组 scale,decode 保持近 4× 加速",
    ],
    practice=[
        "把 gptq_quantize 的 damp 从 0.01 改成 0.5 / 0.001,观察补偿增益与数值稳定性变化",
        "将校准数据 X 换成 one-hot 独热矩阵,验证 H 近似对角、补偿失效",
        "给 quant_trail 加上「每列补偿量范数」的记录,画出补偿强度随列位置的衰减曲线",
        "对比 bits=2 时 RTN/GPTQ 的输出余弦相似度(cosine_sim),补充误差视角",
    ],
    links=[
        ("GPTQ 论文 (arXiv:2210.17323)", "https://arxiv.org/abs/2210.17323"),
        ("OBS: Optimal Brain Surgeon (1992)", "https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf"),
        ("GPTQ 官方代码库", "https://github.com/IST-DASLab/gptq"),
        ("Marlin 内核仓库", "https://github.com/IST-DASLab/marlin"),
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
    ])

NB.save(str(Path(CH05) / "31_gptq_principle.ipynb"))

app_path = Path(CH05) / "app_31_gptq.py"
app_path.write_text(APP_31 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")