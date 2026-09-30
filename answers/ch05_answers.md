# 第 5 章 · 量化 — 参考答案（ch05，第 28–34 课）

> **参考答案 · 建议先自己动手再看。** 本文件与仓库 `exercises/ch05/` 下 28–34 号 notebook 一一对应。
> 代码片段假设与你 notebook 中**已运行过的定义一致**才可直接独立运行（会标注依赖）；其余片段自带完整定义。
> 术语保留英文，答案用中文。

---

## 第 28 课 · 量化基础：位宽、scale 与 zero-point

> 对应 `exercises/ch05/28_quantization_basics.ipynb`。依赖：`symm_quantize` / `symm_dequant` / `asymm_quantize` / `asymm_dequant`（作者在 notebook 第 3、4 节定义的函数）。

### 练习 1：bits=4 量化 1000 个均匀点，看误差分布直方图
**思路**：把量化位宽压到 4 会让格子变粗，误差分布应集中在 `[0, s/2]` 内且无超界值。

```python
import torch, matplotlib.pyplot as plt
x = torch.linspace(-1, 1, 1000)
q, s = symm_quantize(x, bits=4)            # Qmax = 2^3 - 1 = 7
x_hat = symm_dequant(q, s)
plt.hist((x - x_hat).abs().numpy(), bins=50)
plt.title("4-bit 量化误差分布"); plt.show()
print(f"max|err| = {(x-x_hat).abs().max():.4f}   s = {float(s):.5f}   s/2 = {float(s)/2:.5f}")
```
**预期结果**：`s = max|x|/Qmax = 1/7 ≈ 0.1429`，误差直方图呈正偏，峰值贴近 `s/2 ≈ 0.071`，**没有任何误差超过 s/2**（这正是均匀量化的误差上界）。

### 练习 2：推导对称量化的最大误差上界是 s/2
**思路**：`round` 是"四舍五入到最近整数"。
**推导**：`q = round(x/s)`，因此 `x/s` 与 `q` 的差不超过 `0.5`，即 `|x/s - q| ≤ 0.5`。重建值 `x̂ = q·s`，故 `|x - x̂| = |x/s - q|·s ≤ 0.5·s = s/2`。
**结论**：只要用 round-to-nearest，单元素重建误差恒 ≤ s/2，与数值无关——这是对称量化的全部秘密。

### 练习 3：对全负向量做非对称量化，看 zero-point 落到哪
**思路**：非对称量化把 `[rmin, rmax]` 映射到 `[0, 255]`；全负数据浮点 0 对应上界，`zp` 应落在最大值 255 附近。

```python
x_neg = torch.linspace(-2, 0, 100)          # 全负，模拟 pre-ReLU 激活
q, s, zp = asymm_quantize(x_neg, bits=8)    # zp 应为 int32 标量
print(f"zero-point = {int(zp)}   scale = {float(s):.4f}")
```
**预期结果**：`zp ≈ 255`（≈qmax）。因为 `rmax ≈ 0`，浮点 0 精确映到整数 255 附近，激活全负时零点被推到上界。

### 练习 4：int8 量化同一个 0.5，对称 vs 非对称，为何非对称更准
**思路**：对称把正负半轴各分一半格子；数据全正时它在负半轴浪费了一半刻度，非对称用满全部 256 格。

```python
q_s, s_s = symm_quantize(torch.tensor([-0.5, 0.5]), 8)
q_a, s_a, z_a = asymm_quantize(torch.tensor([-0.5, 0.5]), 8)
print(f"对称 scale = {float(s_s):.5f} (0.5/127)；非对称 scale = {float(s_a):.5f} (0.5/255)")
```
**分析**：对数据范围 `[-0.5, 0.5]`，对称格子宽 `0.5/127 ≈ 0.00394`，非对称格子宽 `0.5/255 ≈ 0.00196`，**非对称格子约细一半**。本例 0.5 恰为边界的最大值，两者都精确落到整数端点；换成任意非格子值（如 0.37）时，格子越细 → 舍入误差越小 → 非对称更准。工程上因此"权重（零对称）用对称、激活（ReLU 全正）用非对称"。

---

## 第 29 课 · 对称 vs 非对称量化：per-tensor 与 per-channel

> 对应 `exercises/ch05/29_sym_asym_quant.ipynb`。依赖：`symm_quantize` / `asymm_quantize` / `symm_dequant` / `asymm_dequant`。

### 练习 1：bits 8 时对称/非对称差距如何缩小
**思路**：位数升高 → 格子变细 → 两者都更准；对称"浪费半格"的相对代价随格子变细而减小。

```python
x = torch.rand(1000) * 6.0 + 0.5            # 右偏全正分布
for b in [4, 8]:
    qs, ss = symm_quantize(x, b); es = float(((x - symm_dequant(qs, ss))**2).mean())
    qa, sa, za = asymm_quantize(x, b); ea = float(((x - asymm_dequant(qa, sa, za))**2).mean())
    print(f"{b}bit 对称MSE={es:.4e} 非对称MSE={ea:.4e} 比值={es/max(ea,1e-30):.1f}")
```
**预期结果**：4 bit 时非对称明显更优（比值 > 1）；**升到 8 bit 两者 MSE 都掉一两个数量级，比值趋近 1**——精度足够时零点是否有偏不再重要。

### 练习 2：per-channel 非对称量化，测右偏矩阵
**思路**：把第 29 课的 `symm_quantize_channel` 平移成非对称版，每输出行独立求 `[rmin,rmax]→[0,255]`。

```python
def asymm_quantize_channel(x, bits=8):
    qmin, qmax = 0, 2**bits - 1
    rmin = x.min(dim=1, keepdim=True).values
    rmax = x.max(dim=1, keepdim=True).values
    scale = ((rmax - rmin) / (qmax - qmin)).clamp_min(1e-12)
    zp = torch.round(qmin - rmin/scale).clamp(qmin, qmax).to(torch.int32)
    q  = torch.round(x/scale + zp.float()).clamp(qmin, qmax).to(torch.int32)
    return q, scale, zp

W = torch.rand(16, 64) * torch.logspace(0, 2, 16).unsqueeze(1)  # 右偏、各行幅度差
q, s, z = asymm_quantize_channel(W)
Wh = (q.float() - z.float()) * s
print(f"per-channel 非对称 MSE = {((W-Wh)**2).mean():.4e}  scale shape = {tuple(s.shape)}")
```
**预期结果**：每行一把独立的非对称尺子（scale 形状 `(16,1)`），对右偏且各行幅度差大的矩阵，误差明显低于 per-tensor 非对称。

### 练习 3：构造双峰分布，判断哪种量化更适合
**思路**：双峰（两个高斯混合）常常关于 0 对称，最大值/最小值幅度相近 → 负半轴被充分使用 → 对称量化已够好。

```python
from torch.distributions import Normal, Categorical
mix = Categorical(torch.tensor([0.5, 0.5]))
comps = torch.stack([Normal(2,0.5), Normal(-2,0.5)]).sample((4000,))
x = comps.gather(1, mix.sample((4000,))[:, None]).squeeze()
qs, ss = symm_quantize(x, 8); es = ((x - symm_dequant(qs, ss))**2).mean()
qa, sa, za = asymm_quantize(x, 8); ea = ((x - asymm_dequant(qa, sa, za))**2).mean()
print(f"对称MSE={float(es):.4e}  非对称MSE={float(ea):.4e}")
```
**预期结果**：两峰对称、幅度相等时对称与非对称 MSE 相当——**说明分布关于 0 对称时不必引入 zero-point**；若两峰不对称（正负幅度差大），非对称才更优。

### 练习 4：统计量化后不同取值的个数 ≤ 2^b
**思路**：量化整数必然落在 `[-Qmax, Qmax]` 的离散格点上，用 `unique()` 直接数。

```python
q, s = symm_quantize(torch.randn(100000), 8)   # 大样本
n = q.unique().numel()
print(f"不同取值数 = {n}   ≤ 2^b = {2**8} ？ {n <= 2**8}")
```
**预期结果**：`True`——对称 int8 最多 255 个不同整数（`[-127,127]`），必然 `≤ 256 = 2^b`。

---

## 第 30 课 · 量化误差分析：MSE、SNR 与校准

> 对应 `exercises/ch05/30_quantization_error.ipynb`。依赖：`mse` / `max_abs_err` / `cosine_sim` / `snr_db` / `symm_quantize` / `symm_dequant`，以及本课的 `w` 权重张量。

### 练习 1：bits=8 重画 U 型曲线，最优分位是否更靠近 100%
**思路**：位数升高 → 舍入误差变小 → 钳位误差相对占比上升 → 敢于牺牲更少（最优分位更靠近 100%）。

```python
import numpy as np
pcts = np.linspace(95, 100, 41)
for b in [4, 8]:
    qmax = 2**(b-1) - 1
    m, best = 1e30, None
    for p in pcts:
        c = float(torch.quantile(w.float().abs(), p/100))
        s = max(c, 1e-12)/qmax
        Wh = torch.clamp(torch.round(w.float()/s), -qmax, qmax).float()*s
        if float(((w.float()-Wh)**2).mean()) < m:
            m = float(((w.float()-Wh)**2).mean()); best = p
    print(f"{b}bit 最优分位 ≈ {best:.1f}%")
```
**预期结果**：8 bit 的最优分位比 4 bit 的更高（更接近 100%）。因为位宽高时格子很细，一点点钳位损失就比舍入损失"更贵"。

### 练习 2：写 auto_calibrate，网格搜索使 MSE 最小的钳位分位
**思路**：对分位百分比做一维网格扫描，对每个分位 `c` 做"钳位+量化"，返回使 MSE 最小的 `c`。

```python
def auto_calibrate(x, bits=8):
    qmax = 2**(bits-1) - 1
    best_c, best_m = None, float("inf")
    for c in [float(torch.quantile(x.abs().double(), t)) for t in np.linspace(0.5, 1.0, 300)]:
        s = max(c, 1e-12)/qmax
        Wh = torch.clamp(torch.round(x/s), -qmax, qmax).float()*s
        mse = float(((x - Wh)**2).mean())
        if mse < best_m: best_m, best_c = mse, c
    return best_c, best_m
c, m = auto_calibrate(w.float())
print(f"最优钳位阈值 c={c:.4f}, 对应 MSE={m:.3e}")
```
**预期结果**：返回一个中等偏大的 `c`（牺牲少量离群值换整体更细的格子），MSE 显著低于 max 校准——对应"总误差 = 钳位误差 + 舍入误差"的 U 型谷底。

### 练习 3：用 torch.quantile 取 99.99/99.9/99.5 分位，统计被钳样本比例
**思路**：`quantile(p)` 返回的分位阈值恰是"比它大的样本约 1-p"，直接数超阈值个数即钳掉比例。

```python
x = w.float().abs().reshape(-1)
for p in [0.9999, 0.999, 0.995]:
    c = float(torch.quantile(x, p))
    clipped = (x > c).sum().item()
    print(f"{p*100:.2f}% 分位: 钳掉 {clipped} 个, 占 {clipped/len(x):.4%}")
```
**预期结果**：99.99% 钳掉约 0.01% 的样本，99.9% 约 0.1%，99.5% 约 0.5%——与"1−p"一致，构成经典的偏差/方差权衡。

### 练习 4：为什么校准集要与真实推理数据同分布
**思路**：分位数是**经验统计量**。
**结论**：`scale`/钳位阈值是从校准集样本统计出来的。若校准集分布与真实推理数据不一致，统计出的分位数是"错误的经验值"，会对量化误差/容忍度给出错误估计——可能高估精度（实际掉点）或过度保守（白损失显存）。这就是 GPTQ/AWQ 等必须用贴近真实分布的通用语料（如 C4）做校准的原因。

---

## 第 31 课 · GPTQ 思想：误差补偿

> 对应 `exercises/ch05/31_gptq_principle.ipynb`。依赖：`gptq_quantize`（作者 notebook 定义的逐列补偿实现），以及 `W` / `X` 校准数据。

### 练习 1：damp 改为 0.5 / 0.001，观察补偿增益与稳定性
**思路**：damp 是加到 Hessian 对角线上的阻尼项——太小则 `H⁻¹` 接近奇异导致数值不稳，太大则掩盖真实 Hessian、让 `H⁻¹` 退化成单位阵、补偿失效。

```python
for damp in [0.001, 0.01, 0.5]:
    Q = gptq_quantize(W, X, bits=4, damp=damp, compensate=True)
    e = float(((X @ Q.T - X @ W.T)**2).mean())
    print(f"damp={damp:>7}: 输出MSE = {e:.4e}")
```
**预期结果**：`damp=0.01`（合理值）时输出 MSE 最低、补偿增益最大；`damp=0.5` 时 H 被过阻尼、输出 MSE 明显升高（接近 RTN）；`damp=0.001` 时数值稳定性变差甚至产生较大误差。量级上 4bit 下 GPTQ 输出 MSE 可比 RTN 低数倍到数量级。

### 练习 2：换 one-hot 校准 X，验证 H 近似对角、补偿失效
**思路**：one-hot 的各列正交 → `XᵀX` 非对角元 ≈ 0 → Hessian 对角 → 每列量化误差没有"邻居列"可摊 → 补偿无事可做。

```python
X_ohe = torch.eye(d_in)                     # one-hot：(d_in, d_in)
for name, Xc in [("one-hot", X_ohe), ("强相关", X)]:
    H = 2.0 * Xc.T @ Xc
    offdiag = (H - torch.diag(torch.diagonal(H))).abs().mean()
    Q = gptq_quantize(W, Xc, bits=4, damp=0.01, compensate=True)
    e = float(((Xc @ Q.T - Xc @ W.T)**2).mean())
    print(f"{name}: 非对角元均值 = {float(offdiag):.3e}, GPTQ输出MSE ≈ {e:.4e}")
```
**预期结果**：one-hot 的非对角元均值 ≈ 0，GPTQ 与 RTN 输出误差几乎相同（补偿失效）；强相关校准数据的非对角元大，补偿显著有效。**列相关性（非对角元）就是补偿收益的来源**。

### 练习 3：记录每列补偿量范数，画衰减曲线
**思路**：把 `gptq_quantize` 循环里每次 `torch.outer(err, L_factor[i, ...])` 的范数记录下来。

```python
comp_norms = []
W_c = W.clone(); Lf = torch.linalg.cholesky(torch.linalg.inv(2.0*X.T@X + 0.01*torch.eye(d_in)), upper=True)
for i in range(d_in):
    w = W_c[:, i]; s = max(float(w.abs().max()), 1e-12)/qmax
    W_c[:, i] = torch.round(w/s).clamp(-qmax, qmax)*s
    if i+1 < d_in:
        err = (w - W_c[:, i]) / Lf[i, i]
        delta = torch.outer(err, Lf[i, i+1:])
        comp_norms.append(delta.norm().item())
        W_c[:, i+1:] -= delta
plt.plot(comp_norms); plt.ylabel("每列补偿范数"); plt.xlabel("列序号"); plt.show()
```
**预期结果**：补偿强度随列序号**总体衰减**——前几列把误差摊给后面大量未量化列，之后每列可摊的空间越来越小，曲线大致单调递减。说明 GPTQ 的补偿在贪心顺序里"先下手为强"。

### 练习 4：bits=2 时 RTN vs GPTQ 的输出余弦相似度
**思路**：bits=2 精度极差，从"接近"走向"扭曲方向"，用余弦相似度看方向漂移；依赖第 30 课 `cosine_sim`。

```python
Q_rtn = gptq_quantize(W, X, bits=2, compensate=False)
Q_gptq= gptq_quantize(W, X, bits=2, compensate=True)
out = X @ W.T
cos_rtn = cosine_sim(out, X @ Q_rtn.T)
cos_gptq= cosine_sim(out, X @ Q_gptq.T)
print(f"bits=2 输出余弦: RTN={cos_rtn:.4f}  GPTQ={cos_gptq:.4f}  增益={cos_gptq-cos_rtn:+.4f}")
```
**预期结果**：bits=2 下两者余弦都低于高 bit 情况，但 **GPTQ 的余弦明显高于 RTN**（补偿把误差方向拉回），低位宽下补偿收益相对更显著。

---

## 第 32 课 · AWQ 思想：激活感知的权重保护

> 对应 `exercises/ch05/32_awq_principle.ipynb`。依赖：`awq_quantize` / `rel_err`，以及 `W` / `act_scale` / `X` 数据集。

### 练习 1：激活换成正态（无偏斜），验证 AWQ 增益趋近 1
**思路**：AWQ 的保护只对"显著通道"有意义；各通道激活幅度相同时 `s≈1`，α 不起作用，增益应趋近 1。

```python
act_norm = torch.relu(torch.randn(d_in)) + 0.1         # 幅度大致均匀、无离群
for alpha in [0.0, 0.5, 1.0]:
    Wq = awq_quantize(W, act_norm, bits=3, alpha=alpha)
    print(f"alpha={alpha}: rel_err = {rel_err(Wq):.4%}")
print("对照(LogNormal 重偏斜) alpha=0.5:", f"{rel_err(awq_quantize(W, act_scale, 3, 0.5)):.4%}")
```
**预期结果**：无偏斜时三个 α 的 rel_err 几乎一致（增益 ≈ 1）；而重偏斜时 α=0.5 相对 α=0 有数倍误差降低——**保护对象的"偏心度"决定 AWQ 收益**。

### 练习 2：α 扫描加入 bits=2/3/4 三条曲线，看最优 α 移动
**思路**：位宽越低量化越粗糙，显著通道越需要保护 → 最优 α 往往更大。

```python
import numpy as np
alphas = np.linspace(0, 1, 11)
for bits in [2, 3, 4]:
    errs = [rel_err(awq_quantize(W, act_scale, bits=bits, alpha=a)) for a in alphas]
    best = alphas[int(np.argmin(errs))]
    print(f"bits={bits}: 最优 α ≈ {best:.1f}  (min rel_err={min(errs):.4%})")
```
**预期结果**：一般来说位宽越低（bits=2），最优 α 越大/保护越必要；高位宽（bits=4）时 α 的影响变小、最优 α 更低。保护强度与量化粗糙度负相关。

### 练习 3：per-channel scale 的显存开销统计（s 要不要量化存）
**思路**：AWQ 的 `s` 是每输入通道一个浮点（`.shape=(d_in,)`），相对权重 `(d_out, d_in)` 开销极小。

```python
d_out, d_in = W.shape
s_bytes = d_in * 4                       # 每通道一个 fp32 scale
W_bytes = d_out * d_in * 2               # 假设 fp16 权重
print(f"scale 显存 = {s_bytes/1024:.1f} KB, 占比 = {s_bytes/(s_bytes+W_bytes):.3%}")
print(f"每通道 scale 相对权重开销 ≈ 4/d_out = {4/d_out:.3%}")
```
**分析**：scale 体积 = `d_in×4B`，权重体积 = `d_out×d_in×2B`，占比 ≈ `4/d_out`，通常 <1%（`d_out` 很大），可忽略。**但 s 若用 int 再量化会引入二次误差**——工程上 s 常以 fp16/fp32 独立存储甚至前向即时算出，常规做法是**不量化 scale**。

### 练习 4：对比"冻结 Top1% 通道(mixed precision)"与 AWQ 的误差与硬件代价
**思路**：混合精度留高精度通道会让一个矩阵乘退化成"快速低 bit 为主、慢速高 bit 为辅"的分段运算。

```python
# mixed precision：按激活幅度选 Top1% 通道不量化(fp32 原样)，其余 3bit 量化
top = torch.argsort(act_scale, descending=True)[: max(1, int(0.01*d_in))]
W_mix = awq_quantize(W, act_scale, bits=3, alpha=0.0).clone()
W_mix[:, top] = W[:, top]                        # 冻结：离群通道保留高精度
err_mix = rel_err(W_mix); err_awq = rel_err(awq_quantize(W, act_scale, 3, 0.5))
print(f"输出相对误差: mixed-precision={err_mix:.4%}  AWQ={err_awq:.4%}")
```
**结论**：mixed-precision 的误差通常比 AWQ 更低，但硬件极不友好——GPU tensor core 无法对"大部分 int、少部分 fp32"的统一 GEMM 提速，需要 split 成两次独立乘法再合并，吞吐大幅下降；**AWQ 全员量化、仅缩放，可走标准量化 GEMM（硬件友好），用更小代价获得接近的精度**。

---

## 第 33 课 · FP8 与 KV Cache 量化

> 对应 `exercises/ch05/33_fp8_kv_quant.ipynb`。依赖：`fp8_decode` / `fp8_all_values` / `kv_bytes_per_token`。

### 练习 1：为 E4M3 加 NaN 判定，统计 256 码里 NaN/Inf 各几个
**思路**：E4M3FN 中指数全 1 且尾数全 1（`1111 111`）编码为 NaN，且此格式**没有 Inf**。

```python
n_nan = sum(1 for c in range(256)
            if ((c>>3) & 0xF) == 0xF and (c & 0x7) == 0b111)
print(f"E4M3 256 个编码: NaN = {n_nan} 个, Inf = 0 个(E4M3FN 无 Inf)")
```
**预期结果**：正负各 1 个共 2 个 NaN；**Inf 为 0**（区分 E5M2：尾数全 0 指数全 1 才是 ±Inf）。

### 练习 2：用自己的模型算 100 万 token 的 KV 账
**思路**：改 `kv_bytes_per_token` 的入参为你模型 `config.json` 里的 `num_layers` / `num_key_value_heads`（如 LLaMA-3-70B：`L=80, h_kv=8`）。

```python
def kv_per_token(L, h_kv, d, b=2): return 2*L*h_kv*d*b
L, h_kv, d = 80, 8, 128            # 替换成你自己的模型配置
bpt = kv_per_token(L, h_kv, d, 2)
print(f"每 token {bpt/1024:.0f} KB fp16 → 100万 token 共 {bpt*1_000_000/1024**3:.1f} GB")
print(f"若 FP8(1B): {bpt*1_000_000/2/1024**3:.1f} GB（减半）")
```
**预期结果**：KV 随 `layers × key_value_heads` 线性增长；100 万 token 在 70B 级模型可达数十 GB 量级，FP8 直接减半。

### 练习 3：t 分布重尾数据做 E4M3，观察超范围值如何被钳位
**思路**：E4M3 最大有限值 = `448`，重尾分布大量绝对值 >448 的值会被钳位到 ±448。

```python
import torch
data = torch.distributions.StudentT(3).sample((100000,)) * 200.0   # 重尾、涌现大量 >448
MAX = 448.0
clamped = torch.clamp(data, -MAX, MAX)
n_over = (data.abs() > MAX).sum().item()
print(f"超过 ±448 的值: {n_over} 个,占 {n_over/len(data):.2%}; 钳位误差MSE = {((data-clamped)**2).mean():.3e}")
```
**预期结果**：重尾分布产生一批绝对值超过 448 的离群值，全部被锁到 ±448；这些值的钳位误差成为 FP8 的主要误差来源。这解释了 FP8 适合长尾不重的激活/权重，遇到重尾分布需要 KP 拆分或更高位宽。

### 练习 4：GQA 32→8 KV 头省多少，再叠加 FP8 省多少
**思路**：KV 显存 ∝ `kv_heads × dtype_bytes`，两个因子独立相乘。

```python
# GQA:KV 头 32→8 → 除以 4
print(f"GQA 32头→8头: KV 显存降至 1/4 (省 75%)")
# FP8:每元素 2B→1B → 除以 2
print(f"再叠加 FP8: 再减半 → 总计 1/4 × 1/2 = 1/8,省 87.5%")
print(f"换算：32头 fp16 为基准 → 8头 fp8 后 KV = 1/8 × 基准")
```
**预期结果**：GQA 单独省 75%，叠加 FP8 后总计省到 87.5%（原值的 1/8）。工程上两者常常同时启用。

---

## 第 34 课 · 综合实验：精度-速度-显存的三角权衡

> 对应 `exercises/ch05/34_quant_tradeoff.ipynb`。依赖：`symm_quantize` / `symm_dequant` / `symm_quantize_channel` / `symm_dequant_channel` / `simulate_gemm` / `rel_mse` / `bench`，以及 `X` / `W`，基准 `y_ref = X @ W.T`。

### 练习 1：bits 改成 4，重测 per-tensor/per-channel 误差，看差距拉大
**思路**：`simulate_gemm` 内部把 bits 写死成 8，需自写一个带 bit 参数的版本。

```python
def sim(X, W, mode, bits):
    qx, sx = symm_quantize(X, bits); Xh = symm_dequant(qx, sx)
    if mode == "channel":
        qw, sw = symm_quantize_channel(W, bits); return Xh @ symm_dequant_channel(qw, sw).T
    qw, sw = symm_quantize(W, bits); return Xh @ symm_dequant(qw, sw).T
y_ref = X @ W.T
for b in [8, 4]:
    for mode in ["per-tensor", "channel"]:
        e = ((sim(X, W, mode, b) - y_ref)**2).mean() / (y_ref**2).mean()
        print(f"{b}bit {mode}: 相对输出误差 = {float(e):.4%}")
```
**预期结果**：8 bit 两者都 <0.05%；4 bit 误差放大一个量级以上，且 **per-channel 相对 per-tensor 的优势从 8bit 的数倍扩大到 4bit 的更大倍数**——低 bit 下"每行一把尺子"更关键。

### 练习 2：bench 加 warmup=5、iters=50，对比计时波动
**思路**：`bench(fn, warmup, iters)` 已支持两参数，只改调用参数即可。

```python
t1 = bench(你的被测fn, warmup=2,  iters=10)
t2 = bench(你的被测fn, warmup=5,  iters=50)
print(f"默认(w2/i10) = {t1:.4f} ms  稳定(w5/i50) = {t2:.4f} ms")
print(f"两档相对偏差 = {abs(t1-t2)/t2*100:.2f}%")
```
**预期结果**：iters=50 + warmup=5 时均值更稳、首调抖动被彻底排除；多次重跑该档结果波动明显小于默认档。`bench` 已定义，仅需换参数即可验证。

### 练习 3：帕累托图上加 int2 与 fp4 假想点，推断位置
**思路**：帕累托前沿是"越往左上越好（显存小+精度高）"；int2 更省显存但精度暴跌，fp4 因硬件原生在相近显存下精度更高。

```python
# 假想点 (相对fp16显存, 精度/10 分)
pts = {"fp16":(1.0, 10.0), "fp8":(0.5, 9.5), "int4-GPTQ":(0.25, 8.8),
       "int2假想":(0.125, 5.0), "fp4假想":(0.125, 7.5)}
for n,(mb,acc) in pts.items():
    print(f"{n:<12}: 相对显存={mb:.3f}, 精度={acc:.1f}/10")
```
**推断**：`int2` 把前沿继续往左下推（显存 0.125），但精度塌方（约 5 分）——"再省显存就得付精度税"的铁证；`fp4` 在同样 0.125 显存下因 Blackwell 原生支持，精度明显高于 int2（可到 7.5 分），构成一个更优的新前沿点。它们在图上应落在既有前沿的左下延伸带上。

### 练习 4：查 GPU 规格，算 FP8 相对 FP16 的理论峰值比
**思路**：比率 = GPU 规格表里 FP8 张量核心峰值 TFLOPS ÷ FP16/BF16 峰值。
**参考量级**（NVIDIA 官方规格）：A100 `BF16≈312`、无 FP8；H100 `FP8≈989`、`BF16≈989`（≈1×）；RTX 40 系 `FP8 ≈ FP16 × 2`（如 RTX 4090 FP16≈660、FP8≈1301，约 2×）；B 系/GB 系列亦有 FP8 支持。
**结论**：对这类原生支持 FP8 的卡，比率约为 **2**，与文中"FP8 几乎无损的约 2 倍吞吐/显存收益"一致；A100 无 FP8 内核故该比值不适用。以你自己手头的 GPU 型号查对应 TFLOPS 相除即可对照。

---