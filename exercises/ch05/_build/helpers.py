# -*- coding: utf-8 -*-
"""ch05 生成公共组件:Notebook 构建 + 各课共享的量化函数源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH05 = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch05"
CHAPTER = "第 5 章 · 量化"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 共享源码
INT8_SYMM = D('''
import torch

def symm_quantize(x, bits=8):
    """对称量化:int8 里正负各一半,zero-point 固定为 0。
    scale = max(|x|) / Qmax,量化整数落在 [-Qmax, Qmax]。
    返回 (量化整数 q, 量化步长 scale)。"""
    qmax = 2 ** (bits - 1) - 1          # int8 → 127
    scale = x.abs().max() / qmax
    scale = scale.clamp_min(1e-12)      # 防止全零张量除零
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int32)
    return q, scale

def symm_dequant(q, scale):
    """反量化:整数 × scale,近似还原浮点值"""
    return q.float() * scale
''')

ASYMM_QUANT = D('''
def asymm_quantize(x, bits=8):
    """非对称量化:引入 zero-point,让浮点 0 精确映射到整数 zp。
    适合分布有偏的场景(如 ReLU 激活全为正)。"""
    qmin, qmax = 0, 2 ** bits - 1       # uint8 → [0, 255]
    rmin, rmax = x.min(), x.max()
    scale = ((rmax - rmin) / (qmax - qmin)).clamp_min(1e-12)
    zp = torch.round(qmin - rmin / scale).clamp(qmin, qmax).to(torch.int32)
    q = torch.round(x / scale + zp.float()).clamp(qmin, qmax).to(torch.int32)
    return q, scale, zp

def asymm_dequant(q, scale, zp):
    return (q.float() - zp.float()) * scale
''')

ERROR_METRICS = D('''
def mse(a, b):
    """均方误差 MSE"""
    return float(((a - b) ** 2).mean())

def max_abs_err(a, b):
    """最大绝对误差"""
    return float((a - b).abs().max())

def cosine_sim(a, b):
    """余弦相似度:衡量向量方向(输出漂移)是否保持一致"""
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
''')

PER_CHANNEL = D('''
def symm_quantize_channel(x, bits=8):
    """按输出通道对称量化:权重矩阵 (out_features, in_features) 沿 dim=1 每行求 scale。
    x : (out_features, in_features) 权重矩阵,每个输出通道(dim=0 的一行)各配一把尺子。
    返回 (量化整数 q, 每通道 scale,shape=(out_features,1))。"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=1, keepdim=True) / qmax   # 每输出行(dim=1)一把尺子
    scale = scale.clamp_min(1e-12)
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int32)
    return q, scale

def symm_dequant_channel(q, scale):
    return q.float() * scale
''')

GPTQ_CORE = D('''
import torch

def gptq_quantize(W, X, bits=4, damp=0.01, compensate=True):
    """简化版 GPTQ:逐列量化 + 误差补偿。
    Hessian 近似 H = 2·X^T·X,补偿用 H^{-1} 的 Cholesky 因子(论文 Algorithm 1 的单批版)。
    W : (d_out, d_in) 权重;X : (n, d_in) 校准激活。返回量化后的 W_hat(浮点重建)。
    compensate=False 时退化为逐列 RTN(round-to-nearest)。"""
    d = W.shape[1]
    H = 2.0 * X.T @ X                                  # (d_in, d_in) Hessian 近似
    H += damp * torch.diagonal(H).mean() * torch.eye(d, device=W.device)  # 阻尼项,防止奇异
    # H^{-1} 的 Cholesky 上三角因子 L:一次分解,后续每列把“求逆-向量”运算换成 L 的
    # 稀疏行运算,免去逐列反复求逆——这是 GPTQ 相对 OBQ 提速的关键工程技巧之一。
    L_factor = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)
    W = W.clone()
    Q = torch.zeros_like(W)
    qmax = 2 ** (bits - 1) - 1
    for i in range(d):
        w = W[:, i]                                    # 当前列(已被之前列的补偿更新过)
        s = float(w.abs().max()) / qmax
        s = max(s, 1e-12)
        q = torch.round(w / s).clamp(-qmax, qmax)      # 量化到 int
        Q[:, i] = q * s
        if compensate and i + 1 < d:
            err = (w - Q[:, i]) / L_factor[i, i]       # 残差按 Cholesky 对角缩放
            W[:, i + 1:] -= torch.outer(err, L_factor[i, i + 1:])  # 摊到未量化列
    return Q
''')

AWQ_CORE = D('''
import torch

def awq_quantize(W, act_scale, bits=4, alpha=0.5):
    """简化版 AWQ:给“显著通道”(激活幅度大)的权重乘上缩放 s 再量化,推理时再除回。
    W : (d_out, d_in) 权重;act_scale : (d_in,) 每个输入通道的平均激活幅度。
    alpha : 保护强度,alpha=0 即无保护的普通 RTN 量化。返回重建权重。"""
    s = (act_scale / act_scale.mean()) ** alpha       # 激活越大 → 该通道权重被撑得越大
    s = s / s.min()                                   # 归一化:最小的通道不缩放
    qmax = 2 ** (bits - 1) - 1
    Ws = W * s                                        # 预缩放:显著通道“撑大”后离格子更近
    scale = Ws.abs().amax(dim=1, keepdim=True) / qmax # 每个输出通道一把尺子(行内共享,缩放才有意义)
    scale = scale.clamp_min(1e-12)
    q = torch.round(Ws / scale).clamp(-qmax, qmax)
    return (q * scale) / s                            # 反量化并除回缩放
''')

FP8_PARSE = D('''
import math

def fp8_decode(code, exp_bits, mant_bits, exp_bias):
    """把 FP8 的 8 位整数编码还原成浮点值(含 subnormal 与特殊指数模式)。
    code : 0~255 的整数,按 sign|exp|mant 三段切分
    exp_bits/mant_bits : 指数/尾数的位宽(E4M3 为 4/3,E5M2 为 5/2)
    exp_bias : 指数偏移(E4M3 为 7,E5M2 为 15)
    特殊模式(指数全 1):E4M3FN 的 1111 111 为 NaN;E5M2 的 11111 000 为 ±inf,其余为 NaN。
    """
    sign = (code >> (exp_bits + mant_bits)) & 1
    exp = (code >> mant_bits) & ((1 << exp_bits) - 1)
    mant = code & ((1 << mant_bits) - 1)
    max_exp = (1 << exp_bits) - 1
    if exp == max_exp:                        # 特殊指数模式(指数全 1)
        if mant_bits == 2:                    # E5M2:000 → ±inf,其余 → NaN
            if mant == 0:
                return float("-inf") if sign else float("inf")
            return float("nan")
        if mant == (1 << mant_bits) - 1:      # E4M3FN:1111 111 → NaN
            return float("nan")
        # E4M3FN 其余 exp=15 码(000..110)仍是有限值,落到下面的 normal 计算
    if exp == 0:                          # subnormal / 零:隐含位为 0
        val = (mant / (2 ** mant_bits)) * (2.0 ** (1 - exp_bias))
    else:                                 # normal:隐含位为 1
        val = (1.0 + mant / (2 ** mant_bits)) * (2.0 ** (exp - exp_bias))
    return -val if sign else val

def fp8_all_values(exp_bits, mant_bits, exp_bias, positive_only=True):
    """枚举某个 FP8 格式的所有可表示值(默认只取非负有限值,便于对数坐标画图)。"""
    vals = []
    for code in range(256):
        v = fp8_decode(code, exp_bits, mant_bits, exp_bias)
        if positive_only and v < 0:
            continue
        if not math.isfinite(v):          # 跳过 NaN / ±inf,只留有限刻度
            continue
        vals.append(v)
    return sorted(set(vals))
''')
