# -*- coding: utf-8 -*-
"""ch07 生成公共组件:Notebook 构建 + 各课共享的注意力算法源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH07 = r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch07"
CHAPTER = "第 7 章 · Attention Kernel 实战"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 通用头部
# 每个 notebook 第一段代码:打印设备与库版本,统一 GPU 环境
CUDA_HEADER = D('''
# -*- coding: utf-8 -*-
# 环境自检:确认 GPU、torch、triton 状态(每课都会先跑这一段)
import torch, numpy as np
import math

print("torch           :", torch.__version__)
print("CUDA 可用        :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU              :", torch.cuda.get_device_name(0))
    print("CUDA 计算能力     :", torch.cuda.get_device_capability(0))
try:
    import triton
    import triton.language as tl
    print("triton           :", triton.__version__, "(可用)")
except Exception as e:
    print("triton           : 不可用 →", type(e).__name__, e)
''')

# ---------------------------------------------------------------- 统一设备头部（有 CUDA 就跑真实 GPU）
# ch07 所有 attention 实验统一在这里自检:有 CUDA 就走 RTX（naive/分块/SDPA 真实时延），无才回退 CPU。
# 名字保留为 CPU_HEADER 以不破坏 build_42..47 的旧引用；其内容已升级为「自动选设备」。
CPU_HEADER = D('''
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OMP 库冲突防护
import math, time, torch, numpy as np

dev = "cuda" if torch.cuda.is_available() else "cpu"     # 有 CUDA 就跑真实 GPU（RTX 5060）
torch.manual_seed(0)
print("torch  :", torch.__version__)
print("device :", dev, "| GPU:", torch.cuda.get_device_name(0) if dev == "cuda" else "无(CPU)")
print("num_threads :", torch.get_num_threads())
print("SDPA flash 可用         :", torch.backends.cuda.flash_sdp_enabled())
print("SDPA mem_efficient 可用 :", torch.backends.cuda.mem_efficient_sdp_enabled())
''')

# ---------------------------------------------------------------- 朴素注意力(三步)
NAIVE_ATTN = D('''
def naive_attention(Q, K, V):
    """朴素注意力:把 QK^T / softmax / 加权和 三个大张量全部写进 HBM。
    Q, K, V: (B, H, N, d)。返回与输入同形状的输出 O。
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])          # 1/√d 缩放因子
    S = torch.einsum("bhnd,bhmd->bhnm", Q, K) * scale   # ① QK^T: (B,H,N,N)
    P = torch.softmax(S, dim=-1)                  # ② softmax: (B,H,N,N)
    O = torch.einsum("bhnm,bhmd->bhnd", P, V)     # ③ 加权和: (B,H,N,d)
    return O
''')

# ---------------------------------------------------------------- 分块 FlashAttention(torch 手写)
FLASH_CHUNKED = D('''
def flash_attention_chunked(Q, K, V, block_M=64):
    """分块 FlashAttention(torch 手写版):online softmax + K/V 分块扫描。
    只在寄存器/片上保存 running max(m)、running sum(l) 与累加器 O,
    大张量 S/P 从不整体落进 HBM——这就是 FlashAttention 的核心思想。
    Q, K, V: (B, H, N, d)。block_M 控制 K/V 方向上每次读入的块宽。
    """
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)                                   # 输出累加器
    m = torch.full((B, H, N, 1), float("-inf"), device=Q.device)  # 逐行 running max
    l = torch.zeros((B, H, N, 1), device=Q.device)                # 逐行 running sum
    scale = 1.0 / math.sqrt(d)
    for j in range(0, N, block_M):                            # 沿 K/V 分块扫描
        Kj = K[:, :, j:j + block_M, :]                        # 读一块 K (B,H,Bm,d)
        Vj = V[:, :, j:j + block_M, :]                        # 读一块 V (B,H,Bm,d)
        S = torch.einsum("bhnd,bhmd->bhnm", Q, Kj) * scale    # 块内打分 (B,H,N,Bm)
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)   # 更新 running max
        P = torch.exp(S - m_new)                              # 以新 max 归一(防溢出)
        l_new = l * torch.exp(m - m_new) + P.sum(dim=-1, keepdim=True)  # 更新 running sum
        O = O * torch.exp(m - m_new) + torch.einsum("bhnm,bhmd->bhnd", P, Vj)  # 校正旧累加
        m = m_new                                             # 滚到下一块
        l = l_new
    return O / l                                              # 最终除以分母
''')

# ---------------------------------------------------------------- 三种 softmax(第 43 课)
SOFTMAX_NAIVE = D('''
def softmax_naive(x):
    """① 朴素 softmax:直接 exp 再归一。输入偏大时 exp 溢出成 inf,结果变成 NaN。"""
    e = torch.exp(x)                # 直接 exp —— 危险!
    return e / e.sum(dim=-1, keepdim=True)
''')

SOFTMAX_TWOPASS = D('''
def softmax_twopass(x):
    """② 两遍 softmax:先减最大值(第一遍),再 exp 归一(第二遍)。数值稳定,但要读两遍数据。"""
    m = x.max(dim=-1, keepdim=True).values   # 第一遍:找最大值
    e = torch.exp(x - m)                     # 减去 max 再 exp,永不溢出
    return e / e.sum(dim=-1, keepdim=True)   # 第二遍:求和归一
''')

SOFTMAX_ONLINE = D('''
def softmax_online(x, block=2):
    """③ online softmax:一趟扫描,边读边维护 running max(m) 与 running sum(l)。
    每看到一个块,就用「旧 m 与当前块最大值」校正已累积的 l —— 这就是 FlashAttention 的数学基础。
    """
    m = torch.full((x.shape[0], 1), float("-inf"), device=x.device)   # running max
    l = torch.zeros((x.shape[0], 1), device=x.device)                 # running sum
    for i in range(0, x.shape[-1], block):
        xb = x[:, i:i + block]                                        # 读一块
        m_new = torch.maximum(m, xb.max(dim=-1, keepdim=True).values) # 更新 max
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new).sum(dim=-1, keepdim=True)  # 校正旧和
        m = m_new
    return torch.exp(x - m) / l                                       # 最终归一(再读一遍 x)
''')

# ---------------------------------------------------------------- FLOPs / 显存公式(第 42 课)
FLOP_MEM = D('''
def attn_flops(N, H, D):
    """注意力部分的 FLOPs,拆成三步:
    ① QK^T:  N×N 个点积,每个点积长度 d,含 N·N·d 次乘加 → 2·N²·H·D
    ② softmax: 每个元素约 5 次浮点操作(exp/求和/除法)→ 约 5·N²·H
    ③ 加权和 AV: 同 QK^T → 2·N²·H·D
    """
    qk = 2 * N * N * H * D
    softmax = 5 * N * N * H
    av = 2 * N * N * H * D
    return qk, softmax, av, qk + softmax + av

def proj_flops(N, H, D):
    """QKV 投影 + 输出投影的 FLOPs(每层)。
    QKV 投影: X@W_qkv,输入 N×d_model,权重 d_model×3d_model → 2·N·d_model·3d_model = 6N·d_model²
    输出投影: 2·N·d_model²。合计每层 ≈ 8N·d_model²,d_model = H·D。
    """
    dm = H * D
    qkv = 6 * N * dm * dm
    out = 2 * N * dm * dm
    return qkv, out, qkv + out

def attn_memory(N, H, D, dtype_bytes=2):
    """naive attention 的显存开销(单层、单 batch)。
    Q/K/V 各 N·d_model 个元素;打分 S 与概率 P 各 H·N² 个元素(这是大头!)。"""
    qkv = 3 * N * H * D * dtype_bytes
    scores = H * N * N * dtype_bytes
    probs = H * N * N * dtype_bytes
    return qkv, scores, probs, qkv + scores + probs
''')

# ---------------------------------------------------------------- 真实可调用的 FLOPs / 访存函数(供 build 脚本与 app 用)
def flop_attention(N, H, D):
    """attention 总 FLOPs ≈ 4·N²·H·D(QK^T 与 PV 各约 2·N²·H·D)。"""
    return 4 * N * N * H * D

def hbm_naive_bytes(N, H, D, dtype_bytes=2):
    """naive attention 的 HBM 访存:5·N·H·D + 4·H·N² 字节。"""
    return 5 * N * H * D * dtype_bytes + 4 * H * N * N * dtype_bytes

def hbm_flash_bytes(N, H, D, dtype_bytes=2):
    """flash attention 的 HBM 访存:4·N·H·D 字节(Q/K/V 各读一次,O 写一次)。"""
    return 4 * N * H * D * dtype_bytes

def mem_naive_bytes(N, H, D, dtype_bytes=2):
    """naive 峰值显存:QKV + S + P + O。"""
    return (3 * N * H * D + 2 * H * N * N + N * H * D) * dtype_bytes

def mem_flash_bytes(N, H, D, dtype_bytes=2):
    """flash 峰值显存:QKV + O,S/P 不落盘。"""
    return (3 * N * H * D + N * H * D) * dtype_bytes

# ---------------------------------------------------------------- 计时工具(第 45/47 课)
BENCH = D('''
def bench(fn, *args, warmup=3, iters=10):
    """GPU 计时:先 warmup 预热(编译、缓存),再跑 iters 次取平均,返回毫秒。"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0   # 毫秒
''')

# ---------------------------------------------------------------- Triton FlashAttention 参考 kernel(第 45 课)
# 说明:这是 triton 官方教程的因果(下三角)前向 kernel,完整可编译。本环境 `import triton` 成功、
# 后端识别为 cuda arch=120,但沙箱禁止 triton 编译时对系统临时目录的写入,故本课以「参考源码 +
# torch 分块可执行版」呈现(详见第 45 课正文)。
TRITON_FLASH_KERNEL = D('''
@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, L, O,
                stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk,
                stride_vz, stride_vh, stride_vn, stride_vk,
                stride_oz, stride_oh, stride_om, stride_ok,
                Z, H, N_CTX,
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    """因果 FlashAttention 前向 kernel(单 program 处理 Q 的一行块)。"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)          # (batch, head) 拍平成的一维索引
    qvk_offset = off_hz * stride_qh
    # —— 用 block pointer 描述 Q/K/V/O 在 HBM 上的分块视图 ——
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(base=O + qvk_offset, shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    # —— 行号、列号 ——
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # —— online softmax 的 running 状态(全部驻留在寄存器/SRAM)——
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # —— 读入 Q 行块并缩放 ——
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * sm_scale).to(tl.float16)
    # —— 沿 K/V 的 N 方向逐块扫描(只扫到对角,即因果掩码)——
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.dot(q, k)                                   # 块内 QK^T (BLOCK_M, BLOCK_N)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))               # 更新 running max
        p = tl.math.exp2(qk - m_ij[:, None])                # 以新 max 归一(exp2 更快)
        l_ij = tl.sum(p, 1)                                 # 本块的分母贡献
        alpha = tl.math.exp2(m_i - m_ij)                    # 旧累加器的校正系数
        l_i = l_i * alpha + l_ij                            # 更新 running sum
        acc = acc * alpha[:, None]                          # 校正旧输出
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)                             # 块内加权和累加
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # —— 收尾:除以分母,写回输出与 logsumexp ——
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.dtype.element_ty), boundary_check=(0,))
    l_i = m_i + tl.math.log2(l_i)
    tl.store(L + off_hz * N_CTX + offs_m, l_i)
''')
