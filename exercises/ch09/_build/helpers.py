# -*- coding: utf-8 -*-
"""ch09 生成公共组件:Notebook 构建 + Triton 各课共享的 kernel/工具源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH09 = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch09"
CHAPTER = "第 9 章 · Triton 编程"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 通用头部(每课第一段代码)
# 要点:Windows OMP 防护 + 指定可用的系统 ptxas(内嵌 ptxas 在本机报“内存分配失败”)
TRITON_HEADER = D('''
# -*- coding: utf-8 -*-
# 环境自检:KMP 防护 + 指定可用 ptxas + 打印 GPU/triton 信息
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")          # Windows OMP 冲突防护
_PTXAS = r"D:\\CUDA\\v13.3\\bin\\ptxas.exe"                     # 系统 CUDA 自带 ptxas
if os.path.exists(_PTXAS):                                      # 内嵌 ptxas 在本机编译失败
    os.environ["TRITON_PTXAS_PATH"] = _PTXAS
import math, time
import numpy as np
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font=["Microsoft YaHei", "DejaVu Sans"])
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200

torch.manual_seed(0)
print("torch   :", torch.__version__)
print("triton  :", triton.__version__)
print("CUDA    :", torch.cuda.is_available())
print("GPU     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("ptxas   :", os.environ.get("TRITON_PTXAS_PATH", "内嵌"))
''')

# ---------------------------------------------------------------- 通用 GPU 计时工具
BENCH = D('''
def bench(fn, *args, warmup=3, iters=10):
    """GPU 计时:先 warmup 预热(编译/缓存),再跑 iters 次取平均,返回毫秒。
    GPU 是异步执行的,必须 torch.cuda.synchronize() 等待内核真正完成,否则时间不准。"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0   # 毫秒
''')

# ---------------------------------------------------------------- 第 56/57 课共用:向量加法 kernel
VEC_ADD = D('''
@triton.jit
def vec_add_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    """向量加法:x + y -> o,每个 program 处理 BLOCK 个元素。"""
    pid = tl.program_id(0)                       # 第几个 program(CTAx)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 本 program 负责的全局索引
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

def vec_add(x, y, BLOCK=256, num_warps=4, num_stages=3):
    n = x.numel()
    o = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    vec_add_kernel[grid](x, y, o, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    return o
''')

# ---------------------------------------------------------------- 第 57/56 课共用:简化 GEMM kernel
GEMM = D('''
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """简化 GEMM:每 program 算一个 (BLOCK_M, BLOCK_N) 输出 tile。
    a: (M, K), b: (K, N), c: (M, N) —— 指针+stride 显式传入,适配任意步长布局。"""
    pid = tl.program_id(0)                        # 一维 program id
    num_pid_m = tl.cdiv(M, BLOCK_M)               # M 方向 tile 数
    num_pid_n = tl.cdiv(N, BLOCK_N)               # N 方向 tile 数
    pid_m = pid // num_pid_n                      # 拆成 (m, n) 二维
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # fp32 累加器
    for k in range(0, K, BLOCK_K):                # 沿 K 分块循环
        a = tl.load(a_ptrs)                       # 读一块 A
        b = tl.load(b_ptrs)                       # 读一块 B
        acc = tl.dot(a, b, acc)                   # tile 级矩阵乘
        a_ptrs += BLOCK_K * stride_ak             # 挪到下一块
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)              # 写回,带边界掩码

def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3):
    """matmul 封装:自动铺 grid、算 stride,返回 fp32 结果。a: (M,K), b: (K,N)。"""
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                        num_warps=num_warps, num_stages=num_stages)
    return c
''')

# ---------------------------------------------------------------- 第 58 课:paged decode attention kernel
PAGED_DECODE = D('''
@triton.jit
def paged_decode_attn_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr, block_table_ptr,
    seq_len_ptr, scale, num_kv_heads, max_blocks,
    BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr):
    """mini 版 triton paged decode attention(教学用)。
    一个 program 处理一条序列的一个 head;Q 是单个 token(query),
    K/V 从按物理块组织的 KV cache 里通过 block_table 收集。
    - k_cache / v_cache: (num_blocks, num_kv_heads, block_size, head_size)
    - block_table: (num_seqs, max_blocks),记录每条序列的物理块号
    - seq_len_ptr: 每条序列当前 token 数
    内部用 online softmax(m/l/acc)在块间滚动,数学与 FlashAttention 一致。"""
    pid = tl.program_id(0)
    seq_id = pid // num_kv_heads
    head_id = pid % num_kv_heads
    q_offs = seq_id * num_kv_heads * BLOCK_D + head_id * BLOCK_D + tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + q_offs)                          # (BLOCK_D,) 单个 query
    seq_len = tl.load(seq_len_ptr + seq_id)
    m = tl.full([1], float("-inf"), dtype=tl.float32)    # running max
    l = tl.zeros([1], dtype=tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)          # 输出累加器
    num_full = seq_len // BLOCK_SIZE                     # 完整块数
    for bi in range(0, max_blocks):                      # 逐块扫描物理块
        if bi < num_full:
            block_id = tl.load(block_table_ptr + seq_id * max_blocks + bi)
            base = block_id * num_kv_heads * BLOCK_SIZE * BLOCK_D + head_id * BLOCK_SIZE * BLOCK_D
            rows = tl.arange(0, BLOCK_SIZE)[:, None]
            cols = tl.arange(0, BLOCK_D)[None, :]
            kk = tl.load(k_cache_ptr + base + rows * BLOCK_D + cols)   # (BS, D)
            s = tl.sum(q[None, :] * kk, axis=1) * scale  # 块内打分 (BS,)
            m_new = tl.maximum(m, tl.max(s))             # 更新 max
            p = tl.exp(s - m_new)                        # 归一化(exp 而非 exp2!)
            alpha = tl.exp(m - m_new)                    # 旧累加校正系数
            l = l * alpha + tl.sum(p)
            vv = tl.load(v_cache_ptr + base + rows * BLOCK_D + cols)
            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m = m_new
    if seq_len % BLOCK_SIZE != 0:                        # 尾部不满一块:mask 读入
        block_id = tl.load(block_table_ptr + seq_id * max_blocks + num_full)
        base = block_id * num_kv_heads * BLOCK_SIZE * BLOCK_D + head_id * BLOCK_SIZE * BLOCK_D
        rows = tl.arange(0, BLOCK_SIZE)
        cols = tl.arange(0, BLOCK_D)
        mask = rows[:, None] < (seq_len - num_full * BLOCK_SIZE)
        kk = tl.load(k_cache_ptr + base + rows[:, None] * BLOCK_D + cols[None, :],
                     mask=mask, other=0.0)
        s = tl.sum(q[None, :] * kk, axis=1) * scale
        s = tl.where(rows < (seq_len - num_full * BLOCK_SIZE), s, float("-inf"))  # 关键:掩掉多余行
        m_new = tl.maximum(m, tl.max(s))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p)
        vv = tl.load(v_cache_ptr + base + rows[:, None] * BLOCK_D + cols[None, :],
                     mask=mask, other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
        m = m_new
    acc = acc / l                                        # 除以分母
    tl.store(o_ptr + q_offs, acc)


def paged_decode_attn(q, k_cache, v_cache, block_table, seq_lens, head_size, block_size=16):
    """wrapper:q (bs, nheads, hs);k_cache/v_cache (num_blocks, nkv, bs, hs);
    block_table (bs, max_blocks);seq_lens (bs,)。返回 out (bs, nheads, hs)。"""
    bs, nheads, hs = q.shape
    max_blocks = block_table.shape[1]
    out = torch.empty_like(q)
    grid = (bs * nheads,)
    paged_decode_attn_kernel[grid](
        q, k_cache, v_cache, out, block_table, seq_lens, hs ** -0.5, nheads, max_blocks,
        BLOCK_SIZE=block_size, BLOCK_D=head_size)
    return out
''')

# ---------------------------------------------------------------- 第 60 课:三个经典算子 kernel
OPLIB_KERNELS = D('''
# ---- ① LayerNorm ----
@triton.jit
def layer_norm_fwd(x_ptr, y_ptr, w_ptr, b_ptr, eps, M, N, row_stride,
                   BLOCK_N: tl.constexpr):
    """LayerNorm:每行一个 program。mean/var 在块内一次算完,最后乘 γ 加 β。"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * row_stride + offs, (xc * rstd) * w + b, mask=mask)


def layer_norm_triton(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    layer_norm_fwd[(M,)](x, y, w, b, eps, M, N, N, BLOCK_N=triton.next_power_of_2(N))
    return y


# ---- ② GELU(精确 erf 版)----
@triton.jit
def gelu_fwd(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    o = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))   # x/√2
    tl.store(o_ptr + offs, o, mask=mask)


def gelu_triton(x, BLOCK=1024):
    n = x.numel()
    o = torch.empty_like(x)
    gelu_fwd[(triton.cdiv(n, BLOCK),)](x, o, n, BLOCK=BLOCK)
    return o


# ---- ③ Softmax(数值稳定:减 max)----
@triton.jit
def softmax_fwd(x_ptr, o_ptr, M, N, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)                        # 先减 max,防 exp 溢出
    p = tl.exp(x - m)
    s = tl.sum(p, axis=0)
    tl.store(o_ptr + row * row_stride + offs, p / s, mask=mask)


def softmax_triton(x, BLOCK_N=128):
    M, N = x.shape
    o = torch.empty_like(x)
    softmax_fwd[(M,)](x, o, M, N, N, BLOCK_N=triton.next_power_of_2(N))
    return o
''')

# ---------------------------------------------------------------- 对拍工具(第 59/60 课)
ASSERT_CLOSE = D('''
def assert_allclose(actual, expected, atol=1e-5, rtol=1e-5, name=""):
    """对拍验证:对比 triton 输出与参考实现,打印 max/mean 误差。
    返回是否通过 torch.allclose 的 atol/rtol 检查。"""
    a = actual.float().detach().cpu()
    e = expected.float().detach().cpu()
    diff = (a - e).abs()
    rel = diff / (e.abs() + 1e-12)
    ok = torch.allclose(a, e, atol=atol, rtol=rtol)
    print(f"[{name}] max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}"
          f"  max_rel={rel.max().item():.3e}  allclose(atol={atol},rtol={rtol}) = {ok}")
    return ok
''')
