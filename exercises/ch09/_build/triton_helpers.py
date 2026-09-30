# -*- coding: utf-8 -*-
"""ch09 第 51-55 课专用生成组件(Triton 编程章节)。
独立模块(不占用共享 helpers.py),供 build_51..build_55 使用。"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH09 = r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch09"
CHAPTER = "第 9 章 · Triton 编程"

LOG2E = 1.4426950408889634


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 绘图统一样式头(TASK4: matplotlib + seaborn 静态图)
PLT_STYLE = D('''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font=["DejaVu Sans"])
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
''')

# ---------------------------------------------------------------- 每课第一段代码:环境自检 + KMP 保护
ENV_HEADER = D('''
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OMP 库冲突防护
import math, time, torch, numpy as np
import triton
import triton.language as tl

torch.manual_seed(0)
print("torch   :", torch.__version__)
print("triton  :", triton.__version__)
print("CUDA    :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
''')

# ---------------------------------------------------------------- GPU 计时工具(warmup + 同步)
BENCH = D('''
def bench(fn, *args, warmup=5, iters=20):
    """GPU 计时:先 warmup 预热(编译、缓存),再跑 iters 次取平均,返回毫秒。
    GPU kernel 是异步的,必须 torch.cuda.synchronize() 才能真正拿到耗时。"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0   # 毫秒
''')

# ---------------------------------------------------------------- 第 51 课:生命周期演示 kernel(把每个元素的 block/lane 写出来)
LIFECYCLE_KERNEL = D('''
@triton.jit
def grid_demo(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                     # 我是第几个 program(block)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 本 block 负责的元素下标
    lane = tl.arange(0, BLOCK_SIZE)                 # 块内“通道”编号
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, pid * 1.0 + lane * 1e-3, mask=mask)
''')

# ---------------------------------------------------------------- 第 52 课:向量加法 kernel
ADD_KERNEL = D('''
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                     # 本 program 的编号(决定管哪一段)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 这一段的下标
    mask = offsets < n_elements                     # 边界保护:越界的元素忽略
    x = tl.load(x_ptr + offsets, mask=mask)         # 读一块 x
    y = tl.load(y_ptr + offsets, mask=mask)         # 读一块 y
    tl.store(out_ptr + offsets, x + y, mask=mask)   # 逐元素相加并写回
''')

# ---------------------------------------------------------------- 第 54 课:GEMM tile kernel(fp16 张量核)
GEMM_KERNEL = D('''
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  sm, ak, bk, bn, cm, cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)                       # 行方向块编号
    pid_n = tl.program_id(1)                       # 列方向块编号
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 本 tile 的行下标
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # 本 tile 的列下标
    offs_k = tl.arange(0, BLOCK_K)                 # K 方向(归约轴)下标
    a_ptrs = A + offs_am[:, None] * sm + offs_k[None, :] * ak   # A 的 (BLOCK_M, BLOCK_K) 块指针
    b_ptrs = B + offs_k[:, None] * bk + offs_bn[None, :] * bn   # B 的 (BLOCK_K, BLOCK_N) 块指针
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)        # 累加器
    for k in range(0, tl.cdiv(K, BLOCK_K)):        # 沿 K 方向逐步归约
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                    # 块内矩阵乘并累加
        a_ptrs += BLOCK_K * ak                     # 前进到下一块
        b_ptrs += BLOCK_K * bk
    c_ptrs = C + offs_am[:, None] * cm + offs_bn[None, :] * cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
''')

# ---------------------------------------------------------------- 第 55 课:FlashAttention kernel(fp16)
FA_KERNEL = D('''
LOG2E = 1.4426950408889634     # 1/ln2:把 exp2 换成自然底数 e,等价于标准 softmax

@triton.jit
def fa_kernel(Q, K, V, O, sm_scale, LOG2E, M, N, D,
              sqm, sqk, skn, skk, svn, svk, som, sok,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    # ---- 本 program 负责 Q 的第 start_m 行块 ----
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=(offs_m[:, None] < M), other=0.0)
    q = (q * sm_scale).to(tl.float16)              # 预缩放打分
    # ---- online softmax 的 running 状态(全在寄存器/SRAM,不落盘) ----
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)         # 输出累加器
    for start_n in range(0, tl.cdiv(N, BLOCK_N)):  # 沿 K/V 方向分块扫描
        offs_nn = start_n * BLOCK_N + offs_n
        mask_k = (offs_d[:, None] < D) & (offs_nn[None, :] < N)   # K^T 块 (D, N)
        mask_v = (offs_nn[:, None] < N) & (offs_d[None, :] < D)   # V 块 (N, D)
        mask_qk = (offs_m[:, None] < M) & (offs_nn[None, :] < N)  # 打分块 (M, N)
        k = tl.load(K + offs_nn[None, :] * skn + offs_d[:, None] * skk, mask=mask_k, other=0.0)
        v = tl.load(V + offs_nn[:, None] * svn + offs_d[None, :] * svk, mask=mask_v, other=0.0)
        qk = tl.dot(q, k)                          # 块内 QK^T (M, N)
        qk = tl.where(mask_qk, qk, float("-inf"))  # 越界位置置 -inf
        m_ij = tl.maximum(m_i, tl.max(qk, 1))      # 更新 running max
        p = tl.math.exp2((qk - m_ij[:, None]) * LOG2E)   # 以新 max 归一(exp2+LOG2E = e^x)
        l_ij = tl.sum(p, 1)                        # 本块分母贡献
        alpha = tl.math.exp2((m_i - m_ij) * LOG2E) # 旧累加器的校正系数
        l_i = l_i * alpha + l_ij                   # 更新 running sum
        acc = acc * alpha[:, None]                 # 校正旧输出
        acc = tl.dot(p.to(tl.float16), v, acc)     # 块内加权和累加
        m_i = m_ij
    acc = acc / l_i[:, None]                       # 最终除以分母
    tl.store(O + offs_m[:, None] * som + offs_d[None, :] * sok,
             acc.to(O.dtype.element_ty), mask=(offs_m[:, None] < M))
''')
