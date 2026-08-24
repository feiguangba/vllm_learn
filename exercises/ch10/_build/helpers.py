# -*- coding: utf-8 -*-
"""ch10 生成公共组件:Notebook 构建 + 共享样式/工具源码"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH10 = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch10"
CHAPTER = "第 10 章 · AI 编译器原理"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


def finalize(path):
    """后处理 kernelspec name -> uv_cuda(与 ch04 一致)"""
    import nbformat
    p = Path(path)
    nb = nbformat.read(p, as_version=4)
    nb.metadata.setdefault("kernelspec", {})["name"] = "uv_cuda"
    nb.metadata["kernelspec"]["display_name"] = "Python 3 (uv_cuda)"
    nb.metadata["kernelspec"]["language"] = "python"
    nbformat.write(nb, p)
    print(f"[finalize] {p}")


def app_cell(name, src):
    """生成 %%writefile cell:把 app 源码原样写回磁盘"""
    return "%%writefile " + name + "\n" + src


# ------------------------------------------------------------------ 统一的 KMP 保护 + 会议论文风格头(第一个 code cell)
STYLE = D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch 的 OMP 库冲突(Windows)
import time, json, io, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")              # 开启 TF32,加速并消除告警
print("torch =", torch.__version__,
      "| CUDA =", torch.cuda.is_available(),
      "| device =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---------- 会议论文风格绘图:matplotlib + seaborn ----------
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 中文
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
''')

# ------------------------------------------------------------------ GPU 计时工具(lesson 62/65/66/67/69 共用)
TIME_CUDA = D('''
def bench_cuda_ms(fn, warmup=10, repeat=50):
    """用 torch.cuda.Event 精确计时:返回平均单次耗时(ms)。fn 为无参可调用对象。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat
''')

# ------------------------------------------------------------------ 从编译产物里数 Triton kernel(lesson 66/67 共用)
TRITON_COUNT = D('''
import re

def count_triton_kernels(compiled_fn, *args):
    """用 get_triton_code 取回 inductor 生成的 Triton 源码,数一数里面定义了几个 kernel。
    返回 (kernel 数量, 源码字符串, kernel 名字列表)。args 为编译函数的真实输入。"""
    from torch._inductor.utils import get_triton_code
    code = get_triton_code(compiled_fn, *args)
    names = re.findall(r"def (triton_[a-zA-Z0-9_]+)\\(", code)
    return len(names), code, names
''')
