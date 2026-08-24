# -*- coding: utf-8 -*-
"""ch11 生成公共组件:Notebook 构建 + 共享样式/工具源码(华为昇腾与 MindSpore 生态)
本机无昇腾硬件、未安装 MindSpore → 全章用 torch 类比 + matplotlib 图示讲解真实概念。"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH11 = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch11"
CHAPTER = "第 11 章 · 华为昇腾与 MindSpore 生态"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


def finalize(path):
    """后处理 kernelspec name -> uv_cuda(与 ch10 一致)"""
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
HEADER = D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OMP 冲突防护
import math, json, time, numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font=["Microsoft YaHei", "DejaVu Sans"])
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0); np.random.seed(0)

print("torch  :", torch.__version__)
print("注意: 本机无昇腾硬件 / MindSpore,本课用 torch 类比 + matplotlib 图示讲解真实概念。")
''')

STYLE = D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 OMP 库冲突(Windows)
import time, math
import numpy as np
import pandas as pd
import torch

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
print("本机未安装 MindSpore / 无昇腾硬件:全章用 torch + matplotlib 类比讲解真实概念。")
''')
