# -*- coding: utf-8 -*-
"""ch04 生成公共组件:Notebook 构建 + 共享的迷你 GPT 模型 / 计时工具源码"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH04 = r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch04"
CHAPTER = "第 4 章 · 模型执行与 CUDA Graph"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


def app_src(name):
    """读取已存在的 streamlit app 源码(与本目录 .py 完全一致)"""
    return (Path(CH04) / name).read_text(encoding="utf-8")


def finalize(path):
    """后处理 kernelspec name -> uv_cuda(参考 ch04 已做的后处理)"""
    import nbformat
    p = Path(path)
    nb = nbformat.read(p, as_version=4)
    nb.metadata.setdefault("kernelspec", {})["name"] = "uv_cuda"
    nb.metadata["kernelspec"]["display_name"] = "Python 3 (uv_cuda)"
    nb.metadata["kernelspec"]["language"] = "python"
    nbformat.write(nb, p)
    print(f"[finalize] {p}")


# ------------------------------------------------------------------ 迷你 GPT
# 一台"麻雀虽小五脏俱全"的迷你 GPT:H=256、2 层、4 头、词表 5000。
# 前向 = embedding → N×(注意力 + 门控 MLP) → lm_head。22/25/26/27 课共用。
MINI_GPT = D('''
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")   # 开启 TF32,加速并消除告警

class AttnBlock(nn.Module):
    """多头因果自注意力块:LayerNorm → QKV → 分头 → SDPA → 输出投影 + 残差"""
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        y = self.norm(x)                       # [T, H]
        q, k, v = self.qkv(y).chunk(3, dim=-1) # 各 [T, H]
        q = q.view(-1, self.n_heads, self.head_dim).transpose(0, 1)  # [heads, T, d]
        k = k.view(-1, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, self.n_heads, self.head_dim).transpose(0, 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [heads, T, d]
        out = out.transpose(0, 1).reshape(-1, self.n_heads * self.head_dim)  # [T, H]
        return x + self.o(out)

class MLPBlock(nn.Module):
    """门控 MLP:LayerNorm → gate/up 融合 → SiLU 门控 → down + 残差"""
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.gate_up = nn.Linear(hidden, 4 * hidden, bias=False)
        self.down = nn.Linear(2 * hidden, hidden, bias=False)

    def forward(self, x):
        y = self.norm(x)                       # [T, H]
        g, u = self.gate_up(y).chunk(2, dim=-1)  # 各 [T, 2H]
        return x + self.down(F.silu(g) * u)

class MiniGPT(nn.Module):
    """迷你 GPT:vLLM 术语里的 embedding → layers → lm_head 三段结构"""
    def __init__(self, vocab=5000, hidden=256, n_layers=2, n_heads=4, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.layers = nn.ModuleList([
            nn.ModuleList([AttnBlock(hidden, n_heads), MLPBlock(hidden)])
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.hidden, self.vocab = hidden, vocab

    def forward(self, input_ids, positions):
        # input_ids / positions: [T] 长整型(紧凑拼接后的词元流,见第 21 课)
        h = self.embed(input_ids) + self.pos(positions)   # [T, H]
        for attn, mlp in self.layers:
            h = attn(h)
            h = mlp(h)
        logits = self.lm_head(h)                          # [T, V]
        return logits
''')

# ------------------------------------------------------------------ 计时工具
TIME_EVENTS = D('''
import torch

def time_ms(fn, warmup=5, repeat=50):
    """用 torch.cuda.Event 精确计时:返回平均单次耗时(ms)。
    fn 是零参数可调用对象;warmup 让 CUDA 上下文与缓存就绪后再测。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat
''')
