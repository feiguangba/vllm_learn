# -*- coding: utf-8 -*-
"""shared real-data library — 跨章共用的「真实 CUDA 微基准 + 小 GPT」。

设计动机:全仓库 ipynb 要接入真实数据。很多课(推理基础、KV Cache、调度、Attention、
并行…)背后的**数**(prefill/decode 耗时、吞吐、显存、FLOPs)可统一由一个小型 GPT 后端
在 GPU 上实测得到,供各章直接引用。

本模块只做一件事:把「跑一个真实小模型/算子,拿到真实数字」集中、缓存、可复现地提供,
让所有章节的 ipynb 与 streamlit app 都调用它,而不是各自手写 np.arange。

用法(在 ipynb / app 中):
    import sys; sys.path.insert(0, r"<repo>/exercises")
    from vllm_real import TinyGPT, bench_prefill_decode, cuda_info

硬件探测:自动在 CUDA/CPU 间切换;结果用 lru_cache 缓存,避免重复测。
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _lazy_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


_TORCH = _lazy_torch()


def cuda_available() -> bool:
    return bool(_TORCH is not None and _TORCH.cuda.is_available())


def cuda_info() -> str:
    if not cuda_available():
        return "CPU(未检测到 CUDA)"
    try:
        import torch as t
        return f"{t.cuda.get_device_name(0)} · vram {t.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB"
    except Exception:
        return "CUDA(cuda:0)"


class TinyGPT:
    """一个只用线性层 + softmax 的「可真实前向」小 GPT,用于时序/显存/吞吐微基准。

    刻意**不含注意力/位置编码**,只用纯矩阵乘 + 量化吞吐上界,便于在任意机器上跑。
    参数化 (d, layers, vocab_size, n_kv_slots),尺寸小到能推理又能体现 prefill/decode 差异。
    """

    def __init__(self, d: int = 256, layers: int = 8, vocab_size: int = 1024,
                 device: str = None):
        self.d, self.layers, self.vocab = d, layers, vocab_size
        self.device = device
        torch = _TORCH
        g = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Parameter(torch.randn(vocab_size, d, generator=g) * 0.02)
        self.ws = [torch.nn.Parameter(torch.randn(d, d, generator=g) * (1.0 / d ** 0.5))
                   for _ in range(layers)]
        self.head = torch.nn.Parameter(torch.randn(d, vocab_size, generator=g) * 0.02)
        if self.device and torch is not None:
            self.to(self.device)

    def to(self, device: str):
        if _TORCH is None:
            return self
        self.embed = self.embed.to(device)
        self.ws = [w.to(device) for w in self.ws]
        self.head = self.head.to(device)
        self.device = device
        return self

    def params_count(self) -> int:
        return self.vocab * self.d + self.layers * self.d * self.d + self.d * self.vocab

    def forward_logits(self, x):
        """输入 (B, T),输出 (B, T, vocab) 的 logits(不含 softmax)。"""
        torch = _TORCH
        h = torch.tanh(self.embed[x])          # (B, T, d)
        for w in self.ws:
            h = torch.tanh(h @ w)               # (B, T, d)
        return h @ self.head                    # (B, T, vocab)


@lru_cache(maxsize=16)
def bench_prefill_decode(d: int = 256, layers: int = 8, L: int = 256, steps: int = 64,
                         reps: int = 7, vocab_size: int = 1024, device: str = None) -> Dict:
    """在 GPU 上实测「一次并行 prefill」vs「逐 token decode」的同量耗时。

    prefill:一次过 (L,) 个 token,得到 logits,计时;
    decode :把同样的 L 个 token **逐个**过 (1,) 并累计,计时(模拟逐字生成)。

    返回 dict: device, d, layers, params, L, prefill_ms, decode_ms, ratio,
                prefill_ms_per_token, throughput_prefill_tok_s, ...
    """
    t = _TORCH
    if t is None:
        return {"simulated": True, "device": "no-torch", "reason": "torch 不可用"}
    if t.cuda.is_available() and (device is None or device == "cuda"):
        device = "cuda"
    else:
        device = "cpu" if device is None else device

    model = TinyGPT(d=d, layers=layers, vocab_size=vocab_size, device=device)
    n_params = model.params_count()
    L = int(L)
    x_prefill = t.randint(0, model.vocab, (1, L), device=device)
    x_one = x_prefill[:, :1]

    def timeit(fn, reps):
        # 预热
        with t.no_grad():
            for _ in range(2):
                fn()
            if t.cuda.is_available():
                t.cuda.synchronize()
        runs = []
        with t.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                if t.cuda.is_available():
                    t.cuda.synchronize()
                runs.append(time.perf_counter() - t0)
        return float(np.median(runs))

    with t.no_grad():
        tp = timeit(lambda: model.forward_logits(x_prefill), reps)          # 一次并行
        td = timeit(lambda: model.forward_logits(x_one), reps)              # 一步
    total_decode = td * L                                                    # L 步 = 生成 L token

    return {
        "simulated": False, "device": device,
        "d": d, "layers": layers, "vocab": model.vocab,
        "params": int(n_params), "L": L,
        "prefill_ms": tp * 1e3,          # 一次并行前向 L token
        "decode_step_ms": td * 1e3,      # 每一步(1 token)
        "decode_total_ms": total_decode * 1e3,   # 逐 token 完成 L 个
        "ratio": total_decode / max(tp, 1e-12),  # decode/prefill 耗时比
        "prefill_tok_per_s": L / max(tp, 1e-12),
        "decode_tok_per_s": L / max(total_decode, 1e-12),
    }


@lru_cache(maxsize=8)
def bench_throughput_curve(d: int = 256, layers: int = 8, batch=list,
                           token_len: int = 16, reps: int = 5,
                           device: str = None) -> Tuple[list, list]:
    """给定 batch,测「单位时间生成 token 数」吞吐(近似 vLLM decode 吞吐思想)。

    逐 decode 步,每步把 batch 个单 token 一起前向(continuous batching 的雏形),
    测不同 batch 的 tokens/s,返回 (batch_list, tokens_per_s_list, ms_per_step_list)。
    """
    t = _TORCH
    if t is None:
        return [], [], []
    if t.cuda.is_available() and (device is None or device == "cuda"):
        device = "cuda"
    else:
        device = "cpu" if device is None else device
    model = TinyGPT(d=d, layers=layers, device=device)
    steps = int(token_len)
    b_list, tps, mps = [], [], []
    for b in batch:
        b = int(b)
        x = t.randint(0, model.vocab, (b, 1), device=device)
        _ = model.forward_logits(x)  # warmup
        if t.cuda.is_available():
            t.cuda.synchronize()
        runs = []
        with t.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                for _ in range(steps):
                    x = t.randint(0, model.vocab, (b, 1), device=device)
                    model.forward_logits(x)
                if t.cuda.is_available():
                    t.cuda.synchronize()
                runs.append(time.perf_counter() - t0)
        ms_step = float(np.median(runs)) / steps * 1e3
        b_list.append(b)
        mps.append(ms_step)
        tps.append(b / (ms_step / 1e3))
    return b_list, tps, mps


def real_table_html(bench: Dict) -> str:
    """把 bench 结果渲染成精简 HTML(notebook 里 markdown/HTML 展示)。"""
    if bench.get("simulated"):
        return f"<i>模拟(无 torch):{bench.get('reason','')}</i>"
    return (f"设备 **{bench['device']}** · 参数量 ~{bench['params']/1e6:.1f}M · L={bench['L']}<br>"
            f"prefill 一次并行: **{bench['prefill_ms']:.2f} ms**<br>"
            f"decode 逐 token(总数相同): **{bench['decode_total_ms']:.2f} ms** · "
            f"耗时比 **{bench['ratio']:.1f}×**")