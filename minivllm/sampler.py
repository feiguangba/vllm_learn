# -*- coding: utf-8 -*-
"""采样策略（对应第 03 课）：greedy / temperature / top-k / top-p。"""
import torch


class Sampler:
    def __init__(self, temperature=1.0, top_k=0, top_p=1.0, greedy=False, seed=None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.greedy = greedy
        self.gen = None if seed is None else torch.Generator().manual_seed(seed)

    def __call__(self, logits):
        """logits: (B, V) → 返回 (B,) 的采样 token id。"""
        if self.greedy or self.temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / self.temperature

        # top-k：只保留分数最高的 k 个
        if self.top_k and self.top_k > 0:
            kth = logits.topk(self.top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        # top-p：从高到低累积概率，截到第一次超过 p 为止
        if 0 < self.top_p < 1.0:
            sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = cum - probs > self.top_p          # 保留第一个超过 p 的项
            logits = logits.scatter(-1, idx, sorted_logits.masked_fill(mask, float("-inf")))

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=self.gen).squeeze(-1)
