# -*- coding: utf-8 -*-
"""迷你 BPE 分词器（对应第 01 课）。

实现最简 byte-pair encoding：训练时每轮合并语料中出现次数最多的相邻对，
编码时贪心套用学到的合并规则。够教 BPE 的「压缩」本质，不追求吞吐。
"""
import re
from collections import Counter


class MiniBPETokenizer:
    def __init__(self, vocab_size=256):
        # 词表：0..255 是字节 token，之后每学会一个合并就多一个 token
        self.merges = {}          # (a, b) -> 新 id
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.target = vocab_size

    # ---------- 训练 ----------
    def train(self, texts, verbose=False):
        corpus = list("".join(texts).encode("utf-8"))
        for step in range(self.target - 256):
            pairs = Counter(zip(corpus, corpus[1:]))
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            new_id = 256 + step
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            corpus = self._merge(corpus, (a, b), new_id)
            if verbose and step % 20 == 0:
                print(f"merge {step}: {(a, b)} -> {new_id}  ({len(corpus)} tokens)")
        return self

    @staticmethod
    def _merge(seq, pair, new_id):
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ---------- 编码 / 解码 ----------
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # 按学会的顺序重放合并（保证 (a,b) 的 a/b 已是更早的 token）
        for pair, new_id in self.merges.items():
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def compression_ratio(self, text):
        n = len(self.encode(text))
        return len(text) / n if n else float("inf")
