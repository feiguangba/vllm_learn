# -*- coding: utf-8 -*-
"""TinyGPT（对应第 04/22 课）：一个随机初始化的 2 层小 GPT。

它只负责「形状与数据流正确」——权重是随机的，输出是伪文本，
但 prefill → 逐 token decode → 分页 KV 读写 → 采样，整条推理路径是真的。
想接真模型：把本类换成任意能逐步吐 logits 的 decoder-only 模型即可。

路径约定：
    prefill     块内自带因果注意力，块负责「两段残差」，返回更新后的 x 与 (q,k,v)
    decode_step 引擎先 gather 物理块拼 K/V，再让块走 step() 手工两段残差
"""
import math
import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=512, seed=42):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.head_dim = d_model // n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        torch.manual_seed(seed)            # 固定初始化，输出可复现
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------- 形状辅助 ----------
    def _split(self, x):
        """(B,T,d_model) → (B,H,T,head_dim)。"""
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _attend(self, q, k, v):
        """q:(B,H,Tq,D) k/v:(B,H,Tk,D) → (B,Tq,d_model)。无未来掩码：
        prefill 的因果掩码在 _Block 内部做；decode 只看得到已写入缓存的 token。"""
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v                                    # (B,H,Tq,D)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[-2], self.d_model)

    # ---------- 两条推理路径 ----------
    def prefill(self, ids, cache, seq_id):
        """整段 prompt 并行前向（算力瓶颈），K/V 逐层写入分页缓存。

        返回最后一个位置的 logits，形状 (1, vocab)。"""
        T = len(ids)
        x = self.tok_emb(torch.tensor(ids))[None] + self.pos_emb(torch.arange(T))[None]
        for li, blk in enumerate(self.blocks):
            _, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
            k, v = self._split(k), self._split(v)
            for pos in range(T):               # 逐 token 落盘（教学版不做批量写）
                cache.write(seq_id, li, k[0, :, pos].detach(), v[0, :, pos].detach())
            x = blk.forward_with_kv(x)         # 块内两段残差
        return self.lm_head(self.ln_f(x))[:, -1, :]      # (1, vocab) 末位 logits

    def decode_step(self, token_id, cache, seq_id):
        """单 token 增量解码（带宽瓶颈）：Q=当前 token，K/V=物理块 gather。

        当前 token 的 K/V 在注意力拼好后才写入缓存 → 自己能看到自己、看不到未来。
        返回 (1, vocab) 的 logits。"""
        pos = cache.seqlens[seq_id]
        x = self.tok_emb(torch.tensor([token_id]))[None] + self.pos_emb(torch.tensor([pos]))[None]
        for li, blk in enumerate(self.blocks):
            q, k_t, v_t = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)   # 各 (1,1,d)
            q4, k4, v4 = self._split(q), self._split(k_t), self._split(v_t)
            k_hist, v_hist = cache.gather(seq_id, li)             # (T,H,D) 已有历史
            if k_hist is not None:
                k = torch.cat([k_hist.permute(1, 0, 2).unsqueeze(0), k4], dim=2)
                v = torch.cat([v_hist.permute(1, 0, 2).unsqueeze(0), v4], dim=2)
            else:
                k, v = k4, v4
            att = self._attend(q4, k, v)
            cache.write(seq_id, li, k4[0, :, 0].detach(), v4[0, :, 0].detach())
            x = x + att
            x = x + blk.ffn(blk.ln2(x))
        return self.lm_head(self.ln_f(x))[:, -1, :]          # (1, vocab)


class _Block(nn.Module):
    """pre-LN transformer 块。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward_with_kv(self, x):
        """prefill 路径：带因果掩码的自注意力 + 两段残差，返回更新后的 x。"""
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        sp = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = sp(q) @ sp(k).transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        x = x + (att @ sp(v)).transpose(1, 2).reshape(B, T, -1)
        x = x + self.ffn(self.ln2(x))
        return x
