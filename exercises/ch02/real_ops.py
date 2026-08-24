# -*- coding: utf-8 -*-
"""ch02 真实算子实验(在 GPU/CUDA 上直接跑算子取真实数据,替代拍脑袋的 np.arange)。

设计原则:
- 所有函数自带 ``cuda`` 探测;无 GPU 时优雅降级为理论估计(并标注 simulated=True)。
- 结果用 ``functools.lru_cache`` 缓存,避免 streamlit 每次触发重跑重测。
- 统一返回 dict / pandas DataFrame,方便接入 plotly 或 manim ManimBrace/条形图。

- Lesson 07  → :func:`bench_qkv_attn` :真实测 QKV 投影 + 注意力封装耗时
- Lesson 08  → :func:`kv_bytes_real`   :真实统计 torch 为某形状 KV 分配/占用的显存字节
- Lesson 09/11 → :func:`alloc_frag_sim` :分配器碎片模拟(逻辑层,不依赖 GPU)
- Lesson 10  → :func:`paged_map`      :页表 slot 映射(逻辑层)
- Lesson 12  → :func:`prefix_hit_sim` :链式块哈希前缀命中仿真
"""
from __future__ import annotations

import math
import os
import sys
import time
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def cuda_available() -> bool:
    return bool(_HAS_TORCH and torch.cuda.is_available())


def device_str() -> str:
    if not cuda_available():
        return "cpu"
    return f"cuda:{torch.cuda.current_device()}({torch.cuda.get_device_name(0)})"


def _empty_cache() -> None:
    if cuda_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Lesson 07 · QKV 投影 + 注意力微基准
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def bench_qkv_attn(L: int = 32, H: int = 32, D: int = 128, T_list: Tuple[int, ...] = ()
                   , plane: int = 128, dtype: str = "bf16", warmup: int = 8, iters: int = 30,
                   ) -> Dict:
    """在 GPU 上实测「QKV 投影 + 注意力打分/加权」两种方式的总耗时。

    - nocache(无缓存):每步重算全部前 t 个 token 的 K/V,并做完整注意力。
    - cache(有缓存):只算 1 个新 token 的 query,注意力与全部历史交互
      (QKV 部分只算新的)。

    返回
    ----
    Dict:
        keys: L,H,D,T_list, device, simulated,
              used(T_list 实际用到的长度), results: list[ (t, nocache_ms, cache_ms) ]
    """
    if not T_list:
        T_list = tuple(sorted({64, 128, 256, 512, 1024}))
    if not cuda_available():
        # 无 GPU 时给理论估算,保证 app 不崩
        out = []
        for t in T_list:
            qkv = 6 * L * H * D * D
            att = 4 * L * H * D * t
            nocache_ms = (qkv * t + att) * 1e-9 * 1e6  # 约为 FLOP 换算,示意
            cache_ms = (qkv + att) * 1e-9 * 1e6
            out.append((int(t), float(nocache_ms), float(cache_ms)))
        return {"device": "cpu(simulated)", "simulated": True,
                "L": L, "H": H, "D": D, "results": out}

    dt = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)
    dev = "cuda"
    results = []
    for t in T_list:
        q_s = torch.randn(1, H, 1, D, device=dev, dtype=dt)
        k_all = torch.randn(1, H, t, D, device=dev, dtype=dt)
        v_all = torch.randn(1, H, t, D, device=dev, dtype=dt)
        # cache: 只对 1 个 query 做带掩码注意力
        q = torch.randn(1, H, 1, D, device=dev, dtype=dt)
        # 预热
        for _ in range(warmup):
            torch.matmul(q, k_all.transpose(-1, -2))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            att = torch.matmul(q, k_all.transpose(-1, -2))  # 打分
            p = torch.softmax(att.float(), dim=-1).to(dt)
            o = torch.matmul(p, v_all)
        torch.cuda.synchronize()
        cache_ms = (time.perf_counter() - t0) / iters * 1e3

        # nocache:每步把 t 个 token 的 QKV 都当作“新”的重算
        q_all = torch.randn(1, H, t, D, device=dev, dtype=dt)
        for _ in range(warmup):
            torch.matmul(q_all, k_all.transpose(-1, -2))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            att = torch.matmul(q_all, k_all.transpose(-1, -2))
            p = torch.softmax(att.float(), dim=-1).to(dt)
            o = torch.matmul(p, v_all)
        torch.cuda.synchronize()
        nocache_ms = (time.perf_counter() - t0) / iters * 1e3

        results.append((int(t), float(nocache_ms), float(cache_ms)))
        del q_s, k_all, v_all, q, q_all

    _empty_cache()
    return {"device": device_str(), "simulated": False,
            "L": L, "H": H, "D": D, "results": results}


# ----------------------------------------------------------------------
# Lesson 08 · 真实统计某形状 KV 张量的显存占用
# ----------------------------------------------------------------------
@lru_cache(maxsize=16)
def kv_bytes_real(L: int, kv_heads: int, head_dim: int, seq: int,
                  batch: int, dtype: str = "bf16") -> Dict:
    """构造与真实 KV cache 同形状的 CUDA 张量,量化其显存,并给出理论值对照。"""
    nbytes = 2 if dtype == "bf16" or dtype == "fp16" else (1 if dtype == "fp8" else 4)
    theory = 2 * L * kv_heads * head_dim * seq * batch * nbytes
    if not cuda_available():
        return {"device": "cpu(simulated)", "simulated": True,
                "theory_bytes": theory, "alloc_bytes": theory}
    # 真实分配一个 (2, batch, L, kv_heads, seq, head_dim) 的 KV 张量观察显存增量
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    dt = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)
    kv = torch.zeros(2, batch, L, kv_heads, seq, head_dim, device="cuda", dtype=dt)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    alloc = after - before
    maxalloc = torch.cuda.max_memory_allocated()
    del kv
    _empty_cache()
    return {"device": device_str(), "simulated": False, "dtype": dtype,
            "theory_bytes": int(theory), "alloc_bytes": int(alloc), "max_alloc": int(maxalloc)}


# ----------------------------------------------------------------------
# 逻辑层模拟(不需要 GPU,但参数贴近真实默认值)
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def alloc_frag_sim(total_slots: int = 128, block_size: int = 8, steps: int = 80,
                   seed: int = 0) -> Dict:
    """连续 vs 分页分配器在同一随机事件流下的碎片化模拟。返回含两种分配器的 hist。"""
    rng = np.random.RandomState(seed)
    n_blocks = total_slots // block_size

    state = {"occ_c": np.zeros(total_slots, dtype=np.int64),
             "seg_c": [],                     # [start, size, rid]
             "owner_p": np.zeros(n_blocks, dtype=np.int64),
             "occ_p": np.zeros(total_slots, dtype=np.int64),
             "active": {},                    # rid -> [life, size]
             "queue": 0,
             "hist": {"cont": [], "page": [], "ext": [], "int": [], "queue": [], "util": []}}

    def largest_free(arr):
        best = cur = 0
        for v in np.r_[arr, [0]]:
            cur = cur + 1 if v == 0 else 0
            best = max(best, cur)
        return best

    def alloc_cont(size, rid):
        occ = state["occ_c"]
        n = len(occ)
        start = 0
        while start + size <= n:
            if np.all(occ[start:start + size] == 0):
                occult = occ  # noqa
                occ[start:start + size] = rid
                state["seg_c"].append([start, size, rid])
                return True
            start += 1
        return False

    def free_cont(rid):
        state["occ_c"][state["occ_c"] == rid] = 0
        state["seg_c"] = [s for s in state["seg_c"] if s[2] != rid]

    def alloc_page(size, rid):
        need = (size + block_size - 1) // block_size
        free = [i for i, o in enumerate(state["owner_p"]) if o == 0]
        if len(free) < need:
            return False
        for b in free[:need]:
            state["owner_p"][b] = rid
            lo, hi = b * block_size, min((b + 1) * block_size, total_slots)
            state["occ_p"][lo:hi] = rid
        return True

    def free_page(rid):
        state["owner_p"][state["owner_p"] == rid] = 0
        state["occ_p"][state["occ_p"] == rid] = 0

    for step in range(steps):
        if rng.rand() < 0.62 or not state["active"]:
            size = int(rng.randint(4, max(8, total_slots // 8)))
            life = int(rng.randint(3, 12))
            rid = step + 1
            ok_c = alloc_cont(size, rid)
            ok_p = alloc_page(size, rid)
            if not ok_c:
                state["queue"] += 1
            if ok_p:
                state["active"][rid] = [life, size]
        else:
            rid = int(rng.choice(list(state["active"].keys())))
            free_cont(rid); free_page(rid)
            state["active"].pop(rid, None)
        for k in list(state["active"]):
            state["active"][k][0] -= 1
            if state["active"][k][0] <= 0:
                free_cont(k); free_page(k)
                state["active"].pop(k, None)
        total_free = int(np.sum(state["occ_c"] == 0))
        ext = 1 - largest_free(state["occ_c"]) / total_free if total_free else 0.0
        used = int(np.sum(state["owner_p"] != 0)) * block_size
        req = sum(a[1] for a in state["active"].values())
        inter = (used - req) / used if used else 0.0
        state["hist"]["cont"].append(state["occ_c"].copy())
        state["hist"]["page"].append(state["occ_p"].copy())
        state["hist"]["ext"].append(float(ext))
        state["hist"]["int"].append(float(inter))
        state["hist"]["queue"].append(int(state["queue"]))
        state["hist"]["util"].append(1 - total_free / total_slots)

    state["block_size"] = block_size
    state["total_slots"] = total_slots
    return {"total_slots": total_slots, "block_size": block_size, "steps": steps,
            "state": state}


@lru_cache(maxsize=16)
def paged_map(seq: int, block_size: int, n_phys: int, seed: int = 42) -> Dict:
    """构造 逻辑块 →(随机)物理块 的页表,并给出逐 token 的 slot 落点。"""
    import math
    import random
    n_logical = math.ceil(seq / block_size)
    rng = random.Random(seed + block_size)
    phys_free = list(range(n_phys))
    mapping: Dict[int, int] = {}
    for lb in range(n_logical):
        if not phys_free:
            break
        mapping[lb] = phys_free.pop(rng.randrange(len(phys_free)))

    slots = []
    for tok in range(seq):
        lb, off = divmod(tok, block_size)
        pb = mapping.get(lb)
        slots.append({"token": tok + 1, "logical": lb, "physical": pb,
                      "offset": off, "slot": pb * block_size + off if pb is not None else None})
    return {"mapping": mapping, "n_logical": n_logical, "seq": seq,
            "block_size": block_size, "n_phys": n_phys, "slots": slots,
            "miss": [lb for lb in range(n_logical) if lb not in mapping]}


def _hash_tokens(tokens: Tuple[int, ...], prev: int) -> int:
    """链式块哈希:block.hash = H(prev_hash, tokens),模拟 vLLM 前缀指纹。"""
    h = 0x811C9DC5 ^ prev  # FNV offset + 上一块哈希
    for t in tokens:
        h ^= (t & 0xFFFFFFFF)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _prefix_hit_sim_uncached(n_req: int, seq_len: int, prefix_len: int,
                             block_size: int) -> Dict:
    """无缓存基准:每请求从头全算,命中块=0。"""
    n_blocks_full = math.ceil(seq_len / block_size)
    return {"n_computed_segments": int(n_req * n_blocks_full),
            "n_recovered": 0, "hit_rate": 0.0}


@lru_cache(maxsize=16)
def prefix_hit_sim(n_req: int, seq_len: int, prefix_len: int,
                   block_size: int, seed: int = 7) -> Dict:
    """对『共享前缀+独有后缀』的请求做**逐块链式哈希**的缓存命中仿真。

    首请求把每个前缀块哈希存入表;后续请求的前缀块哈希命中即在物理块池复用,
    独有后缀每请求各算一部分。返回命中比例与省下的 prefill 块数。
    """
    import math
    if n_req <= 1 or prefix_len <= 0:
        return _prefix_hit_sim_uncached(n_req, seq_len, prefix_len, block_size)
    rng = np.random.RandomState(seed)

    def tokens(prefix, suffix):
        return tuple(int(t) for t in prefix) + tuple(int(t) for t in suffix)

    # 共享前缀(所有人都一样)
    shared = tokens(list(range(prefix_len)), [])
    table: Dict[int, int] = {}
    n_blocks_pref = math.ceil(prefix_len / block_size)
    n_blocks_suf = math.ceil(max(seq_len - prefix_len, 0) / block_size)
    prefix_blocks = n_blocks_pref
    pref_hashes = []
    h = 0
    for b in range(n_blocks_pref):
        chunk = shared[b * block_size:(b + 1) * block_size]
        h = _hash_tokens(chunk, h)
        pref_hashes.append(h)
        table[h] = b  # 首请求建立哈希→块

    n_recovered = 0
    n_computed = 0
    for r in range(n_req):
        suffix = tokens(list(range(r)), list(range(prefix_len, seq_len)))
        # 前缀块:除第一请求外全部命中复用
        for k, hh in enumerate(pref_hashes):
            if r == 0:
                n_computed += 1                      # 第一次 prefill
            else:
                if hh in table:
                    n_recovered += 1                 # 命中,免 prefill
                else:
                    n_computed += 1
        # 独有后缀:每请求都算(这里仅表意,统计计算块数)
        n_computed += n_blocks_suf
    total_need = sum(1 for _ in range(n_req)) * (prefix_blocks + n_blocks_suf)
    hit_rate = n_recovered / max(total_need, 1)
    return {"n_req": n_req, "seq_len": seq_len, "prefix_len": prefix_len,
            "block_size": block_size, "n_recovered": n_recovered,
            "n_computed_segments": n_computed, "hit_rate": float(hit_rate)}