# -*- coding: utf-8 -*-
import math, time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| SDPA flash enabled:", torch.backends.cuda.flash_sdp_enabled())

# ---------- chunked flash attention (online softmax + tiling, fp32) ----------
def flash_attention_chunked(Q, K, V, block_M=64):
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)
    m = torch.full((B, H, N, 1), float("-inf"), device=Q.device)
    l = torch.zeros((B, H, N, 1), device=Q.device)
    scale = 1.0 / math.sqrt(d)
    for j in range(0, N, block_M):
        Kj = K[:, :, j:j + block_M, :]
        Vj = V[:, :, j:j + block_M, :]
        S = torch.einsum("bhnd,bhmd->bhnm", Q, Kj) * scale
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)
        P = torch.exp(S - m_new)
        l_new = l * torch.exp(m - m_new) + P.sum(dim=-1, keepdim=True)
        O = O * torch.exp(m - m_new) + torch.einsum("bhnm,bhmd->bhnd", P, Vj)
        m = m_new
        l = l_new
    return O / l

B, H, N, d = 2, 4, 256, 64
Q = torch.randn(B, H, N, d, device=dev)
K = torch.randn(B, H, N, d, device=dev)
V = torch.randn(B, H, N, d, device=dev)

ref = F.scaled_dot_product_attention(Q, K, V)  # (B,H,N,d), masked out (no causal)
out = flash_attention_chunked(Q, K, V, block_M=64)
err = (out - ref).abs().max().item()
print("chunked vs SDPA max abs err =", err)

# ---------- naive attention (materialize full scores) for correctness ----------
def naive_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.einsum("bhnd,bhmd->bhnm", Q, K) * scale
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", P, V)

naive = naive_attention(Q, K, V)
print("naive vs SDPA max abs err =", (naive - ref).abs().max().item())

# ---------- timing at larger N ----------
N2 = 2048
Q2 = torch.randn(1, 8, N2, 64, device=dev)
K2 = torch.randn(1, 8, N2, 64, device=dev)
V2 = torch.randn(1, 8, N2, 64, device=dev)

def bench(fn, *a, iters=5):
    for _ in range(2):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

t_eager = bench(naive_attention, Q2, K2, V2)
t_sdpa = bench(lambda q, k, v: F.scaled_dot_product_attention(q, k, v), Q2, K2, V2)
t_flash = bench(flash_attention_chunked, Q2, K2, V2, 128)
print(f"N={N2} d=64: naive={t_eager:.2f}ms  sdpa={t_sdpa:.2f}ms  chunked_flash={t_flash:.2f}ms")

# ---------- FLOPs / memory formulas sanity ----------
def attn_flops(N, H, d):
    return 4 * N * N * H * d  # dominant 4N^2 d_model
print("attention FLOPs @N=2048,H=8,d=64:", f"{attn_flops(2048, 8, 64):.3e}", "=", attn_flops(2048, 8, 64) / 1e9, "GFLOP")
