# 第 7 章 · Attention 内核 — 参考答案（ch07，第 42–47 课）

> **参考答案 · 建议先自己动手再看。** 本文件与仓库 `exercises/ch07/` 下 42–47 号 notebook 一一对应。
> 代码假设与你 notebook 中**已运行过的定义一致**才可独立运行（会标注依赖）；术语保留英文，答案用中文。

---

## 第 42 课 · FlashAttention 原理

> 对应 `exercises/ch07/42_flash_attention_principle.ipynb`。依赖：`naive_attention` / `hbm_bytes` / `mem_bytes` / `flash_attention_chunked` / `bench_ms`。

### 练习 1：hbm_bytes 的 dtype_bytes 从 2 改成 4（fp32），重算访存
**思路**：`dtype_bytes` 在 naive 与 flash 里以同样倍数缩放，因此**节省倍数不变**（比例只由 N 决定），但绝对字节翻倍。

```python
for dtb in [2, 4]:
    nv, fl = hbm_bytes(4096, 8, 64, dtb)
    print(f"dtype={dtb}B: naive={nv/1e6:8.1f}MB  flash={fl/1e6:8.1f}MB  节省×{nv/fl:.1f}")
```
**预期结果**：naive/flash 节省倍数在两种 dtype 下相同（因为 `dtype_bytes` 在分子分母都被乘），N=4096 时约几十倍；但 fp32 的绝对访存是 fp16 的 2 倍，峰值显存（`mem_bytes`）压力也翻倍。结论：dtype 不影响"节省倍数"，只影响绝对量。

### 练习 2：推导 naive 里 S/P 为什么占 4 次 N² 访问
**思路**：`S` 和 `P` 各写一次 + 各读一次，共 4 次对 `H·N²` 元素的往返。

```
S = QK^T      : 写 S  (H·N²)                          —— 1 次
softmax(S)    : 读 S  (H·N²)                          —— 1 次
P = softmax   : 写 P  (H·N²)                          —— 1 次
O = P V       : 读 P  (H·N²)                          —— 1 次
合计 = 4 · H · N² · dtype_bytes
```
单次字节数：`4·H·N²·dtb`（对照 `hbm_bytes` 里的 `+4*H*N*N*dtype_bytes`）。这 4 次是 naive 比 flash 多出的 O(N²) 访存来源，flash 把 S/P 留在片上因此省掉这 4·N² 往返。

### 练习 3：把 block_M 改成 16/32/64/128，重跑数值验证，看误差与耗时
**思路**：block 越大则沿 K/V 方向扫描次数 `N/block_M` 越少（越省时），但分块用 running max/sum 的 online softmax 数值始终等价。

```python
B, H, N, d = 1, 4, 256, 64
Q = torch.randn(B, H, N, d); K = torch.randn(B, H, N, d); V = torch.randn(B, H, N, d)
ref = naive_attention(Q, K, V)
for bm in [16, 32, 64, 128]:
    O = flash_attention_chunked(Q, K, V, block_M=bm)
    err = (O - ref).abs().max().item()
    t = bench_ms(lambda: flash_attention_chunked(Q, K, V, block_M=bm))
    print(f"block_M={bm:3d}: 误差={err:.2e}  耗时={t:.4f}ms")
```
**预期结果**：四种 block 误差都在 ~1e-6~1e-5 量级（全部正确）；耗时随 block 增大而下降（扫描次数从 16 降到 2）。block 过大（超过片上容量）时性能反而不升，64 附近是常见甜点。

### 练习 4：给 flash_attention_chunked 加下三角因果掩码，对比 SDPA(is_causal=True)
**思路**：因果性要求 query `i` 只看 key `j ≤ i`；对每个块把未来位置的 `S` 置为 `-inf`。

```python
import torch.nn.functional as F
def flash_causal(Q, K, V, block_M=64):
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q); scale = 1.0/math.sqrt(d)
    m = torch.full((B,H,N,1), -math.inf); l = torch.zeros_like(m)
    for j in range(0, N, block_M):
        S = torch.einsum("bhnd,bhmd->bhnm", Q, K[:,:,j:j+block_M,:]) * scale
        col = j + torch.arange(S.shape[-1])                  # 全局 key 列号
        mask = (torch.arange(N)[:, None] >= col[None, :]).to(Q.device)   # i>=j
        S = S.masked_fill(~mask, -math.inf)
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)
        P = torch.exp(S - m_new)
        l = l*torch.exp(m-m_new) + P.sum(dim=-1, keepdim=True)
        O = O*torch.exp(m-m_new) + torch.einsum("bhnm,bhmd->bhnd", P, V[:,:,j:j+block_M,:])
        m, l = m_new, l
    return O / l
O = flash_causal(Q, K, V)
ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(f"flash-causal vs SDPA(is_causal=True) 最大误差 = {(O-ref).abs().max().item():.2e}")
```
**预期结果**：误差 ~1e-6，与官方 SDPA 的因果版一致。注意 `masked_fill(-inf)` 后 `P=exp` 对 -inf 得 0，自动不参与 softmax。

### 练习 5：在 app_42 加"节省倍数"指标卡，观察随 N 的增长
**思路**：用 `hbm_bytes` 的 naive/flash 比值作为"节省倍数"指标，对多个 N 求值连线。

```python
# streamlit / 直接用数据画:
Ns = [512, 1024, 2048, 4096, 8192]
ratio = [ hbm_bytes(n, 8, 64)[0] / hbm_bytes(n, 8, 64)[1] for n in Ns ]
for n, r in zip(Ns, ratio):
    print(f"N={n:>5}: 节省倍数 ≈ {r:.1f}x")
# st.line_chart({"N": Ns, "节省倍数": ratio})
```
**预期结果**：节省倍数随 N 从 5× 一路涨到几十×（naive O(N²) vs flash O(N)），曲线近似线性上升——N 越大，FlashAttention 越值得。

---

## 第 43 课 · 在线 Softmax

> 对应 `exercises/ch07/43_online_softmax.ipynb`。依赖：`softmax_naive` / `softmax_twopass` / `softmax_online` / `online_trace` / `softmax_online_batched`。

### 练习 1：online_trace 改成记录每步校正系数 e^(m-m_new)
**思路**：`e^(m_old - m_new)` 是新 max 出现时旧块贡献要被压缩的倍率；m_new ≥ m_old 故它 ≤ 1。

```python
def trace_corr(x, block):
    m, l = -math.inf, 0.0; corr = []
    for i in range(0, len(x), block):
        xb = x[i:i+block]; m_new = max(m, float(xb.max()))
        corr.append(math.exp(m - m_new)); l = l*corr[-1] + float(torch.exp(xb - m_new).sum()); m = m_new
    return corr, l, m
corr, l, m = trace_corr(torch.tensor([1.0, 2.0, 5.0, 3.0]), 2)
print(f"校正系数序列 = {[round(c,4) for c in corr]}  (首块=1;出现新高 max 时 <1)")
```
**预期结果**：第一个块校正系数 `= e^0 = 1`（max 未变）；当某块冒出一个更大的 max（如本例块2 max=5 覆盖块1的 2）时，校正系数 `e^(2-5)≈0.0498`，把旧的 `l` 大幅压缩。系数序列随"何时出现新高 max"呈现骤降。

### 练习 2：给 softmax_online_batched 加 bf16 版本，比较误差
**思路**：bf16 只有 8 位尾数，running max/sum 累积误差明显大于 fp32。

```python
def softmax_online_bf16(x, block):
    x = x.half()
    m = torch.full((x.shape[0],1), -torch.inf, dtype=x.dtype)
    l = torch.zeros_like(m)
    for i in range(0, x.shape[1], block):
        xb = x[:, i:i+block]
        m_new = torch.maximum(m, xb.max(dim=-1, keepdim=True).values)
        l = l*torch.exp(m-m_new).half() + torch.exp(xb-m_new).sum(dim=-1, keepdim=True).half()
        m = m_new
    return torch.exp(x-m) / l
xf = torch.randn(64, 4096)*5
full = torch.softmax(xf, -1)
print("bf16 误差:", (softmax_online_bf16(xf, 64).float() - full).abs().max().item())
print("fp32 误差:", (softmax_online_batched(xf, 64) - full).abs().max().item())
```
**预期结果**：fp32 误差 ~1e-6；bf16 误差升到 ~1e-3 量级。正因为 bf16 尾数不足，工程上 online softmax 的 running `m`/`l` 常用 fp32 维护（FA2 中 QK 累加也保 fp32）。

### 练习 3：在 2D 矩阵上实现逐行 online softmax，并用 torch.softmax(dim=-1) 验证
**思路**：`softmax_online_batched(x, block)` 已是逐行实现，只需对随机矩阵取不同 block 验证。

```python
def online_2d(x, block):
    B, N = x.shape
    m = torch.full((B,1), -torch.inf, device=x.device); l = torch.zeros_like(m)
    for i in range(0, N, block):
        xb = x[:, i:i+block]
        m_new = torch.maximum(m, xb.max(dim=-1, keepdim=True).values)
        l = l*torch.exp(m-m_new) + torch.exp(xb-m_new).sum(dim=-1, keepdim=True); m = m_new
    return torch.exp(x-m) / l
x = torch.randn(64, 4096)*5
for blk in [64, 256, 2048]:
    ok = torch.allclose(torch.softmax(x, -1), online_2d(x, blk), atol=1e-5, rtol=1e-5)
    print(f"block={blk}: 与 torch.softmax(dim=-1) allclose = {ok}")
```
**预期结果**：任意 block 都 `allclose`。含义：online softmax 对每行各自维护 running 态，天然按行并行，不必跨行通信。

### 练习 4：推导为什么 m 单调不减、l 在校正时可能下降，并给数值例子
**思路**：`m = max(已见 max)`，新 max ≥ 旧 max 故单调不减；而 `l = Σ e^(x_j - m)`，m 变大时每一项都乘上 `e^(m_old-m_new) < 1`。

```python
# 旧块在后一块出现大 max 时被压缩 → l 可能变小时刻
x = torch.tensor([1.0, 1.0, 1.0, 10.0])
m, l = -math.inf, 0.0
for i in range(0, 4, 2):
    xb = x[i:i+2]; m_new = max(m, float(xb.max()))
    l = l*math.exp(m - m_new) + float(torch.exp(xb - m_new).sum()); m = m_new
    print(f"处理完前 {i+2} 个元素: m={m}, l={l:.4f}")
```
**预期结果**：处理 `[1,1]` 后 `m=1, l=3`；处理下一个 `[1,10]` 时 `m_new=10`，`l = 3·e^(1-10) + (e^-9 + e^0) ≈ 1.0005 < 3`——**l 从 3 掉到约 1**。原因：校正把旧三项全部除以 e^9，而新最大值自己的贡献才 1。这是 online softmax 的正确性而非 bug：`l` 只在"当前 m"下才有意义。

### 练习 5：读论文 Algorithm 3 / Theorem 1，证明 l_V = Σ e^(x_j − m_V)
**思路**：用归纳法证明 running sum 恒等于"减当前 max 后的 softmax 分母"。
**证明（自己的话）**：设 `m_V = max_j x_j`（全局最大值），online 过程维护的 `l` 增量地满足 `l = Σ_j e^(x_j - m_current)`。归纳：初始 `l=0,m=-inf` 为空集成立；加入一块时用 `m_new` 压缩旧项得 `l·e^(m-m_new)`，恰把旧每一位 `e^(x_j-m)` 换成 `e^(x_j-m_new)`（因 `e^(x_j-m)·e^(m-m_new)=e^(x_j-m_new)`），再加新块。扫描结束时 `m=m_V`，故 `l = Σ_j e^(x_j - m_V)`，与定义一致。这就是 Theorem 1 的直观内核。

---

## 第 44 课 · Attention Kernel

> 对应 `exercises/ch07/44_attention_kernel.ipynb`。依赖：`naive_attention` / `flash_attention_chunked` / `triton_fa_fwd` / `triton_flash_attention` / `bench`。

### 练习 1：block_M 改成 32/128，观察耗时与精度
**思路**：同第 42 课练习 3——块越大扫描数越少越省时，数值始终等价。

```python
for bm in [32, 128]:
    O = flash_attention_chunked(Q, K, V, block_M=bm)
    err = (O - naive_attention(Q, K, V)).abs().max().item()
    t = bench(lambda: flash_attention_chunked(Q, K, V, block_M=bm))
    print(f"block_M={bm:3d}: 误差={err:.2e}  耗时={t:.3f}ms")
```
**预期结果**：两者误差都在 ~1e-6 量级（都正确）；`block_M=128` 比 `32` 更快（扫描数 8 vs 2）。但超过片上容量后不会再快，需实测甜点。

### 练习 2：加下三角因果掩码，对比 SDPA(is_causal=True)
**思路**：同 42 课练习 4，在分块循环里对 `S` 按 `i≥j` 置 `-inf`。

```python
import torch.nn.functional as F
def flash_causal(Q, K, V, block_M=64):
    B,H,N,d = Q.shape; scale = 1.0/math.sqrt(d)
    O = torch.zeros_like(Q)
    m = torch.full((B,H,N,1), -math.inf); l = torch.zeros_like(m)
    for j in range(0,N,block_M):
        S = torch.einsum("bhnd,bhmd->bhnm", Q, K[:,:,j:j+block_M,:])*scale
        col = j + torch.arange(S.shape[-1])
        mask = (torch.arange(N)[:,None] >= col[None,:]).to(Q.device)
        S = S.masked_fill(~mask, -math.inf)
        m_new = torch.maximum(m, S.max(-1,keepdim=True).values)
        P = torch.exp(S - m_new)
        l = l*torch.exp(m-m_new) + P.sum(dim=-1,keepdim=True)
        O = O*torch.exp(m-m_new) + torch.einsum("bhnm,bhmd->bhnd", P, V[:,:,j:j+block_M,:])
        m, l = m_new, l
    return O/l
O = flash_causal(Q, K, V)
print("与 SDPA(is_causal=True) 最大误差:", (O - F.scaled_dot_product_attention(Q,K,V,is_causal=True)).abs().max().item())
```
**预期结果**：误差 ~1e-6，与官方因果 SDPA 一致。未来 token 被置 -inf 后经 exp 得 0，不污染归一化。

### 练习 3：triton_fa_fwd 的 BM/BN 改成 32 或 128，重跑误差与耗时
**思路**：`BM/BN` 是 kernel 块的 `tl.constexpr`；改小则块多、launch/调度开销略增，改大则可能超出寄存器预算。

```python
for bm, bn in [(32,32),(64,64),(128,128)]:
    O = triton_flash_attention(Q, K, V, bm=bm, bn=bn)
    err = (O.float() - ref.float()).abs().max().item()
    t = bench(lambda: triton_flash_attention(Q, K, V, bm=bm, bn=bn))
    print(f"BM={bm} BN={bn}: 误差={err:.2e}  耗时={t:.3f}ms")
```
**预期结果**：三种块输出的结果一致（误差 ~1e-3 内）；耗时通常 `64` 最优，`32` 因块碎片稍慢、`128` 若超 shared memory/寄存器也会慢。块大小是"片上容量 vs 调度开销"的平衡。

### 练习 4：给 triton_fa_fwd 加因果掩码
**思路**：在 kernel 循环里用 `offs_m[:,None] >= offs_n[None,:]` 构造下三角，把未来位置置 `-inf`。

```python
# 在 triton_fa_fwd 里、tl.dot(q,k) 之后:
offs_n = n_off + tl.arange(0, BN)                    # 本块 key 列号
causal = offs_m[:, None] >= offs_n[None, :]          # 因果: i >= j
qk = tl.dot(q.to(tl.float16), k)                     # (BM, BN)
qk = tl.where(causal, qk, float("-inf"))             # 未来 → -inf
# 注意:因果下每行块扫到末尾即可提前 break(非必需)
```
**预期结果**：未来位置原子被置 `-inf`，`tl.exp` 后为 0；与 `torch` 因果 attention 结果一致。相比非因果版仅多一条 `tl.where`，几乎不增加开销。

### 练习 5：在 app_44 加"误差随 block 大小变化"折线图
**思路**：对角标定几个 block，分别算分块实现相对 naive 的误差和耗时，画折线。

```python
blocks = [16, 32, 64, 128]
errs  = [float((flash_attention_chunked(Q,K,V,block_M=b).float()-ref).abs().max()) for b in blocks]
ts    = [bench(lambda: flash_attention_chunked(Q,K,V,block_M=b)) for b in blocks]
for b,e,t in zip(blocks,errs,ts): print(f"block={b:3d}: err={e:.2e}  time={t:.3f}ms")
# st.line_chart({"误差": errs, "耗时": [t*1e3 for t in ts]})
```
**预期结果**：误差在 block 增大时基本持平（都在 ~1e-6），耗时则随 block 增大单调下降——直观展示"分块无精度损失、却换来访存节省"。

---

## 第 45 课 · PagedAttention

> 对应 `exercises/ch07/45_paged_attention.ipynb`。依赖：`paged_attention` / `paged_attn` / `contiguous_attn` / `mem_contiguous` / `mem_paged`，以及 `phys_k` / `phys_v` / `block_table`。

### 练习 1：total_blocks 调大、请求数调多，观察分页节省比例趋势
**思路**：连续分配按"最长请求"预留（含空洞），分页按实际用量；请求越多、长短越不齐，空洞越多，节省越大。

```python
rng = np.random.default_rng(7); d, bs = 8, 16
def savings(num_req):
    req_blocks = rng.integers(2, 20, num_req)               # 每请求 2..19 块
    mc = req_blocks.max()*bs*num_req*d*4                    # 连续:都按最长预留
    mp = int(req_blocks.sum())*bs*d*4                       # 分页:按实际
    return 1 - mp/mc
for n in [10, 100, 1000]:
    print(f"请求数={n:>4}: 分页节省 ≈ {savings(n):.1%}  (≈ 1 - 平均/最长)")
```
**预期结果**：每个请求等到多个块时 `平均块数/最长块数` 越低，节省越高（例：平均 10 / 最长 19 ⇒ 约 47%）。请求越多、长短分布越尖，总节省越接近 `1 - 平均/最长`，分页收益越显著。

### 练习 2：给 paged_attention 换成多查询（批量 Q），返回完整 (Nq, d)
**思路**：把单 `q:(d,)` 推广为 `q:(Nq,d)`，打分变矩阵乘法、softmax 沿 key 维。

```python
def paged_attn_multi(q, blocks, K, V, d):
    Kc = K[blocks].reshape(-1, d)      # (tokens, d)
    Vc = V[blocks].reshape(-1, d)
    S = Kc @ q.T / math.sqrt(d)        # (tokens, Nq)
    P = torch.softmax(S, dim=0)        # 对每个 query 在 key 上归一
    return P.T @ Vc                    # (Nq, d)
q = torch.randn(4, d)                  # 4 个 query
o = paged_attn_multi(q, block_table[0], phys_k, phys_v, d)
print(f"多个查询输出 shape = {tuple(o.shape)}  <- (Nq=4, d)")
```
**预期结果**：返回 `(Nq, d)`，每个 query 对同一组 KV 打分并加权求和；等价于把单查询循环化的矩阵运算。

### 练习 3：手动构造含空洞的物理池，验证 K[blocks] 仍能正确收集
**思路**：`K[blocks]` 是 PyTorch advanced indexing，按"逻辑块→物理块号"收集，物理块是否连续/含空洞无影响。

```python
pool = torch.randint(0, 10, (10, 2, d)).float()      # 10 个物理块
blocks = torch.tensor([7, 2, 9])                     # 逻辑连续但物理跳(有空洞)
K_cont = pool[blocks].reshape(-1, d)                 # 按 [7,2,9] 收集
ref = torch.cat([pool[7], pool[2], pool[9]]).reshape(-1, d)
print("分页收集 == 手动拼接:", torch.equal(K_cont, ref))
```
**预期结果**：`True`。advanced indexing 保证取第 7、2、9 号物理块再拼接，与物理块是否相邻/中间是否有空洞无关。

### 练习 4：PagedAttention 的 kernel 为什么要把"收集"与"attention"融合
**思路**：若不融合，需要先按 block table gather 出连续的 KV（产生一个中间大张量），再跑 attention。
**结论**：把"按 `block_table` 定位物理块 → 读取 KV → 在线 attention"融进一个 kernel，让**非连续读取直接在片上/寄存器内完成**，省掉中间 gather 张量的 HBM 写入与再读取（省显存 + 省带宽）。这与第 42 课"不落中间大张量"是同一思想——PagedAttention 的 vLLM 实现（FlattenedKVCache）正是如此。

### 练习 5：读论文 §4.4，解释 copy-on-write 如何共享 prompt 的物理块
**思路**：并行采样时多个 sequence 从同一 prompt 延续，前缀 token 的 KV 完全相同。
**结论**：这些 sequence 让它们的**前缀共用同一组物理块**（不复制）；每个 sequence 只在**自己新增的 token**上分配新物理块。当某个 sequence 需要改写一个共享块时才触发 copy（fork 出新块），未改写的共享块继续复用。这样 prompt 部分无论多少并行序列都只存一份物理 KV，显存大幅节省。代价是 block table 需要记录共享关系与引用计数。

---

## 第 46 课 · Attention 性能

> 对应 `exercises/ch07/46_attention_perf.ipynb`。依赖：`naive_attention` / `flash_attention_chunked` / `bench` / `attention_ai` / `flash_ai`。

### 练习 1：把 bench 的 iters 调大（如 20），比较计时噪声对结论的影响
**思路**：iters 越大均值越稳；三实现的相对排序一般不变。

```python
for it in [5, 20]:
    tn = bench(naive_attention, Q, K, V, iters=it)
    tf = bench(flash_attention_chunked, Q, K, V, iters=it)
    print(f"iters={it}: naive={tn:.4f}ms  flash={tf:.4f}ms  naive/flash={tn/tf:.1f}x")
```
**预期结果**：iters=20 时波动更小；naive 明显慢于 flash 的相对关系稳定（同一 N 下 flash 省 N² 访存）。仅当差异本来就小时，加大 iters 才能让排序可信。

### 练习 2：用 H=16, d=64 重跑对比，观察头数对耗时的影响
**思路**：H 影响放在 `N²` 前的并行度系数，多头时 flash 的分块优势更明显。

```python
for (H, d) in [(8, 64), (16, 64)]:
    Q = torch.randn(1, H, 256, d); K = torch.randn_like(Q); V = torch.randn_like(Q)
    tn = bench(naive_attention, Q, K, V)
    tf = bench(flash_attention_chunked, Q, K, V)
    print(f"H={H}, d={d}: naive={tn:.4f}ms  flash={tf:.4f}ms  比值={tn/tf:.1f}x")
```
**预期结果**：H 增加 → `H·N²` 总访存增加，两者绝对时间上升，但 naive 的 N² 项更快膨胀，naive/flash 比值可能进一步拉大。头数不改变"flash 省 N²"这一本质。

### 练习 3：推导 attention 的算术强度(FLOPs/Byte)，标到 roofline 上
**思路**：算术强度 = FLOPs ÷ 访存字节；naive 因 S/P 的 N² 访存拖低强度，flash 去掉 N² 项后大幅升高。

```python
def ai(N, H, d, dtb=2, flash=True):
    fl = 4*N*N*H*d
    by = (4*N*H*d)*dtb if flash else (4*N*H*d + 2*H*N*N)*dtb
    return fl/by
N, H, d = 4096, 32, 128
ai_n = ai(N,H,d,flash=False); ai_f = ai(N,H,d,flash=True)
print(f"naive AI={ai_n:.0f}   flash AI={ai_f:.0f}  (N={N})")
for name, r in {"A100(BF16)":156, "H100(BF16)":295}.items():
    print(f"  {name} ridge={r}: naive={'mem' if ai_n<r else 'compute'}-bound, flash={'mem' if ai_f<r else 'compute'}-bound")
```
**预期结果**：naive 的 AI 较低（受 N² 访存拖累，易落回 memory-bound 侧）；flash 去掉 S/P 后 AI 高一个量级，在长序列下更接近 compute-bound。roofline 图上对应"flash 点右移、翻过 ridge"。

### 练习 4：在 app_46 加"随 d 变化"的扫描维度
**思路**：对一组 `d` 分别测三实现耗时，画 d–耗时曲线。

```python
import torch
for d_ in [32, 64, 128, 256]:
    Q = torch.randn(1, 16, 512, d_); K = torch.randn_like(Q); V = torch.randn_like(Q)
    tn = bench(naive_attention, Q, K, V); tf = bench(flash_attention_chunked, Q, K, V)
    print(f"d={d_:3d}: naive={tn:.3f}ms  flash={tf:.3f}ms  naive/flash={tn/tf:.1f}x")
```
**预期结果**：d 增大 → 每个元素访存字节上升、N² 分量相对下降 → 算术强度升高、更接近 compute-bound；naive/flash 差距随 d 增大而收窄（N² 项占比下降）。曲线揭示 d 是决定"访存受限程度"的关键参数。

### 练习 5：对比 decode(batch=1) 与 prefill(长 N) 的算术强度，解释为何 prefill 更 compute-bound
**思路**：prefill FLOPs ∝ N²、访存 ∝ N ⇒ AI ∝ N；decode 每步 N=1 ⇒ FLOPs ∝ 1、访存 ∝ d ⇒ AI 极小。

```python
print(f"prefill(N=4096) AI = {attention_ai(4096, 1, 128):.0f} FLOPs/Byte")
print(f"decode (N=1)    AI = {attention_ai(1, 1, 128):.3f} FLOPs/Byte")
```
**预期结果**：prefill 的 AI 达数千（远超任何 GPU sheet，明显 compute-bound）；decode 的 AI 不足个位数（memory-bound，带宽受限）。这正是"prefill 阶段由算力墙决定、decode 阶段由带宽墙决定"的量化依据，也是长 prompt 首字比单 token 续写"更贴算力"的原因。

---

## 第 47 课 · Attention 后端选择

> 对应 `exercises/ch07/47_attn_backend_select.ipynb`。依赖：`select_backend` / `BACKENDS` 能力表 / `arch_ok`。

### 练习 1：给 select_backend 增加 need_sliding 参数
**思路**：在能力表加一列 `sliding`，选择函数加过滤分支（类似 `need_paged`）。

```python
# BACKENDS 每个条目加 1 个布尔:…, sliding)
BACKENDS = {
    "FLASH_ATTN":  ("CUDA","sm_80",True,True, True),   # 末位 sliding
    "FLASHINFER":  ("CUDA","sm_80",True,True, True),
    "TRITON_ATTN": ("CUDA/ROCm","sm_80",True,True, True),
    "TORCH_SDPA":  ("CUDA/CPU","不限",True,False, False),  # 无原生 sliding
    "MATH":        ("全平台","不限",True,False, False),
    "ROCM_FLASH":  ("ROCm","不限",True,True, True),
}
def select_backend(platform, arch, need_causal=True, need_paged=False, need_sliding=False):
    for name, (plat, min_arch, causal, paged, sliding) in BACKENDS.items():
        if need_sliding and not sliding: continue
        ...
print(select_backend("CUDA", "sm_90", need_causal=True, need_paged=True, need_sliding=True)[0])
```
**预期结果**：勾选 `need_sliding` 后，`TORCH_SDPA`/`MATH`（无 sliding window）被过滤，候选只剩 FA2/FlashInfer/Triton 等原生支持 sliding window 的后端。

### 练习 2：架构列表换成实际 GPU 型号（RTX 5060 = sm_120）
**思路**：`arch_ok` 比较的是 `sm_` 后的数字，`sm_120 ≥ sm_80` 故 FA2 等全部可用。

```python
# GPU 型号→架构映射示例:{RTX 5060: sm_120, RTX 4090: sm_89, A100: sm_80, T4: sm_75}
chosen, chain = select_backend("CUDA", "sm_120", need_causal=True, need_paged=True)
print(f"RTX 5060(sm_120): 默认后端 = {chosen}  候选链 = {chain}")
```
**预期结果**：默认选中高优的 `FLASH_ATTN`（sm_80 ≤ sm_120 通过架构过滤），候选链含 FA2/FlashInfer。若换 T4(sm_75)，FA2(sm_80) 会被过滤，退回 SDPA/MATH。

### 练习 3：能力表加 alibi 支持列，观察功能过滤
**思路**：`alibi` 是 ALiBi 位置编码相关；加一列并在过滤里按 `need_alibi` 淘汰不支持的后端。

```python
# BACKENDS 再加 1 列 alibi;过滤时
def select(platform, arch, need_causal=True, need_paged=False, need_alibi=False):
    for name, (plat, min_arch, causal, paged, sliding, alibi) in BACKENDS.items():
        if need_alibi and not alibi: continue
        ...
# 只保留 alibi 支持的后端(FA2/FlashInfer 通常支持)
```
**预期结果**：勾选 alibi 后，MATH/SDPA 等不支持 ALiBi 的被裁掉，候选收窄到支持 alibi 的高性能后端——功能过滤逐列生效，选择逻辑与 causal/paged/sliding 完全同构。

### 练习 4：在 app_47 加"对比两个平台默认后端"的双栏视图
**思路**：用 `st.columns(2)` 并排渲染两个平台各自的选择结果。

```python
col1, col2 = st.columns(2)
with col1:
    st.subheader("CUDA sm_90"); c1,_ = select_backend("CUDA","sm_90",True,True)
    st.metric("默认后端", c1)
with col2:
    st.subheader("ROCm (AMD)"); c2,_ = select_backend("ROCm","不限",True,True)
    st.metric("默认后端", c2)
```
**预期结果**：双栏分别展示 CUDA 平台的默认（FLASH_ATTN/FLASHINFER）与 AMD 平台的默认（ROCM_FLASH），便于横向比较平台差异。

### 练习 5：读 vllm/v1/attention/selector.py 的 AttentionSelectorConfig，列出功能字段
**思路**：查源码（open-ended，以实际为准）。
**预期字段**：`AttentionSelectorConfig` 大致包含 `is_encoder_decoder_model`、`is_encoder_only`、`num_heads` / `num_kv_heads`、`max_model_len`、`block_size`、`paged_attn`、`causal` / `enable_cuda_graph`、`alibi_slopes`、`sliding_window` 等布尔/规格字段。它们被喂给选择逻辑，决定最终选 FA2 / FlashInfer / SDPA / Triton 等——正是本课 `BACKENDS` 能力表的源码版。建议在源码里 `grep "class AttentionSelectorConfig"` 精确定位并逐一对照。

---