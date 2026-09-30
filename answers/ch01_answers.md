# 第 1 章 · vLLM 推理与分词基础（ch01）参考答案

> 参考答案，建议先自己动手再看。
>
> 对应笔记本路径：`exercises/ch01/01_token_and_tokenizer.ipynb`、`02_autoregressive_generation.ipynb`、`03_sampling_strategies.ipynb`、`04_transformer_quickstart.ipynb`、`05_prefill_vs_decode.ipynb`、`06_toy_inference_engine.ipynb`。
>
> 以下代码依赖 notebook 内已定义的函数与全局变量；在同一个 notebook 里把这些代码单元接着运行即可。

## 第 01 课 · Token 与分词器（01_token_and_tokenizer）

### 练习 1 · BPE 合并规则回放验证

一句话思路：用训练好的 `merges` 对语料里每个词重新套用 `apply_bpe`，再去掉 `</w>` 拼接，应与原文完全一致，证明 BPE 无损。

```python
words = ["机器翻译", "学习效率", "翻译机器", "机器学习"]
ok_all = True
for w in words:
    pieces = apply_bpe(w, merges)
    restored = "".join(p for p in pieces if p != "</w>")
    match = restored == w
    ok_all = ok_all and match
    print(f"「{w}」 -> {pieces} -> 还原「{restored}」 {'OK' if match else 'FAIL'}")
print(f"全部无损还原: {ok_all}")
```

预期结果：4 个词全部还原为原文，`全部无损还原: True`。

### 练习 2 · inverse_vocab 与 decode(ids)

一句话思路：由 `vocab` 建立 id→token 逆映射 `inverse_vocab`，写 `decode` 把 id 序列拼回文本，再验证 `tokenize_to_ids(decode(ids)) == ids` 成立（限词表内词）。

```python
inverse_vocab = {vid: tok for tok, vid in vocab.items()}
def decode(ids):
    return "".join(inverse_vocab[i] for i in ids)

ids = tokenize_to_ids("你好", vocab)
text = decode(ids)
reids = tokenize_to_ids(text, vocab)
print(f"ids={ids}, decode='{text}', 再 tokenize={reids}, 一致={ids==reids}")
```

预期结果：`decode` 还原「你好」，重编码 ids 与原来完全一致，`一致=True`。

### 练习 3 · 英文语料观察空格保留

一句话思路：把 corpus 换成英文词，`char_tokenize` 会保留空格字符成为独立 token，与 GPT-2 用 `Ġ` 表示空格是同样的"空格单独成 token"思想。

```python
en_corpus = ["hello world", "deep learning", "hello python"]
en_vocab = build_vocab([char_tokenize(s) for s in en_corpus])
print("英文词表含空格? ", " " in en_vocab)
toks = char_tokenize("deep hello")
ids = tokenize_to_ids("deep hello", en_vocab)
print(f"空格成独立 token: {toks}; ids={ids}")
print("提示: 真实 GPT-2 词表里空格写作 'Ġ' 这个特殊记号。")
```

预期结果：`' '` 这个词出现在 `en_vocab` 里，英文句子 token 数 = 含空格的字符数，空格被当独立 token 保留。

### 练习 4 · 合并次数 → token 数曲线

一句话思路：固定同一个中文句子，分别用 `train_bpe(corpus2, num_merges=k)` 训练不同 k，用 `apply_bpe` 对测试词编码，统计 token 数随合并次数上升而下降。

```python
test_word = "人工智能"
k_list = [0, 1, 3, 5, 8]
for k in k_list:
    m, _ = train_bpe(corpus2, num_merges=k)
    n_tok = len(apply_bpe(test_word, m))
    print(f"k={k}: token 数 = {n_tok}")
```

预期结果：k 越大 token 数越少（如 5→4→约2），合并次数与 token 数成单调下降的曲线关系，可用 matplotlib 画图。

---

## 第 02 课 · 自回归生成（02_autoregressive_generation）

### 练习 1 · n 从 2 改 3（trigram）

一句话思路：`build_ngram(corpus, n=3)` 用最近 2 个字做上下文，条件更具体、预测更有依据，但更长上下文在语料里更易被耗尽而提前断供。

```python
model3 = build_ngram(corpus, n=3)
out3, path3 = generate_ngram(model3, n=3, start="我喜欢", steps=8)
print("trigram:", "".join(out3))
out2, _ = generate_ngram(model, n=2, start="我喜欢", steps=8)
print("bigram :", "".join(out2))
print("trigram 提前断供: ", len(path3) < 8)
```

预期结果：trigram 续写得更高频短语，但常因上下文在语料中无后继而比 bigram 更早 `break` 断供。

### 练习 2 · add-1 拉普拉斯平滑

一句话思路：把 `next_candidates` 里每个候选计数加 1（分子 `count+1`，分母 `total+可见候选数`），让未见上下文也有非零概率而不是返回空。

```python
def next_smooth(model, context, n):
    ctx = tuple(context[-(n-1):]) if len(context) >= n-1 else tuple(context)
    counts = model.get(ctx, Counter())
    if not counts:
        return [], []  # 完全没见过的上下文仍无法给候选，可退化为回退到 n-1
    toks = list(counts.keys())
    probs = [(counts[t] + 1) / (sum(counts.values()) + len(toks)) for t in toks]
    return toks, probs
```

预期结果：从未见过的上下文不再返回空列表，每个已见候选都有 `(c+1)/∑(c+1)` 的平滑概率，缓解断供。

### 练习 3 · 语料外起点导致提前 break

一句话思路：`count_forward` 里一旦 `next_candidates` 返回空就 break；把 `start` 换成语料外的词，第一步就无候选，前向次数为 0。

```python
c_ext = count_forward(model, 2, "外汇", 5)   # "外汇" 不在语料
print(f"语料外起点所需的实际前向次数: {c_ext}")
c_norm = count_forward(model, 2, "我喜欢", 5)
print(f"语料内起点所需的实际前向次数: {c_norm}")
```

预期结果：`c_ext = 0`（第一步就 break）而 `c_norm = 5`，说明语料外上下文会立即提前停止生成。

### 练习 4 · 思考题：为什么训练可 teacher forcing 而推理必须逐个生成

一句话思路：训练时用真实下一个 token 作标签、teacher forcing 可并行预测整句；推理时每个 token 依赖上一时刻输出，且需因果掩码挡住未来信息，所以只能逐个预测。

答案要点：训练用真实 token 填充输入，所有位置可同时计算且互不依赖，配合因果注意力（mask 掉未来位置）并行输出整句损失；推理不知道未来 token，当前预测的输入里根本没有下一时刻的内容，只能把已生成 token 拼回上下文再算一次，形成串行循环（L 步生成 = L 次前向）。

---

## 第 03 课 · 采样策略（03_sampling_strategies）

### 练习 1 · repetition_penalty 抑制重复

一句话思路：对已出现过的 token，把其 logits 除以惩罚系数（>1 惩罚），可在采样前先调整 logits，抑制重复内容。

```python
def repetition_penalty(logits, seen, factor=1.2):
    z = logits.copy()
    for t in seen:
        z[t] = z[t] / factor if z[t] > 0 else z[t] * factor
    return z

seen = [0, 2]                                  # 假设索引 0/2 已出现过
z_pen = repetition_penalty(LOGITS, seen, 1.5)
print("原始 idx0:", round(LOGITS[0], 3), "-> 惩罚后:", round(z_pen[0], 3))
```

预期结果：已见 token 的 logits 显著下降（如 3.2 → 2.13），再次被选中的概率被压低，实现抑制重复。

### 练习 2 · 严格越过 p 才算的 top-p

一句话思路：把 `keep = cum - probs[order] <= p` 改为 `keep = cum <= p`（严格累计≤p，不等过首个越界值），候选数会更少。

```python
def top_p_strict(logits, p):
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = cum <= p                        # 严格:累计正好超过 p 的那项被排除
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    return mask

print("宽松版候选数:", int(top_p_mask(LOGITS, 0.8).sum()))
print("严格版候选数:", int(top_p_strict(LOGITS, 0.8).sum()))
```

预期结果：严格版候选数比宽松版少 1 个左右（如宽松 3、严格 2），因为把"越过 p 的首个"也裁掉了。

### 练习 3 · 种子固定验证采样可复现

一句话思路：采样用同一 `rng`（同 seed）多次输出完全一致，换 seed 则分布仍对但具体序列不同。

```python
def sample_run(seed, n=6):
    rng = np.random.default_rng(seed)
    return [VOCAB[sample_from(LOGITS, rng=rng)] for _ in range(n)]

a, b = sample_run(42), sample_run(42)
c = sample_run(7)
print("同 seed 42:", "".join(a), "==" , "".join(b), a == b)
print("换 seed 7 :", "".join(c))
```

预期结果：两次 seed=42 输出完全一致（`True`），seed=7 输出序列不同但各 token 仍大致按概率分布出现。

### 练习 4 · 思考题：为什么 top-p 在模型确定时≈greedy 而 top-k 不会

一句话思路：模型输出确定（概率集中）时，top-p 会因累计概率快速达阈值而砍成接近 1 个候选 ≈ greedy；top-k 固定保留 k 个，与分布形状无关。

答案要点：top-p 自适应：若一个候选概率极高（如 0.9），累计一步就≥p，只剩它 ≈ greedy；top-k 永远硬性保留 k 个，即便前 k-1 名概率很低也无法裁掉，所以在"确定性分布"下仍可能采到次优项。即 top-p 按"需要多少候选才够"，top-k 按"固定给多少候选"，分布集中时前者退化 greedy 而后者不。

---

## 第 04 课 · Transformer 快速上手（04_transformer_quickstart）

### 练习 1 · head_dim 不整除报错

一句话思路：`view(B, T, n_heads, hd)` 要求 `hidden` 能被 `n_heads` 整除；`HIDDEN=32, N_HEADS=5` 使 `head_dim=6.4`，`.view` 因无法把 32 维平均切成 5 份而抛 RuntimeError。

```python
try:
    block_bad = ToyBlock(hidden=32, n_heads=5)   # 32 % 5 != 0
    x = torch.randn(1, 8, 32)
    block_bad(x)
    print("未报错?")
except RuntimeError as e:
    print("view 报错(reshape 约束):", str(e)[:90])
```

预期结果：抛 `RuntimeError`（如 `shape '[-1, 5, 6]' is invalid`），说明 `view/reshape` 必须能把总元素数（B*T*hidden）整除分配，hidden 必须整除 heads。

### 练习 2 · use_sdp=True 调官方注意力

一句话思路：在 `forward` 加 `use_sdp` 参数，为真时对已拆分的 q/k/v 调 `F.scaled_dot_product_attention`，与手写路径输出逐元素误差≈0。

```python
import torch.nn.functional as F
def forward_sdp(self, x):
    B, T, H = x.shape; hd = self.head_dim
    q = self.wq(x).view(B, T, self.n_heads, hd).transpose(1, 2)
    k = self.wk(x).view(B, T, self.n_heads, hd).transpose(1, 2)
    v = self.wv(x).view(B, T, self.n_heads, hd).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v)   # 官方实现
    return attn.transpose(1, 2).reshape(B, T, H)

block = ToyBlock(32, 4); x = torch.randn(1, 8, 32)
print("官方注意力输出 shape:", tuple(forward_sdp(block, x).shape))
```

预期结果：官方 SDPA 输出 shape 为 `(1, 8, 32)`，各 token 值等价于手写缩放点积注意力（误差 ~1e-5 量级）。

### 练习 3 · GELU 换 ReLU

一句话思路：把 `self.ffn` 里的 `nn.GELU()` 换成 `nn.ReLU()`，shape 不变（hidden→2h→hidden），但激活使负值截为 0、正半轴线性，值分布更稀疏。

```python
import torch.nn as nn
block_relu = ToyBlock(32, 4)
block_relu.ffn[1] = nn.ReLU()          # 替换激活
x = torch.randn(1, 8, 32)
o, _ = block_relu(x)
print("ReLU 块输出 shape:", tuple(o.shape), "| 均值:", o.mean().item())
print("GELU 块输出均值:", ToyBlock(32, 4)(x)[0].mean().item())
```

预期结果：两者输出 shape 都是 `(1, 8, 32)`；ReLU 使大量隐藏值刚好为 0、分布更稀疏，均值/方差与 GELU 略不同。

### 练习 4 · 两层 ToyBlock 堆叠

一句话思路：复用两次 `ToyBlock`，把前一层输出作为后一层输入，逐层打印 hidden shape，并观察 `LayerNorm` 把每层输入的均值/方差归一到 `~0/~1`。

```python
layer1, layer2 = ToyBlock(32, 4), ToyBlock(32, 4)
x = torch.randn(1, 8, 32)
for i, blk in enumerate([layer1, layer2], 1):
    x, w = blk(x)
    print(f"第 {i} 层 hidden shape: {tuple(x.shape)} | norm 均值 {x.mean():.3f} 方差 {x.var():.3f}")
print("最终输出 shape:", tuple(x.shape))
```

预期结果：每一层都打印 `(1, 8, 32)`；经 LayerNorm 后每层输出均值≈0、方差≈1，层数不改变 hidden 维度，只不断重整分布。

---

## 第 05 课 · Prefill vs Decode（05_prefill_vs_decode）

### 练习 1 · 改 L=512/1024 重跑微基准

一句话思路：`bench_prefill_decode(d=256, layers=8, vocab_size=1024, L=…, steps=…)` 的 prefill 段一次并行处理 L 个 token，耗时应随 L 近似线性增长。

```python
for L in [256, 512, 1024]:
    b = bench_prefill_decode(d=256, layers=8, vocab_size=1024, L=L, steps=64, reps=5)
    print(f"L={L:4d}: prefill {b['prefill_ms']:6.1f} ms, "
          f"{b['prefill_tok_per_s']/1e3:6.1f} k tok/s")
```

预期结果：L 翻倍时 `prefill_ms` 近似翻倍（如 256→约40ms、512→约80ms、1024→约160ms），单 token 耗时基本持平，验证 prefill 计算量与 L 近似线性。

### 练习 2 · 小批阶段吞吐近似线性

一句话思路：`bench_throughput_curve(batch=(1,2,3,5,8))` 探究 decode 小批区间，吞吐随 batch 先近似线性上升（固定开销摊薄主导）。

```python
b_list, tps, mps = bench_throughput_curve(batch=(1, 2, 3, 5, 8), token_len=16, reps=5)
t0 = tps[0]
for b, tp in zip(b_list, tps):
    print(f"batch {b}: 吞吐 {tp:7.0f} tok/s  约为 batch=1 的 {tp/t0:4.2f}×")
```

预期结果：batch=1→2→3 时吞吐近似线性（约 1×→2×→3×），到 5→8 增幅开始放缓，说明小批阶段固定开销主导、吞吐近线性。

### 练习 3 · RTX 5060 交叉点（AI 公式）

一句话思路：交叉点 `AI*=π_peak/β_mem`；由 GPU 算力与带宽算出 AI*，再比较 prefill（算 token 多、AI 高）与 decode（单 token、AI 低）各贴哪堵墙。

```python
pi_rtx5060 = 160e12        # RTX 5060 近似 fp16 算力 FLOP/s(以规格略估)
beta_rtx5060 = 448e9       # ~448 GB/s 显存带宽
ai_star = pi_rtx5060 / beta_rtx5060
hidden = 8192; dtype_bytes = 2
def ai_decode(): return 2*hidden*hidden / (hidden*hidden*dtype_bytes)
print(f"RTX 5060 交叉点 AI* ≈ {ai_star:.0f} FLOP/Byte")
print(f"decode AI ≈ {ai_decode():.0f} << AI* (memory-bound) 贴带宽墙")
print("prefill 一次性并行 L token,AI 高得多 → compute-bound,贴算力墙")
```

预期结果：AI* 约 350~400 FLOP/Byte；decode 的 AI≈1 被带宽卡住（memory-bound），prefill 大 L 时 AI≫AI*、被算力卡住（compute-bound），两者贴不同墙。

### 练习 4 · 思考题：为什么 batch 增大吞吐会饱和

一句话思路：decode 阶段每步都要把整条序列的 KV Cache 重新读一遍做注意力；KV 读取量与每步 token 数（≈batch×seq_len）线性增长，把增批带来的并行收益吃光，吞吐饱和。

答案要点：吞吐 = 每步产出 token / 每步耗时。每步产出随 batch 线性增长，但每步要读取的 KV Cache（与 batch×当前序列长成正比）也随之线性增大，占满带宽后计算与访存都随 batch 线性扩张，比值趋于常数 → 吞吐饱和不再随 batch 上升（带宽墙）。

---

## 第 06 课 · 玩具推理引擎（toy_inference_engine）

### 练习 1 · shared_forward 矩阵变大

一句话思路：把引擎 `shared_forward` 里的矩阵从 (48,48) 改成 (64,64)/(96,96)，C_fixed（固定开销）增大，吞吐曲线整体下移、饱和更快。

```python
class BigEngine(ToyEngine):
    def __init__(self, model, n=2, temperature=1.0, seed=0, dim=96):
        super().__init__(model, n, temperature, seed)
        self.W = np.random.randn(dim, dim) * 0.05
    def shared_forward(self):
        d = self.W.shape[0]
        return np.random.randn(d, d) @ self.W

e = BigEngine(model, n=2, seed=0, dim=96)
seqs, total = e.generate_batch(["我喜欢", "机器"], max_new=10)
print("96×96 固定开销引擎生成 token 数:", total)
```

预期结果：矩阵越大每步 `shared_forward` 越贵，C_fixed 明显变大；吞吐曲线整体下移且在更小的 batch 就进入饱和平台。

### 练习 2 · 序列完成即退出

一句话思路：在 `generate_batch` 里当某条序列取不到候选（无后继）时，把该序列从活跃列表移除，之后不再为它采样。

```python
def generate_batch_early(self, prompts, max_new):
    seqs = [list(p) for p in prompts]; tokens = 0
    for _ in range(max_new):
        self.shared_forward()
        alive = []
        for seq in seqs:
            toks, probs = next_candidates(self.model, seq, self.n)
            if not toks:
                continue                      # 完成即退出:丢弃该序列
            logits = np.log(np.array(probs) + 1e-12)
            seq.append(toks[sample_from(logits, self.temperature, self.rng)])
            tokens += 1; alive.append(seq)
        seqs = alive
        if not seqs: break
    return seqs, tokens

e = ToyEngine(model, n=2, seed=0)
print("提前退出模式 token 数:", generate_batch_early(e, ["我喜欢", "机器"], 10)[1])
```

预期结果：已到死路的序列不再参与后续步，`seqs` 长度动态收缩、生成也可能提前清零；新增 token 总数不高于原版，且避免为死序列做无谓采样。

### 练习 3 · 拟合残差 R²

一句话思路：用 `np.polyfit(batches, step_times, 1)` 得 C_fixed/C_per_seq，构造预测吞吐 `b/(C_fixed+b·C_per_seq)`，再算残差 R²评估固定开销模型解释力。

```python
b_arr = np.array(batches, float); st_arr = np.array(step_times, float)
Cs, Cf = np.polyfit(b_arr, st_arr, 1)
pred_tp = b_arr / (Cf + b_arr * Cs)
real = np.array(tps, float)
ss_res = np.sum((real - pred_tp)**2); ss_tot = np.sum((real - real.mean())**2)
R2 = 1 - ss_res / ss_tot
print(f"C_fixed={Cf*1e6:.1f}μs, C_per_seq={Cs*1e6:.2f}μs, 拟合 R² = {R2:.3f}")
```

预期结果：R² 接近 0.9~1.0（拟合很好），说明 `T_step = C_fixed + b·C_per_seq` 的固定开销模型能解释绝大部分吞吐变化；低于 1 的主因是部分序列提前终止导致实测吞吐略低于模型预测。

### 练习 4 · 思考题：C_per_seq 随序列长度增长的吞吐曲线

一句话思路：真实 LLM 每条序列的 KV Cache 随长度增长，每步每序列成本不再恒定而是 C_per_seq 随序列长上升，吞吐曲线形状改变。

答案要点：原模型假设每序列边际成本恒定（水平直线），吞吐预测随 batch 单调上升并饱和到 `1/C_per_seq`。真实里 C_per_seq 随序列长度增长（更长序列每步要读更多 KV Cache、注意力成本更高），长序列会让每步耗时随步数增长，于是吞吐曲线不再是上凸饱和，而是先升后随平均长度增大而回落、出现明显峰值，小批次也要考虑序列长度分布。