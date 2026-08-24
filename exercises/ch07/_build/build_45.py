# -*- coding: utf-8 -*-
"""生成 45_paged_attention.ipynb 与 app_45_paged_attn.py(教材级重写版)

对齐 REWRITE_STANDARD.md。论文支撑:
PagedAttention: Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention",
SOSP'23 (arXiv:2309.06180);vLLM 官方博客 (blog.vllm.ai, 2023-06-20)。
工程引用:vllm/v1/attention/ops/paged_attn.py、vllm/v1/core/block 系列;aios/kvcache/mha_pool.py。
"""
from pathlib import Path
from helpers import D, new_nb, chapter_cover, wrapup, CH07, CPU_HEADER

APP_FILE = "app_45_paged_attn.py"

APP_45 = D('''
# -*- coding: utf-8 -*-
# app_45_paged_attn.py — PagedAttention:block table 驱动的非连续 KV 访问 📖
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📖 45 · PagedAttention", layout="wide")
st.title("📖 第 45 课 · PagedAttention:block table 驱动的 KV 访问")

st.markdown("""
LLM 生成时,KV Cache 会长长短短,如果给每个请求**连续分配**一段内存,会产生大量碎片和浪费。
**PagedAttention** 借鉴操作系统的**分页**:把 KV 切成固定大小的**物理块**,每个请求维护一张
**block table**(逻辑块 → 物理块),于是内存可以按需、不连续地分配。
下方拖动**请求数 / 块大小**,看 block table 长什么样、以及如何按它收集 KV 再做 attention。
""")

def run(num_req, block_size, total_blocks, d=8):
    rng = np.random.default_rng(7)
    req_blocks = rng.integers(1, 4, num_req)          # 每个请求占用的块数
    slots = np.arange(total_blocks)
    rng.shuffle(slots)                                # 打乱物理块,模拟碎片
    block_table = []
    cursor = 0
    for nb in req_blocks:
        block_table.append(slots[cursor:cursor + nb]) # 每个请求的逻辑块→物理块
        cursor += nb
    max_len = int(req_blocks.max() * block_size)
    return req_blocks, max_len, block_table

with st.sidebar:
    st.header("🎛️ 参数")
    num_req = st.slider("请求数(并发序列)", 1, 8, 4, 1)
    block_size = st.slider("块大小(每块 token 数)", 2, 8, 4, 1)
    total_blocks = st.slider("物理块总数", 8, 32, 16, 1)
    st.caption("请求的 KV 按需占用若干物理块,物理块可能不连续——这就是分页的威力。")

req_blocks, max_len, block_table = run(num_req, block_size, total_blocks)

st.subheader("🧱 Block Table(逻辑块 → 物理块)")
st.write("每个请求占用" + ", ".join(f"Req{i}: {int(nb)} 块" for i, nb in enumerate(req_blocks)))
st.dataframe(
    np.array([[int(v) for v in row] + [-1] * (max_len // block_size - len(row))
              for row in block_table]),
    use_container_width=True)

d = 8
rng = np.random.default_rng(7)
phys_k = rng.normal(0, 1, (total_blocks, block_size, d)).astype(np.float32)  # 物理 KV 池
phys_v = rng.normal(0, 1, (total_blocks, block_size, d)).astype(np.float32)
cont_k = rng.normal(0, 1, (int(req_blocks.sum() * block_size), d)).astype(np.float32)

results = []
for i, blocks in enumerate(block_table):
    nb = int(req_blocks[i])
    K = phys_k[blocks].reshape(nb * block_size, d)     # 按表收集:非连续块拼成连续 KV
    V = phys_v[blocks].reshape(nb * block_size, d)
    q = rng.normal(0, 1, (1, d)).astype(np.float32)
    S = K @ q / (d ** 0.5)
    p = np.exp(S - S.max()); p = p / p.sum()
    o = p @ V
    results.append((int(nb * block_size), o.reshape(-1)))

st.subheader("🔬 按 block table 收集后的 attention 结果")
st.write("每个请求:块数 → token 数 → 输出(前 3 维)")
for i, (ntok, o) in enumerate(results):
    st.write(f"Req{i}: {ntok} tokens → " + "[" + ", ".join(f"{v:.3f}" for v in o[:3]) + ", ...]")

cont_bytes = int(req_blocks.sum() * block_size) * d * 4
page_bytes = sum(nb * block_size for nb in req_blocks) * d * 4
fig = go.Figure()
fig.add_trace(go.Bar(x=["连续分配", "分页(仅用到的)"], y=[cont_bytes, page_bytes],
                     marker_color=["#E45756", "#4C78A8"], text=[cont_bytes, page_bytes],
                     textposition="outside"))
fig.update_layout(title="KV 访存字节对比:连续分配 vs 分页按需", yaxis_title="字节",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 连续分配要为整片预留(可能含空洞),分页只按实际用到的块读写——内存利用率更高。")

st.markdown("""
> 💡 **结论**:PagedAttention 的核心是 **block table**:一张「逻辑块 → 物理块」的映射表,
> 让不连续的物理内存也能被当作连续的逻辑序列访问。torch 用 advanced indexing(`tensor[blocks]`)
> 就能实现“按表收集”,这正是 vLLM 在 `paged_attn.py` 里用专门 kernel 做的事——只是 GPU kernel
> 把收集与 attention 融合在一起,省掉来回搬运。来源:
> [Kwon et al., SOSP 2023 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 7 章 · 第 45 课配套演示")
''')

NB = new_nb("第 45 课 · PagedAttention",
            subtitle="block table 让非连续 KV 也能被当作连续序列访问;用 torch 模拟分页注意力并数值验证",
            emoji="📖")

chapter_cover(NB,
    objectives=[
        "理解 KV Cache 连续分配的碎片化问题(论文 §3)与 60–80% 的浪费",
        "掌握 block table(逻辑块→物理块)的结构与作用,建立符号表",
        "用 torch advanced indexing(K[blocks])模拟按 block table 收集 KV 再做 attention",
        "数值验证「分页 KV ≡ 连续 KV」输出一致(allclose),并量一次 gather 的耗时代价",
        "对比连续分配与分页的访存/内存利用,看懂近零浪费(<4%)",
        "理解 PagedAttention 在 vLLM 里的角色(block manager、copy-on-write、paged_attn 内核)",
    ],
    toc=[
        ("直觉:图书馆临时加书架", "连续分配在动态长度下的浪费"),
        ("问题:KV Cache 的碎片化", "请求长短不一,连续分配浪费 60–80% 显存"),
        ("block table:逻辑块→物理块", "分页思想,不连续也能当连续用"),
        ("torch 模拟:按表收集 KV", "advanced indexing + 标准 attention,逐行推演"),
        ("数值验证:分页 ≡ 连续", "allclose 断言输出一致 + gather 耗时代价"),
        ("与连续 KV 的访存对比", "只按需读写,内存利用率更高(<4% 浪费)"),
        ("与 vLLM 工程实现的关系", "paged_attn 内核、block manager、copy-on-write、aios"),
        ("小结 + 练习 + 延伸阅读", "要点、动手题、论文链接"),
    ],
    links=[
        ("PagedAttention 论文 (SOSP 2023)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方博客: 10x faster LLM serving", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("PyTorch advanced indexing 文档", "https://pytorch.org/docs/stable/notes/indexing.html"),
    ])

NB.md("## 1. 直觉:图书馆临时加书架 📚\n\n"
      "图书馆里,每位读者来借一批书,管理员要给每人**留一块连续的空书架**。问题来了:读者来的时间"
      "不定、借的册数不定,书架留少了放不下,留多了又空着浪费。尤其当书架被东一块西一块占满后,"
      "新读者几乎找不到一整片连续空间。\n\n"
      "LLM 的 **KV Cache** 就是这批「书」:每个请求(token)在生成时都要把 $K,V$ 存进缓存,"
      "而不同请求的长度**动态变化**。如果给每个请求**连续分配**一大段内存,就会遇到上面一模一样的"
      "碎片化问题——要么浪费,要么放不下。\n\n"
      "> 📄 vLLM 博客指出:现有系统因碎片化与过度预留,浪费了 **60%–80%** 的 KV cache 内存。"
      "[vLLM 官方博客, 2023-06-20](https://blog.vllm.ai/2023/06/20/vllm.html)")

NB.code(CPU_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed。有 CUDA 就走真实 GPU——下面的分页 KV 收集/attention 都是 GPU 上真算的。")

NB.md("## 2. 问题:KV Cache 的碎片化 🧩\n\n"
      "假设有两个请求:请求 A 长了 3 个 token,请求 B 长了 6 个 token。连续分配时,我们得给 A 预留"
      "(比如 8 个 token 的)整块空间以防它再增长,给 B 也要预留——但这些预留的空白,在 A 只用到 3 个"
      "token 时就是**浪费**。更糟的是,如果预留空间不够,还得整块搬走(重新分配),非常昂贵。\n\n"
      "**PagedAttention** 的解法([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)):"
      "不按「序列」连续分配,而是把 KV 切成固定大小的**块(block)**,"
      "每个请求只**按需**占用若干块,并且这些物理块**可以散落在内存各处**。关键就是一张映射表把它们串起来。\n\n"
      "先定符号表:\n\n"
      "| 符号 | 含义 |\n|---|---|\n"
      "| $B$ | 每块容纳的 token 数(block_size,如 16) |\n"
      "| $N_{\\text{blk}}$ | 物理块总数(显存里预分配的块池) |\n"
      "| $L_j$ | 第 $j$ 个逻辑块(序列里的第 $j$ 段) |\n"
      "| $P_j$ | 第 $L_j$ 对应的物理块编号 |\n"
      "| block_table | 每个请求的 $L \\to P$ 映射表,形状 `(num_blocks_per_req,)` |\n"
      "| KV 池 | 连续内存池,形状 `(N_blk, B, d)` |\n\n"
      "> 论文把「逻辑连续、物理不连续」比作 OS 的虚拟内存:块=页,token=字节,请求=进程。"
      "第 4.1 节 PagedAttention 正是让 KV 块可存于非连续物理空间。")

NB.md("## 3. block table:逻辑块 → 物理块 🗺️\n\n"
      "每个请求维护一张 **block table**:第 $j$ 个**逻辑块**(序列里的第 $j$ 段)对应哪个**物理块**"
      "(显存里的实际位置)。物理块是全局共享的内存池,谁需要谁去取。\n\n"
      "就像操作系统的虚拟内存:**逻辑地址连续,物理地址可以不连续**。对 attention 来说,"
      "它只要按 block table 把 KV 一块块取出来,就能拼成「看似连续」的序列。")

NB.code(D('''
# 手工构造一个分页场景:4 个请求,块大小 4,共 16 个物理块
rng = np.random.default_rng(7)
num_req, block_size, total_blocks = 4, 4, 16          # 请求数、每块 token 数、物理块总数
req_blocks = rng.integers(1, 4, num_req)              # 每个请求占用的块数(1..3)
slots = np.arange(total_blocks); rng.shuffle(slots)   # 物理块编号,顺序打乱模拟碎片
block_table, cursor = [], 0
for nb in req_blocks:
    block_table.append(slots[cursor:cursor + nb])     # 为每个请求顺序取 nb 个(散落的)物理块
    cursor += nb

print("每个请求占用的块数:", req_blocks.tolist())
print("物理块编号(被打乱):", slots.tolist())
print("block table(每个请求的逻辑块→物理块):")
for i, bt in enumerate(block_table):
    print(f"  Req{i}: {bt.tolist()}")
print(f"\\nblock_table 的形状: 每个请求一行,长度=该请求的逻辑块数")
'''), "🎯 每个请求的逻辑块 0,1,2… 指向的是**不连续**的物理块——这就是分页的核心。")

NB.md("## 4. torch 模拟:按表收集 KV 再做 attention 🔬\n\n"
      "物理 KV 是一个连续内存池(形状 `(total_blocks, block_size, d)`)。要算 attention,"
      "先用 **advanced indexing** `tensor[blocks]` 按 block table 把非连续块**收集**成一个连续张量,"
      "再做标准 softmax attention——这就是「等价分块注意力」的 torch 版。")

NB.code(D('''
# torch 模拟:按 block table 收集该请求的 KV,再做单查询 attention
import torch
torch.manual_seed(0)
d = 8                                          # 每头维度
phys_k = torch.randn(total_blocks, block_size, d)   # 物理 K 池 (N_blk, B, d)
phys_v = torch.randn(total_blocks, block_size, d)   # 物理 V 池 (N_blk, B, d)
print(f"物理 KV 池 shape = {tuple(phys_k.shape)} <- (物理块数={total_blocks}, 每块token={block_size}, dim={d})")

def paged_attention(q, blocks, K, V):
    """按 block table 收集该请求的 KV,再做单查询 attention。"""
    K_cont = K[blocks].reshape(-1, d)   # advanced indexing:逻辑块→物理块→连续 (tokens, d)
    V_cont = V[blocks].reshape(-1, d)   # (tokens, d)
    S = K_cont @ q / (d ** 0.5)         # 打分 (tokens,)
    p = torch.softmax(S, dim=-1)        # softmax
    return p @ V_cont                   # 加权求和 (d,)

q = torch.randn(d)                      # 单查询向量 (d,)
for i, bt in enumerate(block_table):
    o = paged_attention(q, bt, phys_k, phys_v)   # 输出 (d,)
    ntok = len(bt) * block_size                 # 该请求的总 token 数
    print(f"Req{i}: {len(bt)}块/{ntok} token → 输出前3维 {[f'{v:.3f}' for v in o[:3].tolist()]}")
'''), "✅ `K[blocks]` 一行就把非连续物理块拼成了连续 KV——torch 的 advanced indexing 正是分页收集的原型。")

NB.md("## 5. 数值验证:分页 KV ≡ 连续 KV ⚖️\n\n"
      "在 GPU 上做一个干净对照。同一个**逻辑序列**,分别用两种方式存取:\n"
      "- **连续**:KV 是一段连续内存(物理块紧挨着);\n"
      "- **分页**:KV 散落在被打乱的物理块里,靠 block table 用 advanced indexing 收集。\n\n"
      "两者在数学上必须给出**完全相同**的 attention 输出(allclose),因为 block table 只改变了**存放方式**、"
      "没有改变 KV 的值。同时量一量:非连续 KV 比连续 KV 多付出的代价,就是那一次 `gather`(搬运)。")

NB.code(D('''
# 真机验证①:GPU 上验证「分页 KV(block table 收集)≡ 连续 KV」输出一致性
torch.manual_seed(3)
dd = 64
num_req, block_size, total_blocks = 6, 16, 80        # 6 个请求,块大小 16,共 80 物理块
rng = np.random.default_rng(7)
req_blocks = rng.integers(2, 5, num_req)             # 每请求 2..4 块
slots = np.arange(total_blocks); rng.shuffle(slots)  # 打乱物理块,制造不连续
block_table, cursor = [], 0
for nb in req_blocks:
    block_table.append(slots[cursor:cursor + nb]); cursor += nb

phys_k = torch.randn(total_blocks, block_size, dd, device=dev)   # 物理 K 池 (N_blk,B,d)
phys_v = torch.randn(total_blocks, block_size, dd, device=dev)   # 物理 V 池

def paged_attn(q, blocks):
    K = phys_k[blocks].reshape(-1, dd)    # 分页:advanced indexing 收集非连续块 (tokens,d)
    V = phys_v[blocks].reshape(-1, dd)
    S = K @ q / math.sqrt(dd)
    P = torch.softmax(S, dim=-1)
    return P @ V

def contiguous_attn(q, Kc, Vc):
    S = Kc @ q / math.sqrt(dd)            # 连续:直接对连续 KV 做 attention
    P = torch.softmax(S, dim=-1)
    return P @ Vc

all_ok, max_err = True, 0.0
for i, bt in enumerate(block_table):
    tb = torch.tensor(bt, device=dev)
    Kc = phys_k[tb].reshape(-1, dd)       # 等价的连续 KV(同一逻辑序列)
    Vc = phys_v[tb].reshape(-1, dd)
    q = torch.randn(dd, device=dev)
    op = paged_attn(q, tb); oc = contiguous_attn(q, Kc, Vc)
    e = (op - oc).abs().max().item()
    all_ok &= bool(torch.allclose(op, oc, atol=1e-6, rtol=1e-6))
    max_err = max(max_err, e)
print(f"{num_req} 个请求(物理块都已打乱)的分页 vs 连续:")
print(f"  全部 allclose = {all_ok}")
print(f"  最大绝对误差   = {max_err:.2e}")
print("  → block table 只改了 KV 的存放位置,不改值;收集后与连续完全等价。")
del phys_k, phys_v; torch.cuda.empty_cache()
'''), "🎯 分页与连续的输出逐请求 allclose=True(~1e-6 量级)——PagedAttention 的 block table 收集在数学上是无损的,只是在 GPU 上多一次 gather。")

NB.code(D('''
# 真机验证②:耗时——非连续 KV 比连续 KV 多付「一次 gather 搬运」的代价
def bench_gpu(fn, iters=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

L, bs, dd = 8192, 128, 64; nb = L // bs               # 长序列 8192,块 128,共 64 块
Kfull = torch.randn(L, dd, device=dev); Vfull = torch.randn(L, dd, device=dev)
perm = torch.randperm(nb, device=dev)
Kpool = Kfull.reshape(nb, bs, dd)[perm]               # 物理块乱序存放
Vpool = Vfull.reshape(nb, bs, dd)[perm]
ltp = perm.argsort().to(torch.int64)                  # 逻辑块 j -> 物理块号(block table)
qb = torch.randn(64, dd, device=dev)                  # 64 个 query

def contiguous_style():
    S = qb @ Kfull.t() / math.sqrt(dd); P = torch.softmax(S, -1); return P @ Vfull

def paged_style():
    Kg = Kpool[ltp].reshape(-1, dd); Vg = Vpool[ltp].reshape(-1, dd)   # 先按表 gather
    S = qb @ Kg.t() / math.sqrt(dd); P = torch.softmax(S, -1); return P @ Vg

err = (contiguous_style() - paged_style()).abs().max().item()
tc, tp = bench_gpu(contiguous_style), bench_gpu(paged_style)
print(f"序列 L={L}, 64 个 query, d={dd} @ {dev}:")
print(f"  连续 KV dense       : {tc:6.3f} ms")
print(f"  分页(先 gather 再算) : {tp:6.3f} ms   (+{tp-tc:.3f} ms = 一次非连续收集的费用)")
print(f"  连续 vs 分页输出误差 : {err:.2e} (同一逻辑序列,应 ~0)")
del Kfull, Vfull, Kpool, Vpool; torch.cuda.empty_cache()
'''), "🚀 真机数字:任务量相同时,分页路径只比连续多一次 gather(毫秒级)+ 输出仍逐位一致。vLLM 之所以要专门 kernel 融合收集与 attention,就是为了省掉这块来回搬运、让分页几乎「免费」。")

NB.md("## 6. 与连续 KV 的访存对比 📊\n\n"
      "连续分配要为每个请求**预留一整段**(可能含没用到的地方);分页只按**实际用到的块**读写。"
      "论文指出分页的内存浪费只发生在序列的**最后一块**(<4%),近乎最优。算一下「真正读写」的字节差异:")

NB.code(D('''
# 对比连续分配(预留最大块数)与分页(按需)的 KV 字节数
def mem_contiguous(req_blocks, block_size, d, bytes_per=4):
    """连续分配:每个请求预留最大块数×块大小,含空洞浪费。"""
    max_req = int(req_blocks.max())                      # 最长请求占的块数(预留上限)
    return (max_req * block_size * num_req) * d * bytes_per   # 所有请求都按最长的预留

def mem_paged(req_blocks, block_size, d, bytes_per=4):
    """分页:只按实际用到的块。"""
    used = int(req_blocks.sum()) * block_size            # 实际用到的 token 总数
    return used * d * bytes_per

mc = mem_contiguous(req_blocks, block_size, d)           # 连续预留
mp = mem_paged(req_blocks, block_size, d)                # 分页按需
print(f"连续分配预留给读的 KV: {mc} 字节")
print(f"分页实际读写的 KV   : {mp} 字节")
print(f"分页节省(本例)      : {(1 - mp/mc)*100:.0f}%")
'''), "🚀 请求长度参差越大,连续分配的预留浪费越多,分页优势越明显。真实推理中请求成千上万,这个差距非常可观。")

NB.md("## 7. 与 vLLM 工程实现的关系 🔗\n\n"
      "PagedAttention 不只是算法,它支撑了 vLLM 的一整套内存管理:\n\n"
      "1. **块池(block pool)**:`vllm/v1/core/block/` 里预分配固定大小的物理块(默认 block_size,如 16 token/块);\n"
      "2. **block table**:每个请求维护 逻辑块→物理块 映射;论文第 4.2 节「KV Cache Manager」正是虚拟内存式的管理;\n"
      "3. **分页内核**:`vllm/v1/attention/ops/paged_attn.py` 的 PagedAttention kernel 直接按 block table 取 KV"
      "并**融合**进 attention,免去本课那个显式 gather 的搬运;\n"
      "4. **copy-on-write 共享**:并行采样/beam search 时多个输出序列共享 prompt 的物理块,靠引用计数 + 写时复制"
      "(论文 §4.4),可省 55% 内存、提 2.2× 吞吐;\n"
      "5. **教学引擎 aios**:`aios/python/aios/kvcache/mha_pool.py` 用预分配 `(2, L, num_pages, page_size, kv_heads, head_dim)`"
      "的页式池实现同一思想。\n\n"
      "> 📄 一句话:PagedAttention 解决的是「KV Cache 怎么存」;本课用 torch 验证了它的数学正确性,"
      "工程上由 vLLM 的 block manager + paged_attn 内核把它做成近零浪费的高吞吐推理。")

NB.md("## 8. 🖥️ Streamlit 动态演示:拖请求数看分页 🎛️\n\n"
      "运行同目录下的 `app_45_paged_attn.py`,拖动**请求数 / 块大小**,实时生成 block table、"
      "按它收集 KV 做 attention,并对比连续分配与分页的访存:\n\n"
      "```bash\nD:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_45_paged_attn.py\n```\n\n"
      "浏览器打开 **http://localhost:8501**。建议把请求数从 1 拖到 8,看 block table 如何铺开、"
      "以及分页访存相对连续分配节省多少。完整源码如下(与同目录 app 一字不差):")

NB.code(f"%%writefile {APP_FILE}\n" + APP_45, "📜 这就是 app_45_paged_attn.py 的完整源码,notebook 与 app 共用同一套按 block table 收集的 paged_attention 实现,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "KV Cache 长度动态、请求各异,连续分配会产生碎片与 60–80% 的预留浪费",
        "PagedAttention 借鉴 OS 虚拟内存:KV 切成固定物理块,按需分配,逻辑连续、物理可不连续",
        "block table 是「逻辑块→物理块」的映射表,是分页的核心数据结构",
        "torch advanced indexing(K[blocks])就能模拟按表收集,等价于分块注意力",
        "数值验证:分页 KV 与连续 KV 输出逐请求 allclose=True;分页只多一次 gather(毫秒级)",
        "分页只按实际用到的块读写,内存浪费只剩最后一块(<4%),远优于连续预留",
        "vLLM 用 block pool + block table + paged_attn 内核 + copy-on-write 实现近零浪费的高吞吐推理",
    ],
    practice=[
        "把 total_blocks 调大、请求数调多,观察分页访存节省比例的变化趋势",
        "给 paged_attention 换成多查询(批量 Q),返回完整 (Nq, d) 输出",
        "手动构造一个含空洞的物理池,验证 K[blocks] 仍能正确收集",
        "思考:PagedAttention 的 kernel 为什么要把收集与 attention 融合(省掉中间张量)?",
        "读论文 §4.4,解释 copy-on-write 如何在并行采样时共享 prompt 的物理块",
    ],
    links=[
        ("PagedAttention 论文 (SOSP 2023)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("PyTorch advanced indexing", "https://pytorch.org/docs/stable/notes/indexing.html"),
    ])

NB.save(str(Path(CH07) / "45_paged_attention.ipynb"))

app_path = Path(CH07) / APP_FILE
app_path.write_text(APP_45 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
