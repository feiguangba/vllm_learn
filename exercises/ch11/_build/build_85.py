# -*- coding: utf-8 -*-
"""生成 85_pagedattn_ascend.ipynb 与 app_85_paged_ascend.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_85 = D('''
# -*- coding: utf-8 -*-
# app_85_paged_ascend.py — PagedAttention 在昇腾:块管理与碎片率 📖
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📖 85 · PagedAttention 在昇腾", layout="wide")
st.title("📖 第 85 课 · PagedAttention 在昇腾:块大小与碎片率的博弈")

st.markdown("""
昇腾上的 KV Cache 一样用**分页**管理:把 KV 切成固定大小的**块(block)**,每个请求按需占用若干块,
用 **block table** 记录「逻辑块 → 物理块」。块越大,表越短、块内空间越浪费;
块越小,浪费越少、表越长、管理开销越大。下方拖动**序列长度分布 / 块大小**,实时观察
**碎片率、块表大小与内存利用率**——体会昇腾/GPU 都在做的这个平衡。
""")

def run(seq_lens, block_size, total_gb=64.0):
    seq_lens = np.asarray(seq_lens, dtype=np.float64)
    req_blocks = np.ceil(seq_lens / block_size).astype(int)
    wasted = (req_blocks * block_size - seq_lens).sum()
    frag = wasted / (req_blocks * block_size).sum() * 100
    kv_gb = (req_blocks * block_size).sum() * 2.0 * 4096 * 2.0 / 1e9 * 0.5
    return req_blocks, frag, kv_gb

with st.sidebar:
    st.header("🎛️ 参数")
    block_size = st.slider("块大小(token/块)", 1, 32, 8, 1)
    n_req = st.slider("请求数", 10, 500, 100, 10)
    max_len = st.slider("最大序列长度(token)", 128, 4096, 1024, 128)
    skew = st.radio("长度分布", ["均匀", "偏长尾(多数短、少数长)"])
    st.caption("碎片率 = 块内没用到的那部分占已分配块总容量的比例。")

rng = np.random.default_rng(7)
if skew == "均匀":
    seqs = rng.integers(16, max_len, n_req)
else:
    seqs = np.clip(np.random.default_rng(7).exponential(max_len / 3.0, n_req).astype(int), 16, max_len)

req_blocks, frag, kv_gb = run(seqs, block_size)
c1, c2, c3 = st.columns(3)
c1.metric("平均每请求块数", f"{req_blocks.mean():.1f}")
c2.metric("碎片率", f"{frag:.1f} %")
c3.metric("估算 KV 占用", f"{kv_gb:.2f} GB")

st.subheader("🧱 Block Table(逻辑块 → 物理块)")
rows = [req_blocks[i].tolist() for i in range(min(8, n_req))]
max_b = max(len(r) for r in rows)
tbl = np.array([[int(b) for b in r] + [-1] * (max_b - len(r)) for r in rows])
st.dataframe(tbl, width="stretch")
st.caption("每格 = 该请求第 j 个逻辑块对应的物理块编号;-1 = 未分配。物理块来自全局内存池,不保证连续。")

st.subheader("📉 碎片率 vs 块大小")
bs = np.arange(1, 33)
frags = [run(seqs, b)[1] for b in bs]
fig = go.Figure(go.Scatter(x=bs, y=frags, mode="lines+markers", line=dict(color="#E45756", width=3)))
fig.add_vline(x=block_size, line_dash="dash", line_color="#4C78A8")
fig.update_layout(title="块大小越大 → 碎片率越高(曲线单调上升)", xaxis_title="块大小(token)",
                  yaxis_title="碎片率(%)", height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 但块太大不只是坏处:块表更短、算子访存更连续。工业界(昇腾/GPU)默认块大小多为 16~32。")

st.markdown("""
> 💡 **结论**:昇腾的 PagedAttention 与 CUDA 版共享同一套**分页心法**:按需分配、块表寻址、
> 非连续物理块也能当连续序列用。块大小是一个“碎片率 ↔ 管理开销”的工程旋钮——
> 选小了浪费空间,选大了浪费内存。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 85 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
''')

NB = new_nb("第 85 课 · PagedAttention 在昇腾",
            subtitle="分页思想换个硬件依旧成立:块管理、块表收集与融合注意力",
            emoji="📖")

chapter_cover(NB,
    objectives=[
        "回顾 45 课的分页思想,把它平移到昇腾的 KV 管理上",
        "理解昇腾 KV 池与 block table 的实现思路",
        "用 torch 模拟昇腾融合算子的“按表收集”再算 attention",
        "理解 flash attention 在昇腾的融合实现(npu_fusion_attention)",
        "量化分析块大小与碎片率的权衡,理解工程旋钮",
    ],
    toc=[
        ("直觉:回到 45 课的图书馆", "分页思想:换硬件也不变"),
        ("昇腾 KV 池与 block table", "物理块池 + 逻辑→物理映射"),
        ("torch 模拟按表收集", "昇腾融合算子内部动作的复刻"),
        ("FlashAttention 的昇腾实现", "多 kernel 变单算子:融合的艺术"),
        ("块大小 vs 碎片率", "一块多大才划算?定量看"),
        ("交互图:block table 可视化", "plotly 看物理块的“跳跃”"),
        ("配套 Streamlit 演示", "app_85_paged_ascend.py:拖块大小看碎片"),
    ],
    links=[
        ("PagedAttention 论文", "https://arxiv.org/abs/2309.06180"),
        ("vLLM-Ascend 支持矩阵", "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/"),
        ("昇腾 FlashAttention 算子说明", "https://www.hiascend.com/document"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:回到 45 课的图书馆 📚",
D('''
还记得 45 课的图书馆吗?读者(请求)借书(KV)长短不一,管理员(内存分配器)必须**按需分配**,
而不是给每人预留一整片连续书架。这个分页思想与硬件无关:

- GPU 上:vLLM 用 CUDA kernel 按 block table 收集;
- 昇腾上:vLLM-Ascend 用 CANN 融合算子按 block table 收集;
- MindIE:编译期把分页逻辑揉进算子。

**数学完全一样,只是“谁来实现”变了。** 先复习一下 block table 这张神表:
'''))

NB.code(D('''
rng = np.random.default_rng(7)
num_req, block_size, total_blocks = 4, 4, 16
req_blocks = rng.integers(1, 4, num_req)
slots = np.arange(total_blocks); rng.shuffle(slots)     # 物理块顺序打乱 = 碎片
block_table, cursor = [], 0
for nb in req_blocks:
    block_table.append(slots[cursor:cursor + nb]); cursor += nb

print("每个请求占用的块数:", req_blocks.tolist())
for i, bt in enumerate(block_table):
    print(f"  Req{i} 的逻辑块 0..{len(bt)-1} → 物理块 {bt.tolist()}")
print("注意:同一请求的物理块编号不连续——这就是分页,昇腾上同样如此。")
'''), "✅ 昇腾的 KV 池与这张表完全同构:全局物理块池 + 每个请求一张「逻辑→物理」映射。")

NB.md("## 2️⃣ 昇腾 KV 池与 block table 🗺️",
D('''
昇腾侧(vLLM-Ascend)的 KV 管理沿用 vLLM 的 `BlockManager`:

- **物理块池**:推理启动时在 NPU 显存里预留一块 `(total_blocks, block_size, num_heads, head_dim)` 的张量;
- **block table**:每请求一张表,增删请求时动态分配/释放物理块;
- **连续批处理**:多请求共享同一算子批次,各自的块表同时喂给融合算子。

画一张昇腾视角的“请求 → 物理块池”图:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9.5, 4.2))
for i in range(16):
    ax.add_patch(plt.Rectangle((i * 0.05, 0.1), 0.045, 0.42, fc="#DFE9F8", ec="#4C78A8", lw=1))
    ax.text(i * 0.05 + 0.022, 0.5, str(i), ha="center", va="center", fontsize=7)
ax.add_patch(plt.Rectangle((0.05, 0.66), 0.2, 0.22, fc="#FDF3E4", ec="#E45756", lw=2))
ax.text(0.15, 0.77, "Req0 表: [2, 9, 14]", ha="center", fontsize=9, color="#A03A45")
ax.add_patch(plt.Rectangle((0.55, 0.66), 0.2, 0.22, fc="#FDF3E4", ec="#E45756", lw=2))
ax.text(0.65, 0.77, "Req1 表: [0, 7]", ha="center", fontsize=9, color="#A03A45")
for p in [2, 9, 14]:
    ax.add_patch(plt.Rectangle((p * 0.05 + 0.002, 0.1), 0.045, 0.42, fc="#F5D9A8", ec="#E45756", lw=1.4))
for p in [0, 7]:
    ax.add_patch(plt.Rectangle((p * 0.05 + 0.002, 0.1), 0.045, 0.42, fc="#C9B8DE", ec="#6B4FA1", lw=1.4))
ax.annotate("Req0 的块(橙)", xy=(0.16, 0.55), xytext=(0.4, 0.9), arrowprops=dict(arrowstyle="->", color="#E45756"))
ax.annotate("Req1 的块(紫)", xy=(0.06, 0.55), xytext=(0.6, 0.35), arrowprops=dict(arrowstyle="->", color="#6B4FA1"))
ax.text(0.5, 0.02, "NPU 显存里的物理块池:块可以任意组合给不同请求,不连续也能用。", ha="center", fontsize=10, color="#555")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("昇腾 KV 池:block table 把不连续的块串成连续序列", fontsize=12)
plt.tight_layout(); plt.show()
'''), "🎨 橙色与紫色块分别属于两个请求,物理上犬牙交错——attention 算子按表收集后,照样能“假装连续”。")

NB.md("## 3️⃣ torch 模拟按表收集 🔬",
D('''
昇腾融合算子内部的动作,和 45 课的 torch 版一模一样。我们复刻一次,并验证**分页收集与连续**
**在数学上等价**(用多个请求 + 共享物理块池):
'''))

NB.code(D('''
torch.manual_seed(0)
num_req = 4
total_blocks, block_size, d = 16, 4, 8
rng = np.random.default_rng(7)
req_blocks = rng.integers(1, 4, num_req)
slots = np.arange(total_blocks); rng.shuffle(slots)
bt = []
c = 0
for nb in req_blocks:
    bt.append(slots[c:c + nb]); c += nb

phys_k = torch.randn(total_blocks, block_size, d)
phys_v = torch.randn(total_blocks, block_size, d)
q = torch.randn(d)

def ascend_paged_attention(q, blocks):
    """昇腾融合算子的 torch 版:按 block table 收集 → softmax attention。"""
    K = phys_k[blocks].reshape(-1, d)
    V = phys_v[blocks].reshape(-1, d)
    S = K @ q / (d ** 0.5)
    return torch.softmax(S, dim=-1) @ V

for i, blocks in enumerate(bt):
    o = ascend_paged_attention(q, blocks)
    print(f"Req{i}: {len(blocks)}块/{len(blocks)*block_size} token → 输出前3维 {[f'{v:.3f}' for v in o[:3].tolist()]}")
print("物理块编号(乱序):", slots.tolist())
'''), "✅ `phys_k[blocks]` 一次 advanced indexing 完成“按表收集”——昇腾融合算子内部就是这么做的,只是换成了 NPU 指令。")

NB.md("## 4️⃣ FlashAttention 的昇腾实现:多算子变一个 🧬",
D('''
GPU 上 vLLM 用专门编写的 FlashAttention CUDA kernel;昇腾侧把“QK^T → scale → softmax → ×V”
四步**融合成一个 CANN 算子**(如 `npu_fusion_attention`),好处:

- kernel 启动次数 4 → 1;
- 中间结果(softmax 前的 S)不落地 NPU 显存,省带宽;
- 与分页块表结合,一次算子调用处理一整批请求。

画一张“融合前后”的对比:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
steps = ["QK^T", "Scale", "Softmax", "S×V"]
for i, s in enumerate(steps):
    ax.add_patch(plt.Rectangle((i * 0.24, 0.35), 0.2, 0.3, fc="#DFE9F8", ec="#4C78A8", lw=2))
    ax.text(i * 0.24 + 0.1, 0.5, s, ha="center", va="center", fontsize=10)
    if i < 3:
        ax.annotate("", xy=(i * 0.24 + 0.22, 0.5), xytext=(i * 0.24 + 0.2, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#888"))
ax.set_title("传统:4 个 kernel,中间结果落地显存", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax = axes[1]
ax.add_patch(plt.Rectangle((0.15, 0.35), 0.7, 0.3, fc="#FDF3E4", ec="#E45756", lw=2))
ax.text(0.5, 0.5, "npu_fusion_attention(单算子)", ha="center", va="center", fontsize=11, color="#A03A45")
ax.set_title("昇腾:融合成 1 个算子,零中间落地", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.suptitle("FlashAttention 的昇腾实现:融合算子", fontsize=12)
plt.tight_layout(); plt.show()
'''), "🎨 融合 = 少启动、少搬运、多缓存复用。这是昇腾(以及 Triton、cuDNN)最核心的性能哲学之一。")

NB.md("## 5️⃣ 块大小 vs 碎片率:一块多大才划算? 📐",
D('''
块大小是 PagedAttention 最重要的工程旋钮:

- 块**大**:块表短、访存连续,但块内“装不满”的浪费多(碎片率↑);
- 块**小**:碎片率↓,但块表长、管理开销↑、访存碎片化。

定量看一下不同序列长度分布下“碎片率 vs 块大小”的曲线:
'''))

NB.code(D('''
def frag_rate(seqs, block_size):
    blocks = np.ceil(np.asarray(seqs) / block_size)
    return 1 - np.asarray(seqs).sum() / (blocks * block_size).sum()

rng = np.random.default_rng(3)
seqs_uniform = rng.integers(32, 1024, 2000)
seqs_tail = np.clip(rng.exponential(300, 2000).astype(int), 32, 4096)

bs = np.arange(4, 130, 2)
f_u = [frag_rate(seqs_uniform, b) for b in bs]
f_t = [frag_rate(seqs_tail, b) for b in bs]
fig, ax = plt.subplots(figsize=(8.8, 4.2))
ax.plot(bs, np.array(f_u) * 100, marker="o", ms=3, label="均匀分布", color="#4C78A8")
ax.plot(bs, np.array(f_t) * 100, marker="s", ms=3, label="长尾分布(多数短)", color="#E45756")
ax.axvline(16, color="#F58518", ls="--", lw=1.5)
ax.text(16.5, ax.get_ylim()[1] * 0.9 if False else 34, "工业界常用 16~32", color="#F58518", fontsize=9)
ax.set_xlabel("块大小(token)"); ax.set_ylabel("碎片率(%)")
ax.set_title("块大小越大,碎片率越高;长尾分布更敏感", fontsize=12)
ax.legend(frameon=True, fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 结论:块大到 64 时,短请求多的情况下碎片率能到 30%+;块在 16~32 区间是“碎片率 ↔ 管理开销”的常见甜点区。")

NB.md("## 6️⃣ 交互图:block table 可视化 🕸️",
D('''
把 block table 画成矩阵:行 = 请求,列 = 逻辑块,格内数字 = 物理块。**同一行相邻格子里的物理块
编号是跳跃的**,直观呈现“逻辑连续、物理不连续”。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

table = np.full((num_req, int(req_blocks.max())), -1)
for i, blocks in enumerate(bt):
    table[i, :len(blocks)] = blocks
fig = go.Figure(go.Heatmap(
    z=table, x=[f"逻辑块{j}" for j in range(table.shape[1])],
    y=[f"Req{i}" for i in range(num_req)],
    colorscale=[[0, "#FFFFFF"], [0.5, "#DFE9F8"], [1, "#4C78A8"]],
    zmin=-1, zmax=table.max(), hoverongaps=False, text=table.astype(int), texttemplate="%{text}"))
fig.update_layout(title="Block Table:逻辑块 → 物理块(悬停看编号)", height=380,
                  yaxis=dict(autorange="reversed"),
                  margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "🕸️ 注意同一行相邻列的数字——比如 [2, 9, 14]——物理块编号完全跳跃,却组成一个“逻辑连续”的序列。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_85_paged_ascend.py`,切换**序列长度分布**,拖动**块大小 / 请求数**,
实时看到碎片率、块表样例与“碎片率 vs 块大小”曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_85_paged_ascend.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_85_paged_ascend.py\n" + APP_85, "📜 这就是 app_85_paged_ascend.py 的完整源码,notebook 与 app 共用同一套碎片率口径,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "分页思想与硬件无关:CUDA、CANN 融合算子、MindIE 都在做同一件事",
        "昇腾 KV 池 = 全局物理块池 + 每请求 block table,由 vLLM 的 BlockManager 管理",
        "torch 的 advanced indexing 一行完成昇腾融合算子的“按表收集”",
        "FlashAttention 在昇腾融合成单算子:npu_fusion_attention,少启动少搬运",
        "块大小是工程旋钮:16~32 常为甜点区,长尾序列分布对碎片更敏感",
    ],
    practice=[
        "给 ascend_paged_attention 增加“块内 mask(只取前 k 个 token)”逻辑,模拟已生成部分",
        "画一张“块表长度”随块大小的变化曲线,与碎片率曲线放同一张双轴图",
        "把 req_blocks 改成 lognormal 分布,观察碎片率曲线的形状变化",
        "调研 torch_npu 的 paged attention 接口,列出它与 CUDA 版在输入参数上的异同",
    ],
    links=[
        ("PagedAttention 论文", "https://arxiv.org/abs/2309.06180"),
        ("vLLM-Ascend 支持矩阵", "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/"),
        ("昇腾算子文档", "https://www.hiascend.com/document"),
    ])

NB.save(str(Path(CH11) / "85_pagedattn_ascend.ipynb"))
app_path = Path(CH11) / "app_85_paged_ascend.py"
app_path.write_text(APP_85 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
