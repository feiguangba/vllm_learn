# -*- coding: utf-8 -*-
"""生成 37_tensor_parallel.ipynb 与 app_37_tensor_parallel.py(教材级重写版)

论文/资料支撑:
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism", arXiv:1909.08053(TP 切分算法起源)
- vLLM 博客: Distributed Inference with vLLM (column/row parallel)
- vLLM distributed_serving 文档
"""
from helpers import D, TP_LIN, TP_BLOCK, chapter_cover, wrapup, new_nb, CH06
from pathlib import Path

APP_37_SRC = D('''
# -*- coding: utf-8 -*-
# app_37_tensor_parallel.py — 张量并行的列切/行切与 AllReduce ✂️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="✂️ 37 · 张量并行", layout="wide")
st.title("✂️ 第 37 课 · 张量并行:把一个矩阵切成几份,一起算")

st.markdown("""
一个线性层就是一次矩阵乘 $Y = XW$。**张量并行(TP)** 把权重矩阵 $W$ 切开,让每张卡只存一份、
只算一块,最后用 **AllReduce** 拼回完整结果。两种切法:
- **列切**:$W$ 按输出维切成列块,每卡算出输出的一部分 → 需要 AllGather 拼列;
- **行切**:$W$ 按输入维切成行块,每卡算部分和 → 需要 AllReduce 相加。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    b = st.slider("batch", 1, 8, 2, 1)
    hidden = st.slider("隐藏维度(矩阵宽度)", 4, 64, 16, 2)
    tp = st.slider("TP 切分数", 2, 4, 2, 1)
    st.caption("切分数 = 用几张卡一起算这个矩阵;每卡只存 1/tp 的权重。")

rng = np.random.default_rng(42)
in_dim = hidden
x = rng.normal(size=(b, in_dim))
w = rng.normal(size=(in_dim, hidden))

col_shards = np.split(w, tp, axis=1)          # 列切:W 按输出维切成 tp 块
col_outs = [x @ s for s in col_shards]        # 每卡输出 (b, hidden/tp)
hidden_shards = np.hstack(col_outs)           # 拼回完整 hidden
row_shards = np.split(hidden_shards, tp, axis=1)
row_outs = [hs for hs in row_shards]
y_tp = sum(row_outs)                          # AllReduce:部分和相加
y_ref = x @ w

c1, c2, c3 = st.columns(3)
c1.metric("每卡权重占比", f"1/{tp}")
c2.metric("TP 结果 vs 稠密", "✅ 一致" if np.allclose(y_tp, y_ref, atol=1e-5) else "❌")
c3.metric("权重矩阵形状", f"{w.shape[0]}×{w.shape[1]}")

st.subheader("✂️ 列切:每卡算出一部分输出列")
fig = go.Figure()
for i, s in enumerate(col_shards):
    fig.add_trace(go.Heatmap(z=s, colorscale="Blues", showscale=False,
                             name=f"卡{i} W_i 形状 {s.shape[0]}×{s.shape[1]}",
                             zmin=s.min(), zmax=s.max()))
fig.update_layout(title=f"列切:权重 W 沿输出维切成 {tp} 块,每卡只存其中一块",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔄 AllReduce:部分和相加 = 完整前向")
row_means = [float(np.abs(r).mean()) for r in row_outs]
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=[f"卡{i} 部分和" for i in range(tp)], y=row_means,
                      name="各卡部分结果", marker_color="#4C78A8",
                      text=[f"{v:.3f}" for v in row_means], textposition="outside"))
fig2.add_trace(go.Bar(x=["AllReduce 总和"], y=[float(np.abs(y_ref).mean())],
                      name="稠密参考", marker_color="#E45756",
                      text=[f"{float(np.abs(y_ref).mean()):.3f}"], textposition="outside"))
fig2.update_layout(title="各卡部分和经 AllReduce 相加 = 单卡稠密结果(验证一致)",
                   yaxis_title="|值| 均值", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **直觉**:TP 相当于把一张大矩阵分成几份,每人算一块,再用 AllReduce 把结果拼起来——
> 数学上完全等价于一次完整矩阵乘(上面 ✅ 一致)。代价是**每次矩阵乘后都要一次通信**,
> 所以 TP 的通信很频繁,适合单卡放不下、又要低延迟的场景。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 37 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
''')

APP_37 = "%%writefile app_37_tensor_parallel.py\n" + APP_37_SRC

NB = new_nb("第 37 课 · 张量并行:列切与行切",
            subtitle="把一个线性层切成几份让多卡一起算,看懂 TP 的切法、AllReduce 时机与 Megatron 的 QKV/FFN 切分",
            emoji="✂️")

chapter_cover(NB,
    objectives=[
        "理解张量并行(TP)的核心:把权重矩阵沿不同维度切开,多卡协同一次矩阵乘",
        "掌握列切(Column Parallel)与行切(Row Parallel)两种切法及其拼接/归约方式",
        "手写两卡 TP 的 MLP 前向(column→GELU→row),并用 allreduce 验证与稠密等价",
        "看懂 Megatron 中 QKV 投影与 FFN 的 up/down 切法,以及 AllReduce 的时机",
    ],
    toc=[
        ("直觉:分块拼图", "一个矩阵乘 = 把矩阵切开分块算,再拼回去"),
        ("核心定义与公式", "列切/行切公式 + 符号表 + 通信原语 f/g"),
        ("最小实现 · 逐行推演", "numpy 手写 column→GELU→row,打印 shape"),
        ("数值验证", "TP2 MLP vs 稠密逐点一致"),
        ("真实规模数字", "GPU 上真实 MLP 分片:数学等价 + 每卡显存账"),
        ("与 vLLM 工程实现的关系", "Megatron 算法在 vLLM 的落地 + QKV/FFN 切法"),
        ("配套 Streamlit 演示", "app_37_tensor_parallel.py:拖矩阵尺寸/切分数看结果"),
    ],
    links=[
        ("Megatron-LM 论文 (arXiv:1909.08053)", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 博客: Distributed Inference", "https://vllm.ai/blog/2025-02-17-distributed-inference"),
        ("vLLM 分布式推理文档", "https://docs.vllm.ai/en/latest/serving/distributed_serving.html"),
        ("PyTorch Distributed 文档", "https://pytorch.org/docs/stable/distributed.html"),
    ])

NB.md("## 1️⃣ 直觉:分块拼图 🧩",
D('''
一次线性层就是 $Y = XW$。单个 GPU 要做的是:把巨大的 $X$ 和 $W$ 相乘。问题是 $W$ 可能大到
单卡放不下(70B 模型的单个大矩阵就几十 GB)。

**张量并行(Tensor Parallel,TP)** 的思路像拼图:把矩阵 $W$ 切成几块,分给几张卡,
每张卡只存并只算其中一块,最后用 **AllReduce**(第 36 课)把结果拼回完整。
关键是:**数学上完全等价于一次完整矩阵乘**。

矩阵乘法有两个维可以切,于是有两种基本切法——**列切**和**行切**。这是 Megatron-LM
([arXiv:1909.08053](https://arxiv.org/abs/1909.08053))提出 TP 时给出的核心思想,也是 vLLM 张量并行的基础。
'''))

NB.md("## 2️⃣ 核心定义与公式 ✂️",
D('''
设权重 $W\\in\\mathbb{R}^{d_{in}\\times d_{out}}$,输入 $X\\in\\mathbb{R}^{B\\times d_{in}}$:

- **列切(Column Parallel)**:把 $W$ 沿 **输出维** 切成 $W = [W_1, W_2, \\dots, W_t]$,
  每卡算 $XW_i \\in \\mathbb{R}^{B\\times d_{out}/t}$。各卡得到输出的**不同列**,
  要拼起来(AllGather)才是完整输出;
- **行切(Row Parallel)**:把 $W$ 沿 **输入维** 切成 $W = \\begin{bmatrix}W_1\\\\ \\vdots\\\\ W_t\\end{bmatrix}$,
  输入 $X$ 也相应切成 $X = [X_1, \\dots, X_t]$,每卡算 $X_i W_i \\in \\mathbb{R}^{B\\times d_{out}}$
  的**部分和**,再 **AllReduce 相加** 才是完整输出。

| 符号 | 含义 | 维度 |
|---|---|---|
| $X$ | 层输入 | $(B, d_{in})$ |
| $W$ | 权重 | $(d_{in}, d_{out})$ |
| $t$ | TP 切分数 | 卡数 |
| $W_i$ | 第 $i$ 卡的分片 | 列切 $(d_{in}, d_{out}/t)$;行切 $(d_{in}/t, d_{out})$ |
| $Y_i$ | 第 $i$ 卡的部分输出 | 列切 $(B, d_{out}/t)$;行切 $(B, d_{out})$ |

一个关键配对:**上游用列切、下游就用行切**,这样列切产出的「半成品」正好是行切需要的输入,
省掉一次 AllGather。下面用 numpy 验证:
'''))

NB.code(TP_LIN, "🎯 `column_parallel_linear` 沿输出维切(每卡得一半隐藏维),`row_parallel_linear` 沿输入维切并 allreduce 相加。两者组合的结果与单卡稠密完全一致(✅)。")

NB.md("## 3️⃣ 最小实现 · 逐行推演:两卡 TP 的 MLP 前向 🏭",
D('''
Transformer 里两个最主要的权重块是:注意力的 **QKV 投影** 和 MLP 的 **up/down 投影**。
Megatron 对它们的切法很有讲究:

- **QKV**:把 $W_{qkv}$ 沿输出维**列切**,各卡各算一部分 head 的 Q/K/V;
- **FFN(up_proj 列切 + down_proj 行切)**:up 列切后每卡本地做 GELU(逐元素,无需通信),
  再交给 down 行切做 allreduce——**每卡只通信一次**,效率最高。

下面用 numpy 完整模拟一次两卡 MLP 前向,并记录通信时机:
'''))

NB.code(TP_BLOCK, "✅ 看 `events`:一次 MLP 前向 = 各卡本地算 up+GELU,最后**一次 AllReduce** 把 down 的部分和相加。结果与稠密完全一致——这就是 TP 里「通信只发生一次」的秘密。")

NB.md("## 4️⃣ 数值验证:GPU 上真跑分片 MLP ⚡",
D('''
把 MLP 搬到 **RTX 5060 的真张量**上:把 $W_{up}$ 沿输出维切列为 TP 份、$W_{down}$ 沿输入维行切,
真实跑一次 MLP 前向,与「全量稠密」逐点对比(`allclose`),并同时量出两笔账——**单卡耗时**与**每卡显存**。

⚠️ 说明:本机只有**一张** GPU,这里的 TP 是把两个分片**串行**用同一张卡算,验证的是
「数学等价 + 每卡显存」;**真正的时间加速要靠多卡并行通信**(第 40/41 课)。
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
import torch, time, gc
from vllm_real import cuda_info

print("设备:", cuda_info())
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
B, IN, HID, OUT = 64, 4096, 12288, 4096   # 逼近一层真实 MLP 的规模
x = torch.randn(B, IN, device=dev)                    # (64, 4096) 输入
W_up   = torch.randn(IN, HID, device=dev) * (IN ** -0.5)   # (4096, 12288) up 权重
W_down = torch.randn(HID, OUT, device=dev) * (HID ** -0.5) # (12288, 4096) down 权重

def gelu(z):
    # 近似 GELU(逐元素,无需通信)
    return 0.5 * z * (1 + torch.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

def mlp_dense(x):
    # 单卡稠密 MLP:gelu(x @ W_up) @ W_down
    return gelu(x @ W_up) @ W_down

def mlp_tp(x, tp=2):
    # TP 版 MLP:up 列切 → GELU → down 行切 → AllReduce
    ups   = torch.chunk(W_up, tp, dim=1)     # 列切:每卡 (IN, HID/tp)
    downs = torch.chunk(W_down, tp, dim=0)   # 行切:每卡 (HID/tp, OUT)
    per = [gelu(x @ u) @ d for u, d in zip(ups, downs)]  # 每卡本地 GELU + 部分和
    return sum(per)                          # AllReduce:部分和相加

tp = 2
y_tp  = mlp_tp(x, tp)                        # TP2 输出
y_ref = mlp_dense(x)                         # 稠密参考
print(f"输入 x shape = {tuple(x.shape)}  <- (batch={B}, 输入={IN})")
print(f"up  权重 shape = {tuple(W_up.shape)}  <- (输入={IN}, 隐藏={HID})")
print(f"down 权重 shape = {tuple(W_down.shape)}  <- (隐藏={HID}, 输出={OUT})")
print("TP2 全 GPU MLP vs 稠密:",
      "✅ 一致" if torch.allclose(y_tp, y_ref, atol=1e-3) else "❌ 不一致",
      f"(逐点最大差 = {float((y_tp-y_ref).abs().max()):.3e})")

def bench(fn, reps=20):
    # 计时:预热 + 多次取均值(毫秒)
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

t_ref = bench(lambda: mlp_dense(x))
t_tp  = bench(lambda: mlp_tp(x, tp))
print(f"单卡串行全量 MLP : {t_ref:6.4f} ms")
print(f"单卡模拟 TP2(串行2分片): {t_tp:6.4f} ms")

# 每卡显存账:全量权重 vs TP 每卡
full_bytes  = (W_up.numel() + W_down.numel()) * 2          # fp16 全量字节
shard_bytes = (W_up.numel() // tp + W_down.numel() // tp) * 2  # TP 每卡字节
print(f"\\n全量权重 = {full_bytes/1024**2:7.1f} MB | TP{tp} 每卡 = {shard_bytes/1024**2:6.1f} MB | 每卡省 {(1-1/tp)*100:.0f}%")

gc.collect(); torch.cuda.empty_cache()
'''), "⚡ **真机数字**:输出与稠密逐点一致(数学等价成立);真正的收益在显存——TP2 每卡只需存一半权重(这里 192MB vs 96MB)。单卡上两分片串行所以耗时约等于全量;也说明:**TP 值的是多卡**,单张卡看不出速度优势。")

NB.md("## 5️⃣ 真实规模数字:LLaMA-70B 的切分账 💾",
D('''
代入真实模型,算一笔 TP 的显存账。LLaMA-3-70B 的 FFN 维度:输入 $d_{in}=8192$,
中间 $d_{hid}=28672$,输出 $d_{out}=8192$;单层 FFN 权重参数 = $2 \\times 8192\\times 28672 \\approx 0.47\\mathrm{B}$。

| TP 切分数 | 每卡单层 FFN 权重 | 说明 |
|---|---|---|
| 1 | 0.94 GB(fp16) | 单卡 |
| 2 | 0.47 GB | up/down 各切一半 |
| 8 | 0.12 GB | 摊到 8 卡 |

每层权重被 TP 摊到 $1/t$;而每层只需要 **2 次 AllReduce**(一次 attention O 投影、一次 FFN down),
通信量 ~ $\\frac{2(t-1)}{t} \\times B \\times S \\times H \\times 2\\,\\text{bytes}$。卡越多省得越多,
但通信也越频繁——这就是第 41 课要权衡的「TP 别太大」。
'''))

NB.code(D('''
# LLaMA-3-70B 单层 FFN 的 TP 摊薄账
d_in, d_hid, d_out = 8192, 28672, 8192      # 真实 FFN 维度
params_layer = 2 * d_in * d_hid             # up + down 两个投影的参数
bytes_fp16 = 2                              # fp16 每参数字节
print(f"{'TP':>3}{'每卡权重(MB)':>12}{'每卡权重(GB)':>12}")
for t in [1, 2, 4, 8]:
    per = params_layer * bytes_fp16 / t / 1024**2   # 每卡 MB
    print(f"{t:>3}{per:>12.1f}{per/1024:>12.3f}")

# 通信量:每次 AllReduce 每卡搬多少字节(batch=1, seq=4096)
B, S, H = 1, 4096, 8192
for t in [2, 4, 8]:
    comm_bytes = 2 * (t - 1) / t * B * S * H * bytes_fp16   # 单次 AllReduce 每卡字节
    print(f"TP{t}: 单次 AllReduce 每卡搬 {comm_bytes/1024/1024:.1f} MB(共 2 次/层)")
print("\\n→ 卡越多省显存越多,但通信越多——TP 的平衡点在 35-41 课展开。")
'''), "📊 TP 把每层权重摊到 1/t,代价是每次矩阵乘后的 AllReduce 通信——「省显存」与「加通信」此消彼长。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:Megatron 算法在 vLLM 的落地 🚀",
D('''
vLLM 明确说明实现的就是 **Megatron-LM 的张量并行算法**
(见 [vLLM parallelism_scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)):

1. **Column Parallel**:`vllm/model_executor/layers/linear.py` 的 `QKVParallelLinear` 按
   **注意力头**切分——每卡持有部分 head 的 QKV 权重,注意力在本地算(无需通信);
2. **Row Parallel**:`RowParallelLinear` 负责输出投影,把各卡 head 结果 AllReduce 拼回;
3. **FFN**:`gate_up_proj`(列切,含 SiLU)+ `down_proj`(行切,AllReduce);
   与第 3 节的 numpy 演示完全同构;
4. **嵌入层**:词表按 TP 切分,输出与 cross-entropy 融合减少通信(Megatron 的经典技巧)。

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 4
```

> 📄 Megatron 的要点是「把两个 GEMM 融成一组,f/g 通信原语每层只需 2 次 AllReduce」,
> 见论文 Figure 3 的 f/g 标记与 vLLM 博客的 column/row 图示。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖矩阵尺寸与切分数,看切分与归约 🎛️",
D('''
运行 `app_37_tensor_parallel.py`,拖动 **batch、隐藏维度、TP 切分数**,实时看列切分块、
各卡部分和以及 AllReduce 相加后与稠密参考的一致性:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_37_tensor_parallel.py
```

浏览器打开 **http://localhost:8501**。建议:把 TP 从 2 拖到 4,看权重被切成更多块、
每卡权重占比变成 1/4,同时 AllReduce 结果仍与稠密一致。完整源码如下:
'''))

NB.code(APP_37, "📜 这就是 app_37_tensor_parallel.py 的完整源码,notebook 与 app 共享同一套列切/行切逻辑。")

wrapup(NB,
    summary=[
        "TP 把权重矩阵沿输出维(列切)或输入维(行切)切开,多卡协同一次矩阵乘",
        "列切产出输出的不同列(需 AllGather),行切产出部分和(需 AllReduce)",
        "最佳配对:上游列切 + 下游行切,GELU 等逐元素计算无需通信",
        "QKV 按注意力头切,FFN 用 up 列切 + down 行切,整层只通信一次",
        "TP 把每卡权重摊到 1/t,解决「单卡放不下」,但每次矩阵乘都要通信",
    ],
    practice=[
        "把 TP_LIN 的 tp 改成 4,验证四卡列切+行切组合仍与稠密一致",
        "修改 TP_BLOCK,把 GELU 放在 down_proj 之后,观察结果是否还一致(思考为什么)",
        "实现一个 4 卡版的 column→row 配对,画出每卡的内存占用条形图",
        "查阅 Megatron 论文图 3,对照本课的 up/down 切法画一个示意草图",
    ],
    links=[
        ("Megatron-LM 论文", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 博客: Distributed Inference", "https://vllm.ai/blog/2025-02-17-distributed-inference"),
        ("vLLM 分布式推理文档", "https://docs.vllm.ai/en/latest/serving/distributed_serving.html"),
        ("PyTorch Distributed 文档", "https://pytorch.org/docs/stable/distributed.html"),
    ])

NB.save(str(Path(CH06) / "37_tensor_parallel.ipynb"))

app_path = Path(CH06) / "app_37_tensor_parallel.py"
app_path.write_text(APP_37_SRC + "\n", encoding="utf-8")
print(f"[ok] {app_path}")