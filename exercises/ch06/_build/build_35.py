# -*- coding: utf-8 -*-
"""生成 35_parallel_overview.ipynb 与 app_35_parallel_overview.py(教材级重写版)

论文/资料支撑:
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism", arXiv:1909.08053(TP/PP 起源)
- vLLM 分布式推理文档: https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- vLLM 博客: Distributed Inference with vLLM (2025-02-17)
"""
from helpers import D, MEM_ESTIMATE, chapter_cover, wrapup, new_nb, CH06
from pathlib import Path

APP_35_SRC = D('''
# -*- coding: utf-8 -*-
# app_35_parallel_overview.py — DP/TP/PP/EP 分布式并行总览 🗺️
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🗺️ 35 · 并行总览", layout="wide")
st.title("🗺️ 第 35 课 · 分布式并行总览:模型太大,拆开一起干")

st.markdown("""
大模型太大,单卡塞不下、算不动,于是把模型**拆开放到多张卡上**一起干。拆法有四种:
- 🧑‍🤝‍🧑 **DP 数据并行**:每卡一份完整模型,各吃各的数据(切「数据」);
- ✂️ **TP 张量并行**:每卡拿一半权重,一起算一个矩阵(切「权重/矩阵」);
- 🏭 **PP 流水线并行**:按层切开,像流水线一样接力(切「层」);
- 🧩 **EP 专家并行**:MoE 模型把不同「专家」放到不同卡(切「专家」)。
""")

def est_weights_gb(params_b, dtype_bytes):
    # 权重显存:参数量 × 每参数字节数
    return params_b * dtype_bytes

def est_kv_gb(layers, kv_heads, head_dim, ctx, batch, dtype_bytes):
    # KV cache 显存:每 token KV 字节 = 2×L×h_kv×d_head×b,再乘 token 总数
    return 2 * layers * kv_heads * head_dim * dtype_bytes * ctx * batch / 1024 ** 3

MODEL_CFG = {
    "Llama-3-8B":    dict(params=8,  layers=32, kv_heads=8,  head_dim=128),
    "Qwen2.5-32B":   dict(params=32, layers=64, kv_heads=8,  head_dim=128),
    "Llama-3-70B":   dict(params=70, layers=80, kv_heads=8,  head_dim=128),
    "Mixtral-8x7B(EP)": dict(params=47, layers=32, kv_heads=8, head_dim=128),
}

with st.sidebar:
    st.header("🎛️ 参数")
    model = st.selectbox("模型", list(MODEL_CFG.keys()), index=2)
    dtype = st.radio("权重精度", ["fp16 (2B)", "int8 (1B)", "int4 (0.5B)"], index=0)
    tp = st.slider("TP 张量并行", 1, 8, 2, 1)
    pp = st.slider("PP 流水线并行", 1, 8, 1, 1)
    batch = st.slider("并发请求数", 1, 64, 8, 1)
    st.caption("并行度 = 用多少张卡一起干;TP/PP 会把权重摊薄到每卡。")

cfg = MODEL_CFG[model]
params_b = cfg["params"]
dtype_bytes = {"fp16 (2B)": 2, "int8 (1B)": 1, "int4 (0.5B)": 0.5}[dtype]
gpus = tp * pp

weights_total = est_weights_gb(params_b, dtype_bytes)
weights_per = weights_total / gpus            # TP/PP 都摊薄权重
kv_total = est_kv_gb(cfg["layers"], cfg["kv_heads"], cfg["head_dim"], 4096, batch, dtype_bytes)
kv_per = kv_total / tp                        # KV 被 TP 摊薄(每个 DP 组各用各的)
per_card = weights_per + kv_per

c1, c2, c3, c4 = st.columns(4)
c1.metric("总权重显存(GB)", f"{weights_total:.0f}")
c2.metric("权重 / 卡(GB)", f"{weights_per:.1f}")
c3.metric("KV / 卡(GB)", f"{kv_per:.1f}")
c4.metric("每卡合计(GB)", f"{per_card:.1f}")

st.subheader("📊 显存摊薄:并行度越高,每卡负担越轻")
gpus_range = list(range(1, 17))
w_per = [weights_total / g for g in gpus_range]
kv_per_g = [kv_total / min(tp, g) if g >= tp else kv_total / g for g in gpus_range]
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"{g} 卡" for g in gpus_range], y=w_per, name="权重/卡",
                     marker_color="#4C78A8"))
fig.add_trace(go.Bar(x=[f"{g} 卡" for g in gpus_range], y=kv_per_g, name="KV/卡",
                     marker_color="#72B7B2"))
fig.update_layout(title="随卡数增加,每卡显存负担下降(权重被 TP/PP 摊薄)",
                  xaxis_title="总卡数", yaxis_title="显存(GB)", height=420,
                  barmode="stack", margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧭 四种并行,各切什么")
st.markdown("""
| 并行 | 切什么 | 通信模式 | 适用场景 |
|------|--------|----------|----------|
| 🧑‍🤝‍🧑 DP | 数据/请求 | AllReduce(梯度) | 训练、推理扩吞吐 |
| ✂️ TP | 权重矩阵 | AllReduce/AllGather(频繁) | 大模型放不下、低延迟 |
| 🏭 PP | 网络层 | 点对点(稀疏) | 超大模型、层间解耦 |
| 🧩 EP | 专家 | 路由 AllToAll | MoE 稀疏模型 |
""")
st.caption("💡 生产环境常 TP×PP×DP 组合使用(第 41 课细讲)。")
st.markdown("""
> 💡 **结论**:单卡放不下→用 TP/PP 摊权重;单卡算不动(吞吐不够)→用 DP 摊数据。
> 并行度不是越高越好,通信会成为新瓶颈。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 6 章 · 第 35 课配套演示")

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

APP_35 = "%%writefile app_35_parallel_overview.py\n" + APP_35_SRC

NB = new_nb("第 35 课 · 分布式并行总览:DP / TP / PP / EP",
            subtitle="模型太大放不下?从四种并行各自的「切法」讲起,看懂大模型为什么要并行、怎么并行",
            emoji="🗺️")

chapter_cover(NB,
    objectives=[
        "理解四种并行:DP(数据)、TP(权重矩阵)、PP(层)、EP(专家)各自切什么",
        "掌握每种并行的通信模式与适用场景(训练 vs 推理)",
        "用显存估算公式看明白 7B / 70B 为什么单卡放不下、必须并行",
    ],
    toc=[
        ("直觉:一条生产线,四把剪刀", "把「模型」拆开这件事,可以发生在四个不同的维度"),
        ("核心定义与公式", "DP/TP/PP/EP 定义 + 通信模式 + 显存公式符号表"),
        ("最小实现 · 逐行推演", "手写显存估算函数,打印 7B/70B 每卡账"),
        ("数值验证:真机显存账", "GPU 上量 1M 参数 fp16 显存,外推 7B/70B 放不放得下"),
        ("真实规模数字:70B 摊薄曲线", "并行度越高每卡越轻,但通信是新瓶颈"),
        ("与 vLLM 工程实现的关系", "vLLM 的 distributed_serving 与 TP/PP 参数"),
        ("配套 Streamlit 演示", "app_35_parallel_overview.py:选模型/并行度看显存"),
    ],
    links=[
        ("vLLM 官方文档:分布式推理", "https://docs.vllm.ai/en/latest/serving/distributed_serving.html"),
        ("vLLM 博客: Distributed Inference with vLLM", "https://vllm.ai/blog/2025-02-17-distributed-inference"),
        ("Megatron-LM 论文(TP+PP 起源)", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.md("## 1️⃣ 直觉:一条生产线,四把剪刀 ✂️",
D('''
想象一家工厂要生产 100 万件定制衬衫,单条生产线一天根本做不完。聪明的厂长会从**四个不同角度**
把任务拆开:

1. **加机器(数据并行 DP)**:买一模一样的几条生产线,每条各做一批订单——**切的是「订单/数据」**;
2. **分工位(张量并行 TP)**:一条生产线内部,把「裁剪→缝纫→熨烫」拆到几个工位并行——**切的是「工序/矩阵」**;
3. **分车间(流水线并行 PP)**:裁缝、缝纫工、质检员分别待在三个车间,半成品车间间传递——**切的是「层」**;
4. **分专科(专家并行 EP)**:请一批「衬衫专家」「牛仔裤专家」,来料先路由给对应专家——**切的是「专家」**。

大模型的「并行(parallelism)」就是这四把剪刀:把一个巨大的模型,从数据、权重矩阵、层、专家
四个维度分别切开,交给多张 GPU 一起算。这就是 vLLM 文档里 `distributed_serving` 页讲的核心内容:
[分布式推理](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)。
'''))

NB.md("## 2️⃣ 核心定义与公式:四种并行,各切什么 🧭",
D('''
四种并行的本质区别,一句话就能记住——**它们切的是模型的不同维度**:

| 并行 | 英文 | 切什么 | 通信模式 | 主要作用 |
|------|------|--------|----------|----------|
| 🧑‍🤝‍🧑 **DP** | Data Parallel | 数据/请求 | AllReduce(梯度) | 训练扩吞吐;推理负载均衡 |
| ✂️ **TP** | Tensor Parallel | 权重矩阵 | AllReduce / AllGather | 摊薄权重显存、降单卡算力需求 |
| 🏭 **PP** | Pipeline Parallel | 网络层 | 点对点(稀疏) | 摊薄权重,但引入「气泡」 |
| 🧩 **EP** | Expert Parallel | 专家(FFN 子网络) | AllToAll 路由 | MoE 模型把专家分布到多卡 |

一个直观的取舍:**TP 通信最频繁**(每个矩阵乘都要 allreduce),**PP 通信最稀疏**(层间才传一次),
**DP 之间几乎不通信**(推理时各干各的,只有训练梯度要同步)。所以工程上——
TP 适合「单卡放不下、又要低延迟」,PP 适合「模型大到单节点都放不下」,DP 适合「提升吞吐」。

**显存账公式**(贯穿全章):

$$\\text{权重显存} = P \\times \\text{bytes/参数}, \\qquad
\\text{权重/卡} = \\frac{P \\times \\text{bytes}}{TP \\times PP}$$

| 符号 | 含义 |
|---|---|
| $P$ | 参数量(如 70B = $70\\times10^9$) |
| $\\text{bytes/参数}$ | fp16=2, int8=1, int4=0.5 |
| $TP$ | 张量并行度(切权重) |
| $PP$ | 流水线并行度(切层) |
| $DP$ | 数据并行度(切数据) |
'''))

NB.md("## 3️⃣ 最小实现 · 逐行推演:显存估算函数 🧮",
D('''
把显存账写成函数,对几个主流模型**逐行打印**每卡负担。**每行代码都有注释**:
'''))

NB.code(D('''
def estimate_weights_gb(params_b, dtype_bytes=2):
    # 权重显存(GB):参数量(B 单位)× 每参数字节数( fp16=2, int8=1, int4=0.5 )
    return params_b * dtype_bytes

def estimate_kv_gb(layers, kv_heads, head_dim, ctx_len=4096, batch=1, dtype_bytes=2):
    # KV cache 显存(GB):每 token KV 字节 = 2×L×h_kv×d_head×b,再乘 token 总数。
    # L=层数, h_kv=KV 头数(GQA 后共享), d_head=每头维度, b=每元素字节。
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes   # 每 token KV 字节
    return bytes_per_token * ctx_len * batch / 1024 ** 3               # 总字节 → GB

print(f"{'模型':>16}{'权重GB':>8}{'KV(4k×8并发)GB':>16}{'合计GB':>8}")
for name, (p, L, hkv, d) in {
    "Qwen3-1.7B":  (1.7, 28, 4, 128),
    "Llama-3-8B":  (8, 32, 8, 128),
    "Qwen2.5-32B": (32, 64, 8, 128),
    "Llama-3-70B": (70, 80, 8, 128),
    "Llama-3.1-405B": (405, 126, 8, 128),
}.items():
    w = estimate_weights_gb(p, 2)              # fp16 权重显存(GB)
    kv = estimate_kv_gb(L, hkv, d, 4096, 8)    # 8 并发 × 4k 上下文的 KV(GB)
    print(f"{name:>16}: {w:6.1f} GB + {kv:8.1f} GB = {w + kv:7.1f} GB")
print("\\n→ 70B 光权重就 140GB,RTX 5060 Laptop 只有 8GiB——单卡必然放不下。")
'''), "🎯 看清了吗:Llama-3-70B 权重就要 140GB,再加 KV,单卡根本放不下——**这就是必须并行的根本原因**。")

NB.md("## 4️⃣ 数值验证:真机显存账 ⚡",
D('''
上面的公式估算是把尺子,现在把尺子拿到**真实 GPU** 上量一次:让一个 1M 参数 fp16 张量坐上
RTX 5060,看精确占多少 MB;再线性外推 7B / 70B,对照本机显存容量,判断单卡到底放不放得下。
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 OMP 库冲突防护
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
import torch, gc
from vllm_real import cuda_info

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", cuda_info())
total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if dev == "cuda" else 0.0
print(f"实测本卡显存 = {total_gib:.1f} GiB")

# 真机实测:1M 参数 fp16 权重占多少显存
torch.manual_seed(0)
N = 1_000_000
w = torch.randn(N, device=dev).half()              # (1000000,) fp16 权重
mb16 = w.element_size() * w.numel() / 1024**2      # 每元素字节 × 元素数 → MB
per_param_bytes = w.element_size()                 # 每参数 2 字节
print(f"\\n权重张量 w shape = {tuple(w.shape)}  <- (N={N:,} 个参数) · fp16")
print(f"真机:1M 参数 fp16 权重 = {mb16:.4f} MB(每参数 {per_param_bytes} 字节)")

# 线性外推到真实模型配置,对照本机显存
print("\\n按 fp16(2B/参数)+KV(公式 2·L·h_kv·d_head·b,4k×8并发)估算单卡是否放得下:")
for name, p, L, hkv, d in [("Qwen3-1.7B",1.7,28,4,128), ("Llama-3-8B",8,32,8,128),
                           ("Qwen2.5-32B",32,64,8,128), ("Llama-3-70B",70,80,8,128)]:
    w_gb = p * 2                                  # fp16 权重(GB)
    kv_gb = estimate_kv_gb(L, hkv, d, 4096, 8)    # KV 公式
    fits = (w_gb + kv_gb) < total_gib             # 放不放得下
    print(f"  {name:>12}: 权重 {w_gb:6.1f} GB + KV {kv_gb:5.1f} GB = {w_gb+kv_gb:6.1f} GB"
          f"  -> 单卡{'放得下' if fits else '放不下'}")

gc.collect(); torch.cuda.empty_cache()
'''), "⚡ **真机数字**:8GiB 显存下,7B 的 fp16 权重(14GB)单卡就放不下,更别说 70B(140GB)——这就是必须 TP/PP 摊权重的根本原因。")

NB.md("## 5️⃣ 真实规模数字:70B 摊薄曲线 📉",
D('''
把并行度一路加大,每卡显存怎么变?**TP/PP 越多,每卡权重越少;KV 主要被 TP 摊薄**。
注意曲线会一路走低,但代价是通信量上升(第 36-37 课细讲),所以并不是卡越多越好。
'''))

NB.code(D('''
import numpy as np
import torch
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 16.0

params_b, dtype_bytes, kv_total = 70, 2, 60     # 70B 模型, fp16, 并发下 KV≈60GB
gpus_range = list(range(1, 17))                 # 1..16 卡
w_per = [params_b * dtype_bytes / g for g in gpus_range]      # 权重 / 卡
kv_per = [kv_total / g for g in gpus_range]                   # KV / 卡(理想线性)

fig = go.Figure()
fig.add_trace(go.Scatter(x=gpus_range, y=w_per, mode="lines+markers",
                         name="权重/卡", line=dict(width=3, color="#4C78A8")))
fig.add_trace(go.Scatter(x=gpus_range, y=kv_per, mode="lines+markers",
                         name="KV/卡", line=dict(width=3, color="#E45756")))
fig.add_trace(go.Scatter(x=gpus_range, y=[w + k for w, k in zip(w_per, kv_per)],
                         mode="lines+markers", name="合计/卡",
                         line=dict(width=3, dash="dot", color="#2F3B52")))
fig.add_hline(y=total_gib, line_dash="dash", line_color="gray",
              annotation_text=f"本机显存 {total_gib:.0f}GiB", annotation_position="top right")
fig.update_layout(title="70B 模型:并行度越高,每卡显存负担越低",
                  xaxis_title="总卡数", yaxis_title="显存(GB)", height=430,
                  legend=dict(orientation="h", y=1.12),
                  margin=dict(l=10, r=10, t=60, b=10))
fig
print("本机显存(实测):", round(total_gib, 1), "GiB")
print("8 卡时每卡权重:", params_b * dtype_bytes / 8, "GB")
print("16 卡时每卡权重:", params_b * dtype_bytes / 16, "GB")
print("→ 8 卡时 70B 权重每卡降到 17.5GB,16 卡时 8.75GB;要落到虚线(本机 8GiB)以下还需再摊 KV/精度。")
'''), "🎨 悬停看数值:并行度越高每卡越轻,但要落到本机 8GiB 虚线以下仍需继续摊薄——这就是多卡并行的意义。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:TP/PP 两个参数 🚀",
D('''
vLLM 把并行做成了**两个命令行参数**,与 Megatron-LM 的算法一一对应
(见 [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)):

```bash
# 单节点多卡:模型太大一张卡装不下 → 张量并行
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 4

# 多节点:单节点都装不下 → TP(节点内) + PP(节点间)
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2
```

工程事实:

1. **TP 是默认主力**:vLLM 实现的就是 Megatron-LM 的张量并行算法(第 37 课展开);
   模型放不下但单节点够 → 用 TP;
2. **PP 补充跨节点**:TP 的 AllReduce 跨节点太慢,所以「TP 设节点内卡数、PP 设节点数」
   是最常见配置;
3. **推理中的「超线性收益」**:vLLM 博客实测 TP1→TP2 时 KV cache 块可增加 13.9×,
   吞吐提升 3.9×——因为并行后省下的显存全给了 KV cache,这也呼应第 5 节的摊薄曲线。

> 📄 算法起源是 [Megatron-LM (arXiv:1909.08053)](https://arxiv.org/abs/1909.08053):
> 每层用 f/g 两个通信原语,把权重矩阵切开、用 AllReduce 拼回。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:选模型与并行度,实时看显存 🎛️",
D('''
运行同目录下的 `app_35_parallel_overview.py`,可以**选择模型、切换精度、拖动 TP/PP/并发数**,
每卡显存负担与四种并行对比表实时刷新:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_35_parallel_overview.py
```

浏览器打开 **http://localhost:8501**。建议:把模型切到 Llama-3-70B、fp16,再把 TP 从 1 拖到 8,
观察每卡权重从 140GB 一路掉到 17.5GB;再对比 int8/int4 精度下的差异。完整源码如下:
'''))

NB.code(APP_35, "📜 这就是 app_35_parallel_overview.py 的完整源码,notebook 与 app 共享同一套显存估算函数,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "四种并行按「切什么」区分:DP 切数据、TP 切权重矩阵、PP 切层、EP 切专家",
        "TP 通信最频繁、最省显存;PP 通信最稀疏;DP 推理时几乎不通信、只做负载均衡",
        "权重显存 = 参数量 × 字节数:70B fp16 就要 140GB,单卡必然放不下",
        "KV cache 随并发与上下文暴涨,是推理显存的另一大头",
        "并行度不是越高越好:显存摊薄的同时通信成为新瓶颈(后面几课展开)",
    ],
    practice=[
        "把 KV 估算的 ctx_len 从 4096 改成 128k,重算 KV 显存,体会长上下文压力",
        "给 pyecharts 架构图增加第三个属性「延迟影响」,比较四种并行的相对大小",
        "推导:70B 模型用 int8 时,最少要几张 16GB 卡才能只放权重(不算 KV)",
        "查 vLLM 文档,了解 vLLM 默认在什么时候自动启用 TP(单卡放不下模型时)",
    ],
    links=[
        ("vLLM 分布式推理文档", "https://docs.vllm.ai/en/latest/serving/distributed_serving.html"),
        ("vLLM 博客: Distributed Inference", "https://vllm.ai/blog/2025-02-17-distributed-inference"),
        ("Megatron-LM 论文(TP+PP 起源)", "https://arxiv.org/abs/1909.08053"),
        ("vLLM 官方文档", "https://docs.vllm.ai"),
    ])

NB.save(str(Path(CH06) / "35_parallel_overview.ipynb"))

app_path = Path(CH06) / "app_35_parallel_overview.py"
app_path.write_text(APP_35_SRC + "\n", encoding="utf-8")
print(f"[ok] {app_path}")