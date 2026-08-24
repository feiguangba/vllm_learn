# -*- coding: utf-8 -*-
"""生成 41_parallel_combination.ipynb 与 app_41_combination.py(教材级重写版)

论文/资料支撑:
- Shoeybi et al., "Megatron-LM", arXiv:1909.08053(TP+PP+DP 组合训练)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM", arXiv:2104.04473(PTD-P,万亿参数)
- vLLM parallelism_scaling 文档(TP×PP 配置建议)
- Brown et al., "GPT-3: Language Models are Few-Shot Learners", arXiv:2005.14165
"""
from helpers import D, COMBO_EST, chapter_cover, wrapup, new_nb, CH06
from pathlib import Path

APP_41_SRC = D('''
# -*- coding: utf-8 -*-
# app_41_combination.py — TP × PP × DP 组合估算 🧩
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🧩 41 · 并行组合", layout="wide")
st.title("🧩 第 41 课 · 并行组合:TP × PP × DP,一手算清通信/显存/气泡")

st.markdown("""
现实里三种并行**组合使用**:**TP 摊矩阵、PP 摊层、DP 摊数据**。本页给你一套估算公式,
实时算出指定组合下的**每卡权重显存、KV 显存、TP 通信量、PP 气泡占比**,并列出
固定总卡数下所有合法组合的对比表,帮你理解「为什么这样配」。
""")

def estimate_combo(params_b, layers, hidden, gpus, tp, pp, dp, ctx=4096, batch=8,
                   dtype_bytes=2, m=8, kv_heads=8, head_dim=128):
    weights_gb = params_b * dtype_bytes / tp / pp
    kv_gb = 2 * layers * kv_heads * head_dim * dtype_bytes * ctx * batch / 1024 ** 3 / tp
    ar_mb = 2 * (layers / pp) * 2 * (tp - 1) / tp * batch * ctx * hidden * dtype_bytes / 1e6
    bubble = (pp - 1) / (m + pp - 1)
    return weights_gb, kv_gb, ar_mb, bubble

with st.sidebar:
    st.header("🎛️ 参数")
    params_b = st.slider("模型参数量(B)", 7, 405, 70, 1)
    layers = st.slider("总层数 L", 32, 96, 80, 1)
    hidden = st.slider("隐藏维度 hidden", 4096, 8192, 8192, 256)
    gpus = st.slider("总卡数", 2, 32, 8, 1)
    tp = st.slider("TP", 1, 8, 2, 1)
    pp = st.slider("PP", 1, 8, 2, 1)
    m = st.slider("micro-batch m", 4, 32, 8, 1)
    st.caption("DP 自动算为 gpus/(TP×PP)。TP 需能整除总卡数;TP/PP 越大越省显存但通信/气泡更高。")

dp = gpus // (tp * pp)
w_gb, k_gb, ar_mb, bubble = estimate_combo(params_b, layers, hidden, gpus, tp, pp, dp, m=m)
valid = (tp * pp * dp == gpus)

c1, c2, c3, c4 = st.columns(4)
c1.metric("组合", f"TP{tp}×PP{pp}×DP{dp}" + ("" if valid else " (无效)"))
c2.metric("每卡权重(GB)", f"{w_gb:.1f}")
c3.metric("每卡 KV(GB)", f"{k_gb:.1f}")
c4.metric("PP 气泡占比", f"{bubble:.1%}")

st.subheader("🧮 每卡负担分解")
fig = go.Figure()
labels = ["每卡权重", "每卡 KV"]
fig.add_trace(go.Bar(x=labels, y=[w_gb, k_gb], marker_color=["#4C78A8", "#72B7B2"],
                     text=[f"{w_gb:.1f}", f"{k_gb:.1f}"], textposition="outside"))
fig.update_layout(title=f"TP{tp}×PP{pp}×DP{dp} 下每卡显存(GB)", yaxis_title="GB",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 所有合法组合对比(纯计算)")
rows = []
for tt in [1, 2, 4, 8]:
    for pp_ in [1, 2, 4, 8]:
        dd = gpus // (tt * pp_)
        if tt * pp_ * dd == gpus:
            w, k, a, bub = estimate_combo(params_b, layers, hidden, gpus, tt, pp_, dd, m=m)
            rows.append(dict(组合=f"TP{tt}×PP{pp_}×DP{dd}", TP=tt, PP=pp_, DP=dd,
                             权重GB=round(w, 1), KV_GB=round(k, 1),
                             TP通信MB=round(a, 1), 气泡=round(bub, 3)))
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

st.markdown("""
> 💡 **选型直觉**:显存紧张→加大 TP/PP;吞吐不足→加大 DP;气泡敏感→少用 PP 或加大 m。
> 实际配置还要看单卡显存容量、总线拓扑与负载特征,本页是「方向正确」的估算,非精确仿真。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 6 章 · 第 41 课配套演示")

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

APP_41 = "%%writefile app_41_combination.py\n" + APP_41_SRC

NB = new_nb("第 41 课 · 并行组合:TP × PP × DP",
            subtitle="三种并行叠加,用一套估算公式算清通信量、显存与气泡,看懂 8 卡该怎样组合",
            emoji="🧩")

chapter_cover(NB,
    objectives=[
        "理解为什么 TP / PP / DP 可以组合使用(它们摊的是不同维度)",
        "掌握组合下的估算公式:每卡权重、KV 显存、TP 通信量、PP 气泡",
        "纯计算列出固定总卡数下的所有合法组合,对比各项开销",
        "理解最优选型的权衡:显存 vs 通信 vs 气泡 vs 吞吐",
    ],
    toc=[
        ("直觉:三层包装的包裹", "TP 摊矩阵、PP 摊层、DP 摊数据,互不冲突可叠加"),
        ("核心定义与公式", "每卡权重 / KV / TP 通信 / PP 气泡,一套公式算清"),
        ("最小实现 · 逐行推演", "手写组合估算函数,打印 8 卡所有组合"),
        ("数值验证", "8 卡组合对比表:没有「全胜」方案,只有取舍"),
        ("真实规模数字", "GPU 实测锚点:2B/参数、GEMM TFLOPS、搬运带宽"),
        ("与 vLLM 工程实现的关系", "vLLM 的 TP×PP 配置建议与 Megatron 3D 并行"),
        ("配套 Streamlit 演示", "app_41_combination.py:拖卡数/TP/PP/DP 实时算"),
    ],
    links=[
        ("Megatron-LM 论文(TP+PP+DP 三组合)", "https://arxiv.org/abs/1909.08053"),
        ("Megatron-LM 万亿参数论文 (arXiv:2104.04473)", "https://arxiv.org/abs/2104.04473"),
        ("vLLM parallelism_scaling 文档", "https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"),
        ("GPT-3 (arXiv:2005.14165)", "https://arxiv.org/abs/2005.14165"),
    ])

NB.md("## 1️⃣ 直觉:三层包装的包裹 📦",
D('''
为什么三种并行能组合?因为它们**摊的是不同维度,互不冲突**:

- **TP 张量并行**:把一个矩阵切开(摊「矩阵」),解决单卡放不下大权重;
- **PP 流水线并行**:把网络层切成舞台(摊「层」),解决模型层数太多;
- **DP 数据并行**:每卡一份完整模型、各吃各的数据(摊「数据」),解决吞吐不够。

就像寄一个大包裹:TP 负责把每个箱子拆成小块(矩阵),PP 负责把一摞箱子分到几个仓库(层),
DP 则是复制好几套包裹分给不同快递员(数据)。三者从不同方向分担,所以能**同时叠加**:

$$G = TP \\times PP \\times DP$$

其中 $G$ 是总卡数。生产环境(如 Megatron-LM、DeepSpeed 的 3D parallelism)正是这么配的。
'''))

NB.md("## 2️⃣ 核心定义与公式 🧮",
D('''
给定模型(参数量 $P$、层数 $L$、隐藏维 $h$)和组合(TP、PP、DP),四项关键开销可估算:

- **每卡权重显存**:$\\dfrac{P \\times \\text{bytes}}{TP \\times PP}$(权重被 TP 和 PP 都摊薄);
- **每卡 KV 显存**:$\\propto \\dfrac{1}{TP}$(KV 主要被 TP 摊薄;每个 DP 组各服务各的请求);
- **每卡 TP 通信量**:$\\propto 2\\times\\dfrac{L}{PP}\\times\\dfrac{2(TP-1)}{TP}\\times B\\times h$
  (PP 下每卡只跑 $L/PP$ 层,每层一次 AllReduce);
- **PP 气泡**:$\\dfrac{PP-1}{m+PP-1}$(第 38 课公式)。

| 符号 | 含义 |
|---|---|
| $P$ | 参数量(70B = 70) |
| $L$ | 总层数(如 80) |
| $h$ | 隐藏维度(如 8192) |
| $TP, PP, DP$ | 三种并行度,$G = TP\\times PP\\times DP$ |
| $B$ | batch / 并发请求数 |
| $\\text{bytes}$ | 每参数字节(fp16=2) |
| $m$ | micro-batch 数(气泡公式用) |

下面用这套公式算一个 70B 模型在 8 卡下的所有组合:
'''))

NB.code(COMBO_EST, "🎯 对比各行:TP 越大权重/KV 越省但 TP 通信越大;PP 越大权重越省但气泡越高;DP 越大吞吐越强但不省显存——**没有免费的午餐**。")

NB.md("## 3️⃣ 数值验证:8 卡组合的取舍 📊",
D('''
把 8 卡下所有合法组合摊开对比,读表的几个结论:

- **要省显存**:优先加大 TP(同时摊掉权重和 KV);
- **要低通信延迟**:TP 别太大(TP 通信最频繁);
- **要低气泡**:PP 别太大,或用更大 m;
- **要吞吐**:DP 大(但受限于总卡数减掉 TP/PP 后剩多少)。

以 70B fp16(权重 140GB)为例,8 卡下几乎必须 TP≥4 或 TP2×PP2 才能摊到每卡 ~17-35GB。
而 32B 模型可能 TP2×DP4 就够,TP 更省通信。
'''))

NB.code(D('''
# 用纯 Python 打印 8 卡组合表,避免构造 pandas DataFrame(见 helpers 注释)
rows = []
for tp, pp, dp in all_combos(8):             # 枚举所有合法组合
    est = estimate_combo(70, 80, 8192, gpus=8, tp=tp, pp=pp, dp=dp, batch=4)  # 70B 模型
    rows.append((f"TP{tp}×PP{pp}×DP{dp}", round(est["weights_gb"], 1),
                 round(est["kv_gb"], 1), round(est["tp_comm_mb"], 1),
                 round(est["bubble"], 3)))
print(f"{'组合':<16}{'权重GB/card':>11}{'KV_GB/card':>11}{'TP通信MB/fwd':>13}{'气泡占比':>8}")
for name, wg, kv, cm, bub in rows:
    print(f"{name:<16}{wg:>11.1f}{kv:>11.1f}{cm:>13.1f}{bub:>8.3f}")
print("\\n读表:TP1×PP1×DP8 权重 140GB/卡(放不下);TP8×PP1×DP1 权重 17.5GB 但通信最高;")
print("      TP2×PP2×DP2 是均衡点——权重 35GB、通信与气泡都居中。")
combo_rows = rows
'''), "📊 8 卡组合没有「全胜」方案:省显存 → 大 TP,低延迟 → 小 TP,低气泡 → 小 PP——全部是取舍。")

NB.md("## 4️⃣ 真实规模数字:GPU 实测锚点 ⚡",
D('''
COMBO_EST 里的 `dtype_bytes=2` 等是**几何事实**,但「算力多快、搬数据多快」才是需要真机才填得准的变量。
这一步在 **RTX 5060** 上实测三笔硬账:每 B 参数 fp16 权重显存、一次大 GEMM 的计算吞吐(算力墙)、
内存搬运带宽(通信量级锚点);然后用**真实 2B/参数**回填 8 卡各组合,画出「每卡显存 ↔ DP 吞吐」的权衡图。
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
import gc, time
import torch
from vllm_real import cuda_info

print("设备:", cuda_info())
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ① 每 B 参数 fp16 权重的硬账(几何事实)
print(f"每 1B 参数 fp16 权重 = {2.0:.0f} GB(1e9 参数 × 2B)")

# ② 实测一次大 GEMM 的计算吞吐(算力墙 / 计算墙上界)
n = 4096
a = torch.randn(n, n, device=dev, dtype=torch.float16)
b = torch.randn(n, n, device=dev, dtype=torch.float16)
print(f"GEMM 矩阵 a/b shape = {tuple(a.shape)}  <- (n=4096, n=4096) fp16")
for _ in range(5): a @ b                       # 预热
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(30): c = a @ b
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 30 * 1e3
flops = 2 * n ** 3                             # 一次 GEMM 的 FLOPs = 2n³
print(f"fp16 GEMM {n}³ 单次 {ms:.3f} ms → 计算吞吐 ≈ {flops/(ms/1e3)/1e12:.1f} TFLOPS")

# ③ 实测内存搬运带宽(通信带宽量级锚点)
x = torch.randn(8_000_000, device=dev); y = torch.empty_like(x)
def copy_ms(reps=20):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): y.copy_(x)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps * 1e3
for _ in range(5): y.copy_(x)                  # 预热
cms = copy_ms()
print(f"device 间 copy {x.numel()*2/1e6:.0f} MB(收发) 单次 {cms:.3f} ms "
      f"→ 约 {x.numel()*4/(cms/1e3)/1e9:.1f} GB/s")

# ④ 用真实 2B/参数 + estimate_combo 估 8 卡各组合
combos = [(1, 1, 8), (2, 1, 4), (4, 1, 2), (2, 2, 2), (8, 1, 1), (4, 2, 1), (2, 4, 1)]
names, wlist, dplist, bub = [], [], [], []
for tp, pp, dp in combos:
    est = estimate_combo(70, 80, 8192, gpus=8, tp=tp, pp=pp, dp=dp, batch=4)
    names.append(f"TP{tp}×PP{pp}×DP{dp}")
    wlist.append(est["weights_gb"]); dplist.append(dp); bub.append(est["bubble"])
print("\\n8 卡各组合:每卡权重(GB) ↔ DP 副本数:")
for nm, w, d in zip(names, wlist, dplist):
    print(f"  {nm:<14} 权重 {w:6.1f} GB/card × DP={d}")
print("\\n读表:要 DP 大(吞吐高)就得少留给 TP/PP → 每卡权重上升;要每卡显存小就得多用 TP/PP →"
      " DP 变小、气泡变高。左下方点省显存、右上方点高吞吐,中间是平衡——没有免费午餐。")

gc.collect(); torch.cuda.empty_cache()
'''), "⚡ **真实锚点**:fp16 每 B 参数 = 2GB(几何),实测 GEMM 吞吐与搬运带宽化作「算力墙/通信墙」的真数写照;8 卡各组合的权重/DP/气泡坐标全部来自同一套 `estimate_combo`,比纯三选一更有据。")

NB.md("## 5️⃣ 与 vLLM 工程实现的关系:组合怎么配 🚀",
D('''
vLLM 的组合配置建议([parallelism_scaling 文档](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)):

1. **TP 设节点内卡数、PP 设节点数**:「TP 跨节点慢,PP 跨节点划算」是最常见配置;
   例如 16 卡 / 2 节点 → `--tensor-parallel-size 8 --pipeline-parallel-size 2`;
2. **模型装得下单节点**:只用 TP(`--tensor-parallel-size 4`),不需要 PP;
3. **GPU 数不能整除模型 / 无 NVLink 节点**:L40S 等场合,PP 反而比 TP 吞吐更高、
   通信更少(文档明确建议);
4. **推理的「DP 在 PP 内部」**:跨 PP 组的同层 stage 互为 DP 副本,各自服务不同请求——
   这就是组合里 DP 的推理形态。

训练侧的集大成者是 **Megatron-DeepSpeed 的 3D 并行**(PTD-P):
[arXiv:2104.04473](https://arxiv.org/abs/2104.04473) 用 TP(节点内)+ PP(节点间)+ DP
组合训练了万亿参数模型,吞吐 502 PFLOPS——与 vLLM 推理的 TP×PP 思路一脉相承。

> 📄 GPT-3([arXiv:2005.14165](https://arxiv.org/abs/2005.14165))训练时就是
> 「每个矩阵乘内部做模型并行 + 跨层做流水并行 + 数据并行」的组合——三者并用从 2020 年起就是主流。
'''))

NB.md("## 6️⃣ 最优选型讨论:决策树 🧭",
D('''
给一个实用的「决策树」思路:

1. **先满足显存**:让每卡(权重+KV+激活+缓存)装得下,不够就加大 TP/PP;
2. **再控通信**:TP 别超过单节点内互联支持的规模(跨节点 TP 代价高),PP 别太深(气泡);
3. **最后调吞吐**:剩余卡数都堆给 DP,提升并发吞吐。

以 70B fp16(权重 140GB)为例,8 卡下几乎必须 TP≥4 或 TP2×PP2 才能摊到每卡 ~17-35GB。
而 32B 模型可能 TP2×DP4 就够,TP 更省通信。**实际还要看单卡显存、NVLink/PCIe 拓扑与负载特征**,
本课的公式是「方向正确」的估算,不是精确仿真——这正是 vLLM 文档建议先小规模验证、再放大的原因。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖卡数与并行度,实时算开销 🎛️",
D('''
运行 `app_41_combination.py`,拖动**模型参数量、层数、隐藏维、总卡数、TP、PP、micro-batch**,
实时算出每卡权重/KV/通信/气泡,并列出所有合法组合对比表:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_41_combination.py
```

浏览器打开 **http://localhost:8501**。建议:把模型设成 70B、总卡数 8,然后切换不同 TP/PP,
观察显存、通信、气泡如何此消彼长;再换到 32B 模型,看哪些组合开始「显存充裕」。完整源码如下:
'''))

NB.code(APP_41, "📜 这就是 app_41_combination.py 的完整源码,notebook 与 app 共享同一套组合估算函数。")

wrapup(NB,
    summary=[
        "TP / PP / DP 摊的是不同维度(矩阵/层/数据),可组合叠加,总卡数 = TP×PP×DP",
        "每卡权重 ∝ 1/(TP×PP),KV ∝ 1/TP,TP 通信 ∝ L/PP×(TP-1)/TP,气泡 = (PP-1)/(m+PP-1)",
        "8 卡下组合很多,没有「全胜」方案:省显存→大 TP,低延迟→小 TP,低气泡→小 PP",
        "选型先满足显存、再控通信、最后堆 DP 提吞吐",
        "本课公式是方向正确的估算,实际要结合显存容量、拓扑与负载验证",
    ],
    practice=[
        "把 estimate_combo 的 batch 从 4 改到 64,看 KV 显存如何暴涨、哪些组合装不下",
        "写一个函数:给定单卡显存,自动筛选出「能放下」的合法组合",
        "对 405B 模型重跑 all_combos(8),看是否没有组合能在 8 卡放下,讨论需要多少卡",
        "在 pyecharts 图里加第三个指标「每卡总显存」,做三维取舍的可视化",
    ],
    links=[
        ("Megatron-LM 论文", "https://arxiv.org/abs/1909.08053"),
        ("Megatron-LM 万亿参数论文", "https://arxiv.org/abs/2104.04473"),
        ("GPT-3 论文", "https://arxiv.org/abs/2005.14165"),
        ("vLLM parallelism_scaling 文档", "https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"),
    ])

NB.save(str(Path(CH06) / "41_parallel_combination.ipynb"))

app_path = Path(CH06) / "app_41_combination.py"
app_path.write_text(APP_41_SRC + "\n", encoding="utf-8")
print(f"[ok] {app_path}")