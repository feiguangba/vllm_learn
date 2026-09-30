# -*- coding: utf-8 -*-
"""生成 38_pipeline_parallel.ipynb 与 app_38_pipeline_parallel.py(教材级重写版)

论文/资料支撑:
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using
  Pipeline Parallelism", arXiv:1811.06965(气泡与 micro-batch)
- Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN
  Training", arXiv:1806.03377(1F1B 调度)
- vLLM parallelism_scaling 文档(PP 工程落地)
"""
from helpers import D, PP_SCHED, PP_BUBBLE, chapter_cover, wrapup, new_nb, CH06
from pathlib import Path

APP_38_SRC = D('''
# -*- coding: utf-8 -*-
# app_38_pipeline_parallel.py — 流水线并行的 GPipe vs 1F1B 甘特图 🏭
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🏭 38 · 流水线并行", layout="wide")
st.title("🏭 第 38 课 · 流水线并行:切层 + 微批,看懂气泡")

st.markdown("""
**流水线并行(PP)** 把模型按层切成若干「舞台(stage)」,像工厂流水线一样接力处理一批样本。
为了不让上游舞台干等,把大 batch 拆成 **micro-batch** 流水推进。
但流水线会留下 **气泡(bubble)**——某些舞台在某些时刻在空等。对比两种调度:
- **GPipe**:每舞台先做完全部前向,再做反向(U 形);
- **1F1B(PipeDream-Flush)**:稳态期一个前向接一个反向,气泡更小。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    p = st.slider("舞台数(层分组)", 2, 8, 4, 1)
    m = st.slider("micro-batch 数", 2, 24, 8, 1)
    f = st.slider("前向耗时 f", 0.5, 3.0, 1.0, 0.1)
    b = st.slider("反向耗时 b", 1.0, 5.0, 2.0, 0.1)
    st.caption("m 越大气泡越小,但激活驻留越高(1F1B 缓解);f/b 为每个 micro-batch 的计算耗时。")

def run_pipeline(order_per_stage, p, f, b):
    # 事件驱动流水线模拟器
    t0 = [0.0] * p
    fin = {}
    ptr = [0] * p
    events = []
    remaining = sum(len(q) for q in order_per_stage)
    while remaining:
        progressed = False
        for s in range(p):
            if ptr[s] >= len(order_per_stage[s]):
                continue
            mb, typ = order_per_stage[s][ptr[s]]
            if typ == "F":
                dep = fin.get((s - 1, mb, "F"), 0.0) if s > 0 else 0.0
            else:
                if s < p - 1:
                    if (s + 1, mb, "B") not in fin:
                        continue         # 下游同 mb 的 B 还没排到,本轮到下一圈再处理
                    dep = fin[(s + 1, mb, "B")]
                else:
                    dep = 0.0
            cost = f if typ == "F" else b
            start = max(t0[s], dep)
            events.append((s, mb, typ, start, start + cost))
            fin[(s, mb, typ)] = start + cost
            t0[s] = start + cost
            ptr[s] += 1
            remaining -= 1
            progressed = True
        assert progressed, "调度死锁"
    return events

def schedule_gpipe(p, m, f, b):
    return run_pipeline([[(mb, "F") for mb in range(m)] + [(mb, "B") for mb in reversed(range(m))]
                         for s in range(p)], p, f, b)

def schedule_1f1b(p, m, f, b):
    orders = []
    for s in range(p):
        w = min(p - 1 - s, m)
        q = [(mb, "F") for mb in range(w)]
        for i in range(m - w):
            q.append((w + i, "F")); q.append((i, "B"))
        for i in range(m - w, m):
            q.append((i, "B"))
        orders.append(q)
    return run_pipeline(orders, p, f, b)

def stats(events, p):
    makespan = max(e[4] for e in events)
    busy = sum(e[4] - e[3] for e in events)
    return makespan, 1 - busy / (p * makespan)

ev_g = schedule_gpipe(p, m, f, b)
ev_1 = schedule_1f1b(p, m, f, b)
ms_g, bub_g = stats(ev_g, p)
ms_1, bub_1 = stats(ev_1, p)
formula = (p - 1) / (m + p - 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("GPipe 总时长", f"{ms_g:.1f}")
c2.metric("GPipe 气泡占比", f"{bub_g:.1%}")
c3.metric("1F1B 气泡占比", f"{bub_1:.1%}")
c4.metric("公式 (p-1)/(m+p-1)", f"{formula:.1%}")

def gantt(events, title):
    fig = go.Figure()
    for s, mb, typ, stt, enn in events:
        fig.add_trace(go.Bar(x=[enn - stt], y=[f"Stage {s}"],
                             base=stt, orientation="h", name="",
                             marker_color="#72B7B2" if typ == "F" else "#E45756",
                             hovertemplate=f"mb{mb} {typ} [{stt:.1f},{enn:.1f}]<extra></extra>",
                             showlegend=False))
    fig.update_layout(title=title, xaxis_title="时间", yaxis_title="舞台",
                      barmode="stack", height=360, bargap=0.1,
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

left, right = st.columns(2)
with left:
    st.subheader("GPipe")
    st.plotly_chart(gantt(ev_g, "GPipe(先全部 F 再全部 B)"), use_container_width=True)
with right:
    st.subheader("1F1B")
    st.plotly_chart(gantt(ev_1, "1F1B(1 前向接 1 反向)"), use_container_width=True)
st.caption("🟦 前向 / 🟥 反向。1F1B 的反向更早开始、气泡更小,且激活驻留更低。")

st.markdown(f"""
> 💡 **直觉**:舞台越多气泡越大,微批越多气泡越小。公式 **(p-1)/(m+p-1)** 在理想假设下精确成立
> (与 f、b 无关)。当前:公式 {formula:.1%},1F1B 实测 {bub_1:.1%}。1F1B 相比 GPipe 把
> 「同时驻留的激活数」从 m 降到约 p,更省显存,所以现代框架(含 vLLM 的 PP)普遍用它。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 6 章 · 第 38 课配套演示")

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

APP_38 = "%%writefile app_38_pipeline_parallel.py\n" + APP_38_SRC

NB = new_nb("第 38 课 · 流水线并行:切层、微批与气泡",
            subtitle="手写事件驱动模拟器,画出 GPipe 与 1F1B 的甘特图,推演气泡公式 (p-1)/(m+p-1)",
            emoji="🏭")

chapter_cover(NB,
    objectives=[
        "理解流水线并行(PP):按层切舞台,大 batch 拆成 micro-batch 流水推进",
        "手写事件驱动流水线模拟器,得到 GPipe 与 1F1B 的时间线",
        "理解「气泡(bubble)」的成因,并验证公式 bubble = (p-1)/(m+p-1)",
        "对比 GPipe 与 1F1B 的气泡与激活驻留差异",
    ],
    toc=[
        ("直觉:工厂流水线", "按层切车间、拆微批流水,但会有空等(气泡)"),
        ("核心定义与公式", "PP 定义 + 气泡公式推导 + 符号表"),
        ("最小实现 · 逐行推演", "事件驱动调度器:GPipe 与 1F1B 各自的舞台顺序"),
        ("数值验证", "气泡公式 (p-1)/(m+p-1) 逐格核对"),
        ("真实规模数字", "真实 prefill 耗时换算成毫秒级空等账"),
        ("与 vLLM 工程实现的关系", "vLLM 的 pipeline_parallel_size 与 PP 落地"),
        ("配套 Streamlit 演示", "app_38_pipeline_parallel.py:拖层数/微批数看甘特图"),
    ],
    links=[
        ("GPipe 论文 (arXiv:1811.06965)", "https://arxiv.org/abs/1811.06965"),
        ("PipeDream 论文 (arXiv:1806.03377)", "https://arxiv.org/abs/1806.03377"),
        ("vLLM parallelism_scaling 文档", "https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"),
    ])

NB.md("## 1️⃣ 直觉:工厂流水线 🏭",
D('''
想象一家工厂要把一批产品做出来,工序分「裁剪→缝纫→质检」三道,分别在三个车间。如果一件做完
下一件才开始,大部分车间都在空等。**流水线并行(PP)** 的办法是:

1. **按层切舞台**:把 Transformer 的 $L$ 层切成 $p$ 份,每个舞台(stage)负责一段层;
2. **拆 micro-batch**:把一个 batch 拆成 $m$ 个小批(micro-batch),让它们像流水线工件一样
   依次流过各个舞台。

这样各舞台能**同时**在处理不同的 micro-batch,大幅提高利用率。但流水线启动和收尾阶段,
上游/下游舞台会**空等**——这段空等时间占比就叫 **气泡(bubble)**,是 PP 固有的开销。
'''))

NB.md("## 2️⃣ 核心定义与公式 🧮",
D('''
**流水线并行(PP)**:把模型 $L$ 层按深度切成 $p$ 段,第 $s$ 个舞台只持有并计算自己的 $L/p$ 层;
一个 batch 被拆成 $m$ 个 micro-batch,像工件一样依次流过各舞台。

**气泡公式推导**(理想假设:所有舞台的前向耗时同为 $f$、反向同为 $b$,且 $m$ 足够填充流水线):

- 总时长:$T = (m + p - 1)(f + b)$(首尾各 $p-1$ 个时隙在灌水/放空,中间 $m$ 个时隙满载);
- 气泡时长:$(p-1)(f+b)$(每舞台有 $p-1$ 个时隙空等);

$$\\text{bubble} = \\frac{(p-1)(f+b)}{(m+p-1)(f+b)} = \\frac{p-1}{m+p-1}$$

| 符号 | 含义 |
|---|---|
| $L$ | 模型总层数 |
| $p$ | 舞台数(Pipeline 深度) |
| $m$ | micro-batch 数 |
| $f, b$ | 单 micro-batch 前向 / 反向耗时 |
| $\\text{bubble}$ | 气泡占比(空等时间 / 总时间) |

- **舞台越多($p$ 大)**,启动/收尾空等越多 → 气泡越大;
- **微批越多($m$ 大)**,流水线被填满 → 气泡越小($m\\to\\infty$ 时气泡 $\\to 0$)。

**两种经典调度**:
- **GPipe**:每舞台先做完 $m$ 个前向,再做 $m$ 个反向(倒序)→ 呈 U 形;
- **1F1B(PipeDream-Flush)**:稳态期「一个前向接一个反向」→ 气泡与 GPipe 相同,
  但**同时驻留的激活数从 $m$ 降到约 $p$**,更省显存。
'''))

NB.md("## 3️⃣ 最小实现 · 逐行推演:事件驱动流水线模拟器 ⏱️",
D('''
要精确算气泡,我们写一个**事件驱动的流水线模拟器**:每个舞台按自己的固定顺序执行
(micro-batch, 前向/反向)队列;前向要等上游同 micro-batch 的前向完成,反向要等下游同
micro-batch 的反向完成。**每行代码都有注释**:
'''))

NB.code(PP_SCHED, "🎯 `schedule_gpipe` 是「先全 F 再全 B」的 U 形,`schedule_1f1b` 是稳态 1F1B。对比 `gantt_stats`:理想假设下两者气泡相同,但 1F1B 的**峰值驻留激活更少**(从 m 降到约 p),更省显存。")

NB.md("## 4️⃣ 数值验证:气泡公式逐格核对 🧮",
D('''
在理想假设(各舞台 $f$、$b$ 相同)下,用模拟器把 $p \\in [2,4,8]$、$m \\in [4,8,16,32]$ 全部算一遍,
与公式 $(p-1)/(m+p-1)$ 对照——下面用 1F1B 调度验证:
'''))

NB.code(PP_BUBBLE, "✅ 每行「模拟气泡」与「公式 (p-1)/(m+p-1)」完全一致——公式在理想模型下是精确的,不是近似。")

NB.md("## 5️⃣ 真实规模数字:气泡换算成真实毫秒 ⚡",
D('''
上面的气泡都是**无量纲比例**,工程上要换算成真实毫秒才知道痛不痛。这里用跨章共享库
`vllm_real.TinyGPT` 在 **RTX 5060** 上量一次真实前向(prefill 128 token)的耗时,
再把「气泡占比 × 真实前向耗时」放成「每批空等多少毫秒」的实物账:
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
import gc
import torch
from vllm_real import bench_prefill_decode, cuda_info

print("设备:", cuda_info())
b = bench_prefill_decode(d=256, layers=6, L=128, steps=64, reps=5)   # 真实前向微基准
print(f"一次真实并行前向(prefill {b['L']} token): {b['prefill_ms']:.3f} ms")
print(f"单 decode 步(1 token)              : {b['decode_step_ms']:.3f} ms")
print(f"参数量                               : {b['params']/1e6:.2f} M")

# 把气泡占比挂到真实前向耗时上,变成「白等多少毫秒」
for p, m in [(2, 8), (4, 8), (4, 16), (8, 32)]:
    bubble = (p - 1) / (m + p - 1)           # 气泡公式
    idle = bubble * b["prefill_ms"]          # 空等毫秒 = 气泡比例 × 单次前向
    print(f"p={p:>2} m={m:>2}: 气泡 {bubble:6.1%}  → 按 prefill 前向"
          f" {b['prefill_ms']:.2f} ms 算,每批约空等 {idle:5.2f} ms")

gc.collect(); torch.cuda.empty_cache()
'''), "⚡ **真机数字**:真实前向当「一步多慢」的尺子,气泡比例一乘就成了毫秒级空等账——这就是为什么工程上要压 p、加 m 或用 1F1B。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:pipeline_parallel_size 🚀",
D('''
vLLM 对 PP 的工程事实([parallelism_scaling 文档](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)):

1. **参数就是 PP 度**:`--pipeline-parallel-size 2` 表示把模型切成 2 段;
2. **与 TP 组合**:「TP = 节点内卡数,PP = 节点数」是最常见配置,因为 TP 的 AllReduce
   跨节点太慢、PP 的点对点通信对带宽要求低;
3. **vLLM 的调度**:vLLM 采用 1F1B 风格的流水调度 + micro-batch,让各舞台在 decode 时
   尽量不空等;相比 GPipe 的 U 形调度,1F1B 主要省的是**激活驻留显存**(从 m 降到约 p);
4. **推理里 PP 的价值**:PP 不降低单请求延迟(层是串行的),但能摊薄显存、提高吞吐;
   吞吐提升还常带「超线性」——省下的显存变成更大的 KV cache 和 batch。

```bash
# 8 卡:TP4(节点内)+ PP2(两个节点)
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 4 --pipeline-parallel-size 2
```

> 📄 气泡与 micro-batch 的关系、re-computation 技巧见 [GPipe 论文](https://arxiv.org/abs/1811.06965);
> 1F1B 与权重版本管理见 [PipeDream](https://arxiv.org/abs/1806.03377)。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖层数与微批数,看甘特图与气泡 🎛️",
D('''
运行 `app_38_pipeline_parallel.py`,拖动**舞台数、micro-batch 数、前向/反向耗时**,
实时重算 GPipe 与 1F1B 的甘特图和气泡占比:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_38_pipeline_parallel.py
```

浏览器打开 **http://localhost:8501**。建议:先把 m 设小(如 4)看气泡大,再拖大到 16 看气泡缩小;
把 p 从 2 拖到 8,看气泡随舞台数上升。完整源码如下:
'''))

NB.code(APP_38, "📜 这就是 app_38_pipeline_parallel.py 的完整源码,notebook 与 app 共享同一套事件驱动调度器。")

wrapup(NB,
    summary=[
        "PP 按层切舞台,大 batch 拆 micro-batch 流水推进,让各舞台同时工作",
        "气泡是流水线启动/收尾的空等占比,公式 bubble = (p-1)/(m+p-1)",
        "舞台越多气泡越大、微批越多气泡越小,但微批太多会撑大激活显存",
        "GPipe 先全 F 再全 B(U 形);1F1B 稳态 1F1B 交替,气泡与 GPipe 相同但激活驻留更低",
        "现代框架(含 vLLM)的 PP 普遍采用 1F1B 风格的调度",
    ],
    practice=[
        "把模拟器的 f 设为 2、b 设为 1,重算气泡,看是否还等于 (p-1)/(m+p-1)(验证公式假设)",
        "给模拟器增加 stage_peak_inflight 的输出,对比 GPipe(m 个驻留)与 1F1B(p 个驻留)",
        "实现一种「交错式」调度(interleaved pipeline),画出甘特图并比较气泡",
        "把 p 固定 8、m 从 2 到 64,画气泡下降曲线,找「性价比最高」的 m 区间",
    ],
    links=[
        ("GPipe 论文", "https://arxiv.org/abs/1811.06965"),
        ("PipeDream 论文(1F1B)", "https://arxiv.org/abs/1806.03377"),
        ("vLLM parallelism_scaling 文档", "https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"),
    ])

NB.save(str(Path(CH06) / "38_pipeline_parallel.ipynb"))

app_path = Path(CH06) / "app_38_pipeline_parallel.py"
app_path.write_text(APP_38_SRC + "\n", encoding="utf-8")
print(f"[ok] {app_path}")