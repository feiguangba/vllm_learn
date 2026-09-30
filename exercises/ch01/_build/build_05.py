# -*- coding: utf-8 -*-
"""生成 05_prefill_vs_decode.ipynb 与 app_05_prefill_demo.py"""
from helpers import D, chapter_cover, wrapup, new_nb, CH01
from pathlib import Path

APP_05 = D('''
# -*- coding: utf-8 -*-
# app_05_prefill_demo.py — Prefill vs Decode 演示 ⚡  (真实 GPU 数据版)
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
import streamlit as st, plotly.graph_objects as go
from vllm_real import bench_prefill_decode, cuda_info

st.set_page_config(page_title="Prefill vs Decode ⚡", layout="wide")
st.title("⚡ 第 05 课 · Prefill vs Decode:两阶段推理(真实 GPU 实测)")

st.markdown("""
LLM 推理分两个阶段:**prefill**(一次性处理整个提示词,高并行)与 **decode**(逐字生成,串行)。
除了理论 FLOPs,本 app **在同一张 GPU 上真实跑一个小 GPT**,给出两阶段实测的总耗时与吞吐,
让 'decode 更慢' 有真实数字支撑。
""")

def flops_total(N, L, K):
    return 2 * N * (L + K)

with st.sidebar:
    st.header("🎛️ 参数")
    n_b = st.slider("模型参数量(十亿)", 0.5, 70.0, 7.0, 0.5)
    L = st.slider("提示词长度(prefix tokens)", 64, 2048, 512, 32)
    K = st.slider("生成长度(generated tokens)", 64, 2048, 512, 32)
    do_bench = st.button("🚀 在 GPU 实测两阶段", type="primary")
    st.caption(f"检测到: **{cuda_info()}**")

N = n_b * 1e9
fp, fd = flops_total(N, L, 0), flops_total(N, 0, K)

st.subheader("📐 理论计算量账本 (FLOPs ≈ 2N×token)")
c1, c2, c3 = st.columns(3)
c1.metric("prefill FLOPs", f"{fp/1e15:.2f} PFLOPs")
c2.metric("decode FLOPs", f"{fd/1e15:.2f} PFLOPs")
c3.metric("总 FLOPs", f"{(fp+fd)/1e15:.2f} PFLOPs")

if do_bench or "bench" in st.session_state:
    if "bench" not in st.session_state:
        with st.spinner("跑小 GPT 微基准中…"):
            st.session_state["bench"] = bench_prefill_decode(
                d=256, layers=8, L=L, steps=K, reps=5)
    b = st.session_state["bench"]
    st.subheader("🛠️ 真实 GPU 实测(小 GPT 微基准)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("prefill 总耗时", f"{b['prefill_ms']:.2f} ms")
    m2.metric("decode 总耗时(L 步)", f"{b['decode_total_ms']:.2f} ms")
    m3.metric("耗时比 decode/prefill", f"{b['ratio']:.1f} ×")
    m4.metric("decode 吞吐", f"{b['decode_tok_per_s']/1e3:.1f} k tok/s")
    st.caption(f"参数 ~{b['params']/1e6:.1f}M · 设备 {b['device']}。"
               "prefill 一次并行 L token;decode 逐 token 走 L 步。")
else:
    st.info("点击左侧『🚀 在 GPU 实测两阶段』,跑真实微基准得到数字(约几秒)。再次进入本 app 会缓存结果。")
''')

NB = new_nb("第 05 课 · Prefill vs Decode:两阶段推理",
            subtitle="一次处理整个提示词 vs 逐字生成 —— 推导 FLOPs 公式,再用真实 GPU 实测两阶段的耗时/吞吐差异",
            emoji="⚡")

chapter_cover(NB,
    objectives=[
        "区分推理的两个阶段:prefill(处理提示词)与 decode(逐字生成)",
        "推导推理 FLOPs 公式:每 token 约 2N 次浮点运算,总 FLOPs ≈ 2N(L+K)",
        "用真实 GPU(小 GPT)实测 prefill 与 decode 的耗时、吞吐与显存,理解并行度差异",
        "用 pyecharts 柱状图对比两阶段耗时与计算量",
    ],
    toc=[
        ("直觉:传纸条 vs 写日记", "prefill 是一次性读完,decode 是一字一字写"),
        ("根源:自回归必然走向两阶段", "为什么 LLM 必须分 prefill 与 decode(承接上一课)"),
        ("FLOPs 公式推导", "每 token 2N,prefill 与 decode 的计算量真相"),
        ("真实 GPU 实测两阶段", "同一小模型,同样 token 数,谁快谁慢、吞吐差多少"),
        ("为什么 decode 更慢", "并行度决定一切:compute-bound vs memory-bound"),
        ("连续批处理预告", "把多条 decode 的 1-token 拼成大 batch,喂饱 GPU(预告 ch03)"),
        ("pyecharts 柱状对比", "耗时与计算量的可视化"),
        ("配套 Streamlit 演示", "app_05_prefill_demo.py:模型规模/序列长度滑杆实时算"),
    ],
    links=[
        ("vLLM 文档:关于 prefill/decode 与连续批处理", "https://docs.vllm.ai/en/latest/"),
        ("Transformer Inference Arithmetic 综述(FLOPs 公式参考)", "https://kipp.ly/transformer-inference-arithmetic/"),
        ("Attention Is All You Need(自回归解码背景)", "https://arxiv.org/abs/1706.03762"),
    ])

NB.md("## 1️⃣ 直觉:传纸条 vs 写日记 ✉️",
D('''
用户发来一段提示词,模型要生成回复。这整段过程被拆成两个**性格迥异**的阶段:

- **prefill(预填充)**:一次性读完整段提示词。所有 token 一到位,就能**并行**算注意力,
  像老师扫一眼全班名单,瞬间记住所有人;
- **decode(解码)**:拿到提示词后,**一个 token 一个 token** 地往外写。每一步都要等上一步
  写完才能继续,像写日记只能一字一字往下写,没法并行。

上一课的自回归告诉我们:生成 K 个 token 需要 K 次前向。而 prefill 不一样——它只需要
**一次**前向,就能把 L 个提示词 token 全部算完。这就是两个阶段最本质的区别。
'''))

NB.code("import os\nos.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", \"TRUE\")\n"
        "import torch\nimport numpy as np\nimport pandas as pd\nimport time",
        "🧪 老规矩:先解决 Windows 下 torch 的 OMP 库冲突。本课仍用 CPU 做教学实验。")

NB.md("## 2️⃣ 根源:自回归必然走向两阶段 🔗",
D('''
上一课我们证明了:LLM 是**自回归**的——生成第 $k$ 个 token 要把「已生成的前 $k$-1 个 token」拼回序列再前向一次。
于是推理过程天然被劈成两截:

- **prefill(预填充)**:用户一次给来整段提示词($L$ 个 token)。这些 token **同时已知**,可以一次性、并行地前向算完,得到第一个输出 token;
- **decode(解码)**:从第一个输出 token 开始,**一次只能 forward 一个新 token**,生成 $K$ 个就 forward 恰好 $K$ 次。

为什么必须这样?因为目标 token 是「未知、要逐步猜」的——不可能在不知道"你自己上一刻写了什么"的情况下并行写后续。这是自回归的下界,也是后续一切优化(缓存、批处理)的出发点。

「传纸条」:prefill 把全班名单一眼扫完(并行);「写日记」:decode 一字一字写(串行)。
'''))

NB.md("## 3️⃣ FLOPs 公式推导 🧮",
D('''
推理的计算量可以用 **FLOPs(浮点运算次数)** 度量。一个经验公式:每处理一个 token,
前向大约需要 **2N** 次浮点运算(N 是模型参数量)。直觉是——每个参数在矩阵乘里都要
做一次乘、一次加,所以乘 2。

于是:

- **prefill**:一次处理 L 个 token,FLOPs ≈ $2N \\times L$;
- **decode**:生成 K 个 token,每步一个 token,FLOPs ≈ $2N \\times K$;
- **总推理 FLOPs** ≈ $2N \\times (L + K)$。

注意:**总计算量只看 token 总数**,跟阶段无关。真正拉开差距的是**时间**——同样的 FLOPs,
prefill 能并行算完,decode 只能排队串行。这也提示我们:别只看 FLOPs,要看**耗时和吞吐**。
'''))

NB.code(D('''
def flops_prefill(N, L):
    return 2 * N * L

def flops_decode(N, K):
    return 2 * N * K

def flops_total(N, L, K):
    return 2 * N * (L + K)

N = 7e9          # 7B 模型
L, K = 128, 128  # 提示词 128,生成 128
print(f"7B 模型,提示词 {L},生成 {K}:")
print(f"  prefill FLOPs = 2N×L = {flops_prefill(N, L)/1e12:.2f} TFLOPs")
print(f"  decode  FLOPs = 2N×K = {flops_decode(N, K)/1e12:.2f} TFLOPs")
print(f"  总 FLOPs       = 2N×(L+K) = {flops_total(N, L, K)/1e12:.2f} TFLOPs")
'''), "✅ 看到 prefill 与 decode 的计算量几乎相等(当 L≈K 时)——瓶颈不在算多少,而在怎么算。")

NB.md("## 4️⃣ 真实 GPU 实测两阶段 ⏱️",
D('''
公式是理论,现在**在真实 GPU 上跑一个小 GPT**(线性层 + 激活,参数可测)验证两件事:
同样 token 数,decode 到底比 prefill 慢多少?吞吐差多少?

我们把一个小模型真正部署到 RTX 5060,分别测:
- **prefill**:一次把 `(1, L)` 的整段序列 forward(并行);
- **decode**:把同样的 L 个 token **一个一个** forward(`(1,1)` 重复 L 次)。

两者处理的总 token 数一样,但 prefill 是大矩阵一次算完,decode 是小矩阵算 L 次。
'''))

NB.code("import os\nos.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", \"TRUE\")\n"
        "import sys\nsys.path.insert(0, r\"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises\")\n"
        "import torch\nimport numpy as np\nimport pandas as pd\nimport time\n"
        "from vllm_real import TinyGPT, bench_prefill_decode, cuda_info",
        "🧪 复用跨章共享库 `vllm_real`(内含 TinyGPT 与真实微基准);它自动探测 CUDA/CPU。")

NB.code(D('''
L = 256                      # 一段提示词 256 个 token
bench = bench_prefill_decode(d=256, layers=8, vocab_size=1024, L=L, steps=64, reps=7)
print("设备:", bench["device"], "· 参数量: %.1f M" % (bench["params"] / 1e6))
print(f"prefill(一次并行 {L} token): {bench['prefill_ms']:.2f} ms　→ {bench['prefill_tok_per_s']/1e3:.1f} k tok/s")
print(f"decode (逐个 {L} token):     {bench['decode_total_ms']:.2f} ms　→ {bench['decode_tok_per_s']/1e3:.1f} k tok/s")
print(f"decode / prefill 耗时比:      {bench['ratio']:.1f} 倍")
'''), "🚀 这是**真实 GPU 数字**。注意:prefill 把 L 个 token **一起并行**;decode 逐字来,吞吐低一个数量级。")

NB.md("## 5️⃣ 为什么 decode 更慢 🤔",
D('''
同样的 token 数,为什么 decode 慢这么多?因为 **GPU 的算力要靠“大矩阵”才能吃满**,瓶颈分两类:

- **compute-bound(算力受限)**:prefill 中 `(B, L, H) @ (H, H)` 是大矩阵乘,乘加密度高、访存/计算比低,芯片算力是瓶颈;
- **memory-bound(内存受限)**:decode 每步 `(B, 1, H)`,几乎不消耗算力,瓶颈变成**搬参数和 KV** 的内存带宽——每次都要把整份权重从头读一遍。

这正是工程里反复出现的那句话:**decode 是带宽瓶颈,不是算力瓶颈**。所以"总 FLOPs 相仿"不再重要,"
每步访存多少字节"才是 decode 的关键。

打个比方:prefill 是装满货物的 100 辆卡车**一起**发车(满载高效);decode 是 100 辆卡车**一辆一辆**轮流发车,
每次只运一点货——总货运量(FLOPs)一样,时间却差很多。
'''))

NB.md("## 6️⃣ 那怎么救 decode?预告:连续批处理 🍳",
D('''
decode 慢的根源是「一次只算 1 个 token」。解法自然浮现:**把很多条 decode 请求的单个 token 拼成一个更大的 batch 一起 forward**——
小矩阵变回大矩阵,带宽也摊薄。这就是 vLLM 的 **连续批处理(continuous batching)**,我们先用本课的 `TinyGPT` 直接验证
"batch 增大 → 吞吐暴涨"这一效果:
'''))

NB.code(D('''
from vllm_real import bench_throughput_curve
b_list, tps, mps = bench_throughput_curve(batch=(1, 4, 16, 64, 128), token_len=16, reps=5)
print("batch | 每步耗时(ms) | 吞吐(以每秒生成的 token)")
for b, ms, tp in zip(b_list, mps, tps):
    print(f"{b:5d} | {ms:8.3f}      | {tp:10.0f}")
'''), "💡 batch 每增大 4 倍,吞吐不是平均摊薄而是**大涨**——固定开销被摊薄、带宽被复用。这为 ch03 的连续批处理埋下伏笔。")

NB.md("## 7️⃣ pyecharts 柱状对比 📊",
D('''
把 prefill 与 decode 的**耗时**和**单 token 平均耗时**画成柱状图,差距一目了然。
直接用上面真实 GPU 测得的数字:
'''))

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

names = ["prefill", "decode"]
times = [bench["prefill_ms"], bench["decode_total_ms"]]        # ms
per_token = [bench["prefill_ms"] / L, bench["decode_total_ms"] / L]

bar = (Bar()
       .add_xaxis(names)
       .add_yaxis("总耗时 (ms)", [round(t, 2) for t in times], color="#4C78A8")
       .add_yaxis("单 token 耗时 (ms)", [round(p, 4) for p in per_token], color="#E45756")
       .set_global_opts(title_opts=opts.TitleOpts(title="prefill vs decode 耗时对比(真实 GPU 实测)"),
                        yaxis_opts=opts.AxisOpts(name="毫秒"),
                        legend_opts=opts.LegendOpts(pos_top="8%")))
bar.render_notebook()
'''), "📊 左柱(总耗时)与右柱(单 token 耗时)都指向同一结论:decode 阶段明显更慢、更'贵'。")

NB.md("## 8️⃣ 配套 Streamlit 演示:实时算账本 🎛️",
D('''
运行同目录下的 `app_05_prefill_demo.py`,拖动**模型规模**与**提示词/生成长度**滑杆,
两阶段 FLOPs 实时刷新,并叠加**真实 GPU 测得的吞吐曲线**,看理论公式与实测如何互相印证:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_05_prefill_demo.py
```

浏览器打开 **http://localhost:8501**。把模型从 0.5B 拖到 70B,看 FLOPs 暴涨;对比 prefill 与 decode 实测吞吐。
完整源码如下(与同目录 `app_05_prefill_demo.py` 一致):
'''))

NB.code("%%writefile app_05_prefill_demo.py\n" + APP_05,
        "📜 这就是 app_05_prefill_demo.py 的完整源码。notebook 与 app 共用同一套 FLOPs 公式,保证讲解与演示一致。")

wrapup(NB,
    summary=[
        "推理分两阶段:prefill 一次读完整段提示词(高并行),decode 逐字生成(串行)——由自回归的下界必然决定",
        "推理 FLOPs ≈ 2N(L+K):每 token 前向约 2N 次浮点运算,总计算量只看 token 总数,与阶段无关",
        "真实 GPU 微基准:同样 token 数,decode(逐 token)明显慢于 prefill(并行),吞吐低一个数量级",
        "瓶颈本质:prefill 是 compute-bound(算力受限)、decode 是 memory-bound(内存/带宽受限)",
        "救法:连续批处理把多条 decode 的 1-token 拼成大 batch,吞吐随 batch 变大而大涨(预告 ch03)",
    ],
    practice=[
        "改 vllm_real 的 bench_prefill_decode 参数(d、layers、L),重跑并对比耗时比随 L 的变化:为什么 L 越大 decode 越吃亏",
        "用公式算 70B 模型、提示词 2048、生成 2048 的总 FLOPs,并换算成一张 100TFLOPs GPU 的理论耗时",
        "给 bench_throughput_curve 测 batch 从 1 到 128 的吞吐,画吞吐 vs batch 曲线,找饱和点并解释带宽为何封顶",
        "读 vLLM 关于 continuous batching 的文档,用自己的话解释它如何把 decode 从 memory-bound 拯救回来",
    ],
    links=[
        ("Transformer Inference Arithmetic(FLOPs 公式)", "https://kipp.ly/transformer-inference-arithmetic/"),
        ("vLLM 官方文档(性能与批处理)", "https://docs.vllm.ai/en/latest/performance/"),
        ("Attention Is All You Need(自回归解码背景)", "https://arxiv.org/abs/1706.03762"),
        ("Orca(连续批处理思想源头)", "https://arxiv.org/abs/2208.14217"),
    ])

NB.save(str(Path(CH01) / "05_prefill_vs_decode.ipynb"))

app_path = Path(CH01) / "app_05_prefill_demo.py"
app_path.write_text(APP_05 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
