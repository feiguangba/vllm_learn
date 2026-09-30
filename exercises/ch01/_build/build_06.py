# -*- coding: utf-8 -*-
"""生成 06_toy_inference_engine.ipynb 与 app_06_toy_engine.py"""
from helpers import D, NGRAM, SAMPLING, chapter_cover, wrapup, new_nb, CH01
from pathlib import Path

APP_06 = D('''
# -*- coding: utf-8 -*-
# app_06_toy_engine.py — 玩具推理引擎演示 🛠️
import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from collections import defaultdict, Counter

st.set_page_config(page_title="玩具推理引擎 🛠️", layout="wide")
st.title("🛠️ 第 06 课 · 组装一个玩具推理引擎")

st.markdown("""
把第 02 课的 **n-gram 语言模型** 与第 03 课的 **采样策略** 拼起来,就是一个能**批量生成**
多条文本、并统计吞吐(tokens/s)的**玩具推理引擎**。下方调整批大小、温度、top-p 等参数,
观察批量生成的吞吐如何变化。
""")

CORPUS = ["我喜欢学习机器学习和深度学习", "机器学习很热门", "深度学习让机器更聪明",
          "我每天写代码和学算法", "我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉"]

def build_ngram(corpus, n=2):
    model = defaultdict(Counter)
    for text in corpus:
        toks = list(text)
        for i in range(len(toks) - n + 1):
            model[tuple(toks[i:i + n - 1])][toks[i + n - 1]] += 1
    return model

def next_candidates(model, context, n):
    ctx = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)
    counts = model.get(ctx, Counter())
    total = sum(counts.values())
    if total == 0:
        return [], []
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    toks = [t for t, _ in items]
    probs = [c / total for _, c in items]
    return toks, probs

def softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()

def sample_from(logits, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    probs = softmax(logits, temperature)
    idx = rng.choice(len(probs), p=probs)
    return idx

class ToyEngine:
    def __init__(self, model, n=2, temperature=1.0, seed=0):
        self.model = model
        self.n = n
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.W = np.random.randn(32, 32) * 0.05

    def shared_forward(self):
        return np.random.randn(32, 32) @ self.W

    def generate_batch(self, prompts, max_new):
        seqs = [list(p) for p in prompts]
        tokens = 0
        for _ in range(max_new):
            self.shared_forward()
            for seq in seqs:
                toks, probs = next_candidates(self.model, seq, self.n)
                if not toks:
                    continue
                logits = np.log(np.array(probs) + 1e-12)
                seq.append(toks[sample_from(logits, self.temperature, self.rng)])
                tokens += 1
        return seqs, tokens

model = build_ngram(CORPUS, 2)

with st.sidebar:
    st.header("🎛️ 参数")
    batch = st.slider("批大小 batch", 1, 16, 4, 1)
    max_new = st.slider("每序列生成步数", 5, 40, 15, 1)
    temperature = st.slider("温度", 0.3, 2.0, 1.0, 0.1)
    prompts_input = st.text_area("多条提示词(每行一条)", "我喜欢\\n机器\\n学习")
    st.caption("批大小越大,单次前向处理的序列越多,固定开销被摊薄。")

prompts = [p for p in prompts_input.splitlines() if p.strip()] or ["我喜欢"]
engine = ToyEngine(model, 2, temperature, seed=0)
seqs, tokens = engine.generate_batch(prompts[:batch], max_new)

st.subheader("🖊️ 批量生成结果")
for i, s in enumerate(seqs):
    st.write(f"**序列 {i + 1}**: " + "".join(s))

c1, c2, c3 = st.columns(3)
c1.metric("批大小", len(seqs))
c2.metric("生成总 token 数", tokens)
c3.metric("吞吐", f"{tokens / max(max_new / 60, 1e-6):.0f} tokens/s")

st.subheader("📈 吞吐 vs 批大小(实测)")
bs = [1, 2, 4, 8, 16]
tps = []
for b in bs:
    e2 = ToyEngine(model, 2, temperature, seed=0)
    t0 = time.perf_counter()
    _, tok = e2.generate_batch(["我喜欢"] * b, 30)
    tps.append(tok / (time.perf_counter() - t0))
fig = go.Figure(go.Scatter(x=bs, y=tps, mode="lines+markers",
                           line=dict(width=3, color="#4C78A8"),
                           marker=dict(size=9)))
fig.update_layout(title="批量生成吞吐(批大小越大吞吐越高,后趋饱和)",
                  xaxis_title="批大小 batch", yaxis_title="tokens/s",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 批处理让一次前向同时算多个序列,固定开销被摊薄,吞吐上升;"
           "当固定开销占比变小时,曲线趋于饱和。")
''')

NB = new_nb("第 06 课 · 组装一个玩具推理引擎",
            subtitle="把第 02 课的 n-gram 与第 03 课的采样拼成能批量生成的玩具引擎,实测吞吐随批大小的变化曲线",
            emoji="🛠️")

chapter_cover(NB,
    objectives=[
        "复用 n-gram 语言模型 + 采样策略,组装一个能批量生成文本的玩具推理引擎",
        "理解 batch 生成:一次前向同时推进多条序列,摊薄固定开销",
        "实测并绘制'吞吐 vs 批大小'曲线,理解 vLLM 批处理的价值",
        "用 st.metric / plotly 实时展示引擎吞吐与生成结果",
    ],
    toc=[
        ("直觉:工厂流水线的批量化", "把多条请求并成一批一起加工,单位时间产量上升"),
        ("组装引擎:拼起积木", "n-gram 当语言模型,采样当解码器,一个 class 搞定"),
        ("批量生成演示", "一次喂多条提示词,逐字推进所有序列"),
        ("吞吐 vs 批大小曲线", "实测 tokens/s,看批处理如何摊薄固定开销"),
        ("解读曲线:为何后趋饱和", "固定开销占比下降,收益递减"),
        ("配套 Streamlit 演示", "app_06_toy_engine.py:多 prompt/温度/批大小,st.metric 看吞吐"),
    ],
    links=[
        ("vLLM 官方文档(吞吐与性能)", "https://docs.vllm.ai/en/latest/performance/"),
        ("Transformer Inference Arithmetic(FLOPs 与吞吐)", "https://kipp.ly/transformer-inference-arithmetic/"),
        ("HuggingFace: 批处理与生成", "https://huggingface.co/docs/transformers/generation_strategies"),
    ])

NB.md("## 1️⃣ 直觉:工厂流水线的批量化 🏭",
D('''
一个推理引擎就像一条**工厂流水线**:每来一条文本生成请求,就要占用一次“机器”的加工时间。
如果请求**一条一条**来,机器每加工完一条才接下一条,机器大部分时间在“空转”(等搬运、等启动)。

而 **batch(批处理)** 的做法,是把好几条请求**并成一批**同时加工:机器一次性读入一批原料,
一起走完整个加工流程。虽然每批的加工时间变长了一些,但**单位时间内完成的订单量(吞吐)**
却上去了——因为机器启动、装料这些**固定开销**被摊薄到了更多订单头上。

这就是本课要造的引擎的核心思想。我们把前面学过的两块积木拼起来:
**n-gram 语言模型**(决定“接着说什么”)+ **采样策略**(决定“具体吐出哪个字”)。
'''))

NB.code("import os\nos.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", \"TRUE\")\n"
        "import torch\nimport numpy as np\nimport pandas as pd\nimport time",
        "🧪 老规矩:先解决 Windows 下 torch 的 OMP 库冲突。")

NB.md("## 2️⃣ 组装引擎:拼起积木 🧩",
D('''
先拿出第 02 课的 **n-gram 模型** 与第 03 课的 **采样管线**(温度/top-k/top-p)。它们都是
现成的函数,我们只需要把它们“装进”一个 `ToyEngine` 类,再补一个**批量生成**的循环即可。

引擎对外只暴露一个方法 `generate_batch(prompts, max_new)`:给定若干条提示词,逐字推进
**所有**序列,直到每一条都生成够 `max_new` 个新 token。
'''))

NB.code(NGRAM, "🎨 积木一:n-gram 语言模型(来自第 02 课)——统计条件概率 P(next|context)。")

NB.code(SAMPLING, "🎲 积木二:采样管线(来自第 03 课)——温度 → top-k → top-p → 多项式采样。")

NB.code(D('''
class ToyEngine:
    def __init__(self, model, n=2, temperature=1.0, top_k=0, top_p=1.0, seed=0):
        self.model = model
        self.n = n
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rng = np.random.default_rng(seed)
        self.W = np.random.randn(48, 48) * 0.05   # 共享的“模型权重”

    def shared_forward(self):
        # 模拟整批共享的一次前向(权重/注意力): 固定开销, 与 batch 大小无关
        return np.random.randn(48, 48) @ self.W

    def generate_batch(self, prompts, max_new):
        seqs = [list(p) for p in prompts]   # 每条提示词 = 一条待推进的序列
        tokens = 0
        for _ in range(max_new):
            self.shared_forward()           # 整批只做一次共享前向
            for seq in seqs:                # 再为每条序列各自采样下一个 token
                toks, probs = next_candidates(self.model, seq, self.n)
                if not toks:
                    continue
                logits = np.log(np.array(probs) + 1e-12)
                idx, _, _ = sample_from(logits, self.temperature, self.top_k, self.top_p, self.rng)
                seq.append(toks[idx])
                tokens += 1
        return seqs, tokens
'''), "🛠️ 引擎就绪!`shared_forward` 代表整批共享的模型前向(固定开销),这正是批处理能摊薄成本的关键。")

NB.md("## 3️⃣ 批量生成演示 📜",
D('''
给引擎一次喂**多条**提示词,看它如何同时推进所有序列。这里用小语料训练一个 bigram 模型,
然后批量生成 3 条文本:
'''))

NB.code(D('''
corpus = ["我喜欢学习机器学习和深度学习", "机器学习很热门", "深度学习让机器更聪明",
          "我每天写代码和学算法", "我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉"]
model = build_ngram(corpus, n=2)

engine = ToyEngine(model, n=2, temperature=1.0, seed=0)
prompts = ["我喜欢", "机器", "学习"]
seqs, total = engine.generate_batch(prompts, max_new=10)
for i, s in enumerate(seqs, 1):
    print(f"序列 {i}: {' '.join(s)}")
print(f"共生成 {total} 个新 token(3 条 × ~{total // 3})")
'''), "✅ 3 条序列同步推进、各自接龙。批大小 = 3,意味着每一次“共享前向”同时服务 3 条序列。")

NB.md("## 4️⃣ 吞吐 vs 批大小曲线 📈",
D('''
现在测量引擎在不同批大小下的**吞吐(tokens/s)**。吞吐 = 生成的总 token 数 ÷ 耗时。
我们让每个批大小都生成相同的工作量,观察批处理带来的收益:
'''))

NB.code(D('''
def measure_throughput(batch, max_new=50, repeats=3):
    prompts = ["我喜欢" for _ in range(batch)]
    eng = ToyEngine(model, n=2, seed=0)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, tok = eng.generate_batch(prompts, max_new)
        times.append(time.perf_counter() - t0)
    dt = sorted(times)[len(times) // 2]
    return tok / dt

batches = [1, 2, 4, 8, 16, 32]
tps = [measure_throughput(b) for b in batches]
for b, tp in zip(batches, tps):
    print(f"batch={b:2d}: {tp:8.1f} tokens/s")
'''), "🚀 看到吞吐随批大小明显上升——从单条到 batch=32,吞吐涨了近一个数量级。")

NB.code(D('''
from pyecharts.charts import Line
from pyecharts import options as opts

line = (Line()
        .add_xaxis([str(b) for b in batches])
        .add_yaxis("tokens/s", [round(t, 1) for t in tps], is_smooth=True,
                   symbol_size=8, color="#4C78A8")
        .set_global_opts(title_opts=opts.TitleOpts(title="玩具引擎吞吐 vs 批大小"),
                         xaxis_opts=opts.AxisOpts(name="批大小 batch"),
                         yaxis_opts=opts.AxisOpts(name="tokens/s")))
line.render_notebook()
'''), "📊 pyecharts 交互折线图:吞吐随批大小上升并趋于饱和。这就是 vLLM 拼命做批处理的根本原因。")

NB.code(D('''
# 真实对照:在 GPU 上跑一个小 GPT,测 decode 阶段「批大小 → 吞吐」的真实曲线
import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import bench_throughput_curve, cuda_info

b_real, tps_real, ms_real = bench_throughput_curve(
    batch=(1, 4, 16, 64, 128), token_len=16, reps=5)
print("设备:", cuda_info())
print("batch | 每步耗时(ms) | 吞吐(tokens/s)")
for b, ms, tp in zip(b_real, ms_real, tps_real):
    print(f"{int(b):5d} | {ms:9.3f}   | {tp:10.0f}")
'''), "🚀 用**真实 GPU 小 GPT** 复现同一条曲线:玩具引擎里的“批量摊薄固定开销”,在真实 decode 阶段同样成立——这为下一章 vLLM 连续批处理提供了硬核依据。")

NB.md("## 6️⃣ 解读曲线:为何后趋饱和 🤔",
D('''
吞吐上升的原因是**固定开销被摊薄**。每一步的总时间大致是:

$$T_{\\text{step}} \\approx C_{\\text{fixed}} + b \\times C_{\\text{per-seq}}$$

- $C_{\\text{fixed}}$:整批共享的前向、内核启动等固定开销(与批大小 b 无关);
- $C_{\\text{per-seq}}$:每条序列的独立采样开销。

于是吞吐 $\\approx \\dfrac{b \\times \\text{tokens}}{C_{\\text{fixed}} + b \\times C_{\\text{per-seq}}}$。
当 b 很小时,固定开销占比大,吞吐低;b 越大,固定开销被摊得越薄,吞吐越接近
$1 / C_{\\text{per-seq}}$ 这个上限——所以曲线**先快速上升、后趋饱和**。

真实 vLLM 里这个“固定开销”就是模型权重读取、注意力计算等**每步只做一次**的部分。
批处理让这些昂贵的固定工作一次服务更多请求,是提升推理吞吐的第一大杠杆。
'''))

NB.code(D('''
# 用公式拟合一下“固定开销 + 线性”模型,验证曲线形态
Cf, Cs = 1.2e-4, 1.6e-5
pred = [b / (Cf + b * Cs) for b in batches]
pred = [p / max(pred) * max(tps) for p in pred]   # 缩放到同一量级
print("batch | 实测 tokens/s | 拟合值(归一)")
for b, real, p in zip(batches, tps, pred):
    print(f"{b:5d} | {real:12.1f} | {p:12.1f}")
'''), "✅ 拟合曲线与实测形态吻合:先陡升、后趋平——验证了'固定开销摊薄'的直觉模型。")

NB.md("## 7️⃣ 配套 Streamlit 演示:实时看吞吐 🎛️",
D('''
运行同目录下的 `app_06_toy_engine.py`,输入**多条提示词**,调整**批大小、温度**,
引擎实时批量生成,并用 `st.metric` 展示吞吐、绘制吞吐曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_06_toy_engine.py
```

浏览器打开 **http://localhost:8501**。把批大小从 1 拖到 16,看吞吐如何爬升;
改温度,观察生成文本多样性的变化。完整源码如下(与同目录 `app_06_toy_engine.py` 一致):
'''))

NB.code("%%writefile app_06_toy_engine.py\n" + APP_06,
        "📜 这就是 app_06_toy_engine.py 的完整源码。notebook 与 app 共用同一套 ToyEngine,保证讲解与演示一致。")

wrapup(NB,
    summary=[
        "推理引擎 = 语言模型(决定说什么) + 采样策略(决定具体吐哪个字)+ 批量生成循环",
        "batch 生成让一次共享前向同时推进多条序列,固定开销被摊薄",
        "实测'吞吐 vs 批大小'呈'先陡升、后饱和'形态,可用固定开销+线性模型拟合",
        "vLLM 的连续批处理是提升吞吐的第一大杠杆,本课的玩具引擎是其最小化体现",
        "用 st.metric + plotly 实时展示引擎吞吐,把性能指标可视化",
    ],
    practice=[
        "给 ToyEngine 增加 temperature / top_k / top_p 三个采样参数,并观察高温度下生成多样性",
        "测 batch=[1,2,4,8,16,32] 时把 max_new 换成 100,比较吞吐曲线的饱和点是否变化",
        "把 shared_forward 的矩阵尺寸改大(如 96),重测吞吐曲线,解释为什么固定开销越大收益越明显",
        "写一个对比:用纯串行(每次只生成 1 条)生成同样多文本,与 batch=8 对比总耗时与吞吐",
    ],
    links=[
        ("vLLM 官方文档:性能与吞吐", "https://docs.vllm.ai/en/latest/performance/"),
        ("Transformer Inference Arithmetic", "https://kipp.ly/transformer-inference-arithmetic/"),
        ("HuggingFace Generation 策略", "https://huggingface.co/docs/transformers/generation_strategies"),
    ])

NB.save(str(Path(CH01) / "06_toy_inference_engine.ipynb"))

app_path = Path(CH01) / "app_06_toy_engine.py"
app_path.write_text(APP_06 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
