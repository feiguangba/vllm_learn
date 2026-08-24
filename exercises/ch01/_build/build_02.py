# -*- coding: utf-8 -*-
"""生成 02_autoregressive_generation.ipynb 与 app_02_generate_demo.py"""
from helpers import D, NGRAM, SAMPLING, chapter_cover, wrapup, new_nb, CH01
from pathlib import Path

APP_02 = D('''
# -*- coding: utf-8 -*-
# app_02_generate_demo.py — 自回归生成演示 🔁
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict, Counter

st.set_page_config(page_title="自回归生成 🔁", layout="wide")
st.title("🔁 第 02 课 · 自回归生成:一字接一字")

st.markdown("""
大模型生成文本就像**接龙游戏**:它一次只预测**下一个** token,再把新 token 拼回上下文,
如此循环 L 次就得到 L 个新 token(需要 **L 次前向**)。下面用一个 n-gram 玩具模型
在选定的中文语料上,逐步预测下一个字,并展示每一步的候选概率。
""")

CORPORA = {
    "动物与生活": ["我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉", "小狗喜欢奔跑和玩耍", "小猫喜欢睡觉"],
    "机器学习": ["我喜欢学习机器学习和深度学习", "机器学习很热门", "深度学习让机器更聪明", "我每天写代码和学算法"],
    "美食": ["我喜欢吃火锅和饺子", "饺子很好吃", "火锅很辣很香", "我喜欢喝豆浆和牛奶"],
}

def build_ngram(corpus, n):
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

with st.sidebar:
    st.header("🎛️ 参数")
    corpus_name = st.selectbox("语料主题", list(CORPORA))
    n = st.select_slider("n-gram 阶数", options=[2, 3],
                         format_func=lambda v: "bigram" if v == 2 else "trigram")
    start = st.text_input("起始文本", "我喜欢")
    steps = st.slider("生成步数", 1, 15, 6, 1)
    st.caption("阶数越高,条件越具体,预测越“有依据”。")

model = build_ngram(CORPORA[corpus_name], n)
path = list(start)
cand_hist = []
for _ in range(steps):
    toks, probs = next_candidates(model, path, n)
    if not toks:
        break
    path.append(toks[0])
    cand_hist.append((toks, probs))

generated = list(start) + path[len(start):]
st.subheader("🖊️ 生成结果")
st.write(" " + " ".join(generated))
c1, c2, c3 = st.columns(3)
c1.metric("提示词长度", len(start))
c2.metric("生成 token 数", len(generated) - len(start))
c3.metric("所需前向次数", len(generated) - len(start))

st.subheader("🧪 每一步的候选概率(greedy 取最高)")
for step, (toks, probs) in enumerate(cand_hist):
    ctx_show = "".join(generated[:len(start) + step])
    fig = go.Figure(go.Bar(x=toks, y=probs,
                           text=[f"{p:.2f}" for p in probs],
                           textposition="outside",
                           marker_color="#4C78A8"))
    fig.update_layout(title=f"第 {step + 1} 步 · 上下文“{ctx_show}”",
                      yaxis_title="概率", height=230,
                      margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.caption("⭐ 观察:每一步的候选概率都来自“上下文里最近 n-1 个字”的条件分布;"
           "生成 L 个 token 就要做 L 次这样的预测,无法并行。")
''')

NB = new_nb("第 02 课 · 自回归生成:一字接一字",
            subtitle="从 n-gram 玩具模型看懂'预测下一个 token'——生成长度 L 需要 L 次前向,解码路径逐字可视化",
            emoji="🔁")

chapter_cover(NB,
    objectives=[
        "理解自回归(autoregressive)生成:每一步只预测下一个 token,再拼回输入",
        "用 bigram/trigram n-gram 玩具模型在小中文语料上做逐步预测",
        "理解'生成长度 L 需要 L 次前向'的因果链条与串行本质",
        "用 plotly 可视化逐步解码路径与每一步的候选概率分布",
    ],
    toc=[
        ("直觉:接龙游戏与自动补全", "LLM 生成就是'逐字接龙',一次只吐一个字"),
        ("n-gram 玩具模型", "统计'看到什么上下文,接着说什么'的条件概率"),
        ("自回归生成:把上一个 token 拼回上下文", "new token 追加进 context 再预测下一步"),
        ("L 次前向:为什么生成快不了", "串行依赖让生成长度 L 必须做 L 次推理"),
        ("解码路径可视化", "plotly 把提示词与生成 token 逐字铺开"),
        ("候选概率分布", "每一步模型心里都在给候选 token '打勾'"),
        ("配套 Streamlit 演示", "app_02_generate_demo.py:选语料/阶数/起点/步数逐字生成"),
    ],
    links=[
        ("vLLM 官方文档(解码与采样概览)", "https://docs.vllm.ai/en/latest/features/sampling_params.html"),
        ("Andrej Karpathy: The GPT-2 讲解(生成循环)", "https://www.youtube.com/watch?v=kCc8FmEb1nY"),
        ("HuggingFace: 文本生成策略", "https://huggingface.co/docs/transformers/generation_strategies"),
    ])

NB.md("## 1️⃣ 直觉:接龙游戏与自动补全 🧩",
D('''
想象你在玩**词语接龙**:我说“机器”,你接“器学”,再接“学习”……大模型生成文本就是
这么一回事——它一次只看当前已生成的整段话,只预测**下一个** token,然后把这个新 token
拼到末尾,作为下一次预测的输入。这个过程叫**自回归(autoregressive)生成**,意思是
“把自己过去生成的输出,当作新的输入”。

打个比方:它不是一次画好一整幅画,而是一笔一笔地描,每一笔都依赖前面所有笔的落点。
所以模型不会“整段凭空出现”,而是**一个 token 一个 token 地往外吐**。这也是为什么
打字机式的流式输出看着很“聪明”——其实那是自回归的必然节奏。
'''))

NB.code("import os\nos.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", \"TRUE\")\n"
        "import torch\nimport numpy as np\nimport pandas as pd",
        "🧪 老规矩:先解决 Windows 下 torch 的 OMP 库冲突,再导入常用库。")

NB.md("## 2️⃣ n-gram 玩具模型:统计'看到什么,接着说什么' 📖",
D('''
为了看懂自回归,我们造一个**玩具语言模型**。最朴素的思路是 **n-gram**:假设下一个 token
只依赖它前面最近的 **n-1 个** token,然后在语料里统计:

$$P(x_t \mid x_{t-n+1},\dots,x_{t-1}) = \frac{\\text{count}(x_{t-n+1},\dots,x_{t})}{\\text{count}(x_{t-n+1},\dots,x_{t-1})}$$

- **bigram(n=2)**:只看前 1 个字,`喜欢 -> 学` 的概率;
- **trigram(n=3)**:看前 2 个字,`我喜欢 -> 学` 的概率。

这就像一位只看你“最近一句话”的接龙高手:它记不住全部历史,只按最近的口吻接话。
模型很小,却完整承载了“条件概率 + 生成”两个核心概念。
'''))

NB.code(SAMPLING, "🎲 这是第 03 课要用的**采样管线**(温度/top-k/top-p)。这里先定义好,让 `generate_ngram` 能直接用上——第 06 课还会整段复用。")

NB.code(NGRAM, "🎨 n-gram 模型就三件事:统计相邻 token 对的出现次数 → 查上下文 → 归一化成概率。`sample_from` 会帮我们在候选里按概率挑一个字。")

NB.md("## 3️⃣ 自回归生成:把上一个 token 拼回上下文 🔁",
D('''
有了玩具模型,自回归生成就一个循环:

1. 拿当前已生成的 token 序列作**上下文**;
2. 用 n-gram 查表得到“下一个 token 的候选概率”;
3. 选一个(这里先 greedy 取最高概率);
4. 把这个新 token **追加**到上下文末尾;
5. 回到第 1 步,直到凑够 L 个 token。

关键就在第 4 步:**新 token 会改变下一次的上下文**,这正是“自回归”名字的来源。
下面在“动物与生活”小语料上用 bigram 从“我喜欢”开始生成:
'''))

NB.code(D('''
corpus = ["我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉", "小狗喜欢奔跑和玩耍", "小猫喜欢睡觉"]
model = build_ngram(corpus, n=2)
out, path = generate_ngram(model, n=2, start="我喜欢", steps=8)
print("生成结果:", "".join(out))
print("提示词:  我喜欢   |  模型新增:", "".join(out[len("我喜欢"):]))
print()
print("逐步解码路径:")
for step, (ctx, toks, probs, chosen, _, _) in enumerate(path, 1):
    cand = " ".join(f"{t}({p:.2f})" for t, p in zip(toks, probs))
    print(f"  第 {step} 步 上下文“{''.join(ctx)}” -> 候选[{cand}] -> 选中“{chosen}”")
'''), "💡 你会看到:每一步只吐一个字,且第 2 步起上下文里多了上一步刚生成的字——生成结果会反过来影响后续预测。")

NB.md("## 4️⃣ L 次前向:为什么生成快不了 ⏱️",
D('''
注意上面循环**无法并行**:要算第 3 个 token,必须等第 2 个 token 已经生成;要算第 2 个,
又得等第 1 个。于是“生成长度 L”这件事,本质是一条**串行链**:

> 生成 L 个新 token ≈ 需要 **L 次前向推理**,一次一个,排着队来。

这与“读提示词”不同——提示词一开始就在那里,可以一次性全部喂进去(这就是下一课要讲的
**prefill**)。而**生成**阶段只能步步为营,这也是 LLM 推理“慢”的根源之一。
vLLM 的核心优化之一,就是想办法让这 L 次串行前向**一次同时跑很多序列**(批处理),
从而摊薄每次前向的固定开销——这正是第 06 课吞吐曲线的由来。
'''))

NB.code(D('''
# 用一个计数实验验证“L 个 token = L 次前向”:统计 generate_ngram 内部实际调用次数
def count_forward(model, n, start, steps):
    context, calls = list(start), 0
    for _ in range(steps):
        toks, _ = next_candidates(model, context, n)
        if not toks:
            break
        calls += 1
        context.append(toks[0])
    return calls

for L in [1, 3, 5, 8]:
    c = count_forward(model, 2, "我喜欢", L)
    print(f"想生成 {L} 个 token -> 需要 {c} 次前向")
'''), "✅ 结论实锤:生成几个 token,就做几次前向。生成长度线性决定前向次数,无法靠加大 batch 缩短这条串行链(但可以靠批处理摊薄单次成本)。")

NB.md("## 5️⃣ 解码路径可视化:逐字铺开 🎨",
D('''
把整条解码路径用 **plotly** 画出来:蓝色是提示词,红色是模型一个个生成的 token,
从左到右就是模型“接龙”的落子顺序。一眼就能看出生成是**串行追加**的。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

def viz_decode(tokens, start_len):
    n = len(tokens)
    colors = ["#9ECAE1" if i < start_len else "#E45756" for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=[0] * n, mode="lines",
                             line=dict(color="#BBBBBB", width=1)))
    fig.add_trace(go.Scatter(x=list(range(n)), y=[0] * n, mode="markers+text",
                             text=tokens, textposition="top center",
                             marker=dict(size=20, color=colors)))
    fig.update_layout(title="自回归解码路径(蓝=提示词,红=模型生成)",
                      xaxis_title="token 位置", yaxis=dict(showticklabels=False,
                      range=[-0.6, 0.6]), height=280,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

out, _ = generate_ngram(model, n=2, start="我喜欢", steps=8)
viz_decode(out, len("我喜欢"))
'''), "🎨 红字一个个接在蓝字后面:这就是'逐字接龙'的可视化。每个红字都依赖它左边所有字,所以只能从左往右依次产生。")

NB.md("## 6️⃣ 候选概率分布:模型心里在'打勾' 🎯",
D('''
每生成一步,模型并不是只“知道”一个字,而是给**所有候选 token** 都打了分(概率)。
greedy 只是挑了概率最高那个。把每一步的候选概率画成柱状图,就能看清模型“心里在想什么”:
同一上下文下,哪些字更可能被选中一目了然。
'''))

NB.code(D('''
def viz_probs(toks, probs):
    fig = go.Figure(go.Bar(x=toks, y=probs, text=[f"{p:.3f}" for p in probs],
                           textposition="outside", marker_color="#4C78A8"))
    fig.update_layout(title="下一步候选概率分布(greedy 选最高)",
                      xaxis_title="候选 token", yaxis_title="概率",
                      height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig

ctx = ["我", "喜"]
toks, probs = next_candidates(model, ctx, 2)
print("上下文“我 喜” -> 候选:", toks)
print("概率:", [f"{p:.3f}" for p in probs])
viz_probs(toks, probs)
'''), "🎯 柱越高越容易被选中。注意概率总和为 1——这就是一个离散条件概率分布,也是下一课'采样策略'的输入。")

NB.md("## 7️⃣ 配套 Streamlit 演示:动手玩接龙 🎛️",
D('''
运行同目录下的 `app_02_generate_demo.py`,可以**选语料主题、切换 bigram/trigram、
改起始文本、调生成步数**,每一步的候选概率柱状图与最终生成结果实时刷新:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_02_generate_demo.py
```

浏览器打开 **http://localhost:8501**。试试把阶数从 2 换成 3,观察预测是否变得更“有依据”;
把步数拖大,看模型如何一步步接下去。完整源码如下(与同目录 `app_02_generate_demo.py` 一致):
'''))

NB.code("%%writefile app_02_generate_demo.py\n" + APP_02,
        "📜 这就是 app_02_generate_demo.py 的完整源码。notebook 与 app 共用同一套 build_ngram / next_candidates,保证讲解与演示一致。")

wrapup(NB,
    summary=[
        "自回归生成 = 逐字接龙:每一步只预测下一个 token,并把它拼回上下文",
        "n-gram 玩具模型用条件概率 P(next|context) 刻画'看到什么接着说什么'",
        "生成 L 个 token 需要 L 次前向,是一条无法并行的串行链,这是 LLM 推理慢的根源之一",
        "每一步模型都会给所有候选 token 打概率分,greedy 只是挑最高那个",
        "plotly 把解码路径与候选概率可视化,建立对'生成'过程的直观感受",
    ],
    practice=[
        "给 generate_ngram 的 greedy 改成按概率随机采样,观察多次生成结果是否不同",
        "把语料换成一段英文(如 \"the cat sat on the mat\"),用 bigram 生成,观察 the 的跟随分布",
        "对比 bigram 与 trigram 在相同起始文本下的生成路径差异,解释为什么 trigram 更'有依据'",
        "写一个函数统计:给定 n 与语料,bigram/trigram 各覆盖了多少种上下文,体会 n 增大后数据稀疏问题",
    ],
    links=[
        ("vLLM 采样参数文档(下节课的主角)", "https://docs.vllm.ai/en/latest/api/sampling_params.html"),
        ("HuggingFace Generation 策略", "https://huggingface.co/docs/transformers/generation_strategies"),
        ("Attention Is All You Need(自回归解码背景)", "https://arxiv.org/abs/1706.03762"),
    ])

NB.save(str(Path(CH01) / "02_autoregressive_generation.ipynb"))

app_path = Path(CH01) / "app_02_generate_demo.py"
app_path.write_text(APP_02 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
