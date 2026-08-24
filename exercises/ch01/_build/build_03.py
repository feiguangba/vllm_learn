# -*- coding: utf-8 -*-
"""生成 03_sampling_strategies.ipynb 与 app_03_sampling_demo.py"""
from helpers import D, SAMPLING, chapter_cover, wrapup, new_nb, CH01
from pathlib import Path

APP_03 = D('''
# -*- coding: utf-8 -*-
# app_03_sampling_demo.py — 采样策略演示 🎲
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="采样策略 🎲", layout="wide")
st.title("🎲 第 03 课 · 采样策略:greedy / temperature / top-k / top-p")

st.markdown("""
模型每次生成前都会给候选 token 打分(logits)。**采样策略**决定怎么把 logits 变成
真正吐出来的那个字:温度控制分布的“陡峭度”,top-k / top-p 决定候选被压缩到多小。
下方拖动滑杆,实时观察概率分布如何变化,以及采样最终抽中的 token。
""")

VOCAB = ["猫", "狗", "鱼", "鸟", "马", "牛"]
LOGITS = np.array([3.2, 2.1, 1.5, 0.6, 0.2, -0.5], dtype=np.float64)

def softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()

def top_k_mask(logits, k):
    mask = np.zeros_like(logits, dtype=bool)
    if k <= 0 or k >= len(logits):
        mask[:] = True
    else:
        mask[np.argsort(logits)[-k:]] = True
    return mask

def top_p_mask(logits, p):
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = cum - probs[order] <= p
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    return mask

with st.sidebar:
    st.header("🎛️ 参数")
    temperature = st.slider("温度 temperature", 0.1, 3.0, 1.0, 0.1)
    top_k = st.slider("top-k(0=关闭)", 0, 6, 6, 1)
    top_p = st.slider("top-p(1.0=关闭)", 0.1, 1.0, 1.0, 0.05)
    n_sim = st.slider("采样次数(点图)", 10, 500, 200, 10)
    st.caption("温度>1 更随机,<1 更确定;top-k / top-p 压缩候选范围。")

probs = softmax(LOGITS, temperature)
mask = np.ones_like(probs, dtype=bool)
if top_k < 6:
    mask &= top_k_mask(LOGITS, top_k)
if top_p < 1.0:
    mask &= top_p_mask(LOGITS, top_p)
final = probs * mask
final = final / final.sum()

fig = go.Figure()
fig.add_trace(go.Bar(x=VOCAB, y=probs, name="温度后原始概率",
                     marker_color="#C7CDD9", opacity=0.8,
                     text=[f"{p:.3f}" for p in probs], textposition="outside"))
fig.add_trace(go.Bar(x=VOCAB, y=final, name="top-k/top-p 后最终概率",
                     marker_color="#E45756",
                     text=[f"{p:.3f}" for p in final], textposition="outside"))
fig.update_layout(title="候选概率分布:灰色=温度后,红色=再经 top-k/top-p 过滤",
                  yaxis_title="概率", barmode="overlay", height=380,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

rng = np.random.default_rng()
counts = np.zeros(len(VOCAB))
for _ in range(n_sim):
    counts[rng.choice(len(final), p=final)] += 1

pick = rng.choice(len(final), p=final)
st.subheader("🎯 采样结果")
c1, c2, c3 = st.columns(3)
c1.metric("抽中的 token", f"“{VOCAB[pick]}”")
c2.metric("单次最高概率候选", VOCAB[int(np.argmax(final))])
c3.metric("有效候选数", int(mask.sum()))

fig2 = go.Figure(go.Bar(x=VOCAB, y=counts, marker_color="#4C78A8",
                        text=counts, textposition="outside"))
fig2.update_layout(title=f"采样 {n_sim} 次的命中频次(近似于红色概率)",
                   yaxis_title="次数", height=300,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 抽中频次大致正比于最终概率:概率高的候选被抽中的次数更多。")
''')

NB = new_nb("第 03 课 · 采样策略:从 logits 到一句话",
            subtitle="greedy / temperature / top-k / top-p —— 一行公式看清概率如何被重塑,再用玩具 logits 亲手实验三种截断",
            emoji="🎲")

chapter_cover(NB,
    objectives=[
        "理解 greedy 解码:每次选概率最高的 token,简单但容易重复",
        "掌握带温度的 softmax 公式:温度如何控制分布的陡峭与平坦",
        "实现并理解 top-k 与 top-p(nucleus)两种截断策略",
        "对比四种策略在玩具 logits 上的输出,联系 vLLM 的 SamplingParams",
    ],
    toc=[
        ("直觉:从'打分'到'选字'", "logits 是模型给候选打的原始分,采样负责把它变成输出"),
        ("greedy:每次都选最高分", "最简单,却常陷入重复与空洞"),
        ("温度:softmax 的'放大镜'", "T>1 拉平、T<1 拉尖的数学本质"),
        ("top-k:只看前 k 个", "固定数量截断,简单直接"),
        ("top-p(nucleus):按累积概率截断", "动态数量,按质量保留候选"),
        ("四策略对比与 vLLM 连接", "玩具实验 + SamplingParams 文档"),
        ("配套 Streamlit 演示", "app_03_sampling_demo.py:滑杆实时看分布与采样"),
    ],
    links=[
        ("vLLM SamplingParams 官方文档", "https://docs.vllm.ai/en/latest/api/sampling_params.html"),
        ("Holtzman 等《The Curious Case of Neural Text Degeneration》(top-p 出处)", "https://arxiv.org/abs/1904.09751"),
        ("HuggingFace: 集束搜索、采样、top-k、top-p", "https://huggingface.co/blog/how-to-generate"),
    ])

NB.md("## 1️⃣ 直觉:从'打分'到'选字' ⚖️",
D('''
上一课我们看到,模型每生成一步,会先给**所有候选 token** 打一个原始分,叫 **logits**
(未归一化的分数,可正可负)。logits 只是“分数”,还不是概率,更不能直接拿来选字。
**采样策略(sampling strategy)** 负责这最后一步:怎么把 logits 变成真正吐出来的那个字。

想象一场选秀:评委(模型)给每位选手打分(logits),但**最终谁晋级**由“导演”(采样策略)
决定。导演可以:

- 永远让最高分晋级(**greedy**);
- 给分数加点“随机扰动”,让低分选手偶尔逆袭(**temperature**);
- 只让前几名参赛(**top-k**);
- 只让“实力够格”的晋(累积概率达标,**top-p**)。

不同的导演,拍出来的片子风格完全不同。下面我们逐一认识他们。
'''))

NB.code("import os\nos.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", \"TRUE\")\n"
        "import torch\nimport numpy as np\nimport pandas as pd",
        "🧪 老规矩:先解决 Windows 下 torch 的 OMP 库冲突,再导入常用库。")

NB.md("## 2️⃣ greedy:每次都选最高分 🏆",
D('''
**greedy(贪心)** 是最朴素的策略:每次直接选 logits(或概率)最大的那个 token。
它的优点是完全确定、可复现;缺点也很明显——一步选错会一路错下去,而且容易陷入
“好的、很好、非常好”这类重复空洞的循环,因为模型总是选当下最顺口的那个。

在上一课的 n-gram 里,我们其实已经用过 greedy(`toks[0]`)。这里的公式就是:

$$\\hat x_t = \\arg\\max_{x} P(x \\mid x_{<t})$$
'''))

NB.code(SAMPLING, "🎨 这是第 02 课定义过的采样管线,这里展开成 4 个独立小函数:`softmax`(温度)、`top_k_mask`、`top_p_mask`、`sample_from`(完整链路)。下面用一个玩具 logits 逐一演示。")

NB.code(D('''
VOCAB = ["猫", "狗", "鱼", "鸟", "马", "牛"]
LOGITS = np.array([3.2, 2.1, 1.5, 0.6, 0.2, -0.5], dtype=np.float64)

print("候选:", VOCAB)
print("logits:", LOGITS)
greedy_idx = int(np.argmax(LOGITS))
print(f"greedy 选中: {VOCAB[greedy_idx]}  (logits={LOGITS[greedy_idx]})")
'''), "🏆 greedy 只看分数最高那位——这里是“猫”。")

NB.md("## 3️⃣ 温度:softmax 的'放大镜' 🔥",
D('''
要让低分候选也有机会,先把 logits 变成概率。标准 softmax 是:

$$P_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

再加一个**温度 T** 变成:

$$P_i = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$$

温度的作用像**放大镜的焦距**:

- **T → 0**:分布越来越尖,趋近 greedy(最高分概率 → 1);
- **T = 1**:标准 softmax;
- **T > 1**:分布被“摊平”,低分候选概率上升,输出更随机、更多样。

vLLM 里这就是 `temperature` 参数。用代码把不同温度下的分布画出来,一目了然:
'''))

NB.code(D('''
print(f"{'温度':>5} | " + "  ".join(f"{v:>6}" for v in VOCAB))
for T in [0.3, 0.8, 1.0, 2.0]:
    p = softmax(LOGITS, T)
    row = "  ".join(f"{x:.4f}" for x in p)
    print(f"T={T:<4.1f} | {row}")

# 验证:温度越高,最高概率越低(分布更平)
for T in [0.3, 1.0, 3.0]:
    p = softmax(LOGITS, T)
    print(f"T={T}: 最高候选“{VOCAB[int(np.argmax(p))]}”的概率 = {p.max():.4f}")
'''), "🔥 看 T=0.3 时“猫”几乎垄断;T=3.0 时大家概率趋近均分——温度就是随机性的旋钮。")

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

fig = go.Figure()
for T in [0.3, 0.8, 1.0, 2.0, 4.0]:
    p = softmax(LOGITS, T)
    fig.add_trace(go.Bar(x=VOCAB, y=p, name=f"T={T}"))
fig.update_layout(title="不同温度下的概率分布", yaxis_title="概率",
                  barmode="group", height=360,
                  margin=dict(l=10, r=10, t=50, b=10))
fig
'''), "🎨 温度越高,整组柱子越矮越齐(分布平坦);温度越低,柱子越尖(集中到最高分)。这就是'陡峭度'的可视化。")

NB.md("## 4️⃣ top-k:只看前 k 个 ✂️",
D('''
**top-k** 非常直白:只保留 logits 最大的 **k 个**候选,其余概率直接归零,再在剩下的里
重新归一化采样。它固定截断数量,实现简单;缺点是有时候 k 太大(保留了垃圾候选)或太小
(把好候选也砍了)——不同上下文的“合适 k”并不一样。

公式:只对最大的 k 个保留,其余置 0:
'''))

NB.code(D('''
def demo_topk(k):
    mask = top_k_mask(LOGITS, k)
    keep = [v for v, m in zip(VOCAB, mask) if m]
    probs = softmax(LOGITS)
    re = probs * mask
    re = re / re.sum()
    print(f"top-k={k}: 保留 {keep}  重新归一化后概率: {[f'{x:.3f}' for x in re]}")

for k in [1, 3, 5]:
    demo_topk(k)
'''), "✂️ top-k 是'固定人数'的裁切:无论上下文多复杂,永远只看前 k 名。vLLM 里对应 `top_k`。")

NB.md("## 5️⃣ top-p(nucleus):按累积概率截断 🎯",
D('''
**top-p(也叫 nucleus,核心采样)** 更聪明:它不固定人数,而是**按质量**截断——从概率
最大的 token 开始往下累积,直到累积概率刚好超过 **p**。这样候选集合的大小是**动态的**:

- 若某个 token 概率极高(模型很有把握),候选只有 1 个,几乎就是 greedy;
- 若一堆 token 概率都差不多(模型没把握),候选会多留几个,保留多样性。

这正是 Holtzman 等人在《The Curious Case of Neural Text Degeneration》里提出的,
是目前 GPT 系列默认使用的策略之一。vLLM 里对应 `top_p`。
'''))

NB.code(D('''
def demo_topp(p):
    mask = top_p_mask(LOGITS, p)
    keep = [v for v, m in zip(VOCAB, mask) if m]
    print(f"top-p={p}: 保留 {len(keep)} 个候选 {keep}")

for p in [0.5, 0.8, 0.95]:
    demo_topp(p)
'''), "🎯 top-p 的候选数量会随分布形态自动伸缩:分布集中时少留,分散时多留。这就是它与 top-k 的本质区别。")

NB.md("## 6️⃣ 四策略对比与 vLLM 连接 🔗",
D('''
把四种策略放到一起看:greedy 只认最高;top-k 固定人数;top-p 动态人数;温度则改变
整体“陡峭度”。实践中它们常**叠加**使用:先 temperature 缩放,再 top-k 或 top-p 截断,
最后归一化采样。

在 **vLLM** 中,这一切都封装在 `SamplingParams` 里——你只需在 `LLM.generate()` 或
`Chat` 时传一个 `SamplingParams(temperature=0.8, top_p=0.95, top_k=50)` 即可。
'''))

NB.code(D('''
rng = np.random.default_rng(42)
print(f"{'策略':<22} | 10 次采样结果")
configs = [
    ("greedy", dict(top_k=1)),
    ("temperature=0.5", dict(temperature=0.5)),
    ("top-k=3", dict(top_k=3)),
    ("top-p=0.7", dict(top_p=0.7)),
]
for name, kw in configs:
    picks = []
    for _ in range(10):
        idx, _, _ = sample_from(LOGITS, rng=rng, **kw)
        picks.append(VOCAB[idx])
    print(f"{name:<22} | " + " ".join(picks))
'''), "🎲 greedy 永远是“猫猫猫…”;温度/top-k/top-p 则给出有变化但符合分布的采样。注意 `sample_from` 已自动做了温度→top-k→top-p→归一化的完整链路。")

NB.md("## 7️⃣ 配套 Streamlit 演示:动手拧旋钮 🎛️",
D('''
运行同目录下的 `app_03_sampling_demo.py`,**温度 / top-k / top-p 三根滑杆**任意拖动,
概率分布柱状图实时重绘,右侧还能看到采样几百次后的命中频次:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_03_sampling_demo.py
```

浏览器打开 **http://localhost:8501**。把温度从 0.1 拖到 3,看柱子由尖变平;把 top-p 调小,
看候选被一步步压缩。完整源码如下(与同目录 `app_03_sampling_demo.py` 一致):
'''))

NB.code("%%writefile app_03_sampling_demo.py\n" + APP_03,
        "📜 这就是 app_03_sampling_demo.py 的完整源码。notebook 与 app 共用同一套 softmax / top_k / top_p 实现。")

wrapup(NB,
    summary=[
        "logits 是模型给候选打的原始分,采样策略负责把它变成最终输出的 token",
        "greedy 每次选最高分,确定但易重复空洞;温度通过软化分布引入可控随机性",
        "top-k 固定数量截断候选;top-p(nucleus)按累积概率动态截断,更贴近真实分布形态",
        "四策略可叠加:温度→top-k/top-p→归一化→采样,vLLM 里用 SamplingParams 一行配置",
        "不同策略适用于不同场景:创作求多样、问答求稳定,是生产里常见的调参权衡",
    ],
    practice=[
        "把 LOGITS 换成一组差距很小的分数,观察 top-k 与 top-p 谁保留的候选更多",
        "写一个函数,固定 seed 下比较 temperature=0.1 与 temperature=2 各生成 20 次的去重 token 种类数",
        "给 sample_from 增加 'temperature 为 0 时退化为 greedy' 的保护逻辑,并验证",
        "读 vLLM SamplingParams 文档,找出默认 temperature / top_p / top_k 的值,思考生产为何如此配置",
    ],
    links=[
        ("vLLM SamplingParams 官方文档", "https://docs.vllm.ai/en/latest/api/sampling_params.html"),
        ("Holtzman 等 top-p 论文", "https://arxiv.org/abs/1904.09751"),
        ("HuggingFace 博客: How to generate", "https://huggingface.co/blog/how-to-generate"),
    ])

NB.save(str(Path(CH01) / "03_sampling_strategies.ipynb"))

app_path = Path(CH01) / "app_03_sampling_demo.py"
app_path.write_text(APP_03 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
