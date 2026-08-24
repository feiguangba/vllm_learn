# -*- coding: utf-8 -*-
"""生成 01_token_and_tokenizer.ipynb 与 app_01_token_demo.py"""
from helpers import D, TOK_CHAR, TOK_BPE, chapter_cover, wrapup, new_nb, CH01
from pathlib import Path

APP_01 = D('''
# -*- coding: utf-8 -*-
# app_01_token_demo.py — 认识 Token 与分词器 🏷️
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(page_title="Token 与分词器 🏷️", layout="wide")
st.title("🏷️ 第 01 课 · 认识 Token 与分词器")

st.markdown("""
语言模型吃的不是“字”,而是 **token** —— 词汇表里最小的意义单位。这就像搭乐高:
一段文字先被拆成一块块积木(token),每块积木对应一个编号(id),模型只认得这些编号。
下方输入任意文字,切换**字符级**或**简化 BPE** 分词,实时观察它被切成哪些积木、每块出现几次。
""")

# ---------------------------------------------------------------- 分词器(与 notebook 一致)
def char_tokenize(text):
    return list(text)

def merge_pair(tokens, pair):
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(pair[0] + pair[1]); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out

def get_pair_counts(token_lists):
    pairs = Counter()
    for tokens in token_lists:
        for a, b in zip(tokens, tokens[1:]):
            pairs[(a, b)] += 1
    return pairs

def train_bpe(corpus, num_merges):
    vocab = [list(w) + ["</w>"] for w in corpus]
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = [merge_pair(w, best) for w in vocab]
        merges.append(best)
    return merges

def bpe_tokenize(text, merges):
    tokens = list(text) + ["</w>"]
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens

# ---------------------------------------------------------------- 侧边栏参数
CORPUS = ["机器学习", "深度学习", "机器翻译", "学习效率", "翻译机器", "学习机器"]
with st.sidebar:
    st.header("🎛️ 参数")
    mode = st.radio("分词方式", ["字符级", "简化 BPE"])
    num_merges = st.slider("BPE 合并次数", 0, 20, 6, 1)
    show_unk = st.checkbox("展示 token → id 映射", value=True)
    st.caption("BPE 合并次数越多,词表里越可能出现更长的“复合 token”。")

text = st.text_area("✏️ 输入要分词的文字", "机器学习让翻译机器更聪明")
merges = train_bpe(CORPUS, num_merges) if mode == "简化 BPE" else []

# ---------------------------------------------------------------- 执行分词
if mode == "字符级":
    tokens = char_tokenize(text)
    method = "字符级:每个字符(含空格)一个 token"
else:
    tokens = bpe_tokenize(text, merges)
    method = f"简化 BPE:已合并 {len(merges)} 对高频相邻字符"

freq = Counter(tokens)
vocab = {t: i for i, t in enumerate(freq)}

c1, c2, c3, c4 = st.columns(4)
c1.metric("token 总数", len(tokens))
c2.metric("去重后词表大小", len(vocab))
c3.metric("平均 token 长度", f"{sum(len(t) for t in tokens) / max(len(tokens), 1):.2f} 字符")
c4.metric("最高频 token", f"“{max(freq, key=freq.get)}” ×{freq[max(freq, key=freq.get)]}")
st.caption(method)

# ---------------------------------------------------------------- token 切分展示
st.subheader("🧩 Token 切分结果")
st.write(" | ".join(f"[{t}]" for t in tokens))

if show_unk:
    st.subheader("🔢 token → id 映射")
    st.write("  ".join(f"{t}→{i}" for t, i in vocab.items()))

# ---------------------------------------------------------------- plotly 频次柱状图
order = [t for t, _ in freq.most_common()]
counts = [freq[t] for t in order]
fig = go.Figure(go.Bar(
    x=order, y=counts,
    marker_color=["#4C78A8" if len(t) == 1 else "#E45756" for t in order],
    text=counts, textposition="outside"))
fig.update_layout(
    title="各 token 出现频次(蓝=单字符,红=多字符 token)",
    xaxis_title="token", yaxis_title="频次",
    height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 观察:BPE 合并次数增加后,红色柱(多字符 token)会变多变长,"
           "说明“机器”“学习”这类高频组合被合并成了一个整体。")

st.markdown("""
> 💡 **结论**:分词器决定了模型“看世界”的粒度。字符级最细但 token 多、序列长;
> BPE 通过合并高频相邻对,在“词表大小”和“序列长度”之间取得平衡,是 GPT 系列等
> 主流模型的默认选择(实际用的是更细的 byte-level BPE,原理同源)。
""")
''')

NB = new_nb("第 01 课 · 认识 Token 与分词器",
            subtitle="模型吃的不是字,是 token —— 从字符级到 BPE,手写一个迷你分词器,看懂 token → id 的映射",
            emoji="🏷️")

chapter_cover(NB,
    objectives=[
        "理解 token 与词表(vocabulary)是什么,为什么模型只认 token 不认字",
        "手写字符级分词器与简化 BPE(字节对编码)训练循环",
        "掌握 token → id 的双向映射,理解 <unk> 与词尾标记 </w> 的作用",
        "用 pyecharts + plotly 双图可视化 token 切分与频次",
    ],
    toc=[
        ("直觉:乐高积木与 token", "文字是一堆积木,分词器负责把它们拆成最小可拼装单元"),
        ("字符级分词:最朴素的做法", "每个字符一个 token,三行代码搞定"),
        ("token → id:模型只认数字", "词表把 token 翻译成编号,<unk> 兜底生僻字符"),
        ("BPE:合并最频繁的相邻对", "从字符出发反复合并,词表与序列长度的平衡术"),
        ("中文多 token:一个词拆几块?", "没有空格分词的中文,如何被切成多个 token"),
        ("双图可视化:频次与切分", "pyecharts 柱状图 + plotly 交互图"),
        ("配套 Streamlit 演示", "app_01_token_demo.py:输入文字实时看切分与频次"),
    ],
    links=[
        ("GPT-2 的 byte-level BPE 说明", "https://github.com/openai/gpt-2/blob/master/src/encoder.py"),
        ("Sennrich 等《Neural Machine Translation ... Subword Units》(BPE 出处)", "https://arxiv.org/abs/1508.07909"),
        ("HuggingFace Tokenizer 文档", "https://huggingface.co/docs/transformers/main_classes/tokenizer"),
    ])

NB.md("## 1️⃣ 直觉:乐高积木与 token 🧱",
D('''
你给大模型发一句话,它并不能直接“读”这句话——它读到的是一串数字。把文字变成数字的过程,
分两步:

1. **分词(tokenization)**:把文字切成一个个小片段,每个片段叫一个 **token**;
2. **查表映射**:查一张叫**词表(vocabulary)**的对照表,把每个 token 换成它的编号(id)。

想象你在拼乐高:一段文字是一幅成品图,token 是一块块积木,id 是每块积木的出厂编号。
模型只认编号,不认积木长什么样。所以“怎么切”这件事至关重要——同样一句话,
切成 5 块还是 15 块,直接决定了模型看到的序列长度、以及它能不能认出你写的词。

一个 token 可能是:一个字符、一个英文单词、单词的一部分(子词)、甚至一个中文词。
下面我们从最朴素的切法开始,一路走到工业界主流的 BPE。
'''))

NB.code(TOK_CHAR, "**字符级分词器**的核心就是 `list(text)`:把字符串拆成一个字符列表。虽然简单,但它建立了“文本 → token 序列”的第一种约定。")

NB.md("## 2️⃣ 字符级分词:最朴素的切法 ✂️",
D('''
字符级分词的规则只有一条:**每个字符(包括空格、标点、换行)都单独成为一个 token**。
它什么都不需要训练、不会“不认识”任何字,但代价是:

- 一个 10 个字母的英文单词会变成 10 个 token,序列被拉得很长;
- 模型很难学到“单词”这个概念,因为它看到的永远是单个字母。

我们先拿一句中英混合的句子试试:
'''))

NB.code(D('''
text = "LLM 爱学习 deep learning"
tokens = char_tokenize(text)
print("原文:", text)
print("tokens:", tokens)
print("token 数:", len(tokens))
'''), "✅ 你会看到空格、字母、汉字都被一一拆开。接下来要解决的是:模型如何“读懂”这些 token?")

NB.md("## 3️⃣ token → id:模型只认数字 🔢",
D('''
模型内部的矩阵运算只处理数字,所以必须有一张**词表**把 token 映射成整数 id。
词表就是 `token → id` 的字典;反过来 `id → token` 叫逆词表,生成时把数字再翻译回文字。

词表里还会塞几个特殊 token:

- `<unk>`(unknown):遇到词表外的字符时兜底,避免程序崩溃;
- `<s>` / `</s>`(起止符):标记一句话的开头和结尾;
- `</w>`(词尾标记):BPE 里用来标记“一个词到这儿结束”。

下面在字符级分词基础上建立词表,并把一段文字转成 id 序列:
'''))

NB.code(D('''
corpus_texts = ["你好世界", "深度学习", "hello world"]
vocab = build_vocab([char_tokenize(s) for s in corpus_texts])
print("词表大小:", len(vocab))
print("词表:", vocab)

ids = tokenize_to_ids("你好", vocab)
print("“你好” 的 id 序列:", ids)
'''), "🔍 观察:`tokenize_to_ids` 把每个字符查表换成 id;若把词表导出成文件,就得到了模型“字典”的雏形。")

NB.md("## 4️⃣ BPE:合并最频繁的相邻对 🔗",
D('''
字符级分词太碎,那能不能让模型自动“学会”把常见字符组合成一个 token?这正是
**BPE(Byte Pair Encoding,字节对编码)** 的思路:

1. 初始化:每个词拆成字符,词尾加 `</w>`;
2. 反复统计:在所有词里,哪一对**相邻 token** 出现得最多;
3. 合并:把这对合并成一个新 token,更新所有词;
4. 重复 2-3 步若干次,记录下每次合并的规则(merges)。

训练完后,面对新词就按 `merges` 的顺序贪心合并。这样“机 + 器”出现得多了,就会被合并成
“机器”一个 token;“学习”亦然。词表大小和序列长度之间,于是有了一根可调的旋钮(合并次数)。
'''))

NB.code(TOK_BPE, "**简化 BPE** 的训练循环就三件事:统计相邻对 → 挑最频繁的 → 合并。`merges` 就是学到的“合并规则本”。")

NB.code(D('''
corpus = ["机器学习", "机器翻译", "学习效率", "翻译机器"]
merges, trained_vocab = train_bpe(corpus, num_merges=4)
print("学到的合并规则(按顺序):")
for i, p in enumerate(merges, 1):
    print(f"  第 {i} 次合并: {p[0]} + {p[1]} -> {p[0] + p[1]}")
print()
print("训练后“机器翻译”的切分:", apply_bpe("机器翻译", merges))
print("训练后“学习效率”的切分:", apply_bpe("学习效率", merges))
'''), "🎯 你会看到:“机 + 器”率先被合并成“机器”,随后“机器 + 翻译”可能再合并——子词一步步长出来。")

NB.md("## 5️⃣ 中文多 token:一个词拆几块? 🀄",
D('''
英文用空格天然分词,中文则没有空格,所以“一个中文词会被切成几个 token”完全取决于词表。
这是中文 LLM 推理成本更高的一个微观原因:同样的语义,中文往往需要更多 token。

用上面学到的 BPE 合并规则,对比不同合并次数下“人工智能”的切分粒度:
'''))

NB.code(D('''
corpus2 = ["人工智能", "机器学习", "人工成本", "智能家居", "语言模型"]
for k in [0, 1, 3, 5]:
    m, _ = train_bpe(corpus2, num_merges=k)
    pieces = apply_bpe("人工智能", m)
    print(f"合并 {k} 次: 人工智能 -> {pieces}   ({len(pieces)} 个 token)")
'''), "⚠️ 坑:同一个词在不同词表里 token 数可能不同。评价一个分词器时,常看“平均每个词拆成几个 token”,它直接影响序列长度与推理成本。")

NB.md("## 6️⃣ 双图可视化:频次与切分 📊",
D('''
先用 **pyecharts 柱状图**看一段中文文本的 token 频次分布——哪个 token 出现最多,一目了然。
再用 **plotly 交互图**对比字符级与 BPE 两种切法下的 token 序列(悬停可看 token 内容)。
'''))

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts
from collections import Counter

sample = "机器翻译和机器学习都是人工智能的重要方向,机器正在学会学习"
toks = char_tokenize(sample)
freq = Counter(toks)
order = [t for t, _ in freq.most_common()]

bar = (Bar()
       .add_xaxis(order)
       .add_yaxis("出现次数", [freq[t] for t in order], color="#4C78A8")
       .set_global_opts(title_opts=opts.TitleOpts(title="字符级 token 频次分布"),
                        xaxis_opts=opts.AxisOpts(name="token", axislabel_opts=opts.LabelOpts(rotate=45)),
                        yaxis_opts=opts.AxisOpts(name="频次")))
bar.render_notebook()
'''), "📊 pyecharts 交互图:可缩放、可悬停看数值。字符级分词下“机”“器”“学”“习”各算各的,频次被摊薄了。")

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

def viz_tokens(char_toks, bpe_toks):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(char_toks))), y=[1]*len(char_toks),
                         text=char_toks, textposition="inside", name="字符级",
                         marker_color="#4C78A8"))
    fig.add_trace(go.Bar(x=list(range(len(bpe_toks))), y=[1]*len(bpe_toks),
                         text=bpe_toks, textposition="inside", name="简化 BPE",
                         marker_color="#E45756"))
    fig.update_layout(barmode="group", title="同一句话:字符级 vs BPE 的 token 序列",
                      yaxis=dict(showticklabels=False), height=320,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

m, _ = train_bpe(["机器翻译", "机器学习", "人工智能"], num_merges=5)
viz_tokens(char_tokenize("机器翻译人工智能"), apply_bpe("机器翻译人工智能", m)).show()
'''), "🎨 plotly 图:蓝色是字符级(每块积木=1 字符),红色是 BPE(相邻字符被合并成大块)。同样一句话,积木数量明显变少。")

NB.md("## 7️⃣ 配套 Streamlit 演示:输入文字,实时看切分 🎛️",
D('''
光看静态图不过瘾?运行同目录下的 `app_01_token_demo.py`,可以**实时输入任意文字**、
切换字符级 / BPE、拖动合并次数,切分结果与频次柱状图随之刷新:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_01_token_demo.py
```

浏览器打开 **http://localhost:8501**。建议输入一句中英混合的话,把 BPE 合并次数从 0 拖到 20,
观察“机器学习”这类高频组合如何一步步合并成一个 token。完整源码如下(与同目录
`app_01_token_demo.py` 一字不差):
'''))

NB.code(APP_01, "📜 这就是 app_01_token_demo.py 的完整源码,notebook 与 app 共用同一套 char_tokenize / train_bpe 实现,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "token 是模型词汇表里的最小单位,分词器把文字切成 token,词表把 token 映射成 id",
        "字符级分词最简单但序列长;BPE 通过反复合并最频繁相邻对,平衡词表大小与序列长度",
        "特殊 token(<unk>、</w>)承担兜底与边界标记职责,是词表不可缺的部分",
        "中文无空格分词,同一词的 token 数随词表而异,是中文推理成本更高的微观原因之一",
        "pyecharts + plotly 双图把“切分粒度”可视化,建立对 token 的直觉",
    ],
    practice=[
        "把 corpus 换成一段英文文本,重跑 BPE,观察 the/ing 等常见片段是否被合并出来",
        "给 build_vocab 增加特殊 token(<unk>、<s>、</s>),并让 tokenize_to_ids 处理词表外字符",
        "写一个统计函数:给定一段中文和 merges,返回平均每个词被切成几个 token",
        "对比“字符级 vs BPE”在 100 字中文上的 token 总数,体会序列长度的差异",
    ],
    links=[
        ("Sennrich 等 BPE 论文", "https://arxiv.org/abs/1508.07909"),
        ("GPT-2 encoder.py(byte-level BPE 参考实现)", "https://github.com/openai/gpt-2/blob/master/src/encoder.py"),
    ])

NB.save(str(Path(CH01) / "01_token_and_tokenizer.ipynb"))

app_path = Path(CH01) / "app_01_token_demo.py"
app_path.write_text(APP_01 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
