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
