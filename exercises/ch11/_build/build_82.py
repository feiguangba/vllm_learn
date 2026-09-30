# -*- coding: utf-8 -*-
"""生成 82_mindir.ipynb 与 app_82_mindir.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_82 = D('''
# -*- coding: utf-8 -*-
# app_82_mindir.py — MindIR 中间表示浏览器:计算图 DAG 一览 📦
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📦 82 · MindIR 中间表示", layout="wide")
st.title("📦 第 82 课 · MindIR 中间表示:像“话单”一样的模型文件")

st.markdown("""
**MindIR(MindSpore IR)** 是把训练好的模型固化成的一个**计算图文件**(`.mindir`):
里面存着算子列表(nodes)、张量流向(edges)与权重(parameters),不含任何 Python 代码。
它就像餐厅之间传递的**菜谱**——后厨(MindSpore)写完菜谱,任何会看菜谱的厨师
(云侧 Runtime / Lite / MindIE)都能照着做出同一道菜。
下方选择一个示例计算图,浏览它的节点 / 边,并在网络图中交互拖拽。
""")

GRAPHS = {
    "线性网络(3 算子)": {
        "nodes": [("input", "Parameter"), ("matmul", "MatMul"), ("add_bias", "Add"),
                  ("relu", "ReLU"), ("output", "Output")],
        "edges": [("input", "matmul"), ("matmul", "add_bias"), ("add_bias", "relu"), ("relu", "output")],
        "desc": "MindIR 里最常见的形状:一条笔直的数据流水线。",
    },
    "残差块(分支 + 汇合)": {
        "nodes": [("input", "Parameter"), ("conv", "Conv2D"), ("bn", "BatchNorm"),
                  ("skip", "Add(直连)"), ("relu", "ReLU"), ("output", "Output")],
        "edges": [("input", "conv"), ("conv", "bn"), ("bn", "skip"), ("input", "skip"), ("skip", "relu"), ("relu", "output")],
        "desc": "ResNet 残差块:主路卷积,旁路“抄近道”,最后汇合——IR 里就是一个分叉再合并的 DAG。",
    },
    "注意力算子(fused)": {
        "nodes": [("query", "Parameter"), ("key", "Parameter"), ("value", "Parameter"),
                  ("qk", "MatMul"), ("scale", "Mul"), ("softmax", "Softmax"), ("attn", "MatMul"), ("output", "Output")],
        "edges": [("query", "qk"), ("key", "qk"), ("qk", "scale"), ("scale", "softmax"),
                  ("softmax", "attn"), ("value", "attn"), ("attn", "output")],
        "desc": "昇腾上常被“融合”成一个算子的注意力模式:MatMul→Scale→Softmax→MatMul。",
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    gname = st.selectbox("选择示例计算图", list(GRAPHS.keys()))
    show_weights = st.checkbox("展示 Parameter 权重形状", value=True)
    node_size = st.slider("节点半径", 12, 30, 18, 1)
    st.caption("MindIR 的“节点”就是算子,“边”就是张量依赖。")

G = GRAPHS[gname]
nodes, edges = G["nodes"], G["edges"]
names = [n[0] for n in nodes]
kinds = [n[1] for n in nodes]

# 简单分层布局:按拓扑排序(先输入的在前)
pos = {n: i for i, n in enumerate(names)}
x = [pos[n] for n in names]
y = [0.5 - (0.6 if k in ("Parameter", "Output") else 0.0) for n, k in zip(names, kinds)]

c1, c2, c3 = st.columns(3)
c1.metric("节点数", len(nodes))
c2.metric("边数", len(edges))
c3.metric("是否含环", "否(DAG)")
st.caption(G["desc"])

# ---------------- 节点 / 边表格 ----------------
st.subheader("🗂️ 节点与边清单")
st.dataframe(pd.DataFrame({"节点": names, "类型": kinds}), width="stretch")
st.dataframe(pd.DataFrame({"源节点": [e[0] for e in edges], "目标节点": [e[1] for e in edges]}),
             width="stretch")
if show_weights:
    st.caption("Parameter 节点还携带权重张量(形状 / 类型),例如 MatMul 的 weight 形状 (16, 16)。")

# ---------------- 网络图 ----------------
fig = go.Figure()
for s, t in edges:
    fig.add_trace(go.Scatter(x=[x[pos[s]], x[pos[t]]], y=[y[pos[s]], y[pos[t]]],
                             mode="lines", line=dict(color="#B8C4D8", width=2),
                             hoverinfo="none", showlegend=False))
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers+text", text=names, textposition="top center",
    marker=dict(size=node_size * 2, color=["#E45756" if k == "Output" else "#4C78A8"
                                          for n, k in zip(names, kinds)], line=dict(width=1, color="white")),
    hovertext=[f"{n} ({k})" for n, k in zip(names, kinds)], showlegend=False))
fig.update_layout(title=f"MindIR 计算图: {gname}",
                  xaxis=dict(showticklabels=False, title="拓扑顺序"), yaxis=dict(showticklabels=False),
                  height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 鼠标悬停看算子类型,拖拽缩放看整条依赖链。残差块里的“分叉再汇合”就是 DAG 的标志。")

st.markdown("""
> 💡 **结论**:MindIR = 计算图(DAG)+ 权重 + 元信息,一次导出、多端复用。
> 它和 ONNX 同属“图 IR”,和 TorchScript 同属“图 + 代码”的过渡形态。
> 任何会“读图”的运行时都能执行它——这就是中间表示的威力。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 82 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os, subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
''')

NB = new_nb("第 82 课 · MindIR 与模型转换",
            subtitle="模型的“通用话单”:MindIR 中间表示、转换工具与 ONNX 互通",
            emoji="📦")

chapter_cover(NB,
    objectives=[
        "理解中间表示(IR)为何是跨框架、跨硬件复用的关键",
        "看懂 MindIR 的结构:算子节点、张量边、权重与子图",
        "掌握 MindIR 导出 / 转换 / ONNX 互通的方法",
        "用 torch 手写一个迷你 IR(DAG + 解释器)跑通“生成→序列化→执行”",
        "对比 MindIR / ONNX / TorchScript 三者的取舍",
    ],
    toc=[
        ("直觉:能搬运的菜谱", "IR 把“做法”从“厨具”里剥离出来"),
        ("MindIR 长什么样", "算子、边、权重、子图——一张图讲透"),
        ("手写迷你 IR", "JSON 描述 DAG + 解释器执行,torch 实现"),
        ("转换与 ONNX 互通", "ms.export / 转换器 / ONNX 格式的桥"),
        ("三张菜谱横评", "MindIR vs ONNX vs TorchScript"),
        ("交互图:DAG 网络浏览", "plotly 拖拽看计算图结构"),
        ("配套 Streamlit 演示", "app_82_mindir.py:选择图、看节点与边"),
    ],
    links=[
        ("MindSpore 模型导出文档", "https://www.mindspore.cn/docs/zh-CN/r2.4/design/mindir.html"),
        ("ONNX 官网", "https://onnx.ai"),
        ("TorchScript 文档", "https://pytorch.org/docs/stable/jit.html"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:能搬运的菜谱 🧑‍🍳",
D('''
思考一个问题:你在 PyTorch 里训练的模型,凭什么能跑到昇腾、端侧、甚至是 vLLM 里?
答案是——**中间表示(Intermediate Representation, IR)**。

把模型想成一道菜,训练代码是**后厨的现场做法**(食材、锅具、火候全在 Python 里),
而 IR 是一张**写成通用格式的菜谱**:不看厨房,只记“食材、步骤、顺序”。
换一个会读菜谱的厨师(另一个框架 / 另一块硬件),照做即可。

- **ONNX** 是“全行业通用的菜谱”;
- **TorchScript** 是“PyTorch 自家菜谱,兼带执行环境”;
- **MindIR** 是“MindSpore 自家菜谱,专为昇腾打磨”。

下面画一张“菜谱(IR)”在整条流水线里的位置:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(10, 4.2))
boxes = [("训练框架", "PyTorch / MindSpore", 0.03), ("中间表示 IR", "MindIR / ONNX / TorchScript", 0.38),
         ("推理运行时", "Lite / vLLM / MindIE", 0.73)]
for label, sub, x in boxes:
    ax.add_patch(plt.Rectangle((x, 0.35), 0.24, 0.3, fc="#DFE9F8", ec="#4C78A8", lw=2))
    ax.text(x + 0.12, 0.58, label, ha="center", fontsize=11, fontweight="bold")
    ax.text(x + 0.12, 0.42, sub, ha="center", fontsize=9, color="#444")
for a, b in [(0.27, 0.38), (0.62, 0.73)]:
    ax.annotate("", xy=(b + 0.01, 0.5), xytext=(a, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#E45756", lw=2.5))
ax.text(0.5, 0.85, "一次导出(MindIR/ONNX)→ 多端复用:菜谱只写一遍,谁都能照做",
        fontsize=13, ha="center", fontweight="bold")
ax.text(0.5, 0.14, "中间表示把“模型是什么(图)”与“怎么跑(硬件)”彻底解耦。", fontsize=11, ha="center", color="#555")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
plt.tight_layout(); plt.show()
'''), "🎨 关键洞察:IR 一旦生成,就和“训练框架怎么写的”没关系了——这是它能跨端复用的根本原因。")

NB.md("## 2️⃣ MindIR 长什么样 🗃️",
D('''
MindIR 是一个**函数式计算图**(FunctionGraph),核心成员是:

- **算子节点(Node)**:MatMul / Add / ReLU 等,描述“做什么”;
- **张量边(Edge)**:节点间的数据依赖,描述“顺序”;
- **权重(Parameter / ValueNode)**:常量张量,如 weight、bias;
- **子图(SubGraph)**:条件、循环等控制流被拆成若干子图。

导出时 `mindspore.export` 会做一轮**图优化**(常量折叠、算子融合),再把图序列化进
`.mindir` 文件。下面我们用 torch 画一个典型的注意力计算图 MindIR 形态:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(10, 4.6))
ops = [("Q", "Parameter"), ("K", "Parameter"), ("V", "Parameter"),
       ("MatMul(Q,Kᵀ)", "MatMul"), ("Scale", "Mul"), ("Softmax", "Softmax"),
       ("MatMul(S,V)", "MatMul"), ("Out", "Output")]
n = len(ops)
for i, (label, kind) in enumerate(ops):
    color = "#E45756" if kind == "Output" else "#4C78A8" if kind == "Parameter" else "#F58518"
    ax.add_patch(plt.Rectangle((i * 0.94 / (n - 1), 0.32), 0.94 / (n - 1), 0.36,
                               fc=color, ec="white", lw=1, alpha=0.85))
    ax.text(i * 0.94 / (n - 1) + 0.47 / (n - 1), 0.5, label, ha="center", va="center",
            fontsize=9, color="white")
    if i < n - 1:
        ax.annotate("", xy=(i * 0.94 / (n - 1) + 0.94 / (n - 1) + 0.01, 0.5),
                    xytext=(i * 0.94 / (n - 1) + 0.94 / (n - 1) - 0.04, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))
ax.text(0.5, 0.82, "一个注意力算子在 MindIR 里的“分形”表示", fontsize=13, ha="center", fontweight="bold")
ax.text(0.5, 0.14, "昇腾上常把这串算子融合成一个 FlashAttention 大算子——图级融合就是 IR 优化的第一步。",
        fontsize=10, ha="center", color="#555")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
plt.tight_layout(); plt.show()
'''), "🎨 每条边代表“前一个算子输出 → 后一个算子输入”。所谓图优化,就是在这张 DAG 上做等价变换(融合、删减、折叠)。")

NB.md("## 3️⃣ 手写迷你 IR:DAG + 解释器 🔬",
D('''
理论讲完了,动手造一个**迷你 IR**:用 JSON 描述计算图(节点带算子类型和权重,边表示依赖),
再写一个**解释器**按拓扑顺序执行它。这一套东西,正是 MindIR 运行时的最小内核。
'''))

NB.code(D('''
def make_graph():
    """用 JSON 描述一个 y = W2 @ relu(W1 @ x + b1) + b2 的迷你计算图(形如 MindIR 序列化)。"""
    return {
        "nodes": [
            {"name": "x",   "type": "input",  "shape": [4]},
            {"name": "w1",  "type": "param",  "shape": [8, 4], "data": "W1"},
            {"name": "b1",  "type": "param",  "shape": [8],    "data": "B1"},
            {"name": "matmul1", "type": "MatMul", "inputs": ["x", "w1"]},
            {"name": "add1", "type": "Add", "inputs": ["matmul1", "b1"]},
            {"name": "relu", "type": "ReLU", "inputs": ["add1"]},
            {"name": "w2",  "type": "param",  "shape": [2, 8], "data": "W2"},
            {"name": "b2",  "type": "param",  "shape": [2],    "data": "B2"},
            {"name": "matmul2", "type": "MatMul", "inputs": ["relu", "w2"]},
            {"name": "add2", "type": "Add", "inputs": ["matmul2", "b2"]},
            {"name": "out", "type": "output", "inputs": ["add2"]},
        ],
    }

graph = make_graph()
print("迷你 IR 节点数:", len(graph["nodes"]))
print("示例节点:", graph["nodes"][3]["name"], "→", graph["nodes"][3]["type"],
      "依赖:", graph["nodes"][3]["inputs"])
'''), "✅ 这个字典就是“序列化之后的 MindIR”的微型缩影:节点类型 + 输入依赖 + 形状信息。")

NB.code(D('''
def run_graph(graph, tensors):
    """迷你解释器:按依赖拓扑依次执行,类似 MindIR 运行时逐算子调度。"""
    env = {}
    for node in graph["nodes"]:
        t = node["type"]
        if t == "input":
            env[node["name"]] = tensors[node["name"]]
        elif t == "param":
            env[node["name"]] = tensors[node["data"]]
        elif t == "MatMul":
            env[node["name"]] = env[node["inputs"][0]] @ env[node["inputs"][1]].T
        elif t == "Add":
            env[node["name"]] = env[node["inputs"][0]] + env[node["inputs"][1]]
        elif t == "ReLU":
            env[node["name"]] = torch.relu(env[node["inputs"][0]])
        elif t == "output":
            env[node["name"]] = env[node["inputs"][0]]
    return env["out"]

x = torch.randn(4)
W1, B1 = torch.randn(8, 4), torch.randn(8)
W2, B2 = torch.randn(2, 8), torch.randn(2)
y = run_graph(graph, {"x": x, "W1": W1, "B1": B1, "W2": W2, "B2": B2})
y_ref = W2 @ torch.relu(W1 @ x + B1) + B2
print("迷你 IR 解释执行:", y.detach().numpy().round(3))
print("PyTorch 直接计算:", y_ref.detach().numpy().round(3))
print("结果一致:", torch.allclose(y, y_ref))
'''), "✅ 解释器按“依赖就绪就执行”的顺序跑完整个图——MindIR 的 GraphCell 运行时就是这样一层层调度算子的。")

NB.md("## 4️⃣ 转换与 ONNX 互通 🌉",
D('''
MindSpore 提供双向桥:

- **导出**:`mindspore.export(net, input_x, file_name="net.mindir", file_format="MINDIR")`;
- **ONNX**:MindSpore 支持把模型导出为 ONNX(`file_format="ONNX"`),反之也可从 ONNX 加载;
- **转换器**:端侧用 `converter_lite` 把 MindIR/ONNX 转成 `.ms`,并顺手做算子裁剪与量化(第 83 课)。

ONNX 是“最大公约数”:PyTorch 导出 ONNX → MindSpore 消费 ONNX,成为生态互通的高速公路。
下面把「导出 → 转换 → 执行」三类文件的关系画出来:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 3.8))
formats = [("PyTorch\\nsafetensors/pt", "torch 导出", "#4C78A8"),
           ("ONNX\\n.onnx", "通用桥", "#F58518"),
           ("MindSpore\\nMindIR(.mindir)", "ms.export", "#4C78A8"),
           ("Lite\\n.ms", "converter_lite", "#E45756")]
for label, mid, color in formats:
    ax.add_patch(plt.Rectangle((0.03, 0.4), 0.2, 0.26, fc="#DFE9F8", ec=color, lw=2))
    ax.text(0.13, 0.53, label, ha="center", va="center", fontsize=10)
ax.annotate("", xy=(0.72, 0.53), xytext=(0.24, 0.53), arrowprops=dict(arrowstyle="-|>", color="#555", lw=2))
ax.text(0.48, 0.56, "converter_lite\\n(转换)", ha="center", fontsize=9)
ax.annotate("", xy=(0.28, 0.53), xytext=(0.24, 0.53), arrowprops=dict(arrowstyle="-|>", color="#555", lw=2))
ax.text(0.32, 0.34, "也可从 ONNX 反向加载", fontsize=8, color="#888")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("模型文件的“换乘站”:导出 → 转换 → 端侧", fontsize=13)
plt.tight_layout(); plt.show()
'''), "🎨 格式转换的本质是“重排菜谱”,不是“重新做菜”:图信息保真,权重照搬,只在边界处做等价改写。")

NB.md("## 5️⃣ 三张菜谱横评:MindIR vs ONNX vs TorchScript ⚖️",
D('''
三种 IR 各有所长。用雷达图对比四个维度:跨框架、跨硬件、保真度(含控制流/动态形状)、工具链成熟度:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(6.6, 5.2))
dims = ["跨框架互通", "跨硬件执行", "控制流/动态\\n形状保真", "工具链成熟度"]
mindir = [35, 95, 75, 60]
onnx = [95, 85, 60, 90]
ts = [30, 55, 90, 70]
angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
angles += angles[:1]
for name, vals, color in [("MindIR", mindir, "#4C78A8"), ("ONNX", onnx, "#F58518"),
                          ("TorchScript", ts, "#E45756")]:
    v = vals + vals[:1]
    ax.plot(angles, v, marker="o", label=name, color=color, lw=2)
    ax.fill(angles, v, alpha=0.08, color=color)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims, fontsize=10)
ax.set_ylim(0, 100); ax.set_ylabel("能力(示意)")
ax.set_title("三种中间表示的取舍(示意)", fontsize=13)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.0), frameon=True, fontsize=10)
plt.tight_layout(); plt.show()
'''), "📊 ONNX 胜在“到处能用”,TorchScript 胜在“保真 Python 语义”,MindIR 胜在“为昇腾深度优化”。真实项目往往混用:训练用 TorchScript 保真,部署转 ONNX/MindIR 求通用。")

NB.md("## 6️⃣ 交互图:DAG 网络浏览 🕸️",
D('''
用 plotly 把刚才的“残差块”计算图画成可拖拽的网络图:节点 = 算子,边 = 数据依赖,悬停看类型。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

names = ["input", "conv", "bn", "skip", "relu", "output"]
kinds = ["Parameter", "Conv2D", "BatchNorm", "Add", "ReLU", "Output"]
edges = [("input", "conv"), ("conv", "bn"), ("bn", "skip"), ("input", "skip"), ("skip", "relu"), ("relu", "output")]
posx = [0, 1, 2, 2.2, 3, 4]
posy = [0, 0, 0, 0.6, 0, 0]
fig = go.Figure()
for s, t in edges:
    fig.add_trace(go.Scatter(x=[posx[names.index(s)], posx[names.index(t)]],
                             y=[posy[names.index(s)], posy[names.index(t)]],
                             mode="lines", line=dict(color="#B8C4D8", width=2), hoverinfo="none",
                             showlegend=False))
fig.add_trace(go.Scatter(x=posx, y=posy, mode="markers+text", text=names, textposition="top center",
                         marker=dict(size=34, color=["#E45756" if k == "Output" else "#4C78A8"
                                                     for k in kinds], line=dict(width=1, color="white")),
                         hovertext=[f"{n} ({k})" for n, k in zip(names, kinds)], showlegend=False))
fig.update_layout(title="ResNet 残差块在 MindIR 里的 DAG 表示", height=380,
                  xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                  margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "🕸️ 注意 `skip` 节点:一条从 input 直达的旁路边——这就是残差连接在 IR 里的样子,也是图优化器重点盯住的模式。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_82_mindir.py`,用**下拉框切换三种示例计算图**(线性 / 残差 / 注意力),
浏览节点与边的表格,并在可拖拽的 plotly 网络图中观察 DAG:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_82_mindir.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_82_mindir.py\n" + APP_82, "📜 这就是 app_82_mindir.py 的完整源码,notebook 与 app 共用同一套 DAG 描述,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "IR(中间表示)把“模型是什么(图)”与“怎么跑(硬件)”解耦,实现一次导出、多端复用",
        "MindIR = 算子节点 + 张量边 + 权重 + 子图,序列化在 .mindir 文件里",
        "MindSpore 支持导出 MindIR 与 ONNX,转换器可双向互通,端侧用 converter_lite 转 .ms",
        "手写迷你 IR(JSON DAG + 解释器)完整复现了 MindIR 运行时的最小内核",
        "ONNX 胜在通用、TorchScript 胜在保真、MindIR 胜在昇腾深度优化,三者常混用",
    ],
    practice=[
        "给迷你 IR 增加一个 softmax 节点类型,并把注意力图的 JSON 描述跑通解释器",
        "把 run_graph 改成“先拓扑排序再执行”,并验证有依赖的节点不会先于输入执行",
        "调研 ms.export 的 file_format 参数:除了 MINDIR/ONNX,还能导出什么格式?",
        "对比 torch.jit.trace 与 torch.jit.script:哪个保留的控制流信息更多?与 MindIR 的子图机制有何异同",
    ],
    links=[
        ("MindSpore MindIR 设计文档", "https://www.mindspore.cn/docs/zh-CN/r2.4/design/mindir.html"),
        ("ONNX 官网", "https://onnx.ai"),
        ("PyTorch TorchScript 文档", "https://pytorch.org/docs/stable/jit.html"),
    ])

NB.save(str(Path(CH11) / "82_mindir.ipynb"))
app_path = Path(CH11) / "app_82_mindir.py"
app_path.write_text(APP_82 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
