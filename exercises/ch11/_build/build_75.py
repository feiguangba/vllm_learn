# -*- coding: utf-8 -*-
"""生成 75_mindspore_tensor.ipynb 与 app_75_mindspore.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_75 = D('''
# -*- coding: utf-8 -*-
# app_75_mindspore.py — MindSpore 张量运算交互 🧮
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch

st.set_page_config(page_title="MindSpore 张量 🧮", layout="wide")
st.title("🧮 第 75 课 · MindSpore 基础:张量运算交互")

st.markdown("""
MindSpore(昇思)的编程模型与 PyTorch 高度神似:**张量(Tensor)+ 算子(ops)+ 网络(nn)+
自动微分(autograd)**。本演示用 **torch 模拟 MindSpore 的 API 设计理念**:
调整张量的形状、精度与运算,实时看结果统计、内存占用与自动微分示例。
""")

shape_r = st.sidebar.slider("行数", 1, 16, 4, 1)
shape_c = st.sidebar.slider("列数", 1, 16, 6, 1)
dtype = st.sidebar.selectbox("dtype", ["fp32", "fp16", "bf16"])
op = st.sidebar.selectbox("运算", ["square 平方", "exp 指数", "relu", "matmul × 转置"])
show_grad = st.sidebar.checkbox("展示自动微分示例", value=True)
st.sidebar.caption("mindspore.Tensor 与 torch.tensor 概念一一对应,API 可查官方映射表。")

shape = (shape_r, shape_c)
torch.manual_seed(0)
x = torch.randn(*shape)
bytes_el = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
xt = {"fp32": x.float(), "fp16": x.half(), "bf16": x.bfloat16()}[dtype]

out = {"square 平方": xt ** 2, "exp 指数": torch.exp(xt.float()),
       "relu": torch.relu(xt), "matmul × 转置": xt @ xt.t()}[op]

c1, c2, c3 = st.columns(3)
c1.metric("shape", f"{list(shape)}")
c2.metric("dtype", dtype)
c3.metric("内存占用", f"{shape_r * shape_c * bytes_el / 1024:.2f} KB")

st.subheader("📊 输入张量视图(热力图)")
data = xt.float().cpu().numpy()
fig = go.Figure(data=go.Heatmap(z=data, colorscale="YlGnBu",
                                text=np.round(data, 2), texttemplate="%{text}",
                                colorbar_title="数值"))
fig.update_layout(title=f"输入张量 {shape} · {dtype}", height=300,
                  yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"🔧 运算结果: {op}")
ov = out.float().cpu().numpy().ravel()
c1, c2, c3, c4 = st.columns(4)
c1.metric("min", f"{ov.min():.4f}")
c2.metric("max", f"{ov.max():.4f}")
c3.metric("mean", f"{ov.mean():.4f}")
c4.metric("std", f"{ov.std():.4f}")

fig2 = go.Figure(go.Bar(x=[f"r{i}" for i in range(shape_r)],
                        y=out.float().abs().sum(dim=1).cpu().numpy(),
                        marker_color="#4C78A8", textposition="outside"))
fig2.update_layout(title="每行 L1 范数(绝对值求和)", height=300,
                   xaxis_title="行", yaxis_title="Σ|值|", margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 观察:改变 shape / dtype / 运算,热力图与行范数实时变化 —— 张量是这一切的公共语言。")

if show_grad:
    st.subheader("🧮 自动微分示例(模拟 mindspore.GradOperation)")
    xg = torch.tensor([[2.0, 3.0]], requires_grad=True)
    y = (xg ** 2).sum() + 3 * xg.sum() + 1
    y.backward()
    st.markdown(f"对 **f(x)=x²+3x+1** 在 x=[2,3] 处求梯度:dy/dx = **{xg.grad.tolist()}**(解析式 2x+3 的精确值)")
    st.markdown("MindSpore 里写法是 `mindspore.GradOperation()(net, inputs)` → 返回同样的梯度向量。")
    st.caption("grad 是逐个元素求导:∂f/∂x = 2x+3,在 x=2 时为 7,在 x=3 时为 9。")

st.markdown("""
> 💡 **一句话**:MindSpore 四大件(Tensor / ops / nn / GradOperation)与 PyTorch 同构,
> 区别主要在『图模式编译』与『昇腾下沉』(第 76-77 课)。会用 torch,就能很快上手 MindSpore。
""")
''')

NB = new_nb("第 75 课 · MindSpore 基础:张量、自动微分与计算图思维",
            subtitle="Tensor / ops / nn / GradOperation 四大件 —— 用 torch 类比 MindSpore 的 API 设计理念",
            emoji="🧮")

chapter_cover(NB,
    objectives=[
        "认识 MindSpore 的定位:华为自研、全场景、与昇腾深度绑定的 AI 框架",
        "掌握四大件:Tensor、ops、nn、自动微分(GradOperation)",
        "用 torch 类比 mindspore 的 API 设计,跑通张量运算与梯度计算",
        "画出计算图 DAG:张量运算如何构成一张图",
        "看懂官方『PyTorch 与 MindSpore API 映射表』的价值",
        "跑通配套 App:张量运算交互 + 自动微分示例",
    ],
    toc=[
        ("直觉:换个马甲的同一套功夫", "Tensor / ops / nn / Grad 四大件对照"),
        ("Tensor 基础", "创建、dtype、shape、与 numpy 互转"),
        ("自动微分", "mindspore.GradOperation vs torch autograd"),
        ("计算图思维", "把张量运算画成 DAG(matplotlib)"),
        ("官方 API 映射表", "PyTorch ↔ MindSpore 速查表"),
        ("配套 Streamlit 演示", "app_75_mindspore.py:张量运算交互"),
    ],
    links=[
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
        ("MindSpore API 文档", "https://www.mindspore.cn/docs/zh-CN/master/index.html"),
        ("PyTorch 与 MindSpore API 映射表", "https://www.mindspore.cn/docs/zh-CN/master/mapping/pytorch_api_mapping.html"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头(本机未装 MindSpore,用 torch 严格类比)。")

NB.md("## 1️⃣ 直觉:换个马甲的同一套功夫 🥋",
D('''
如果你会用 PyTorch,那 MindSpore 几乎是"换了名字的熟面孔"。看四大件的对照:

| 概念 | PyTorch | MindSpore | 本课演示 |
|------|---------|-----------|----------|
| 张量 | `torch.tensor` | `mindspore.Tensor` | 用 torch 跑 |
| 算子 | `torch.add / F.relu` | `mindspore.ops.Add / ReLU` | 用 torch 跑 |
| 网络层 | `torch.nn.Linear` | `mindspore.nn.Dense` | 用 torch 跑 |
| 自动微分 | `autograd` / `.backward()` | `mindspore.GradOperation` | 用 torch 模拟 |

MindSpore 官方甚至维护了一张 **PyTorch 与 MindSpore API 映射表**(见本课参考链接),
从张量到优化器逐项列全 —— 这本身就说明:两家框架在**编程体验上刻意对齐**。
那么 MindSpore 到底多了什么?答案是两点:**图模式编译**(第 76 课)和**昇腾整图下沉**。
先用 torch 把基础四大件跑一遍。
'''))

NB.md("## 2️⃣ Tensor 基础:框架的通用语言 🔤",
D('''
张量(Tensor)是任何深度学习框架的"通用货币"。MindSpore 的张量与 PyTorch 一样:
有 shape、dtype、设备,能广播,能与 numpy 互转。区别只在于**默认设备与内存布局**:
MindSpore 张量默认就是昇腾(NPU)或 GPU 设备张量,而 PyTorch 默认在 CPU。
'''))

NB.code(D('''
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones(2, 2)
print("创建与运算:")
print("  a + b =\\n", a + b)          # mindspore.ops.Add 同义
print("  a @ b =\\n", a @ b)          # mindspore.ops.MatMul 同义

print("\\ndtype / shape 体系:")
for t in [a, a.half(), a.bfloat16()]:
    print(f"  {str(t.dtype):<10} shape={list(t.shape)}")

print("\\n与 numpy 互转:")
arr = a.numpy()
print("  torch → numpy →", type(arr).__name__, arr.tolist())
'''),
"🔍 注意:MindSpore 里这些 API 名几乎一致(mindspore.ops.Add / MatMul / cast),只是包名不同 —— 迁移成本极低。")

NB.md("## 3️⃣ 自动微分:框架教你求导 📐",
D('''
**自动微分(autograd)** 是框架的杀手锏:你只写**前向**(forward),框架自动按计算图
做**反向**(backward)求梯度。MindSpore 的入口是 `mindspore.GradOperation()`:
给定网络与输入,返回一个"能算梯度"的函数;PyTorch 则是 `backward()` + `.grad`。

两者数学本质完全一样:**链式法则 + 计算图反向传播**。下面用 torch 验证一个二次函数的
梯度,并与解析解对照:
'''))

NB.code(D('''
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2 + 3 * x + 1).sum()       # f(x)=x²+3x+1, 对向量 x 求和
y.backward()
grad = x.grad
print("x          =", x.tolist())
print("dy/dx(框架) =", grad.tolist())
print("解析解 2x+3  =", (2 * x).tolist())      # d(x²+3x+1)/dx = 2x+3
print("一致?", torch.allclose(grad, 2 * x))
'''),
"✅ 验证:框架自动算出的梯度与解析解 2x+3 完全一致 —— 求导这件事,框架替你做了。")

NB.md("## 4️⃣ 计算图思维:把张量运算画成 DAG 🌳",
D('''
自动微分之所以可能,是因为框架默默记住了一笔"账":**计算图** —— 每个节点是一次运算,
每条边是数据依赖。反向传播就是沿着边从输出走回输入。把上面的 f(x)=x²+3x+1 画出来:
'''))

NB.code(D('''
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axis("off")
nodes = [
    (0.3, 1.0, "输入 x", "#d6eaf8", "#1f4e79"),
    (3.3, 1.0, "square 平方", "#e8daef", "#5b2c8f"),
    (3.3, 0.0, "linear 3x+1", "#e8daef", "#5b2c8f"),
    (6.3, 0.5, "sum 求和", "#fce4d6", "#c55a11"),
    (8.7, 0.5, "输出 y", "#d5f5e3", "#1e8449"),
]
for nx, ny, label, fc, ec in nodes:
    ax.add_patch(plt.Rectangle((nx, ny), 1.6, 0.85, facecolor=fc, edgecolor=ec, lw=1.8))
    ax.text(nx + 0.8, ny + 0.42, label, ha="center", va="center", fontsize=10, fontweight="bold", color=ec)
ax.annotate("", xy=(3.3, 1.42), xytext=(1.9, 1.42), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.annotate("", xy=(3.3, 0.42), xytext=(1.9, 0.42), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.annotate("", xy=(6.3, 0.92), xytext=(4.9, 1.42), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.annotate("", xy=(6.3, 0.92), xytext=(4.9, 0.42), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.annotate("", xy=(8.7, 0.92), xytext=(7.9, 0.92), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.annotate("", xy=(7.4, 0.6), xytext=(8.0, 0.6),
            arrowprops=dict(arrowstyle="->", lw=2, ls="--", color="#E45756"))
ax.text(7.6, 0.72, "backward", fontsize=8, color="#E45756")
ax.set_xlim(0, 10.5); ax.set_ylim(-0.35, 2.0)
ax.set_title("f(x) = Σ(x² + 3x + 1) 的计算图:前向画边,反向求导", fontsize=13)
plt.tight_layout()
'''),
"🎨 计算图 DAG:黑色实线是前向数据流,红色虚线是反向梯度流 —— 静态图模式(下一课)正是把这张图拿去编译。")

NB.md("## 5️⃣ 官方 API 映射表:PyTorch ↔ MindSpore 速查 🗺️",
D('''
MindSpore 官方维护的映射表(见链接)是迁移时的救命稻草。摘几条最常见的对照:

| 功能 | PyTorch | MindSpore |
|------|---------|-----------|
| 张量 | `torch.tensor(shape)` | `mindspore.Tensor(shape)` |
| 全连接 | `torch.nn.Linear` | `mindspore.nn.Dense` |
| ReLU | `torch.nn.functional.relu` | `mindspore.ops.ReLU` |
| 梯度 | `x.grad` / `torch.autograd.grad` | `mindspore.GradOperation()` |
| 模型保存 | `torch.save(state_dict)` | `mindspore.save_checkpoint` |
| 数据加载 | `torch.utils.data.DataLoader` | `mindspore.dataset` |

记住一句话:**语法对得上,语义对得上,但『图模式 + 昇腾下沉』是 MindSpore 的独门功夫**。
所以迁移时的核心功课不是改 API,而是理解它对"计算图"的态度 —— 这正是 76/77 两课的内容。
'''))

NB.md("## 6️⃣ 配套 Streamlit 演示:张量运算交互 🎛️",
D('''
运行同目录下的 `app_75_mindspore.py`,可以**调整张量 shape / dtype / 运算**,实时看
热力图、行范数与自动微分示例:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_75_mindspore.py
```

浏览器打开 **http://localhost:8501**。建议把 dtype 从 fp32 切到 bf16,观察内存占用减半;
再把运算切到 exp 与 matmul,体会不同算子对结果分布的影响。完整源码如下
(与同目录 `app_75_mindspore.py` 一字不差):
'''))

NB.code(app_cell("app_75_mindspore.py", APP_75),
"📜 运行本 cell 会覆盖写入 `app_75_mindspore.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "MindSpore 四大件与 PyTorch 同构:Tensor / ops / nn / GradOperation",
        "自动微分 = 计算图 + 链式法则,框架替你求导;GradOperation 与 autograd 等价",
        "计算图 DAG 是理解框架的钥匙:前向是数据流,反向是梯度流",
        "官方『PyTorch 与 MindSpore API 映射表』让迁移成本接近『换包名』",
        "MindSpore 的独门功夫在『图模式编译 + 昇腾整图下沉』,下一课展开",
    ],
    practice=[
        "用 torch 手写三层线性网络 + loss,并打印每层权重的梯度,体会 autograd 的自动性",
        "把第 4 节的计算图改成 f=sin(x)+x²,重画 DAG 并标出反向路径上的每一步导函数",
        "在映射表里补充 5 条你熟悉的 API(MaxPool、CrossEntropyLoss、Adam、cat、stack)",
        "对比 mindspore.Tensor 与 torch.tensor 的默认设备,说说 MindSpore 为什么默认 NPU",
    ],
    links=[
        ("MindSpore 官方文档", "https://www.mindspore.cn"),
        ("MindSpore API 文档", "https://www.mindspore.cn/docs/zh-CN/master/index.html"),
        ("PyTorch 与 MindSpore API 映射表", "https://www.mindspore.cn/docs/zh-CN/master/mapping/pytorch_api_mapping.html"),
    ])

out = str(Path(CH11) / "75_mindspore_tensor.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_75_mindspore.py"
app_path.write_text(APP_75 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
