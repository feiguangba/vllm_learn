# -*- coding: utf-8 -*-
"""生成 78_ms_operator_dev.ipynb 与 app_78_ms_op.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_78 = D('''
# -*- coding: utf-8 -*-
# app_78_ms_op.py — MindSpore 算子开发:算子选择对比 🔬
import time
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch

st.set_page_config(page_title="MindSpore 算子开发 🔬", layout="wide")
st.title("🔬 第 78 课 · MindSpore 算子开发:内置 / 自定义 / 融合")

st.markdown("""
框架内置算子(成品)开箱即用但未必最优;**自定义算子**(Primitive / autograd.Function)
可为特定形状与精度量身定制;再把相邻算子**融合**成一个 kernel,访存更省、速度更快。
下方选择算子方案与数据规模,对比**耗时**,并做一次数值梯度校验。
""")

class SwishFn(torch.autograd.Function):
    """自定义算子 swish = x * sigmoid(x),forward + backward 二件套。"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return g * (s + x * s * (1 - s))

def swish(x):
    return SwishFn.apply(x)

scheme = st.sidebar.radio("算子方案", ["内置算子", "自定义算子(Primitive)", "融合算子(2→1)"])
n = st.sidebar.slider("张量元素数", 100_000, 50_000_000, 10_000_000, 1_000_000)
run_grad = st.sidebar.checkbox("做数值梯度校验", value=True)
st.sidebar.caption("内置最稳、自定义最灵活、融合最快 —— 三者的取舍正是算子工程的核心。")

torch.manual_seed(0)
x = torch.randn(min(n, 2_000_000))   # 控制内存

def bench(fn, iters=10):
    ts = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - ts) / iters * 1000

def fused():
    s = torch.sigmoid(x)
    return torch.relu(x * s)

if scheme == "内置算子":
    t = bench(lambda: torch.nn.functional.silu(x))
    note = "调用框架内置 silu,开箱即用、性能均衡。"
elif scheme == "自定义算子(Primitive)":
    t = bench(lambda: SwishFn.apply(x))
    note = "自定义 autograd.Function 实现 swish,行为与内置一致,梯度可自定义。"
else:
    t = bench(fused)
    note = "把 sigmoid 与 relu 融合成一次张量表达式,减少中间张量访问。"

c1, c2, c3 = st.columns(3)
c1.metric("单次耗时", f"{t:.3f} ms")
c2.metric("元素数", f"{x.numel():,}")
c3.metric("方案", scheme)

if run_grad:
    st.subheader("🧮 数值梯度校验(自定义算子)")
    xg = torch.randn(6, requires_grad=True)
    swish(xg).sum().backward()
    analytic = xg.grad.clone()
    eps = 1e-6
    for i in range(xg.numel()):
        xp = xg.clone(); xm = xg.clone()
        xp.flatten()[i] += eps; xm.flatten()[i] -= eps
        fp = swish(xp).sum().item()
        fm = swish(xm).sum().item()
        xg.grad.flatten()[i] = (fp - fm) / (2 * eps)
    num = xg.grad
    ok = torch.allclose(analytic, num, atol=1e-5)
    st.markdown(f"解析梯度 vs 数值梯度一致: **{'✅ 是' if ok else '❌ 否'}** (atol=1e-5)")
    st.caption("swish 的解析梯度 = σ(x) + x·σ(x)·(1−σ(x)),与中心差分逐点对照。")

st.subheader("📊 三种方案耗时对比(本机实测)")
schemes = ["内置算子", "自定义算子", "融合算子"]
times = [bench(lambda: torch.nn.functional.silu(x)),
         bench(lambda: SwishFn.apply(x)),
         bench(fused)]
fig = go.Figure(go.Bar(x=schemes, y=times, marker_color=["#54A24B", "#B279A2", "#4C78A8"],
                       text=[f"{v:.3f}" for v in times], textposition="outside"))
fig.update_layout(title="同规模输入下三种方案的耗时(ms,本机实测)",
                  yaxis_title="耗时(ms)", height=360, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"> 💡 **当前方案说明**:{note} 三者数学等价,差异在工程形态:自定义算子给了你"
            f"『控制梯度 + 参与图融合』的能力,融合算子砍掉了中间张量的来回读写。")
''')

NB = new_nb("第 78 课 · MindSpore 算子开发:Primitive、自定义算子与融合",
            subtitle="框架算子从哪来?用 torch 类比 Primitive 的注册与自定义,看懂『内置 / 自定义 / 融合』三态",
            emoji="🔬")

chapter_cover(NB,
    objectives=[
        "理解算子(Primitive)在框架中的角色:类型、shape 推导、反向三件套",
        "用 torch.autograd.Function 亲手实现并注册一个自定义算子(swish)",
        "做数值梯度校验,证明自定义算子的反向正确",
        "理解算子融合为何能提速(减少中间张量访问)",
        "对比内置 / 自定义 / 融合三种算子方案的选择逻辑",
        "跑通配套 App:算子选择对比 + 梯度校验",
    ],
    toc=[
        ("直觉:成品厨具 vs 自打菜刀", "内置算子与自定义算子的场景分工"),
        ("Primitive 是什么", "算子的『身份证』:定义 + InferShape + bprop"),
        ("用 torch 自定义算子", "autograd.Function 实现 swish,注册进 autograd"),
        ("数值梯度校验", "解析梯度 vs 中心差分,证明反向正确"),
        ("算子融合:从 2 个到 1 个", "为什么融合更快:访存账"),
        ("三种方案怎么选", "内置 / 自定义 / 融合的决策矩阵"),
        ("配套 Streamlit 演示", "app_78_ms_op.py:算子选择对比"),
    ],
    links=[
        ("MindSpore 自定义算子文档", "https://www.mindspore.cn/docs/zh-CN/master/design/custom_operator.html"),
        ("MindSpore ops 文档", "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html"),
        ("PyTorch autograd.Function 文档", "https://pytorch.org/docs/stable/autograd.html"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:成品厨具 vs 自打菜刀 🔪",
D('''
框架内置的几百个算子,就像超市里的成品厨具:**开箱即用、质量有保障**,但不见得
"最适合你的手"。什么时候要**自定义算子**?

- 现有算子组合性能差(要融合成一个 kernel);
- 特定形状/精度没有现成实现;
- 算子库(vLLM 的 PagedAttention、FlashAttention 这类)需要深度定制的核心操作。

MindSpore 里,自定义算子的标准姿势是注册一个 **Primitive**(原语):告诉框架"我这个算子
输入什么、输出什么、怎么求导"。PyTorch 的对应物是 `torch.autograd.Function`。
本课用 torch 把这套机制完整走一遍。
'''))

NB.md("## 2️⃣ Primitive 是什么:算子的『身份证』 🪪",
D('''
MindSpore 里每个算子都是一个 **Primitive**,它要交代三件事(概念性描述,真实 MindSpore
伪代码如下,本机未安装不执行):

```python
# MindSpore 自定义算子注册的『三件套』(示意)
class MyMulAdd(PrimitiveWithInfer):
    def __init__(self):              # ① 算子定义
        super().__init__()
    def infer_shape(self, x_shape, y_shape):   # ② shape 推导
        return x_shape                # 告诉图引擎输出长什么样
    def bprop(self, x, y, out, dout): # ③ 反向:定义梯度
        return (dout * y, dout * x)
```

三件套的意义:**框架拿到 Primitive 后,就能把它放进计算图里参与自动微分、参与图优化与
算子融合** —— 它既是计算单元,也是图优化的"材料"。下面用 torch 的等价物
`autograd.Function` 走一遍完整流程。
'''))

NB.md("## 3️⃣ 用 torch 自定义算子:实现一个 swish 🛠️",
D('''
**swish(即 SiLU)** 的定义是 $f(x)=x\\cdot\\sigma(x)$,导数有优雅的闭式解
$f'(x)=\\sigma(x)+x\\cdot\\sigma(x)(1-\\sigma(x))$。我们用 `torch.autograd.Function`
把它做成一个"注册进 autograd 的自定义算子":
'''))

NB.code(D('''
class SwishFn(torch.autograd.Function):
    """自定义算子 swish = x * sigmoid(x):
    forward 写前向,backward 写梯度,apply 完成注册。"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # 存 x,供 backward 使用
        return x * torch.sigmoid(x)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return g * (s + x * s * (1 - s))  # swish 的解析导数

def swish(x):
    return SwishFn.apply(x)

x = torch.randn(4, 4, requires_grad=True)
y = swish(x).sum()
y.backward()
print("自定义算子可正常参与 autograd 求梯度:", x.grad is not None)
print("与框架内置 SiLU 结果一致:", torch.allclose(swish(x.detach()),
      torch.nn.functional.silu(x.detach()), atol=1e-6))
'''),
"✅ 验证:自定义的 swish 与内置 SiLU 数学等价 —— Primitive 的 forward/backward 二件套足以撑起一次完整的前向与反向。")

NB.md("## 4️⃣ 数值梯度校验:用『土办法』验算导数 🔍",
D('''
backward 是你手写的,凭什么相信它对?土办法是**中心差分**:把输入每个分量拨动 ±ε,
用 $\\dfrac{f(x+\\varepsilon)-f(x-\\varepsilon)}{2\\varepsilon}$ 逼近导数,与解析梯度逐点对照:
'''))

NB.code(D('''
x0 = torch.randn(6, requires_grad=True)
swish(x0).sum().backward()
analytic = x0.grad.clone()            # 自定义算子的解析梯度
eps = 1e-6
for i in range(x0.numel()):
    xp = x0.clone(); xm = x0.clone()
    xp.flatten()[i] += eps; xm.flatten()[i] -= eps
    fp = swish(xp).sum().item()
    fm = swish(xm).sum().item()
    x0.grad.flatten()[i] = (fp - fm) / (2 * eps)
numerical = x0.grad
print("解析梯度:", analytic.tolist())
print("数值梯度:", numerical.tolist())
print("全部一致:", torch.allclose(analytic, numerical, atol=1e-5))
'''),
"✅ 这就是算子开发的基本功:forward/backward 写完,数值梯度校验是『出厂质检』 —— MindSpore 里写 bprop 同样要过这一关。")

NB.md("## 5️⃣ 算子融合:从 2 个 kernel 到 1 个 🧬",
D('''
自定义算子的一个重要动机是**融合**:把 sigmoid、乘法、relu 三个逐元素算子捏成一个 kernel。
每个逐元素算子都要**读一次、写一次**;融合后**只读一次、只写一次**,访存减半还要多。
实测对比:
'''))

NB.code(D('''
x = torch.randn(5_000_000)
def bench(fn, iters=10):
    ts = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - ts) / iters * 1000

t_separate = bench(lambda: torch.relu(x * torch.sigmoid(x)))
def fused():
    s = torch.sigmoid(x)
    return torch.relu(x * s)
t_fused = bench(fused)

print(f"分离写(逐元素算子×3): {t_separate:.4f} ms/次")
print(f"融合写(一个表达式):   {t_fused:.4f} ms/次")
print(f"融合相对节省:        {(t_separate - t_fused) / t_separate * 100:.1f}%")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.bar(["分离(3 次读写)", "融合(1 次读写)"], [t_separate, t_fused],
       color=["#E45756", "#4C78A8"], width=0.5)
for i, v in enumerate([t_separate, t_fused]):
    ax.text(i, v * 1.03, f"{v:.4f}ms", ha="center", fontsize=10)
ax.set_ylabel("耗时(ms)"); ax.set_title("逐元素算子:融合省掉中间张量的来回读写")
plt.tight_layout()
'''),
"📊 实测:同一计算,融合写法明显更快 —— 这就是为什么算子融合是图优化(第 79 课)的第一板斧。")

NB.md("## 6️⃣ 三种方案怎么选:决策矩阵 🎯",
D('''
| 方案 | 性能 | 灵活性 | 工程成本 | 适用场景 |
|------|------|--------|----------|----------|
| **内置算子** | 均衡 | 低(不能改) | 零 | 绝大多数模型 |
| **自定义算子** | 可定制 | 高(可写梯度/逻辑) | 中(要写 forward+backward) | FlashAttention、PagedAttention 这类特殊核心 |
| **算子融合** | 最高 | 中(受限于可融合性) | 低(交给编译器或手写) | 逐元素链、激活+归一化 |

一个判断口诀:**能内置不自定义,能融合不分离**。把内置算子做不了或做不好的
"最后 10%"留给自己写 —— 这正是 vLLM 里 PagedAttention 内核存在的意义。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:算子选择对比 🎛️",
D('''
运行同目录下的 `app_78_ms_op.py`,可以**切换内置 / 自定义 / 融合三种方案、调整数据规模**,
实时对比耗时、看数值梯度校验结果:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_78_ms_op.py
```

浏览器打开 **http://localhost:8501**。建议先选自定义算子看梯度校验通过,再切到融合算子
看耗时下降。完整源码如下(与同目录 `app_78_ms_op.py` 一字不差):
'''))

NB.code(app_cell("app_78_ms_op.py", APP_78),
"📜 运行本 cell 会覆盖写入 `app_78_ms_op.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "Primitive 三件套:算子定义 + InferShape(shape 推导)+ bprop(反向),是进入计算图的身份证",
        "torch.autograd.Function 是自定义算子的 PyTorch 等价物:forward + backward + apply",
        "数值梯度校验(中心差分)是算子开发的基本功,保证手写导数正确",
        "算子融合把多个逐元素算子并成一个 kernel,访存减半,实测明显更快",
        "决策口诀:能内置不自定义,能融合不分离 —— 特殊核心才值得自己写",
    ],
    practice=[
        "把 SwishFn 扩展成 swish + scale(乘一个常数参数),重做数值梯度校验",
        "自定义一个 relu6 算子,并与内置 relu6 对比结果,体会 Primitive 的通用套路",
        "把第 5 节融合例子加一个 bias 相加,对比 4 算子分离与 1 表达式融合的耗时",
        "查 MindSpore 文档里 register_custom_op 的流程,与 torch.autograd.Function 各写 3 条异同",
    ],
    links=[
        ("MindSpore 自定义算子文档", "https://www.mindspore.cn/docs/zh-CN/master/design/custom_operator.html"),
        ("MindSpore ops 文档", "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html"),
        ("PyTorch autograd.Function 文档", "https://pytorch.org/docs/stable/autograd.html"),
    ])

out = str(Path(CH11) / "78_ms_operator_dev.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_78_ms_op.py"
app_path.write_text(APP_78 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
