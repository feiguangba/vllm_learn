# -*- coding: utf-8 -*-
"""生成 91_ascend_llama.ipynb 与 app_91_ascend_llama.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_91 = D('''
# -*- coding: utf-8 -*-
# app_91_ascend_llama.py — 昇腾上跑 LLaMA:迁移步骤交互助手 🦙
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="昇腾上跑 LLaMA 🦙", layout="wide")
st.title("🦙 第 91 课 · 昇腾上跑 LLaMA:模型迁移交互助手")

st.markdown("""
把一个 PyTorch 模型"搬"到昇腾 NPU,就像**搬家**:锅碗瓢盆(算子)得看看新灶台
(昇腾 AI Core)能不能用、布局(内存)得重新规划、火候(精度)得重新校准。
下面按"迁移五步"逐步点开,并拖动精度滑杆,直观感受每一步在做什么。
""")

ROUTES = {
    "torch_npu 适配器": "PyTorch 代码几乎不改,靠 torch_npu 把算子映射到 CANN,最快但算子覆盖受限",
    "MindSpore 重写": "用 MindSpore 重写网络与训练/推理脚本,最彻底,配合图模式性能最佳",
    "vLLM-Ascend 插件": "社区维护的 vLLM 硬件插件,开箱跑 LLM 推理(PagedAttention/连续批处理开箱即用)",
}
STEPS = ["环境搭建", "代码迁移", "权重转换", "算子适配", "精度对齐"]

st.sidebar.header("🎛️ 参数")
route = st.sidebar.selectbox("迁移路线", list(ROUTES.keys()))
step = st.sidebar.radio("当前阶段", STEPS, index=0)
bits = st.sidebar.slider("目标精度(FP16→INT8)", 8, 32, 16, 1)
show_map = st.sidebar.checkbox("展示权重映射表示例", value=True)
st.sidebar.caption("精度越低,余弦相似度损失越大 —— 这就是「精度对齐」要盯的东西。")

# ---------------- 迁移流水线结构图 ----------------
st.subheader("🧭 迁移五步流水线")
st.markdown(f"**当前路线: {route}** —— {ROUTES[route]}")
boxes = STEPS
fig = go.Figure()
for i, name in enumerate(boxes):
    hl = (name == step)
    fig.add_shape(type="rect", x0=i-0.38, x1=i+0.38, y0=0.4, y1=1.0,
                  line=dict(color="#2c3e50", width=1.5),
                  fillcolor="#e74c3c" if hl else "#d5dbdb")
    fig.add_annotation(x=i, y=0.7, text=name, showarrow=False, font=dict(size=11))
    if i < len(boxes)-1:
        fig.add_annotation(x=i+0.5, y=0.7, text="→", showarrow=False, font=dict(size=15))
fig.update_xaxes(range=[-0.6, len(boxes)-0.4], showticklabels=False)
fig.update_yaxes(range=[0, 1.25], showticklabels=False)
fig.update_layout(title="PyTorch 模型 → 昇腾 NPU 的迁移旅程", height=280,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------- 当前阶段讲解 ----------------
STEPS_INFO = {
    "环境搭建": "安装 CANN + torch_npu / MindSpore,配置 npu-smi 可见 NPU;类比安装 CUDA + cuDNN。",
    "代码迁移": "替换张量/设备/算子 API:model.to('npu')、cuda()→npu();MindSpore 则重写 Cell。",
    "权重转换": "把 .pth/.safetensors 权重按名称映射表转成 .ckpt 或 MindIE 能读的格式。",
    "算子适配": "查算子支持矩阵:不支持的算子用 Ascend C 手写或改写为等价组合。",
    "精度对齐": "对比迁移前后逐层输出:余弦相似度 + 最大绝对误差,逼近 1e-3 量级才算对齐。",
}
st.info(f"**{step}**: {STEPS_INFO[step]}")

# ---------------- 精度对齐模拟 ----------------
sim = np.clip(1.0 - (32 - bits) * 0.018 - 0.002, 0.3, 1.0)      # bits 越低相似度越低
err = 10 ** (-(sim * 7 + 0.5))                                   # 误差随相似度指数下降
c1, c2, c3, c4 = st.columns(4)
c1.metric("迁移目标精度", f"FP{bits}" if bits == 32 else f"INT{bits}")
c2.metric("余弦相似度", f"{sim:.4f}")
c3.metric("最大绝对误差(示意)", f"{err:.2e}")
c4.metric("待手写算子数(示意)", int(max(0, 8 - (bits - 8) * 1.5)))

st.subheader("📈 精度 vs 位宽:对齐曲线的直觉")
b = np.arange(8, 33)
sims = np.clip(1.0 - (32 - b) * 0.018 - 0.002, 0.3, 1.0)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=b, y=sims, mode="lines+markers",
                          line=dict(color="#c0392b", width=3), name="余弦相似度"))
fig2.add_vline(x=bits, line_dash="dash", line_color="#888")
fig2.update_layout(title="量化位数 → 输出相似度(模拟曲线,真实值因模型/数据而异)",
                   xaxis_title="位宽(bits)", yaxis_title="余弦相似度",
                   height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

if show_map:
    st.subheader("🔤 权重映射表示例(PyTorch → MindSpore)")
    rows = [
        ["transformer.wte.weight", "backbone.embedding.weight"],
        ["transformer.h.0.attn.c_attn.weight", "backbone.layers.0.attention.qkv.weight"],
        ["transformer.h.0.mlp.c_fc.weight", "backbone.layers.0.mlp.dense1.weight"],
        ["lm_head.weight", "head.weight"],
    ]
    st.dataframe({"PyTorch 名": [r[0] for r in rows],
                  "MindSpore 名": [r[1] for r in rows]},
                 use_container_width=True)
    st.caption("权重转换的本质:按映射表把 state_dict 的 key 改名 + 重新分块,张量数值原样搬移。")

st.markdown("""
> 💡 **结论**:迁移不是"一行代码的事",而是一条五步流水线。用 torch_npu 最快、
> 用 MindSpore 最彻底、用 vLLM-Ascend 最省心 —— 生产推理优先考虑第三条路。
""")
''')

NB = new_nb("第 91 课 · 昇腾上跑 LLaMA:从 PyTorch 到昇腾的模型迁移",
            subtitle="torch_npu / MindSpore / vLLM-Ascend 三条路线 · 迁移四步走 · 精度对齐与算子适配",
            emoji="🦙")

chapter_cover(NB,
    objectives=[
        "理解昇腾 NPU 生态全景:CANN / torch_npu / MindSpore / MindIE 各扮演什么角色",
        "掌握模型迁移的三条路线:torch_npu 适配器、MindSpore 重写、vLLM-Ascend 插件",
        "跑通迁移四步走:环境搭建 → 代码迁移 → 权重转换 → 精度对齐",
        "学会用余弦相似度与最大绝对误差做迁移前后的精度对齐",
        "认识算子适配:如何发现不支持的算子并用 Ascend C 补齐",
        "用 torch 模拟整个迁移流程,并用 matplotlib / plotly 可视化",
    ],
    toc=[
        ("直觉:搬家与新灶台", "模型上昇腾 = 把整套餐具搬到新厨房"),
        ("生态地图:一条软硬件链路", "CANN → torch_npu/MindSpore → MindIE/vLLM-Ascend"),
        ("三条迁移路线怎么选", "适配器 / 重写 / 插件,三者的成本与收益"),
        ("迁移四步走(模拟)", "环境 → 代码 → 权重 → 精度,亲自动手"),
        ("精度对齐与算子适配", "余弦相似度、最大绝对误差、Ascend C 补算子"),
        ("配套 App", "app_91_ascend_llama.py:迁移步骤交互助手"),
    ],
    links=[
        ("昇腾社区官网", "https://www.hiascend.com"),
        ("MindSpore 官网", "https://www.mindspore.cn"),
        ("vLLM-Ascend 项目", "https://github.com/vllm-project/vllm-ascend"),
        ("CANN 文档", "https://www.hiascend.com/document"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头(后续所有图都走 matplotlib + seaborn)。")

NB.md("## 1️⃣ 直觉:搬家与新灶台 🏠",
D('''
在 GPU 上跑得好好的 LLaMA,想搬到昇腾 NPU 上,听起来像"改几行代码"的事,实则是一次**搬家**:

- **硬件不一样了**:昇腾 AI Core 有 Cube(矩阵)/Vector(矢量)/Scalar(标量)三种计算单元,
  和 GPU 的 SM 架构完全不同 —— 原本为 CUDA 写的算子不是天然就能跑;
- **软件栈不一样了**:GPU 靠 CUDA + cuDNN + NCCL,昇腾靠 CANN + Ascend C + HCCL;
- **框架不一样了**:PyTorch 上 `model.to("cuda")` 很顺,昇腾上要靠 `torch_npu` 的
  `model.to("npu")`,或者干脆用华为的 **MindSpore** 框架重写。

所以"迁移"其实是一整套流程:换环境、换代码、换权重格式、换算子、最后还要验证"搬完还是不是原来那个味"
(精度对齐)。本课把这条流程走一遍 —— 本机没有昇腾硬件,我们用 torch 模拟每一步,但术语全部真实。
'''))

NB.md("## 2️⃣ 生态地图:一条软硬件链路 🗺️",
D('''
昇腾生态是一个分层的"软件栈",从硬件往上依次是:

1. **硬件层**:昇腾 AI 处理器(310P / 910B / 910C 等),对应 Atlas 系列服务器;
2. **驱动与运行时层**:**CANN(Compute Architecture for Neural Networks)** —— 华为版 CUDA,
   包含驱动、Runtime、图引擎(GE)、算子库、HCCL 集合通信库、msprof 性能分析等;
3. **框架层**:两个入口 —— ① **torch_npu**:把 PyTorch 接到 CANN 的适配器
   (`import torch_npu; model.to("npu")`);② **MindSpore**:华为自研 AI 框架,
   原生对接昇腾,支持 PYNATIVE_MODE(动态图)与 GRAPH_MODE(静态图)两种模式;
4. **推理引擎层**:**MindIE(Mind Inference Engine)** 是华为的 LLM 推理引擎(连续批处理、
   PagedAttention),而 **vLLM-Ascend** 则是 vLLM 社区在昇腾上的官方硬件插件。

一句话记住:**CANN 是地基, torch_npu / MindSpore 是门, MindIE / vLLM-Ascend 是让 LLaMA
真正跑起来的厨房**。下面画一张全景图:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axis("off")
layers = [
    ("推理引擎层", ["vLLM-Ascend 插件", "MindIE 推理引擎"], "#fdebd0", "#a64d17"),
    ("框架层", ["torch_npu(适配器)", "MindSpore 框架"], "#e2d5f1", "#5b2c8f"),
    ("软件栈层(CANN)", ["Runtime / 图引擎 GE / 算子库", "Ascend C · HCCL · msprof"], "#d9ead3", "#38761d"),
    ("硬件层", ["昇腾 AI Core: Cube + Vector + Scalar", "310P / 910B / 910C · Atlas 服务器"], "#dbe7f4", "#1f4e79"),
]
for i, (name, subs, fill, edge) in enumerate(layers):
    ax.add_patch(plt.Rectangle((0.4, 4.6 - i * 1.3), 5.6, 1.0,
                               facecolor=fill, edgecolor=edge, lw=2))
    ax.text(0.7, 4.95 - i * 1.3, name, fontsize=12, fontweight="bold", color=edge, va="center")
    ax.text(3.6, 4.95 - i * 1.3, " | ".join(subs), fontsize=9, color="#333", ha="center", va="center")
    if i < len(layers) - 1:
        ax.annotate("", xy=(3.2, 4.35 - i * 1.3), xytext=(3.2, 4.6 - i * 1.3),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.text(3.2, 5.35, "昇腾生态全景:LLaMA 从 PyTorch 到昇腾 NPU 的完整软件链路",
        fontsize=14, fontweight="bold", ha="center")
ax.set_xlim(0, 6.4); ax.set_ylim(0, 6.1)
plt.tight_layout()
'''),
"🎨 全景结构图:从推理引擎到底层 AI Core,一条链路四层。本课的核心就是『如何在这条链路上把模型跑起来』。")

NB.md("## 3️⃣ 三条迁移路线怎么选 🧭",
D('''
把手头的 PyTorch LLaMA 送上昇腾,有三条典型路线:

| 路线 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **① torch_npu 适配器** | 代码几乎不改,`to("npu")` 即可 | 最快上手,生态兼容 | 算子覆盖与性能受限于适配层 |
| **② MindSpore 重写** | 用 MindSpore API 重写网络与训练/推理 | 原生性能最佳、图模式优化彻底 | 重写工作量大、踩坑多 |
| **③ vLLM-Ascend 插件** | 用 vLLM 的硬件插件接口跑 LLM 推理 | LLM 推理开箱即用(连续批处理/PagedAttention) | 面向推理服务,训练场景不适用 |

生产环境推理建议优先 **③**:vLLM-Ascend 已经实现了硬件可插拔接口,`vllm serve` 的用法
在昇腾上和 GPU 几乎一致,只是底层换成 CANN + torch_npu。下面模拟一次"四步走"的完整迁移。
'''))

NB.md("## 4️⃣ 迁移四步走(用 torch 模拟)👣",
D('''
我们造一个迷你 LLaMA 骨架(Embedding + 两层 Transformer 块 + 输出头),模拟完整流程:

**Step 1 环境**:正常 import torch(真机上这里是 CANN + torch_npu);
**Step 2 代码迁移**:把 `state_dict` 当作迁移前后的"权重容器";
**Step 3 权重转换**:PyTorch 的权重名(如 `transformer.wte.weight`)要按映射表改名为
MindSpore/MindIE 认识的格式 —— 张量数值原样搬移,只是"改名 + 重新打包";
**Step 4 精度对齐**:对比 fp32 与 fp16 推理输出,用余弦相似度与最大绝对误差量化差距。
'''))

NB.code(D('''
import torch.nn as nn
import torch.nn.functional as F

class MiniLLaMA(nn.Module):
    """迷你 LLaMA 骨架:embedding + 2 层 transformer 块 + lm_head"""
    def __init__(self, vocab=128, d=64, nhead=4, nlayer=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, nhead, dim_feedforward=d*4,
                                     batch_first=True, activation="relu") for _ in range(nlayer)])
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.lm_head(h)

model = MiniLLaMA().eval()
with torch.no_grad():
    logits = model(torch.randint(0, 128, (2, 16)))

print("Step1 环境就绪:torch 模拟 CANN + torch_npu 环境")
print("Step2 代码迁移:state_dict 键名数量 =", len(model.state_dict()))

# ---- Step 3 权重转换:按映射表改名 ----
NAME_MAP = {
    "embed.weight": "backbone.embedding.weight",
    "lm_head.weight": "head.weight",
    "lm_head.bias": "head.bias",
}
converted = {}
for k, v in model.state_dict().items():
    new_key = NAME_MAP.get(k, "backbone.layers." + k)
    converted[new_key] = v.numpy()
print("Step3 权重转换:示例映射")
for src, dst in list(NAME_MAP.items()):
    print(f"    {src:16s} → {dst}")
print(f"    转换后键数 = {len(converted)},数值保持原样(np.float32)")
'''),
"🔧 Step1-3:环境 → 代码 → 权重。注意转换只改『名字和打包格式』,权重数值不动 —— 这是迁移最省心的一环。")

NB.md("## 5️⃣ 精度对齐:搬完还是原来的味吗?⚖️",
D('''
权重搬过去之后,最大的风险是**数值精度漂移**:fp32 权重换成 fp16/INT8 存储、算子换成昇腾实现,
每一层都可能有微小误差,累积起来模型就"变味"了。工程上拿两个数字验收:

- **余弦相似度(cosine similarity)**:两个输出的方向一致性,越接近 1.0 越好(要求 > 0.99);
- **最大绝对误差(max abs error)**:逐元素最大差,通常要求 1e-3 量级(INT8 量化放宽)。

下面对比 fp32 与 fp16 两套权重跑同一输入的输出,亲手算出这两个指标:
'''))

NB.code(D('''
import torch.nn.functional as F

def cosine_sim(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item())

def max_abs_err(a, b):
    return float((a.float() - b.float()).abs().max().item())

with torch.no_grad():
    x = torch.randint(0, 128, (2, 16))
    logits_f32 = model(x).float()

    # 模拟权重被转成 fp16 再读回来
    st_fp16 = {k: v.half() for k, v in model.state_dict().items()}
    model_h = MiniLLaMA().eval()
    model_h.load_state_dict({k: v.float() for k, v in st_fp16.items()})
    logits_f16 = model_h(x).float()

sim = cosine_sim(logits_f32, logits_f16)
err = max_abs_err(logits_f32, logits_f16)
print(f"余弦相似度 = {sim:.6f}   (验收线: > 0.99)")
print(f"最大绝对误差 = {err:.3e}   (fp32→fp16 通常 1e-3 量级)")

# 可视化:不同存储精度下的相似度
precisions = ["fp32→fp32", "fp32→fp16", "fp32→int8(模拟)"]
sims = [1.0, sim, 0.9820]           # int8 为示意
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(precisions, sims, color=["#27ae60", "#2e86c1", "#e67e22"], width=0.55)
ax.axhline(0.99, color="#c0392b", ls="--", lw=1.5, label="验收线 0.99")
for b, v in zip(bars, sims):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.4f}", ha="center", fontsize=10)
ax.set_ylim(0.96, 1.005)
ax.set_ylabel("余弦相似度")
ax.set_title("不同存储精度的输出对齐程度(值越高越接近原模型)")
ax.legend(); plt.tight_layout()
'''),
"🎯 看数字:fp16 几乎无损(相似度 ~0.999),INT8 明显下降 —— 这就是『精度对齐』要盯住的核心指标。")

NB.md("## 6️⃣ 算子适配:锅碗瓢盆不合用怎么办?🔧",
D('''
迁移时最麻烦的是**算子支持矩阵**不完整:昇腾的算子库(CANN 算子库)没有某些 PyTorch 算子的等价实现。
处理策略按成本从低到高:

1. **等价替换**:用几个支持算子拼出同样的效果(如把某些融合算子拆开);
2. **图改写**:让 GE 图引擎 / 编译器自动改写;
3. **手写 Ascend C**:用华为的算子开发语言自己写(下一课 92 详细讲)。

我们用 `torch.fx` 把迷你模型拆成算子清单,模拟『算子盘点 → 标出缺失 → 决定处理方式』:
'''))

NB.code(D('''
import torch.fx as fx
from collections import Counter

gm = fx.symbolic_trace(model)
kinds = []
for node in gm.graph.nodes:
    if node.op == "call_module":
        kinds.append(type(gm.get_submodule(node.target)).__name__)   # 模块类名即"算子种类"
    elif node.op == "call_function":
        kinds.append(node.target.__name__)
    elif node.op == "call_method":
        kinds.append(node.target)

support = {"Embedding", "Linear", "LayerNorm", "ReLU", "Dropout", "Softmax",
           "matmul", "mul", "add", "transpose", "getitem"}
missing = [k for k in kinds if k not in support]
print("算子盘点:共", len(kinds), "个节点;已支持", len(kinds) - len(missing),
      "个;缺失(需适配)", len(set(missing)), "类:")
for m, c in Counter(missing).most_common():
    print(f"  ✗ {m} × {c}")

# 处理方式分布(示意:按成本从低到高)
method = {"等价替换": 3, "图改写": 2, "手写 Ascend C": 1, "无需适配": 0}
fig, ax = plt.subplots(figsize=(7, 3.8))
names = list(method.keys()); vals = list(method.values())
ax.bar(names, vals, color=["#2e86c1", "#27ae60", "#c0392b", "#95a5a6"], width=0.55)
ax.set_ylabel("命中算子数(示意)")
ax.set_title("缺失算子的三类处理方式(示例分布)")
plt.tight_layout()
'''),
"📊 算子盘点:缺失的算子按成本分流 —— 能替换的先替换,必须手写的才写 Ascend C(成本最高)。")

NB.md("## 7️⃣ 部署与验证:让 LLaMA 真正服务起来 🚀",
D('''
如果走 **vLLM-Ascend** 路线,迁移的最后一步是部署,用法和 GPU 上几乎一样:

```bash
# 真机(昇腾 910B)示例:启动 OpenAI 兼容服务
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2-7B-Instruct \
    --trust-remote-code --dtype float16 --max-model-len 8192
```

验证指标就是第 50 课那套:TTFT / TPOT / 吞吐。下面用 plotly 画一个"迁移流程"交互图,
悬停可看每步的产出物:
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

steps = ["环境搭建", "代码迁移", "权重转换", "算子适配", "精度对齐", "部署服务"]
outs = ["CANN + torch_npu", "model.to('npu')", "state_dict → 映射改名", "Ascend C / 替换",
        "余弦相似度 > 0.99", "vllm serve / MindIE"]
times = [0.5, 1.0, 0.5, 3.0, 1.5, 0.5]      # 相对耗时(示意)
fig = go.Figure(go.Bar(x=steps, y=times, text=outs, textposition="outside",
                       marker_color=["#5dade2", "#2e86c1", "#27ae60",
                                     "#e67e22", "#e74c3c", "#8e44ad"]))
fig.update_layout(title="迁移六步的相对工作量与产出物(悬停/点读)",
                  xaxis_title="步骤", yaxis_title="相对工作量(示意)",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''),
"🎨 plotly 交互图:悬停看每步的产出物 —— 『算子适配』通常是迁移成本最高的环节,和我们的直觉一致。")

NB.md("## 8️⃣ 配套 App:迁移步骤交互助手 🎛️",
D('''
运行同目录的 `app_91_ascend_llama.py`,可**切换迁移路线、点开五个阶段、拖动精度滑杆**,
实时看流水线高亮与精度对齐曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_91_ascend_llama.py
```

浏览器打开 **http://localhost:8501**(本课 App 端口也可用 `--server.port 8691` 指定)。
完整源码如下(与同目录 `app_91_ascend_llama.py` 一字不差):
'''))

NB.code(app_cell("app_91_ascend_llama.py", APP_91),
"📜 运行本 cell 会覆盖写入 `app_91_ascend_llama.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "昇腾生态分层:CANN(地基)→ torch_npu/MindSpore(门)→ MindIE/vLLM-Ascend(厨房)",
        "三条迁移路线:torch_npu 最快、MindSpore 最彻底、vLLM-Ascend 最适合 LLM 推理生产",
        "迁移四步走:环境 → 代码 → 权重转换(改名打包)→ 精度对齐",
        "精度对齐看两个数:余弦相似度(>0.99)与最大绝对误差(1e-3 量级)",
        "算子适配三策略:等价替换、图改写、手写 Ascend C —— 手写成本最高、收益也最大",
    ],
    practice=[
        "给 MiniLLaMA 增加一个 LayerNorm 权重,补一条映射表规则,重跑权重转换",
        "把模型换成 fp16 后重算余弦相似度,试试更大输入规模下误差是否放大",
        "用 torch.fx 列出模型全部算子,对照昇腾算子清单(查 CANN 文档)找出真正缺失的",
        "查一下 vLLM-Ascend 的 support matrix,看 Qwen2-7B 需要哪几个 CANN 版本",
    ],
    links=[
        ("昇腾开发者社区", "https://www.hiascend.com"),
        ("CANN 文档中心", "https://www.hiascend.com/document"),
        ("MindSpore 安装与教程", "https://www.mindspore.cn"),
        ("vLLM-Ascend 支持矩阵", "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/"),
    ])

out = str(Path(CH11) / "91_ascend_llama.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_91_ascend_llama.py"
app_path.write_text(APP_91 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
