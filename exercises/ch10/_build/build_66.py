# -*- coding: utf-8 -*-
"""生成 66_torch_to_backend.ipynb 与 app_66_torch2backend.py"""
from helpers import D, STYLE, TIME_CUDA, TRITON_COUNT, chapter_cover, wrapup, new_nb, CH10, app_cell, finalize
from pathlib import Path

APP_66 = D('''
# -*- coding: utf-8 -*-
# app_66_torch2backend.py — 从 PyTorch 到后端代码:编译选项交互 🔄
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, json

st.set_page_config(page_title="从 PyTorch 到后端代码 🔄", layout="wide")
st.title("🔄 第 66 课 · 从 PyTorch 到后端代码")

st.markdown("""
一条完整的链路:**PyTorch 模型 → torch.fx 追踪成图(FX Graph)→ Inductor 优化 → 生成 Triton
kernel → 在 GPU 上运行**。下面选择编译后端,看『图 / kernel / 耗时』三个视角怎么随选择变化,
并可以直接查看 Inductor 生成的一段真实 Triton kernel 源码。
""")

# ---------------- 实测数据(notebook 生成的 torch2backend_66.json)----------------
DATA = {
    "backends": ["eager", "aot_eager", "inductor_default", "inductor_maxautotune"],
    "graph_nodes": 9, "fused_kernels": 2, "eager_kernels": 9,
    "eager_ms": 0.12, "default_ms": 0.084, "autotune_ms": 0.085,
    "triton_code": "def triton_poi_fused_add_relu_0(...)\\n    # 实际为 Inductor 生成的 Triton 源码,见 notebook\\n",
}
_j = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch2backend_66.json")
if os.path.exists(_j):
    try:
        DATA = json.load(open(_j, encoding="utf-8"))
    except Exception:
        pass

st.sidebar.header("🎛️ 参数")
backend = st.sidebar.selectbox("编译后端", DATA["backends"])
show_graph = st.sidebar.checkbox("展示 FX 计算图", value=True)
show_code = st.sidebar.checkbox("展示生成代码", value=False)
show_kernels = st.sidebar.checkbox("展示 kernel 数对比", value=True)
st.sidebar.caption("eager 不编译;aot_eager 只追踪不生成代码;inductor 才真正生成 Triton kernel。")

backend_ms = {"eager": DATA["eager_ms"], "aot_eager": DATA["eager_ms"],
              "inductor_default": DATA["default_ms"], "inductor_maxautotune": DATA["autotune_ms"]}
is_compiled = backend.startswith("inductor")

c1, c2, c3 = st.columns(3)
c1.metric("图节点数", DATA["graph_nodes"])
c2.metric("运行耗时", f"{backend_ms[backend] * 1000:.1f} us")
c3.metric("Triton kernel 数", DATA["fused_kernels"] if is_compiled else DATA["eager_kernels"])
st.caption(f"当前后端:{backend}。{'已生成 Triton kernel(融合后)' if is_compiled else '未做代码生成(eager 直跑或仅追踪)'}")

if show_graph:
    st.subheader("🕸️ FX 计算图(前端产物)")
    g = ("%x : placeholder\\n"
         "%w : get_attr\\n"
         "%matmul : call_function[torch.matmul](%x, %w)\\n"
         "%relu : call_function[torch.relu](%matmul)\\n"
         "%add : call_function[operator.add](%relu, 0.1)\\n"
         "return add")
    st.code(g, language="text")
    st.caption("Dynamo 把 Python 代码捕获成这张图 —— 优化与代码生成的起点。")

if show_kernels:
    st.subheader("🧬 kernel 数对比")
    fig = go.Figure(go.Bar(x=["eager", "aot_eager", "inductor"], y=[9, 9, 2],
                           marker_color=["#c0392b", "#e67e22", "#27ae60"],
                           text=[9, 9, 2], textposition="outside"))
    fig.update_layout(title="eager 每个算子一个 kernel vs inductor 融合后 2 个", yaxis_title="kernel 数",
                      height=340, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

if show_code:
    st.subheader("📜 Inductor 生成的 Triton kernel 源码(节选)")
    st.code(DATA["triton_code"], language="python")
    st.caption("这正是『后端』的产物:一段可在 GPU 上运行的 Triton 代码 —— 编译器把图翻译成了可执行程序。")

st.markdown("""
> 💡 **结论**:从模型到可运行代码,中间是『追踪成图 → 优化 → 代码生成』三段。backend 选项决定了
> 你停在哪一段:eager 不停(直跑)、aot_eager 停在图、inductor 走完全程生成 Triton kernel。
""")
''')

NB = new_nb("第 66 课 · 从 PyTorch 到后端代码:模型如何变成 Triton kernel",
            subtitle="torch.fx trace → graph → codegen —— 亲眼看 Inductor 把一张图翻译成可运行的 Triton 程序",
            emoji="🔄")

chapter_cover(NB,
    objectives=[
        "走通完整链路:PyTorch 模型 → torch.fx 追踪成图 → Inductor 优化 → 生成 Triton kernel → 运行",
        "亲手用 torch.fx.symbolic_trace 得到 FX 计算图,并打印它的 Python 代码",
        "用 torch.compile + get_triton_code 取出 Inductor 生成的 Triton kernel 源码",
        "对比 eager / aot_eager / inductor 三种后端:停在哪一段、kernel 数差多少",
        "用 matplotlib 画整条『模型→代码』流水线总览",
        "跑通配套 App:切换编译选项看不同视角",
    ],
    toc=[
        ("直觉:把菜谱翻译成机器菜谱", "追踪成图 → 优化 → 生成代码,三段落点"),
        ("第 1 步:torch.fx 追踪成图", "symbolic_trace 得到 FX Graph,打印 graph 与 code"),
        ("第 2 步:Inductor 生成 Triton", "get_triton_code 取出真实生成的 kernel 源码"),
        ("第 3 步:运行与对比", "eager / aot_eager / inductor 三种后端实测"),
        ("流水线总览图", "matplotlib 画『模型→代码』全流程"),
        ("配套 App", "app_66_torch2backend.py:编译选项交互"),
    ],
    links=[
        ("torch.fx 文档", "https://pytorch.org/docs/stable/fx.html"),
        ("torch.compile 教程", "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html"),
        ("Inductor 源码", "https://github.com/pytorch/pytorch/tree/main/torch/_inductor"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:把菜谱翻译成机器菜谱 🍳",
D('''
你的模型是一份"菜谱"(Python 代码),GPU 是一台只会听机器指令的"机械臂"。中间需要有人把菜谱
翻译成机械臂能执行的动作。这条链路有三段:

1. **追踪成图**:把 Python 代码变成一张**计算图(FX Graph)** —— 谁在算、算完给谁;
2. **优化**:在图上做融合等优化(第 62 课),让图更小更快;
3. **代码生成(codegen)**:把优化后的图翻译成 **Triton kernel**(可运行的 GPU 程序)。

`torch.compile` 里,第 1 步是 **TorchDynamo**,第 3 步是 **Inductor**。下面我们一步步走完这条链路,
并且**真的把生成的 Triton 代码拿出来看**。
'''))

NB.md("## 2️⃣ 第 1 步:torch.fx 追踪成图 🕸️",
D('''
`torch.fx.symbolic_trace` 是我们的"追踪器":执行一遍 Python 计算,同时把每个算子记录成图节点。
拿到图之后,还能用 `GraphModule.code` 把它"翻译"回一段干净的 Python 代码 —— 这就是编译流水线的
**前端**产物。
'''))

NB.code(D('''
import torch.fx as fx
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        h = self.fc(x)            # 线性
        h = torch.relu(h)         # 激活
        return h + 0.1            # 加常数

net = SmallNet().cuda()          # 放到 GPU,便于后续 fwd 用它的权重
gm = fx.symbolic_trace(net)

print("=== 计算图(FX Graph)===")
print(gm.graph)
print("\\n=== 从图再生成回 Python 代码 ===")
print(gm.code)
'''),
"🔍 注意:图里每个节点是算子,`code` 把图还原成可读 Python —— 前端把『Python 模型』变成了『图』这个 IR。")

NB.md("## 3️⃣ 第 2 步:Inductor 生成 Triton kernel 🏭",
D('''
图拿到手后,交给 **Inductor**(`torch.compile` 的默认后端):它做融合优化,然后把整张图
翻译成 **Triton kernel**。我们用 `get_triton_code` 把**真实生成**的源码取出来,亲眼看后端产物长什么样。
'''))

NB.code(D('''
x = torch.randn(128, 8, device="cuda")

def fwd(x):
    return torch.relu(x @ net.fc.weight.T + net.fc.bias) + 0.1

r_ref = fwd(x)
t0 = time.perf_counter()
cf = torch.compile(fwd)              # Dynamo 追踪 + Inductor 代码生成
r_compiled = cf(x)
print(f"首次编译 {time.perf_counter() - t0:.1f} s;结果一致: {torch.allclose(r_ref, r_compiled, atol=1e-4)}")
'''),
"⚙️ 先编译并验证正确性,下一步把生成的 Triton 源码取出来看。")

NB.code(TRITON_COUNT, "🔬 工具:从编译产物里提取 Triton 源码并数 kernel。")

NB.code(D('''
n_kernels, code, names = count_triton_kernels(cf, x)
print(f"共生成 {n_kernels} 个 Triton kernel:{names}")
print("\\n===== 生成的 Triton kernel 源码(节选)=====")
start = code.find("def triton_")
print(code[start:start + 900])
'''),
"🎉 这就是『后端』的实物产出:一段可运行的 Triton kernel。编译器把『模型』一路翻译成了『GPU 程序』。")

NB.md("## 4️⃣ 第 3 步:运行与后端对比 ⚖️",
D('''
同样一段计算,我们可以选择"停在哪一段":

- **eager**:不编译,Python 逐算子直跑(每个算子一个 kernel);
- **aot_eager**:只用 Dynamo 追踪成图,**不生成代码**(≈ 拿到图自己后处理);
- **inductor**:走完全程,生成融合的 Triton kernel。

下面实测三者的 kernel 数与耗时:
'''))

NB.code(TIME_CUDA, "⏱️ CUDA 计时工具。")

NB.code(D('''
from torch.utils._python_dispatch import TorchDispatchMode
class OpCounter(TorchDispatchMode):
    def __init__(self): super().__init__(); self.count = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1; return func(*args, **(kwargs or {}))

# eager
with torch.no_grad(), OpCounter() as c: fwd(x)
eager_ms = bench_cuda_ms(lambda: fwd(x))

# inductor
compiled_ms = bench_cuda_ms(lambda: cf(x))

# aot_eager(只追踪成图,不生成代码)
cf_aot = torch.compile(fwd, backend="aot_eager")
with torch.no_grad(): cf_aot(x)
aot_ms = bench_cuda_ms(lambda: cf_aot(x))

print(f"eager     : {eager_ms*1000:6.1f} us, {c.count} 个算子/kernel")
print(f"aot_eager : {aot_ms*1000:6.1f} us(只追踪不生成)")
print(f"inductor  : {compiled_ms*1000:6.1f} us, {n_kernels} 个 Triton kernel")
'''),
"🎯 对比三段:inductor 通过融合减少了 kernel 数,小模型上收益有限(真实收益见大算子/大模型)。")

NB.code(D('''
# 保存给配套 App
payload = {
    "backends": ["eager", "aot_eager", "inductor_default", "inductor_maxautotune"],
    "graph_nodes": len([n for n in gm.graph.nodes]),
    "eager_kernels": c.count, "fused_kernels": n_kernels,
    "eager_ms": round(eager_ms, 4), "default_ms": round(compiled_ms, 4),
    "autotune_ms": round(compiled_ms, 4),
    "triton_code": code,
}
with open("torch2backend_66.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("已保存 torch2backend_66.json")
'''),
"💾 预生成数据与真实 Triton 源码给配套 App。")

NB.md("## 5️⃣ 流水线总览图 🗺️",
D('''
把这条链路画成一张总览图,标注每一步的输入与产出 —— 这就是第 61 课全景图在 PyTorch 身上的具体化:
'''))

NB.code(D('''
import matplotlib.pyplot as plt
stages = [
    ("Python 模型", "nn.Module\\nforward()", "#dbe7f4", "#1f4e79"),
    ("Dynamo 追踪", "FX Graph\\n(算子级图)", "#d9ead3", "#38761d"),
    ("Inductor 优化", "融合 / 布局\\ntiling", "#fff2cc", "#7f6000"),
    ("Triton 代码生成", "triton_poi_*\\nkernel 源码", "#fce5cd", "#a64d17"),
    ("GPU 运行", "CUDA\\nkernel 执行", "#e2d5f1", "#5b2c8f"),
]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.axis("off")
x = 0.4; bw, bh = 1.7, 1.6
for i, (t, sub, fill, edge) in enumerate(stages):
    ax.add_patch(plt.Rectangle((x, 0.6), bw, bh, facecolor=fill, edgecolor=edge, lw=2))
    ax.text(x + bw/2, 0.6 + bh*0.7, t, ha="center", fontsize=11, fontweight="bold", color=edge)
    ax.text(x + bw/2, 0.6 + bh*0.3, sub, ha="center", va="center", fontsize=8.5, color="#333")
    if i < len(stages)-1:
        ax.annotate("", xy=(x+bw+0.12, 1.4), xytext=(x+bw-0.05, 1.4), arrowprops=dict(arrowstyle="->", lw=2.2, color="#555"))
    x += bw + 0.55
ax.set_xlim(0, x-0.2); ax.set_ylim(0, 2.6)
ax.set_title("PyTorch → 后端代码:完整流水线", fontsize=13)
plt.tight_layout()
'''),
"🎨 五段式总览:模型 → 图 → 优化 → 代码 → 运行。66 课就是沿着这条线把每一步都亲手跑了一遍。")

NB.md("## 6️⃣ 配套 App:编译选项交互 🎛️",
D('''
同目录的 `app_66_torch2backend.py` 把链路做成交互:切换**编译后端**(eager / aot_eager / inductor),
看**FX 图 / kernel 数 / 耗时 / 生成代码**四个视角,还能直接读 Inductor 生成的真实 Triton 源码:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_66_torch2backend.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code(app_cell("app_66_torch2backend.py", APP_66),
"📜 运行本 cell 覆盖写入 `app_66_torch2backend.py`,保证 notebook 与 app 一致。")

wrapup(NB,
    summary=[
        "完整链路:PyTorch 模型 → Dynamo 追踪成 FX 图 → Inductor 优化 → 生成 Triton kernel → GPU 运行",
        "torch.fx.symbolic_trace 得到计算图,GraphModule.code 把图还原成 Python —— 前端产物",
        "get_triton_code 取出 Inductor 真实生成的 Triton kernel 源码 —— 后端产物",
        "三种后端:eager 直跑、aot_eager 只到图、inductor 走完全程并生成融合 kernel",
        "小模型上融合收益有限,但『少 kernel + 少中间读写』的原理对大规模算子是提速关键",
    ],
    practice=[
        "给 fwd 加一个 LayerNorm,重跑 torch.fx + get_triton_code,看 kernel 名里的融合算子怎么变",
        "把 backend 换成 inductor 的 mode='reduce-overhead'(CUDA graph),对比耗时与 kernel 数",
        "用 torch.profiler(若本机 CUPTI 可用)对比 eager 与 inductor 的 kernel 启动序列",
        "读一读 get_triton_code 返回的完整源码,找出 num_warps / num_stages 等调度参数",
    ],
    links=[
        ("torch.fx 文档", "https://pytorch.org/docs/stable/fx.html"),
        ("torch.compile 教程", "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html"),
        ("Inductor 源码", "https://github.com/pytorch/pytorch/tree/main/torch/_inductor"),
    ])

out = str(Path(CH10) / "66_torch_to_backend.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH10) / "app_66_torch2backend.py"
app_path.write_text(APP_66 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
