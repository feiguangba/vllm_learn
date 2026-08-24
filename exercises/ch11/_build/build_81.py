# -*- coding: utf-8 -*-
"""生成 81_ms_inference.ipynb 与 app_81_ms_inference.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_81 = D('''
# -*- coding: utf-8 -*-
# app_81_ms_inference.py — MindSpore 推理部署:链路与代价估算 🚀
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚀 81 · MindSpore 推理部署", layout="wide")
st.title("🚀 第 81 课 · MindSpore 推理部署:导出 → 加载 → 推理")

st.markdown("""
**MindSpore(昇思)** 是华为的 AI 框架,推理侧标准链路是:
`训练导出(MindIR)→ 模型转换 → 加载推理`。这和我们熟悉的
`PyTorch 导出(torch.jit / safetensors)→ vLLM 加载 → serve` 是同构的。
下方拖动**模型规模 / 批量 / 输入长度**,估算一条推理链路的**时延构成**与 **KV Cache 显存**,直观体会"模型越大、批量越大、prompt 越长 → 越贵"。
""")

# ---------------- 推理代价模型(与 notebook 一致,均为量级估算) ----------------
def est_latency(params, batch, prompt_tokens, threads, d=4096, model_len=8192):
    p = float(params)                                   # 参数量(十亿)
    fwd_flops = 2.0 * p * 1e9 * batch * prompt_tokens   # prefill 一次前向
    tops = 60.0 * threads / 8.0                         # 等效 NPU 算力(模拟)
    prefill_ms = fwd_flops / (tops * 1e12) * 1e3
    per_tok_ms = (2.0 * p * 1e9 / (tops * 1e12)) * 1e3  # 单 token 解码
    decode_ms = per_tok_ms * batch
    kv_gb = 2.0 * d * 2.0 / 1e9 * prompt_tokens * batch * 0.5
    return prefill_ms, decode_ms, per_tok_ms, kv_gb

with st.sidebar:
    st.header("🎛️ 参数")
    params = st.select_slider("模型参数量(十亿)", options=[0.5, 1.0, 3.0, 7.0, 13.0, 32.0], value=7.0)
    batch = st.slider("批量(batch)", 1, 64, 8, 1)
    prompt_tokens = st.slider("Prompt 长度(token)", 128, 8192, 1024, 128)
    threads = st.slider("等效算力(模拟,个 NPU AICore 单位)", 4, 64, 16, 1)
    st.caption("时延为「量级估算」,用于建立直觉,非真实硬件数据。")

prefill_ms, decode_ms, per_tok_ms, kv_gb = est_latency(params, batch, prompt_tokens, threads)
total = prefill_ms + decode_ms * (prompt_tokens // 32 + 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prefill 时延", f"{prefill_ms:.1f} ms")
c2.metric("单步 Decode 时延", f"{decode_ms:.1f} ms")
c3.metric("估算 KV Cache", f"{kv_gb:.2f} GB")
c4.metric("链路合计(约)", f"{total:.0f} ms")

# ---------------- 时延构成图 ----------------
fig = go.Figure()
fig.add_trace(go.Bar(x=["Prefill", f"Decode ×{prompt_tokens // 32 + 1} 步", "总时延"],
                     y=[prefill_ms, decode_ms * (prompt_tokens // 32 + 1), total],
                     marker_color=["#4C78A8", "#F58518", "#E45756"],
                     text=[f"{prefill_ms:.0f} ms", f"{decode_ms*(prompt_tokens//32+1):.0f} ms", f"{total:.0f} ms"],
                     textposition="outside"))
fig.update_layout(title="推理链路时延构成:Prefill vs Decode", yaxis_title="毫秒(ms)",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ Prompt 越长,Decode 步数越多;批量越大,每步时延越高——两把钳子夹住吞吐。")

# ---------------- 批量 vs 时延曲线 ----------------
bs = np.arange(1, 65)
lat = [est_latency(params, b, prompt_tokens, threads)[1] for b in bs]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=bs, y=lat, mode="lines+markers",
                          name="decode 时延", line=dict(color="#4C78A8", width=3)))
fig2.add_vline(x=batch, line_dash="dash", line_color="#E45756")
fig2.update_layout(title=f"{params:.1f}B 模型:Decode 时延随批量增长", xaxis_title="批量",
                   yaxis_title="单步 decode 时延 (ms)", height=380,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:MindSpore 推理与 vLLM 推理是"同一件事、两套栈":
> 导出/转换负责把训练图固化,加载/推理负责把固化图搬上硬件跑。
> 差别主要在**调度器**:vLLM/MindIE 这类服务引擎还会叠加连续批处理、
> PagedAttention、预分配 KV 池——这些正是第 84/85/88 课的主角。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 81 课配套演示")

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

NB = new_nb("第 81 课 · MindSpore 推理部署",
            subtitle="模型从训练端到推理端:导出 MindIR → 加载 → 推理,一条与 vLLM 同构的链路",
            emoji="🚀")

chapter_cover(NB,
    objectives=[
        "理解 MindSpore(昇思)在华为 AI 栈中的定位:昇腾芯片的“PyTorch”",
        "掌握推理三件套:导出(MindIR)→ 转换 → 加载推理",
        "用 torch 类比亲手走一遍“导出→加载→推理”流程",
        "对比 MindSpore 推理与 vLLM 推理的职责划分",
        "区分 MindSpore Lite 端侧推理与云侧推理的取舍",
    ],
    toc=[
        ("直觉:昇腾的“PyTorch”", "MindSpore 与 CANN、昇腾硬件的关系版图"),
        ("推理三件套:导出→转换→加载", "真实 API 术语讲透 MindIR 导出与加载推理"),
        ("torch 类比:亲手走一遍流程", "torch.jit 导出 + 加载,模拟 MindIR 生命周期"),
        ("推理链路代价模型", "prefill/decode 时延与 KV 显存的量级估算"),
        ("与 vLLM 推理对比", "两家厨房各司其职:框架 vs 服务引擎"),
        ("MindSpore Lite 端侧 vs 云侧", "手机上的推理与机房里的推理"),
        ("配套 Streamlit 演示", "app_81_ms_inference.py:拖参数看链路时延构成"),
    ],
    links=[
        ("MindSpore 官网", "https://www.mindspore.cn"),
        ("MindSpore 推理文档", "https://www.mindspore.cn/docs/zh-CN/r2.4/migration_guide/model_development/inference_and_training.html"),
        ("昇腾社区", "https://www.hiascend.com"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:昇腾的“PyTorch” 🧭",
D('''
你已经在前面十章把 vLLM 摸了个透:连续批处理、PagedAttention、量化、多卡并行……这些概念全部
建在 **PyTorch + CUDA** 这条“西方栈”上。华为给昇腾 NPU 配的“东方栈”长这样:

- **昇腾硬件**:910B / 910C 训练卡、310P 推理卡,算力单位是 NPU、AICore;
- **CANN**:昇腾的“CUDA”,提供算子库、运行时、通信库;
- **MindSpore(昇思)**:昇腾的“PyTorch”,训练与推理框架;
- **MindIE / vLLM-Ascend**:昇腾的“服务引擎”,负责把模型高效地跑成服务。

MindSpore 与 PyTorch 的对照可以写成一张表——下面的柱状图直观展示两者在“训练/推理/生态”上的
分工:MindSpore 更像“全家桶”(框架自带推理与转换),PyTorch 更像“组件超市”(靠 vLLM/TorchServe 补齐)。
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 3.6))
cats = ["训练", "模型导出", "推理运行时", "服务引擎", "算子库"]
pts = [90, 85, 80, 55, 40]
sp = [85, 40, 30, 80, 35]
x = np.arange(len(cats))
ax.bar(x - 0.19, pts, 0.38, label="MindSpore 全家桶", color="#4C78A8")
ax.bar(x + 0.19, sp, 0.38, label="PyTorch 组件超市", color="#F58518")
for xi, v in zip(x - 0.19, pts):
    ax.text(xi, v + 1.5, str(v), ha="center", fontsize=9)
for xi, v in zip(x + 0.19, sp):
    ax.text(xi, v + 1.5, str(v), ha="center", fontsize=9)
ax.set_ylabel("生态成熟度(示意)")
ax.set_ylim(0, 105)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_title("MindSpore 与 PyTorch 生态分工(示意图)")
ax.legend(frameon=True, fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 注意:MindSpore 从框架自带推理/导出/转换(左三根高),而 PyTorch 的推理能力主要靠 vLLM 等引擎补齐(服务引擎一栏更高)。这只是“生态分工示意”,不代表绝对好坏。")

NB.md("## 2️⃣ 推理三件套:导出 → 转换 → 加载推理 📦",
D('''
MindSpore 推理的标准链路,和你在前几章看到的 `torch.save → vllm.load` 几乎一模一样,只是名字换了:

1. **导出**:训练好的 `nn.Cell` 用 `mindspore.export` 固化成 **MindIR** 文件(`.mindir`)。
   MindIR 是昇思的中间表示(下一课 82 会细讲),把计算图 + 权重打包;
2. **转换**(可选):端侧还要用 `converter_lite` 把 MindIR 转成 MindSpore Lite 的 `.ms` 格式(第 83 课);
3. **加载推理**:云侧 `mindspore.load` 读回 MindIR 交给 `nn.GraphCell` 跑;端侧由 Lite 运行时加载 `.ms` 执行。

下面用一张流程图把它串起来——方框是产物,箭头是动作:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(10, 4.0))
steps = ["① 训练\\nnn.Cell", "② 导出\\nms.export → MindIR", "③ 加载\\nms.load", "④ 推理\\nnn.GraphCell"]
xs = [0.05, 0.38, 0.68, 0.92]
for x, s in zip(xs, steps):
    ax.add_patch(plt.Rectangle((x - 0.16, 0.35), 0.32, 0.3, fc="#DFE9F8", ec="#4C78A8", lw=2))
    ax.text(x, 0.5, s, ha="center", va="center", fontsize=11)
for a, b in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(b - 0.17, 0.5), xytext=(a + 0.17, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#E45756", lw=2))
ax.text(0.05, 0.78, "MindSpore 推理链路:同一个图,换不同的“搬运工”", fontsize=13, fontweight="bold")
ax.text(0.05, 0.12, "MindIR 是中间产物:训练端产出,推理端消费,跨端复用的“通用话单”。", fontsize=11, color="#555")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
plt.tight_layout(); plt.show()
'''), "🎨 图中第 2 步产出的 MindIR 是“一次导出、多端复用”的关键——云侧、端侧、服务引擎都从它开始。")

NB.md("## 3️⃣ torch 类比:亲手走一遍流程 🔬",
D('''
没有昇腾硬件,我们照样可以体会这条链路的“形状”。PyTorch 里最接近 MindIR 的产物是
**TorchScript**:一个把 Python 模型“固化”成图的中间表示。下面我们走一遍
`导出 → 保存 → 加载 → 推理` 四步,动作一一对应 MindSpore 的 `export → MindIR → load → GraphCell`。
'''))

NB.code(D('''
class MiniLM(torch.nn.Module):
    def __init__(self, d=8, hidden=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, hidden)
        self.fc2 = torch.nn.Linear(hidden, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MiniLM()
example = torch.randn(1, 8)
traced = torch.jit.trace(model, example)     # ① 导出:把模型固化成图
traced.save("mini_mindir_analog.zip")        # ② 保存:类似 .mindir 文件
loaded = torch.jit.load("mini_mindir_analog.zip")   # ③ 加载:读回图
out = loaded(example)                         # ④ 推理:直接跑图,不再经过 Python 层
print("推理输出形状:", tuple(out.shape))
print("注意:导出后不再需要训练代码,只要文件在手就能跑——这正是 MindIR 的定位。")
'''), "✅ 导出后模型与训练代码解耦:文件即模型,加载即运行。MindSpore 的 MindIR 就是为“跨端、跨运行时复用”而生的同一思路。")

NB.md("## 4️⃣ 与 vLLM 推理对比:两家厨房 🍜",
D('''
一句话总结:**MindSpore 负责“把图搬上硬件跑”,vLLM 负责“让很多人同时用这台硬件”。**
两者不是竞争关系,而是上下游:

- 用 MindSpore 训练/导出的模型,可以喂给 **vLLM-Ascend**(第 84 课)或 **MindIE**(第 88 课)做服务;
- vLLM 在 GPU 上做的连续批处理、PagedAttention,昇腾侧由 MindIE/vLLM-Ascend 提供等价能力。

下面把「框架级推理」与「引擎级推理」的差异画成对比图:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
topics = ["批处理", "KV 管理", "调度策略", "服务 API", "多卡并行"]
framework = [20, 15, 20, 10, 25]
engine = [90, 95, 90, 95, 85]
axes[0].barh(topics[::-1], framework[::-1], color="#4C78A8")
axes[0].set_title("框架级推理(MindSpore)", fontsize=12)
axes[0].set_xlim(0, 100); axes[0].set_xlabel("能力完备度(示意)")
axes[1].barh(topics[::-1], engine[::-1], color="#F58518")
axes[1].set_title("引擎级推理(vLLM / MindIE)", fontsize=12)
axes[1].set_xlim(0, 100); axes[1].set_xlabel("能力完备度(示意)")
fig.suptitle("推理职责分工:框架负责“能跑”,引擎负责“跑得快且多人用”", fontsize=12)
plt.tight_layout(); plt.show()
'''), "📊 框架级推理 = 单请求、串行、简单;引擎级推理 = 批量、调度、并发。所以真实生产环境几乎总是用引擎(vLLM / MindIE),框架只负责喂模型。")

NB.md("## 5️⃣ MindSpore Lite 端侧 vs 云侧 📱",
D('''
MindSpore 还分两个“流派”:

- **云侧(MindSpore 完整版)**:跑在昇腾/GPU 服务器,训练 + 推理一把抓;
- **端侧(MindSpore Lite)**:跑在手机/盒子/摄像头,模型要先用 `converter_lite` 转成 `.ms`,
  再做算子裁剪、INT8 量化,才能在有限算力上跑起来。

端侧推理的灵魂是“**少即是多**”:模型更小、精度更低、算子更少、功耗更省。把它与云侧对比:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
parts = ["模型权重", "算子库", "运行时", "量化后"]
vals = [100, 60, 20, 0]
ax.bar(["云侧", "端侧"], [200, 80], color=["#4C78A8", "#F58518"], width=0.5)
ax.text(0, 202, "大而全", ha="center"); ax.text(1, 82, "小而精", ha="center")
ax.set_ylim(0, 240); ax.set_title("端侧 vs 云侧“负担”对比(示意)", fontsize=12)
ax = axes[1]
stages = ["FP32 模型", "FP16 转换", "INT8 量化"]
size = [100, 50, 25]
ax.bar(stages, size, color=["#4C78A8", "#F58518", "#E45756"], width=0.5)
for i, v in enumerate(size):
    ax.text(i, v + 3, f"{v}%", ha="center")
ax.set_ylim(0, 120); ax.set_title("端侧模型瘦身链路:体积占比(示意)", fontsize=12)
plt.tight_layout(); plt.show()
'''), "📊 端侧模型一路瘦身:FP16 减一半、INT8 再减一半,配合算子裁剪,才能在手机 NPU 上实时推理。下一课(83)专门拆解这套流程。")

NB.md("## 6️⃣ 交互图:推理链路时延构成 ⏱️",
D('''
推理不是“一次计算”,而是 **Prefill + 多次 Decode** 的接力赛:先一次性处理整个 prompt(prefill),
再一个个 token 往后“蹦”(decode)。prompt 越长、批量越大,链路越贵。用 plotly 画一张交互图,
悬停可见各部分时延:
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

params = 7.0; batch = 8; prompt = 2048
top = 60.0 * 16 / 8.0   # 模拟等效算力
prefill_ms = 2.0 * params * 1e9 * batch * prompt / (top * 1e12) * 1e3
decode_ms = 2.0 * params * 1e9 * batch / (top * 1e12) * 1e3
labels = [f"Prefill\\n{prompt} token", "Decode\\n每步", "总时延(约 64 步)"]
values = [prefill_ms, decode_ms, prefill_ms + decode_ms * 64]
fig = go.Figure(go.Bar(x=labels, y=values,
                       marker_color=["#4C78A8", "#F58518", "#E45756"],
                       text=[f"{v:.1f} ms" for v in values], textposition="outside"))
fig.update_layout(title=f"{params}B 模型 · batch={batch}:Prefill 与 Decode 的时延构成",
                  yaxis_title="毫秒(ms)", height=400, margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "📊 悬停可见数值。你会看到:prompt 越长,prefill 占比越高;输出越长,decode 步数越多——这就是服务引擎用连续批处理来“摊薄”它的原因(第 88 课)。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_81_ms_inference.py`,拖动**模型规模 / 批量 / Prompt 长度**,实时看到
`Prefill 时延 → Decode 时延 → KV 显存` 的连锁反应,并画出“批量 vs 单步时延”的增长曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_81_ms_inference.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_81_ms_inference.py\n" + APP_81, "📜 这就是 app_81_ms_inference.py 的完整源码,notebook 与 app 共用同一套时延估算逻辑,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "MindSpore 是昇腾芯片的“PyTorch”,推理链路 = 导出 MindIR → 转换 → 加载推理",
        "MindIR 是跨端复用的话单:训练端产出、云侧/端侧/引擎共同消费",
        "torch.jit 的“导出→保存→加载→推理”完整复现了这条链路的形状",
        "框架级推理负责“能跑”,引擎级推理(vLLM / MindIE)负责“跑得快且多人用”",
        "MindSpore Lite 端侧推理通过 FP16/INT8 量化与算子裁剪换取小体积、低功耗",
    ],
    practice=[
        "把 MiniLM 换成 2 层 Transformer,用 torch.jit.script(而非 trace)导出,观察图结构差异",
        "给 MiniLM 加一个 batch=16 的输入,估算并对比 prefill 与 decode 的 FLOPs",
        "搜索 mindspore.export 的参数(file_format / input_names),写一段伪代码导出带动态轴的 MindIR",
        "对比 torch.jit 与 MindIR:为什么 MindIR 强调“一次导出、多端复用”?列出至少 2 个理由",
    ],
    links=[
        ("MindSpore 推理与训练文档", "https://www.mindspore.cn/docs/zh-CN/r2.4/migration_guide/model_development/inference_and_training.html"),
        ("MindSpore 官网", "https://www.mindspore.cn"),
        ("昇腾社区", "https://www.hiascend.com"),
    ])

NB.save(str(Path(CH11) / "81_ms_inference.ipynb"))
app_path = Path(CH11) / "app_81_ms_inference.py"
app_path.write_text(APP_81 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
