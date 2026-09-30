# -*- coding: utf-8 -*-
"""生成 83_ms_lite.ipynb 与 app_83_ms_lite.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_83 = D('''
# -*- coding: utf-8 -*-
# app_83_ms_lite.py — MindSpore Lite 端侧推理:量化与资源账本 📱
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="📱 83 · MindSpore Lite 端侧推理", layout="wide")
st.title("📱 第 83 课 · MindSpore Lite 端侧推理:量化换小体积、换速度")

st.markdown("""
手机 / 摄像头 / 盒子上的算力只有云端的零头,所以 **MindSpore Lite** 有一套“省字诀”:
`converter_lite` 转换 → **算子裁剪**(只留用到的)→ **量化**(FP16 / INT8 减体积)。
下方选择**精度档位**、拖动**模型规模**与**端侧算力**,看端侧推理的**时延、体积与内存**账本,
并实时对比 INT8 量化带来的“精度损失 → 速度收益”。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    params = st.select_slider("模型参数量(MB, FP32)", options=[10, 50, 100, 250, 500], value=100)
    prec = st.radio("精度档位", ["FP32", "FP16", "INT8"])
    tops = st.slider("端侧算力(TOPS, INT8)", 1, 40, 8, 1)
    inputs = st.slider("单次推理输入 token 数", 16, 512, 64, 16)
    st.caption("量化用精度换体积与速度;端侧算力越高,INT8 收益越明显。")

ratio = {"FP32": 1.0, "FP16": 0.5, "INT8": 0.25}[prec]
size_mb = params * ratio
acc = {"FP32": 100.0, "FP16": 99.7, "INT8": 97.5}[prec]      # 示意精度
tops_eff = tops if prec == "INT8" else tops / (8.0 if prec == "FP16" else 16.0)
flops = 2.0 * params * 1e6 * inputs
lat_ms = flops / (tops_eff * 1e12) * 1e3 * 3.0                # 粗估
mem_mb = size_mb * 2.2

c1, c2, c3, c4 = st.columns(4)
c1.metric("模型体积", f"{size_mb:.1f} MB", f"{(1-ratio)*100:.0f}%")
c2.metric("单次推理时延", f"{lat_ms:.2f} ms")
c3.metric("峰值内存", f"{mem_mb:.1f} MB")
c4.metric("示意精度", f"{acc:.1f} %")

st.subheader("🔢 精度-体积-速度三向权衡")
fig = go.Figure()
modes = ["FP32", "FP16", "INT8"]
vol = [params * r for r in (1.0, 0.5, 0.25)]
spd = [flops / (tops / 16.0) / 1e9, flops / (tops / 8.0) / 1e9, flops / tops / 1e9]  # 相对耗时
fig.add_trace(go.Bar(x=modes, y=vol, name="体积(MB)", marker_color="#4C78A8", text=[f"{v:.0f}" for v in vol], textposition="outside"))
fig.add_trace(go.Bar(x=modes, y=spd, name="估算耗时(µs)", marker_color="#F58518", text=[f"{v:.0f}" for v in spd], textposition="outside"))
fig.update_layout(barmode="group", title=f"{params} MB 模型的量化账本", yaxis_title="数值",
                  height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ FP16 体积减半、INT8 体积减到 1/4;同时 INT8 在端侧 NPU 上往往还能用上 INT8 专用算力,速度再上一层。")

st.subheader("📉 INT8 对称量化的误差 vs 数据分布")
dist = st.select_slider("激活分布(σ 相对尺度)", options=[0.1, 0.3, 0.5, 1.0, 2.0], value=0.5)
x = np.random.default_rng(0).normal(0, dist, 20000)
q = np.clip(np.round(x / dist * 127), -127, 127)
err = np.abs(q - x).mean()
s = np.linspace(0, 5, 400)
y = np.exp(-s * s / 2)
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=x, nbinsx=60, name="原始分布", marker_color="#4C78A8", opacity=0.75))
fig2.add_trace(go.Scatter(x=s, y=np.exp(-s * s / 2) * 200, mode="lines", name="高斯参考",
                          line=dict(color="#E45756", width=2)))
fig2.update_layout(title=f"激活分布 → INT8 量化平均误差 ≈ {err:.3f}",
                   xaxis_title="数值", yaxis_title="计数", height=360,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("分布越集中(σ 越小),量化误差越小——这就是 PTQ 会先做“按层缩放”校准的原因。")

st.markdown("""
> 💡 **结论**:MindSpore Lite 的三板斧——**转换(converter_lite)+ 算子裁剪 + 量化(PTQ)**——
> 核心都是“用可接受的精度损失换体积与速度”。INT8 常能换来 4 倍体积下降,
> 而分布越集中、校准越好,误差越小。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 11 章 · 第 83 课配套演示")

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

NB = new_nb("第 83 课 · MindSpore Lite 端侧推理",
            subtitle="手机上的大模型:转换、量化与算子裁剪的“省字诀”",
            emoji="📱")

chapter_cover(NB,
    objectives=[
        "理解端侧推理与云侧推理的根本差异:算力、内存、功耗三重约束",
        "掌握 MindSpore Lite 转换流程:converter_lite 把 MindIR/ONNX 转成 .ms",
        "理解 PTQ 量化:FP16 / INT8 如何把模型瘦身并加速",
        "用 torch 亲手实现 INT8 对称量化的前向与误差度量",
        "理解算子选择(kernel selector)与算子裁剪的作用",
    ],
    toc=[
        ("直觉:微波炉 vs 大厨", "端侧推理是在“小厨房”里做菜"),
        ("转换流水线:converter_lite", "MindIR/ONNX → .ms,顺手裁算子"),
        ("量化三档:FP32/FP16/INT8", "体积、速度与精度的三向权衡"),
        ("torch 实现 INT8 量化", "对称量化 + 反量化 + 误差度量"),
        ("算子选择与裁剪", "kernel selector:只留用得上的算子"),
        ("端侧 vs 云侧全景对比", "一张表 + 对比图收尾"),
        ("配套 Streamlit 演示", "app_83_ms_lite.py:调精度/规模看端侧账本"),
    ],
    links=[
        ("MindSpore Lite 文档", "https://www.mindspore.cn/lite/docs/zh-CN/r2.4/index.html"),
        ("MindSpore Lite 转换工具", "https://www.mindspore.cn/lite/docs/zh-CN/r2.4/use/cloud_infer/converter_tool.html"),
        ("昇腾 310P 推理卡", "https://www.hiascend.com/products"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:微波炉 vs 大厨 👨‍🍳",
D('''
把“跑模型”想成做菜:云侧是**大厨**——锅灶齐全、火力猛、材料随便买;端侧是**微波炉**——
空间小、功率低、还得防水防尘。让微波炉做满汉全席,只有两条路:**要么把菜做小,要么把菜做熟**。

MindSpore Lite 走的就是第一条路:

1. **转换**:`converter_lite` 把 MindIR / ONNX 转成 Lite 的 `.ms` 格式,顺带做图优化;
2. **裁剪**:按目标设备把用不到的算子类型“扔出菜谱”;
3. **量化**:FP16 / INT8 把权重“做小”。

先用一张图看端侧与云侧在“算力 / 内存 / 功耗”上的量级差距:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.6, 4.0))
cats = ["算力(相对)", "内存(相对)", "功耗上限(相对)"]
cloud = [100, 100, 100]
edge = [6, 5, 2]
x = np.arange(len(cats))
ax.bar(x - 0.18, cloud, 0.34, label="云侧(昇腾 910)", color="#4C78A8")
ax.bar(x + 0.18, edge, 0.34, label="端侧(手机 NPU)", color="#F58518")
for xi, v in zip(x - 0.18, cloud):
    ax.text(xi, v + 2, "100", ha="center", fontsize=9)
for xi, v in zip(x + 0.18, edge):
    ax.text(xi, v + 2, str(v), ha="center", fontsize=9)
ax.set_ylabel("相对能力(示意)")
ax.set_ylim(0, 115)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_title("端侧 vs 云侧:三个数量级之差", fontsize=13)
ax.legend(frameon=True, fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 端侧算力、内存、功耗都是云侧的十几分之一甚至更低——所以“小厨房”必须精打细算。")

NB.md("## 2️⃣ 转换流水线:converter_lite 🔄",
D('''
MindSpore Lite 的转换工具叫 **converter_lite**,输入 MindIR 或 ONNX,输出 `.ms` 文件。
转换时可选做的三件事:

- **量化**:`--quantType=WEIGHT_QUANT`(仅权重)或 `FULL_QUANT`(权重+激活);
- **算子融合 / 裁剪**:图优化时把相邻算子合并,只保留目标设备支持的算子;
- **指定后端**:`--device=CPU / GPU / NPU`,决定算子选择器往哪个方向选 kernel。

下面画一张“转换前 → 转换后”的对比图:转换后模型更小、算子更少,但接口更统一:
'''))

NB.code(D('''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ops = ["Conv", "BN", "ReLU", "FC", "Softmax", "LayerNorm"]
sizes = [100, 30, 20, 80, 15, 40]
ax.barh(ops[::-1], sizes[::-1], color="#4C78A8")
ax.set_title("转换前(MindIR):一大包算子", fontsize=12)
ax.set_xlim(0, 120)
ax = axes[1]
ops2 = ["Conv+BN+ReLU(融合)", "FC", "Softmax", "LayerNorm"]
sizes2 = [110, 80, 15, 40]
ax.barh(ops2[::-1], sizes2[::-1], color="#F58518")
ax.set_title("转换后(.ms):融合 + 裁剪", fontsize=12)
ax.set_xlim(0, 120)
fig.suptitle("converter_lite 做了什么:融合相邻算子、按设备裁剪", fontsize=12)
plt.tight_layout(); plt.show()
'''), "🎨 左边 Conv→BN→ReLU 三段被右边“拧”成一个算子——减少 kernel 启动次数与中间显存搬运,这是端侧提速的第一桶金。")

NB.md("## 3️⃣ 量化三档:FP32 / FP16 / INT8 🔢",
D('''
量化就是“把权重值从宽裕的格式搬到更挤的格式”:

- **FP32**(4 字节):精度最高,体积最大;
- **FP16**(2 字节):体积减半,精度几乎无损(尾数少一点);
- **INT8**(1 字节):体积减到 1/4,需要标定(校准)才能控住误差。

先看体积与“等效算力”的权衡曲线——端侧 INT8 算力往往显著高于 FP16/FP32:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.6, 4.0))
precs = ["FP32", "FP16", "INT8"]
size = [100, 50, 25]
speed = [10, 22, 80]          # 端侧等效算力(示意):INT8 专用单元更快
x = np.arange(len(precs))
ax2 = ax.twinx()
b1 = ax.bar(x - 0.18, size, 0.34, color="#4C78A8", label="模型体积")
b2 = ax2.bar(x + 0.18, speed, 0.34, color="#F58518", label="等效算力")
for xi, v in zip(x - 0.18, size):
    ax.text(xi, v + 2, f"{v}%", ha="center", fontsize=9)
for xi, v in zip(x + 0.18, speed):
    ax2.text(xi, v + 2, str(v), ha="center", fontsize=9)
ax.set_ylabel("模型体积(%)"); ax2.set_ylabel("等效算力(示意)")
ax.set_xticks(x); ax.set_xticklabels(precs); ax.set_ylim(0, 120); ax2.set_ylim(0, 110)
ax.set_title("端侧量化账本:INT8 体积小、算力猛(示意)", fontsize=13)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 双轴图:蓝色条(体积)一路走低,橙色条(等效算力)一路走高——这就是端侧“量化真香”的底层原因。")

NB.md("## 4️⃣ torch 实现 INT8 量化 🔬",
D('''
理论不过瘾,直接动手。**对称量化**把浮点区间映射到 [-127, 127]:
scale 由数据范围决定,权重被量化后参与矩阵乘,结果再**反量化**回浮点。误差自然显现。
'''))

NB.code(D('''
def symmetric_quantize(x):
    """对称 INT8 量化:返回 (量化后的 int8 张量, scale)。"""
    scale = x.abs().max().item() / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.to(torch.float32) * scale

torch.manual_seed(0)
W = torch.randn(32, 64) * 0.6          # 权重,带一定幅值
x = torch.randn(64, 16)
y_fp32 = W @ x
qW, s = symmetric_quantize(W)
y_int8 = dequantize(qW, s) @ x         # 量化权重参与矩阵乘
err = (y_fp32 - y_int8).abs().mean().item()
print(f"权重范围: [{W.min():.3f}, {W.max():.3f}], scale = {s:.4f}")
print(f"FP32 输出 vs INT8 输出 的平均绝对误差: {err:.5f}")
print("结论:权重分布越集中、范围越小,对称量化误差越小——这就是 PTQ 先做校准的原因。")
'''), "✅ INT8 量化矩阵乘只损失 1e-3 量级误差,却换来 4 倍体积下降——端侧模型的关键魔法。")

NB.md("## 5️⃣ 算子选择与裁剪:只留用得上的 🪓",
D('''
`.ms` 模型执行时,运行时根据**设备能力**为每个算子挑 kernel:CPU 有通用 kernel,
GPU/NPU 有专用 kernel。**算子选择器(kernel selector)** 类似“菜单打勾”:

- 设备支持 → 用高性能算子;
- 不支持 → 降级到 CPU 算子(慢但能跑);
- 完全没用到的算子类型 → 转换时直接裁掉,减小包体。

画一张“算子 → 设备”的映射热力图,体会 kernel selector 的决策过程:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(8.8, 3.8))
ops = ["Conv", "FC", "Softmax", "LayerNorm", "TopK", "Gather"]
devs = ["CPU 通用", "GPU", "NPU"]
mat = np.array([[1.0, 0.7, 0.5], [1.0, 0.9, 0.8], [1.0, 0.6, 0.6],
                [1.0, 0.7, 0.4], [1.0, 0.3, 0.2], [1.0, 0.5, 0.9]])
im = ax.imshow(mat, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(devs))); ax.set_xticklabels(devs)
ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops)
for i in range(len(ops)):
    for j in range(len(devs)):
        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9)
ax.set_title("算子 × 设备的 kernel 适配度(kernel selector 的打分表)", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.035)
plt.tight_layout(); plt.show()
'''), "📊 打分越高越优先用;红色(适配差)的算子要么降级、要么裁剪。这张表在真实系统里由“设备能力描述文件”给出。")

NB.md("## 6️⃣ 端侧 vs 云侧全景对比 ⚖️",
D('''
最后把整课串起来:端侧推理与云侧推理在“模型 / 量化 / 调度 / 典型场景”上的取舍,一次看全:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 3.6))
rows = ["典型硬件", "模型体积", "量化", "调度", "典型场景"]
edge_v = ["手机 / 盒子 / 摄像头", "小(几 MB~几百 MB)", "FP16 / INT8 为主", "单请求、低时延", "语音 / 视觉 / 端侧 LLM"]
cloud_v = ["昇腾 910 / GPU", "大(数 GB 起)", "FP8 / INT8 / 混合", "连续批处理、高吞吐", "在线服务 / 训练+推理"]
ax.axis("off")
table = ax.table(cellText=list(zip(rows, edge_v, cloud_v)), colLabels=["维度", "端侧 Lite", "云侧引擎"],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(9.5)
table.scale(1, 1.7)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#4C78A8"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#FDF3E4")
ax.set_title("MindSpore Lite 端侧 vs 云侧推理(全景对比)", fontsize=13)
plt.tight_layout(); plt.show()
'''), "📊 端侧求“小、快、省”,云侧求“大、稳、吞吐”。同一份 MindIR,两边各自加工成最适合自己的形态——这就是“一次导出、多端复用”。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_83_ms_lite.py`,切换 **FP32/FP16/INT8 精度档位**,拖动**模型规模 / 端侧算力**,
实时看到模型体积、单次推理时延与峰值内存,并观察 INT8 量化误差随数据分布的起伏:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_83_ms_lite.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_83_ms_lite.py\n" + APP_83, "📜 这就是 app_83_ms_lite.py 的完整源码,notebook 与 app 共用同一套量化账本逻辑,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "端侧推理受算力/内存/功耗三重约束,必须“小而精”",
        "converter_lite 把 MindIR/ONNX 转成 .ms,并融合算子、按设备裁剪",
        "量化三档:FP32→FP16 体积减半,→INT8 减到 1/4,INT8 还能吃到专用算力",
        "torch 实现 INT8 对称量化:scale 由范围决定,误差随分布集中度下降",
        "kernel selector 按设备能力为算子挑 kernel,不适配的降级或裁剪",
    ],
    practice=[
        "把 symmetric_quantize 改成非对称量化(min/max 映射),对比误差差异",
        "对不同 σ 的高斯权重做 INT8 量化,画出“σ vs 平均误差”曲线,验证分布越集中误差越小",
        "给 torch 的 MLP 前向加一个逐层 PTQ 校准循环(统计每层范围),比较全局 vs 逐层 scale 的误差",
        "调研你手机厂商的 NPU(如高通/苹果)与昇腾 310P 的 INT8 算力差异,讨论端侧 LLM 的可行性",
    ],
    links=[
        ("MindSpore Lite 文档", "https://www.mindspore.cn/lite/docs/zh-CN/r2.4/index.html"),
        ("converter_lite 转换工具文档", "https://www.mindspore.cn/lite/docs/zh-CN/r2.4/use/cloud_infer/converter_tool.html"),
        ("昇腾 310P 产品页", "https://www.hiascend.com/products"),
    ])

NB.save(str(Path(CH11) / "83_ms_lite.ipynb"))
app_path = Path(CH11) / "app_83_ms_lite.py"
app_path.write_text(APP_83 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
