# -*- coding: utf-8 -*-
"""生成 86_cann_oplib.ipynb 与 app_86_cann_oplib.py"""
from helpers import D, HEADER, chapter_cover, wrapup, new_nb, CH11
from pathlib import Path

APP_86 = D('''
# -*- coding: utf-8 -*-
# app_86_cann_oplib.py — CANN 算子库浏览器:类别、融合与对照 🗃️
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🗃️ 86 · CANN 算子库", layout="wide")
st.title("🗃️ 第 86 课 · CANN 算子库与融合:昇腾的“cuDNN/cuBLAS”")

st.markdown("""
**CANN(Compute Architecture for Neural Networks)** 是昇腾的异构计算平台,
地位相当于 CUDA。它内置一套高性能**算子库**(GEMM、卷积、归一化、FlashAttention、通信……),
并支持把多个小算子**融合**成一个大算子,减少 kernel 启动与中间数据搬运。
下方选择**算子类别**,浏览其与 CUDA 生态的**对照关系**,并拖动算子数量观察**融合的收益曲线**。
""")

DATA = {
    "矩阵乘 GEMM": {
        "ops": ["GEMM", "Strided Batch GEMM", "MatMul-Add 融合", "MatMul-GELU 融合"],
        "cuda": ["cuBLAS gemm", "cuBLAS strided_batched_gemm", "Fusion(cuBLASLt)", "Fusion(cuDNN)"],
        "comm": "对标 cuBLAS:矩阵乘最核心的高性能算子",
    },
    "神经网络 NN": {
        "ops": ["Conv2D", "BatchNorm", "LayerNorm", "Softmax", "FlashAttention"],
        "cuda": ["cuDNN conv", "cuDNN batchnorm", "ATen layernorm", "cuDNN softmax", "FlashAttention kernel"],
        "comm": "对标 cuDNN:卷积 / 归一化 / 注意力一族",
    },
    "通信 HCCL": {
        "ops": ["AllReduce", "AllGather", "ReduceScatter", "P2P Send/Recv"],
        "cuda": ["NCCL AllReduce", "NCCL AllGather", "NCCL ReduceScatter", "NCCL P2P"],
        "comm": "对标 NCCL:多卡集合通信(张量并行 / 专家并行的命脉)",
    },
    "运行时 AscendCL": {
        "ops": ["aclrtMalloc", "aclrtLaunch", "aclrtMemcpyAsync", "aclmdlExecute"],
        "cuda": ["cudaMalloc", "cudaLaunchKernel", "cudaMemcpyAsync", "cudaGraphLaunch"],
        "comm": "对标 CUDA Runtime:显存管理、kernel 启动、模型执行",
    },
}

with st.sidebar:
    st.header("🎛️ 参数")
    cat = st.selectbox("算子类别", list(DATA.keys()))
    n_ops = st.slider("融合前算子数量", 2, 12, 6, 1)
    data_gb = st.slider("中间数据规模(GB)", 0.5, 16.0, 4.0, 0.5)
    st.caption("融合的核心收益:少启动 kernel + 中间结果不落显存。")

D = DATA[cat]
c1, c2 = st.columns(2)
c1.metric("算子数", len(D["ops"]))
c2.metric("CUDA 对照", D["cuda"][0].split(" ")[0])
st.caption(D["comm"])

st.subheader("🗂️ CANN 算子 ↔ CUDA 生态对照")
st.dataframe(pd.DataFrame({"昇腾(CANN)": D["ops"], "CUDA 生态": D["cuda"]}), width="stretch")

# ---------------- 融合收益曲线 ----------------
launch_ms = 0.05 * n_ops                    # kernel 启动开销
traffic_gb = n_ops * data_gb * 0.4          # 中间搬运(融合后几乎为 0)
fig = go.Figure()
fig.add_trace(go.Bar(x=["非融合\\n(逐个算子)", "融合\\n(单算子)"],
                     y=[launch_ms + traffic_gb * 2, 0.05 + 0.05 * data_gb],
                     marker_color=["#E45756", "#4C78A8"],
                     text=["高", "低"], textposition="outside"))
fig.update_layout(title=f"融合前后:kernel 启动 + 中间数据搬运(示意)", yaxis_title="相对开销",
                  height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("⭐ 融合省掉的“中间数据落显存再读回”往往是最大的开销——这就是 FlashAttention 快的原因。")

# ---------------- 带宽收益曲线 ----------------
ns = list(range(2, 13))
fig2 = go.Figure(go.Scatter(x=ns, y=[n * data_gb * 0.4 for n in ns], mode="lines+markers",
                            line=dict(color="#F58518", width=3),
                            name="非融合搬运量"))
fig2.add_hline(y=0.05 * data_gb, line_dash="dash", line_color="#4C78A8",
               annotation_text="融合后搬运量")
fig2.update_layout(title="中间数据搬运量随算子数增长(融合后趋近于 0)", xaxis_title="算子数",
                   yaxis_title="搬运量(GB·次)", height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **结论**:CANN 不是“昇腾版的 CUDA”这么简单——它同时提供**运行时、算子库、通信库、
> 编译工具链**。理解它最划算的方式,就是与 CUDA 生态**逐个对照**:
> `AscendCL ↔ CUDA Runtime`、`GEMM/NN 算子库 ↔ cuBLAS/cuDNN`、`HCCL ↔ NCCL`、`Ascend C ↔ CUDA C`。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 11 章 · 第 86 课配套演示")

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

NB = new_nb("第 86 课 · CANN 算子库与融合",
            subtitle="昇腾的“cuDNN/cuBLAS”:算子库、融合算子与 CUDA 生态逐项对照",
            emoji="🗃️")

chapter_cover(NB,
    objectives=[
        "理解 CANN 的分层架构:运行时、算子库、通信库、编译工具链",
        "建立 CUDA ↔ CANN 的逐项对照表(AscendCL/cuBLAS/cuDNN/NCCL/Ascend C)",
        "理解融合算子的收益:少启动、少搬运、多缓存复用",
        "认识 Ascend C 算子编程范式与 torch 模拟融合的对比",
        "用数据量化“融合 vs 非融合”的开销差异",
    ],
    toc=[
        ("直觉:CANN = 昇腾的 CUDA", "一张对照表认识整个异构平台"),
        ("分层架构:从框架到硬件", "应用 → MindSpore → AscendCL → GE → NPU"),
        ("逐项对照:谁是谁", "cuBLAS/cuDNN/NCCL/nvcc 的昇腾表亲"),
        ("融合算子:把 4 变 1", "torch 模拟融合 vs 非融合的搬运账本"),
        ("Ascend C:昇腾的 CUDA C", "一种写昇腾 kernel 的编程范式"),
        ("量化收益曲线", "算子越多,融合越香"),
        ("配套 Streamlit 演示", "app_86_cann_oplib.py:选类别、看对照、看收益"),
    ],
    links=[
        ("昇腾社区 CANN 主页", "https://www.hiascend.com/cann"),
        ("Ascend C 编程文档", "https://www.hiascend.com/zh/developer/techarticles"),
        ("CUDA 文档", "https://docs.nvidia.com/cuda/"),
    ])

NB.code(HEADER, "✅ 第一段代码:KMP 保护 + 固定 seed + 会议论文风绘图环境;本机无昇腾硬件,全课用 torch 类比讲解。")

NB.md("## 1️⃣ 直觉:CANN = 昇腾的 CUDA 🧭",
D('''
在 GPU 上,你想直接写算子,得学 CUDA;**昇腾上则学 CANN**。
CANN 不是“一个库”,而是昇腾的**整层异构平台**,包含:

- **AscendCL**(Ascend Computing Language):运行时 API,管显存、启 kernel——对标 CUDA Runtime;
- **GE(Graph Engine)**:图引擎,负责图优化与算子调度;
- **算子库**:GEMM、卷积、归一化、FlashAttention、通信等高性能算子——对标 cuBLAS/cuDNN/NCCL;
- **Ascend C**:写昇腾 kernel 的编程语言——对标 CUDA C。

用一张“俄罗斯套娃”图看它在整条栈里的位置:
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 4.2))
layers = [("模型与应用(LLM / 训练 / 推理)", 0.14, "#DFE9F8", "#4C78A8"),
          ("框架层:MindSpore / PyTorch(torch_npu)", 0.28, "#DFE9F8", "#4C78A8"),
          ("CANN: AscendCL + GE + 算子库 + HCCL", 0.42, "#FDF3E4", "#E45756"),
          ("昇腾硬件:910B/910C/310P", 0.56, "#EDE4F5", "#6B4FA1")]
for label, y, fc, ec in layers:
    ax.add_patch(plt.Rectangle((0.12, y), 0.76, 0.12, fc=fc, ec=ec, lw=2))
    ax.text(0.5, y + 0.06, label, ha="center", va="center", fontsize=11,
            color="#333" if y != 0.42 else "#A03A45", fontweight="bold" if y == 0.42 else "normal")
ax.text(0.5, 0.92, "昇腾软件栈:CANN 是框架与硬件之间的“万能适配层”", fontsize=13, ha="center", fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
plt.tight_layout(); plt.show()
'''), "🎨 橙色那层就是 CANN:上面谁都能来(框架),下面谁都能接(硬件)。vLLM 的昇腾版正是透过这一层调用一切。")

NB.md("## 2️⃣ 逐项对照:谁是谁 🗂️",
D('''
记 CANN 最轻松的方法,是把它跟 CUDA 生态**逐项配对**:

| CUDA 生态 | CANN 对照 | 干什么 |
|---|---|---|
| CUDA Runtime(`cudaMalloc`…) | AscendCL(`aclrtMalloc`…) | 显存 / 上下文 / kernel 启动 |
| cuBLAS(GEMM) | GE 中的高性能 GEMM 算子 | 矩阵乘 |
| cuDNN(卷积/RNN/attention) | CANN NN 算子库 | 神经网络算子 |
| cuDNN Graph / cuBLASLt | GE 图优化 + 融合算子 | 算子融合 |
| NCCL(集合通信) | HCCL | AllReduce / AllGather… |
| CUDA C(`__global__`) | Ascend C | 写昇腾 kernel |
| cuDNN flash-attention | `npu_fusion_attention` | 融合注意力 |

画成条形对照图(按“生态成熟度”示意):
'''))

NB.code(D('''
fig, ax = plt.subplots(figsize=(9, 4.0))
pairs = [("CUDA Runtime ↔ AscendCL", 95), ("cuBLAS ↔ GEMM 算子", 92),
         ("cuDNN ↔ NN 算子库", 88), ("NCCL ↔ HCCL", 85), ("CUDA C ↔ Ascend C", 78),
         ("FlashAttention ↔ npu_fusion_attention", 90)]
labels = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
bars = ax.barh(labels[::-1], vals[::-1], color="#4C78A8")
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
ax.set_xlim(0, 105)
ax.set_xlabel("生态成熟度(示意)")
ax.set_title("CUDA ↔ CANN 逐项对照:功能一一对应,成熟度略有差异", fontsize=12)
plt.tight_layout(); plt.show()
'''), "📊 每个 CUDA 组件在 CANN 里都能找到“表亲”。记住这六对,昇腾的开发地图就打开了。")

NB.md("## 3️⃣ 融合算子:把 4 个 kernel 变 1 个 🔬",
D('''
昇腾性能优化的第一课是**融合**:把相邻的多个算子合并成一个。以注意力为例,
`QK^T → Scale → Softmax → ×V` 在旧做法里是 4 个 kernel、中间结果 3 次“落显存再读回”;
融合后是 1 个 kernel、0 次中间落地。

用 torch 模拟这个账本:分别跑“4 次独立计算 + 中间存张量”和“一次函数内完成”,比较
“虚拟搬运量”:
'''))

NB.code(D('''
def unfused_attn(q, k, v, scale):
    """非融合:每步一个“kernel”,中间结果落地(用 .clone() 模拟搬回显存)。"""
    s = (q @ k.T) * scale
    s = s.clone()                       # 模拟中间落显存
    p = torch.softmax(s, dim=-1)
    p = p.clone()                       # 模拟中间落显存
    return p @ v, 2                     # 返回 中间落地次数=2

def fused_attn(q, k, v, scale):
    """融合:单次函数内完成,中间结果只在寄存器/缓存里流动。"""
    return torch.softmax((q @ k.T) * scale, dim=-1) @ v, 0

torch.manual_seed(0)
q, k, v = (torch.randn(16, 32) for _ in range(3))
o1, w1 = unfused_attn(q, k, v, 1.0 / 32 ** 0.5)
o2, w2 = fused_attn(q, k, v, 1.0 / 32 ** 0.5)
print(f"非融合:中间落地 {w1} 次 → 每次搬运 32×32×4B ≈ 4KB 显存往返")
print(f"融合  :中间落地 {w2} 次 → 零搬运")
print("结果一致:", torch.allclose(o1, o2, atol=1e-6))
'''), "✅ 数学上完全等价,但融合版本少 2 次“显存往返”。序列越长、头越多,省下的带宽越可观。")

NB.md("## 4️⃣ Ascend C:昇腾的 CUDA C ✏️",
D('''
如果现有算子库不够用,你可以用 **Ascend C** 自己写昇腾 kernel——就像用 CUDA C 写 GPU kernel。
思维模型基本一致:

- 数据先搬进 **NPU 的本地内存(local memory)**;
- 在 **AICore**(AI 计算核)上算;
- 结果搬回**全局内存(global memory)**。

下面用伪代码看一个 Ascend C 算子的“形”(本机无昇腾硬件,仅讲解不执行):
'''))

NB.code(D('''
ascend_c_demo = r"""
#include "kernel_operator.h"
using namespace AscendC;

class KernelRelu {                      // 一个 ReLU kernel
public:
    __aicore__ inline KernelRelu(GM_ADDR x, GM_ADDR y) {
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
    }
    __aicore__ inline void Process() {
        // 1. 从全局内存搬到本地(Tensor)
        LocalTensor<half> xLocal = tQueIn.AllocTensor<half>();
        DataCopy(xLocal, xGm, len);          // 搬运
        // 2. 在 AICore 上算
        Max(xLocal, xLocal, (half)0.0f);     // ReLU = max(x, 0)
        // 3. 搬回全局内存
        DataCopy(yGm, xLocal, len);
        tQueIn.FreeTensor(xLocal);
    }
private:
    GlobalTensor<half> xGm, yGm;             // 全局内存句柄
};
"""
print(ascend_c_demo)
print("核心三连:DataCopy(搬入) → 计算 → DataCopy(搬出),与 CUDA kernel 的 load→compute→store 同构。")
'''), "✏️ Ascend C 与 CUDA C 的思维模型几乎一比一:全局↔本地内存搬运 + AICore 计算。只是名字和工具链不同。")

NB.md("## 5️⃣ 量化收益:算子越多,融合越香 📈",
D('''
融合的收益随“算子链长度”和“中间数据规模”增长。画一条曲线:非融合的总开销(kernel 启动 +
中间搬运)随算子数线性上涨,而融合后几乎水平:
'''))

NB.code(D('''
ns = np.arange(2, 16)
launch = 0.05 * ns                       # 每次启动固定开销
traffic = 0.4 * ns                       # 中间搬运随算子数线性涨
total_unfused = launch + traffic
total_fused = 0.05 + 0.05 * np.ones_like(ns)
fig, ax = plt.subplots(figsize=(8.8, 4.2))
ax.plot(ns, total_unfused, "o-", color="#E45756", lw=2, label="非融合:启动 + 搬运")
ax.plot(ns, launch, "--", color="#F58518", lw=1.5, label="其中 kernel 启动")
ax.plot(ns, total_fused, "s-", color="#4C78A8", lw=2, label="融合:几乎恒定")
ax.fill_between(ns, total_fused, total_unfused, color="#E45756", alpha=0.08)
ax.set_xlabel("算子链长度"); ax.set_ylabel("相对开销")
ax.set_title("融合收益随链长增长:越长的链,融合越划算", fontsize=12)
ax.legend(frameon=True, fontsize=9)
plt.tight_layout(); plt.show()
'''), "📊 链条越长,橙色线越高、蓝色线越低——所以 attention 这类“长链、大中间量”的算子,融合收益最显著。")

NB.md("## 6️⃣ 交互图:算子库浏览 🗃️",
D('''
用 plotly 把“算子库 → 依赖它的典型模型”画成层次条形图,直观感受昇腾算子库的覆盖范围:
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

cats = ["GEMM 族", "NN 族", "通信族", "运行时"]
counts = [18, 26, 8, 12]
fig = go.Figure(go.Bar(x=cats, y=counts, marker_color=["#4C78A8", "#F58518", "#E45756", "#6B4FA1"],
                       text=counts, textposition="outside"))
fig.update_layout(title="昇腾算子库类别与算子数(示意)",
                  yaxis_title="算子数", height=380, margin=dict(l=10, r=10, t=50, b=10))
fig.show()
'''), "🗃️ 柱状图只是示意,真实数量随 CANN 版本持续增长——关键在于:你需要的算子大概率已经有人写好并调优了。")

NB.md("## 7️⃣ 配套 Streamlit 演示 🎛️",
D('''
运行同目录下的 `app_86_cann_oplib.py`,用下拉框切换**算子类别**,浏览 CANN 算子 ↔ CUDA 生态的
**逐项对照表**,并拖动算子数观察**融合收益曲线**:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_86_cann_oplib.py
```

浏览器打开 **http://localhost:8501**。完整源码如下:
'''))

NB.code("%%writefile app_86_cann_oplib.py\n" + APP_86, "📜 这就是 app_86_cann_oplib.py 的完整源码,notebook 与 app 共用同一套 CUDA↔CANN 对照表,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "CANN = 昇腾的 CUDA:运行时(AscendCL)+ 图引擎(GE)+ 算子库 + 通信库(HCCL)",
        "CUDA ↔ CANN 六对表亲:CUDA Runtime↔AscendCL、cuBLAS↔GEMM、cuDNN↔NN 库、NCCL↔HCCL、CUDA C↔Ascend C",
        "融合算子的账本:少 kernel 启动、中间结果零落地,attention 这类长链收益最大",
        "Ascend C 与 CUDA C 同构:DataCopy 搬入 → AICore 计算 → DataCopy 搬出",
        "算子库 + 融合是昇腾性能的两大支柱,理解它们就能读懂昇腾上的一切优化",
    ],
    practice=[
        "把 unfused_attn 的中间落地次数改成 3(QK^T、Softmax、P×V 各一次),重算对比",
        "写一段“逐算子”与“融合”的 Conv→BN→ReLU torch 代码,统计两者数值差异",
        "调研 AscendCL 的 aclrtLaunch / aclrtSynchronizeStream 与 CUDA 的流 API 有何对应关系",
        "对比 Triton kernel 与 Ascend C kernel 的思维模型,写一段二者对照的伪代码",
    ],
    links=[
        ("昇腾 CANN 主页", "https://www.hiascend.com/cann"),
        ("Ascend C 技术文章", "https://www.hiascend.com/zh/developer/techarticles"),
        ("NVIDIA cuDNN/cuBLAS 文档", "https://developer.nvidia.com/cudnn"),
    ])

NB.save(str(Path(CH11) / "86_cann_oplib.ipynb"))
app_path = Path(CH11) / "app_86_cann_oplib.py"
app_path.write_text(APP_86 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
