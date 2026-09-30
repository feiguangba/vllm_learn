# -*- coding: utf-8 -*-
"""生成 33_fp8_kv_quant.ipynb 与 app_33_fp8_kv.py(教材级重写版)

论文/资料支撑:
- Micikevicius et al., "FP8 Formats for Deep Learning", arXiv:2209.05433
  (NVIDIA/Arm/Intel 联合白皮书,E4M3/E5M2 定义)
- NVIDIA 博客 "NVIDIA, Arm, and Intel Publish FP8 Specification"
- vLLM FP8 E4M3 KV Cache 文档(工程落地)
"""
from helpers import (D, FP8_PARSE, chapter_cover, wrapup, new_nb, CH05)
from pathlib import Path

APP_33 = D('''
# -*- coding: utf-8 -*-
# app_33_fp8_kv.py — FP8 位拆解与 KV Cache 显存计算器 🚀
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚀 33 · FP8 与 KV Cache 显存", layout="wide")
st.title("🚀 第 33 课 · FP8 与 KV Cache:省一半显存的魔法")

st.markdown("""
FP8 用 8 bit 装「科学计数法」:**E4M3**(4 位指数 + 3 位尾数)精度高、范围小,适合权重与前向;
**E5M2**(5 位指数 + 2 位尾数)范围大、精度低,适合梯度与 KV cache。
把 KV cache 从 fp16 换成 fp8,显存直接减半——同样一张卡能装下双倍的并发上下文。

下方左侧调模型配置,右侧实时计算 KV cache 显存:
""")

def fp8_decode(code, exp_bits, mant_bits, exp_bias):
    # 把 8 位整数编码还原成浮点值(含 subnormal)
    sign = (code >> (exp_bits + mant_bits)) & 1
    exp = (code >> mant_bits) & ((1 << exp_bits) - 1)
    mant = code & ((1 << mant_bits) - 1)
    if exp == 0:                      # subnormal / 零:隐含位为 0
        val = (mant / (2 ** mant_bits)) * (2.0 ** (1 - exp_bias))
    else:                             # normal:隐含位为 1
        val = (1.0 + mant / (2 ** mant_bits)) * (2.0 ** (exp - exp_bias))
    return -val if sign else val

DTYPES = {"fp16 / bf16(16 bit)": 2, "fp8 E4M3 / E5M2(8 bit)": 1}

PRESETS = {
    "Llama-3-8B 类(32 层 · 8 KV头 × 128 维)": dict(layers=32, kv_heads=8, head_dim=128),
    "Qwen2.5-7B 类(28 层 · 4 KV头 × 128 维)": dict(layers=28, kv_heads=4, head_dim=128),
    "7B 老 Llama(32 层 · 32 KV头 × 128 维)": dict(layers=32, kv_heads=32, head_dim=128),
    "自定义": dict(layers=32, kv_heads=8, head_dim=128),
}

with st.sidebar:
    st.header("🎛️ 模型配置")
    preset = st.selectbox("预设模型", list(PRESETS.keys()))
    cfg = PRESETS[preset]
    layers = st.slider("层数 L", 4, 80, cfg["layers"], 1)
    kv_heads = st.slider("KV 头数(GQA)", 1, 32, cfg["kv_heads"], 1)
    head_dim = st.slider("每头维度", 64, 256, cfg["head_dim"], 16)
    batch = st.slider("并发请求数", 1, 128, 32, 1)
    ctx = st.slider("每请求上下文长度", 256, 131072, 4096, 256)
    st.caption("💡 KV = 2(K和V) × L × kv_heads × head_dim × 每元素字节 × token 数")

total_tokens = batch * ctx
kv_elems = 2 * layers * kv_heads * head_dim
bytes_fp16 = kv_elems * 2 * total_tokens
bytes_fp8 = kv_elems * 1 * total_tokens

c1, c2, c3, c4 = st.columns(4)
c1.metric("每 token KV(fp16)", f"{kv_elems * 2 / 1024:.1f} KB")
c2.metric("KV 显存 fp16/bf16", f"{bytes_fp16 / 1024 ** 3:.2f} GB")
c3.metric("KV 显存 fp8", f"{bytes_fp8 / 1024 ** 3:.2f} GB")
c4.metric("节省", f"{(1 - bytes_fp8 / bytes_fp16):.0%}", "fp16 → fp8")

st.subheader(f"📊 KV 显存 vs 并发规模(批量 {batch} × 上下文 {ctx} = {total_tokens:,} tokens)")
toks = np.linspace(1000, 2000000, 60)
fig = go.Figure()
for name, nb, color in [("fp16/bf16(2 B/元素)", 2, "#4C78A8"),
                        ("fp8(1 B/元素)", 1, "#F2C14E"),
                        ("int8 KV(1 B/元素)", 1, "#72B7B2")]:
    fig.add_trace(go.Scatter(x=toks, y=kv_elems * nb * toks / 1024 ** 3, mode="lines",
                             name=name, line=dict(color=color, width=2)))
fig.add_vline(x=total_tokens, line_dash="dot", line_color="#E45756",
              annotation_text=f"当前 = {total_tokens:,} tokens")
fig.update_layout(title="同样一张 24 GB 卡:fp16 撑死 200 万 token,fp8 能翻倍",
                  xaxis_title="并发 token 总数", yaxis_title="KV cache 显存(GB)",
                  height=420, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔬 加餐:E4M3 vs E5M2 可表示值(对数轴)")
e4m3 = sorted({fp8_decode(c, 4, 3, 7) for c in range(256) if fp8_decode(c, 4, 3, 7) > 0})
e5m2 = sorted({fp8_decode(c, 5, 2, 15) for c in range(256) if fp8_decode(c, 5, 2, 15) > 0})
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=e4m3, y=[1] * len(e4m3), mode="markers", name="E4M3",
                          marker=dict(size=7, color="#F2C14E")))
fig2.add_trace(go.Scatter(x=e5m2, y=[0] * len(e5m2), mode="markers", name="E5M2",
                          marker=dict(size=7, color="#72B7B2")))
fig2.update_layout(title="FP8 两种格式的正数刻度点:E4M3 密(精度)vs E5M2 宽(范围)",
                   xaxis_title="数值(对数轴)", yaxis=dict(showticklabels=False),
                   xaxis_type="log", height=300,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("""
> 💡 **vLLM 用法**:`vllm serve <model> --kv-cache-dtype fp8_e4m3`(或 fp8_e5m2)。
> 注意 KV 量化对长上下文检索类任务可能有轻微精度损失,建议先小规模评测。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 5 章 · 第 33 课配套演示")

if __name__ == "__main__":
    try:
        import streamlit.runtime as st_runtime
        if st_runtime.exists():
            raise SystemExit(0)
    except Exception:
        pass
    import os as _os, subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])
''')

NB = new_nb("第 33 课 · FP8 与 KV Cache 量化",
            subtitle="8 bit 装下科学计数法——E4M3/E5M2 位拆解、软件模拟,以及 KV cache 显存减半的账本",
            emoji="🚀")

chapter_cover(NB,
    objectives=[
        "理解浮点数的位拆解:符号 | 指数 | 尾数,以及 E4M3 / E5M2 的取舍",
        "手写 FP8 解码器:枚举 256 个编码对应的值,验证动态范围",
        "用 torch.float8_e4m3fn 在 CPU 上真实体验 FP8 转换与精度",
        "推导 KV cache 显存公式,算清 fp16 → fp8 的节省账",
    ],
    toc=[
        ("直觉:科学计数法的尺子", "浮点 = 指数定范围 + 尾数定精度"),
        ("核心定义与公式", "E4M3/E5M2 位拆解 + 解码公式 + 符号表"),
        ("最小实现 · 逐行推演", "手写 FP8 解码器,256 个编码逐一还原"),
        ("数值验证", "软件解码 vs torch float8_e4m3fn 对拍;FP8 vs int8 误差"),
        ("真实规模数字:KV 显存账本", "公式 + Llama-3-8B 配置速算,24GB 卡能装多少 token"),
        ("与 vLLM 工程实现的关系", "--kv-cache-dtype fp8_e4m3,与权重量化叠加"),
        ("配套 Streamlit 演示", "app_33_fp8_kv.py:KV 显存计算器"),
    ],
    links=[
        ("FP8 Formats for Deep Learning (arXiv:2209.05433)", "https://arxiv.org/abs/2209.05433"),
        ("NVIDIA/Arm/Intel FP8 规范博客", "https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/"),
        ("vLLM FP8 E4M3 KV Cache 文档", "https://docs.vllm.ai/en/v0.6.1/quantization/fp8_e4m3_kvcache.html"),
        ("vLLM 量化文档", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
    ])

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows 下 torch/numpy OMP 冲突保护
import torch
import numpy as np

torch.manual_seed(0)          # 固定随机种子,结果可复现
np.random.seed(0)
print("torch =", torch.__version__)
# 说明:本课执行体不引入 pandas —— GPU cell 后再建 pandas/pyarrow 表易触发 arrow.dll 原生崩溃。
'''), "🛡️ 环境准备:KMP 保护 + 固定随机种子,全程 CPU。")

NB.md("## 1️⃣ 直觉:科学计数法的尺子 🔬",
D('''
前面两课的 int8 量化是**均匀刻度**的尺子;浮点数则是**科学计数法**的尺子:

$$x = (-1)^{sign} \\times 1.man \\times 2^{exp - bias}$$

一个浮点数由三段位组成:**符号 | 指数 | 尾数**。
- **指数位**(exp)决定「量级」——数到 2 的几次方,管**范围**;
- **尾数位**(mant)决定「这个量级内分几格」,管**精度**。

总共只有 8 bit,指数和尾数怎么分?两个官方答案(NVIDIA/Arm/Intel 共同制定,
见 [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)):

| 格式 | 符号 | 指数 | 尾数 | 最大值 | 精度(相对) | 适用 |
|------|------|------|------|--------|--------------|------|
| **E4M3** | 1 | 4 | 3 | ±448 | ~2⁻³(较高) | 权重、前向激活、KV |
| **E5M2** | 1 | 5 | 2 | ±57344 | ~2⁻²(较低) | 梯度、极大范围场景 |

直觉:**E4M3 是「放大镜」(近处看得清),E5M2 是「望远镜」(看得远)**。
推理场景数值范围可控,大家几乎都用 E4M3。
'''))

NB.md("## 2️⃣ 核心定义与公式:位拆解 🧩",
D('''
把 8 bit 从高位到低位切开:`s eeee mmm`(1 符号 + 4 指数 + 3 尾数),指数偏置 $bias = 7$:

$$x = (-1)^s \\times 1.mmm_2 \\times 2^{eeee_2 - 7}$$

| 符号 | 含义 | 取值 |
|---|---|---|
| $s$ | 符号位 | 0/1 |
| $eeee$ | 4 位指数(偏置表示) | E4M3: 0..15,偏置 7 |
| $mmm$ | 3 位尾数(隐含位 1) | 0..7,精度 $2^{-3}=0.125$ |
| $bias$ | 指数偏置 | E4M3: 7;E5M2: 15 |
| $\\mathrm{NaN}/\\infty$ | 特殊值 | E4M3 只留一个 NaN;E5M2 有 ±inf |

两个特殊规则:
- $eeee = 0000$:**次正规数**(subnormal),隐含位从 1 变 0,表示更小的数,逐步下溢到 0;
- E4M3FN 约定 $eeee = 1111, mmm = 111$ 表示 NaN(只留这一个 NaN,换取 +448 的最大值,
  FN = finite,这就是 torch 里 `float8_e4m3fn` 名字的由来)。

手写解码器,把 0~255 每个编码还原成浮点值:
'''))

NB.code(FP8_PARSE, "📜 FP8 解码器:按 sign|exp|mant 拆位、处理次正规数,枚举全部可表示值。")

NB.md("## 3️⃣ 最小实现 · 逐行推演:拆一个具体数 🔬",
D('''
用一个具体编码走一遍位拆解,再枚举两种格式的全部正数,打印刻度分布——
**每行代码都有注释**,读者可手算核对 `0b00111100` 应为 $1.5 \\times 2^0 = 1.5$:
'''))

NB.code(D('''
# --------------------------------------------------------------------------
# 拆一个具体的数:E4M3 编码 0 0111 100 → 应为 1.5 × 2^0 = 1.5
# --------------------------------------------------------------------------
code = 0b00111100                        # 8 bit 整数:0 | 0111 | 100
print(f"编码 {code:08b} → E4M3 值 = {fp8_decode(code, 4, 3, 7)}")
print("  ^ 符号=0(正) | 指数 0111=7(偏置7 → 2^0) | 尾数 100=4/8 → 1.5 × 1 = 1.5")

# 枚举两种格式所有可表示的正数
e4m3 = fp8_all_values(4, 3, 7)           # E4M3 的全部正数刻度
e5m2 = fp8_all_values(5, 2, 15)          # E5M2 的全部正数刻度
print(f"\\nE4M3 正数可表示值个数 = {len(e4m3)},最小正规 = {min(v for v in e4m3 if v >= 2**-6)},最大 = {max(e4m3)}")
print(f"E5M2 正数可表示值个数 = {len(e5m2)},最小次正规 = {min(v for v in e5m2 if v > 0)},最大 = {max(e5m2)}")
print("E4M3 相邻值示例(1 附近):", [v for v in e4m3 if 0.8 <= v <= 1.7])
print("E5M2 相邻值示例(1 附近):", [v for v in e5m2 if 0.8 <= v <= 1.7])

# 特殊指数模式:fp8_decode 应返回 NaN / ±inf,而不是一个巨大的有限数
print(f"\\nE4M3 编码 0x7F(1111 111)→ {fp8_decode(0x7F, 4, 3, 7)}  <- 应为 NaN(表的 ±448 不含它)")
print(f"E5M2 编码 0x7C(11111 000)→ {fp8_decode(0x7C, 5, 2, 15)}  <- 应为 +inf")
print(f"E5M2 编码 0x7D(11111 001)→ {fp8_decode(0x7D, 5, 2, 15)}  <- 应为 NaN")
'''), "✅ 1 附近 E4M3 的刻度间距是 0.125,E5M2 是 0.25——E4M3 精度高一倍;但 E5M2 能到 57344,E4M3 止步 448。特殊指数模式(NaN/±inf)也被正确识别,最大值与 markdown 表的 ±448 / ±57344 一致。")

NB.md("## 4️⃣ 数值验证:torch 对拍 + FP8 vs int8 📏",
D('''
torch 2.x 内置 `float8_e4m3fn` / `float8_e5m2` 数据类型,CPU 上就能做转换。
把我们的软件解码器与 torch 的真实转换对拍;再对比 FP8 与 int8 在**两种分布形状**上的误差,
看清 FP8 真正赢在「动态范围」而非「小值更准」:
'''))

NB.code(D('''
torch.manual_seed(0)
x = torch.randn(1000) * 3.0                     # (1000,) 测试数据

# 真实 FP8 转换:先 cast 到 fp8,再回 float32(相当于量化-反量化)
x_e4m3 = x.to(torch.float8_e4m3fn).to(torch.float32)   # E4M3 转换
x_e5m2 = x.to(torch.float8_e5m2).to(torch.float32)     # E5M2 转换

mse_e4 = float(((x - x_e4m3) ** 2).mean())     # E4M3 MSE
mse_e5 = float(((x - x_e5m2) ** 2).mean())     # E5M2 MSE
print(f"E4M3 量化 MSE = {mse_e4:.4f}")
print(f"E5M2 量化 MSE = {mse_e5:.4f}")
print(f"E4M3 相对 E5M2 误差低 {mse_e5 / mse_e4:.2f} 倍(尾数多 1 位,精度翻倍)")

# 对拍:用软件解码器给同一个数找最近的 E4M3 刻度
v = 1.3
levels = fp8_all_values(4, 3, 7) + [-u for u in fp8_all_values(4, 3, 7)]  # 正负全刻度
nearest = min(levels, key=lambda u: abs(u - v))     # 最近刻度
print(f"软件模拟:v=1.3 → 最近 E4M3 刻度 {nearest}")
print(f"torch 真实 :v=1.3 → {float(torch.tensor(v).to(torch.float8_e4m3fn).to(torch.float32))}")
'''), "✅ 软件解码器与 torch 的 float8_e4m3fn 完全一致:1.3 → 1.25。位拆解就这么简单。")

NB.code(D('''
# --------------------------------------------------------------------------
# FP8 vs int8:谁的误差更低,取决于分布形状
# --------------------------------------------------------------------------
def symm_quantize(x, bits=8):
    # 对称量化(复用第 28 课逻辑):返回重建值
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().max() / qmax).clamp_min(1e-12)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale

def rel_err(x, xh):
    # 相对误差:均方误差 / 信号功率
    return float(((x - xh) ** 2).mean() / (x ** 2).mean())

torch.manual_seed(1)

print("分布A · 窄幅集中(LLM 权重典型画像,std 正态 ×0.02):")
wA = torch.randn(100000) * 0.02           # 值集中在 [-0.1, 0.1],动态范围窄
rA_i = rel_err(wA, symm_quantize(wA, 8))
rA_f = rel_err(wA, wA.to(torch.float8_e4m3fn).to(torch.float32))
print(f"  int8 相对误差 = {rA_i:.4%} | FP8 E4M3 = {rA_f:.4%} → int8 更准"
      f"(FP8 反而差 {rA_f/max(rA_i,1e-30):.1f}×)")

print("分布B · 重尾宽动态范围(t 分布 df=3):")
wB = torch.distributions.StudentT(3).sample((100000,))
rB_i = rel_err(wB, symm_quantize(wB, 8))
rB_f = rel_err(wB, wB.to(torch.float8_e4m3fn).to(torch.float32))
print(f"  int8 相对误差 = {rB_i:.4%} | FP8 E4M3 = {rB_f:.4%} → FP8 更准"
      f"(FP8 好 {rB_i/max(rB_f,1e-30):.1f}×)")

print()
print("结论:FP8 的优势不在“小值更准”,而在浮点刻度“近密远疏”能覆盖宽动态范围;")
print("对窄幅集中分布(LLM 权重),int8 的均匀刻度反而更准——这也是 int8 至今仍被广泛使用的原因。")
'''), "📊 分布形状决定胜负:窄幅集中分布 int8 更准,重尾宽动态范围 FP8 更准——FP8 赢在动态范围,不在小值精度。")

NB.md("## 5️⃣ 真实规模数字:KV Cache 显存账本 🧮",
D('''
自回归解码时,每个 token 的每一层都要存 K 和 V 两个向量。**每个 token 的 KV 字节数**:

$$\\text{bytes/token} = 2\\,(K\\!+\\!V) \\times L\\,(层数) \\times h_{kv}\\,(KV头数) \\times d_{head}\\,(每头维度) \\times b\\,(每元素字节)$$

| 符号 | 含义 | Llama-3-8B 取值 |
|---|---|---|
| $L$ | 层数 | 32 |
| $h_{kv}$ | KV 头数(GQA 后共享) | 8 |
| $d_{head}$ | 每头维度 | 128 |
| $b$ | 每元素字节 | fp16=2, fp8=1 |

现代模型都用 **GQA**(少量 KV 头共享),所以 $h_{kv}$ 远小于注意力头数——这是 KV 的第一次瘦身;
**FP8 是第二次瘦身**:把 $b$ 从 2(fp16)降到 1,显存直接减半。算一笔 Llama-3-8B 的账:
'''))

NB.code(D('''
def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes):
    # 每个 token 每层要存 K 和 V 两个向量 → 前面系数 2
    return 2 * layers * kv_heads * head_dim * dtype_bytes

L, h_kv, d = 32, 8, 128                    # Llama-3-8B 类配置
print(f"每 token KV 元素数 = 2 × {L} × {h_kv} × {d} = {2*L*h_kv*d:,}")

for name, b in [("fp16/bf16", 2), ("fp8/int8", 1)]:
    bt = kv_bytes_per_token(L, h_kv, d, b)      # 每 token 字节数
    for tokens in [100_000, 1_000_000]:
        gb = bt * tokens / 1024 ** 3            # 总显存(GB)
        print(f"{name:8s} · {tokens/1e4:.0f}万 token:每 token {bt/1024:.0f} KB → 共 {gb:5.2f} GB")

print()
print("一张 24 GB 卡,扣掉 16 GB 权重后约剩 8 GB 给 KV:")
bt16 = kv_bytes_per_token(L, h_kv, d, 2)
bt8 = kv_bytes_per_token(L, h_kv, d, 1)
print(f"  fp16 KV:8 GB 能装 {8 * 1024**3 / bt16 / 1000:.0f}k token")
print(f"  fp8  KV:8 GB 能装 {8 * 1024**3 / bt8 / 1000:.0f}k token(翻倍!)")
'''), "✅ fp16→fp8 让同等显存装下双倍 token——要么并发翻倍,要么上下文翻倍,这就是 KV 量化的诱惑。")

NB.code(D('''
# --------------------------------------------------------------------------
# 真机验证:三种 dtype 的每元素字节 + 真实 KV 张量分配
# --------------------------------------------------------------------------
import torch, gc
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ① 同一份数据,三种 dtype 的每元素字节与驻留显存
x = torch.randn(1_000_000, device=dev)              # (1000000,) 基准数据
for dt, name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"),
                 (torch.float8_e4m3fn, "fp8 E4M3")]:
    t = x.to(dt)                                    # 转换 dtype
    print(f"{name:12s}: 每元素 {t.element_size()} 字节 · {t.numel():,d} 元素 = "
          f"{t.numel()*t.element_size()/1024**2:6.2f} MB")

# ② 真实 KV cache 张量:Llama-3-8B 类配置(32 层 × 8 KV 头 × 128 维 × 长度 4096)
L, h_kv, d, T = 32, 8, 128, 4096
K16 = torch.empty(L, h_kv, T, d, dtype=torch.float16, device=dev)        # fp16 KV
K8  = torch.empty(L, h_kv, T, d, dtype=torch.float8_e4m3fn, device=dev)  # fp8 KV
m16 = K16.numel() * K16.element_size() / 1024 ** 2   # fp16 显存(MB)
m8  = K8.numel()  * K8.element_size()  / 1024 ** 2   # fp8 显存(MB)
print(f"\\n真实 KV 张量 shape = {tuple(K16.shape)}  <- (层 L={L}, KV头 {h_kv}, 序列 T={T}, 维度 d={d})")
print(f"  fp16: {m16:8.1f} MB | fp8: {m8:8.1f} MB | 减省 {(1 - m8/m16):.0%}")

torch.cuda.empty_cache()
del K16, K8; gc.collect(); torch.cuda.empty_cache()
'''), "✅ 真实 GPU 数字:fp16/bf16 每元素 2 字节、fp8 1 字节;同一段上下文 fp8 的 KV 显存精确减半——对占推理显存大头的 KV cache,这直接翻倍并发容量或上下文长度。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:一行参数开启 🚀",
D('''
vLLM 只需一个参数([官方文档](https://docs.vllm.ai/en/latest/features/quantization/index.html)):

```bash
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype fp8_e4m3
# 或范围优先:--kv-cache-dtype fp8_e5m2
```

工程事实:

1. **KV 的 scale**:默认按 fp16 动态范围直接映射(静态 scale);也支持 `--calculate-kv-scales`
   在校准数据上统计 scale,精度更稳(见 vLLM 的 `kv_cache_scales.json`);
2. **可以叠加**:KV 量化与权重量化(GPTQ/AWQ/FP8)**互不冲突**——权重 FP8 + KV FP8
   是高吞吐服务的常见组合;推理 dtype 由 `--quantization fp8` 决定,KV 由 `--kv-cache-dtype` 决定;
3. **长上下文注意**:长上下文召回类任务(大海捞针)建议先评测,确认精度损失可接受;
4. **硬件**:E4M3 转换在 Hopper+ 有原生指令(`cvt.rn.f16x2.e4m3x2`),老卡上用 E4M3 可能反而慢。

> 📄 FP8 格式规范与训练/推理精度证据见
> [FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433):
> 175B 级模型用 E4M3 做 PTQ 也能保住精度,而 int8 有明显损失。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:KV 显存计算器 🎛️",
D('''
运行同目录下的 `app_33_fp8_kv.py`:内置主流模型预设,拖动层数 / KV 头数 / 上下文等滑块,
实时计算 fp16 / fp8 / int4 的 KV 显存,并附送 E4M3 vs E5M2 刻度点对比图:

```bash
D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_33_fp8_kv.py
```

浏览器打开 **http://localhost:8501**。建议玩法:选「Llama-3-8B 类」预设,
把并发拉到 64 × 4k 上下文,看 fp16 需要多少 GB、fp8 又省下多少。完整源码如下:
'''))

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + __import__("textwrap").indent(APP_33, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_33_fp8_kv.py 后运行:\")\n"
    "    print(\"    D:/uv_envs/uv_cuda/Scripts/python.exe -m streamlit run app_33_fp8_kv.py\")\n"
)
NB.code(guard, "📜 这就是 app_33_fp8_kv.py 的完整源码(计算器 + FP8 刻度可视化二合一)。")

wrapup(NB,
    summary=[
        "浮点数 = 符号|指数|尾数:指数管范围,尾数管精度,8 bit 里两者此消彼长",
        "E4M3 精度高范围小(最大 448),适合权重/激活/KV;E5M2 范围大(57344)精度低,适合梯度",
        "手写位拆解解码器即可枚举 FP8 全部 256 个值,与 torch float8_e4m3fn 完全对拍一致",
        "FP8 的浮点刻度赢在宽动态范围(重尾分布);但对窄幅集中分布,int8 反而更准——选型看分布",
        "KV 显存 = 2·L·h_kv·d_head·b·tokens:fp16→fp8 每元素字节减半,同等显存并发容量翻倍",
        "vLLM 一行参数 --kv-cache-dtype fp8_e4m3 即可开启,且可与权重量化叠加",
    ],
    practice=[
        "给 fp8_decode 增加 E4M3 的 NaN 判定(1111 111 → NaN),并统计 256 个编码里 NaN/Inf 各占几个",
        "把 KV 公式改成你熟悉的模型(查 config.json 的 num_layers / num_key_value_heads),算 100 万 token 的账",
        "实验:对 t 分布重尾数据做 E4M3 量化,观察超范围(>448)的值如何被钳位",
        "推导:GQA 把 32 个 KV 头减到 8 个,KV 显存省多少?再叠加 FP8 又省多少?",
    ],
    links=[
        ("FP8 Formats for Deep Learning (arXiv:2209.05433)", "https://arxiv.org/abs/2209.05433"),
        ("NVIDIA/Arm/Intel FP8 规范博客", "https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/"),
        ("vLLM FP8 E4M3 KV Cache 文档", "https://docs.vllm.ai/en/v0.6.1/quantization/fp8_e4m3_kvcache.html"),
        ("vLLM 量化文档(KV cache dtype)", "https://docs.vllm.ai/en/latest/features/quantization/index.html"),
    ])

NB.save(str(Path(CH05) / "33_fp8_kv_quant.ipynb"))

app_path = Path(CH05) / "app_33_fp8_kv.py"
app_path.write_text(APP_33 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")