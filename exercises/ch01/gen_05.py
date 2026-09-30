# -*- coding: utf-8 -*-
"""生成第 05 课 notebook:Prefill vs Decode(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:传纸条直觉 -> prefill/decode 定义与 FLOPs 公式 -> 7B 账本 -> 真实 GPU 微基准 -> compute/memory-bound 分析 -> vLLM 关联
2. 每一行代码都有 inline 注释
3. 每个结果打印数值与单位,标注含义
4. 论文支撑:Kaplan scaling laws(arXiv:2001.08361, 2N/token)、Korthikanti(arXiv:2205.05198)、Orca(arXiv:2208.14217)、PagedAttention(arXiv:2309.06180)
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_05_prefill_demo.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 05 课 · Prefill vs Decode:两阶段推理",
    subtitle="一次处理整个提示词 vs 逐字生成 —— FLOPs 公式推导 + 真实 GPU 微基准 + compute/memory-bound",
    emoji="⚡", chapter="第 1 章 · LLM 推理基础",
)

chapter_cover(
    nb,
    objectives=[
        "理解推理天然分成两阶段:prefill(一次性并行处理提示词)与 decode(逐 token 串行生成)",
        "推导推理 FLOPs 经验公式:每 token 前向约 2N 次浮点运算(Kaplan et al.),总 FLOPs ≈ 2N(L+K)",
        "在真实 GPU 上跑一个小 GPT,实测两阶段的耗时与吞吐,验证公式与现实的差距",
        "用 arithmetic intensity(算术强度)解释为什么 prefill 是 compute-bound、decode 是 memory-bound",
        "理解连续批处理为何能救 decode:批大小 → 吞吐 的真实曲线",
        "建立与 vLLM(prefill/decode 调度、chunked prefill、disaggregation)的联系",
    ],
    toc=[
        ("直觉与动机", "传纸条 vs 写日记"),
        ("核心定义与公式", "两阶段定义、FLOPs 公式、算术强度"),
        ("FLOPs 账本", "7B 模型的理论计算量"),
        ("真实 GPU 实测", "小 GPT 上 prefill vs decode 耗时/吞吐"),
        ("为什么 decode 慢", "compute-bound vs memory-bound(roofline)"),
        ("怎么救 decode", "批处理让吞吐大涨,预告连续批处理"),
        ("与 vLLM 的关系", "调度、chunked prefill、disaggregation"),
        ("Streamlit 动态演示", "实时算账本"),
    ],
    links=[
        ("Kaplan et al.: Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"),
        ("Korthikanti et al.: Reducing Activation Recomputation (FLOPs 分解)", "https://arxiv.org/abs/2205.05198"),
        ("Orca: A Distributed Serving System (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉与动机:传纸条 vs 写日记\n\n"
    "用户发来一段提示词,模型要生成回复。这整段过程被拆成两个**性格迥异**的阶段:\n\n"
    "- **prefill(预填充)**:一次性读完整段提示词。所有 token 一到位就能**并行**算注意力,"
    "像老师扫一眼全班名单,瞬间记住所有人;\n"
    "- **decode(解码)**:拿到提示词后,**一个 token 一个 token** 地生成,"
    "像写日记,写完一个字才能写下一个。\n\n"
    "这个两阶段不是设计者的选择,而是上一课的自回归本质**必然导致**的:"
    "训练时可以一次并行预测整句的每个位置(teacher forcing),但推理时第 $t$ 个 token 必须等前 $t-1$ 个先生成。\n\n"
    "> 📄 两阶段的不同瓶颈(compute-bound vs memory-bound)是当今推理系统所有优化"
    "(KV Cache、连续批处理、prefill/decode 分离)的出发点。"
    "参见 [Orca, OSDI'22](https://arxiv.org/abs/2208.14217) 与 "
    "[PagedAttention, SOSP'23](https://arxiv.org/abs/2309.06180)。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式\n\n"
    "**定义(prefill / decode)** 记提示词长 $L$ 个 token、生成长 $K$ 个 token:\n\n"
    "- **prefill**:把 $L$ 个 token **一次并行**前向,得到首个 token 的 logits。耗时决定 **TTFT**;\n"
    "- **decode**:循环 $K$ 次,每次只前向最新 1 个 token(用 KV Cache 复用历史)。"
    "每次耗时的中位数即 **ITL / TPOT**。\n\n"
    "**FLOPs 经验公式** 推理(仅前向)每处理一个 token 约需 $2N$ 次浮点运算,"
    "其中 $N$ 为参数量。直觉:每个参数在矩阵乘里都做一次乘、一次加。"
    "这个口径来自 scaling-law 文献([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)),"
    "更精细的分解见 [Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)。于是:\n\n"
    "$$ \\text{FLOPs}_{\\text{prefill}} = 2NL, \\qquad "
    "\\text{FLOPs}_{\\text{decode}} = 2NK, \\qquad "
    "\\text{FLOPs}_{\\text{total}} = 2N(L+K) $$\n\n"
    "**算术强度(arithmetic intensity, AI)** 定义运算量/搬移字节数:"
    "$\\text{AI} = \\text{FLOPs} / \\text{bytes}$。GPU 有峰值算力 $\\pi$(FLOP/s)与峰值带宽 $\\beta$(B/s),"
    "交叉点 $\\text{AI}^* = \\pi/\\beta$:AI 大于交叉点是 **compute-bound**,否则 **memory-bound**。\n\n"
    "| 符号 | 含义 | 取值 |\n"
    "|---|---|---|\n"
    "| $N$ | 模型参数量 | 7B / 小 GPT ~17M |\n"
    "| $L$ | 提示词长度 | 128~2048 |\n"
    "| $K$ | 生成长度 | 128~2048 |\n"
    "| $\\pi$ | GPU 峰值算力 | A100 ~312 TFLOPS(fp16) |\n"
    "| $\\beta$ | GPU 内存带宽 | A100 ~2TB/s |\n"
    "| $\\text{AI}^*$ | 交叉点 | ~150 FLOP/Byte |"
)

# =====================================================================
# 第 3 节 · FLOPs 账本
# =====================================================================
nb.md(
    "## 3. FLOPs 账本:7B 模型的理论计算量\n\n"
    "先把公式写成函数,代入真实规模算一笔账。注意一个反直觉结论:"
    "**两阶段的计算量只取决于 token 总数,与阶段无关**——瓶颈不在算多少,而在怎么算。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Windows 下 OpenMP 库重复加载报错

def flops_prefill(N, L):
    """prefill 计算量:一次并行处理 L 个 token,每 token ~2N FLOPs。"""
    return 2 * N * L                                    # 2N × L

def flops_decode(N, K):
    """decode 总计算量:K 步,每步 1 个 token,每 token ~2N FLOPs。"""
    return 2 * N * K                                    # 2N × K

def flops_total(N, L, K):
    """总计算量:与阶段无关,只取决于 token 总数 L+K。"""
    return 2 * N * (L + K)                              # 2N(L+K)

N = 7e9                                                 # 7B 模型参数量
L, K = 128, 128                                         # 提示词 128,生成 128
print(f"7B 模型,提示词 {L} 个,生成 {K} 个:")
print(f"  prefill FLOPs = 2N×L  = {flops_prefill(N, L)/1e12:.2f} TFLOPs")
print(f"  decode  FLOPs = 2N×K  = {flops_decode(N, K)/1e12:.2f} TFLOPs")
print(f"  总 FLOPs        = 2N×(L+K) = {flops_total(N, L, K)/1e12:.2f} TFLOPs")
print()
print("关键:token 总数相同时,prefill 与 decode 的『算的量』几乎一样。")
print("差距只在:prefill 一次算完(并行),decode 拆成 K 次(串行)。")''',
    "✅ **账本结论**:当 L≈K 时两阶段 FLOPs 相等——瓶颈不在『算多少』,而在『怎么算』。"
    "下面用真实 GPU 验证『怎么算』带来的差距。",
)

# =====================================================================
# 第 4 节 · 真实 GPU 实测
# =====================================================================
nb.md(
    "## 4. 真实 GPU 实测:小 GPT 上的两阶段\n\n"
    "公式是理论,现在**在同一张 GPU 上跑一个小 GPT** 验证两件事:\n"
    "同样的 token 数,decode 到底比 prefill 慢多少?吞吐差多少?\n\n"
    "复用跨章共享库 `vllm_real`(内含 TinyGPT 与微基准),它自动探测 CUDA/CPU 并在 GPU 上实测。"
)

nb.code(
    '''# 导入共享库:路径指向 exercises 根目录
import sys
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")  # 模块搜索路径
from vllm_real import bench_prefill_decode, cuda_info   # 两阶段微基准 + 设备信息

L = 256                                                 # 一段提示词 256 个 token
bench = bench_prefill_decode(d=256, layers=8, vocab_size=1024, L=L, steps=64, reps=7)  # 跑微基准
print("设备:", bench["device"], "· 参数量 ~%.1f M" % (bench["params"] / 1e6))
print(f"prefill(一次并行 {L} token): {bench['prefill_ms']:.2f} ms  ->  {bench['prefill_tok_per_s']/1e3:.1f} k tok/s")
print(f"decode (逐 token 共 {L} 步): {bench['decode_total_ms']:.2f} ms  ->  {bench['decode_tok_per_s']/1e3:.1f} k tok/s")
print(f"decode / prefill 耗时比      : {bench['ratio']:.1f}×")''',
    "🚀 **真实 GPU 数字**。注意:prefill 把 L 个 token **一起并行**;"
    "decode 逐字来,同样的 token 数却慢一个数量级——这就是『怎么算』的差距。",
)

nb.code(
    '''# 用 plotly 柱状图把两个数字画出来(总耗时 vs 单 token 耗时)
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"                # 静态渲染器
import plotly.graph_objects as go

names = ["prefill", "decode"]                            # 两个阶段
times = [bench["prefill_ms"], bench["decode_total_ms"]]  # 总耗时(ms)
per_token = [times[0] / L, times[1] / L]                 # 单 token 平均耗时(ms)

fig = go.Figure()                                        # 新建画布
fig.add_trace(go.Bar(x=names, y=times, name="总耗时 (ms)",
                     marker_color="#4C78A8", text=[f"{t:.1f}" for t in times], textposition="outside"))
fig.add_trace(go.Bar(x=names, y=per_token, name="单 token 耗时 (ms)",
                     marker_color="#E45756", text=[f"{p:.4f}" for p in per_token], textposition="outside"))
fig.update_layout(title=f"prefill vs decode 耗时(真实 GPU,L={L})",
                  yaxis_title="毫秒", barmode="group", height=360,
                  margin=dict(l=10, r=10, t=50, b=10))
fig.show()                                               # 展示''',
    "📊 **左柱与右柱指向同一结论**:decode 阶段明显更慢、更『贵』。",
)

# =====================================================================
# 第 5 节 · 为什么 decode 慢
# =====================================================================
nb.md(
    "## 5. 为什么 decode 慢:compute-bound vs memory-bound\n\n"
    "同样的 token 数,为什么 decode 慢这么多?用 **算术强度(AI)** 回答。\n\n"
    "**prefill** 输入 `(B, L, hidden)`,与权重 `(hidden, hidden)` 做的是**大矩阵乘**,"
    "FLOPs/字节 比值高,GPU 算力(compute)是瓶颈 → **compute-bound**;\n\n"
    "**decode** 输入每步只有 `(B, 1, hidden)`,是**向量×矩阵**,"
    "运算少、却要把全部权重(以及越来越长的 KV Cache)从 HBM 搬一遍,"
    "FLOPs/字节 比值极低,内存带宽(memory)是瓶颈 → **memory-bound**。\n\n"
    "| 阶段 | 每步输入 | 主要瓶颈 | 优化方向 |\n"
    "|---|---|---|---|\n"
    "| prefill | (B,L,hidden) 大矩阵 | compute-bound | 更好的注意力内核(FlashAttention) |\n"
    "| decode | (B,1,hidden) 向量×矩阵 | memory-bound | 减少搬移:量化、KV Cache、批处理 |"
)

nb.code(
    '''# 用数字感受算术强度:decode 的 AI 比 prefill 低几个数量级
pi_a100 = 312e12                                        # A100 fp16 峰值算力(FLOP/s)
beta_a100 = 2.0e12                                       # A100 HBM 带宽(B/s)
ai_star = pi_a100 / beta_a100                            # 交叉点(FLOP/Byte)
print(f"A100 交叉点 AI* = {ai_star:.0f} FLOP/Byte (AI 高于它=compute-bound,低于它=memory-bound)")

hidden = 8192                                            # 7B 级隐藏维
dtype_bytes = 2                                          # fp16 每参数 2 字节

# prefill:一次处理 L=2048 个 token,AI 高
L = 2048
flops_p = 2 * L * hidden * hidden                        # 一层矩阵乘 FLOPs
bytes_p = hidden * hidden * dtype_bytes                  # 权重只搬一次(每个 token 复用)
ai_p = flops_p / bytes_p                                 # prefill 算术强度
print(f"prefill (L={L}):  AI ≈ {ai_p:.0f} FLOP/Byte  > 交叉点 -> compute-bound")

# decode:每步只算 1 个 token,权重仍要全搬一遍,AI 低
flops_d = 2 * 1 * hidden * hidden                        # 1 个 token 的矩阵乘
ai_d = flops_d / bytes_p                                 # decode 算术强度
print(f"decode (1 token): AI ≈ {ai_d:.1f} FLOP/Byte  << 交叉点 -> memory-bound")
print(f"\\n差距: {ai_p / ai_d:.0f}× —— 这就是 decode 慢的结构性原因")''',
    "🔬 **量化验证**:prefill 的 AI 比 decode 高约 2048×,"
    "一侧贴算力墙、一侧贴带宽墙。所以给 decode 换更强的算力芯片没用,"
    "**必须减少字节搬移**(量化、KV Cache、批处理)。",
)

# =====================================================================
# 第 6 节 · 怎么救 decode
# =====================================================================
nb.md(
    "## 6. 怎么救 decode:批处理让吞吐大涨\n\n"
    "decode 慢的根源是「一次只算 1 个 token」,且权重只被 1 个 token 复用。"
    "解法自然浮现:**把很多条 decode 请求的单个 token 拼成更大的 batch 一起前向**——"
    "权重从 HBM 搬一次,被 batch 个 token 同时复用,带宽被摊薄,吞吐上升。\n\n"
    "这就是 vLLM 的**连续批处理(continuous batching)** 雏形([Orca, OSDI'22]"
    "(https://arxiv.org/abs/2208.14217));下面在真实 GPU 上测「批大小 → 吞吐」曲线。"
)

nb.code(
    '''# 真实 GPU:decode 阶段「批大小 → 吞吐」曲线
from vllm_real import bench_throughput_curve            # 吞吐曲线微基准

b_list, tps, mps = bench_throughput_curve(batch=(1, 4, 16, 64, 128), token_len=16, reps=5)  # 各批大小实测
print("batch | 每步耗时(ms) | 吞吐(tokens/s)")
for b, ms, tp in zip(b_list, mps, tps):                  # 逐档打印
    print(f"{b:5d} | {ms:8.3f}      | {tp:10.0f}")

import plotly.io as pio
pio.renderers.default = "plotly_mimetype"
import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=b_list, y=tps, mode="lines+markers",
                           line=dict(width=3, color="#4C78A8"), marker=dict(size=9)))
fig.update_layout(title="真实 GPU:decode 吞吐 vs 批大小",
                  xaxis_title="批大小 batch", yaxis_title="tokens/s",
                  height=340, margin=dict(l=10, r=10, t=50, b=10))
fig.show()                                               # 展示''',
    "💡 **观察**:batch 从 1 涨到 128,吞吐不是平均摊薄而是**大涨**——"
    "固定开销被摊薄、带宽被复用。这为第 06 课的玩具引擎与第 2 章的连续批处理埋下伏笔。",
)

# =====================================================================
# 第 7 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 7. 与 vLLM 的关系:把两阶段变成工程决策\n\n"
    "vLLM 围绕「两阶段瓶颈不同」做了一系列工程决策:\n\n"
    "1. **调度区分阶段**:`Scheduler` 把请求分为 prefill 队列与 decode 批次;"
    "decode 批次一旦满了就持续输出,prefill 新请求见缝插针;\n"
    "2. **chunked prefill**:把超长 prefill 切成小块,混进 decode 批次,"
    "避免『长 prefill 挡住所有 decode』(vLLM `--enable-chunked-prefill`,v1 默认开启);\n"
    "3. **prefill/decode 分离(disaggregation)**:把两个阶段放在不同的 GPU/服务上,"
    "各自针对自己的瓶颈优化硬件与调度;更进一步的 PP 流水线切分见 Sarathi/分块解码;\n"
    "4. **KV Cache 复用**:decode 的每步只算新 token,历史 K/V 全在缓存里(第 2 章主题)。\n\n"
    "> 📄 本课的核心一句话:prefill 是 compute-bound、decode 是 memory-bound,"
    "所以**给推理系统加算力对 decode 帮助有限,减字节搬移才是对症下药**。"
)

# =====================================================================
# 第 8 节 · Streamlit
# =====================================================================
nb.md(
    "## 8. 🖥️ Streamlit 动态演示:实时算账本\n\n"
    "运行 `app_05_prefill_demo.py`:拖动**模型规模**与**提示词/生成长度**滑杆,"
    "两阶段 FLOPs 实时刷新,并叠加**真实 GPU 测得的吞吐**。\n\n"
    "### 📜 App 完整源码(`app_05_prefill_demo.py`)"
)

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    + textwrap.indent(APP_CODE, "    ") +
    "\nelse:\n"
    "    print(\"💡 当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请把上方源码保存为 app_05_prefill_demo.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_05_prefill_demo.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_05_prefill_demo.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_05_prefill_demo.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,拖动滑杆观察 FLOPs 变化,点击『🚀 在 GPU 实测两阶段』跑真实数据。"
)

wrapup(
    nb,
    summary=[
        "推理分两阶段:prefill 一次并行处理整段提示词(决定 TTFT),decode 逐 token 串行生成(决定 ITL/TPOT)",
        "推理 FLOPs ≈ 2N(L+K):每 token 前向约 2N 次浮点运算,总计算量只取决于 token 总数(Kaplan et al.)",
        "真实 GPU 实测:相同 token 数下 decode 总耗时比 prefill 慢一个数量级,吞吐差一个数量级",
        "根因是算术强度:prefill 大矩阵乘 AI 高(compute-bound),decode 向量×矩阵 AI 低(memory-bound)",
        "救 decode 的关键是减少字节搬移:批处理摊薄权重搬运、KV Cache 复用、量化压缩",
        "vLLM 把两阶段变成工程决策:连续批处理、chunked prefill、prefill/decode 分离(disaggregation)",
    ],
    practice=[
        "改 L=512/1024 重跑 bench_prefill_decode,观察 prefill 耗时与 L 是否近似线性(attn 部分除外)",
        "把 bench_throughput_curve 的 batch 换成 (1,2,3,5,8),看小批阶段吞吐是否近似线性增长",
        "用本课 AI 公式算 RTX 5060(AI* 按 specs)的交叉点,判断 prefill/decode 各贴哪堵墙",
        "思考题:为什么 batch 增大吞吐会饱和?提示:每步的 KV Cache 读取量随 batch 线性增长",
    ],
    links=[
        ("Kaplan et al.: Scaling Laws", "https://arxiv.org/abs/2001.08361"),
        ("Korthikanti et al.: Activation Recomputation", "https://arxiv.org/abs/2205.05198"),
        ("Orca: Continuous Batching (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch01\05_prefill_vs_decode.ipynb")