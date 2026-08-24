# -*- coding: utf-8 -*-
"""生成 50_metrics_monitor.ipynb 与 app_50_monitor.py —— 指标监控 / 生产运维(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:直觉动机 -> 指标定义与公式(符号表) -> 最小实现逐行推演 -> 数值/负载验证 ->
   真实规模数字 -> Prometheus 命名规范与 vLLM 工程关联 -> 小结+练习+延伸阅读
2. 每一行可执行代码都有 inline 注释;每个数组/张量打印 shape 并标注维度含义
3. 修复审核发现的问题:
   * 旧版模拟指标名含冒号 `vllm:time_to_first_token_seconds`——Prometheus 指标名允许字母数字_
     下划线(冒号虽在正则里合法但**保留给 recording rule**,生产最佳实践禁用);
     新版统一改为下划线 `vllm_time_to_first_token_seconds`(并在文中讲清 vLLM 真实前缀 `vllm:` 的来龙去脉)
4. 论文支撑:真实引用 vLLM Metrics 文档、Prometheus 命名规范、Grafana
5. 保留 streamlit app 直跑入口(%%writefile 同步 + st.runtime.exists() 保护)

调研来源(websearch):
- vLLM Metrics: https://docs.vllm.ai/en/stable/design/metrics 与 usage/metrics
- Prometheus Metric and label naming: https://prometheus.io/docs/practices/naming
- Prometheus Data Model(冒号保留给 recording rule): https://prometheus.io/docs/concepts/data_model
- Grafana: https://grafana.com
"""
from pathlib import Path
from helpers import D, new_nb, chapter_cover, wrapup, CH08

APP_FILE = "app_50_monitor.py"

# =====================================================================
# Streamlit app 源码(与 notebook 共用 simulate,指标名用下划线规范)
# =====================================================================
APP_50 = D('''
# -*- coding: utf-8 -*-
# app_50_monitor.py — 性能指标与监控仪表盘
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="50 · 指标监控", layout="wide")
st.title("第 50 课 · 性能指标与监控:TTFT / TPOT / 吞吐 / Prometheus")

st.markdown("""
衡量一个 LLM 推理服务「快不快」,主要看三类指标:**首字延迟(TTFT)**、**每字延迟(TPOT)** 与
**吞吐(throughput)**。下方用**并发滑杆**模拟不同负载,实时计算这些指标并画成仪表盘。
这对应 vLLM 通过 Prometheus 暴露给运维监控的核心指标(概念一致)。
""")

MODELS = {
    "小模型 (1.5B)":  dict(ttft=0.08, tpot=0.012),
    "中模型 (7B)":    dict(ttft=0.25, tpot=0.035),
    "大模型 (14B)":   dict(ttft=0.50, tpot=0.070),
}

with st.sidebar:
    st.header("负载与模型")
    model = st.selectbox("模型规模", list(MODELS.keys()))
    concurrency = st.slider("并发请求数 (concurrency)", 1, 64, 8, 1,
                            help="同一时刻有多少个请求在服务端并行处理")
    out_tokens = st.slider("平均输出 token 数/请求", 16, 1024, 256, 16)
    alpha = st.slider("争抢系数(并发越高越慢)", 0.0, 0.5, 0.20, 0.01)
    st.caption("并发越高,共享 GPU 与 KV cache 的争抢越严重,单请求延迟越高。")

cfg = MODELS[model]

def simulate(c, ttft_base, tpot_base, out_tokens, alpha):
    """给定并发 c,模拟出 TTFT / TPOT / E2E / 吞吐 四项指标。"""
    ttft = ttft_base * (1 + alpha * (c - 1))        # TTFT 随争抢线性上升
    tpot = tpot_base * (1 + alpha * (c - 1) * 0.5)  # TPOT 上升更缓(系数减半)
    e2e = ttft + out_tokens * tpot                  # 端到端 = TTFT + 输出数×TPOT
    throughput = c * out_tokens / e2e               # 系统吞吐 tokens/s
    return ttft, tpot, e2e, throughput

ttft, tpot, e2e, throughput = simulate(concurrency, cfg["ttft"], cfg["tpot"], out_tokens, alpha)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("TTFT(首字延迟)", f"{ttft*1000:.0f} ms", help="Time To First Token:首个 token 前的时间")
c2.metric("TPOT(每字延迟)", f"{tpot*1000:.1f} ms", help="Time Per Output Token:每输出一个 token 的时间")
c3.metric("E2E(端到端)", f"{e2e:.2f} s", help="从发出请求到收完整段回答")
c4.metric("吞吐", f"{throughput:.0f} tokens/s", help="单位时间整个服务生成的 token 数")
c5.metric("并发", f"{concurrency} req", help="当前模拟负载")
st.caption(f"{model} · 平均输出 {out_tokens} token/请求 · 争抢系数 α={alpha:.2f}")

st.subheader("吞吐与延迟随并发变化")
cs = list(range(1, 65))
ts = [simulate(c, cfg["ttft"], cfg["tpot"], out_tokens, alpha) for c in cs]
e2es = [t[2] for t in ts]
thrs = [t[3] for t in ts]
fig = go.Figure()
fig.add_trace(go.Scatter(x=cs, y=thrs, mode="lines+markers", name="吞吐 (tokens/s)",
                         line=dict(width=3, color="#4C78A8"), yaxis="y"))
fig.add_trace(go.Scatter(x=cs, y=e2es, mode="lines", name="E2E 延迟 (s)",
                         line=dict(width=2, dash="dot", color="#E45756"), yaxis="y2"))
fig.add_vline(x=concurrency, line_dash="dash", line_color="#72B7B2",
              annotation_text=f"当前并发 {concurrency}", annotation_position="top")
fig.update_layout(title="负载扫描:吞吐 vs 端到端延迟(注意吞吐的饱和点)",
                  xaxis_title="并发请求数", height=440,
                  yaxis=dict(title="吞吐 (tokens/s)"),
                  yaxis2=dict(title="E2E 延迟 (s)", overlaying="y", side="right"),
                  legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("观察:并发低时吞吐随并发近线性上涨;超过拐点后延迟暴涨、吞吐趋于饱和——这就是运维要找的「甜蜜点」。")

st.subheader("E2E 延迟组成:TTFT + TPOT×输出数")
t_list = []
for c in cs:
    a, b, e, _ = simulate(c, cfg["ttft"], cfg["tpot"], out_tokens, alpha)
    t_list.append((a, b * out_tokens, e))
ttft_comp = [t[0] for t in t_list]
decode_comp = [t[1] for t in t_list]
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=cs, y=ttft_comp, name="TTFT(预填充)", marker_color="#72B7B2"))
fig2.add_trace(go.Bar(x=cs, y=decode_comp, name="解码(TPOT×输出)", marker_color="#4C78A8"))
fig2.update_layout(barmode="stack", title="E2E 延迟组成随并发变化",
                   xaxis_title="并发请求数", yaxis_title="秒", height=400,
                   legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("长回答时解码部分(TPOT×输出数)占 E2E 大头;短回答时 TTFT 占比上升。所以「快不快」要分场景看。")

st.markdown("""
> **vLLM 的监控**:真实 vLLM 服务通过 **Prometheus** 在 `/metrics` 暴露指标(如 `time_to_first_token_seconds`、
> `num_requests_running`),配合 Grafana 画成仪表盘。
> 注意:真实 vLLM 指标名前缀是 `vllm:`(带冒号,这是 vLLM 的一个已知特例);而**我们自己写的模拟指标**
> 遵循 Prometheus 最佳实践用下划线(`vllm_time_to_first_token_seconds`),因为冒号被保留给 recording rule。
> 见 [vLLM Metrics](https://docs.vllm.ai/en/stable/design/metrics) 与
> [Prometheus Naming](https://prometheus.io/docs/practices/naming)。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 8 章 · 第 50 课配套演示")
''')

# =====================================================================
# 组装 notebook
# =====================================================================
NB = new_nb(
    "第 50 课 · 性能指标与监控:TTFT / TPOT / 吞吐 / Prometheus",
    subtitle="精确定义首字延迟、每字延迟、端到端延迟与吞吐,用 torch 模拟推理计时,并讲清 vLLM 的 Prometheus 监控与命名规范",
    emoji="📊",
)

chapter_cover(NB,
    objectives=[
        "给出 TTFT / TPOT / E2E latency / throughput 的精确定义与公式(含符号表)",
        "用 torch(CPU)模拟 prefill + decode,亲手计时算出这些指标",
        "扫描并发负载,理解吞吐饱和与延迟暴涨的「甜蜜点」",
        "掌握 Prometheus 指标类型(Counter / Gauge / Histogram)与命名规范",
        "修复冒号指标名,用下划线命名模拟指标,并讲清 vLLM 真实 `vllm:` 前缀的来龙去脉",
        "认识 vLLM 通过 /metrics 暴露的指标与 Grafana 仪表盘",
    ],
    toc=[
        ("直觉:餐厅出菜与外卖", "首菜慢、后面快,对应 TTFT 与 TPOT"),
        ("四个核心指标定义", "TTFT / TPOT / E2E / throughput 公式 + 符号表"),
        ("最小实现:torch 模拟计时", "prefill + decode 逐行推演,算出真实数字"),
        ("数值验证:负载扫描", "并发上去,吞吐为何先涨后平(甜蜜点)"),
        ("Prometheus 指标类型与命名", "Counter / Gauge / Histogram + 冒号问题修复"),
        ("真实规模数字与 vLLM 工程", "真实 vLLM 指标清单 + Grafana 仪表盘"),
        ("配套 Streamlit 演示", "app_50_monitor.py:并发滑杆实时看仪表盘"),
    ],
    links=[
        ("vLLM Metrics 文档", "https://docs.vllm.ai/en/stable/design/metrics"),
        ("Prometheus Metric and label naming", "https://prometheus.io/docs/practices/naming"),
        ("Prometheus Data Model", "https://prometheus.io/docs/concepts/data_model"),
        ("Grafana 官网", "https://grafana.com/"),
    ])

# ---------------------------------------------------------------- 第 1 节:直觉
NB.md("## 1. 直觉与动机:为什么推理服务需要监控\n\n"
      "你点一桌菜:第一道菜上得慢(备料、起锅),后面的菜却接二连三地快。这正是 LLM 推理的节奏——\n"
      "**首字延迟慢(TTFT),后续每个字快(TPOT)**。理解了这两者,就看懂了大模型推理延迟的全部结构。\n\n"
      "再想**外卖平台**:同时有很多顾客点单,平台要在「每个顾客尽快拿到(低延迟)」与「单位时间送出最多"
      "(高吞吐)」之间平衡。并发一高,厨房就拥堵(争抢 GPU),每位顾客都变慢,但系统整体的吞吐**先涨、后趋于饱和**——\n"
      "这就是「甜蜜点(sweet spot)」。生产环境必须**用数字监控**来找到这个平衡点,"
      "而 vLLM 通过 **Prometheus** 暴露这些数字。")

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP 冲突防护
import math, time, torch, numpy as np                   # 数学 / 计时 / 张量 / 数值
torch.manual_seed(0)                                    # 固定 torch 随机种子,保证可复现
dev = "cpu"                                             # 本课计时统一 CPU,稳定可复现
print("torch :", torch.__version__)
print("device:", dev, "| threads =", torch.get_num_threads())
'''), "✅ 第一段代码:设置 KMP 保护、固定 seed、统一跑 CPU。")

# ---------------------------------------------------------------- 第 2 节:核心定义
NB.md("## 2. 核心指标定义与公式(含符号表)\n\n"
      "假设一个请求有 $P$ 个输入(prompt)token,生成 $S$ 个输出 token。核心指标:\n\n"
      "- **TTFT**(Time To First Token,首字延迟):从发出请求到收到第一个输出 token 的时间。\n"
      "  主要花在**预填充(prefill)**:一次性并行处理全部 $P$ 个 token;\n"
      "- **TPOT**(Time Per Output Token,每字延迟):生成每个输出 token 的平均时间。\n"
      "  主要花在**解码(decode)**:逐 token 自回归,每次只算 1 个新 token;\n"
      "- **E2E**(End-to-End,端到端延迟):拿到完整回答的总时间,\n"
      "  $$\\text{E2E} = \\text{TTFT} + S \\times \\text{TPOT}$$\n"
      "- **吞吐(throughput)**:单位时间内服务生成的 token 总数(tokens/s)。单请求时\n"
      "  $$\\text{throughput} = \\frac{P + S}{\\text{E2E}}$$\n\n"
      "**符号表**:\n\n"
      "| 符号 | 含义 | 单位 |\n"
      "|---|---|---|\n"
      "| $P$ | prompt 输入 token 数 | 个 |\n"
      "| $S$ | 输出 token 数 | 个 |\n"
      "| $\\text{TTFT}$ | 首字延迟 | 秒 |\n"
      "| $\\text{TPOT}$ | 每字延迟 | 秒/token |\n"
      "| $\\text{E2E}$ | 端到端延迟 | 秒 |\n"
      "| $c$ | 并发请求数 | 个 |\n"
      "| $\\alpha$ | 争抢系数 | 无量纲 |\n\n"
      "直觉:**TTFT 管「第一印象」,TPOT 管「连续体验」,吞吐管「系统产能」**,三者经常互相牵制。")

# ---------------------------------------------------------------- 第 3 节:最小实现
NB.md("## 3. 最小实现:用 torch 模拟 prefill + decode 并计时\n\n"
      "真实 vLLM 里,prefill 一次性并行算 $P$ 个 token,decode 逐 token 自回归。"
      "我们用一个小线性模型(torch,CPU)模拟这两种计算,再用基准测试的标准做法\n"
      "(跑多遍取平均)算出四个指标。")

NB.code(D('''
class TinyLLM(torch.nn.Module):
    """迷你「语言模型」:两个线性层。prefill 处理一批 token,decode 每次只处理 1 个。"""
    def __init__(self, d=256):
        super().__init__()                            # 必须调用父类初始化
        self.w1 = torch.nn.Linear(d, d)               # 第一层:d -> d 线性变换
        self.w2 = torch.nn.Linear(d, d)               # 第二层:d -> d 线性变换
    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))        # 两层 + ReLU 激活的 MLP

model = TinyLLM(256).eval()                           # 实例化并切到推理模式
d = 256                                               # 隐藏维度

def prefill(n_tokens):
    """模拟 prefill:一次性并行处理 n_tokens 个 token。"""
    with torch.no_grad():                             # 推理不计算梯度
        model(torch.randn(1, n_tokens, d))            # 输入 (B=1, S=n_tokens, d)

def decode_step():
    """模拟 decode:每次只处理 1 个新 token(自回归的一步)。"""
    with torch.no_grad():                             # 推理不计算梯度
        model(torch.randn(1, 1, d))                   # 输入 (B=1, S=1, d)

def bench(fn, iters):
    """基准测试:跑 iters 遍取平均耗时(毫秒),减小单次计时噪声。"""
    t0 = time.perf_counter()                          # 开始计时(高精度时钟)
    for _ in range(iters):                            # 重复执行 iters 次
        fn()                                          # 调用被测函数
    return (time.perf_counter() - t0) / iters         # 返回单次平均耗时(秒)
'''), "🔧 计时工具 `bench` 先跑多遍取平均,减小噪声——这正是基准测试的标准做法。")

NB.code(D('''
P = 64          # 输入 prompt token 数
S = 128         # 输出 token 数
prefill_time = bench(lambda: prefill(P), iters=10)    # prefill 平均耗时(秒)
tpot = bench(decode_step, iters=50)                   # 单个输出 token 的平均时间 = TPOT(秒)

TTFT = prefill_time                                   # 首字延迟 ≈ prefill 时间
E2E = TTFT + S * tpot                                 # 端到端 = TTFT + S×TPOT
throughput = (P + S) / E2E                            # 单请求吞吐 tokens/s

print(f"prefill({P} token) = {prefill_time*1000:.3f} ms")
print(f"TPOT(decode 1 token) = {tpot*1000:.3f} ms")
print(f"TTFT = {TTFT*1000:.3f} ms")
print(f"E2E  = {E2E:.3f} s  (= TTFT + {S}×TPOT)")
print(f"throughput = {throughput:.1f} tokens/s")
'''), "🎯 看数字:TTFT 只有几毫秒(一次性并行算完),而 E2E 被 `S×TPOT` 主导——输出越多,解码越占大头。")

# ---------------------------------------------------------------- 第 4 节:负载扫描
NB.md("## 4. 数值验证:负载扫描与吞吐饱和点\n\n"
      "单请求看「延迟」,但生产环境是**很多请求并发**。并发 $c$ 个请求时,它们争抢同一块 GPU,\n"
      "每个请求的延迟会上升(用争抢系数 $\\alpha$ 建模: $\\text{TTFT}(c)=\\text{TTFT}_0(1+\\alpha(c-1))$),\n"
      "而系统吞吐 $\\text{throughput}(c) = \\frac{c \\cdot S}{\\text{E2E}(c)}$。扫一遍 $c$,观察「先涨后平」:\n\n"
      "> 这个「先涨后平」的曲线形态,正是真实 vLLM 负载测试里吞吐-并发关系的简化模型——\n"
      "> 拐点之前的区域就是运维要维持的「甜蜜点」。")

NB.code(D('''
base_ttft = prefill_time                             # 并发=1 时的基础 TTFT
base_tpot = tpot                                     # 并发=1 时的基础 TPOT
alpha = 0.20                                         # 争抢系数(并发越高争抢越严重)

def simulate(c):
    """给定并发 c,模拟四项指标(与 app 完全一致)。"""
    ttft = base_ttft * (1 + alpha * (c - 1))         # TTFT 随争抢线性上升
    tp = base_tpot * (1 + alpha * (c - 1) * 0.5)     # TPOT 上升更缓(系数减半)
    e2e = ttft + S * tp                              # 端到端延迟
    return ttft, tp, e2e, c * S / e2e                # 返回 (TTFT, TPOT, E2E, 吞吐)

rows = []                                            # 收集每行结果
for c in [1, 2, 4, 8, 16, 32, 64]:                   # 扫描一组并发值
    ttft, tp, e2e, thr = simulate(c)                 # 计算四项指标
    rows.append(dict(c=c, ttft_ms=ttft*1000, e2e=e2e, thr=thr))   # 存一行
    print(f"并发 {c:3d}: TTFT={ttft*1000:6.1f}ms  E2E={e2e:7.3f}s  吞吐={thr:7.1f} tokens/s")
'''), "🚦 观察:并发从 1 升到 8,吞吐翻了好几倍;继续加并发,吞吐增长变缓(饱和),而 E2E 一路走高——"
     "甜蜜点就在增速放缓、延迟还可接受的地方。")

NB.code(D('''
# 数值验证吞吐的「饱和」:算相邻并发的吞吐增量,看到增量趋近 0
cs = [1, 2, 4, 8, 16, 32, 64]                        # 并发序列
thrs = [simulate(c)[3] for c in cs]                  # 各并发下的吞吐
print("并发      吞吐(tokens/s)    吞吐增量(相对上一档)")
for i, c in enumerate(cs):
    inc = thrs[i] - thrs[i-1] if i > 0 else 0.0      # 与上一档的吞吐差
    print(f"  {c:3d}      {thrs[i]:9.1f}        {inc:8.1f}")
print("结论:并发越高,吞吐增量越小(趋向饱和),而 E2E 延迟持续上升。")
'''), "✅ **数值验证**。吞吐增量随并发递减并趋近 0(饱和),而 E2E 单调上升——"
     "这就是为什么要监控并发,避免在拐点之后空耗资源。")

# ---------------------------------------------------------------- 第 5 节:Prometheus
NB.md("## 5. Prometheus 指标类型与命名规范\n\n"
      "生产环境不能只算一次,要**持续采集**。vLLM 在 `/metrics` 暴露一套 Prometheus 兼容指标,"
      "Prometheus 定时抓取,再交给 Grafana 画成仪表盘。Prometheus 有三类基本指标:\n\n"
      "| 类型 | 含义 | 例子 |\n"
      "|---|---|---|\n"
      "| **Counter** | 只增不减的累计值 | 累计生成 token 数 |\n"
      "| **Gauge** | 可升可降的瞬时值 | 当前正在运行的请求数 |\n"
      "| **Histogram** | 直方图:分桶计数,含 `_bucket` / `_sum` / `_count` | TTFT 延迟分布 |\n\n"
      "### ⚠️ 指标命名:冒号 vs 下划线\n\n"
      "Prometheus 数据模型规定指标名匹配正则 `[a-zA-Z_:][a-zA-Z0-9_:]*`——**冒号 `:` 语法上合法**,\n"
      "但官方明确**保留给 recording rule(预计算规则)使用**,普通指标应只用**字母、数字、下划线**。\n"
      "所以**我们自己写监控指标时应该用下划线**,例如:\n\n"
      "- `vllm_time_to_first_token_seconds`(不是 `vllm:time_to_first_token_seconds`)\n"
      "- `vllm_num_requests_running`\n\n"
      "> 📄 真实 vLLM 的指标名确实以 `vllm:` 前缀出现(如 `vllm:time_to_first_token_seconds`),\n"
      "> 这是 vLLM 的一个**已知特例**(见 [vLLM Metrics 文档](https://docs.vllm.ai/en/stable/design/metrics))——\n"
      "> 很多下游采集端会做归一化。本课的教学模拟指标遵循 Prometheus 最佳实践,统一用下划线。\n\n"
      "直方图的平均值可用 `sum / count` 计算——这就是监控面板上 TTFT/TPOT 折线的来源。")

NB.code(D('''
import re                                             # 正则表达式(解析指标文本)

# 一段「模拟的 Prometheus 指标文本」——注意:指标名已统一为下划线(修复冒号问题)
fake_metrics = (
    "# HELP vllm_time_to_first_token_seconds Distribution of Time to First Token (TTFT)\\n"
    "# TYPE vllm_time_to_first_token_seconds histogram\\n"
    "vllm_time_to_first_token_seconds_bucket{le=\\\"0.1\\\"} 12\\n"
    "vllm_time_to_first_token_seconds_bucket{le=\\\"0.5\\\"} 45\\n"
    "vllm_time_to_first_token_seconds_sum 18.7\\n"
    "vllm_time_to_first_token_seconds_count 50\\n"
    "# HELP vllm_time_per_output_token_seconds Distribution of TPOT\\n"
    "# TYPE vllm_time_per_output_token_seconds histogram\\n"
    "vllm_time_per_output_token_seconds_sum 4.2\\n"
    "vllm_time_per_output_token_seconds_count 400\\n"
    "vllm_num_requests_running 8\\n"
)

def parse_avg(text, name):
    """从指标文本里提取直方图平均值 = sum / count。name 需转义下划线。"""
    m = re.search(rf"{name}_sum\\s+([0-9.]+)", text)   # 匹配 _sum 行
    c = re.search(rf"{name}_count\\s+([0-9.]+)", text)  # 匹配 _count 行
    if m and c:                                        # 两者都在
        return float(m.group(1)) / float(c.group(1))   # 平均值 = sum / count
    return None                                        # 缺失则返回 None

# 用下划线命名解析(与文本一致)
ttft_avg = parse_avg(fake_metrics, "vllm_time_to_first_token_seconds")
tpot_avg = parse_avg(fake_metrics, "vllm_time_per_output_token_seconds")
running = re.search(r"vllm_num_requests_running\\s+([0-9.]+)", fake_metrics).group(1)
print(f"TTFT 平均 ≈ {ttft_avg*1000:.0f} ms")
print(f"TPOT 平均 ≈ {tpot_avg*1000:.0f} ms")
print(f"当前并发(running) = {running}")

# 校验:指标名里是否还残留冒号(修复验证)
import re as _re
bad = _re.findall(r"[a-z_]+:[a-z_]+", fake_metrics)    # 找含冒号的指标名
print("含冒号的指标名:", bad if bad else "无(已全部改为下划线 ✓)")
'''), "✅ **修复验证**。模拟指标全部用下划线命名;直方图平均值 = sum/count;"
     "最后一段正则确认文本里已无冒号指标名。")

# ---------------------------------------------------------------- 第 6 节:真实规模
NB.md("## 6. 真实规模数字:vLLM 的指标清单与 Grafana\n\n"
      "真实 vLLM 服务在 `/metrics` 暴露的指标与上面我们算的完全对应(见\n"
      "[vLLM Metrics 文档](https://docs.vllm.ai/en/stable/design/metrics))。常用清单:\n\n"
      "| 指标(真实 vLLM 前缀 `vllm:`) | 类型 | 对应本课概念 |\n"
      "|---|---|---|\n"
      "| `time_to_first_token_seconds` | Histogram | TTFT |\n"
      "| `inter_token_latency_seconds` / `time_per_output_token_seconds` | Histogram | TPOT |\n"
      "| `e2e_request_latency_seconds` | Histogram | E2E |\n"
      "| `num_requests_running` / `_waiting` | Gauge | 并发负载与排队 |\n"
      "| `prompt_tokens_total` / `generation_tokens_total` | Counter | 累计 token(吞吐原材料) |\n"
      "| `kv_cache_usage_perc` / `gpu_cache_usage_perc` | Gauge | 显存/KV cache 利用率 |\n\n"
      "**真实量级**(生产经验值):\n"
      "- 对话型应用 TTFT 目标 < 2s;GPU 推理 < 4s;重批处理 < 8s;\n"
      "- 长输出时 TPOT 主导 E2E,输出越长延迟占比越高;\n"
      "- 高并发下 KV cache 成为瓶颈,`kv_cache_usage_perc` 逼近 1 时开始 OOM/换出。\n\n"
      "> 📄 运维工作流:vLLM `/metrics` → Prometheus 抓取(如每秒)→ PromQL 算分位数"
      "(`histogram_quantile(0.95, rate(...))`)→ Grafana 画图 + Alertmanager 告警。")

NB.code(D('''
# 用真实量级代回公式,看「甜蜜点」与 KV cache 的关系
S_real = 512                    # 长回答:输出 512 token
ttft_real = 0.25                # 中模型 7B 的 TTFT ≈ 250ms
tpot_real = 0.035               # 7B 的 TPOT ≈ 35ms
alpha_real = 0.20               # 争抢系数

def throughput(c):
    # 并发 c 下系统吞吐(单位 tokens/s)
    e2e = ttft_real * (1 + alpha_real * (c - 1)) + S_real * tpot_real * (1 + alpha_real * (c - 1) * 0.5)
    return c * S_real / e2e     # 吞吐 = 并发×输出 / 端到端

# 找吞吐「增量显著变小」的拐点(甜蜜点附近)
prev = throughput(1)            # 并发=1 的吞吐
for c in range(2, 33):          # 从 2 扫到 32
    cur = throughput(c)         # 当前并发吞吐
    gain = cur - prev           # 吞吐增量
    if gain < 10:               # 增量跌破 10 tokens/s -> 接近饱和
        print(f"吞吐拐点 ≈ 并发 {c}:吞吐 {cur:.0f} tokens/s,增量仅 {gain:.1f} tokens/s")
        break
    prev = cur                  # 更新上一档吞吐
print(f"对比:并发 1 时吞吐 {throughput(1):.0f} tokens/s;并发 32 时吞吐 {throughput(32):.0f} tokens/s")
'''), "🚀 **真实量级**。代入 7B 模型参数,系统吞吐随并发先快速上涨、后接近饱和——"
     "运维用这组数字决定「并发上限」与「何时扩容」。")

# ---------------------------------------------------------------- 第 7 节:Streamlit app
NB.md("## 7. 配套 Streamlit 演示:并发滑杆,实时看仪表盘\n\n"
      "运行同目录下的 `app_50_monitor.py`,拖动 **并发请求数 / 平均输出 token 数**,\n"
      "实时刷新 TTFT / TPOT / E2E / 吞吐 四个指标卡,以及负载扫描图:\n\n"
      "```\n"
      "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_50_monitor.py\n"
      "```\n\n"
      "浏览器打开 **http://localhost:8501**。建议:把并发从 1 拖到 64,观察吞吐何时见顶、延迟何时暴涨;\n"
      "再换大模型,看甜蜜点如何左移。完整源码如下(与 app 文件一字不差):")

NB.code(f"%%writefile {APP_FILE}\n" + APP_50,
        "📜 这就是 `app_50_monitor.py` 的完整源码。notebook 与 app 共用同一套指标计算逻辑,保证讲解与演示一致。")

guard = (
    "try:\n"
    "    import streamlit as st\n"
    "    _IS_STREAMLIT = bool(st.runtime.exists())\n"
    "except Exception:\n"
    "    _IS_STREAMLIT = False\n\n"
    "if _IS_STREAMLIT:\n"
    "    # 在 streamlit 运行时,直接执行上面 %%writefile 写入的 app 源码\n"
    "    exec(open(\"" + APP_FILE + "\", encoding=\"utf-8\").read())\n"
    "else:\n"
    "    print(\"当前不是 streamlit 环境,跳过执行本 App。\")\n"
    "    print(\"    请运行: D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run " + APP_FILE + "\")\n"
)
NB.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行 app;在普通 notebook 中仅作展示并给出运行命令。")

# ---------------------------------------------------------------- 小结
wrapup(NB,
    summary=[
        "TTFT=首字延迟(主要花在 prefill),TPOT=每字延迟(主要花在 decode),E2E = TTFT + S×TPOT",
        "单请求吞吐 = (P+S)/E2E;并发越高,单请求延迟上升(争抢),但系统吞吐先涨后趋于饱和——存在甜蜜点",
        "Prometheus 三类指标:Counter(累计)/ Gauge(瞬时)/ Histogram(分桶),平均值 = sum/count",
        "指标命名规范:普通指标用下划线(冒号保留给 recording rule),模拟指标用 vllm_time_to_first_token_seconds",
        "真实 vLLM 在 /metrics 暴露 TTFT/TPOT/E2E/并发/KV cache 指标,配合 Grafana 画仪表盘并告警",
    ],
    practice=[
        "把 prefill 的 token 数 P 从 64 改成 512,观察 TTFT 如何变化,解释原因",
        "把争抢系数 alpha 从 0.2 改成 0.05,重跑负载扫描,看吞吐饱和点是否右移",
        "用正则把 fake_metrics 里的三个指标名都改成你自定义的前缀,验证解析仍正确",
        "写一个函数,给定目标 E2E 预算,反推出允许的最大并发数(运维容量规划)",
    ],
    links=[
        ("vLLM Metrics 文档", "https://docs.vllm.ai/en/stable/design/metrics"),
        ("Prometheus Metric and label naming", "https://prometheus.io/docs/practices/naming"),
        ("Prometheus Data Model", "https://prometheus.io/docs/concepts/data_model"),
        ("Grafana 官网", "https://grafana.com/"),
    ])

NB.save(str(Path(CH08) / "50_metrics_monitor.ipynb"))

app_path = Path(CH08) / APP_FILE
app_path.write_text(APP_50 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
