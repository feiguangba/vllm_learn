# -*- coding: utf-8 -*-
"""生成 27_profiling_metrics.ipynb(app_27_profiler.py 已存在,%%writefile 覆盖写入相同内容)"""
from helpers import D, MINI_GPT, chapter_cover, wrapup, new_nb, CH04, app_src, finalize


def app_cell(name):
    return "%%writefile " + name + "\n" + app_src(name)


NB = new_nb("第 27 课 · 性能剖析:tokens/s、TTFT、TPOT 与延迟分布",
            subtitle="给推理引擎装一台心电图仪:先看大指标,再钻到算子,找到「少数吃掉多数时间」的热点",
            emoji="📈")

chapter_cover(NB,
    objectives=[
        "掌握推理服务的四大指标定义与公式:吞吐 tokens/s、TTFT、TPOT、端到端延迟",
        "理解延迟不只看平均值:P50/P99 才能刻画「最差用户体验」",
        "用迷你 GPT 模拟一次完整 serving:prefill 一次(TTFT)+ decode 循环(每步计时)",
        "亲手从计时原始数据算出全部指标,不做只看结论的人",
        "用 torch.profiler 剖析前向,提取 top 算子表与 80/20 累计曲线",
        "跑通配套 App:性能仪表盘 + Top-N 算子浏览器",
    ],
    toc=[
        ("直觉:心电图仪", "先看心率/血压(大指标),再放大看哪段心律不齐(热点算子)"),
        ("指标定义与公式", "TTFT / TPOT / e2e / 吞吐,以及 P50/P99 的意义"),
        ("模拟一次 serving", "prefill + 20 步 decode,每步 perf_counter 计时"),
        ("从原始数据算指标", "TTFT、TPOT、P99、吞吐 —— 全部亲手算并打印"),
        ("可视化延迟分布", "plotly 时间线 + pyecharts 延迟直方"),
        ("torch.profiler 剖析", "top 算子表、80/20 累计曲线,保存给 App"),
        ("配套 App:📈 性能剖析仪表盘", "streamlit 交互演示"),
    ],
    links=[
        ("vLLM V1 使用文档(指标与日志)", "https://docs.vllm.ai/en/latest/design/v1/v1_usage.html"),
        ("torch.profiler 文档", "https://pytorch.org/docs/stable/profiler.html"),
        ("延迟数字的谎言(Tail Latency)", "https://brooker.co.za/blog/2015/03/21/latency.html"),
    ])

NB.md("## 1. 直觉:心电图仪 🩺",
D('''
医院里看病人,不会一上来就开 CT —— 先量**心率、血压、体温**几个大指标,异常了再逐项深挖。
推理服务的体检也是三步:

1. **大指标**:吞吐(tokens/s)、TTFT(首字延迟)、TPOT(每字延迟)—— 心率与血压;
2. **分布**:平均数会说谎,P99 才告诉你「最差的 1% 用户体验有多糟」—— 24 小时心电变异;
3. **热点**:用 profiler 钻到算子层,找到「少数吃掉多数时间」的地方 —— 定位到那一段心律不齐。

然后才是开药:第 24~26 课的 CUDA Graph / torch.compile,治的正是「启动开销病」与「kernel 太碎病」。
'''))

NB.md("## 2. 指标定义与公式 📐",
D('''
设一条请求:prompt 有 $S$ 个词元,生成 $N$ 个词元。

| 指标 | 定义 | 公式 | 关心什么 |
|---|---|---|---|
| **TTFT** | Time To First Token,首字延迟 | 从请求发出到第 1 个词元返回 | 用户「开始看到回答」的等待;由 prefill 决定 |
| **TPOT** | Time Per Output Token,每字延迟 | $\\frac{t_{\\text{e2e}} - t_{\\text{TTFT}}}{N-1}$ | 打字机滚动速度;由 decode 每步决定 |
| **e2e 延迟** | 端到端总延迟 | $t_{\\text{TTFT}} + (N-1)\\times \\text{TPOT}$ | 整体快不快 |
| **吞吐** | 每秒产出词元数 | $\\frac{\\text{生成词元总数}}{\\text{墙钟时间}}$ | 服务赚不赚钱;批越大越高 |

两个坑:

- **吞吐与延迟是跷跷板**:批越大吞吐越高,但单条请求的 TPOT 也越高 —— 服务要在两者间找平衡;
- **平均值会骗人**:延迟看 **P50(中位)/ P99(第 99 百分位)**。P99 差 = 每 100 步就有 1 步卡顿,
  对流式输出非常扎眼。
'''))

NB.code(D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Anaconda/torch OMP 库冲突(Windows)
import time, json
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
print("实验平台:CPU。指标的计算方法与平台无关 —— 换到 GPU 上公式一字不变,只是数字更好看。")
'''),
"🩺 先把体检台搭好:模型用第 22 课的迷你 GPT。")

NB.code(MINI_GPT,
"🏗️ 迷你 GPT(H=256、2 层、4 头、词表 5000)。prefill 吃 T=B×S,decode 每步 T=B。")

NB.md("## 3. 模拟一次 serving:prefill + 20 步 decode 🎬",
D('''
场景:B=4 条请求并发,prompt 各 32 词元(prefill 一次吃 T=128),然后 20 步 decode,
每步每条序列产出 1 个词元(T=4)。**每一步都用 `perf_counter` 计时** —— 这就是最朴素的剖析器。
'''))

NB.code(D('''
model = MiniGPT(vocab=5000, hidden=256, n_layers=2, n_heads=4).eval()
B, S, N_DECODE = 4, 32, 20

with torch.no_grad():                     # 热身:排除首次运行的懒初始化噪声
    for _ in range(3):
        model(torch.randint(0, 5000, (B * S,)), torch.cat([torch.arange(S)] * B))

wall_t0 = time.perf_counter()

# --- prefill:一次吃 4 条 × 32 词元,产出每条的第 1 个词元(TTFT 由此决定)---
ids = torch.randint(0, 5000, (B * S,))
pos = torch.cat([torch.arange(S) for _ in range(B)])
with torch.no_grad():
    t0 = time.perf_counter()
    logits = model(ids, pos)
    torch.argmax(logits[torch.arange(B - 1, B * S, S)], dim=-1)   # 每条序列取最后位置的 logits
    ttft_ms = (time.perf_counter() - t0) * 1e3

# --- decode:每步 T=B=4,每条序列各产 1 词元(TPOT 由此决定)---
cur_pos = torch.full((B,), S, dtype=torch.long)
step_lat_ms = []
with torch.no_grad():
    for _ in range(N_DECODE):
        ids = torch.randint(0, 5000, (B,))          # 简化:用随机词元代替真实采样回填
        t0 = time.perf_counter()
        logits = model(ids, cur_pos)
        torch.argmax(logits, dim=-1)
        step_lat_ms.append((time.perf_counter() - t0) * 1e3)
        cur_pos += 1
wall_s = time.perf_counter() - wall_t0

step_lat = np.array(step_lat_ms)
print(f"TTFT(prefill, 4×32 词元)= {ttft_ms:7.2f} ms")
print(f"decode 每步:均值 {step_lat.mean():.2f} ms | P50 {np.percentile(step_lat, 50):.2f} | P99 {np.percentile(step_lat, 99):.2f} ms")
print(f"墙钟总时间 = {wall_s*1e3:.2f} ms(含计时与词元准备的开销)")
'''),
"🎬 prefill(128 词元)比 decode 单步(4 词元)重一个数量级 —— 这就是 TTFT 与 TPOT 天然量级不同的原因。")

NB.code(D('''
# --- 从原始数据亲手算全部指标 ---
tpot_ms = step_lat.mean()
e2e_ms = ttft_ms + (N_DECODE - 1) * tpot_ms          # 端到端 = TTFT + (N-1)×TPOT
gen_tokens = B * N_DECODE                            # 4 条 × 每条 20 个新词元
throughput = gen_tokens / wall_s                     # 词元/秒(按墙钟,包含全部开销)
busy_pct = (ttft_ms + step_lat.sum()) / (wall_s * 1e3) * 100   # 计算忙占比

print(f"TPOT            = decode 均值               = {tpot_ms:8.3f} ms/词元")
print(f"e2e 延迟        = TTFT + (N-1)×TPOT          = {e2e_ms:8.2f} ms")
print(f"生成词元总数    = {gen_tokens}(B={B} × N={N_DECODE})")
print(f"吞吐            = 词元总数 / 墙钟           = {throughput:8.1f} tokens/s")
print(f"计算忙占比      = 计算时间 / 墙钟            = {busy_pct:8.1f} %")
print(f"校验:e2e vs 墙钟 = {e2e_ms:,.0f} vs {wall_s*1e3:,.0f} ms(差值 = 词元准备等非计算开销)")
'''),
"✅ 每个数字都从上一格的原始计时直接算出 —— 指标不神秘,就是「计时数据的四则运算」。")

NB.md("## 4. 可视化:延迟时间线与分布 📊",
D('''
两条曲线看 decode:时间线(哪一步抖动?)+ 累计视角(TTFT 先付、TPOT 慢慢滚)。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(y=step_lat, mode="lines+markers", name="decode 单步延迟"))
fig.add_hline(y=float(np.percentile(step_lat, 99)), line_dash="dash", line_color="red",
              annotation_text=f"P99 = {np.percentile(step_lat, 99):.2f} ms")
fig.add_hline(y=float(step_lat.mean()), line_dash="dot", line_color="green",
              annotation_text=f"均值 = {step_lat.mean():.2f} ms")
fig.update_layout(title="decode 每步延迟时间线(20 步)", height=360,
                  xaxis_title="decode 步数", yaxis_title="延迟(ms)")
fig
'''),
"📊 CPU 上波动明显(线程调度噪声);GPU + CUDA Graph 后这条线会平得多 —— 方差本身就是优化指标。")

NB.code(D('''
from pyecharts.charts import Bar
from pyecharts import options as opts

n_bins = 8
counts, edges = np.histogram(step_lat, bins=n_bins)
labels = [f"{edges[i]:.1f}~{edges[i+1]:.1f}" for i in range(n_bins)]
hist = (
    Bar()
    .add_xaxis(labels)
    .add_yaxis("步数", [int(c) for c in counts], label_opts=opts.LabelOpts(position="inside"))
    .set_global_opts(title_opts=opts.TitleOpts(title="decode 单步延迟分布(直方图)"),
                     xaxis_opts=opts.AxisOpts(name="延迟区间(ms)"), yaxis_opts=opts.AxisOpts(name="步数"))
)
hist.render_notebook()
'''),
"📈 分布右边的尾巴就是 P99 的来源 —— 平均值看不到它们,用户却每次都撞上。")

NB.md("## 5. torch.profiler:钻到算子层 🔬",
D('''
大指标异常时,下一步是问「时间花在哪些算子上了」。`torch.profiler` 记录每次 aten 调用,
`key_averages()` 聚合出「算子 × 调用次数 × 自身耗时 × 总耗时」表 —— 这就是热点清单。
我们剖析 decode 前向,再画 **80/20 累计曲线**:把算子按耗时降序排列,看前几个吃掉 80%。
'''))

NB.code(D('''
from torch.profiler import profile, ProfilerActivity

ids_d = torch.randint(0, 5000, (B,))
pos_d = torch.full((B,), S, dtype=torch.long)
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(10):                     # 剖析 10 次取平均,样本更稳
            model(ids_d, pos_d)

ka = prof.key_averages()
rows = []
for e in ka:
    if e.count > 0 and e.self_cpu_time_total > 0:
        rows.append({"name": e.key[:40], "calls": int(e.count),
                     "cpu_self": round(e.self_cpu_time_total / 1e3 / 10, 4),      # µs→ms,除以10次
                     "cpu_total": round(e.cpu_time_total / 1e3 / 10, 4)})
rows.sort(key=lambda r: r["cpu_total"], reverse=True)
ops_df = pd.DataFrame(rows)
print(f"共 {len(ops_df)} 种算子;Top-8(python_tracer 等框架开销已自然计入):")
ops_df.head(8)
'''),
"🔬 `cpu_self` 是算子自身耗时(不含子调用),`cpu_total` 含子调用 —— 排热点用 total,找融合候选看 self。")

NB.code(D('''
cum = ops_df["cpu_total"].cumsum() / ops_df["cpu_total"].sum()
n80 = int((cum < 0.8).sum()) + 1
print(f"80/20 检验:前 {n80} / {len(ops_df)} 种算子吃掉 80% 的总耗时")
print(f"最热算子:{ops_df.iloc[0]['name']}(占总耗时 {ops_df.iloc[0]['cpu_total'] / ops_df['cpu_total'].sum() * 100:.0f}%)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=cum, mode="lines+markers", name="累计占比"))
fig2.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% 线")
fig2.add_vline(x=n80, line_dash="dot", line_color="gray", annotation_text=f"前 {n80} 个算子")
fig2.update_layout(title="算子耗时累计占比(降序)", height=340,
                   xaxis_title="算子序号(按耗时降序)", yaxis_title="累计占比")
fig2
'''),
"📉 曲线越靠左上越「头重脚轻」—— 优化前几个算子就能拿到大部分收益,这就是剖析的意义。")

NB.code(D('''
# 保存给配套 App(它需要 profile_summary_27.json 才能启动)
payload = {
    "ops": rows[:30],
    "metrics": {
        "throughput_tok_s": round(float(throughput), 1),
        "latency_mean_ms": round(float(tpot_ms), 4),
        "latency_p99_ms": round(float(np.percentile(step_lat, 99)), 4),
        "sm_busy_pct": round(float(busy_pct), 1),
        "ttft_ms": round(float(ttft_ms), 2),
        "tpot_ms": round(float(tpot_ms), 4),
        "batch": B, "prompt_len": S, "decode_steps": N_DECODE,
    },
}
with open("profile_summary_27.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("已保存 profile_summary_27.json:", payload["metrics"])
'''),
"💾 App 会读这份文件渲染仪表盘;TTFT/TPOT 也一并存进去,方便随时回看实验条件。")

NB.md("## 6. 对应 vLLM 的生产级剖析 🔍",
D('''
vLLM 不用 print 计时,但方法论同构(见 [V1 使用文档](https://docs.vllm.ai/en/latest/design/v1/v1_usage.html)):

- **日志指标**:启动时开 `--enable-otel-metrics` / 查询 `/metrics` 端点,能直接拿到
  `vllm:time_to_first_token_seconds`、`vllm:time_per_output_token_seconds`、
  `vllm:request_success_total` 等本课同名指标(Prometheus 格式);
- **在线剖析**:`VLLM_TORCH_PROFILER_DIR=/tmp/vllm_profile` + API `start_profile/stop_profile`
  导出 torch.profiler trace,用 [Perfetto](https://ui.perfetto.dev) 打开看时间轴;
- **解码统计**:服务日志每 10 步打印一次 `Avg generation throughput`。

流程永远是:**大指标定方向 → 分布看尾巴 → trace 找热点 → 用第 24~26 课的武器优化 → 回测**。
'''))

NB.md("## 7. 配套 App:📈 性能剖析仪表盘 🎛️",
D('''
同目录的 `app_27_profiler.py` 加载刚保存的 `profile_summary_27.json`,渲染成仪表盘:
三个仪表(计算忙占比 / 吞吐 / 平均延迟)、Top-N 算子柱状图或表格(可按 cpu_total / cpu_self / calls 排序)、
以及 80/20 累计曲线。

**运行方法**(在 `ch04` 目录执行;需先跑完本 notebook 生成数据文件):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_27_profiler.py
```

浏览器打开 **http://localhost:8501**(也可加 `--server.port 8627` 换端口)。
下面这个 cell 会把 app 源码原样写入 `app_27_profiler.py`:
'''))

NB.code(app_cell("app_27_profiler.py"),
"📜 运行后覆盖写入相同内容,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "四大指标:TTFT(prefill 决定)、TPOT(decode 每步)、e2e = TTFT+(N−1)×TPOT、吞吐 = 词元/墙钟",
        "延迟要看分布:均值会骗人,P99 才代表最差 1% 的用户体验",
        "指标就是计时数据的四则运算 —— 本课每个数字都从 perf_counter 原始数据亲手算出",
        "torch.profiler + key_averages 得到算子热点表;80/20 曲线告诉你优化前几个算子就够",
        "vLLM 生产环境:OTel/metrics 端点拿同名指标 + torch.profiler trace 用 Perfetto 看",
    ],
    practice=[
        "把 N_DECODE 从 20 改成 100,观察 TPOT 与吞吐是否稳定、P99 变大还是变小",
        "把 B 从 4 改成 16:吞吐提升多少?TPOT 恶化多少?体会「吞吐-延迟跷跷板」",
        "把 S 从 32 改成 256,看 TTFT 如何随 prompt 长度增长(prefill 是平方级注意力)",
        "用 OpCounter(第 26 课)对照 profiler 表:哪些算子调用次数多但耗时低?(融合它们省的是启动开销)",
    ],
    links=[
        ("vLLM V1 使用文档", "https://docs.vllm.ai/en/latest/design/v1/v1_usage.html"),
        ("torch.profiler 文档", "https://pytorch.org/docs/stable/profiler.html"),
        ("Perfetto Trace 查看器", "https://ui.perfetto.dev"),
    ])

from pathlib import Path
out = str(Path(CH04) / "27_profiling_metrics.ipynb")
NB.save(out)
finalize(out)
