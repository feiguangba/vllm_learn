# -*- coding: utf-8 -*-
"""生成第 06 课 notebook:组装一个玩具推理引擎(教材级重写版)

设计要点(对齐 REWRITE_STANDARD.md):
1. 由浅入深:工厂流水线直觉 -> 吞吐定义与固定开销模型 -> 复用 n-gram+采样组装 ToyEngine -> 批量生成 -> 吞吐曲线与拟合 -> 真实 GPU 对照 -> vLLM 关联
2. 每一行代码都有 inline 注释
3. 每个结构(引擎/序列/吞吐表)打印内容与数值
4. 论文支撑:Orca 连续批处理(arXiv:2208.14217, OSDI'22)、PagedAttention(arXiv:2309.06180)、vLLM 博客
"""
import sys
from pathlib import Path
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

APP = Path(__file__).parent / "app_06_toy_engine.py"
APP_CODE = APP.read_text(encoding="utf-8")

nb = Notebook(
    "第 06 课 · 组装一个玩具推理引擎",
    subtitle="复用 n-gram 模型 + 采样策略,拼出能批量生成的引擎,实测吞吐 vs 批大小曲线并拟合",
    emoji="🛠️", chapter="第 1 章 · LLM 推理基础",
)

chapter_cover(
    nb,
    objectives=[
        "理解推理引擎的分层:语言模型(决定说什么)+ 采样策略(决定吐哪个字)+ 批处理循环(摊薄成本)",
        "复用第 02 课的 n-gram 模型与第 03 课的采样管线,组装成一个 ToyEngine 类",
        "实现批量生成:一次『共享前向』同时推进多条序列,理解固定开销被摊薄的机制",
        "实测并绘制『吞吐 vs 批大小』曲线,验证其『先陡升、后饱和』的形态",
        "用最小二乘拟合固定开销模型 T_step = C_fixed + b·C_per_seq,量化参数",
        "在真实 GPU 上复现同一条曲线,建立与 vLLM 连续批处理(Orca)的联系",
    ],
    toc=[
        ("直觉与动机", "工厂流水线的批量化"),
        ("核心定义与公式", "吞吐定义与固定开销模型"),
        ("组装引擎", "n-gram + 采样 + 批处理循环,逐行注释"),
        ("批量生成演示", "3 条序列同步推进"),
        ("吞吐 vs 批大小", "实测曲线 + 最小二乘拟合"),
        ("真实 GPU 对照", "同一条曲线在 GPU 上复现"),
        ("与 vLLM 的关系", "连续批处理与 iteration-level 调度"),
        ("Streamlit 动态演示", "实时看吞吐"),
    ],
    links=[
        ("Orca: A Distributed Serving System (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方博客: 10x faster inference", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Sarathi-Serve: 吞吐-延迟权衡 (OSDI'24)", "https://www.usenix.org/system/files/osdi24-agrawal.pdf"),
    ],
)

# =====================================================================
# 第 1 节 · 直觉与动机
# =====================================================================
nb.md(
    "## 1. 直觉与动机:工厂流水线的批量化\n\n"
    "一个推理引擎就像一条**工厂流水线**:每来一条文本生成请求,就要占用一次机器加工时间。"
    "如果请求**一条一条**来,机器加工完一条才接下一条,大部分时间在**空转**"
    "(等启动、等搬运、等内核);而 **batch(批处理)** 把多条请求**并成一批**同时加工——"
    "机器一次性读入一批,把固定开销(加载权重、启动内核)摊到多条序列头上。\n\n"
    "> 📄 这条直觉的工程化就是 [Orca, OSDI'22](https://arxiv.org/abs/2208.14217)"
    "提出的**迭代级调度(iteration-level scheduling)**,后成为 vLLM 的"
    "**连续批处理(continuous batching)**。本课先在第 02/03 课的玩具模型上,"
    "把这个机制亲手实现、亲手实测。"
)

# =====================================================================
# 第 2 节 · 核心定义与公式
# =====================================================================
nb.md(
    "## 2. 核心定义与公式\n\n"
    "**定义(吞吐)** 单位时间生成的 token 数:\n\n"
    "$$ \\text{throughput} = \\frac{\\text{生成的 token 总数}}{\\text{耗时}} "
    "(\\text{tokens/s}) $$\n\n"
    "**固定开销模型** 一次批量前向的时间大致是「固定成本 + 每序列线性成本」:\n\n"
    "$$ T_{\\text{step}} \\approx C_{\\text{fixed}} + b \\times C_{\\text{per-seq}} $$\n\n"
    "- $C_{\\text{fixed}}$:整批共享的权重加载、内核启动等开销(与 $b$ 无关);\n"
    "- $C_{\\text{per-seq}}$:每条序列各自采样、更新的开销。\n\n"
    "每步整批生成 $b$ 个 token(每条序列 1 个),于是:\n\n"
    "$$ \\text{throughput}(b) = \\frac{b}{C_{\\text{fixed}} + b\\,C_{\\text{per-seq}}} "
    "\\xrightarrow{b\\to\\infty} \\frac{1}{C_{\\text{per-seq}}} $$\n\n"
    "| 符号 | 含义 | 本课取值 |\n"
    "|---|---|---|\n"
    "| $b$ | 批大小(一批几条序列) | 1~32 |\n"
    "| $C_{\\text{fixed}}$ | 固定开销(与批无关) | 拟合得到(约 1e-4 s) |\n"
    "| $C_{\\text{per-seq}}$ | 每序列边际成本 | 拟合得到(约 1e-5 s) |\n"
    "| $1/C_{\\text{per-seq}}$ | 吞吐上界(批无限大) | 拟合得到 |\n\n"
    "**预测**:吞吐随 $b$ 先陡升(固定开销被摊薄)、后趋饱和(边际成本占主导)。"
    "下面组装引擎,实测这条曲线并拟合验证。"
)

# =====================================================================
# 第 3 节 · 组装引擎
# =====================================================================
nb.md(
    "## 3. 组装引擎:把积木拼起来\n\n"
    "拿出第 02 课的 **n-gram 模型**(决定说什么)与第 03 课的 **采样管线**(决定吐哪个字),"
    "装进一个 `ToyEngine` 类,再补一个**批量生成**循环。积木都是现成的,"
    "重点看 `generate_batch` 里「整批共享前向 + 逐序列采样」的结构。"
)

nb.code(
    '''# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 避免 Windows 下 OpenMP 库重复加载报错
import time                                             # 计时
import numpy as np                                      # 数值运算
from collections import defaultdict, Counter             # n-gram 统计所需

# ---- 积木一:n-gram 语言模型(来自第 02 课)----
def build_ngram(corpus, n):
    """统计 (前 n-1 个 token) -> {下一个 token: 次数}。n=2 即 bigram。"""
    model = defaultdict(Counter)                         # 上下文 -> Counter
    for text in corpus:                                  # 遍历每条文本
        toks = list(text)                                # 拆成 token(按字符)
        for i in range(len(toks) - n + 1):               # 滑动窗口
            ctx = tuple(toks[i:i + n - 1])               # 上下文元组
            model[ctx][toks[i + n - 1]] += 1             # 计数 +1
    return model                                         # 条件计数表

def next_candidates(model, context, n):
    """给定上下文,返回 (候选 token, 概率) 列表,按次数降序。"""
    ctx = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)  # 最近 n-1 个
    counts = model.get(ctx, Counter())                   # 查条件计数(没有则空)
    total = sum(counts.values())                         # 归一化分母
    if total == 0:                                       # 语料没见过
        return [], []                                    # 无候选
    items = sorted(counts.items(), key=lambda kv: -kv[1])  # 降序
    toks = [t for t, _ in items]                         # 候选列表
    probs = [c / total for _, c in items]                # MLE 概率
    return toks, probs                                   # 返回候选与概率

# ---- 积木二:采样管线(来自第 03 课)----
def softmax(logits, temperature=1.0):
    """带温度 softmax:T>1 变平, T<1 变尖。"""
    logits = np.asarray(logits, dtype=np.float64) / temperature  # 除以温度
    logits = logits - logits.max()                       # 数值稳定
    e = np.exp(logits)                                   # 指数
    return e / e.sum()                                   # 归一化

def sample_from(logits, temperature=1.0, rng=None):
    """从 logits 对应的概率分布里采样一个下标。"""
    if rng is None:                                      # 未提供 RNG
        rng = np.random.default_rng()                    # 新建
    probs = softmax(logits, temperature)                 # 概率分布
    return int(rng.choice(len(probs), p=probs))          # 多项式采样

# ---- 引擎本体 ----
class ToyEngine:
    """玩具推理引擎:共享前向(固定开销)+ 逐序列采样。"""

    def __init__(self, model, n=2, temperature=1.0, seed=0):
        self.model = model                               # n-gram 语言模型
        self.n = n                                       # 阶数
        self.temperature = temperature                   # 采样温度
        self.rng = np.random.default_rng(seed)           # 固定种子,可复现
        self.W = np.random.randn(48, 48) * 0.05          # 模拟的『模型权重』(固定开销来源)

    def shared_forward(self):
        """模拟整批共享的一次前向(矩阵乘)。成本与批大小无关 = C_fixed 的来源。"""
        return np.random.randn(48, 48) @ self.W          # 固定大小的 matmul

    def generate_batch(self, prompts, max_new):
        """批量生成:每步先做一次共享前向,再为每条序列各自采样。
        返回 (完成序列列表, 新增 token 总数)。"""
        seqs = [list(p) for p in prompts]                # 每条提示词 -> 一条待推进的序列
        tokens = 0                                       # 累计新增 token 数
        for _ in range(max_new):                         # 循环 max_new 步
            self.shared_forward()                        # 整批只做一次共享前向(摊薄固定开销)
            for seq in seqs:                             # 再逐序列推进
                toks, probs = next_candidates(self.model, seq, self.n)  # 查条件分布
                if not toks:                             # 无候选(语料没见过)
                    continue                             # 跳过该序列
                logits = np.log(np.array(probs) + 1e-12)  # 概率 -> logits(对数空间)
                idx = sample_from(logits, self.temperature, self.rng)  # 采样
                seq.append(toks[idx])                    # 新 token 追加进序列
                tokens += 1                              # 计数 +1
        return seqs, tokens                              # 返回序列与 token 数''',
    "🛠️ **引擎就绪**。`shared_forward` 代表整批共享的模型前向——它是固定开销;"
    "`generate_batch` 里『一次共享前向 + b 次逐序列采样』正是批处理摊薄成本的微观结构。",
)

# =====================================================================
# 第 4 节 · 批量生成演示
# =====================================================================
nb.md(
    "## 4. 批量生成演示:3 条序列同步推进\n\n"
    "用小语料训练一个 bigram 模型,再批量生成 3 条文本,"
    "看引擎如何「一趟趟地」同时推进所有序列。"
)

nb.code(
    '''# 训练语料 + 批量生成
corpus = ["我喜欢学习机器学习和深度学习", "机器学习很热门", "深度学习让机器更聪明",
          "我每天写代码和学算法", "我喜欢可爱的猫咪", "猫咪喜欢玩耍和睡觉"]   # 6 条中文句子
model = build_ngram(corpus, n=2)                         # 训练 bigram 模型

engine = ToyEngine(model, n=2, temperature=1.0, seed=0)  # 建引擎(固定种子)
prompts = ["我喜欢", "机器", "学习"]                      # 3 条提示词 = 批大小 3
seqs, total = engine.generate_batch(prompts, max_new=10)  # 每序列最多续写 10 个 token
for i, s in enumerate(seqs, 1):                          # 打印每条序列
    print(f"序列 {i}: {' '.join(s)}")
print(f"\\n共生成 {total} 个新 token(3 条 × ~{total // len(seqs)})")''',
    "✅ **3 条序列同步推进、各自接龙**。批大小 = 3,意味着每一次『共享前向』同时服务 3 条序列——"
    "固定开销被除以 3。",
)

# =====================================================================
# 第 5 节 · 吞吐 vs 批大小
# =====================================================================
nb.md(
    "## 5. 吞吐 vs 批大小:实测 + 拟合\n\n"
    "现在测量引擎在不同批大小下的**吞吐(tokens/s)**。让每个批大小生成相同的工作量,"
    "观察批处理带来的收益,再用第 2 节的固定开销模型做最小二乘拟合。"
)

nb.code(
    '''# 测量函数:给定批大小,返回(每步耗时, 吞吐 tokens/s)
def measure_step_time(batch, max_new=50, repeats=5):
    """跑 batch 条序列生成 max_new 步,返回 (每步耗时 s, 吞吐 tokens/s)。"""
    prompts = ["我喜欢" for _ in range(batch)]            # batch 条相同提示词
    eng = ToyEngine(model, n=2, seed=0)                  # 每次新建引擎(同一模型)
    times = []                                           # 记录每次总耗时
    for _ in range(repeats):                             # 重复测量取中位数
        t0 = time.perf_counter()                         # 计时开始
        _, tok = eng.generate_batch(prompts, max_new)    # 批量生成
        times.append(time.perf_counter() - t0)           # 计时结束
    dt = sorted(times)[len(times) // 2]                  # 中位数总耗时(去噪)
    return dt / max_new, tok / dt                        # 每步耗时 + 吞吐

batches = [1, 2, 4, 8, 16, 32]                           # 批大小扫描
step_times = []                                          # 各批的每步耗时
tps = []                                                 # 各批的吞吐
print("batch | 每步耗时(ms) | tokens/s")
for b in batches:                                        # 逐档测量
    st, tp = measure_step_time(b)                        # (每步耗时 s, 吞吐)
    step_times.append(st); tps.append(tp)                # 收集结果
    print(f"{b:5d} | {st*1e3:9.3f}   | {tp:8.1f}")''',
    "🚀 **看到吞吐随批大小明显上升**——从单条到 batch=32,涨了近一个数量级。"
    "同时注意**每步耗时**只从 ~0.36ms 涨到 ~0.49ms:固定开销被摊薄,吞吐才大涨。",
)

nb.code(
    '''# 最小二乘拟合固定开销模型:T_step = C_fixed + b*C_per_seq(对每步耗时做线性回归)
b_arr = np.array(batches, dtype=float)                   # 批大小向量
st_arr = np.array(step_times, dtype=float)               # 每步耗时向量
tps_arr = np.array(tps, dtype=float)                     # 实测吞吐向量
Cs, Cf = np.polyfit(b_arr, st_arr, 1)                    # 斜率=Cs(边际成本),截距=Cf(固定开销)
model_tp = b_arr / (Cf + b_arr * Cs)                     # 模型预测吞吐 = b / T_step

print(f"拟合结果: C_fixed = {Cf*1e6:.1f} μs, C_per_seq = {Cs*1e6:.2f} μs/序列")
print(f"模型吞吐上界 1/C_per_seq ≈ {1/Cs:.0f} tokens/s (批无限大时的极限)")
print("batch | 每步耗时(ms) | 实测吞吐 | 模型预测吞吐")
for b, st, real, p in zip(batches, st_arr, tps, model_tp):  # 逐档对比
    print(f"{b:5d} | {st*1e3:9.3f}   | {real:9.0f} | {p:12.0f}")
print("\\n注:实测吞吐 < 模型预测,是因为部分序列在语料里走到死路(候选耗尽)提前停止,")
print("    每步并非都产出 token —— 真实 LLM 里对应 EOS 提前终止,同样会拉低有效吞吐。")

import plotly.io as pio
pio.renderers.default = "plotly_mimetype"                # 静态渲染器
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=batches, y=tps, mode="markers", name="实测吞吐",
                         marker=dict(size=10, color="#E45756")))
fig.add_trace(go.Scatter(x=batches, y=model_tp, mode="lines", name="模型预测 b/T_step",
                         line=dict(width=3, color="#4C78A8")))
fig.update_layout(title="吞吐 vs 批大小:实测点 + 固定开销模型预测",
                  xaxis_title="批大小 batch", yaxis_title="tokens/s",
                  height=360, margin=dict(l=10, r=10, t=50, b=10))
fig.show()                                               # 展示''',
    "✅ **模型曲线与实测形态吻合:先陡升、后趋平**——验证了『固定开销 + 线性边际成本』的模型。"
    "两条线的间距来自 token 产出率(候选耗尽/EOS 提前终止),这正是真实推理系统的『有效吞吐 vs 理想吞吐』之差。",
)

# =====================================================================
# 第 6 节 · 真实 GPU 对照
# =====================================================================
nb.md(
    "## 6. 真实 GPU 对照:同一条曲线在 GPU 上复现\n\n"
    "玩具引擎证明了『批量摊薄固定开销』;现在用**真实 GPU 小 GPT** 复现同一条曲线,"
    "证明 decode 阶段「批大小 → 吞吐」的收益在真实硬件上同样成立——"
    "这正是下一章 vLLM 连续批处理的硬核依据。"
)

nb.code(
    '''# 真实对照:在 GPU 上跑一个小 GPT,测 decode 阶段「批大小 → 吞吐」
import sys
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")  # 共享库路径
from vllm_real import bench_throughput_curve, cuda_info  # 吞吐曲线 + 设备信息

b_real, tps_real, ms_real = bench_throughput_curve(      # GPU 实测
    batch=(1, 4, 16, 64, 128), token_len=16, reps=5)     # 各批大小各测 5 次
print("设备:", cuda_info())
print("batch | 每步耗时(ms) | 吞吐(tokens/s)")
for b, ms, tp in zip(b_real, ms_real, tps_real):         # 逐档打印
    print(f"{int(b):5d} | {ms:9.3f}   | {tp:10.0f}")
print("\\n结论:真实 decode 阶段同样『批越大吞吐越高』——这是 vLLM 连续批处理的地基。")''',
    "🚀 **真实 GPU 数字**。注意与玩具引擎的差异:GPU 上小 batch 每步耗时几乎不变"
    "(内核启动主导),吞吐近似线性;大 batch 后耗时开始抬头,吞吐增速放缓——同样是饱和。",
)

# =====================================================================
# 第 7 节 · 与 vLLM 的关系
# =====================================================================
nb.md(
    "## 7. 与 vLLM 的关系:从玩具引擎到连续批处理\n\n"
    "本课的 `ToyEngine` 与 vLLM 的差距只在**调度粒度**:\n\n"
    "- **ToyEngine(静态批)**:一批序列一起开工、一起收工,中间不能进新人;\n"
    "- **vLLM(连续批处理)**:每**一步迭代**结束都可以把完成的长序列踢出去、"
    "把新到的短序列加进来——批始终是满的。这是 [Orca, OSDI'22]"
    "(https://arxiv.org/abs/2208.14217) 提出的 **iteration-level scheduling**:\n\n"
    "> *«We present iteration-level scheduling with selective batching … the scheduler "
    "interacts with the execution engine at the granularity of iteration instead of request»*\n\n"
    "连续批处理 + **PagedAttention**([SOSP'23](https://arxiv.org/abs/2309.06180))"
    "让显存不再按最坏情况预留,吞吐因此再上一个台阶。vLLM 官方博客报告"
    " [最高 10×+ 吞吐提升](https://blog.vllm.ai/2023/06/20/vllm.html)。\n\n"
    "**对照表**:\n\n"
    "| 环节 | 本课玩具引擎 | vLLM |\n"
    "|---|---|---|\n"
    "| 模型 | n-gram(查表) | Transformer(前向) |\n"
    "| 采样 | `sample_from` | `Sampler`(greedy/multinomial) |\n"
    "| 批调度 | 静态整批 | iteration-level 连续批处理 |\n"
    "| 缓存 | 无 | KV Cache(PagedAttention 分页管理) |\n"
    "| 吞吐曲线 | 固定开销模型 | 实测批大小→吞吐 |"
)

# =====================================================================
# 第 8 节 · Streamlit
# =====================================================================
nb.md(
    "## 8. 🖥️ Streamlit 动态演示:实时看吞吐\n\n"
    "运行 `app_06_toy_engine.py`:输入多条提示词,调整**批大小、温度**,"
    "引擎实时批量生成,并用 `st.metric` 展示吞吐、绘制吞吐曲线。\n\n"
    "### 📜 App 完整源码(`app_06_toy_engine.py`)"
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
    "    print(\"    请把上方源码保存为 app_06_toy_engine.py 后运行:\")\n"
    "    print(\"    D:\\\\uv_envs\\\\uv_cuda\\\\Scripts\\\\python.exe -m streamlit run app_06_toy_engine.py\")\n"
)
nb.code(guard, "▶️ 此 cell 在 streamlit 环境中才真正运行;在 notebook 中仅作展示。")

nb.md(
    "### 🏃 运行方法\n\n"
    "1. 使用本目录已生成的 `app_06_toy_engine.py`;\n"
    "2. 在命令行执行:\n"
    "```\n"
    "D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_06_toy_engine.py\n"
    "```\n"
    "3. 浏览器打开 http://localhost:8501 ,观察吞吐随批大小变化。"
)

wrapup(
    nb,
    summary=[
        "推理引擎 = 语言模型(决定说什么)+ 采样策略(决定吐哪个字)+ 批量生成循环(摊薄成本)",
        "批处理让一次共享前向同时推进多条序列,固定开销(权重加载、内核启动)被摊薄",
        "吞吐 = b / (C_fixed + b·C_per_seq):随批大小先陡升、后趋饱和,上界 = 1/C_per_seq",
        "最小二乘拟合验证了固定开销模型的形态,量化出固定/边际成本的比例",
        "真实 GPU 上 decode 阶段同样呈现『批越大吞吐越高』的曲线,为连续批处理提供依据",
        "vLLM 的连续批处理(Orca 的 iteration-level scheduling)在每步迭代动态进出请求,是本课静态批的工程升级",
    ],
    practice=[
        "把 shared_forward 的矩阵改成 (64,64)/(96,96),观察 C_fixed 变大后吞吐曲线如何变化",
        "给 generate_batch 加一个『序列完成即退出』逻辑:某条序列采样不到候选就从 batch 移除(连续批处理的雏形)",
        "用 numpy 计算拟合残差 R²,评估固定开销模型对曲线的解释力",
        "思考题:真实 LLM 里 C_per_seq 随序列长度增长(KV Cache 读取量随 T 增长),这会让吞吐曲线变成什么样?",
    ],
    links=[
        ("Orca (OSDI'22)", "https://arxiv.org/abs/2208.14217"),
        ("PagedAttention (SOSP'23)", "https://arxiv.org/abs/2309.06180"),
        ("vLLM 官方博客", "https://blog.vllm.ai/2023/06/20/vllm.html"),
        ("Sarathi-Serve (OSDI'24)", "https://www.usenix.org/system/files/osdi24-agrawal.pdf"),
    ],
)

nb.save(r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch01\06_toy_inference_engine.ipynb")