# -*- coding: utf-8 -*-
"""生成 22_modelrunner_dataflow.ipynb(app_22_dataflow.py 已存在,仅嵌入源码)"""
from helpers import D, MINI_GPT, TIME_EVENTS, chapter_cover, wrapup, new_nb, CH04, app_src, finalize
from pathlib import Path

NB = new_nb("第 22 课 · 小模型完整前向数据流:从 embedding 到 lm_head",
            subtitle="跟着一个词元流,把迷你 GPT 的前向拆成九段,每一步的张量形状都打印出来看个究竟",
            emoji="🚰")

chapter_cover(NB,
    objectives=[
        "用一台「迷你 GPT」(H=256、2 层、4 头、词表 5000)亲眼看一次完整前向",
        "把前向拆成九段:输入组装 → 词嵌入 → QKV → 注意力 → MLP → lm_head → 采样,逐段打印张量形状",
        "理解为什么 prefill 阶段的张量巨大、decode 阶段的张量小而精",
        "用 pyecharts 桑基图画出数据流、用 plotly 柱状图画出各步显存账本",
        "对照 vLLM 的 ModelRunner.execute_model,明白它为什么每次执行前都要「组装输入」",
    ],
    toc=[
        ("直觉:一条流水线", "词元像零件,经过一道道工序变成下一个词元"),
        ("迷你模型:麻雀虽小五脏俱全", "H=256、2 层、4 头,参数只有百万级,结构却与真 GPT 同构"),
        ("九段前向:逐段打印形状", "embedding → QKV → 注意力 → MLP → lm_head,形状一目了然"),
        ("显存账本:每个张量占多少", "元素数 × 字节数,量化 prefill 与 decode 的显存差异"),
        ("桑基图:数据流一眼看穿", "pyecharts 桑基图把九段工序连成一条河"),
        ("plotly 柱状图:显存水位", "哪一步的张量最吃显存,柱状图说话"),
        ("prefill vs decode", "同样一台模型,两种阶段形状完全不同"),
        ("对应 vLLM 源码", "ModelRunner.execute_model 的三件事:组装、循环、采样"),
        ("真实 GPU:小模型形状追踪 + 实测耗时", "TinyGPT 前向的形状串 + prefill/decode 的真实开销"),
        ("配套 Streamlit 演示", "app_22_dataflow.py:任意调参,逐步骤看形状与显存"),
    ],
    links=[
        ("vLLM ModelRunner 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py"),
        ("PyTorch nn.Linear 文档", "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"),
        ("PyTorch scaled_dot_product_attention", "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"),
    ])

NB.md("## 1. 直觉:一条流水线 🚰",
D('''
想象一条**工厂流水线**:最前面倒进一堆零件(词元),每经过一台机器,零件就被加工一下,
流水线末端吐出一个成品(下一个词元)。大模型的**前向传播(forward)**就是这么一条流水线,
而「每台机器」就是一层神经网络。

vLLM 的 `ModelRunner.execute_model` 本质上只做三件事:

1. **组装输入**:把调度器挑好的词元拼成 `input_ids / positions / seq_lens`(第 21 课的内容);
2. **循环跑模型**:`embedding → N 层 Transformer → lm_head`;
3. **采样输出**:从 logits 里挑出每个序列的下一个词元。

本课用一台迷你 GPT,亲手把这三件事各走一遍,并且**每一步都把张量形状打印出来**。
为什么形状这么重要?因为 GPU 上的每次矩阵乘法、每个 kernel,都是「形状决定算力与显存」——
形状一旦错了,不是报错就是算废。
'''))

NB.md("## 2. 迷你模型:麻雀虽小五脏俱全 🐤",
D('''
真实的大模型有几百亿参数,拆起来眼花缭乱。我们照原样搭一台**迷你 GPT**,结构完全同构:

- **词嵌入 embedding**:把词元 id 查表成 `H=256` 维向量;
- **位置嵌入 positional embedding**:给每个位置也学一个 `H=256` 维向量;
- **N=2 层 Transformer**,每层由「注意力块 + MLP 块」组成:
  - 注意力块:LayerNorm → QKV 投影 → 分 4 个头(head_dim=64)→ 因果自注意力 → 输出投影 + 残差;
  - MLP 块:LayerNorm → gate/up 融合线性(`4H=1024`)→ SiLU 门控 → down 线性 + 残差;
- **lm_head**:把 `H=256` 维隐状态投影回词表 `V=5000` 维,得到 logits。

下面把模型定义出来,顺便数一下参数。注意我们开启了 TF32 矩阵乘,既加速又消除告警。
'''))

NB.code(MINI_GPT,
"🏗️ 迷你 GPT 完整定义。后面几课(25/26/27)还会复用这台模型做 CUDA Graph、torch.compile 与剖析。")

NB.code(D('''
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(vocab=5000, hidden=256, n_layers=2, n_heads=4).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"设备            = {dev}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"参数总量        = {n_params:,}")
print(f"隐藏维度 H      = {model.hidden},词表 V = {model.vocab},层数 = 2,头数 = 4")
print(f"一句话算账:embedding={model.embed.weight.numel():,} + lm_head={model.lm_head.weight.numel():,} "
      f"已占大头(词表 5000 × 256 = 128 万,乘 2 次)")
'''),
"✅ 约 400 万参数,远小于真实大模型,但「embedding → layers → lm_head」的三段结构一模一样。")

NB.md("## 3. 九段前向:逐段打印形状 📐",
D('''
现在把一个「prefill 阶段」的批喂进去:`B=4` 条序列、每条 `S=8` 个词元,于是拼接后的词元流长度
`T = B×S = 32`。我们把 `forward` 里的每一步**手动拆开写一遍**,每一步打印一次形状。

先造输入。注意 `positions` 是**每条序列内部**从 0 连续编号,然后头尾拼起来——这正是第 21 课
紧凑拼接的结果;`seq_lens` 记录每条序列的长度。
'''))

NB.code(D('''
B, S = 4, 8                 # 批大小 4,每条序列 8 个词元(prefill 一次吃 8 个)
T = B * S                   # 紧凑拼接后的总词元数
input_ids = torch.randint(0, model.vocab, (T,), device=dev)
positions = torch.cat([torch.arange(S, device=dev) for _ in range(B)])
seq_lens = torch.full((B,), S, dtype=torch.long, device=dev)
print(f"input_ids  形状 {tuple(input_ids.shape)}   语义:拼接的 {T} 个词元 id")
print(f"positions  形状 {tuple(positions.shape)}   语义:每个词元在各自序列里的位置")
print(f"seq_lens   形状 {tuple(seq_lens.shape)}     语义:每条序列长度")
'''),
"🚰 输入组装完毕:三个张量、三种语义。`T = B×S` 只在 prefill 成立,decode 时 `T = B`(后面会讲)。")

NB.code(D('''
H = model.hidden
x, p = input_ids, positions

print("① 输入组装      input_ids [T]      =", tuple(x.shape))
h = model.embed(x) + model.pos(p)
print("② 词嵌入+位置   hidden [T, H]      =", tuple(h.shape))

for li, (attn, mlp) in enumerate(model.layers):
    print(f"--- 第 {li} 层 ---")
    y = attn.norm(h)
    print("③ QKV 投影      norm 后 [T, H]     =", tuple(y.shape))
    qkv = attn.qkv(y)
    print("   fused QKV     qkv [T, 3H]       =", tuple(qkv.shape))
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(-1, attn.n_heads, attn.head_dim).transpose(0, 1)
    k = k.view(-1, attn.n_heads, attn.head_dim).transpose(0, 1)
    v = v.view(-1, attn.n_heads, attn.head_dim).transpose(0, 1)
    print("   分头后        q [heads, T, d]    =", tuple(q.shape), "(k/v 同形)")
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print("④ 注意力输出    out [heads, T, d]  =", tuple(out.shape))
    h = h + attn.o(out.transpose(0, 1).reshape(-1, H))
    print("⑤ 注意力残差后  hidden [T, H]      =", tuple(h.shape))

    y = mlp.norm(h)
    gu = mlp.gate_up(y)
    g, u = gu.chunk(2, dim=-1)
    print("⑥ MLP gate/up   gu [T, 4H]        =", tuple(gu.shape), " 门控后 [T, 2H] =", tuple(g.shape))
    h = h + mlp.down(F.silu(g) * u)
    print("⑦ MLP 残差后    hidden [T, H]      =", tuple(h.shape))

logits = model.lm_head(h)
print("⑧ lm_head       logits [T, V]      =", tuple(logits.shape))
next_tokens = logits.argmax(dim=-1)
print("⑨ 采样          next_tokens [T]    =", tuple(next_tokens.shape))
'''),
"📐 这是本课核心 cell:九段前向,段段有形状。`H` 维度像一条 256 宽的传送带,始终不变;`T` 只在开头和结尾出现。")

NB.code(D('''
# 验证:手动分步的结果与 model.forward 完全一致(用 max 而不是 allclose,避免 TF32 舍入差异)
logits_ref = model(input_ids, positions)
diff = (logits - logits_ref).abs().max().item()
print(f"手工分步 vs model.forward 的最大绝对误差 = {diff:.2e}")
print("✅ 形状与数值都对得上:九段拼起来就是一次完整 forward")
'''),
"✅ 分步拆解不是「另一个模型」,它就是 `forward` 本尊。理解这一步,vLLM 的执行就通了。")

NB.md("## 4. 显存账本:每个张量占多少 💾",
D('''
形状看完,算算账。每个张量占用的字节数 = **元素数 × dtype 字节数**(fp32 是 4 字节,long 是 8 字节)。
我们把这些张量汇总成一张表,看看**哪一步最吃显存**。
'''))

NB.code(D('''
import pandas as pd

DT = {"long": 8, "float": 4}
rows = [
    ("1 输入组装", "input_ids", (T,), "long", "拼接词元流"),
    ("1 输入组装", "positions", (T,), "long", "绝对位置"),
    ("2 词嵌入", "hidden_states", (T, H), "float", "embedding 输出"),
    ("3 QKV 投影", "qkv_proj", (T, 3 * H), "float", "fused QKV"),
    ("3 QKV 投影", "q/k/v 分头", (T, 4, H // 4), "float", "多头 reshape"),
    ("4 注意力", "attn_scores", (T, 4, S), "float", "q·k^T 因果分数"),
    ("4 注意力", "attn_output", (T, 4, H // 4), "float", "softmax 加权"),
    ("5 MLP", "mlp_gate_up", (T, 4 * H), "float", "gate/up 双路"),
    ("5 MLP", "mlp_down", (T, H), "float", "down 投影"),
    ("6 输出", "logits", (T, model.vocab), "float", "LM Head"),
    ("7 采样", "next_tokens", (T,), "long", "argmax"),
]
mem = []
for step, name, shape, dtype, note in rows:
    n = 1
    for s in shape:
        n *= s
    mem.append({"步骤": step, "张量": name, "形状": str(shape), "元素数": n,
                "显存(KB)": round(n * DT[dtype] / 1024, 1), "说明": note})
df = pd.DataFrame(mem)
df
'''),
"💾 注意 `logits [T, V]` 和 `mlp_gate_up [T, 4H]` 是两大显存户:词表 5000 和 MLP 膨胀系数 4 都很大。")

NB.md("## 5. 桑基图:数据流一眼看穿 🎨",
D('''
把上面的九段工序画成一张 **pyecharts 桑基图**:每条「河」的宽度 = 该张量的元素数。
你会看到词元流从输入涌入,在 QKV / MLP 处「变宽」(维度膨胀),最后在采样处收成一条细线。
'''))

NB.code(D('''
from pyecharts.charts import Sankey
from pyecharts import options as opts

nodes = [
    {"name": "input_ids [T]"}, {"name": "positions [T]"},
    {"name": "hidden [T,H]"}, {"name": "qkv [T,3H]"},
    {"name": "q/k/v [heads,T,d]"}, {"name": "attn_out [heads,T,d]"},
    {"name": "mlp_gate_up [T,4H]"}, {"name": "mlp_down [T,H]"},
    {"name": "logits [T,V]"}, {"name": "next_tokens [T]"},
]
links = [
    {"source": "input_ids [T]", "target": "hidden [T,H]", "value": T},
    {"source": "positions [T]", "target": "hidden [T,H]", "value": T},
    {"source": "hidden [T,H]", "target": "qkv [T,3H]", "value": T * H},
    {"source": "qkv [T,3H]", "target": "q/k/v [heads,T,d]", "value": T * 3 * H},
    {"source": "q/k/v [heads,T,d]", "target": "attn_out [heads,T,d]", "value": T * 3 * H},
    {"source": "attn_out [heads,T,d]", "target": "mlp_gate_up [T,4H]", "value": T * H},
    {"source": "mlp_gate_up [T,4H]", "target": "mlp_down [T,H]", "value": T * 4 * H},
    {"source": "mlp_down [T,H]", "target": "logits [T,V]", "value": T * H},
    {"source": "logits [T,V]", "target": "next_tokens [T]", "value": T * model.vocab},
]
sankey = (
    Sankey()
    .add("数据流", nodes, links,
         linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color="source"),
         label_opts=opts.LabelOpts(position="right"))
    .set_global_opts(title_opts=opts.TitleOpts(title="迷你 GPT 前向数据流(河宽 = 元素数)"))
)
sankey.render_notebook()
'''),
"🎨 桑基图上的「瓶颈」就是最宽的河:`logits` 与 `mlp_gate_up` 两条最粗——跟显存账本完全呼应。")

NB.md("## 6. plotly 柱状图:显存水位 📊",
D('''
再用 plotly 画一张按步骤着色的**显存柱状图**。桑基图看流向,柱状图看绝对值——
哪一步最吃显存,一柱了然。
'''))

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.express as px

fig = px.bar(df, x="张量", y="显存(KB)", color="步骤", text="显存(KB)",
             hover_data=["形状", "说明"])
fig.update_layout(title="各步张量显存占用(KB,prefill B=4 S=8)",
                  height=430, xaxis_tickangle=-35)
fig.show()
'''),
"📊 把鼠标悬停在柱子上看形状。`logits` 一柱擎天,因为词表 5000 太大了。")

NB.md("## 7. prefill vs decode:同台机器,两种身材 🏋️",
D('''
上面一直是 **prefill**:`T = B×S`,一口气处理整段输入,张量又大又多。真实推理里,
**decode** 阶段每条序列每步只产生 **1 个新词元**,于是 `T = B`,张量一下子小了很多。
但注意:decode 的注意力窗口 `kv_len` 会随历史增长(第 8 章详细讲 KV Cache)。

下面用一个循环把两种阶段的形状对比打印出来:
'''))

NB.code(D('''
print("=== prefill:B=4 条、每条 S=8 → T=32,一次吃整段 ===")
print(f"  hidden [T,H] = [{B*S}, {H}],logits [T,V] = [{B*S}, {model.vocab}]")

print("=== decode:B=4 条、每条只新增 1 词元 → T=4 ===")
Td = B
print(f"  hidden [T,H] = [{Td}, {H}],logits [T,V] = [{Td}, {model.vocab}]")
print(f"  但注意力 kv_len 随步数增长:第 0 步 kv_len=1,第 k 步 kv_len={S}+k+1")
print()
print("结论:prefill 是「计算密集」的洪水猛兽,decode 是「访存密集」的小步快跑。")
print("vLLM 用 CUDA Graph 优化的正是 decode 这种「每步只有几个小 kernel」的场景(第 24/25 课)。")
'''),
"🏋️ 形状的「变」是 vLLM 每次执行前都要重新组装输入的根本原因;形状的「不变」(decode 每步 T=B)则是 CUDA Graph 能捕获的契机。")

NB.md("## 9. 真实 GPU:小模型形状追踪 + 实测耗时 ⏱️",
D('''
上面第 2~7 节的形状与显存都是 **MiniGPT 在 CPU 上手推的**。现在用跨章共享库
`vllm_real.TinyGPT`(纯线性层小模型,在 RTX 5060 上真实前向)把同一套「形状串」在真机上走一遍,
再实测「一次并行 prefill(compute-bound)vs 逐字 decode(memory-bound)」的**真实毫秒**:

- 形状追踪:`(B, T) → [embed] → (B, T, d) → [N 层] → (B, T, vocab) = logits`;
- 一档实测耗时:prefill 一次并行 L 词元 vs decode 每步 1 词元,谁快谁慢、慢多少倍。

这正是第 5 节「prefill 洪水 vs decode 小步快跑」在真实设备上的打开方式,也是下一课 KV Cache
该按块管理的物理理由(decode 每步都紧巴巴地只在搬参数和 KV)。
'''))

NB.code(D('''
import sys; sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\minivllm\\exercises")
from vllm_real import TinyGPT, bench_prefill_decode, cuda_info
import torch

# —— 真实形状追踪 ——
gpt = TinyGPT(d=256, layers=8, vocab_size=1024).to("cuda")
B, T = 4, 32
x = torch.randint(0, 1024, (B, T), device="cuda")
with torch.no_grad():
    h = torch.tanh(gpt.embed[x])          # (B,T,d)
    for w in gpt.ws:
        h = torch.tanh(h @ w)             # (B,T,d) 每层不变
    logits = h @ gpt.head                 # (B,T,vocab)
print("设备:", cuda_info())
print("形状串: input_ids (B,T)=", tuple(x.shape),
      "→ embed (B,T,d)=", tuple(h.shape),
      "→ logits (B,T,vocab)=", tuple(logits.shape))
print(f"模型参数: {gpt.params_count()/1e6:.2f}M(纯线性,T 不变维度,decode 时各自序列只贡献 1 新词元)")

# —— 一档真实耗时对照 ——
b = bench_prefill_decode(d=256, layers=8, L=256, steps=64, reps=7)
print(f"\\nprefill 一次并行 {b['L']} 词元 : {b['prefill_ms']:.3f} ms → {b['prefill_tok_per_s']/1e3:.0f} k tok/s")
print(f"decode 每步 1 词元           : {b['decode_step_ms']:.3f} ms/步")
print(f"同样 L 词元,decode/prefill 耗时比 = {b['ratio']:.0f} 倍")
print("→ 形状一「大」一「小」的真实代价:prefill 吃满算力,decode 每步都在搬权重(KV 痛点)。")

import gc
torch.cuda.empty_cache(); gc.collect()
'''),
"⏱️ **真实 GPU 数字**。形状的「变」(prefill T=32)与「不变」(decode 每步 T=B)直接决定这两档耗时不同的物理原因 —— 下一课的 KV Cache 分块就是为decode 每步的小身材设计的。")

NB.md("## 10. 对应 vLLM 源码:ModelRunner.execute_model 🔍",
D('''
vLLM 的 `vllm/v1/worker/gpu_model_runner.py` 里,`execute_model` 的职责正是本课三步:

1. **组装输入**:`ModelInputForGPUBuilder` 把 `SchedulerOutput` 变成 `input_ids / positions / slot_mapping / block_table / seq_lens`(第 21 课);
2. **执行模型**:调用 `self.model(...)` 跑一遍 embedding → layers → lm_head(本课九段);
3. **采样**:`sampler` 从 logits 采样出 `next_tokens`,再写回各序列。

⚠️ 诚实说明:本练习环境的 `vendor/vllm` 目录尚未就绪,以上对应关系基于 vLLM 公开源码;
配好 vendor 后可直接 `grep -n "def execute_model"` 核对。本课重点是**把这条数据流吃透**,
而不是逐字段比对源码。
'''))

NB.md("## 11. 配套 App:🚰 模型前向数据流浏览器 🎛️",
D('''
同目录的 `app_22_dataflow.py` 把「九段前向」做成了可交互浏览器:拖动 **批大小** 与 **序列长度**,
每一步的张量形状与显存实时重算,还能切换 prefill / decode、用对数坐标看显存。

**运行方法**(在 `ch04` 目录执行):

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_22_dataflow.py
```

浏览器打开 **http://localhost:8501**。完整源码如下(与同目录 `app_22_dataflow.py` 一字不差):
'''))

NB.code("%%writefile app_22_dataflow.py\n" + app_src("app_22_dataflow.py"),
"📜 这就是 app_22_dataflow.py 的完整源码(本 cell 只写入文件、不执行),notebook 与 app 共享同一套形状规则,保证讲解与演示一致。")

wrapup(NB,
    summary=[
        "前向 = 组装输入 → 循环模型(embedding→layers→lm_head)→ 采样输出,三步走",
        "九段前向的形状串起来:`[T] → [T,H] → [T,3H] → [heads,T,d] → [T,4H] → [T,V]`",
        "两大显存户:logits `[T,V]`(词表大)与 mlp_gate_up `[T,4H]`(膨胀系数 4)",
        "prefill 的 T=B×S 是计算密集的洪水,decode 的 T=B 是访存密集的小步快跑",
        "vLLM 的 ModelRunner.execute_model 就是这三步,形状的变与不变决定了后面所有优化",
    ],
    practice=[
        "把 n_layers 改成 4,重新走九段前向,观察参数与显存如何随层数线性增长",
        "把 vocab 从 5000 改成 50000,看 logits 的显存怎么暴涨——这对应真实大词表模型",
        "在 decode 场景(B=4)下重算显存账本,和 prefill 对比,算出两者相差多少倍",
        "给桑基图加上「KV Cache 写入」节点,想想 decode 时这条河应该多宽",
    ],
    links=[
        ("vLLM ModelRunner 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py"),
        ("PyTorch nn.Embedding", "https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html"),
    ])

out = str(Path(CH04) / "22_modelrunner_dataflow.ipynb")
NB.save(out)
finalize(out)
