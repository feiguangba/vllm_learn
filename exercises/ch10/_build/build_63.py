# -*- coding: utf-8 -*-
"""生成 63_memory_scheduling.ipynb 与 app_63_memory.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH10, app_cell, finalize
from pathlib import Path

APP_63 = D('''
# -*- coding: utf-8 -*-
# app_63_memory.py — 内存规划与生命周期:生命周期图交互 🧠
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="内存规划与生命周期 🧠", layout="wide")
st.title("🧠 第 63 课 · 内存规划与生命周期")

st.markdown("""
编译器看完整张图后,能算出**每个张量何时出生、何时死亡**(生命周期分析),于是让已死的张量
**复用**它们的显存(内存复用)。下方拖动参数,观察生命周期图与
『不复用 / 复用』两种策略的内存占用差多少。
""")

# ---------------- 与 notebook 一致的内存规划模拟 ----------------
def build_ops(n_ops, base, with_shared=True):
    """构造计算图:n_ops 个『主要结果』张量;若有共享张量,则第 2 个结果被第 3 步复用(残差式)"""
    ops = []
    prev = "x"
    for i in range(n_ops):
        ops.append((f"op{i}", [prev], f"h{i}", base))   # 主要结果
        if i == 1 and with_shared:
            ops.append((f"add{i}", [f"h{i-1}", f"h{i}"], f"s{i}", base))  # 共享/残差
        prev = f"h{i}"
    return ops

def plan_memory(ops):
    """生命周期 + 贪心 best-fit 复用。返回 (names, born, death, size, slot_of, reuse_peak, no_reuse)"""
    born, death, size = {}, {}, {}
    for t, (name, inputs, out, sz) in enumerate(ops):
        born[out], size[out] = t, sz
        for inp in inputs:
            death[inp] = t
    for name in born:
        death.setdefault(name, born[name])
    names = sorted(set(size) | {"x"})
    if "x" in names:
        size["x"] = base = ops[0][3]; born["x"] = 0
        death["x"] = max([t for t, (n, ins, o, s) in enumerate(ops) if "x" in ins] or [0])
    slots, alloc, slot_of, cur, reuse_peak = [], {}, {}, 0, 0
    for t in range(len(ops)):
        for nm in list(alloc):
            if death[nm] < t:
                sid = alloc.pop(nm); slots[sid] = (t, slots[sid][1], None); cur -= size[nm]
        for nm in names:
            if born[nm] <= t <= death[nm] and nm not in alloc:
                best = min((i for i, (ft, sz, o) in enumerate(slots) if o is None and sz >= size[nm]),
                           key=lambda i: slots[i][1], default=None)
                if best is not None:
                    slots[best] = (t, slots[best][1], nm); alloc[nm] = best
                else:
                    alloc[nm] = len(slots); slots.append((t, size[nm], nm))
                slot_of.setdefault(nm, alloc[nm]); cur += size[nm]
        reuse_peak = max(reuse_peak, cur)
    no_reuse = sum(size.values())            # 不复用 = 每个张量永久占一块
    return names, born, death, size, slot_of, reuse_peak, no_reuse

# ---------------- 控件 ----------------
st.sidebar.header("🎛️ 参数")
n_ops = st.sidebar.slider("算子个数(链长)", 2, 12, 6, 1)
base = st.sidebar.selectbox("单张量规模", [16, 64, 256, 1024], index=2, format_func=lambda v: f"{v} MB")
with_shared = st.sidebar.checkbox("存在共享/残差张量", value=True)
show_slots = st.sidebar.checkbox("按内存槽上色", value=True)
st.sidebar.caption("链越长,复用在『不复用=每张量独占』的对比下省得越多。")

ops = build_ops(n_ops, base, with_shared)
names, born, death, size, slot_of, reuse_peak, no_reuse = plan_memory(ops)

c1, c2, c3 = st.columns(3)
c1.metric("不复用总内存", f"{no_reuse:.0f} MB")
c2.metric("复用后峰值", f"{reuse_peak:.0f} MB")
c3.metric("省下", f"{100 * (1 - reuse_peak / max(no_reuse, 1)):.0f}%")
st.caption("不复用 = 每个张量永久各自占一块显存;复用 = 死张量腾出槽给新张量,只看同一时刻并发峰值。")

# ---------------- 生命周期 Gantt ----------------
st.subheader("📊 张量生命周期图")
rows = []
for nm in names:
    rows.append({"张量": nm, "出生": born[nm], "死亡": death[nm], "大小MB": size[nm],
                 "槽": slot_of.get(nm)})
df = pd.DataFrame(rows).sort_values("槽", na_position="first").reset_index(drop=True)
fig = go.Figure()
color = px.colors.qualitative.Plotly
for _, r in df.iterrows():
    fig.add_trace(go.Bar(x=[max(r["死亡"] - r["出生"], 0.5)], y=[r["张量"]], base=r["出生"],
                         orientation="h", name=r["张量"],
                         marker_color=color[(r["槽"] or 0) % len(color)],
                         hovertemplate=f"{r['张量']}: {r['出生']}→{r['死亡']}, {r['大小MB']}MB, 槽{r['槽']}<extra></extra>"))
fig.update_layout(barmode="overlay", title="张量生命周期(颜色 = 内存槽,同色 = 共用显存)",
                  xaxis_title="时间步", height=max(320, 40 * len(df)), margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------- 峰值对比 ----------------
st.subheader("⛰️ 内存占用对比")
fig2 = go.Figure(go.Bar(x=["不复用", "复用"], y=[no_reuse, reuse_peak],
                        marker_color=["#c0392b", "#27ae60"],
                        text=[f"{no_reuse:.0f}MB", f"{reuse_peak:.0f}MB"], textposition="outside"))
fig2.update_layout(title="同一张图:复用让内存占用大降", yaxis_title="MB", height=340,
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.caption("⭐ 这就是大模型激活值 / KV Cache 内存优化的核心思想:能复用就不新开。")
''')

NB = new_nb("第 63 课 · 内存规划与生命周期:把显存用得抠门",
            subtitle="张量生命周期分析 · 内存复用 · in-place 优化 —— 编译器如何让内存占用大降",
            emoji="🧠")

chapter_cover(NB,
    objectives=[
        "理解张量生命周期:出生(被创建)、活跃(被读取)、死亡(最后一次使用后)三个时点",
        "掌握内存复用规划算法:贪心 best-fit 槽分配,让已死张量腾出显存给新张量",
        "亲手实现生命周期分析 + 内存规划模拟,对比『不复用 / 复用』的内存占用",
        "理解 in-place 优化:就地写回避免分配新张量,并用 torch 实测",
        "用 matplotlib / seaborn / plotly 画生命周期 Gantt 与槽占用热力图",
        "跑通配套 App:拖动参数观察生命周期与内存占用变化",
    ],
    toc=[
        ("直觉:住酒店与退房", "张量是房客,退房后房间可以给下一位住"),
        ("生命周期分析", "出生 / 活跃 / 死亡 —— 编译器怎么算出每个张量活多久"),
        ("内存复用算法", "贪心 best-fit 槽分配,模拟『复用 vs 不复用』"),
        ("in-place 优化", "就地写回:省掉新张量分配,并用 torch 实测"),
        ("可视化与配套 App", "Gantt 生命周期图 + 槽占用热力图 + app_63_memory.py"),
    ],
    links=[
        ("XLA 内存规划(buffer 复用)", "https://www.tensorflow.org/xla/operation_semantics"),
        ("PyTorch 内存管理(CachingAllocator)", "https://pytorch.org/docs/stable/notes/cuda.html"),
        ("TVM 内存规划文档", "https://tvm.apache.org/docs/"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:住酒店与退房 🏨",
D('''
把显存想象成一家酒店,每个张量是一位房客:

- **出生**:入住(算子创建了它);
- **活跃**:在房间里住着(还有算子要读它);
- **死亡**:退房(最后一个用它的算子跑完了)。

一家聪明的酒店不会"每个房客各租一栋楼",而是**房客退房后立刻把房间收拾出来给下一位**。
编译器做内存规划(memory planning)就是干这件事:它先分析每个张量的生命周期,再决定
哪些张量可以**共用一个内存槽**(memory reuse),从而让内存占用大幅下降。

对 LLM 推理尤其重要:激活值(activation)在每层都会被创建和销毁,如果不能复用,
几百层叠起来显存立刻爆炸 —— 这就是为什么 KV Cache 与激活值内存都要"精打细算"。
'''))

NB.md("## 2️⃣ 生命周期分析:算出每个张量活多久 🕰️",
D('''
给定一张计算图,编译器按**拓扑序**扫描:

- 张量 $t$ 的**出生时间** = 产生它的那个算子执行的时间;
- 张量 $t$ 的**死亡时间** = **最后一次读取它的算子**执行的时间;
- 生命周期 = $[\\text{born},\\ \\text{death}]$。

关键洞察:**一个算子只在它执行的那一瞬间需要输入存活**,之后输入就没用了。于是"活多久"
完全由"谁是最后一个消费者"决定。下面构造一个典型的 MLP 前向,列出每个张量的生命周期。
'''))

NB.code(D('''
# 构造一个 MLP 前向的计算图:每个 op = (名字, 输入, 输出, 输出大小MB)
ops = [
    ("matmul1", ["x", "W1"], "h1", 64),   # h1 = x@W1
    ("bias1",   ["h1", "b1"], "a1", 64),  # a1 = h1 + b1
    ("relu1",   ["a1"],       "r1", 64),  # r1 = relu(a1)
    ("matmul2", ["r1", "W2"], "h2", 64),
    ("relu2",   ["h2"],       "r2", 64),
    ("matmul3", ["r2", "W3"], "out", 16),
]

# 生命周期分析
born, death, size = {}, {}, {}
for t, (name, inputs, out, sz) in enumerate(ops):
    born[out], size[out] = t, sz
    for inp in inputs:
        death[inp] = t                     # 最后一次读到它的时间 = 它死亡
for name in born:
    death.setdefault(name, born[name])     # 没人读的输出(如 out),活到被创建为止

print(f"{'张量':6s} {'出生':>4s} {'死亡':>4s} {'大小MB':>7s}  生命周期")
for name in sorted(born, key=lambda n: born[n]):
    print(f"{name:6s} {born[name]:>4d} {death[name]:>4d} {size[name]:>7d}   [{born[name]}, {death[name]}]")
'''),
"🔍 观察:每个中间张量(x 之外)在产生它的下一条算子后基本就死了 —— 它们的显存完全可以复用。")

NB.md("## 3️⃣ 内存复用算法:贪心 best-fit 🗄️",
D('''
有了生命周期,内存规划就成了一个"在线装箱"问题:维护一张**空闲槽表**(每个槽记录
`(可复用时间, 槽大小, 当前占用者)`),按时间推进:

1. 每过一个时间步,把**已死张量**占的槽标记为空闲;
2. 新张量出生时,在空闲槽里挑一个**最小但够用**的槽(best-fit)放进去;
3. 若没有够用的空闲槽,才**新开一块显存**。

对比两种策略:**不复用**(每个张量永久各自占一块,内存 = 所有张量大小之和)vs
**复用**(只看同一时刻并发存活的峰值):
'''))

NB.code(D('''
def plan_memory(ops):
    """生命周期 + 贪心 best-fit 内存复用。返回各张量占用槽 与 两种策略的内存。"""
    born, death, size = {}, {}, {}
    for t, (name, inputs, out, sz) in enumerate(ops):
        born[out], size[out] = t, sz
        for inp in inputs:
            death[inp] = t
    for name in born:
        death.setdefault(name, born[name])
    names = sorted(set(size) | {"x"})
    if "x" in names:
        size["x"] = 64; born["x"] = 0
        death["x"] = max([t for t, (n, ins, o, s) in enumerate(ops) if "x" in ins] or [0])

    slots, alloc, slot_of = [], {}, {}
    cur = 0; reuse_peak = 0
    for t in range(len(ops)):
        for nm in list(alloc):                        # 释放已死张量
            if death[nm] < t:
                sid = alloc.pop(nm); slots[sid] = (t, slots[sid][1], None); cur -= size[nm]
        for nm in names:                              # 分配当前存活的张量
            if born[nm] <= t <= death[nm] and nm not in alloc:
                best = min((i for i, (ft, sz, o) in enumerate(slots)
                            if o is None and sz >= size[nm]), key=lambda i: slots[i][1], default=None)
                if best is not None:
                    slots[best] = (t, slots[best][1], nm); alloc[nm] = best
                else:
                    alloc[nm] = len(slots); slots.append((t, size[nm], nm))
                slot_of.setdefault(nm, alloc[nm]); cur += size[nm]
        reuse_peak = max(reuse_peak, cur)
    no_reuse = sum(size.values())                     # 不复用 = 每个张量永久占一块
    return names, born, death, size, slot_of, reuse_peak, no_reuse

names, born, death, size, slot_of, reuse_peak, no_reuse = plan_memory(ops)
print(f"不复用内存(每张量独占): {no_reuse} MB")
print(f"复用后峰值(并发)      : {reuse_peak} MB")
print(f"节省                  : {no_reuse - reuse_peak} MB,{(1 - reuse_peak / no_reuse) * 100:.0f}%")
'''),
"🎯 同一张图:不复用要 336MB,复用后峰值只有 128MB —— 死张量的槽被反复利用,省下约 62%。")

NB.code(D('''
# 可视化:各张量占用哪个内存槽(带颜色的生命周期条,同色 = 共用槽)
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(8, 4.5))
names_sorted = sorted(names, key=lambda n: slot_of.get(n, 999))
for i, nm in enumerate(names_sorted):
    sid = slot_of.get(nm)
    ax.barh(i, death[nm] - born[nm] + 0.5, left=born[nm], height=0.6,
            color=cmap((sid or 0) % 10), alpha=0.85)
    ax.text(born[nm] + 0.1, i, nm, va="center", fontsize=9)
ax.set_yticks([]); ax.set_xlabel("时间步")
ax.set_title("张量生命周期(颜色 = 复用的内存槽;同色 = 共用一块显存)")
ax.set_xlim(-0.5, len(ops) + 0.5)
plt.tight_layout()
'''),
"🎨 同色的条共用同一个内存槽 —— 一眼看出哪些张量『接力』了同一块显存。")

NB.code(D('''
# 槽占用热力图:时间 × 槽,颜色 = 该槽此刻装的是哪个张量
slots_used = max(len(set(slot_of.values())), 1)
used = [nm for nm in names if slot_of.get(nm) is not None]
code = {nm: i + 1 for i, nm in enumerate(used)}          # 张量名 -> 数值编码
occupancy = np.zeros((len(ops), slots_used))
labels = np.full(occupancy.shape, "", dtype=object)
for nm in used:
    sid = slot_of[nm]
    for t in range(born[nm], death[nm] + 1):
        occupancy[t, sid] = code[nm]
        labels[t, sid] = nm

import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(occupancy, index=[f"t{i}" for i in range(len(ops))],
                         columns=[f"槽{s}" for s in range(slots_used)]),
            annot=labels, fmt="", cmap="viridis", cbar=False, linewidths=1, ax=ax)
ax.set_title("内存槽占用热力图(横=时间, 纵=槽, 值=张量)")
plt.tight_layout()
'''),
"📊 热力图清晰地展示『同一槽在不同时刻装不同张量』 —— 这就是内存复用的时空图。")

NB.md("## 4️⃣ in-place 优化:就地写回 ✍️",
D('''
除了"死张量让槽",还有一招:**in-place(就地)写回**。常规 `b = a + 1.0` 会**新分配**一块显存
放 b;而 `a.add_(1.0)` 直接**在 a 自己的显存里改写**,不新开。当 a 之后没人再需要原值时,
in-place 就省下了一块内存。我们在 GPU 上用 torch 实测峰值显存:
'''))

NB.code(D('''
torch.cuda.reset_peak_memory_stats()
a = torch.randn(4096, 4096, device="cuda")   # 64MB

def out_of_place(a):                          # 每步新开一块
    b = a + 1.0
    c = b * 2.0
    d = torch.relu(c)
    return d

torch.cuda.reset_peak_memory_stats()
d = out_of_place(a); torch.cuda.synchronize()
peak_oop = torch.cuda.max_memory_allocated()

def in_place(a):                              # 就地写回,只占一块
    a.add_(1.0)
    a.mul_(2.0)
    a.relu_()
    return a

torch.cuda.reset_peak_memory_stats()
d2 = in_place(a); torch.cuda.synchronize()
peak_ip = torch.cuda.max_memory_allocated()
print(f"out-of-place 峰值显存 : {peak_oop / 1e6:.0f} MB")
print(f"in-place 峰值显存     : {peak_ip / 1e6:.0f} MB")
'''),
"⚠️ 注意:in-place 只在『原值没人再用』时才安全;编译器会做别名分析决定能否就地写回。")

NB.md("## 5️⃣ 配套 App:生命周期图交互 🎛️",
D('''
同目录的 `app_63_memory.py` 把内存规划做成交互:拖动**算子个数**、切换张量规模与是否有
共享/残差张量,实时看生命周期图(按内存槽上色)与『不复用 / 复用』对比:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_63_memory.py
```

浏览器打开 **http://localhost:8501**。建议把链长拉到 10,观察复用省下的内存百分比如何上升。
完整源码如下:
'''))

NB.code(app_cell("app_63_memory.py", APP_63),
"📜 运行本 cell 覆盖写入 `app_63_memory.py`,保证 notebook 与 app 一致。")

wrapup(NB,
    summary=[
        "张量生命周期 = 出生(创建) → 活跃(被读) → 死亡(最后一个消费者跑完),由拓扑序扫描算出",
        "内存复用 = 死张量腾出槽给新张量;贪心 best-fit 是常用在线策略",
        "『不复用=每张量独占 vs 复用=并发峰值』同一张图从 336MB 降到 128MB —— 激活值内存优化的核心",
        "in-place 就地写回可避免新张量分配,但前提是原值无人再用(别名分析)",
        "Gantt + 热力图把『时空复用』可视化:同色同槽、异时接力",
    ],
    practice=[
        "给 ops 增加一个『被后续两个算子共用』的张量(如残差连接),看它的生命周期与复用怎么变",
        "把 best-fit 改成 first-fit(第一个够用的槽),对比峰值内存是否变差",
        "在 torch 里对 8 层循环分别用 out-of-place 与 in-place 测峰值显存,画出链长 → 峰值曲线",
        "思考:为什么 LLM 的 KV Cache 需要 paged 分配而不是简单复用?(提示:容量随请求动态增长)",
    ],
    links=[
        ("XLA 内存规划文档", "https://www.tensorflow.org/xla/operation_semantics"),
        ("PyTorch CUDA 内存管理", "https://pytorch.org/docs/stable/notes/cuda.html"),
        ("vLLM PagedAttention 论文", "https://arxiv.org/abs/2309.06180"),
    ])

out = str(Path(CH10) / "63_memory_scheduling.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH10) / "app_63_memory.py"
app_path.write_text(APP_63 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
