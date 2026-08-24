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
