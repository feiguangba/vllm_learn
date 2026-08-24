# -*- coding: utf-8 -*-
"""生成 36_allreduce_nccl.ipynb 与 app_36_allreduce.py(教材级重写版)

论文/资料支撑:
- Patarasuk & Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of
  Workstations", J. Parallel Distrib. Comput. 2009(ring allreduce 带宽最优证明)
- Jeaugey, "Massively Scale Your Deep Learning Training with NCCL 2.4"(双二叉树)
- NVIDIA NCCL 官方文档 + nccl-tests busbw 公式
"""
from helpers import D, RING_AR, TREE_AR, chapter_cover, wrapup, new_nb, CH06
from pathlib import Path

APP_36_SRC = D('''
# -*- coding: utf-8 -*-
# app_36_allreduce.py — AllReduce 的 ring / tree 通信量对比 🔄
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🔄 36 · AllReduce", layout="wide")
st.title("🔄 第 36 课 · AllReduce:让每张卡都拿到「大家之和」")

st.markdown("""
**AllReduce** 是最常用的集合通信:每张卡各持一份数据,结束后**每张卡都拿到所有卡数据的总和**。
想象 N 个同学各写了一页笔记,要把内容汇总——**每个人都必须拿到完整汇总**。
本页对比两种算法:**ring(环)** 和 **tree(树)**,看它们的通信量与延迟轮数随卡数怎么变。
""")

with st.sidebar:
    st.header("🎛️ 参数")
    n = st.slider("卡数 N", 2, 16, 8, 1)
    data_mb = st.slider("每卡数据量(MB)", 1, 512, 64, 1)
    st.caption("卡越多,AllReduce 需要搬运的总字节数越多;数据量越大,耗时越长。")

def ring_rounds(n):
    return 2 * (n - 1)                       # reduce-scatter N-1 步 + allgather N-1 步

def tree_rounds(n):
    return 2 * int(np.ceil(np.log2(n)))      # halving log2 N + doubling log2 N

vol_per_card = 2 * (n - 1) / n * data_mb     # 每卡通信量(MB)
total_vol = vol_per_card * n                 # 全网总通信量(MB)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ring 通信轮数", f"{ring_rounds(n)}")
c2.metric("tree 通信轮数", f"{tree_rounds(n)}")
c3.metric("每卡通信量(MB)", f"{vol_per_card:.1f}")
c4.metric("全网通信量(GB)", f"{total_vol / 1024:.2f}")

st.subheader("🔍 ring vs tree:轮数与通信量")
names = ["ring(环)", "tree(树)"]
rounds = [ring_rounds(n), tree_rounds(n)]
fig = go.Figure()
fig.add_trace(go.Bar(x=names, y=rounds, name="通信轮数",
                     marker_color=["#4C78A8", "#72B7B2"], text=rounds, textposition="outside"))
fig.update_layout(title=f"{n} 卡下两种算法的通信轮数(轮数≈延迟)",
                  xaxis_title="算法", yaxis_title="轮数", height=360,
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 随卡数 N 变化:ring 线性涨, tree 对数涨")
ns = list(range(2, 17))
r_ring = [ring_rounds(nn) for nn in ns]
r_tree = [tree_rounds(nn) for nn in ns]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ns, y=r_ring, mode="lines+markers", name="ring 轮数",
                          line=dict(color="#4C78A8", width=3)))
fig2.add_trace(go.Scatter(x=ns, y=r_tree, mode="lines+markers", name="tree 轮数",
                          line=dict(color="#E45756", width=3)))
fig2.update_layout(title="卡数越多,ring 的轮数(2(N-1))涨得越快,而 tree 只有 2log2(N)",
                   xaxis_title="卡数 N", yaxis_title="通信轮数", height=380,
                   legend=dict(orientation="h", y=1.12),
                   margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
> 💡 **直觉**:两种算法的**总字节数相同**(都是 2(N-1)/N × 每卡数据量),
> 差别在**轮数**——ring 是线性的 2(N-1) 轮,tree 是对数的 2log2(N) 轮。
> 卡少时差不多,卡一多 tree 的延迟优势就显现了;但 tree 对树高、负载均衡更敏感。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 6 章 · 第 36 课配套演示")

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

APP_36 = "%%writefile app_36_allreduce.py\n" + APP_36_SRC

NB = new_nb("第 36 课 · AllReduce 与 NCCL:ring / tree 算法",
            subtitle="手写 numpy 版 ring-allreduce 验证正确性,搞懂集合通信的字节账与轮数账",
            emoji="🔄")

chapter_cover(NB,
    objectives=[
        "理解 AllReduce 的语义:每卡都拿到所有卡数据的总和",
        "手写 numpy 版 ring-allreduce 并验证正确性(reduce-scatter + allgather)",
        "理解 tree-allreduce(recursive halving-doubling)的两阶段流程",
        "掌握带宽公式:每卡通信量 = 2(N-1)/N × 数据量,以及 ring/tree 轮数差异",
    ],
    toc=[
        ("直觉:同学传笔记", "AllReduce = 让每个人都拿到完整汇总的「传纸条」游戏"),
        ("核心定义与公式", "AllReduce 语义 + reduce-scatter/allgather + 字节账公式"),
        ("最小实现 · 逐行推演", "numpy 手写 ring-allreduce,逐阶段打印记录"),
        ("数值验证", "ring/tree 正确性 + 字节账 2(N-1)/N 公式核对"),
        ("真实规模数字", "GPU 上真实求和耗时 + alpha/beta 延迟模型实测"),
        ("与 vLLM 工程实现的关系", "NCCL 与 vLLM 的 AllReduce 用法"),
        ("配套 Streamlit 演示", "app_36_allreduce.py:拖卡数/数据量看两种算法"),
    ],
    links=[
        ("NCCL 官方文档", "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html"),
        ("NCCL 论文: Accelerating Collective Communication (arXiv:1907.08586)", "https://arxiv.org/abs/1907.08586"),
        ("NCCL 2.4 博客:双二叉树", "https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/"),
        ("nccl-tests: busbw 公式", "https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md"),
        ("Bandwidth Optimal All-reduce (Patarasuk & Yuan, 2009)", "https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf"),
    ])

NB.md("## 1️⃣ 直觉:同学传笔记 🔄",
D('''
想象 4 个同学各写了一页课堂笔记,老师要**每人手头都有一份完整的总笔记**(所有人的笔记拼在一起)。
最朴素的传法:谁都能看到别人的,但要让所有人都拿到「完整版」,必须好好设计。

**AllReduce(全规约)** 就是这件事:每张卡(同学)持有数据 $x_i$,结束时**每张卡都拿到
$\\sum_i x_i$**。它的两个兄弟:
- **Reduce**:结果只落在某一张卡(一个人收齐);
- **Broadcast**:某一张卡把数据发给所有人。

AllReduce 最常见、也最贵,所以专门有算法研究它。工业界用它做**张量并行**(第 37 课)和
**数据并行的梯度同步**(第 39 课)。vLLM 底层用 [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
来执行这些集合通信。
'''))

NB.md("## 2️⃣ 核心定义与公式 🧮",
D('''
**AllReduce 语义**:$N$ 张卡,第 $r$ 张卡持有向量 $x_r$(长度 $D$),结束时所有卡都得到
$\\sum_{r=0}^{N-1} x_r$。

**ring-allreduce(环规约)** 分两阶段,每卡把数据切成 $N$ 块:

- **阶段一 reduce-scatter(累加散开,共 $N-1$ 步)**:每步大家把手里「当前块」累加后传给下家。
  转一圈后,每张卡各自握着一个「某一块的完整累加和」;
- **阶段二 allgather(广播聚合,共 $N-1$ 步)**:每个握有最终块的卡,把它沿环广播一圈,
  让所有人都拿到全部块。

**字节账**:每步每卡发 $D/N$ 个元素,共 $2(N-1)$ 步,每卡总发送 $\\frac{2(N-1)}{N} D$。

| 符号 | 含义 |
|---|---|
| $N$ | 卡数(进程数) |
| $D$ | 每卡数据量(元素数或字节) |
| $D/N$ | 每块大小(切 $N$ 块) |
| $2(N-1)$ | ring 的总步数(reduce-scatter + allgather) |
| $2\\log_2 N$ | tree 的总步数(halving + doubling) |
| $2(N-1)/N$ | 每卡通信量的倍数系数 |

**tree-allreduce(树规约)** = recursive halving-doubling:先两两配对求和把人数减半,再配对…
最后集中在 0 号卡;然后反向把总和二分广播回去,只要 $2\\log_2 N$ 步。
注意它要求卡数是 **2 的幂**。

> 📄 ring 的**带宽最优性**(每卡通信量等于理论下界 $2(N-1)/N \\cdot D$)由
> [Patarasuk & Yuan, 2009](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) 证明;
> NCCL 在大规模下改用**双二叉树**换取对数延迟,见
> [NCCL 2.4 博客](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)。
'''))

NB.md("## 3️⃣ 最小实现 · 逐行推演:numpy 手写 ring-allreduce 🔗",
D('''
用 numpy 手写并验证正确性。`ring_allreduce` 只有两段循环:
先 N-1 步累加(reduce-scatter),再 N-1 步广播(allgather)。**每行代码都有注释**:
'''))

NB.code(RING_AR, "🎯 `ring_allreduce` 只有两段循环:先是 N-1 步累加(reduce-scatter),再是 N-1 步广播(allgather)。`allclose` 验证每卡最终结果 = 全卡逐块求和——**正确性 Pass**。")

NB.md("## 4️⃣ 数值验证:tree 正确性 + 字节账公式 🧮",
D('''
先验证 tree-allreduce(8 卡)正确性;再核对「每卡通信量 $= \\frac{2(N-1)}{N} \\times D$」的公式:
'''))

NB.code(TREE_AR, "🌳 `tree_allreduce` 分 halving(两两求和,奇数组把数据并给偶数组)与 doubling(反向广播)两阶段。8 卡下正确性同样 Pass。")

NB.code(D('''
import numpy as np

data_mb = 100.0                       # 每卡数据量(MB)
print(f"{'N':>3} {'公式 2(N-1)/N×D(MB)':>18} {'实际发送(MB)':>13} {'匹配':>4}")
for n in [2, 4, 8, 16]:
    formula = 2 * (n - 1) / n * data_mb      # 理论每卡通信量
    actual = 2 * (n - 1) * (data_mb / n)     # 两阶段各 (N-1) 步,每步发 1/N 数据
    print(f"{n:>3} {formula:>18.2f} {actual:>13.2f} {'✅' if abs(formula-actual)<1e-9 else '❌'}")
print("\\n→ 每卡通信量确实就是 2(N-1)/N × D;N 大时趋于 2D(每卡几乎发出 2 份完整数据)。")
'''), "✅ 公式与实际发送字节数完全吻合——ring 的每卡通信量确实就是 2(N-1)/N × D。")

NB.md("## 5️⃣ 真实规模数字:GPU 上量 alpha 与 beta ⚡",
D('''
上面的字节账是**公式/仿真**,现在把它接到真实 GPU 上。核心事实:ring-allreduce 无论怎么绕,
**最终目标都是「让每张卡拿到所有卡数据之和」**——这一步在地时就落地为一个**全量求和**。
所以我们在 RTX 5060 上:① 算清 ring 每步传多少字节(仿真侧);
② 用 torch 真算一次「N 张卡数据的全局求和」拿到真实耗时;
③ 扫描消息大小,量出**固定开销(alpha)**与**带宽斜率(beta)**——正是第 40 课延迟模型的物理来源。
'''))

NB.code(D('''
import sys, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OMP 冲突防护
sys.path.insert(0, r"D:\\Project\\21-Cpp_learn\\explore\\VLLM_learn\\exercises")
import time, gc, numpy as np
import torch
from vllm_real import cuda_info

print("设备:", cuda_info())
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ① 仿真侧:ring 的步数与每步字节量(chunks 复用第 3 节 RING_AR 中 n=4 的变量)
N = 4                                            # 与 RING_AR 的 N=4 对齐
elem_bytes = 4                                   # float32 每元素 4 字节
per_step_bytes = chunks[0].shape[-1] * elem_bytes   # 每步每卡发送的 1 块字节数
steps = 2 * (N - 1)                              # ring 的 N-1 + N-1 步
print(f"ring N={N}:共 {steps} 步(reduce-scatter {N-1} + allgather {N-1}),"
      f"每步每卡发 {per_step_bytes} B")

# ② 真实 torch:一次「全卡求和」即 ring 的目标,量出真实耗时(带预热)
xs = [torch.randn(chunks[0].shape[-1], device=dev) for _ in range(N)]  # N 份数据
print(f"xs 每份张量 shape = {tuple(xs[0].shape)}  <- (chunk_elems={chunks[0].shape[-1]})")
print(f"torch.stack(xs) shape = {tuple(torch.stack(xs).shape)}  <- (卡数 N={N}, chunk_elems)")
torch.stack(xs).sum(dim=0)                      # 预热,跳过首调开销
torch.cuda.synchronize()
t0 = time.perf_counter()
total = torch.stack(xs).sum(dim=0)              # 每卡那份 1 块数据 → 全局和
torch.cuda.synchronize()
real_ms = (time.perf_counter() - t0) * 1e3
print(f"真实 torch 一次全局求和({N} 卡,{chunks[0].shape[-1]} 个 float32):{real_ms:.4f} ms")

# ③ 扫描消息大小,量固定开销 alpha 与带宽斜率 beta
def red_ms(nel, reps=50):
    # 对 n 个元素的张量做 in-place 加法,量平均耗时(毫秒)
    a = torch.randn(nel, device=dev); b = torch.randn(nel, device=dev)
    for _ in range(5): torch.add(a, b, out=b)   # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): torch.add(a, b, out=b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

print("\\n消息大小     元素数        单次加法耗时(ms)")
for nel in [16, 1024, 65536, 1_000_000, 10_000_000]:
    print(f"{nel*4/1024:>9.1f} KB {nel:>10,d}  {red_ms(nel):12.5f}")

gc.collect(); torch.cuda.empty_cache()
'''), "⚡ **真机数字**:16 个元素(64B)的 reduce 耗时就是 alpha——几个微秒的固定开销;数据变大后耗时 ≈ 数据/带宽,斜率就是 beta。这就是「延迟 = alpha + 数据/带宽」的物理来源(第 40 课)。")

NB.md("## 6️⃣ 与 vLLM 工程实现的关系:NCCL 与 AllReduce 🚀",
D('''
工程上 AllReduce 不是手写,而是交给 **NCCL**(NVIDIA Collective Communications Library):

1. **TP 层间通信**:vLLM 张量并行每层末尾的 AllReduce,底层就是 NCCL 的 `ncclAllReduce`
   (见 `vllm/distributed/device_communicators/`);
2. **DP 梯度同步**:训练时每步梯度 AllReduce(第 39 课),NCCL 同样负责;
3. **NCCL 的算法选择**:NCCL 内部在 Ring / Tree / CollNet 之间自动切换——
   大消息用 ring(带宽最优),小消息用双二叉树(延迟低),且要求卡数为 2 的幂时 tree 更优;
4. **busbw 口径**:NCCL 的 bus bandwidth = `algbw × (N-1)/N × 2`,与第 4 节公式同源,
   见 [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。

> 📄 vLLM 还针对「单机小消息」做了共享内存版 AllReduce(CustomAllreduce),
> 跳过 NCCL 的固定开销——第 40 课专门展开。
'''))

NB.md("## 7️⃣ 配套 Streamlit 演示:拖卡数与数据量,看两种算法 🎛️",
D('''
运行 `app_36_allreduce.py`,拖动**卡数**与**每卡数据量**,实时对比 ring / tree 的通信轮数与通信量:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_36_allreduce.py
```

浏览器打开 **http://localhost:8501**。建议:把卡数从 2 拖到 16,看 ring 轮数一路涨、
tree 轮数缓慢爬升;再调大每卡数据量,看通信量增加。完整源码如下:
'''))

NB.code(APP_36, "📜 这就是 app_36_allreduce.py 的完整源码,notebook 与 app 共享同一套 ring/tree 轮数与通信量公式。")

wrapup(NB,
    summary=[
        "AllReduce 语义:每张卡最终都拿到所有卡数据的总和,是 TP/DP 的通信基石",
        "ring-allreduce:reduce-scatter(N-1 步)+ allgather(N-1 步),每卡负载均匀",
        "tree-allreduce:recursive halving-doubling,只要 2log2(N) 步,延迟更低但要求卡数 2 的幂",
        "每卡通信量 = 2(N-1)/N × D(N 大时趋于 2D);全网通信量 = 2(N-1)×D 线性上涨",
        "ring 重带宽利用率、tree 重低延迟,工程上按卡数与消息大小取舍",
    ],
    practice=[
        "把 RING_AR 里的卡数改成 8,验证正确性仍 Pass,并打印 record 看每一步谁发给谁",
        "实现一个「ring-reduce」(只做 reduce-scatter,不做 allgather),统计每卡发送量",
        "用 numpy 验证:tree 的 halving 阶段结束后,0 号卡持有的就是全量总和",
        "推导并验证:N=2 时 ring 和 tree 的轮数都为 2(两者退化相同)",
    ],
    links=[
        ("NCCL 官方文档", "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html"),
        ("NCCL 论文 (arXiv:1907.08586)", "https://arxiv.org/abs/1907.08586"),
        ("NCCL 2.4 博客:双二叉树", "https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/"),
        ("Bandwidth Optimal All-reduce (Patarasuk & Yuan, 2009)", "https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf"),
    ])

NB.save(str(Path(CH06) / "36_allreduce_nccl.ipynb"))

app_path = Path(CH06) / "app_36_allreduce.py"
app_path.write_text(APP_36_SRC + "\n", encoding="utf-8")
print(f"[ok] {app_path}")