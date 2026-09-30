# 第 3 章 · 从静态批到连续批与调度器（ch03）参考答案

> 参考答案，建议先自己动手再看。
>
> 对应笔记本路径（7 本，均在 `exercises/ch03/`）：
> - `14_static_batching_problem.ipynb`、`15_continuous_batching.ipynb`、`16_request_state_machine.ipynb`、`17_scheduler_design.ipynb`、`18_preemption.ipynb`、`19_iterative_scheduler_sim.ipynb`、`20_batch_compare_experiment.ipynb`
>
> 下列代码复用各 notebook 已定义的真实函数（make_reqs / simulate_static / simulate_continuous / simulate_events / Scheduler / simulate_preempt / Simulator / harness_static / harness_continuous 等）与数据类。

## 第 14 课 · 静态批处理的缺陷：队头阻塞与尾部浪费

### 练习 1 · 把 make_reqs 的 mode 改为 poisson、rate 调到 1.5 重跑，观察空转率变化

一句话思路：泊松流下请求到得越密，GPU 就不必干等凑批，空转率随之下降；对比 rate=0.5 与 1.5 即可验证。

```python
for rate in [0.5, 1.5]:
    reqs, batches = simulate_static(make_reqs(n=32, mode="poisson", rate=rate, seed=11), batch_size=8)
    df = reqs_df(reqs)
    ms = float(df["end"].max())
    idle = ms - sum(b["end"] - b["start"] for b in batches)
    print(f"rate={rate}: 空转率={idle/ms*100:.1f}%  利用率={df['work'].sum()/ms*100:.1f}%")
```

预期：rate 越大空转越少（rate=1.5 时空转率明显低于 0.5）；rate 太小会长期空转凑不满批。

### 练习 2 · 给 simulate_static 记录每批内每个请求的「陪跑浪费」并画直方图

一句话思路：陪跑浪费 = 批耗时 − 该请求自己的工作量，短请求被长请求拖慢的那段就是浪费。

```python
import matplotlib.pyplot as plt
reqs, batches = simulate_static(make_reqs(n=32, seed=11), batch_size=8)
df = reqs_df(reqs)
waste = []
for b in batches:
    m = df[df.rid.isin(b["members"])]
    bt = b["end"] - b["start"]
    waste += list(bt - m.work)                       # 批耗时 - 每请求工作量
plt.hist(waste, bins=15, color="#C44E52")
plt.xlabel("陪跑浪费 (步)"); plt.ylabel("请求数"); plt.show()
print(f"平均浪费 {np.mean(waste):.1f} 步, 占总工作量 {np.mean(waste)/df.work.mean()*100:.0f}%")
```

预期：分布右偏，短请求的陪跑浪费大、长请求几乎为零——队头阻塞的代价落在短请求头上。

### 练习 3 · batch_size 扫描时额外记录 P95 延迟，验证「平均延迟会撒谎」

一句话思路：平均延迟被多数短请求拉低，掩盖了部分请求被批内长任务卡住的长尾，P95 涨得比均值快。

```python
for bs in range(1, 11):
    reqs, batches = simulate_static(make_reqs(n=32, seed=11), batch_size=bs)
    df = reqs_df(reqs)
    print(f"batch={bs:2d}: 平均延迟={df.latency.mean():6.1f}  P95={np.percentile(df.latency, 95):6.1f}")
```

预期：平均延迟随 batch 缓涨，P95 的斜率明显更陡——只看均值会低估队头阻塞的伤害。

### 练习 4 · n 从 16 增到 64，观察最优批次大小如何移动

一句话思路：请求越多，最后一批的尾部空座影响越被摊薄，凑大/凑中批次的代价变小，最优批移向更大。

```python
for n in [16, 32, 64]:
    best = None
    for bs in range(1, 11):
        reqs, batches = simulate_static(make_reqs(n=n, seed=11), batch_size=bs)
        df = reqs_df(reqs)
        util = df["work"].sum() / float(df["end"].max())
        if best is None or util > best[1]: best = (bs, util)
    print(f"n={n:3d}: 最优 batch_size={best[0]} (利用率 {best[1]*100:.1f}%)")
```

预期：n 越大最优 batch_size 越大，反映尾部浪费与空转的相对权重被摊薄；本质是「批内浪费 vs 空转」的权衡随规模移动。

---

## 第 15 课 · Continuous Batching：随做随上，不凑桌

### 练习 1 · token_budget 从 8 调到 2 和 32，观察吞吐曲线

一句话思路：预算太小每步只能服务少数请求（饿）；预算太大虽省 prefill 开销但都挤在同一步，收益封顶甚至变差。

```python
for tb in [2, 8, 32]:
    r = make_reqs(n=16, seed=42)
    simulate_continuous(r, token_budget=tb)
    ms = max(x.end for x in r)
    tp = sum(x.prompt_len + x.max_new for x in r) / ms
    lat = np.mean([x.end - x.arrive for x in r])
    print(f"token_budget={tb}: 吞吐={tp:.2f} token/步  平均延迟={lat:.1f}")
```

预期：budget=2 吞吐明显偏低（太饿），=8 均衡，=32 不再提升甚至并发度被挤低（太大也无益）。

### 练习 2 · 给 simulate_continuous 加 max_running 上限（并发数约束）模拟显存受限

一句话思路：复刻 simulate_continuous，只在「拉人」处追加 `len(running) < max_running`（配合原 budget），显存满员就停下。

```python
def simulate_continuous_cap(reqs, token_budget=8, max_running=4):
    queue = sorted(reqs, key=lambda r: r.arrive); running, t = [], 0
    while queue or running:
        for r in running:
            if r.prefilled >= r.prompt_len and r.generated >= r.max_new:
                r.state, r.end = "FINISHED", t
        running = [r for r in running
                   if not (r.prefilled >= r.prompt_len and r.generated >= r.max_new)]
        budget = token_budget
        for r in list(queue):
            if r.arrive <= t and budget > 0 and len(running) < max_running:
                running.append(r); queue.remove(r); budget -= min(prefill_chunk, r.prompt_len)
            if budget <= 0: break
        if not running:
            t += 1; continue
        for r in running:
            if r.prefilled < r.prompt_len: r.prefilled += min(prefill_chunk, r.prompt_len - r.prefilled)
            else: r.generated += 1
        t += 1
    return reqs
```

预期：max_running 越小，同时并发越低，尾部堆积越重，延迟升高——刻画真正显存受限的 vLLM。

### 练习 3 · 实现 prefill_chunk=4 的分块 prefill，观察 TTFT 如何摊平

一句话思路：预填被切成小块后，长 prompt 的首个 token 不必等整段算完，TTFT ≈ start + ceil(prompt/chunk)，chunk 越小越早。

```python
for chunk in [None, 4, 2]:
    r = make_reqs(n=16, seed=42)
    simulate_continuous(r, token_budget=8, prefill_chunk=chunk)
    c = 8 if chunk is None else chunk                  # None 时一次算完整个 prompt
    ttft = [x.start + -(-x.prompt_len // c) for x in r]  # TTFT ≈ 首 token 时刻
    print(f"prefill_chunk={chunk}: 平均TTFT={np.mean(ttft):.1f} 步")
```

预期：chunk=4/2 时大 prompt 的 TTFT 显著提前并被摊平，长 prompt 不再独占「整段 prefill」窗口。

### 练习 4 · 用 avg_active 插值法分别算两种调度「算力利用率」= 平均并发 / 理想并发

一句话思路：平均并发代表 GPU 同时服务的请求数，除以理想并发上限即为算力利用率，静态批因陪跑+空转远低于上限。

```python
def utilization(sim_fn, ideal, n=16, seed=42):
    return avg_active(sim_fn, n, seed) / ideal
u_static = utilization(lambda r: simulate_static(r, batch_size=4), ideal=4)
u_cont   = utilization(lambda r: simulate_continuous(r, token_budget=8), ideal=8)
print(f"静态批算力利用率 = {u_static*100:.1f}%  (理想并发 4)")
print(f"连续批算力利用率 = {u_cont*100:.1f}%  (理想并发 8)")
```

预期：静态批利用率明显偏低（整批陪跑+尾部空座），连续批更接近理想并发，接近 1 的高利用率换来更高真实吞吐。

---

## 第 16 课 · 请求状态机：WAITING → RUNNING → FINISHED

### 练习 1 · 抢占策略从 LIFO 改成「踢掉进度最少请求」，看 PREEMPTED 次数变化

一句话思路：把 `running.pop()`（LIFO 弹最近上车）换成取 progress 最小者，避免频繁重算刚跑很少的请求。

```python
def simulate_events_minp(reqs, token_budget=8, max_running=4):
    for r in reqs: r.total = r.prompt_len + r.max_new
    queue = sorted(reqs, key=lambda r: r.arrive); running, t, trans = [], 0, []
    while queue or running:
        for r in running:
            if r.progress >= r.total: r.state, r.end = "FINISHED", t
        running = [r for r in running if r.progress < r.total]
        if len(running) >= max_running:
            victim = min(running, key=lambda r: r.progress)   # 进度最少者
            running.remove(victim); victim.state = "PREEMPTED"
            trans.append((t, victim.rid, "RUNNING", "PREEMPTED", "显存不足")); queue.insert(0, victim)
        # 其余逻辑沿用 simulate_events
```

（提示：直接复制 simulate_events，仅把 `victim = running.pop()` 换成上面两行即得 min-progress 版。）

```python
_, tl = simulate_events(make_sreqs(12, seed=3))
_, tm = simulate_events_minp(make_sreqs(12, seed=3))
pl = sum(1 for x in tl if x[3]=="PREEMPTED"); pm = sum(1 for x in tm if x[3]=="PREEMPTED")
print(f"LIFO 抢占 {pl} 次, 踢进度最少 {pm} 次")
```

预期：两者抢占次数量级相近，但踢进度最少避免了重复重算刚起步的请求，平均延迟更低、重算浪费更小。

### 练习 2 · 迁移日志加 reason='max_tokens' 的 FINISHED，区分正常结束与长度截断

一句话思路：默认到达 max_new 为正常完成，若是被 max_tokens 上限截断则用不同 reason 标记，日志即可分辨两类终点。

```python
def finalize(req, trans, t, cut=False):
    req.state, req.end = "FINISHED", t
    trans.append((t, req.rid, "RUNNING", "FINISHED",
                  "max_tokens" if cut else "完成"))     # 截断 vs 自然结束
# 在清理完成处:
for r in running:
    if r.progress >= r.total:
        finalize(r, trans, t, cut=True)                  # 已达 max_tokens 上限
```

预期：迁移日志中 FINISHED 出现两种 reason，能统计出多少比例是被长度上限硬切而非自然生成完。

### 练习 3 · 给 SReq 加 cancel 事件（随机取消），观察 ABORTED 参与状态分布

一句话思路：每步以概率 p 把某个排队请求置为 ABORTED 并记迁移，取消请求即从分布中退出。

```python
import random
rng = random.Random(1)
def simulate_events_cancel(reqs, token_budget=8, max_running=4, p=0.02):
    # 同 simulate_events，但在 while 开头加：
    for r in list(queue):
        if rng.random() < p:
            queue.remove(r); r.state, r.end = "ABORTED", t
            trans.append((t, r.rid, "WAITING", "ABORTED", "取消"))
```

（在 aggregate 里把 "ABORTED" 纳入 state_distribution 的计数键即可看到它出现。）

预期：分布堆叠图中 ABORTED 短暂占据一小段水位后消失，边缘请求被取消，整池压力略有释放。

### 练习 4 · 状态分布堆叠图改画成 4 条折线，观察每个池子水位涨落节奏

一句话思路：把 fill_between 堆叠改成逐状态折线，直接看 WAITING 先涨后消、RUNNING 平稳、PREEMPTED 尖峰、FINISHED 单调爬的节奏。

```python
import numpy as np
counts = state_distribution(reqs, trans, int(max(reqs, key=lambda r: r.end).end))
t_axis = np.arange(len(counts))
for s in ["WAITING", "RUNNING", "PREEMPTED", "FINISHED"]:
    plt.plot(t_axis, [c[s] for c in counts], label=s, lw=1.8)
plt.xlabel("时间(步)"); plt.ylabel("请求数"); plt.legend(); plt.show()
```

预期：WAITING 呈「先冲高后消退」，RUNNING 近乎平台，PREEMPTED 是若干尖峰，FINISHED 近乎单调爬升——各自水池的涨落节奏一目了然。

---

## 第 17 课 · Scheduler 设计：token 预算 × 调度策略

### 练习 1 · 给 Scheduler 加 WRR（权重轮询）策略，让低优先级按 2:1 权重轮流插入

一句话思路：在 pick_next 里按优先级分组，高优先级每组取 2 个、低优先级取 1 个，轮流取，保证低优先级间歇获得算力、缓解饥饿。

```python
def pick_next_wrr(candidates, w=(2, 1)):
    hp = sorted([c for c in candidates if c.priority == 0], key=lambda r: r.arrive)
    lp = sorted([c for c in candidates if c.priority != 0], key=lambda r: r.arrive)
    out, i, j = [], 0, 0
    while i < len(hp) or j < len(lp):
        for _ in range(w[0]):
            if i < len(hp): out.append(hp[i]); i += 1
        for _ in range(w[1]):
            if j < len(lp): out.append(lp[j]); j += 1
    return out
# 在 Scheduler.pick_next 末尾加：
# if self.policy == "wrr": return pick_next_wrr(candidates)
```

（配合 run_policy 用 priority 字段造请求，SReq(priority=0/1) 区分两档。）

预期：priority 独占资源时低优先级会饿死；WRR 下低优先级获得周期执行，饥饿缓解，代价是高优先级延迟略有上升。

### 练习 2 · run_policy 里统计 SJF 策略下「最长请求的等待时间」

一句话思路：SJF 贪最短剩余，最长请求必然被拖到最后，其 start−arrive 几乎逼近 makespan，量化被拖延的程度。

```python
def run_policy_wait(policy, n=16, token_budget=8, seed=11):
    rng = np.random.default_rng(seed)
    reqs = [SReq(i, 0.0, int(rng.integers(2,16)), int(rng.integers(2,12)), int(rng.integers(0,3)))
            for i in range(n)]
    for r in reqs: r.total = r.prompt_len + r.max_new
    s = Scheduler(policy=policy, token_budget=token_budget); step = 0
    while not all(r.state == "FINISHED" for r in reqs) and step < 6000:
        s.schedule([r for r in reqs if r.state == "WAITING" and r.arrive <= step]); step += 1
    return reqs, max(reqs, key=lambda r: r.total)
_, lg = run_policy_wait("sjf")
print(f"SJF 下最长请求(total={lg.total})等待了 {lg.start - lg.arrive:.1f} 步")
```

预期：SJF 把最长请求的等待拖到接近整个 makespan，说明长任务在 SJF 下被系统性拖延（长尾巨痛）。

### 练习 3 · 第 7 节的双扫描画成热力图（预算 × 策略 × 平均延迟）

一句话思路：把 run_policy_at 铺成二维矩阵，x=预算、y=策略，用颜色标平均延迟，最快定位最优（低延迟）区间。

```python
import matplotlib.pyplot as plt
budgets = [1, 2, 4, 8, 16]; policies = ["fcfs", "sjf", "priority"]
grid = np.array([[run_policy_at(p, b) for b in budgets] for p in policies])
plt.imshow(grid, cmap="YlGnBu", aspect="auto"); plt.colorbar()
plt.xticks(range(5), budgets); plt.yticks(range(3), policies)
for i in range(3):
    for j in range(5): plt.text(j, i, f"{grid[i,j]:.0f}", ha="center", color="k")
plt.xlabel("token 预算"); plt.ylabel("策略"); plt.show()
```

预期：低预算列延迟爆炸式红，预算升到中等后各策略趋于收敛，右上角（中高预算 × SJF/fcfs）是最优区间。

### 练习 4 · 打印 budget=1 与 budget=32 的请求执行顺序

一句话思路：budget=1 每步只处理 1 个 token，长请求独占很久、后到者等待≈整段墙钟；budget 大时请求快速按序完成。

```python
def order(budget):
    s = Scheduler(policy="fcfs", token_budget=budget)
    reqs = [SReq(i, 0.0, 2 + i, 1) for i in range(6)]
    for r in reqs: r.total = r.prompt_len + r.max_new
    step = 0
    while not all(r.state == "FINISHED" for r in reqs) and step < 200:
        s.schedule([r for r in reqs if r.state == "WAITING" and r.arrive <= step]); step += 1
    return [r.rid for r in sorted(reqs, key=lambda r: r.start)]
print("budget=1 执行顺序:", order(1))
print("budget=32 执行顺序:", order(32))
```

预期：budget=1 时后续请求被前面长请求的逐 token 推进完全堵死，等待近乎整段墙钟；=32 时一次放行、几乎同时完成——体现「等待 = 墙钟被低预算拖延」。

---

## 第 18 课 · 抢占：显存不够时，recompute 还是 swap？

### 练习 1 · 抢占策略从 LIFO 改「踢进度最少请求」，比较抢占次数与平均延迟

一句话思路：把 simulate_preempt 的 `victim = running.pop()` 换成取 progress 最小者，减少对刚起步请求的反复重算。

```python
def simulate_preempt_minp(reqs, max_running=4, token_budget=8, mode="recompute", swap_cost=3.0):
    # 复制 simulate_preempt，仅改抢占选 victim 一行：
    #   victim = min(running, key=lambda r: r.progress); running.remove(victim)
    ...

f = lambda rr: np.mean([x.end - x.arrive for x in rr])
r1, e1 = simulate_preempt(make_preq(12, seed=3), max_running=3)
r2, e2 = simulate_preempt_minp(make_preq(12, seed=3), max_running=3)
print(f"LIFO: 抢占{len(e1)}次 延迟{f(r1):.1f} | 踢最少: 抢占{len(e2)}次 延迟{f(r2):.1f}")
```

预期：抢占次数可能相近，但踢进度最少避免了重算刚跑的请求，平均延迟更低、浪费更小。

### 练习 2 · 实现 swap 的「断点续算」——被抢请求恢复时从 progress 处继续

一句话思路：recompute 清零 progress 从头重算；swap 保留 progress 只付搬运费，恢复即续算，省掉整段重算。

```python
# simulate_preempt 抢占分支本就按 mode 区分：
if mode == "recompute":
    victim.progress = 0            # 从头重算（浪费最大）
else:
    t += swap_cost                 # swap：保留 progress，仅付搬运代价
# 验证断点续算省多少: 高压力+长 prompt 下比较 makespan
res = compare_policies(make_preq(12, seed=3, plen=(20,40)), max_running=2, swap_cost=1.0)
print(f"长prompt高压: recompute延迟={res['recompute']['lat']:.1f}  swap延迟={res['swap']['lat']:.1f}")
```

预期：swap 保留进度恢复、只付 swap_cost，长 prompt 高压力场景下比反复全量重算的 recompute 更省、延迟更低。

### 练习 3 · 用第 8 节公式扫描 PCIe 带宽 8GB/s→200GB/s，画 recompute/swap 换手点

一句话思路：recompute_ms = p_tok×t_token 与带宽无关，swap_ms 随带宽反比下降，两条线交叉处的带宽即换手点。

```python
import numpy as np
L, kv_heads, head_dim, dt = 32, 8, 128, 2
r = bench_prefill_decode(L=128, steps=48, reps=5)
t_token = r["prefill_ms"] / r["L"]; p_tok = 2048
kv_b = 2 * L * kv_heads * head_dim * dt
bws = np.linspace(8e9, 200e9, 100)
rec = p_tok * t_token
swap = 2 * (p_tok * kv_b) / bws * 1e3
cross = bws[np.argmin(np.abs(swap - rec))]
plt.plot(bws/1e9, swap, label="swap"); plt.axhline(rec, color="r", label="recompute")
plt.xlabel("PCIe 带宽 (GB/s)"); plt.ylabel("ms"); plt.legend(); plt.show()
print(f"换手点带宽 ≈ {cross/1e9:.0f} GB/s")
```

预期：带宽越大 swap 越便宜，曲线在某处与 recompute 常值相交；带宽高于换手点则 swap 更优，低于则 recompute 更优。

### 练习 4 · simulate_preempt 加 prefix_hit 参数（命中共享前缀缓存的比例）

一句话思路：命中缓存的 prompt 前缀无需重算，recompute 只需重算命中比例之外的部分，命中比越高重算代价被摊薄。

```python
def simulate_preempt_prefix(reqs, max_running=4, token_budget=8, mode="recompute", swap_cost=3.0, prefix_hit=0.0):
    # 在 recompute 抢占分支改：
    if mode == "recompute":
        victim.progress = int(min(victim.progress, victim.total * (1 - prefix_hit)))
    ...

for ph in [0.0, 0.5, 0.9]:
    rr, _ = simulate_preempt_prefix(make_preq(12, seed=3), max_running=3, prefix_hit=ph)
    print(f"prefix_hit={ph}: 平均延迟={np.mean([x.end-x.arrive for x in rr]):.1f}")
```

预期：prefix_hit 越高，被抢请求需重算的 token 越少，recompute 的代价被肉眼可见地摊薄，延迟下降。

---

## 第 19 课 · 完整调度模拟器：参数敏感性实验

### 练习 1 · 把 Simulator 抢占策略参数化（preempt_policy='lifo'/'shortest'），重做敏感性 III

一句话思路：给 Simulator 加 preempt_policy 参数，run() 的抢占分支按它选 victim，再复跑策略×显存热力图观察对抢占与延迟的影响。

```python
class Simulator:
    def __init__(self, policy="fcfs", token_budget=8, max_running=None,
                 prefill_chunk=None, rate=0.5, mode="burst", n=24, seed=1, preempt_policy="lifo"):
        ...; self.preempt_policy = preempt_policy
    # run() 抢占分支：
    if self.max_running and len(running) >= self.max_running:
        if self.preempt_policy == "shortest":
            victim = min(running, key=lambda r: r.progress); running.remove(victim)
        else:
            victim = running.pop()
        victim.state, victim.preempts = "PREEMPTED", victim.preempts + 1
        self.events.append((t, victim.rid)); victim.progress = 0; queue.insert(0, victim)

for pp in ["lifo", "shortest"]:
    m = Simulator(policy="fcfs", token_budget=32, max_running=3, n=40, seed=2, preempt_policy=pp).run()
    print(f"{pp}: 抢占{m['n_preempt']}次 平均延迟={m['avg_latency']:.1f}")
```

预期：shortest 不反复踢刚起步的请求，同压力下抢占更理智、平均延迟更低，多维度敏感性更稳。

### 练习 2 · λ 扫描里同时记录 P95，验证「平均延迟先升」之前 P95 已爆

一句话思路：run() 后请求都已带 end，可单独取各请求延迟算 P95；P95 比均值更早进入爆炸区。

```python
def p95_sim(lam):
    s = Simulator(policy="fcfs", token_budget=8, mode="poisson", rate=lam, n=60, seed=1)
    m = s.run()
    p95 = np.percentile([r.end - r.arrive for r in s.reqs], 95)
    return m, p95
for lam in [0.4, 0.6, 0.8, 1.0]:
    m, p95 = p95_sim(lam)
    print(f"λ={lam}: 平均延迟={m['avg_latency']:6.1f}  P95={p95:6.1f}")
```

预期：平均延迟还在缓升时 P95 已急剧抬升——长尾先于均值失守，说明斜率拐点要盯 P95 而非平均值。

### 练习 3 · 实现 warmup 预热——给 Simulator 加 per-request「思考时间」

一句话思路：给每个请求加 think 字段，先消耗 think 再推进 progress，思考时间吃掉算力使有效容量下降、饱和 λ 提前。

```python
class Simulator:
    def _make_reqs(self, ...):
        ...; reqs.append(...); req.think = int(rng.integers(0, 5))   # 预热思考步数
    # run() 推进分支：
    for r in running:
        if r.think > 0: r.think -= 1               # 先消耗思考时间
        else: r.progress += self.step_tokens(r)    # 再推进实际进度

for lam in [0.4, 0.6, 0.8]:
    m = Simulator(policy="fcfs", token_budget=8, mode="poisson", rate=lam, n=60, seed=1).run()
    print(f"λ={lam}: 吞吐={m['throughput']:.2f} token/步")
```

预期：加入思考时间后有效吞吐天花板下降，到达率饱和点提前（更小 λ 就开始掉头），预热越长饱和越早。

### 练习 4 · 把第 7 节换算做成函数，输入模拟吞吐直接输出真实 token/s

一句话思路：把 `token/步 × (ms/步/1000)⁻¹` 抽成函数，输入模拟吞吐与 batch，查曲线 ms 后输出真实 token/s，方便复用。

```python
def to_real_tok_per_s(tok_per_step, batch=8, curve=None):
    ms = (curve or {}).get(batch, 1.0)
    return tok_per_step / (ms / 1000)

b_list, tps, mps = bench_throughput_curve(batch=(1,2,4,8,16,32), token_len=16, reps=5)
curve = dict(zip(b_list, mps))
sim = Simulator(seed=1).run()
print(f"模拟 {sim['throughput']:.1f} token/步 -> 真实 {to_real_tok_per_s(sim['throughput'], 8, curve):.0f} token/s")
```

预期：同一函数可代入任意模拟吞吐与批次，直接换算真实 token/s，把模拟的风洞结果落地为真机预估。

---

## 第 20 课 · 真实 GPU 对比：静态批 vs Continuous Batching

### 练习 1 · harness_static 的 prefill 改成「整段 prompt 一次喂入」（真 prefill），对比 TTFT

一句话思路：原实现逐 token 喂 prefill（TTFT 需等 max_pl 步），改成把整段 prompt 一步并行前向，TTFT 从 O(max_pl) 降为 O(1)。

```python
def harness_static_trueprefill(model, reqs, batch_size, device):
    for start in range(0, len(reqs), batch_size):
        batch = reqs[start:start+batch_size]; B = len(batch)
        max_pl = max(r["prompt_len"] for r in batch)
        max_work = max(r["prompt_len"] + r["max_new"] for r in batch)
        kv_k = [torch.zeros(B, model.heads, max_work+1, model.head_dim, device=device) for _ in model.layers]
        kv_v = [torch.zeros(B, model.heads, max_work+1, model.head_dim, device=device) for _ in model.layers]
        # 整段 prompt 拼成 (B, max_pl) 一步前向（需 TinyLM 加一个并行 prefill 前向）
        tids = torch.tensor([[batch[i]["rid"] % model.vocab] * max_pl for i in range(B)], device=device)
        seq  = torch.full((B,), max_pl, dtype=torch.long, device=device)
```

（TinyLM 需新增一次算完整 prompt 的 prefill 前向。）

预期：真 prefill 下首批首 token（TTFT）从「等 max_pl 步」缩短到接近一步，首个 token 显著提前，批内排队被后续翻转。

### 练习 2 · harness_continuous 里给每请求随机生成不同 token 序列，消除「占位 token」影响

一句话思路：`tids = rid % vocab` 是固定占位 token，换成每请求随机不同 token，消除规律性占位对测量公平性的干扰。

```python
# 原:
# tids = torch.tensor([[slots[s]["rid"] % model.vocab] for s in active], device=device)
# 改为每请求随机独立 token 序列：
rng = torch.Generator(device=device).manual_seed(0)
tids = torch.randint(0, model.vocab, (len(active), 1), device=device, generator=rng)
```

预期：占位 token 带来的确定性偏差被消除，各请求 token 分布更贴近真实生成，测得的吞吐/加速比更能反映调度差异本身。

### 练习 3 · max_running 从 2 扫到 16，画连续批吞吐曲线，找 TinyLM 饱和 batch

一句话思路：跑 harness_continuous 的 max_running 扫描，吞吐先随并发升后趋于平台，平台起点即 TinyLM 的饱和 batch。

```python
results = []
for mr in [2, 4, 8, 12, 16]:
    model = TinyLM(vocab=256, hidden=64, layers=2, heads=4).to(device)
    ms, tok = harness_continuous(model, reqs, max_running=mr, device=device)
    results.append((mr, tok / (ms / 1000)))
    print(f"max_running={mr:2d}: {tok/(ms/1000):8.0f} token/s")
plt.plot([r[0] for r in results], [r[1] for r in results], "o-"); plt.show()
```

预期：吞吐随 max_running 上升后进入平台（TinyLM 在该 hidden 下饱和 batch 约为中档值），继续加大并发收益趋零。

### 练习 4 · 用更大 hidden=128 重跑主实验，观察加速比是否变大

一句话思路：hidden 越大单请求前向越贵，批内带跑/陪跑的浪费被放大，连续批「完成即补位」的优势被放大，加速比应上升。

```python
for bs in [2, 4, 8]:
    m1 = TinyLM(vocab=256, hidden=128, layers=2, heads=4).to(device)
    ms_s, _ = harness_static(m1, reqs, batch_size=bs, device=device)
    m2 = TinyLM(vocab=256, hidden=128, layers=2, heads=4).to(device)
    ms_c, _ = harness_continuous(m2, reqs, max_running=bs, device=device)
    print(f"hidden=128 batch={bs}: 加速比 {ms_s/ms_c:.2f}×")
```

预期：hidden 增大后静态批整批陪跑代价更重、连续批省下的浪费更大，加速比（ms_s/ms_c）比 hidden=64 时明显变大。