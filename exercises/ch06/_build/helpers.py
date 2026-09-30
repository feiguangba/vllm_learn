# -*- coding: utf-8 -*-
"""ch06 生成公共组件:Notebook 构建 + 各课共享的模拟器源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH06 = r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch06"
CHAPTER = "第 6 章 · 分布式并行"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 35 课:显存估算
MEM_ESTIMATE = D('''
def estimate_weights_gb(params_b, dtype_bytes=2):
    """模型权重显存(GB):参数量 × 每参数字节数( fp16=2, int8=1, int4=0.5 )"""
    return params_b * dtype_bytes


def estimate_kv_gb(layers, kv_heads, head_dim, ctx_len=4096, batch=1, dtype_bytes=2):
    """KV cache 显存(GB):每 token KV 字节数 = 2×L×h_kv×d_head×b,再乘 token 总数。
    L=层数, h_kv=KV 头数(GQA 后共享), d_head=每头维度, b=每元素字节。"""
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes  # 每 token KV 字节
    return bytes_per_token * ctx_len * batch / 1024 ** 3              # 总字节 → GB


# 各模型真实配置(层数 L, KV 头数 h_kv, 每头维度 d_head):取自官方 config.json
for name, (p, L, hkv, d) in {
    "Qwen3-1.7B":  (1.7, 28, 4, 128),
    "Llama-3-8B":  (8, 32, 8, 128),
    "Qwen2.5-32B": (32, 64, 8, 128),
    "Llama-3-70B": (70, 80, 8, 128),
    "Llama-3.1-405B": (405, 126, 8, 128),
}.items():
    w = estimate_weights_gb(p, 2)
    kv = estimate_kv_gb(L, hkv, d, 4096, 8)
    print(f"{name:>16}: 权重 {w:7.1f} GB + KV(4k×8并发) {kv:6.1f} GB = {w + kv:7.1f} GB")
''')

# ---------------------------------------------------------------- 36 课:ring / tree allreduce
RING_AR = D('''
import numpy as np

def ring_allreduce(chunks_per_rank, record=None):
    """numpy 手写 ring-allreduce(每卡数据已切成 N 块)。
    阶段一 reduce-scatter(N-1 步):第 r 步,每卡把手里的一块累加后发给下家;
    阶段二 allgather(N-1 步):把各自算好的"最终块"沿环广播一圈。
    chunks_per_rank: list of N 个数组,每个数组 shape (N, ...),第 i 行是第 i 块。
    record: 可选 list,记录 (阶段, 步, 发送卡, 接收卡, 块号)。
    """
    n = len(chunks_per_rank)
    for step in range(n - 1):                       # ---- reduce-scatter
        snap = [c.copy() for c in chunks_per_rank]  # 本步开始时各卡手里的块(同步语义)
        for r in range(n):
            src, dst = (r - 1) % n, r               # 上一家发给我
            blk = (r - step) % n                    # 我这一步处理的块号
            if record is not None:
                record.append(("RS", step, src, dst, blk))
            chunks_per_rank[r][blk] = snap[r][blk] + snap[src][blk]
    for step in range(n - 1):                       # ---- allgather
        snap = [c.copy() for c in chunks_per_rank]
        for r in range(n):
            src, dst = (r - 1) % n, r
            blk = (r + 1 - step) % n
            if record is not None:
                record.append(("AG", step, src, dst, blk))
            chunks_per_rank[r][blk] = snap[src][blk]
    return chunks_per_rank


def make_chunks(rng, n, chunk_elems):
    """造 n 张卡,每卡持有 n 块数据(每块 chunk_elems 个 float32)"""
    return [rng.normal(size=(n, chunk_elems)).astype(np.float32) for _ in range(n)]


rng = np.random.default_rng(0)
n = 4
chunks = make_chunks(rng, n, 1000)
answer = np.stack(chunks).sum(axis=0)               # 正确答案:全卡逐块求和 (n, chunk_elems)
result = ring_allreduce([c.copy() for c in chunks])
# 每卡最终结果都应等于整张求和表;float32 求和顺序不同会有 ~1e-6 舍入差,故给一个宽松容差
ok = all(np.allclose(result[r], answer, atol=1e-4, rtol=1e-4) for r in range(n))
print(f"ring-allreduce 正确性: {'✅ Pass' if ok else '❌ Fail'}")
print("每卡拿到完整求和结果:", all(np.allclose(result[r], result[0], atol=1e-4, rtol=1e-4) for r in range(n)))
''')

TREE_AR = D('''
def tree_allreduce(data_per_rank, record=None):
    """numpy 手写 tree-allreduce(recursive halving-doubling):
    阶段一 halving: 距离为 2^k 的伙伴两两配对求和, log2(N) 步后 0 号卡拿到总和;
    阶段二 doubling: 沿反方向把总和二分广播回去, 再 log2(N) 步。
    data_per_rank: list of N 个同 shape 数组。"""
    n = len(data_per_rank)
    assert n & (n - 1) == 0, "tree 版要求卡数是 2 的幂"
    d = [x.copy() for x in data_per_rank]
    dist = 1
    while dist < n:                                  # ---- halving: 两两求和
        for r in range(n):
            if (r // dist) % 2 == 1:                 # 奇数组把数据发给伙伴
                if record is not None:
                    record.append(("HALV", dist, r, r - dist))
                d[r - dist] = d[r - dist] + d[r]
        dist *= 2
    dist = n // 2
    while dist >= 1:                                 # ---- doubling: 广播总和
        for r in range(n):
            if (r // dist) % 2 == 1:
                if record is not None:
                    record.append(("DOUB", dist, r - dist, r))
                d[r] = d[r - dist]
        dist //= 2
    return d


rng = np.random.default_rng(1)
n = 8
data = [rng.normal(size=500).astype(np.float32) for _ in range(n)]
answer = np.stack(data).sum(axis=0)
result = tree_allreduce(data)
ok = all(np.allclose(result[r], answer) for r in range(n))
print(f"tree-allreduce 正确性(8 卡): {'✅ Pass' if ok else '❌ Fail'}")
''')

# ---------------------------------------------------------------- 37 课:张量并行前向
TP_LIN = D('''
import numpy as np

rng = np.random.default_rng(42)

def column_parallel_linear(x, W_full, tp=2):
    """列切:W_full (in, out) 沿 out 维切成 tp 份,每卡算 X @ W_i 得到输出的不同列。
    每卡输出需要 all-gather 才能拼成完整结果(或留给行切层消化)。"""
    outs = []
    for w_shard in np.split(W_full, tp, axis=1):
        outs.append(x @ w_shard)                     # 各卡: (batch, out/tp)
    return outs

def row_parallel_linear(xs, A_full, tp=2):
    """行切:A_full (in, out) 沿 in 维切成 tp 份,与列切配对:
    每卡算 X_i @ A_i 得到 (batch, out) 的部分和,再 allreduce 相加。"""
    A_shards = np.split(A_full, tp, axis=0)
    partial = sum(x @ a for x, a in zip(xs, A_shards))   # allreduce = 部分和相加
    return partial

# 验证:列切+行切 组合 == 一次完整前向
B, IN, HID, OUT = 4, 8, 16, 6
X = rng.normal(size=(B, IN))
W = rng.normal(size=(IN, HID))
A = rng.normal(size=(HID, OUT))

hidden_shards = column_parallel_linear(X, W, tp=2)   # 两卡各得一半隐藏维
Y_tp = row_parallel_linear(hidden_shards, A, tp=2)   # 行切 + allreduce
Y_ref = (X @ W) @ A                                  # 单卡稠密参考

print("TP2 前向 vs 单卡稠密:", "✅ 一致" if np.allclose(Y_tp, Y_ref, atol=1e-5) else "❌ 不一致")
''')

TP_BLOCK = D('''
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

def mlp_column_row_shard(x, W_up, W_down, tp=2, events=None):
    """Megatron 式 MLP:up_proj 列切 → 各卡本地 GELU → down_proj 行切 → allreduce。
    x: (B, in)。返回完整输出 (B, out),events 记录通信时机。"""
    up_shards = np.split(W_up, tp, axis=1)           # 列切
    down_shards = np.split(W_down, tp, axis=0)       # 行切
    if events is not None:
        events.append(("计算: 各卡 X@W_up_i + GELU", None))
    partial = None
    for i in range(tp):
        h = gelu(x @ up_shards[i])                   # 卡 i 本地计算
        p = h @ down_shards[i]                       # 卡 i 部分结果
        partial = p if partial is None else partial + p
    if events is not None:
        events.append(("通信: AllReduce(部分和相加)", tp))
    return partial

B, IN, HID, OUT = 4, 64, 128, 64
X = rng.normal(size=(B, IN))
W_up, W_down = rng.normal(size=(IN, HID)), rng.normal(size=(HID, OUT))
events = []
Y_tp = mlp_column_row_shard(X, W_up, W_down, tp=2, events=events)
Y_ref = gelu(X @ W_up) @ W_down

print("TP2 MLP vs 稠密参考:", "✅ 一致" if np.allclose(Y_tp, Y_ref, atol=1e-4) else "❌ 不一致")
print("\\n一次 MLP 前向的时间线:")
for name, _ in events:
    print("  ▸", name)
''')

# ---------------------------------------------------------------- 38 课:流水线调度
PP_SCHED = D('''
def _run_pipeline(order_per_stage, p, f=1.0, b=2.0):
    """事件驱动流水线模拟器:每个舞台按自己的固定顺序执行 (mb, "F"/"B") 队列,
    F(mb) 要等上游舞台同 mb 的 F 完成,B(mb) 要等下游同 mb 的 B 完成。
    返回事件列表 (stage, mb, "F"/"B", start, end)。"""
    t0 = [0.0] * p                       # 每个舞台的空闲时刻
    fin = {}                             # (stage, mb, 类型) → 完成时刻
    ptr = [0] * p                        # 每个舞台执行到队列第几个
    events = []
    remaining = sum(len(q) for q in order_per_stage)
    while remaining:
        progressed = False
        for s in range(p):               # 每轮扫描:能跑的下一步就跑
            if ptr[s] >= len(order_per_stage[s]):
                continue
            mb, typ = order_per_stage[s][ptr[s]]
            if typ == "F":
                dep = fin.get((s - 1, mb, "F"), 0.0) if s > 0 else 0.0
            else:
                if s < p - 1:
                    if (s + 1, mb, "B") not in fin:
                        continue         # 下游同 mb 的 B 还没排到,本轮到下一圈再处理
                    dep = fin[(s + 1, mb, "B")]
                else:
                    dep = 0.0            # 最后一个舞台的反向无下游依赖
            cost = f if typ == "F" else b
            start = max(t0[s], dep)
            events.append((s, mb, typ, start, start + cost))
            fin[(s, mb, typ)] = start + cost
            t0[s] = start + cost
            ptr[s] += 1
            remaining -= 1
            progressed = True
        assert progressed, "调度死锁(检查舞台顺序是否合法)"
    return events


def schedule_gpipe(p, m, f=1.0, b=2.0):
    """GPipe:每舞台先做完所有 m 个前向,再做反向(倒序 micro-batch,呈 U 形)。"""
    orders = []
    for s in range(p):
        orders.append([(mb, "F") for mb in range(m)] +
                      [(mb, "B") for mb in reversed(range(m))])
    return _run_pipeline(orders, p, f, b)


def schedule_1f1b(p, m, f=1.0, b=2.0):
    """1F1B(PipeDream-Flush):舞台 s 先做 warmup=min(p-1-s, m) 个前向,
    稳态期 1 前向 1 反向交替,收尾清剩余反向 → 激活驻留从 m 降到 ≈ p。"""
    orders = []
    for s in range(p):
        w = min(p - 1 - s, m)
        q = [(mb, "F") for mb in range(w)]
        for i in range(m - w):           # 稳态:F 一个、B 一个
            q.append((w + i, "F"))
            q.append((i, "B"))
        for i in range(m - w, m):        # 收尾:只剩 B
            q.append((i, "B"))
        orders.append(q)
    return _run_pipeline(orders, p, f, b)


def stage_peak_inflight(events, s):
    """舞台 s 峰值驻留激活数:前向把激活留下(+1),反向才释放(-1)。"""
    cur = peak = 0
    for s_, mb, typ, st, en in sorted([e for e in events if e[0] == s],
                                      key=lambda e: e[3]):
        cur += 1 if typ == "F" else -1
        peak = max(peak, cur)
    return peak


def gantt_stats(events, p):
    """总时长 / 气泡占比 / 全部舞台的最大峰值驻留"""
    makespan = max(e[4] for e in events)
    busy = sum(e[4] - e[3] for e in events)
    peak = max(stage_peak_inflight(events, s) for s in range(p))
    return dict(makespan=makespan, bubble=1 - busy / (p * makespan), peak_inflight=peak)


ev_g = schedule_gpipe(4, 8)
ev_1 = schedule_1f1b(4, 8)
print("GPipe 4 舞台 × 8 微批:", gantt_stats(ev_g, 4))
print("1F1B  4 舞台 × 8 微批:", gantt_stats(ev_1, 4))
''')

PP_BUBBLE = D('''
# 验证气泡公式:bubble = (p-1)/(m+p-1)
# (理想化假设:各舞台 f、b 相同;总时长 = (m+p-1)*(f+b),气泡 = (p-1)*(f+b),
#  比例恒为 (p-1)/(m+p-1),与 f、b 取值无关。)
print(f"{'p':>3} {'m':>3} {'模拟气泡':>9} {'公式(p-1)/(m+p-1)':>17} {'匹配':>5}")
for p in [2, 4, 8]:
    for m in [4, 8, 16, 32]:
        st = gantt_stats(schedule_1f1b(p, m), p)
        formula = (p - 1) / (m + p - 1)
        match = "✅" if abs(st["bubble"] - formula) < 1e-9 else "❌"
        print(f"{p:>3} {m:>3} {st['bubble']:>9.3f} {formula:>17.3f} {match:>5}")
''')

# ---------------------------------------------------------------- 39 课:数据并行
DP_LB = D('''
import numpy as np
from dataclasses import dataclass

@dataclass
class Req:
    rid: int
    arrive: float
    work: float                       # 处理耗时(时长不均匀 → 考验负载均衡)

def make_stream(n=24, rng=None, wmin=2, wmax=30):
    rng = rng or np.random.default_rng(0)
    t = 0.0
    reqs = []
    for i in range(n):
        t += rng.exponential(0.8)
        reqs.append(Req(i, t, float(rng.uniform(wmin, wmax))))
    return reqs


def dispatch_dp(reqs, n_replica, policy="least_loaded"):
    """推理 DP 的请求分发:每个副本是完整的模型副本,各自独立出答案。
    policy = "round_robin"  轮询:1号→2号→…→N号→1号…
             "least_loaded" 最短队列:总是把请求派给当前排队最短的副本
             "random"       随机分发(反例:容易不均)"""
    loads = [0.0] * n_replica         # 每个副本当前的工作堆积
    lanes = [[] for _ in range(n_replica)]
    rr = 0
    for r in reqs:
        if policy == "round_robin":
            k = rr % n_replica; rr += 1
        elif policy == "least_loaded":
            k = int(np.argmin(loads))
        else:
            k = int(np.random.randint(n_replica))
        lanes[k].append(r)
        loads[k] = max(loads[k], r.arrive) + r.work     # 简化:排队 + 执行
    return lanes


def dp_stats(lanes):
    ends = [r for lane in lanes for r in lane]
    # 每个请求的完成时间 = 它所在 lane 上依次累加
    fin, lats = [], []
    for lane in lanes:
        t = 0.0
        for r in lane:
            start = max(t, r.arrive)
            fin.append(start + r.work)
            lats.append(start + r.work - r.arrive)
            t = start + r.work
    total_work = [sum(r.work for r in lane) for lane in lanes]
    return dict(makespan=max(fin), avg_lat=float(np.mean(lats)),
                p95_lat=float(np.percentile(lats, 95)),
                imbalance=float(np.std(total_work) / np.mean(total_work)))   # 不均衡度


rng = np.random.default_rng(3)
stream = make_stream(24, rng)
for pol in ["round_robin", "least_loaded", "random"]:
    lanes = dispatch_dp([Req(r.rid, r.arrive, r.work) for r in stream], 4, pol)
    print(f"{pol:>12}: {dp_stats(lanes)}")
''')

DP_GRAD = D('''
# 训练 DP 的梯度同步模拟:3 个副本各自在本地 batch 上算梯度,
# AllReduce 求平均后,应与"单卡吃满全量数据"算出的梯度完全一致(数学可证!)
import numpy as np

rng = np.random.default_rng(7)
N, D = 60, 5                                   # 60 个样本,5 维特征
X_all = rng.normal(size=(N, D))
y_all = X_all @ rng.normal(size=D) + 0.3

def linear_grad(X, y, w):
    """均方误差下线性回归的梯度:2/N * X^T (Xw - y)"""
    return 2 / len(X) * X.T @ (X @ w - y)

w = rng.normal(size=D) * 0.1
R = 3                                          # 3 个 DP 副本
shards = np.array_split(np.arange(N), R)       # 数据并行:各副本分到不同样本
local_grads = [linear_grad(X_all[idx], y_all[idx], w) for idx in shards]

avg_grad = np.mean(local_grads, axis=0)        # ← AllReduce(平均) 在做的事
full_grad = linear_grad(X_all, y_all, w)       # 单卡全量梯度

print("AllReduce 平均梯度 vs 全量梯度:",
      "✅ 一致(分布式训练 = 全量训练)" if np.allclose(avg_grad, full_grad, atol=1e-6) else "❌")
print("数值差:", float(np.abs(avg_grad - full_grad).max()))
''')

# ---------------------------------------------------------------- 40 课:CustomAllreduce 延迟模型
CUSTOM_AR = D('''
import numpy as np

# 通信延迟的两段式模型:latency = 固定开销(alpha) + 通信量/带宽(beta)。
# NCCL 每次调用有内核启动 + 协议握手开销(alpha 大),但能跨机、带宽高;
# vLLM 的 CustomAllreduce 用共享内存 + IPC 直写结果(alpha 极小),只适用于单机小消息。
def ar_latency(msg_mb, n_gpus, alpha_s, bw_bps):
    """msg_mb: 消息大小(MB);alpha_s: 固定开销(秒);bw_bps: 带宽(Byte/s)。
    返回总延迟(微秒)。ring-allreduce 每卡通信量系数 = 2(N-1)/N。"""
    msg_bytes = msg_mb * 1e6
    ring_vol = 2 * (n_gpus - 1) / n_gpus
    return (alpha_s + ring_vol * msg_bytes / bw_bps) * 1e6

def custom_ar_latency(msg_mb, n_gpus, alpha_s=2e-6, bw_bps=80e9):
    """CustomAllreduce:共享内存直写,固定开销 ~2us,带宽受限于单机共享内存(~80GB/s)。"""
    return ar_latency(msg_mb, n_gpus, alpha_s, bw_bps)

def nccl_ar_latency(msg_mb, n_gpus, alpha_s=25e-6, bw_bps=25e9):
    """NCCL:固定开销 ~25us(内核启动+协议),带宽受限于 PCIe/NVLink(~25GB/s)。"""
    return ar_latency(msg_mb, n_gpus, alpha_s, bw_bps)

# 找 crossover:消息多小的时候,Custom 明显快于 NCCL?
print(f"{'消息大小(MB)':>14} {'custom(us)':>11} {'nccl(us)':>10} {'谁更快':>8}")
for mb in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    c = custom_ar_latency(mb, 4)
    n = nccl_ar_latency(mb, 4)
    winner = "custom" if c < n else "NCCL"
    print(f"{mb:>14.3f} {c:>11.1f} {n:>10.1f} {winner:>8}")

print()
print("同一消息下,延迟随卡数 N 的变化(固定 0.1MB):")
for ng in [2, 4, 8]:
    print(f"  N={ng}: custom {custom_ar_latency(0.1, ng):.1f} us | "
          f"nccl {nccl_ar_latency(0.1, ng):.1f} us")
''')

# ---------------------------------------------------------------- 41 课:组合估算
COMBO_EST = D('''
def estimate_combo(params_b, layers, hidden, gpus=8, tp=2, pp=1, dp=4,
                   ctx=4096, batch=8, dtype_bytes=2, m=8, kv_heads=8, head_dim=128):
    """估算 TP×PP×DP 组合下:每卡权重显存 / KV 显存 / TP 通信量 / PP 气泡占比。
    模型(简化但方向正确):
      权重/卡 ≈ params×bytes / tp / pp
      KV/卡   ≈ 2×L×h_kv×d_head×b×ctx×batch / tp(每个 DP 组各自服务自己的请求,内部再 TP 切)
      TP通信  /前向 = 2×layers×pp... 注意 PP 下每卡只跑 layers/pp 层 → 每卡通信 = 2×(layers/pp)×2(N-1)/N×b×h×2B
      气泡     = (pp-1)/(m+pp-1)"""
    assert tp * pp * dp <= gpus + 1e-9
    weights_gb = params_b * dtype_bytes / tp / pp
    kv_gb = 2 * layers * kv_heads * head_dim * dtype_bytes * ctx * batch / 1024 ** 3 / tp
    ar_bytes = 2 * (layers / pp) * 2 * (tp - 1) / tp * batch * ctx * hidden * dtype_bytes
    bubble = (pp - 1) / (m + pp - 1)
    return dict(weights_gb=weights_gb, kv_gb=kv_gb,
                tp_comm_mb=ar_bytes / 1e6, bubble=bubble)

def all_combos(gpus):
    out = []
    for tp in [1, 2, 4, 8]:
        for pp in [1, 2, 4, 8]:
            dp = gpus // (tp * pp)
            if tp * pp * dp == gpus:
                out.append((tp, pp, dp))
    return out

# 用纯 Python list 打印,避免在本会话语境(可能有 GPU 前序 cell)构造 pandas DataFrame
# (pandas3 + pyarrow 在 Jupyter 内核接 GPU 运算后可能触发 arrow.dll 崩溃)。画图/下游直接索引 list。
rows = []
for tp, pp, dp in all_combos(8):
    est = estimate_combo(70, 80, 8192, gpus=8, tp=tp, pp=pp, dp=dp, batch=4)
    rows.append((f"TP{tp}×PP{pp}×DP{dp}", round(est["weights_gb"], 1),
                 round(est["kv_gb"], 1), round(est["tp_comm_mb"], 1),
                 round(est["bubble"], 3)))

print(f"{'组合':<16}{'权重GB/card':>11}{'KV_GB/card':>11}{'TP通信MB/fwd':>13}{'气泡占比':>8}")
for name, wg, kv, cm, bub in rows:
    print(f"{name:<16}{wg:>11.1f}{kv:>11.1f}{cm:>13.1f}{bub:>8.3f}")
combo_rows = rows     # 供后续 cell 直接用
''')
