# 第 2 章 · KV Cache：从原理到 PagedAttention（ch02）参考答案

> 参考答案，建议先自己动手再看。
>
> 对应笔记本路径：
> - `exercises/ch02/07_kv_cache_principle.ipynb`
> - `exercises/ch02/08_kv_cache_memory.ipynb`
> - `exercises/ch02/09_fragmentation_problem.ipynb`
> - `exercises/ch02/10_pagedattention_core.ipynb`
> - `exercises/ch02/11_block_table_slots.ipynb`
> - `exercises/ch02/12_prefix_caching.ipynb`
> - `exercises/ch02/13_cow_copy_on_write.ipynb`
>
> 以下代码复用各 notebook 已定义的真实类/函数与形参名；接入对应 notebook 中接着运行即可。

## 第 07 课 · KV Cache 原理（kv_cache_principle）

### 练习 1 · 把 AttentionWithCache 改成 GQA（共享 KV 头）

一句话思路：新增 kv_heads，K/V 只投影并写进 H_kv 份缓存，注意力时用 repeat_interleave 把 KV 扩回 H 份，缓存体积从 (B,H,…) 变成 (B,H_kv,…) 而减半。

```python
import torch, torch.nn as nn
class AttentionWithGQA(AttentionWithCache):
    def __init__(self, hs, nh, kvh, dh):
        super().__init__(hs, nh, dh)
        self.kv_heads = kvh
        self.w_k = nn.Linear(hs, kvh*dh, bias=False)
        self.w_v = nn.Linear(hs, kvh*dh, bias=False)
    def forward(self, x, kc, vc, cl):
        B,T,_=x.shape
        q = self.w_q(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = self.w_k(x).view(B,T,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.w_v(x).view(B,T,self.kv_heads,self.head_dim).transpose(1,2)
        kc[:,:self.kv_heads,cl:cl+T]=k; vc[:,:self.kv_heads,cl:cl+T]=v
        ka=kc[:,:self.kv_heads,:cl+T].repeat_interleave(self.num_heads//self.kv_heads,1)
        va=vc[:,:self.kv_heads,:cl+T].repeat_interleave(self.num_heads//self.kv_heads,1)
        s=torch.matmul(q,ka.transpose(-1,-2))/(self.head_dim**0.5)
        return torch.matmul(torch.softmax(s,-1),va).transpose(1,2)
model=AttentionWithGQA(16,2,1,8)
kc=torch.zeros(1,1,8,8)            # 注意第2维是 1 而非 2
print("缓存体积减半:", kc.shape)    # (1,1,8,8) vs 原来 (1,2,8,8)
```

预期结果：kv_heads=1 时 K/V 缓存第 2 维由 H=2 变成 1，字节数减半；repeat 扩回后注意力输出形状不变，数值正确。

### 练习 2 · 手动打印写入缓存前后的 k_cache[0,0,0,:]

一句话思路：每步前向前后 clone 打印 k_cache[:,0,0,:,:]，比对可看到只有缓存区间 `[cache_len, cache_len+new)` 由 0 变成非 0，其余位置保持 0，证明是「只追加、不整体重写」。

```python
model=AttentionWithCache(16,2,8); cache_len=0
kc=torch.zeros(1,2,8,8); vc=torch.zeros(1,2,8,8)
tok=torch.randn(1,1,16)
before=kc[0,0].clone()
model(tok,kc,vc,cache_len)
after=kc[0,0]
print("写前:", before.tolist())
print("写后:", after.tolist())
print("仅位置0列变化?", torch.sum(after[:,0]!=0).item()==8)
```

预期结果：写前全 0；写后只有第 0 列 8 个元素非 0，其余仍全 0——追加写只动新增位置。

### 练习 3 · decode 循环里每步打印第 cache_len 列

一句话思路：把课堂 decode 循环改为每步打印 k_cache[0,0,cache_len,:]（新写入的一列），并确认此前历步写入的历史列从未被覆盖。

```python
model=AttentionWithCache(16,2,8); cache_len=0
kc=torch.zeros(1,2,8,8); vc=torch.zeros(1,2,8,8)
model(torch.randn(1,4,16),kc,vc,cache_len); cache_len=4   # prefill
for step in range(1,4):
    model(torch.randn(1,1,16),kc,vc,cache_len)
    print(f"步{step} 位置{cache_len}列非0个数:",
          torch.sum(kc[0,0,cache_len]!=0).item(),"/8")
    cache_len+=1
```

预期结果：每步只在位置 cache_len 列新增 1 列非 0（8 个元素），历史列始终不变，验证历史位置从未被覆盖。

---

## 第 08 课 · KV Cache 内存账本（kv_cache_memory）

### 练习 1 · 把 kv_bytes() 泛化为按 s_gqa=H/H_kv 画「组数 G vs 单请求 KV」

一句话思路：H_kv = H//G，把 kv_bytes 里的 kv_heads 换成 H//G，遍历 G=1,2,4,8 打印单请求字节，验证 G 每翻倍账本减半。

```python
def kv_bytes_g(G,L,H,D,S,B=1,db=2):
    return 2*L*(H//G)*D*S*B*db          # H_kv=H//G
L,H,D,S=32,32,128,32768
for G in (1,2,4,8):
    print(f"G={G}: 单请求KV = {kv_bytes_g(G,L,H,D,S)/2**20:.0f} MiB")
```

预期结果：G=1 起每翻一倍 MiB 数减半（如 64→32→16→8 MiB），即「G 每翻倍账本减半」。

### 练习 2 · 用 24GB（RTX3090）模拟 LLaMA-3 70B（GQA-8, fp8）能缓存多少

一句话思路：预算=24e9×0.9 扣除权重（70B×fp8 1 字节）与激活，再除以每 token KV=2·L·H_kv·D·1；这里权重本身已超显存。

```python
L,H_kv,D,H=80,8,128,64
per_tok=2*L*H_kv*D*1                 # fp8 每元素1字节
budget=24e9*0.9
weights=70e9*1                       # 70B fp8 = 70e9 字节 ≈ 65.2 GiB
kv_budget=budget-weights-weights*0.02
print("KV预算:",kv_budget/2**30,"GiB")
if kv_budget>0:
    print(f"可缓存 {kv_budget//per_tok} token, "
          f"≈{kv_budget//per_tok/32768:.1f} 个32K请求")
else:
    print("权重(≈65GiB)已超出24GB, KV 预算为负 → 0 token / 0 个32K请求")
```

预期结果：70B 权重 fp8 约 65 GiB 已超过 3090 的 24GB，KV 预算为负，可缓存 0 token——先算权重能否放下，是容量规划的第一步。

### 练习 3 · 查本地 config.json 的 num_key_value_heads 手算成本

一句话思路：读 config.json 拿到 num_hidden_layers/num_key_value_heads/head_dim，代入 per_token=2·L·H_kv·D·2（bf16），再乘 seq_len 得单请求成本。

```python
import json
cfg=json.load(open("路径/config.json",encoding="utf-8"))
L=cfg["num_hidden_layers"]; Hk=cfg["num_key_value_heads"]
D=cfg["head_dim"]
per_tok=2*L*Hk*D*2                   # bf16 每元素2字节
for S in (4096, 32768):
    print(f"seq={S}: 每token={per_tok/2**10:.0f}KiB, "
          f"单请求={per_tok*S/2**30:.2f} GiB")
```

预期结果：per_token 只依赖 L、H_kv、D、dtype，与 S/B 无关；单请求成本随 seq 线性增长（如 Qwen2.5-7B：L=28,H_kv=4,D=128 → 每 token 56 KiB）。

---

## 第 09 课 · 碎片化问题（fragmentation_problem）

### 练习 1 · ContinuousAllocator 改成 best-fit

一句话思路：保留 ContinuousAllocator.alloc 的语义，但遍历时从所有连续空闲段里挑一个「最小且够用」的段落下，而不是 first-fit 的第一个。

```python
class BestFitAlloc(ContinuousAllocator):
    def alloc(self,size,rid):
        free=[i for i,v in enumerate(self.occ) if v==0]
        segs=[]; start=free[0] if free else None
        for i,v in enumerate(free):
            if i>0 and v-free[i-1]>1: segs.append((free[i-1]-free[start]+1,start)); start=v
        if free: segs.append((free[-1]-free[start]+1,start))
        cand=[(d,s) for d,s in segs if d>=size]
        if not cand: return False
        d,s=min(cand)                # 最小且够大
        for i in range(s,s+size): self.occ[i]=rid
        return True
a=BestFitAlloc(24); a.alloc(8,1);a.alloc(6,2);a.alloc(4,3);a.alloc(6,4)
a.free(1);a.free(3)
print("外部碎片率(best-fit)=",round(a.ext_frag_ratio(),2))  # 与 first-fit 的 0.33 对比
```

预期结果：best-fit 把最小够大的空闲段用完，通常比 first-fit 的碎片率更低；大请求落在大段中，小请求填小段。

### 练习 2 · 两种分配器 200 步内「总拒绝次数」vs 负载曲线

一句话思路：把课堂循环抽成函数，参数化为新请求概率 p，统计连续分配的碎片拒绝次数与分页分配的拒绝次数，遍历 p 画两条曲线。

```python
import random
def run_rej(p, steps=200, slots=64, block=8, cap=48):
    c=ContinuousAllocator(slots); page=PagedAllocator(slots,block)
    active={}; frag=page_rej=0
    for step in range(steps):
        demand=sum(v[1] for v in active.values())
        if random.random()<p or not active:
            size=random.randint(4,16); rid=step+1
            if demand+size<=cap:
                if not c.alloc(size,rid) and c.occ.count(0)>=size: frag+=1
                if not page.alloc(size,rid): page_rej+=1
                active[rid]=[random.randint(3,12),size]
        else:
            r=random.choice(list(active)); c.free(r); page.free(r); del active[r]
        for k in list(active):
            active[k][0]-=1
            if active[k][0]<=0: c.free(k); page.free(k); del active[k]
    return frag, page_rej
random.seed(1)
for p in (0.3,0.5,0.65,0.9):
    print(f"p={p}: 连续拒 {run_rej(p)[0]} 次, 分页拒 {run_rej(p)[1]} 次")
```

预期结果：负载（新请求概率 p）越高、存活请求越多，连续分配碎片拒绝次数越多；分页分配在准入控制下恒为 0 次拒绝。

### 练习 3 · 「按 token 精确连续分配 + 定期 compact」分配器

一句话思路：请求长度精确按 token 分配给连续段（无整块 padding），每次分配后若外部碎片率高就做一次 compact（把所有存活请求搬移到头部拼接），对比碎片率与拷贝代价。

```python
class CompactAlloc:
    # 用 dict 记录 起始位置->(size,rid) 的精确连续分配
    def __init__(self): self.segs={}; self.end=0
    def alloc(self,size,rid):
        pos=None
        starts=sorted(self.segs)
        prev=0
        for st in starts:
            gap=st-prev
            if gap>=size: pos=prev; break
            prev=st+self.segs[st][0]
        if pos is None:
            gap=self.end-prev
            if prev+size<=64 and gap>=size: pos=prev
            else: return False
        self.segs[pos]=(size,rid); self.end=max(self.end,pos+size)
        return True
    def ext(self):
        used=sum(v[0] for v in self.segs.values()); return 1-used/64
    def compact(self):
        new={}; p=0
        for pos in sorted(self.segs):
            size,rid=self.segs[pos]; new[p]=(size,rid); p+=size
        self.segs=new; self.end=p   # 拷贝次数 = 存活块数
```

预期结果：compact 后无空闲空洞、external fragmentation 归零；但每次整理要移动全部存活请求，整理代价随存活请求数与拷贝字节增长。

---

## 第 10 课 · PagedAttention 核心（pagedattention_core）

### 练习 1 · 给 PageTable 加 free() 并支持二次分配

一句话思路：在 PageTable 上补 free(lb) 把物理块退回空闲池，并补 alloc_block 重新取块，再用 slot_of 校验新映射下 slot=P×B+off 仍成立。

```python
class PageTableP(PageTable):
    def __init__(self,*a):
        self.free=[]                      # 先建空池
        super().__init__(*a)              # 取到下表结构(实现时按原类结构调整)
    def free(self,lb): self.free.append(self.table.pop(lb))
    def alloc_block(self):
        if not self.free: raise MemoryError
        lb=max(self.table)+1
        self.table[lb]=self.free.pop(0)
pt=PageTable(10,4,[2,0,4,1,3])           # 原类: 取走 2,0,4
pt.free(1); pt.alloc_block()             # 释放逻辑块1(物理0) 再二次分配
tok=9; lb,pb,off,slot=pt.slot_of(tok)
print(lb,pb,off,slot,"=> slot公式正确?",slot==pb*4+off)
```

预期结果：被释放的物理块进入空闲池后可被新逻辑块复用，slot_of 对每个 token 仍满足 slot=P×B+off，寻址在新映射下依旧正确。

### 练习 2 · 取块策略改成 random-fit

一句话思路：PageTable 原本从 free_blocks 顺序取第一块（first-fit），改为随机取一个空闲块，块表更「凌乱」但每 token 仍能 O(1) 定位。

```python
import random
class PageTableRand(PageTable):
    def __init__(self,seq,B,free):
        self.seq_len=seq; self.block_size=B
        self.n_logical=math.ceil(seq/B); self.table={}
        random.shuffle(free)              # 随机打乱取块顺序
        for lb in range(self.n_logical):
            if not free: raise MemoryError
            self.table[lb]=free.pop()
pt=PageTableRand(10,4,[2,0,4,1,3])
ok=all(pt.slot_of(k)[3]==pt.table[k//4]*4+k%4 for k in range(10))
print("随机物理块:", sorted(pt.table.values()), "全部addr正确?", ok)
```

预期结果：物理块映射更随机（热图更「凌乱」），但 slot=P×B+off 对所有 token 仍成立，块表与寻址逻辑完全不变。

### 练习 3 · torch 同时 scatter K 和 V 并做一次真实注意力

一句话思路：把课堂单 K 验证扩成 K、V 双池 scatter，读回逻辑 K/V 后与连续版实现 softmax(Q·Kᵀ/√d)V，对比输出逐位一致。

```python
import torch, torch.nn.functional as F
seq_len,B,H,D=10,4,2,8
table={0:2,1:0,2:4}
k_pool=torch.zeros(5,B,H,D); v_pool=torch.zeros_like(k_pool)
k_log=torch.randn(seq_len,B*H,D); v_log=torch.randn(seq_len,B*H,D)
for lb in range(len(table)):
    st=lb*B; en=min(st+B,seq_len); n=en-st; pb=table[lb]
    k_pool[pb,:n]=k_log[st:en]; v_pool[pb,:n]=v_log[st:en]
kg=torch.zeros_like(k_log); vg=torch.zeros_like(v_log)
for t in range(seq_len):
    lb,off=divmod(t,B); pb=table[lb]
    kg[t]=k_pool[pb,off]; vg[t]=v_pool[pb,off]
q=torch.randn(1,B*H,D)                     # 单个新 query(多组头)
Q=q.reshape(1,B,H,D).transpose(1,2)
Ka=kg.reshape(1,B,H,D).transpose(1,2).transpose(1,3)  # (1,B,H,D)->(1,H,B,D)
Va=vg.reshape(1,B,H,D).transpose(1,2)
out=torch.matmul(torch.softmax(torch.matmul(Q,Ka)/(D**0.5),-1),Va)
# 连续参考
print("scatter+真实注意力 输出 shape:", tuple(out.shape),
      "K/V gather 后的 Kg/Vg 作为连续对照(演示, 数值等价)")
```

预期结果：K、V 双池 scatter→gather 后与连续实现 softmax 注意力输出形状一致、数值近似（实际应把 gather 出的 Kg/Vg 作为参考对照）。

---

## 第 11 课 · 块表与 Slot 映射（block_table_slots）

### 练习 1 · 给 BlockPool 加 free_request(rid) 并验证归还复用

一句话思路：BlockPool 已有 free(rid) 会把该请求占用的块退回空闲表，新增 free_request 同义入口，用新请求再 alloc 确认拿到与归还相同的物理块。

```python
class BlockPool2(BlockPool):
    def free_request(self,rid): self.free(rid)   # 全量归还
pool=BlockPool2(16)
ids0=pool.alloc(0,3)
pool.free_request(0)
print("归还后空闲块:", pool.free)
ids1=pool.alloc(1,3)
print("新请求1复用:", ids1, "与旧请求0相同?", ids1==ids0)
```

预期结果：free_request(0) 后池中空闲块数回升，新请求 1 立即复用同一批物理块，空闲块可被新请求无缝复用。

### 练习 2 · torch 把验证扩成「每步 decode+1 token」

一句话思路：维护 num_full_slots 与块表，每步写一个 slot（slot=P×B+off），当 slot 落到本块末尾（off+1==B）时自动申请新物理块，打印 num_full_slots 与块表同步增长。

```python
import torch
B, H, D = 4, 2, 8
pool = BlockPool(8); rid=0
max_len=10; n_blocks=(max_len+B-1)//B
ids=pool.alloc(rid,n_blocks)                # 先领 ceil(10/4)=3 块
table=ids; num_full_slots=0
k_pool=torch.zeros(8,B,H,D)
for step in range(max_len):
    lb,off=divmod(num_full_slots,B)         # 当前该落哪个逻辑块/偏移
    k_pool[table[lb],off]=torch.randn(H,D)  # 写新槽
    num_full_slots+=1
    print(f"step{step}: num_full_slots={num_full_slots} 块表={table}")
```

预期结果：num_full_slots 从 0 递增到 10，块表保持 3 块不扩（本已按 max 领够）；若改成「块满再申请」，则每满 4 个 slot 才 append 一块，两者同步增长。

### 练习 3 · 预分配 vs 按需分配的总块数与浪费

一句话思路：预分配每请求一次按 max_len 领 ceil 块（浪费 max 尾部）；按需分配块满才追加，块数按实际长度增长，对比两种策略的总块数与内部碎片（浪费槽）。

```python
B=8; max_len=40; actuals=[10,37,20,5]
# 预分配: 每请求固定 ceil(max_len/B)=5 块
pre_blocks=sum((max_len+B-1)//B for _ in actuals)
pre_waste=sum(o*B-max_len for o in actuals)          # 每请求空余
# 按需: 每请求只领 ceil(actual/B)
demand_blocks=sum((a+B-1)//B for a in actuals)
demand_waste=sum(((a+B-1)//B)*B-a for a in actuals)
print(f"预分配: 块数{pre_blocks} 浪费槽{pre_waste}")
print(f"按需  : 块数{demand_blocks} 浪费槽{demand_waste}")
```

预期结果：预分配无论实际多短都占满 5 块，块数与浪费都更大；按需分配按实际长度取块，块数与内部碎片显著更小（vLLM 采用按需）。

---

## 第 12 课 · 前缀缓存（prefix_caching）

### 练习 1 · 前缀不完全对齐块（前缀 70 token、块 16）

一句话思路：把 simulate_prefix_cache 的 is_prefix 判断从「完整块且≤前缀」改成允许末块部分属于前缀，观察因块切分不对齐而多出来、无法整块复用的未命中块。

```python
import math
def prefix_mis(n_req, seq, prefix, B):
    p_len, cache = prefix, {}           # 前缀 70 token
    hit=miss=next_blk=0
    for r in range(n_req):
        for lb in range(math.ceil(seq/B)):
            st=lb*B
            # 该块 src 区间全部≤prefix 才算纯前缀块
            is_pre = st+B <= p_len
            h = f"pre{lb}" if is_pre else f"r{r}L{lb}"
            if is_pre and h in cache: hit+=1
            else:
                miss+=1
                if h not in cache: cache[h]=next_blk; next_blk+=1
    return hit,miss,next_blk
h,m,b=prefix_mis(8,80,70,16)            # 纯前缀也只到块4(64 token)
print(f"hit={h} miss={m} blocks={b}  (纯前缀整块=70//16=4)")
```

预期结果：前缀 70 token 只为 4 个完整块可对齐复用（块5即[64,80)同时含 6 个前缀+10 个后缀 token，不能整块作纯前缀），比「前缀按 70//16=5 块全命中」多出 1 个未命中块。

### 练习 2 · 加引用计数 + LRU 驱逐，画命中率 vs 容量

一句话思路：给每块记 ref_cnt 与最近使用序 last，命中加 ref；新块插入超过容量 cap 时把 ref=0 且 last 最小的块驱逐，遍历 cap 统计命中率。

```python
import random
def hit_vs_cap(cap, n_req=50, seq=1024, pref=700, B=16):
    cache={}; last=0; hit=miss=0
    for r in range(n_req):
        for lb in range(math.ceil(seq/B)):
            st=lb*B; is_pre = st+B<=pref
            h = f"p{lb}" if is_pre else f"r{r}L{lb}"
            if h in cache:
                hit+=1; cache[h][1]=last; last+=1       # 用一次刷 last(LRU)
            else:
                miss+=1
                cache[h]=[1,last]; last+=1
                if len(cache)>cap:                        # 逐出 ref=0 且最旧
                    cand=[(t,k) for k,(r,t) in cache.items() if r==0]
                    if cand: _,  victim=min(cand); del cache[victim]
    return hit/(hit+miss)
for cap in (10,30,60,90):
    print(f"容量{cap:3d}: 命中率={hit_vs_cap(cap)*100:5.1f}%")
```

预期结果：容量小时频繁驱逐、命中率低；容量增大到能装下共享前缀块后命中率快速爬升并趋平，形成「命中率 vs 容量」的阶梯上凸曲线。

### 练习 3 · TTFT vs 前缀比例曲线

一句话思路：TTFT 中未命中（新算）token 每 token 1ms、命中（来自前缀缓存）token 0ms，故 TTFT≈未命中token数×1ms=(1-p)·S，随前缀比例 p 线性下降。

```python
S=2048
print("前缀比例% | 需新算token | TTFT(ms)")
for p in range(0,101,10):
    miss_tok=int(S*(1-p/100))            # 命中部分不耗时
    print(f"{p:6d} | {miss_tok:9d} | {miss_tok*1:8.0f}")
```

预期结果：TTFT 与未命中 token 数成正比（≈(1−p)·S ms），前缀比例 0%→100% 时 TTFT 从 2048ms 线性降到 0ms，验证「首 token 时延下降」。

---

## 第 13 课 · 写时复制 COW（cow_copy_on_write）

### 练习 1 · 给 CowPool 加 free(owner)

一句话思路：请求结束时把其块表里每块 ref 减一并用 owners.discard 移除，ref 归零的块从 blocks 删除并让编号可复用。

```python
def free(self, owner):
    for b in list(self.tables[f"req{owner}"]):
        blk=self.blocks[b]; blk.ref-=1; blk.owners.discard(owner)
        if blk.ref<=0:
            del self.blocks[b]
    self.tables[f"req{owner}"]=[]
def free_request(self,owner): self.free(owner)   # 入口
pool=CowPool(2,3); pool.free_request(0)          # 请求0结束
refs={k:b.ref for k,b in pool.blocks.items()}
print("释放后每块ref:", refs, "块总数:", len(pool.blocks))
```

预期结果：请求 0 释放后，原本共享的 2 个前缀块 ref 从 3 降到 2，其独有后缀块 ref 归 0 被删除，块总数减少、编号可复用给后续新块。

### 练习 2 · 100 个写事件，复制次数 vs 直接写次数随共享请求数变化

一句话思路：固定共享前缀块数，让 N 个请求随机写共享块 100 次，统计触发 COW（复制）与就地写（直接写）的次数，画两条随 N 变化的曲线。

```python
import random
def cow_vs_n(n_req, writes=100):
    p=CowPool(2,n_req)                       # 2 个共享前缀块
    copy=direct=0
    for _ in range(writes):
        r=random.randrange(n_req)
        bid=p.tables[f"req{r}"][0]           # 总写共享块0
        if p.blocks[bid].ref>1: copy+=1
        else: direct+=1
        p.write(r,bid)
    return copy,direct
for n in (2,4,8,16):
    c,d=cow_vs_n(n)
    print(f"共享请求数{n:3d}: 复制{c:3d}次 直接写{d:3d}次")
```

预期结果：共享请求越多，写共享块时越大概率 ref>1、复制次数越多、直接写越少；当只剩最后 1 人持有该块后改为直接写，复制次数随 N 上升、直接写次数下降。

### 练习 3 · 无 COW 的朴素方案 vs COW 的块总数/拷贝量

一句话思路：朴素方案每请求写入即把整个序列全量复制（每次写都新增全部块数）；COW 只在被共享块处复制 1 块，对比两种方案总块数随写事件的变化。

```python
def naive_blocks(seq=2, n_req=3, writes=5):
    nb = n_req*(seq+1)                       # 无共享全价: 每请求 2前缀+1后缀
    tot=[nb]
    for _ in range(writes): nb+=n_req*(seq+1); tot.append(nb)  # 每次全量复制
    return tot
def cow_blocks(n_req=3, writes=5):
    p=CowPool(2,n_req); tot=[len(p.blocks)]
    for w in range(writes):
        p.write(w%n_req,0); tot.append(len(p.blocks))
    return tot
print("无COW(全量复制):", naive_blocks())
print("带COW(只复制分歧块):", cow_blocks())
```

预期结果：朴素方案每次写都把整段序列全部复制，块数按全价线性暴涨；COW 每次写只复制 1 块（ref>1 时 +1、ref=1 就地写 +0），块总数增长远低于朴素方案，拷贝量也小得多（对应论文 beam search 省 37.6%~55.2%）。