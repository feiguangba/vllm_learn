# 02 · vLLM V1 PagedAttention 与 KV Cache 管理:块池、前缀缓存与多组协调

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `v1/core/block_pool.py:143` = 上游源码中的 `vllm/v1/core/block_pool.py` 第 143 行)。
> **一句话总结**:KV Cache 不再是一整块连续显存,而是被切成固定大小(默认 16 token)的"页/块"(block),由 `BlockPool` 统一分配/回收,`KVCacheManager` + `KVCacheCoordinator` 负责按请求建块表、做前缀缓存(链式哈希 + COW)并对**多个 KV cache group**(全注意力 / 滑窗 / Mamba / 混合模型)协调;显存预算在启动时由 GPU profiling 决定(`available = 总内存×util − 模型占用 − CUDA graph`)。

---

## 一图看懂

```
模型启动(profiling)                                       运行期(EngineCore 每步)
┌──────────────────────────────────────┐          ┌─────────────────────────────────────────────┐
│ gpu_worker.py:475                    │          │ Scheduler.schedule() (scheduler.py:476)       │
│  determine_available_memory          │          │  └─ kv_cache_manager.allocate_slots          │
│   requested = 总内存×util (utils.py:449)│        │      (kv_cache_manager.py:347)               │
│   available = requested              │          │         │ 检查空闲块 ≥ 需求 → 分配 → 缓存      │
│              − 模型占用              │          │         ▼                                     │
│              − cudagraph 估算        │          │ KVCacheManager (kv_cache_manager.py:118)      │
│             (gpu_worker.py:559)      │          │  ├─ get_computed_blocks (前缀缓存查找, :232)  │
└───────────────────┬──────────────────┘          │  ├─ allocate_slots (块分配, :347)             │
                    │ available_memory (字节)      │  └─ free / remove_skipped_blocks / COW        │
                    ▼                              │         │ 委托给                              │
┌──────────────────────────────────────┐          │         ▼                                     │
│ kv_cache_utils.py:1327               │          │ KVCacheCoordinator (kv_cache_coordinator.py:63)│
│  get_kv_cache_config_from_groups     │          │  ├─ BlockPool (block_pool.py:143)             │
│   num_blocks = available             │          │  │   ├─ blocks[]: 所有 KVCacheBlock           │
│              // page_size // group   │          │  │   ├─ free_block_queue (LRU 双向链表)        │
│              (:1359-1361, :1392)     │          │  │   └─ cached_block_hash_to_block (前缀缓存)  │
│  → KVCacheConfig (kv_cache_interface.py:955)   │  └─ single_type_managers[] (每 group 一个)      │
└───────────────────┬──────────────────┘          │       FullAttention / SlidingWindow / Mamba... │
                    │ num_blocks                     └──────────────┬───────────────────────────────┘
                    ▼                                                │
┌──────────────────────────────────────┐          ┌────────────────▼──────────────────────────────┐
│ BlockPool.__init__ (block_pool.py:162)│         │ 请求的块表 = req_to_blocks[req_id]             │
│  blocks[i] = KVCacheBlock(i)         │          │  ([物理块引用, …, null_block, …])              │
│  保留 null_block (block_id=0)  (:190)│          │  每 block 的 block_hash 链式指纹前缀            │
└──────────────────────────────────────┘          │  命中 → touch (ref_cnt+1) / 部分命中 → COW     │
                                                   └───────────────────────────────────────────────┘

hash 链:  block[0] = H(NONE_HASH, t0..t15, extra)      (kv_cache_utils.py:577)
          block[i] = H(block[i-1], t{i*16}.., extra)   ← 每个哈希唯一指纹"到该边界为止的整段前缀"
```

---

## 1. 总体结论:PagedAttention 的动机与块思想

### 1.1 为什么需要 PagedAttention

传统 LLM 推理(如 HuggingFace / FasterTransformer)把 KV cache 分配成**每请求一整段连续显存**。连续分配带来两个问题:

1. **显存碎片**:不同请求长度不同、生命周期交错,预留的连续段无法复用,论文实测显存利用率只有 **20% ~ 40%**(60%~80% 被浪费)。
2. **超额预留**:无法预测生成长度,只能按 `max_model_len` 预留整段,实际只用到一小部分。

PagedAttention(论文 SOSP'23)借用了 OS 的**虚拟内存分页**思想:**把 KV cache 切成固定大小的块(block),每个块是一个"页"**,请求通过一张**块表(block table)**把逻辑上连续的序列映射到物理上任意位置的块;块可以按需分配、按需释放,碎片和超额预留同时消失。这也是 vLLM 吞吐量提升 2~4 倍的核心机制。

> 打个比方:传统方式像给每个进程预分配一整段连续地址空间,分页就像把地址空间切成页、按需映射到物理页——操作系统里被证明有效的抽象,搬到 GPU 显存上依然成立。

### 1.2 默认 block_size = 16

块大小由 `CacheConfig.DEFAULT_BLOCK_SIZE = 16` 定义(`config/cache.py:48`),构造时若未显式指定就套用默认值(`_apply_block_size_default`,`config/cache.py:256-269`,用户指定则置 `user_specified_block_size=True`)。16 是 attention kernel 在**半线程束(半 warp)内处理 token 数**的常用值,能让每个线程负责一个 token 的 KV 读写。多组模型里各组 block_size 可以不同,但调度粒度取所有组的 **LCM**(见 §6)。

几个影响块大小的派生量:

- **`hash_block_size`**:块哈希的计算粒度,默认等于调度块大小;多组模型取各 group block_size 的 GCD(`resolve_kv_cache_block_sizes`,`kv_cache_utils.py:607-668`,见 §5.1)。
- **`scheduler_block_size`**:调度对齐粒度(`num_computed_tokens` 必须按它取整),单组 = `block_size × dcp`(decode context parallelism),多组 = LCM。
- **`prefix_match_unit`**(`config/cache.py:57-68`):可把哈希粒度调得比物理块更细,让命中边界落在物理块内部(§5.3 的部分命中)。

> **历史对照**:V0 的 KV 管理有 `VirtualBlock` / `PhysicalBlock` 两类对象,以及"虚拟块表 + 物理块池"两层概念。V1 把它们**合并成单一的 `KVCacheBlock` 对象**:`block_id` 直接索引物理池 `blocks[]`(block_pool.py:175),而每个请求的 `req_to_blocks[req_id]` 列表(见 §3.3)就是那张"虚拟块表"。

---

## 2. KVCacheBlock:块元数据与物理/虚拟映射

### 2.1 数据结构

核心数据结构 `KVCacheBlock`(`kv_cache_utils.py:118-177`)是一个 `@dataclass(slots=True)`,字段如下:

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                      # 0 .. num_gpu_blocks-1, 物理池内索引 (:123)
    ref_cnt: int = 0                   # 引用计数, 见 §3.2 (:125)
    _block_hash: BlockHashWithGroupId | None = None   # 满块哈希(仅已缓存时非空) (:128)
    _block_hash_num_tokens: int | None = None         # 该哈希覆盖的前缀 token 数 (:131)
    prev_free_block / next_free_block: KVCacheBlock | None   # 空闲双向链表指针 (:135-136)
    is_null: bool = False              # 是否 null 块(永不缓存/永不释放) (:139)
```

要点:

- **物理身份 = `block_id`**:`BlockPool.__init__` 一次性创建 `blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]`(`block_pool.py:175-177`),之后所有分配/释放都是把这些对象在"空闲队列"和"请求块表"之间搬运,对象地址不变,`block_id` 永远稳定——这正是前缀缓存能放心把"块哈希 → block_id"做成表的前提。
- **虚拟身份 = 请求的块表**:每个请求的 `req_to_blocks[request_id]`(`single_type_kv_cache_manager.py:94`)是一个 `list[KVCacheBlock]`,按 token 顺序排列;滑窗等稀疏注意力会把"被跳过的块"替换成 `null_block` 占位(`add_local_computed_blocks`,`single_type_kv_cache_manager.py:273`),保证块表长度与序列位置始终对齐,worker 端可直接按位置索引。
- **`is_null` 与 `null_block`**:`BlockPool` 构造时从空闲队列**队头**拿出一个块充当 `null_block`(`block_pool.py:190-191`),`block_id=0`,`ref_cnt` 不维护,永远不会被释放。它承担"占位符"角色——所有对它的引用都共享同一份全零 KV。
- **哈希元数据**:`set_block_hash`(`kv_cache_utils.py:149`)/`reset_hash`(:160)只能一次性设置/清除;`_block_hash_num_tokens` 支持"部分条目"(前缀终点落在块中间,见 §5.3)。

`block_hash` 字段也常被用作**该块是否在缓存表中**的判据:`get_unhashed_block_ids`(`kv_cache_manager.py:94`)等查询 `block.block_hash is None` 来区分"已缓存块"与"普通块"。

### 2.2 物理块 ↔ 虚拟块表(图示)

以 block_size=4、总块数 8 为例。请求 A 的序列长为 10 token,滑窗只覆盖最后 8 个 token:

```
物理池  blocks[]:   ┌────┬────┬────┬────┬────┬────┬────┬────┐
                    │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
                    └────┴────┴────┴────┴────┴────┴────┴────┘
                     null_block                         空闲空闲

请求 A 块表 req_to_blocks["A"]  (逻辑位置 → 物理块):
    位置0..3    位置4..7      位置8..9(+占位)
   ┌─────────┬──────────┬──────────┐
   │ block 5  │ block 2  │ null(0)  │   ← 第三块只写了 2 个 token,未满
   └─────────┴──────────┴──────────┘
     ref_cnt=1  ref_cnt=1  is_null
     block_hash=X  block_hash=Y   (满块才打哈希)

空闲队列 free_block_queue:  6 → 7 → [1 → 3 → 4](队尾,LRU)
                                  (1,3,4 有哈希:回收后按 FIFO 排,先淘汰最久未用)
```

- 逻辑上的"连续 token 区间"可以散落在任意物理块上;worker 的 attention kernel 通过 block table 逐个块取 KV,块内仍是连续内存,块间无需连续。
- `null_block` 占据块表第 3 个槽位,意味着"这段位置没有任何真实 KV",kernel 读到的是全零——滑窗窗口外的历史就这样被安全"遗忘"而不必移动后续数据。

---

## 3. BlockPool:分配、释放与回收

`BlockPool`(`block_pool.py:143`)是整个 KV 块体系的心脏。构造参数:`num_gpu_blocks`、`enable_caching`(前缀缓存开关)、`hash_block_size`(哈希粒度,见 §5.1)。

### 3.1 FreeKVCacheBlockQueue:空闲块双向链表

`FreeKVCacheBlockQueue`(`kv_cache_utils.py:185`)用**双向链表**组织空闲块,而不是 `deque`:它复用每个块自身的 `prev_free_block`/`next_free_block` 指针(`kv_cache_utils.py:135-136`),因此**从链表中间 O(1) 摘除某块**而不分配任何 Python 对象(`kv_cache_utils.py:186-192`)。链表两端是 fake head/tail(block_id=-1,永不弹出,`kv_cache_utils.py:223-235`)。

核心操作(`kv_cache_utils.py:237-428`):

| 方法 | 语义 |
|---|---|
| `popleft` / `popleft_n` | 从队头取 1/n 个块(分配) |
| `remove` | O(1) 从中间摘除(ref_cnt 从 0 变 1 时把块移出空闲队列) |
| `prepend_n` | 放回**队头**(LIFO 复用) |
| `append_n` | 放回**队尾**(FIFO 复用) |

### 3.2 ref_cnt 引用计数与分配/释放流程

`BlockPool` 的四个核心方法通过 `ref_cnt` 维护块的生命周期:

- **`get_new_blocks(num_blocks)`**(`block_pool.py:647`):`free_block_queue.popleft_n(num_blocks)` 取块。若开启缓存,先 `_maybe_evict_cached_block` 清掉这些块的哈希元数据,再 `ref_cnt += 1`(:661-677)。空闲块不足直接 `raise ValueError`——但调度器路径上会先做 `get_num_free_blocks()` 检查(`kv_cache_manager.py:526-530`),分配失败时改为触发抢占(见文档 03 §5)。
- **`touch(blocks)`**(`block_pool.py:702`):**前缀缓存命中时**调用。`ref_cnt += 1`;若块此刻 `ref_cnt == 0`(空闲队里的淘汰候选),先 `free_block_queue.remove(block)` 把它从空闲队列摘掉——否则"分配者"拿到手的块会同时被"淘汰者"还回队列造成双份引用(:710-715)。
- **`free_blocks(ordered_blocks)`**(`block_pool.py:719`):`ref_cnt -= 1`,归零且非 null 时送回空闲队列。**有哈希的块追加到队尾(FIFO,LRU 淘汰),无哈希的块插入队头(LIFO,提高 GPU 局部性)**(:727-743)。注意传入的块要按"先释放尾部"的顺序排列,注释明确由调用方保证(:720-725)。
- **`_maybe_evict_cached_block(block)`**(`block_pool.py:679`):从 `cached_block_hash_to_block` 中清除该块的全部哈希条目(`_remove_cached_block_hashes`,:571),并发出 `BlockRemoved` 事件。

### 3.3 一个完整的分配/释放时间线

以"两个请求共享前缀"为例走一遍 ref_cnt 的流转:

```
① 请求 A 首次 prefill,需要 2 块:        get_new_blocks(2) → blocks{5,2}  ref_cnt: 0→1
② A 的块满了,cache_full_blocks 打哈希:   block 5→hash X, block 2→hash Y
③ 请求 B 进来,前缀与 A 相同:             find_longest_cache_hit 命中 {5,2}
   add_local_computed_blocks → touch:     blocks{5,2}  ref_cnt: 1→2
④ A 完成,free(A):                       free_blocks([5,2] 逆序)  ref_cnt: 2→1
                                         (仍被 B 引用,不回空闲队列)
⑤ B 完成,free(B):                       free_blocks([5,2])  ref_cnt: 1→0
                                         有哈希 → 追加到空闲队列队尾(等 LRU 淘汰)
⑥ 新请求 C 分配,空闲队列 popleft 弹出 block 5:
   _maybe_evict_cached_block(5) → 清除 hash X → ref_cnt 0→1
```

### 3.4 每请求块表与回收

`req_to_blocks: defaultdict[str, list[KVCacheBlock]]`(`single_type_kv_cache_manager.py:94`)记录每个请求的块表。`pop_blocks_for_free(request_id)`(:497-514)把块表和 `num_cached_block` 记录一次性弹出但不回池,交给调用方(调度器的 `_free_request_blocks`,见文档 03 §5.3)做"延迟释放";`free(request_id)`(:516)则直接逆序回池。

`KVCacheManager` 侧对应的入口:`free`(`kv_cache_manager.py:570`)、`pop_blocks_for_free`(:602)、`remove_skipped_blocks`(:583,滑窗把窗口外的块替换成 null_block 并回池,`single_type_kv_cache_manager.py:624-661`)。调度器每步先 `new_step_starts()`(`kv_cache_manager.py:885` → `kv_cache_coordinator.py:413`)通知各 manager 复位每步状态。

### 3.5 滑窗/局部注意力的"回收式"管理

全注意力块的生存期是"请求存活期间一直持有";而滑窗(SWA)、chunked local attention 等稀疏注意力的块可以**边写边扔**:

- `SlidingWindowManager.get_num_skipped_tokens`(`single_type_kv_cache_manager.py:1063-1098`):计算窗口外已被跳过的 token 数(`max(0, computed - window + 1 - extra_retained_tokens)`),`remove_skipped_blocks`(:624)据此把头部被跳过的块换成 `null_block` 并回池。
- `ChunkedLocalAttentionManager` 类似,但按 attention chunk 边界丢弃。
- 这类"回收式"manager 有每请求的**准入块上限**(`max_admission_blocks_per_request`,`get_manager_for_kv_cache_spec`,`single_type_kv_cache_manager.py:1883-1892`;公式在 `SlidingWindowSpec.max_admission_blocks_per_request`,`kv_cache_interface.py:546`),保证启动时算的内存容量与运行时实际占用一致,避免 chunked prefill 下"第一块检查过、后续 OOM"的死锁(注释里的 issue #39734,`single_type_kv_cache_manager.py:176-188`)。

---

## 4. KV Cache 容量计算:从 GPU profiling 到 num_blocks

### 4.1 可用显存 = 总内存 × util − 模型 − CUDA graph

`Worker.determine_available_memory`(`v1/worker/gpu_worker.py:475`)在启动时做一次 dummy 前向 profiling(`model_runner.profile_run()`,gpu_worker.py:519),计算可给 KV cache 的字节数:

```python
available_kv_cache_memory_bytes = (
    self.requested_memory                  # = ceil(总显存 × gpu_memory_utilization)
    - profile_result.non_kv_cache_memory   # 模型权重 + 激活峰值等占用
    - cudagraph_memory_estimate_applied    # CUDA graph 池(可选估算)
)                                          # gpu_worker.py:559-563
```

- `requested_memory = math.ceil(init_snapshot.total_memory * gpu_memory_utilization)`(`v1/worker/utils.py:444-451`,默认 util=0.92,`config/cache.py:69`)。
- CUDA graph 显存默认按估算扣除(`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`,gpu_worker.py:527-539)。
- 用户也可用 `kv_cache_memory_bytes` 手工指定并跳过 profiling(gpu_worker.py:489-511)。

### 4.2 num_blocks = available // page_size // group_size

把字节换算成块数在 `get_kv_cache_config_from_groups`(`kv_cache_utils.py:1327`)完成,分三种情况:

1. **单组 UniformTypeKVCacheSpecs**(同型异 hidden size,如纯全注意力多 head 模型):`num_blocks = available_memory // page_size_bytes`(`kv_cache_utils.py:1359-1361`),为每层各建一个独立 `KVCacheTensor`。
2. **packed 布局**(DeepSeek V4 / `--enable-cross-layers`):`_get_kv_cache_config_packed`(:1296)按块跨度 `block_stride` 叠放各组的字节布局,`num_blocks = available // block_stride`(:1308)。
3. **通用多组**(混合模型):`group_size = max(各组层数)`,`num_blocks = get_num_blocks(available, group_size, available, page_size)`(:1386-1394),公式在 `get_num_blocks`(`kv_cache_utils.py:973`):

```python
num_blocks = int(available_memory // page_size // num_layers)   # :988
```

其中 `page_size_bytes` = 一个块一层所需的字节:`num_kv_heads × block_size × (head_size + head_size_v) × dtype_size`(`AttentionSpec.unpadded_page_size_bytes`,`kv_cache_interface.py:250-252`;Mamba 则是状态张量体积之和,:677-685)。`num_layers` 即 group_size——因为每组按"每 group 取一层共享一块"的规则,一个块实际承载 group_size 层的 KV。

> **算例**:假设 `page_size = 32KB`,`group_size = 2`(例如 1 个 full + 2 个 sw 的模式,取每模式一层),`available = 80GiB = 85,899,345,920 B`。则 `num_blocks = 85,899,345,920 // 32768 // 2 = 1,310,720`。这个数字就是 `num_gpu_blocks`,会通过握手回传给前端(`core.py:1616` 的 `EngineCoreReadyResponse` 携带 `num_gpu_blocks`/`block_size`)。

最终可被 `num_gpu_blocks_override` 覆盖(`may_override_num_blocks`,`kv_cache_utils.py:942`),并统一取各 worker 的最小值(`kv_cache_utils.py:2213-2226`,PP 下各 stage 层数不同)。产出的 `KVCacheConfig`(`kv_cache_interface.py:955`)包含 `num_blocks`、`kv_cache_tensors`(worker 建张量用)和 `kv_cache_groups`。

### 4.3 内存自检与最大并发

- `check_enough_kv_cache_memory`(`kv_cache_utils.py:834`):若连一个 `max_model_len` 请求都放不下则启动失败(`_check_enough_kv_cache_memory`,:731,会给出建议的 max_model_len)。
- `max_model_len = -1` 时自动二分搜索可装下的最大长度(`_auto_fit_max_model_len`,:1976)。
- 容量换算为 token:并发度 × `max_model_len`(`get_kv_cache_capacity`,:1848),写入 `kv_cache_size_tokens` / `kv_cache_max_concurrency`(:1861)。`max_concurrency = num_blocks / 每请求块数`(`get_max_concurrency_for_kv_cache_config`,:917),即"同时能容纳多少个满长请求"。

---

## 5. 前缀缓存(automatic prefix caching)

### 5.1 链式哈希:每个哈希指纹"整段前缀"

块哈希由 `hash_block_tokens`(`kv_cache_utils.py:577-604`)计算:

```python
block_hash[i] = hash_function((parent_block_hash, tuple(token_ids[i*16:(i+1)*16]), extra_keys))
```

- **父哈希链**:第 i 块的输入包含 `parent_block_hash = block_hash[i-1]`(首块用全局随机种子 `NONE_HASH`,`kv_cache_utils.py:88-115`)。因此**任意边界上的哈希唯一指纹从 0 到该边界的整段 token 序列**——前缀缓存只需逐块比哈希,不需要比对 token。
- **extra_keys**(`generate_block_hash_extra_keys`,`kv_cache_utils.py:539-574`):把**多模态输入**(`identifier, block 内 offset`)、**LoRA 名称**、**cache_salt**(首块)和 **prompt_embeds 哈希** 混入,使相同 token 但不同 MM/LoRA 上下文的块不互相误命中。
- **哈希算法**:默认 sha256(`prefix_caching_hash_algo`,`config/cache.py:98`),可选 xxhash 提速。
- **何时计算**:哈希在 `Request` 构造和输出 token 追加时**增量**计算。`Request.update_block_hashes`(`request.py:265-268`)调用 `_block_hasher`(`get_request_block_hasher`,`kv_cache_utils.py:671`,按 `hash_block_size` 粒度),`append_output_token_ids`(:252-263)在每步 append 后同步补算新满块,存进 `request.block_hashes`(:206)。hasher 在 EngineCore 构造时按 `hash_block_size` 生成(`v1/engine/core.py:223-232`)。
- **粒度**:`hash_block_size` 默认等于 `scheduler_block_size`;多组模型取各 group block_size 的 GCD(`resolve_kv_cache_block_sizes`,`kv_cache_utils.py:607-668`),可通过 `prefix_match_unit`(`config/cache.py:57`)调得更细,使命中边界落在物理块内部(配合 §5.3 的部分命中)。

### 5.2 一个哈希链的算例

设 `hash_block_size = 16`,请求 token 序列 `t0, t1, ..., t63`(64 个,4 块):

```
block[0] = H(  NONE_HASH, (t0..t15), extra0 )          → h0   ← 指纹 {t0..t15}
block[1] = H(  h0,        (t16..t31), extra1 )         → h1   ← 指纹 {t0..t31}
block[2] = H(  h1,        (t32..t47), extra2 )         → h2   ← 指纹 {t0..t47}
block[3] = H(  h2,        (t48..t63), extra3 )         → h3   ← 指纹 {t0..t63}
```

另一个请求若 `t0..t31` 完全相同,则它的 `block[1]` 哈希必然等于 `h1`——不需要比对 token 就能断定前两块可复用。注意 `extra_keys` 只作用于当前块(如 LoRA 名称影响所有块,MM 特征只影响包含该 MM 的块)。

### 5.3 BlockHashToBlockMap:哈希 → 块

`BlockHashToBlockMap`(`block_pool.py:33`)是前缀缓存的主索引:`{BlockHashWithGroupId: KVCacheBlock | dict[int, KVCacheBlock]}`。键 = 块哈希 + 4 字节 group id(`make_block_hash_with_group_id`,`kv_cache_utils.py:58-77`,避免建 tuple 的 GC 开销)。

- **不做去重**(NOTE #1,`block_pool.py:47-52`):相同哈希可以对应多个块(存成 dict),因为块表是 append-only,保证已分配 block_id 不改变。
- 关键方法:`get_one_block`(:61,取任意一个)、`contain`(:74)、`insert`(:88,单块升级成 dict)、`pop`(:106)。`BlockPool.get_cached_block`(`block_pool.py:198`)要求**所有 group 同哈希同时命中**才返回整组块。

### 5.4 查找与 COW:find_longest_cache_hit

调度器对等待队列请求先调 `KVCacheManager.get_computed_blocks`(`kv_cache_manager.py:232`,细节在文档 03 §4.3):`max_cache_hit_length = num_tokens - 1`(:262,必须留最后一个 token 重算以拿 logits),交给 `coordinator.find_longest_cache_hit`(:264)。返回 `(computed_blocks, num_new_computed_tokens, shared_prefix_boundary)`,其中 `computed_blocks` 经 `create_kv_cache_blocks`(:297)包装成 `KVCacheBlocks` 交给调度器。

**全注意力**(`FullAttentionManager.find_longest_cache_hit`,`single_type_kv_cache_manager.py:684`):

1. **Phase 1 左到右扫满块**:因为哈希是链式的,某个块 miss ⇒ 后面全部 miss,可提前 `break`(:735-741)。命中长度按 `block_size` 累加。
2. **Phase 2 部分命中**(fine-grained,`alignment_tokens < block_size` 时):在第一个未满块内部从高到低探测更细的哈希边界,拿最长命中(:745-764)。命中长度再向下取整到 `alignment_tokens`(:775)。
3. **EAGLE/MTP 尾巴丢弃**:投机解码需要命中边界前的 token 重算,`drop_eagle_block` 时从命中长度里减掉一个 hash 单位/块(:770-771)。

**命中 → touch**:命中的缓存块通过 `add_local_computed_blocks`(`single_type_kv_cache_manager.py:229`)挂进请求块表并 `block_pool.touch`(:266),`ref_cnt` 因此等于引用它的请求数——`get_num_common_prefix_blocks` 就靠"`ref_cnt == len(req_to_blocks)`"判公共前缀(`FullAttentionManager.get_num_common_prefix_blocks`,:823-831)。

**部分命中 → COW**:当命中长度落在某个共享块的中间(`num_local_computed_tokens % block_size != 0`),必须把该共享尾块**复制**成私有块,否则两个请求会同时写同一块。V1 的 COW 是**惰性**的:先记 `_partial_hit_reqs[req_id] = (block_idx, source_block)`(single_type_kv_cache_manager.py:113/280-286),`allocate_new_blocks` 时 `_apply_cow`(`single_type_kv_cache_manager.py:402-422`)把请求块表里该位置换成新分配的 `cow_block` 并记入 `_pending_cow_copies`;调度器每步通过 `KVCacheManager.take_kv_cache_block_copies`(`kv_cache_manager.py:840`)取走 `(src_block_id, dst_block_id)` 交给 worker 做真正的 GPU 拷贝,两端块在拷贝完成前都被 extra ref 钉住。

**缓存写入**:每步 `cache_blocks`(`kv_cache_manager.py:763` → coordinator :302 → 各 manager :424)调 `BlockPool.cache_full_blocks`(`block_pool.py:225`)给满块打哈希、`_insert_block_hash`(:607)进索引;块满又已有哈希时先 `_remove_cached_block_hashes` 再插(partial→full 提升,:284-297)。`cache_partial_block`(:445)支持把部分边界注册成缓存条目(细粒度命中用)。**注意缓存只提交"已 finalize"的 token**:投机草稿 token 会被 `request.num_tokens` 封顶排除(`kv_cache_manager.py:562-565`)。

**淘汰**:块 `ref_cnt == 0` 时留在空闲队列尾部,分配新块 `popleft` 到它时 `_maybe_evict_cached_block`(:679)清哈希并发 `BlockRemoved` 事件。`reset_prefix_cache`(`block_pool.py:764`)在 RLHF 换权重等场景整池清哈希。

### 5.5 前缀缓存统计与事件

- **统计**:`KVCacheManager.record_prefix_cache_stats`(`kv_cache_manager.py:221`)记录每次查找的命中 token 数/是否被抢占过,`make_prefix_cache_stats`(:205)取出并重置,汇入 `PrefixCacheStats`。
- **KV cache events**(可选):`enable_kv_cache_events` 开启后,块缓存/命中/淘汰会产生 `BlockStored`/`BlockRemoved`/`AllBlocksCleared` 事件(`block_pool.py:344-443`,`take_events`,`kv_cache_manager.py:680`),供 KV connector / gateway 等外部消费者做分布式缓存或用量统计;`kv_cache_report_mode="full"` 时命中也会发 `BlockStored`(`kv_cache_manager.py:269-287`)。事件里还会按组标注 `kv_cache_spec_kind` 和 sliding window(`kv_cache_manager.py:686-704`)。

---

## 6. 多组 KV cache:混合模型与 KVCacheCoordinator

### 6.1 为什么要有 group

一个模型的不同层可以有**不同的注意力/缓存类型**:全注意力、滑窗注意力(SWA)、chunked local attention、Mamba(线性注意力,缓存的是状态而非 K/V)、cross-attention、MLASpec(DeepSeek)。V1 把"缓存格式相同的层"归为一组(`KVCacheGroupSpec`,`kv_cache_interface.py:940`;分组逻辑 `get_kv_cache_groups`,`kv_cache_utils.py:1741`):

- 纯全注意力 → 1 组,所有层共享同一张块表;
- 混合模型(如 10 个 full + 20 个 sw)→ 按"层模式"拆多组,每组取每模式的一层,块表互不共享(`_get_kv_cache_groups_uniform_page_size`,`kv_cache_utils.py:1106-1246`);
- Mamba 的 `mamba_cache_mode`(`config/cache.py:137-145`):`none`(关前缀缓存)/ `all`(每个块边界存状态)/ `align`(仅调度步边界+块边界,省显存)。

每个 group 对应一个 `SingleTypeKVCacheManager` 子类(注册表工厂 `get_manager_for_kv_cache_spec`,`single_type_kv_cache_manager.py:1852`;内置注册 `register_all_kvcache_specs`,:1897)。manager 与 spec 的对应关系:FullAttention→`FullAttentionManager`、SlidingWindow→`SlidingWindowManager`、ChunkedLocal→`ChunkedLocalAttentionManager`、Mamba→`MambaManager`、CrossAttention→`CrossAttentionManager` 等。

### 6.2 KVCacheCoordinator:统一视图

`KVCacheCoordinator`(`kv_cache_coordinator.py:63`)把"共享一个 BlockPool"和"每个 group 一个 manager"组合起来,向上对 `KVCacheManager` 呈现**单一接口**:

- 构造时创建共享 `BlockPool`(:97-103)和 `single_type_managers` 元组(:135-149)。
- `get_num_blocks_to_allocate`(:159):对每个 manager 求和(cross-attention 按 encoder token 数一次性分配,:197-208)。
- `allocate_new_computed_blocks`(:221):**两阶段**(issue #33775):先 touch 所有组的本地命中块,再分配外部块,避免前一组的外部分配驱逐后一组未 touch 的命中块(:248-265)。
- `allocate_new_blocks`(:267)、`cache_blocks`(:302)、`free`(:324)、`remove_skipped_blocks`(:370)、`get_blocks`(:393)。

工厂函数 `get_kv_cache_coordinator`(`kv_cache_coordinator.py:915`)按配置选三种实现:

| 实现 | 适用 | 特点 |
|---|---|---|
| `KVCacheCoordinatorNoPrefixCache`(:419) | 前缀缓存关闭 | 任意组数(含 0),`find_longest_cache_hit` 恒空 |
| `UnitaryKVCacheCoordinator`(:471) | 单 group | 直接委托 manager 0;断言 `hash_block_size == block_size`(:515) |
| `HybridKVCacheCoordinator`(:559) | 多 group 混合模型 | 见 §6.3 |

### 6.3 Hybrid 的交叉命中协调

`HybridKVCacheCoordinator.find_longest_cache_hit`(`kv_cache_coordinator.py:749-881`)是**迭代不动点算法**:各注意力类型要么接受当前候选命中长度、要么把它调小,有任何一个调小就重启一轮,直到收敛(长度单调降,必收敛,:755-760)。优化:

- 全注意力**向下封闭**(块命中 ⇒ 前缀块必命中),扫一次缓存后只做裁剪(:802-809);
- 简单混合(1 full + 1 other)一轮即收敛(:782-784, :859-860);
- EAGLE 组多留一个块再丢,每次候选长度只 drop 一次(:811-829);
- 返回 `(各 group 命中块, 调和后长度, num_uncached_common_prefix_tokens)`——后者表示"某个稀疏组还没缓存、但全注意力组已命中的公共前缀",由 `get_computed_blocks` 转成 `shared_prefix_boundary`(`kv_cache_manager.py:232-298`)供 Mamba/SWA 保留交界点(`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`,`kv_cache_coordinator.py:33-60`)。

滑窗组的查找是**从右往左**找连续窗口块(`SlidingWindowManager.find_longest_cache_hit`,`single_type_kv_cache_manager.py:903-999`),命中窗口内的块放回对应位置、窗口外补 `null_block`;它只缓存"可达边界"附近的尾巴(`reachable_block_mask`,:1001-1061),避免缓存永远不可能命中的中部块。

### 6.4 一个混合模型命中的走查

假设模型是 **1 组全注意力(A)+ 1 组滑窗(S,window=8)**,`block_size=4`,请求 prompt 长 16 token(4 块)。缓存现状:两组都缓存了块 0..2,但滑窗组因为 `reachable_block_mask` 只缓存了"对齐边界尾巴",块 2 没进缓存表:

```
候选长度从 max_cache_hit_length=15 开始,轮询两组:
  第 1 轮  A 组: 从左到右 0,1,2 全中 → 候选长度保持 15
          S 组: 需要 2 个连续块(⌈8/4⌉)才构成窗口命中;从右往左找到块 1,0 连续,
                但块 2 不在缓存表 → 命中长度被压到 8 → 候选长度 = 8
  第 2 轮  A 组: 只需把已扫的块裁剪到 2 块(向下封闭) → 候选长度保持 8
          S 组: 候选 8 下恰好 2 连续块命中 → 收敛
结果: 命中长度 8, A 组返回 [block0, block1], S 组返回 [block0, block1]
      num_uncached_common_prefix_tokens = 8 (A 组本来可以到 12,被 S 组拉低)
```

这个"被拉低"的公共前缀就是 `shared_prefix_boundary` 的来源——它被钉住,避免 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 把交界点清掉导致后续请求无法复用(`kv_cache_manager.py:289-295`)。

### 6.5 分配防线:watermark 与 reserved_blocks

`allocate_slots` 在判定"能不能分配"时有两道防线(`kv_cache_manager.py:466-530`):

- **watermark 水位线**:`watermark_blocks = int(watermark × num_blocks)`(`kv_cache_manager.py:174`)是**必须保留的最小空闲块数**,只对 WAITING/PREEMPTED 请求生效,且仅当本步已有请求被调度时(`has_scheduled_reqs` 且请求在 `WAITING`/`PREEMPTED`,:469-473)。作用:防止新请求吃光空闲块、把已在跑的请求逼进抢占,制造"反复抢占"的抖动。水位线通常由 `--kv-cache-watermark` 配置。
- **`reserved_blocks`**:调用方(异步 KV connector 加载路径)要求本步分配后仍保留的空闲块数(:526),保证新加载的请求不会吃掉在途 prefill 依赖的块。
- **`full_sequence_must_fit`**:准入门,要求整条序列(非本 chunk)都能放下才分配(:475-491),防止 chunked prefill 下"第一块能过、后续 OOM"。

### 6.6 与异步调度 / 延迟释放的衔接

文档 03 §5.3 提过:异步调度(PP / `max_concurrent_batches>1`)下 `schedule()` 跑在 GPU 写 KV **之前**,被抢占/完成请求的块可能仍被在途 GPU 步骤写入。KV 侧的配合点:

- `Scheduler.defer_block_free` 开启时,释放走 `pop_blocks_for_free`(`kv_cache_manager.py:602`)+ 调度器延迟回池,而不是立即 `free`;`KVCacheManager.take_kv_cache_block_copies`(:840)取走 COW 拷贝任务,两端块由调度器用 `_free_cow_retained_blocks` 在拷贝完成后释放(文档 03 §4.4 的 `take_kv_cache_block_copies` 调用点)。
- `num_in_flight_tokens` / `last_sched_seq`(`request.py:163/171`)配合 `remove_skipped_blocks` 的"processed-token 基准"(`kv_cache_manager.py:507-511`):只按已确认处理的 token 边界释放滑窗块,在途步骤仍然能读到窗口内的旧块。
- `new_step_starts` 每步复位各 manager 的每步状态(`kv_cache_manager.py:885`),PP 多批次时由调度器控制调用节奏。

### 6.7 与调度器/执行器的接口

- `KVCacheManager.allocate_slots`(`kv_cache_manager.py:347`)的块布局注释(:393-425)给出完整分区:`<comp> | <new_comp> | <ext_comp> | <new> | <lookahead>`,调度器对每个请求算好 `num_new_tokens` 后调用它,拿不到块就触发抢占(文档 03 §4.2,scheduler.py:631)。
- 分配的三阶段(注释 :431-439):① 释放 `comp` 段多余块并检查空闲量;② 处理前缀段(`comp+new_comp+ext_comp`,滑窗外的块释放、`ext_comp` 块新分配);③ 为待计算 token 分配新块。
- 返回类型 `KVCacheBlocks`(`kv_cache_manager.py:33-115`):`blocks[i][j]` = 第 i 组第 j 块;`get_block_ids` 输出 worker 需要的 block_id 表,`new_empty` 复用预构造空实例省 GC(:111-115)。
- 需要清零的新块 ID 由 `take_new_block_ids`(`kv_cache_manager.py:805`)收集,`KVCacheConfig.needs_kv_cache_zeroing`(`kv_cache_interface.py:996`)决定是否需要(混合精度 / Mamba 必须清零,防止读到陈旧字节)。

---

## 7. 关键类 / 函数速查(文件:行号)

### `v1/core/kv_cache_utils.py`
- `KVCacheBlock`:118(字段:block_id:123 / ref_cnt:125 / _block_hash:128 / _block_hash_num_tokens:131 / prev·next_free_block:135-136 / is_null:139;`set_block_hash`:149 / `reset_hash`:160)
- `KVCacheBlockCopy`:180;`FreeKVCacheBlockQueue`:185(`popleft`:237 / `popleft_n`:274 / `remove`:307 / `append`:327 / `prepend_n`:350 / `append_n`:371)
- `hash_block_tokens`:577;`init_none_hash`:100;`generate_block_hash_extra_keys`:539
- `get_request_block_hasher`:671;`resolve_block_hashes`:2307;`BlockHashListWithBlockSize`:2231
- `resolve_kv_cache_block_sizes`:607;`get_num_blocks`:973(:988 公式);`get_kv_cache_config_from_groups`:1327;`get_kv_cache_groups`:1741;`get_kv_cache_configs`:2082
- `check_enough_kv_cache_memory`:834;`get_kv_cache_capacity`:1848;`get_max_concurrency_for_kv_cache_config`:917;`may_override_num_blocks`:942

### `v1/core/block_pool.py`
- `BlockHashToBlockMap`:33(`get_one_block`:61 / `contain`:74 / `insert`:88 / `pop`:106)
- `BlockPool`:143;`__init__`:162(建 blocks:175 / null_block:190);`get_cached_block`:198;`cache_full_blocks`:225;`cache_partial_block`:445;`_remove_cached_block_hashes`:571;`_insert_block_hash`:607;`move_block_hashes`:629
- `get_new_blocks`:647;`_maybe_evict_cached_block`:679;`touch`:702;`free_blocks`:719;`evict_blocks`:745;`reset_prefix_cache`:764;`get_num_free_blocks`:800;`get_usage`:808;`take_events`:821

### `v1/core/kv_cache_manager.py`
- `KVCacheBlocks`:33(`__add__`:56 / `get_block_ids`:77 / `get_unhashed_block_ids`:94 / `new_empty`:111)
- `KVCacheManager`:118;`__init__`:119(coordinator:153 / watermark_blocks:174);`get_computed_blocks`:232;`get_computed_blocks_for_connector`:300;`allocate_slots`:347(布局注释:393)
- `free`:570;`remove_skipped_blocks`:583;`pop_blocks_for_free`:602;`evict_blocks`:622;`reset_prefix_cache`:630;`get_num_common_prefix_blocks`:646;`take_events`:680
- `get_blocks`:706;`get_block_ids`:710;`get_block_ids_for_computed_tokens`:714;`estimate_cached_tokens`:734;`cache_blocks`:763;`create_kv_cache_blocks`:774;`truncate_computed_blocks`:780;`take_kv_cache_block_copies`:840;`new_step_starts`:885

### `v1/core/single_type_kv_cache_manager.py`
- `SingleTypeKVCacheManager`:36;`__init__`:44(req_to_blocks:94 / num_cached_block:100);`get_num_blocks_to_allocate`:141;`add_local_computed_blocks`:229;`allocate_external_computed_blocks`:288;`allocate_new_blocks`:327;`_apply_cow`:402;`cache_blocks`:424;`pop_blocks_for_free`:497;`free`:516;`remove_skipped_blocks`:624;`get_num_skipped_tokens`:663;`reachable_block_mask`:477
- `FullAttentionManager`:680(`find_longest_cache_hit`:684 / `cache_blocks`:781 / `_cache_partial_tail_block`:793 / `get_num_common_prefix_blocks`:823)
- `RSWAManager`:834;`SlidingWindowManager`:880(`find_longest_cache_hit`:903 / `get_num_skipped_tokens`:1063 / `reachable_block_mask`:1001);`ChunkedLocalAttentionManager`:1110;`SinkFullAttentionManager`:1826
- `get_manager_for_kv_cache_spec`:1852;`register_all_kvcache_specs`:1897

### `v1/core/kv_cache_coordinator.py`
- `KVCacheCoordinator`:63;`__init__`:70(BlockPool:97 / single_type_managers:135);`get_num_blocks_to_allocate`:159;`allocate_new_computed_blocks`:221;`allocate_new_blocks`:267;`cache_blocks`:302;`free`:324;`remove_skipped_blocks`:370;`get_blocks`:393
- `KVCacheCoordinatorNoPrefixCache`:419;`UnitaryKVCacheCoordinator`:471(`find_longest_cache_hit`:524);`HybridKVCacheCoordinator`:559(`_cache_hit_alignment_tokens`:647 / `verify_and_split_kv_cache_groups`:656 / `find_longest_cache_hit`:749 / `find_longest_cache_hit_per_group`:883);`get_kv_cache_coordinator`:915

### `v1/kv_cache_interface.py`(spec 类型)
- `KVCacheSpec`:141;`AttentionSpec`:217(`page_size_bytes`:254 / `max_num_blocks_per_req`:268);`FullAttentionSpec`:274(`max_memory_usage_bytes`:300);`MLAAttentionSpec`:381;`SlidingWindowSpec`:536(`max_admission_blocks_per_request`:546);`ChunkedLocalAttentionSpec`:496;`RSWASpec`:454;`MambaSpec`:667;`SinkFullAttentionSpec`:740;`UniformTypeKVCacheSpecs`:796
- `KVCacheTensor`:928;`KVCacheGroupSpec`:940;`KVCacheConfig`:955(`has_mamba_layers`:974 / `needs_kv_cache_zeroing`:996);`KVCacheSpecKind`:128;`get_kv_cache_spec_kind`:875;`KVQuantMode`:36

### `v1/request.py`(哈希侧)
- `Request.block_hashes`:206;`append_output_token_ids`:252;`update_block_hashes`:265;`num_computed_tokens`:174;`skip_reading_prefix_cache`:294

### 配置 / Worker
- `CacheConfig.DEFAULT_BLOCK_SIZE = 16`:`config/cache.py:48`;`block_size` 默认值应用:`config/cache.py:256-269`;`gpu_memory_utilization=0.92`:`:69`;`enable_prefix_caching=True`:`:96`;`prefix_caching_hash_algo="sha256"`:`:98`;`prefix_match_unit`:`:57`;`mamba_cache_mode`:`:137-145`
- `Worker.determine_available_memory`:`v1/worker/gpu_worker.py:475`(公式 :559-563);`request_memory`:`v1/worker/utils.py:444`;`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`:`v1/core/kv_cache_coordinator.py:154`
- hasher 组装:`v1/engine/core.py:223-232`;`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`:`v1/worker/gpu_worker.py:535-539`;`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`:`v1/core/kv_cache_utils.py:80-83`

---

## 延伸阅读

- 论文:SOSP'23《Efficient Memory Management for Large Language Model Serving with PagedAttention》 <https://arxiv.org/abs/2309.06180>
- vLLM Blog:《vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention》(块式 KV 与吞吐提升的动机) <https://blog.vllm.ai/2023/06/20/vllm.html>
- vLLM 官方文档:Automatic Prefix Caching(自动前缀缓存原理与命中率观察) <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- vLLM 官方文档:调度器设计说明(与 KV 管理的接口 `allocate_slots`/`get_computed_blocks`) <https://docs.vllm.ai/en/latest/design/scheduler.html>
- vLLM Blog:《Announcing vLLM V1: A Major Architectural Upgrade》(V1 重构 KV 管理动机) <https://blog.vllm.ai/2025/01/27/v1-alpha-release.html>
- 相关文档:本仓库 repowiki `01_system_architecture.md`(KV Cache Manager 在 EngineCore 中的位置)、`03_Scheduler.md`(`allocate_slots` 的调用点 scheduler.py:631、抢占与延迟释放如何与块池交互)