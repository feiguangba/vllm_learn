# 03 · vLLM V1 调度器:Token 预算驱动的一次性调度与请求生命周期

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `v1/core/sched/scheduler.py:476` = 上游源码中的 `vllm/v1/core/sched/scheduler.py` 第 476 行)。
> **一句话总结**:V1 调度器**取消了独立的 prefill/decode 阶段**,把调度彻底泛化为"给每个请求分配 token 预算、让 `num_computed_tokens` 追赶 `num_tokens_with_spec`"的纯预算问题;`schedule()` 每步产出 `SchedulerOutput` 交给模型前向,`update_from_output()` 消费 GPU 输出并推进请求状态机,配合抢占(recompute)与连续批处理实现动态的 token 级流水线。

---

## 一图看懂

```
                          ┌─────────────────────────────────────────────┐
                          │              EngineCore busy loop            │
                          │             (v1/engine/core.py:1375)          │
                          └──────────────────────┬──────────────────────┘
                                                 │ step() (core.py:583)
                                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     Scheduler.schedule()  (scheduler.py:476)              │
│                                                                           │
│   token_budget = max_num_scheduled_tokens   (scheduler.py:496)            │
│   input_budget = max_num_batched_tokens     (scheduler.py:499)            │
│                                                                           │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │ ① 调度 RUNNING 请求 (scheduler.py:523)                            │    │
│   │    num_new_tokens = num_tokens_with_spec + num_output_placeholders │    │
│   │                     - num_computed_tokens        (scheduler.py:558)│    │
│   │    → 预算不足时抢占最低优先级请求 (scheduler.py:643, 678)           │    │
│   │    → kv_cache_manager.allocate_slots(...)  (scheduler.py:631)      │    │
│   ├──────────────────────────────────────────────────────────────────┤    │
│   │ ② 调度 WAITING 请求 (scheduler.py:747)                             │    │
│   │    → 前缀缓存查询 get_computed_blocks (scheduler.py:811-884)       │    │
│   │    → 编码器输入 _try_schedule_encoder_inputs (scheduler.py:981)    │    │
│   │    → allocate_slots → 进入 running → status = RUNNING (scheduler.py:1135)│
│   ├──────────────────────────────────────────────────────────────────┤    │
│   │ ③ 组装 SchedulerOutput (scheduler.py:1270)                         │    │
│   │    scheduled_new_reqs / scheduled_cached_reqs / num_scheduled_tokens│   │
│   └──────────────────────────────────────────────────────────────────┘    │
│   → _update_after_schedule(): 推进 num_computed_tokens (scheduler.py:1379)│
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │ execute_model + sample_tokens (GPU 前向)
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│               Scheduler.update_from_output()  (scheduler.py:1733)         │
│   · 消费 sampled_token_ids, 追加 output token (scheduler.py:1874/2182)     │
│   · spec decode 接受/拒绝 → 回滚 num_computed_tokens (scheduler.py:1830)   │
│   · 检查停止条件 check_stop (scheduler.py:2195)                            │
│   · 停止 → _handle_stopped_request → _free_request → 释放 KV (scheduler.py:1976)│
│   · 产出 EngineCoreOutputs → 前端 (scheduler.py:2102)                      │
└───────────────────────────────────────────────────────────────────────────┘

请求生命周期:  add_request ─WAITING→ [RUNNING ──反复调度─→] → 停止 → FINISHED_*
                (scheduler.py:2301)         ↑ 预算不足             (scheduler.py:2325)
                                            └─ PREEMPTED (num_computed_tokens=0,
                                               scheduler.py:1336) ─→ 重新入 waiting
```

---

## 1. 总体结论:V1 的调度理念

V1 调度器最重要的设计决定,写死在 `schedule()` 顶部的一段注释里(`scheduler.py:478-487`):

> *There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.*

这意味着:

1. **没有阶段,只有预算**。V0 里 prefill 一次吃完整条 prompt、decode 每步只吃 1 个 token 的二元对立被抹平了。任意一个请求,无论它处于"首次 prefill""chunked prefill""已开始 decode""被抢占后重放""spec decode 补充 draft"……在调度器眼里都只是同一件事:**还差多少个 token 没算**(`num_tokens_with_spec - num_computed_tokens`)。
2. **一个通用的 token 分配模型**。这个模型自然覆盖了 chunked prefill、前缀缓存、投机解码,以及未来的"跳跃解码"优化(`scheduler.py:486-487` 注释)——因为它们都只是改变"一次给请求分几个 token"。
3. **调度是迭代级的**。`SchedulerInterface.schedule()` 的 docstring(`interface.py:53-83`)明确:每次调度决策对应模型的一次前向,`schedule()` 被引擎 busy loop 反复调用;核心产物就是一张 `{req_id: num_tokens}` 的映射表,告诉模型这一步每个请求处理几个 token。

> **与 V0 的历史对照**:V0 的调度器(block 分配 + prefill/decode 双队列 + `_schedule_prefills/_schedule_runs`)本质仍是"阶段式"。V1 把它压成"**token 预算 + 两条队列(waiting/running)+ 一个 FCFS/PRIORITY 策略**",调度逻辑的复杂度大幅下降,同时能自然表达连续批处理。

三个核心预算(`scheduler.py:110-124`、`496-499`):

| 预算 | 取值来源 | 作用 |
|---|---|---|
| `token_budget` | `max_num_scheduled_tokens`(默认取 `max_num_batched_tokens`,`scheduler.py:110-115`) | 本步最多调度多少 **GPU 计算** token |
| `input_budget` | `max_num_batched_tokens`(`scheduler.py:499`) | 本步最多吃进多少 **输入位置**(含 draft 预留) |
| `encoder_compute_budget` | `max_num_encoder_input_tokens`(`scheduler.py:232`/`506`) | 本步编码器(视觉/音频)最多处理的 embedding 数 |

`schedule()` 每步都要保证最终约束成立(`scheduler.py:1169-1181`):调度 token 总数 ≤ `max_num_scheduled_tokens`,running 请求数 ≤ `max_num_seqs`。

---

## 2. 请求的核心计数器:num_computed_tokens vs num_tokens_with_spec

`Request`(`request.py:59`)是调度器操作的最小单元。与调度直接相关的三个计数器(`request.py:270-284`):

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)                    # prompt + output

@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)  # + draft

@property
def num_output_tokens(self) -> int:
    return len(self._output_token_ids)
```

- `num_computed_tokens`(`request.py:174`,初值 0):**调度器认为已经算出 KV 的 token 数**。它不是一个稳定值,而是随调度/产出不断前进,也会因 spec 拒绝、KV 加载失败而**回滚**。
- `num_tokens_with_spec`(`request.py:279`):**这个请求"应该"算到的目标位置** = prompt token + 已产出 token + 投机 draft token。
- 调度目标就是 `num_computed_tokens → num_tokens_with_spec`。

另外几个在**异步调度(PP / async scheduling)**下至关重要的计数器(`request.py:151-171`):

- `num_output_placeholders`(:152):已排进 GPU 但尚未消费的输出占位(draft token 也算),保证预算不被重复计数。
- `num_in_flight_tokens`(:163):**在途 token 数**。异步调度时 `schedule()` 跑在 GPU 前向之前,`num_computed_tokens` 是"乐观"推进的;这个字段记录还有几步的产出没回来。
- `num_stale_output_tokens`(:155)/`drop_stale_output`(:158):抢占时在途产出被标记为 stale,返回时仍投递但**不改计数器**。
- `last_sched_seq`(:171):该请求最后一次被调度的 step 序号,用作延迟释放块的**栅栏**(见 §5.3)。

> `num_computed_tokens` 的"乐观推进"发生在 `_update_after_schedule`(`scheduler.py:1392`):调度完立刻把 `num_computed_tokens += num_scheduled_token`、`num_in_flight_tokens += num_scheduled_token`,这样**同一请求可以在下一步立刻再次被调度**(尤其 chunked prefill)。若后续 spec token 被拒,再在 `update_from_output` 里回滚(`scheduler.py:1846`)。

---

## 3. 请求状态机(RequestStatus)

`RequestStatus`(`request.py:351-367`)是一个 `IntEnum`,注意它的**顺序即语义**——`is_finished()` 就是 `status > PREEMPTED`(`request.py:373`):

```
WAITING                        :354   排队等待调度
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR :355   等结构化输出的 grammar 编译好
WAITING_FOR_REMOTE_KVS         :356   KV Connector 异步传输中
WAITING_FOR_STREAMING_REQ      :357   流式会话等待下一段输入
RUNNING                        :358   已入 running 队列,参与每步调度
PREEMPTED                      :359   被抢占,等重新入队
── 以下均为"已完成" ──
FINISHED_STOPPED               :362   正常停止(eos/stop 词)
FINISHED_LENGTH_CAPPED         :363   触达 max_tokens / max_model_len
FINISHED_ABORTED               :364   客户端 abort
FINISHED_IGNORED               :365   prompt 超长被忽略
FINISHED_ERROR                 :366   出错(grammar 拒绝等)
FINISHED_REPETITION            :367   重复惩罚触发
```

状态→`FinishReason` 的映射在 `_FINISHED_REASON_MAP`(`request.py:385-393`),例如 `FINISHED_STOPPED → STOP`、`FINISHED_LENGTH_CAPPED → LENGTH`、`FINISHED_ABORTED → ABORT`。

调度器内部维护两组队列(`scheduler.py:179-191`):

- `waiting` / `skipped_waiting`:等待调度的请求。`skipped_waiting` 专门存放**暂时被阻塞**的请求(等 grammar / 等远程 KV / 等流式输入),见 `_enqueue_waiting_request`(`scheduler.py:2146-2150`)、`_is_blocked_waiting_status`(`scheduler.py:2138`)。`get_request_counts` 把两者合并计数(`scheduler.py:2293-2295`)。
- `running`(`scheduler.py:191`):**一个普通 list**,每步遍历它给每个请求分 token。

调度策略由 `SchedulingPolicy`(`request_queue.py:13-17`)**FCFS / PRIORITY** 决定,对应两种队列实现(`request_queue.py:75` `FCFSRequestQueue`、`:131` `PriorityRequestQueue`),由 `create_request_queue`(`request_queue.py:201`)按策略创建。PRIORITY 模式按 `(priority, arrival_time, request_id)` 比较(`request.py:337-348`)。

---

## 4. schedule():主流程逐步拆解

`schedule(throttle_prefills)`(`scheduler.py:476`)是每步的核心。整体分四段。

### 4.1 重置与预算初始化(`scheduler.py:477-521`)

```python
self.current_step += 1
...
token_budget = self.max_num_scheduled_tokens      # :496
input_budget = self.scheduler_config.max_num_batched_tokens  # :499
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0                              # :500-502 暂停则一个都不调度
...
self.kv_cache_manager.new_step_starts()           # :515 每步 KV 管理器重置
```

`PauseState`(`interface.py:24-35`)有三个档位:`UNPAUSED`(正常)/ `PAUSED_NEW`(不接新请求,继续跑 running)/ `PAUSED_ALL`(全停)。此外还有 `throttle_prefills` 参数,是 **DP prefill 负载均衡**(多个 DP rank 对齐 prefill 节奏)用的,`defer_prefills`(`scheduler.py:519-521`)会在非对齐步延迟 prefill。

### 4.2 第一步:调度 RUNNING 请求(`scheduler.py:523-736`)

遍历 `self.running`,对每个请求算 `num_new_tokens`(`scheduler.py:558-562`):

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

然后逐层裁剪(`scheduler.py:563-606`):

- 受 `long_prefill_token_threshold` 限制(:563)。
- `min(num_new_tokens, token_budget, input_budget - draft_slots)`(:565-567):**同时受计算预算和输入预算约束**。
- 不越过 `max_model_len`(:571-576)。
- Mamba 块对齐 `_mamba_block_aligned_split`(:579-582)。
- 编码器输入预算 `_try_schedule_encoder_inputs`(:584-600)。
- 预留 speculative lookahead `_reserve_prefill_lookahead`(:604)。

如果 `num_new_tokens == 0`(`scheduler.py:608`),`continue` 跳过该请求(**不 `break`**,这是刻意的——不严格守 FCFS,允许后面的低优先级请求被调度,见 :622-624 注释)。

真正决定"能不能上"的是 **KV 块分配**(`scheduler.py:629-635`):

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
    if new_blocks is not None:
        break
    # 分配失败 → 抢占最低优先级请求
    ...
```

分配失败时进入抢占(见 §5):PRIORITY 策略选 `max(running, key=(priority, arrival_time))`(:643-677),FCFS 直接 `running.pop()`(:678),然后 `_preempt_request`(:680-685)。若最后发现自己也被抢了(无法自给自足)就 `break`(:690-692)。

成功调度后记账(`scheduler.py:694-735`):入 `scheduled_running_reqs`、记 `req_to_new_blocks`、`num_scheduled_tokens`、扣减 `token_budget` 和 `input_budget`,并处理 spec token(:704-720)与编码器输入(:722-735)。

### 4.3 第二步:调度 WAITING 请求(`scheduler.py:747-1167`)

只有**没有发生抢占**且**未暂停**时才调度等待队列(`scheduler.py:748`)。每轮从 `_select_waiting_queue_for_scheduling()`(`scheduler.py:2152-2162`)挑队列头:

- FCFS:先 `skipped_waiting` 再 `waiting`(:2153-2154)。
- PRIORITY:比较两队列头谁更小(:2157-2160)。

对每个等待请求依次检查:

1. **阻塞状态提升** `_try_promote_blocked_waiting_request`(`scheduler.py:2766`):远程 KV 已就绪 / grammar 编译好 / 流式输入到达 → 转回可调度态;否则留在 skipped(:767-777)。
2. **stale 输出仍在途** → 跳过(:779-788)。
3. **LoRA 容量** → 跳过(:792-803)。
4. **前缀缓存查询**(仅首次,`num_computed_tokens == 0` 时,`scheduler.py:811-884`):`get_computed_blocks` 找出本地可复用块,得到 `num_new_local_computed_tokens`;若配置了 KVConnector 还查询远端命中 `get_num_new_matched_tokens`(:829-863),合并成 `num_computed_tokens`(:881-883)。
5. **编码器输入调度** `_try_schedule_encoder_inputs`(:981-993,逻辑见 §4.5)。
6. **分配 KV 块** `allocate_slots`(:1033-1045)。
7. **入队 + 置 RUNNING**(`scheduler.py:1115-1135`):

```python
self.running.append(request)
...
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)   # 被抢占后重放的请求
...
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

对 `WAITING_FOR_REMOTE_KVS` 的请求(异步 KV 加载),只置计数、进 `skipped_waiting`、`continue` 不真正运行(:1083-1113)。

### 4.4 第三步:组装 SchedulerOutput 与收尾(`scheduler.py:1169-1315`)

- **约束断言**(:1169-1181):总调度 token、预算非负、running 上限。
- **公共前缀块数**(:1183-1191):为 cascade attention 预留,取任一 running 请求的最长公共前缀。
- **构造 `NewRequestData`**(:1194-1211):新请求的完整 prompt/block 信息;V2 runner 还带 `_all_token_ids`。
- **构造 `CachedRequestData`**(`_make_cached_request_data`,:1472-1529):老请求只需增量(新 block、num_computed_tokens 等),降低通信量。
- **组装 `SchedulerOutput`**(:1270-1291,结构见 output.py:193)。
- **CoW 副本 / connector 元数据**:`take_kv_cache_block_copies`(:1243)、KVConnector meta(:1297-1299)、ECConnector meta(:1302-1306)。
- **推进 `sched_step_seq`**(仅非空步,:1310-1311),为延迟释放的栅栏铺路。
- **`_update_after_schedule`**(:1313-1314,见 §6)。

### 4.5 编码器输入调度 `_try_schedule_encoder_inputs`(`scheduler.py:1531`)

多模态/编码器-解码器模型的编码器部分由独立预算管理:

- 哪些输入要调度:其输出 token 与本步计算区间 `[num_computed_tokens, num_computed_tokens+num_new_tokens)` 重叠且未缓存(`scheduler.py:1543-1549`、`1573-1584`)。
- 已缓存(`check_and_update_cache`)→ 跳过(:1616-1619)。
- 无编码器预算/缓存 → 回退只调度到编码器输入之前的 token(:1637-1656)。
- 返回 `(encoder_inputs_to_schedule, num_new_tokens, encoder_compute_budget, external_load_encoder_input)`。

---

## 5. 抢占(preemption):recompute 语义

### 5.1 何时抢占

在 §4.2 的 `allocate_slots` 循环里,当**新请求需要 KV 块但池子不够**时,就抢占一个低优先级请求释放它的块(`scheduler.py:643-688`)。抢占本身委托给 `_preempt_request`。

### 5.2 `_preempt_request`(`scheduler.py:1336-1377`)

```python
assert request.status == RequestStatus.RUNNING     # 只有 running 能抢
self._free_request_blocks(request)                 # 释放 KV 块(:1352)
self.encoder_cache_manager.free(request)           # 释放编码器缓存(:1353)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED           # :1355
request.num_computed_tokens = 0                    # ← recompute 语义(:1356)
if request.spec_token_ids:
    request.spec_token_ids = []                    # :1357-1358
request.drop_stale_output = ...                    # :1366-1368
request.num_stale_output_tokens = request.num_in_flight_tokens  # :1369
request.num_output_placeholders = 0                # :1370
request.num_preemptions += 1                       # :1371
...
self.waiting.prepend_request(request)              # 回到 waiting 队头(:1376)
self.reset_preempted_req_ids.add(request.request_id)  # :1377
```

**关键点**:`num_computed_tokens = 0`(`scheduler.py:1356`)意味着这个请求的 KV 被**整体作废**——它重新进 waiting 后,要么从前缀缓存重新命中一部分,要么**从头重算(recompute)**。这就是 vLLM 的 **recompute 抢占**(相对 V0 还有过 swap 抢占;V1 里 KV 不能换到 CPU,只能丢弃重算)。`num_preemptions`(:1371)被统计,前缀缓存命中统计也会区分是否被抢占过(`scheduler.py:1073`)。

### 5.3 延迟释放与 last_sched_seq 栅栏

异步调度(PP / async scheduling / KV Connector)下,`schedule()` 跑在 GPU 写 KV **之前**,被抢占/完成的请求其块可能仍被在途的 GPU 步骤写入。V1 用 `defer_block_free` + `last_sched_seq` 栅栏解决(`scheduler.py:132`、`request.py:171`):

- `defer_block_free`(`scheduler.py:132/155-157`)在**有多个在途批次**(`max_concurrent_batches>1`,PP 或异步)或 KV Connector 消费端时开启。
- 每次调度非空步推进 `sched_step_seq`(`scheduler.py:1310-1311`),并把每个请求的 `last_sched_seq` 记为该步序号(`_update_after_schedule`,`scheduler.py:1396`)。
- `_free_request_blocks`(`scheduler.py:2429-2442`)判断:若 `request.last_sched_seq <= processed_step_seq`(最后一步的 GPU 写已处理完)就立即释放;否则把块塞进 `deferred_frees` FIFO(`scheduler.py:334`)。
- `update_from_output` 每消费完一步就 `processed_step_seq += 1` 并 `_drain_deferred_frees`(`scheduler.py:1750-1752`),`_drain_deferred_frees`(`scheduler.py:2455-2468`)把 fence ≤ processed 的块真正归还块池。

> 一句话:**块的物理释放被推迟到"最后一次可能写它的 GPU 步骤确认完成"之后**,`last_sched_seq` 就是这个确认的栅栏序号。

---

## 6. 调度后的状态推进:_update_after_schedule(`scheduler.py:1379`)

在 `schedule()` 末尾调用(注意它发生在**返回 SchedulerOutput 之前**,但产出的是给"下一步"用的状态):

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request.num_computed_tokens += num_scheduled_token   # :1392 乐观推进
    request.num_in_flight_tokens += num_scheduled_token  # :1393
    if self.defer_block_free:
        request.last_sched_seq = self.sched_step_seq     # :1396 栅栏
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders)  # :1397-1399
    ...
self.finished_req_ids = set()        # :1426 清空上一步的 finished
self.reset_preempted_req_ids = set() # :1427 清空抢占标记
```

- `is_prefill_chunk`(:1397):只要 computed 还没到目标就仍算 prefill chunk,驱动 Mamba 对齐、结构化输出等分支。
- `finished_req_ids` / `reset_preempted_req_ids` 之所以 `= set()` 而不是 `.clear()`(`scheduler.py:1424-1427` 注释):因为它们已被 `SchedulerOutput` 引用,新建空集避免影响已构造的 output。

---

## 7. update_from_output:消费 GPU 输出(`scheduler.py:1733-2136`)

这一步是"输出侧"的回调,把模型前向的结果(采样 token、logprobs、spec 接受情况、KV connector 反馈)转成 `EngineCoreOutputs` 并推进请求状态。整体结构:

1. **延迟释放闸门**(`scheduler.py:1750-1752`):非空步则 `processed_step_seq += 1` + `_drain_deferred_frees()`。
2. **无效 KV 块处理**(`scheduler.py:1761-1769`):`_handle_invalid_blocks` 处理远端 KV 加载失败,回滚受影响请求的 computed token 数(§5.3 相关)。
3. **主循环**(`scheduler.py:1797-2028`):遍历 `num_scheduled_tokens`,每个请求:
   - 扣减 `num_in_flight_tokens`(:1802),并按在途比例扣减 stale 输出(:1804-1807)。
   - 取采样 token `sampled_token_ids[req_index]`(:1825-1828)。
   - **spec decode 接受/拒绝**(:1830-1855):`num_accepted = len(generated) - num_sampled`、`num_rejected = num_draft - num_accepted`;拒绝则回滚 `num_computed_tokens` 和 `num_output_placeholders`(:1845-1848)——这是"乐观推进"后的校正。
   - **追加输出 token + 检查停止** `_update_request_with_output`(:1874,定义 :2182-2199):`append_output_token_ids` → `check_stop(request, max_model_len)`(`utils.py:94`)逐个 token 检查 eos/stop 词,命中则裁剪 token 并置 `stopped`。
   - **结构化输出语法**推进(:1893-1919),grammar 拒绝 token 则 `FINISHED_ERROR`。
   - **处理停止**(:1971-1983):`_handle_stopped_request`(定义 :2164-2180,可 resumable 的流式请求转为 `WAITING_FOR_STREAMING_REQ` 等待下一段输入,否则返回 True 表示真正结束);结束后 `_free_request`(:1978)。
   - **组装 `EngineCoreOutput`**(:2007-2025):`new_token_ids`、`finish_reason`、`new_logprobs`、`pooling_output`、`prefill_stats`、`routed_experts` 等。
4. **移除已停止请求**(:2030-2036):`remove_all(running, ...)`(`utils.py:62`)。
5. **收尾**:错误请求 `finish_requests(FINISHED_ERROR)`(:2043-2056)、KV/EC connector 状态更新(:2058-2064)、发布 KV 事件(:2096-2098)、把各 client 的输出聚成 `engine_core_outputs` 字典返回(:2102-2136)。

### 7.1 哈希更新:append 时同步

`_update_request_with_output` 调 `request.append_output_token_ids`(`request.py:252-263`),它把 token 追加进 `_output_token_ids` 和 `_all_token_ids`,并调用 `update_block_hashes()`(`request.py:265-268`)增量计算新满块的哈希,供前缀缓存(文档 02 的 KV Cache Manager)使用。

### 7.2 停止条件汇总

停止检查发生在 `check_stop`(`v1/core/sched/utils.py:94`),触发后进入 `_handle_stopped_request`。真正结束与否取决于 `request.resumable`(`request.py:216`):普通请求直接结束;流式请求(reasoning/流式会话)若 `streaming_queue` 还有下一段输入则转 `WAITING_FOR_STREAMING_REQ` 继续(`scheduler.py:2169-2180`)。结束后统一走 `_free_request`(`scheduler.py:2388-2415`)→ 释放编码器缓存、登记 `finished_req_ids`、必要时释放 KV 块。

---

## 8. 连续批处理与请求生命周期(串起来)

连续批处理(continuous batching)在 V1 里是**token 预算调度的自然结果**,不需要专门的状态机:

- **每步自由混批**:`schedule()` 同时考虑 running(续算/decode)和 waiting(新 prefill/chunked prefill),用预算把它们塞进同一批(`scheduler.py:523` 与 `:747` 两段)。GPU 空出来就立刻有新请求补上,无需等整批结束。
- **请求的动态进出**:
  - 进入:`add_request`(`scheduler.py:2301-2323`)把请求放进 waiting(或追加到流式会话),`schedule` 中置 RUNNING。
  - 运行:反复被 `schedule` 分配 token、`update_from_output` 推进 computed/output。
  - 暂停:`_preempt_request` 置 PREEMPTED、清零 computed、回 waiting。
  - 结束:`_update_request_with_output` 检测停止 → `_handle_stopped_request` → `_free_request` → `FINISHED_*`。
  - 外部中止:`finish_requests`(`scheduler.py:2325-2386`,客户端 disconnect / abort 时由引擎调用)可批量结束任意请求并立即释放其块。
- **空闲判定**:`has_requests`(`scheduler.py:2494`)综合未完成请求、待清理的 finished 请求以及 connector 的未决推送工作,决定引擎是否还能继续 loop。

---

## 9. 关键类 / 函数速查(文件:行号)

### `v1/core/sched/scheduler.py`(核心)
- `Scheduler`:69;`__init__`:70;`schedule`:476;`_get_local_prefix_cache_hit`:443;`_reserve_prefill_lookahead`:455;`_mamba_block_aligned_split`:366
- `_preempt_request`:1336;`_update_after_schedule`:1379;`_make_cached_request_data`:1472;`_try_schedule_encoder_inputs`:1531
- `get_grammar_bitmask`:1709;`update_from_output`:1733;`update_draft_token_ids`:2234;`update_draft_token_ids_in_output`:2256
- `_is_blocked_waiting_status`:2138;`_enqueue_waiting_request`:2146;`_select_waiting_queue_for_scheduling`:2152
- `_handle_stopped_request`:2164;`_update_request_with_output`:2182;`_free_encoder_inputs`:2201
- `get_request_counts`:2293;`get_kv_cache_usage`:2297;`add_request`:2301;`finish_requests`:2325
- `_free_request`:2388;`_free_blocks`:2417;`_free_request_blocks`:2429;`_free_cow_retained_blocks`:2444;`_drain_deferred_frees`:2455
- `get_num_unfinished_requests`:2470;`has_finished_requests`:2482;`has_requests`:2494;`reset_prefix_cache`:2511
- `_update_waiting_for_remote_kv`:2723;`_try_promote_blocked_waiting_request`:2766;`_update_requests_with_invalid_blocks`:2831;`_handle_invalid_blocks`:2934
- `make_stats`:2583;`make_spec_decoding_stats`:2621;`shutdown`:2640

### `v1/core/sched/interface.py`
- `PauseState`:24(UNPAUSED=0 / PAUSED_NEW=1 / PAUSED_ALL=2)
- `SchedulerInterface`:38(抽象基类:schedule / update_from_output / add_request / finish_requests / reset_prefix_cache 等)

### `v1/core/sched/request_queue.py`
- `SchedulingPolicy`:13(FCFS / PRIORITY);`RequestQueue`:20;`FCFSRequestQueue`:75;`PriorityRequestQueue`:131;`create_request_queue`:201

### `v1/core/sched/output.py`
- `NewRequestData`:35(`from_request`:51);`CachedRequestData`:116;`ScheduledEncoderInputStats`:185;`SchedulerOutput`:193(`make_empty`:272);`GrammarOutput`:286

### `v1/core/sched/utils.py`
- `remove_all`:62;`check_stop`:94

### `v1/request.py`
- `StreamingUpdate`:33;`Request`:59;`num_tokens`:275;`num_tokens_with_spec`:279;`num_output_tokens`:283;`append_output_token_ids`:252;`update_block_hashes`:265;`is_finished`:307
- `RequestStatus`:351;`is_finished`:373;`get_finished_reason`:377;`_FINISHED_REASON_MAP`:385

### 相关环境变量 / 配置
- `--max-num-batched-tokens` → `max_num_batched_tokens`(token 预算与 `input_budget` 来源,`scheduler.py:114/499`)。
- `--max-num-seqs` → `max_num_seqs`(running 请求上限,`scheduler.py:110`)。
- `--enable-chunked-prefill` → `enable_chunked_prefill`(允许长请求被拆分调度,`scheduler.py:959`)。
- `--long-prefill-token-threshold` → `long_prefill_token_threshold`(超长 prefill 的 token 上限,`scheduler.py:563/952`)。
- `--scheduler-policy` → `policy`(fcfs / priority,`scheduler.py:182`)。
- `--max-concurrent-batches` → 是否开启 `defer_block_free`(`scheduler.py:155`)。
- `--kv-load-failure-policy` → `recompute_kv_load_failures`(fail / recompute,`scheduler.py:148`)。

---

## 延伸阅读

- vLLM 官方文档:调度器设计说明 <https://docs.vllm.ai/en/latest/design/scheduler.html>
- vLLM 官方文档:架构总览(EngineCore / 调度循环) <https://docs.vllm.ai/en/latest/design/arch_overview.html>
- 论文:Orca —— 连续批处理(continuous batching)与迭代级调度 <https://arxiv.org/abs/2208.14217>
- vLLM Blog:《vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention》(PagedAttention 与块式 KV 管理) <https://blog.vllm.ai/2023/06/20/vllm.html>
- vLLM Blog:《Announcing vLLM V1: A Major Architectural Upgrade》(V1 调度器泛化的动机) <https://blog.vllm.ai/2025/01/27/v1-alpha-release.html>
- 论文:SOSP'23 PagedAttention <https://arxiv.org/abs/2309.06180>
- 相关文档:本仓库 repowiki `01_system_architecture.md`(三层进程拓扑,调度器在 EngineCore 中的位置)、`02_*`(KV Cache Manager,`allocate_slots`/`free` 的底层实现)
