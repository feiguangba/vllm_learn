# 05 · vLLM V1 ModelRunner 与 CUDA Graph:从 SchedulerOutput 到 GPU 前向

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `v1/worker/gpu/model_runner.py:158` = 上游源码中的 `vllm/v1/worker/gpu/model_runner.py` 第 158 行)。
> **一句话总结**:ModelRunner 是 Worker 进程内的"GPU 前端",负责把调度器产出的 `SchedulerOutput` 翻译成一组固定形状的 GPU 张量(`input_ids`/`positions`/`query_start_loc`/block table/slot mapping)并驱动模型前向;为了让一次前向的数千次 kernel 启动变成一次 `graph.replay()`,vLLM 先用 `profile_run()` 测定显存、按 `cudagraph_capture_sizes` 预热并捕获一组 **shape 固定**的 CUDA Graph(共享一个 graph 内存池),运行期再按"取最近的捕获尺寸向上 padding"派发,而 `torch.compile`(VLLM_COMPILE + VllmBackend)负责把图拆成可捕获的 piecewise 子图并做 Inductor 融合与缓存。

---

## 一图看懂

```
                       Worker (v1/worker/gpu_worker.py:142)
   ┌───────────────────────────────────────────────────────────────────────┐
   │ 启动期                                                                  │
   │  load_model ─── model_runner.load_model()   (gpu/model_runner.py:350)  │
   │  determine_available_memory ── profile_run() + memory_profiling        │
   │        (gpu_worker.py:475)      (gpu/model_runner.py:790)              │
   │  initialize_from_config ── 按测得内存分配 KV cache (gpu_worker.py:665)   │
   │  compile_or_warm_up_model (gpu_worker.py:694):                         │
   │     ├─ warmup_sizes: _dummy_run(size)  编译预热 (gpu/model_runner.py:633)│
   │     ├─ kernel_warmup                                                    │
   │     └─ capture_model()  (gpu/model_runner.py:848)                      │
   │          └─ ModelCudaGraphManager.capture() (cudagraph_utils.py:307)   │
   │               warmup(NONE) → torch.cuda.graph(pool) → replay 候选表      │
   └───────────────────────────────┬───────────────────────────────────────┘
                                    │ 运行期 (每步, busy loop 驱动)
                                    ▼
   execute_model(scheduler_output)  (gpu/model_runner.py:1408)
     ├─ add_requests / update_requests: 请求状态写进 req_states + block_tables
     ├─ dispatch_cg_and_sync_dp: 用 CudagraphDispatcher 决定 cg_mode
     │    (NONE / PIECEWISE / FULL) + batch_desc  (cudagraph_dispatcher.py:235)
     ├─ prepare_inputs: 组装 InputBatch (固定 buffer 上重填, :1102)
     ├─ prepare_attn: block_tables + slot_mappings (:1303)
     ├─ model_state.prepare_attn → 各层 AttentionMetadata (backend.py:455/463)
     ├─ 前向三态:
     │    FULL      → cudagraph_manager.run_fullgraph(batch_desc)  (cudagraph_utils.py:412)
     │    PIECEWISE → run_pw_graph (torch-compiled piecewise cudagraph / breakable)
     │    NONE      → self.model(**model_inputs) 直接 eager
     └─ 存 execute_model_state (NamedTuple, :1980)
   sample_tokens(grammar_output)  (gpu/model_runner.py:1707)
     ├─ sample(): compute_logits → (grammar 过滤) → Sampler / RejectionSampler
     ├─ postprocess_sampled(): 更新 num_computed_tokens / last_sampled_tokens
     └─ 产出 ModelRunnerOutput → 调度器 update_from_output (文档 03)
```

---

## 1. 总体结论:ModelRunner 到底在干什么

在 V1 三层进程模型里(文档 01),Worker 进程只负责一件事:**模型前向与采样**。而 `GPUModelRunner` 就是这段逻辑的全部载体——它既不是"推理引擎",也不管 KV 分配(那是调度器的 `KVManager`),它只做三件事:

1. **翻译调度输出**:把 `SchedulerOutput`(每请求分了多少 token、新老请求、block 分配、LoRA、spec token 等,见文档 03 的 `output.py:193`)变成模型能吃的**固定形状张量**。
2. **执行前向**:按 `CUDAGraphMode` 决定这一批是 replay 一个捕获好的 CUDA Graph(FULL)、跑 piecewise 编译图(PIECEWISE)还是裸跑 eager(NONE)(`gpu/model_runner.py:1629-1666`)。
3. **采样与回写**:在末位 PP rank 上把 hidden states 换算成 logits,经 `Sampler`(或 spec decode 的 `RejectionSampler`)得到 token,再通过 `postprocess_sampled` 把 `num_computed_tokens`/`last_sampled_tokens` 更新回去,供调度器下一步决策。

> **历史对照**:V0 时代同一职责的类是 `vllm/worker/model_runner.py` 的 `ModelRunner`,靠 `prepare_model_input` + `forward` 两步;V1 把它重写为"**输入缓冲池(InputBuffers)+ 每步重填**"的模式,并把采样拆分到独立的 `sample_tokens` 调用——这让 GPU 前向与 CPU 调度真正可以流水(engine 用 `execute_model(non_block=True)` 后立刻 `sample_tokens`)。

V1 的 GPUModelRunner 在 `v1/worker/gpu_model_runner.py:500`,V2 的在 `v1/worker/gpu/model_runner.py:158`,两者共享同一个 `Worker`(`gpu_worker.py:142`)与同一个 `execute_model`/`sample_tokens` 调用契约。EngineCore 侧的 Executor 只认识这两个方法(`v1/executor/abstract.py:38` 的 `execute_model`/`sample_tokens` 签名),runner 具体是 V1 还是 V2 对引擎完全透明——**切换成本被压缩在 Worker 初始化一处**(`gpu_worker.py:424-438`)。

### 1.1 ModelRunner 与调度器/执行器的边界

| 谁 | 持有 | 做什么 |
|---|---|---|
| Scheduler(文档 03) | 请求状态机 + KV 分配 | 决定"谁算几个 token、给哪些块",产出 `SchedulerOutput` |
| ModelRunner | 模型 + 输入缓冲 + CUDA Graph + Sampler | 把 `SchedulerOutput` 变成 GPU 计算,吐出 sampled tokens |
| KV Cache Manager(文档 02) | 块池 / slot 映射 | runner 只读它的分配结果(`block_ids`),不参与分配 |

这条边界最直接的证据是 `SchedulerOutput` 里的字段被 runner **原样消费**:`scheduled_new_reqs`(新请求的完整 token/block 信息)进 `add_requests`,`scheduled_cached_reqs`(增量更新)进 `update_requests`,`scheduled_spec_decode_tokens` 进 `prepare_inputs` 的 draft 分支(`gpu/model_runner.py:1125`)。调度器从不碰 GPU 张量,runner 从不碰请求队列。

---

## 2. 执行链拆解(V2 runner 为主,标注 V1 差异)

### 2.1 请求状态先落地:add_requests / update_requests

`execute_model` 开头不是直接跑模型,而是先**把调度器的决策同步进 runner 的持久状态**(`gpu/model_runner.py:1417-1430`):

- `finish_requests`(`:931`)/`free_states`(`:942`):清掉 finished/preempted 请求的 slot。
- `add_requests`(`:955`):新请求写入 `req_states`(token 计数器、sampling params)、`block_tables.append_block_ids`、LoRA、多模态 `encoder_cache`。
- `update_requests`(`:1013`):老请求增量更新 `num_computed_tokens`,并为新分配的空 KV 块清零、执行 CoW 副本。
- 若 `total_num_scheduled_tokens == 0`,直接返回空输出(`:1425-1430`),不碰 GPU。

V1 对应的是 `_update_states`(`gpu_model_runner.py:1241`)+ 一个可延迟执行的 `deferred_state_corrections_fn`。

### 2.2 决定执行形态:dispatch_cg_and_sync_dp

`execute_model` 第二步是问 `CudagraphDispatcher` 该批用什么形态跑(`:1456-1466`):

```python
batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
    self.cudagraph_manager, num_reqs, num_toks, uniform_tok_count,
    self.dp_size, self.dp_rank, max_query_len=max_query_len,
    need_eager=is_profile or skip_compiled, num_active_loras=num_active_loras)
```

- `CudagraphDispatcher.dispatch()`(`v1/cudagraph_dispatcher.py:235`)按 `(num_tokens, num_reqs, uniform, num_active_loras)` 查捕获过的 key 表,命中 FULL 图→`CUDAGraphMode.FULL`;命中 PIECEWISE 图→`PIECEWISE`;否则 `NONE`(eager)。
- DP>1 时所有 rank 的 token 数必须对齐(`uniform_tok_count`),`dispatch_cg_and_sync_dp`(`gpu/dp_utils.py:98`)负责跨 rank 同步并补 padding。
- 一个重要的"逃逸阀":encoder-decoder 模型调度到编码器输入时强制 `skip_compiled=True`(`:1449-1454`),因为跨注意力缓存需要动态更新,不能进图。

V1 里同样的决策在 `_determine_batch_execution_and_padding`(`gpu_model_runner.py:4055`),额外还处理 cascade attention 与 micro-batching(UB/DBA)。

**为什么 dispatch 需要"num_reqs 也参与匹配"**:同一 token 数下,请求数不同意味着 `query_start_loc`/`seq_lens` 的排布不同;FULL 图按最坏情况统一成"每请求同样 token 数"(uniform decode),而 PIECEWISE 图允许 `num_reqs=None`(不特化请求数),因此 `dispatch` 对 PIECEWISE 会构造一个放宽的 key(`replace(batch_desc, num_reqs=None, uniform=False)`,`cudagraph_dispatcher.py:316`)做二次匹配——这也是"PIECEWISE 更省图数量、FULL 更快但更占内存"的根源。

### 2.3 prepare_inputs:填满"固定形状"的输入缓冲池

`prepare_inputs`(`gpu/model_runner.py:1102`)是 V2 最核心的"翻译器",产物是一个 `InputBatch`(`gpu/input_batch.py:42`)。它写进的所有张量都来自一个**预先分配好、永不释放**的 `InputBuffers`(`gpu/input_batch.py:17`):

| 缓冲(InputBuffers) | 形状 | 用途 |
|---|---|---|
| `input_ids` / `positions` / `is_padding` | `[max_num_tokens]` | 每个 token 的 id、位置、是否 graph padding |
| `query_start_loc` | `[max_num_reqs+1]` | 每个请求的 token 区间前缀和(FA 后端需要单调不减,`query_start_loc_np[num_reqs+1:] = num_tokens` 补齐,:1178) |
| `seq_lens` / `dcp_local_seq_lens` | `[max_num_reqs]` | 每请求长度(DCP 时按上下文切分) |

关键点:这些缓冲的**地址在 CUDA Graph 捕获时被写死在图里**,所以运行期永远不能 realloc,只能**原地覆写**——这就是 CUDA Graph 模式必须 shape 固定的底层原因(详见 §4.4)。

`prepare_inputs` 内部依次:
1. 处理 spec decode 的 draft token(没有 draft 时 `cu_num_logits = arange(num_reqs+1)`,有 draft 时按 `num_bonus_tokens + len(draft)` 重算,:1125-1152)。
2. 算 `query_start_loc`(:1170-1191)。
3. `prepare_prefill_inputs`(:1194)把 prefill 的下一段 token 从 `all_token_ids` 拷进 `input_ids`。
4. `prepare_pos_seq_lens`(:1206)按 `num_computed_tokens` 算每个 token 的绝对位置。
5. `combine_sampled_and_draft_tokens`(:1230)把上一轮的 `last_sampled_tokens` 和 draft tokens 拼进输入,并产出 `logits_indices`(采样时从 hidden states 里取哪些行)。
6. 组装 `InputBatch` 返回(:1263-1300)。

V1 的对等物是 `_prepare_inputs`(`gpu_model_runner.py:2014`),产物是 `logits_indices` + `SpecDecodeMetadata`,并直接在 `self.input_batch`(一个 V1 的持久 batch 对象)上填。

### 2.4 prepare_attn:AttentionMetadata 与 SamplingMetadata

- `prepare_attn`(`gpu/model_runner.py:1303`):从 `BlockTables.gather_block_tables`(:1310)与 `compute_slot_mappings`(:1316)拿到每层需要的 **block table(行=请求,列=KV 块)** 和 **slot mapping(token→KV slot)**;随后 `model_state.prepare_attn`(:1535)按各注意力后端把它们包装成 `AttentionMetadata`。
- `AttentionMetadata`(`v1/attention/backend.py:455`)与 `CommonAttentionMetadata`(:463)是**各注意力后端的公共契约**(FA2/FA3/FlashInfer/flashmla 各自实现子类),由 `AttentionMetadataBuilder`(:678)在每次前向构建。
- `SamplingMetadata`(`v1/sample/metadata.py:15`)在 V1 中由 `_prepare_inputs` 阶段的 sampling 逻辑构建并拷到 GPU(`gpu_model_runner.py:1248` 注释);**V2 不再有独立的 SamplingMetadata 对象**——采样所需的温度/种子等状态被装进 `Sampler` 内部的 `sampling_states`,请求级计数全部走 `req_states`。

### 2.5 前向与采样分离:execute_model → sample_tokens

V2 刻意把一步拆成两次调用(`gpu/model_runner.py:1408` 与 `:1707`):

- `execute_model`:返回前把本次前向的所有中间结果塞进 `ExecuteModelState`(NamedTuple,:1980),**非末位 PP rank 返回 `IntermediateTensors`** 用于流水线通信(:1700-1703)。
- `sample_tokens`:消费 `ExecuteModelState` → `sample()`(:1333,`compute_logits` + 语法 bitmask + `Sampler`/`RejectionSampler`)→ `postprocess_sampled`(:1367)更新计数器 → 组 `ModelRunnerOutput`,并把结果异步拷到 CPU 流(`AsyncOutput`,:1782)。末位 rank 还会把 sampled tokens 通过 `pp_handler.broadcast`(:1753)广播给非末位 rank。

这种分离让 EngineCore 可以 `execute_model(non_block=True)` 之后立即处理调度/输出,GPU 拷贝与 CPU 工作重叠。

#### 2.5.1 一个真实 decode 步的流水线(TP=1,PP=1)

```
schedule() 产出 SchedulerOutput
  ├─ num_scheduled_tokens = {reqA:1, reqB:1, ...}  (decode 每请求 1 token)
  └─ total_num_scheduled_tokens = N
execute_model:
  add_requests/update_requests → req_states 更新
  dispatch: num_tokens=N ≤ capture_size 中最近的值 M → FULL, batch_desc(M, uniform)
  prepare_inputs: 把 N 个 token 填进 input_ids[:N], query_start_loc 尾部填到 M
  prepare_attn: 生成 M 行的 block table / slot mapping
  run_fullgraph(batch_desc): graph.replay()  ← 唯一一次 GPU 驱动调用
sample_tokens:
  sample → 取 logits_indices 上的 N 行 → 每请求 1 个新 token
  postprocess_sampled → num_computed_tokens += 1
ModelRunnerOutput → scheduler.update_from_output (文档 03 §7)
```

这套流水里,**M−N 个 padding token 也在图里被算**,但它们的位置是重复/无效的,采样只取真实 token 的 logits——代价是少量浪费的 FLOPs,换来整步只付出一次图回放的开销。

### 2.6 多模态 / 编码器输入的特殊路径

支持多模态时,`execute_model` 在模型前向**之前**要准备 `inputs_embeds`:

- 仅首 PP rank 执行(`:1551`);`dummy_run` 时用 `model_state.dummy_inputs_embeds` 造同形状假 embedding(:1556-1558),保证编译/捕获出的 shape 一致。
- 真实请求时经 `ec_connector.maybe_get_output(...)` 上下文取 `get_mm_embeddings`(:1570-1583);纯编码器模型(embedding 模式)只执行编码、不跑语言模型(:1573-1579)。
- 若模型声明 `requires_raw_input_tokens`,`input_ids` 保留;否则置 None 只喂 `inputs_embeds`(:1584-1585)。
- 调度器侧对应的 `scheduled_encoder_inputs` 在文档 03 §4.5(`_try_schedule_encoder_inputs`,`scheduler.py:1531`),两处共享同一个"编码器预算"概念。

### 2.7 前向上下文(forward context)

`set_forward_context(...)`(:1644)把 `attn_metadata`、`batch_descriptor`、`cudagraph_runtime_mode`、`slot_mapping`、`is_padding` 等打进一个**线程局部上下文**,模型的层级代码(`AttentionImpl`、融合 kernel、`CUDAGraphWrapper.__call__`)从 `get_forward_context()` 读取而不是层层传参:

- 这是 CUDA Graph 派发的关键通道:`CUDAGraphWrapper.__call__`(`compilation/cuda_graph.py:240-242`)正是从这里取 `batch_descriptor` 与 `cudagraph_runtime_mode` 决定 capture 还是 replay。
- `skip_compiled`(encoder-decoder 逃逸阀,§2.2)也经由它传给模型,让 `@support_torch_compile` 装饰的层临时退回 eager(`compilation/decorators.py:86` 附近)。
- PIECEWISE 模式下同一个上下文里还会带 `BatchDescriptor`,供 `PiecewiseBackend.__call__` 按 `RangeEntry` 查找编译产物(`piecewise_backend.py:343-358`)。

---

## 3. V1 vs V2:谁在跑,怎么选

### 3.1 选择机制

`Worker.__init__` 里按 `use_v2_model_runner` 二选一(`gpu_worker.py:424-438`):

```python
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
else:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
    self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

`use_v2_model_runner`(`config/vllm.py:615`)的默认判定链:

1. 环境变量 `VLLM_USE_V2_MODEL_RUNNER`(`envs.py:292`,解析在 `envs.py:2029`):显式指定则直接采信。
2. PCP(prefix context parallel)>1、DSpark、DFlash 多 KV 组、diffusion 模型 → **强制 V2**(V1 不支持)。
3. `_is_default_v2_model_runner_model`(`config/vllm.py:674`):非 generate 任务、attention-free、非默认架构的 hybrid MoE → V1;默认架构列表来自 `default_v2_model_runner_architectures`(`config/vllm.py:84`)。
4. 没有 Triton → V1(打 warning);有 `_get_v2_model_runner_unsupported_features()` 列出的不支持特性 → 降级 V1。

### 3.2 核心差异

| 维度 | V1(`gpu_model_runner.py`) | V2(`gpu/model_runner.py`) |
|---|---|---|
| 类定义 | `:500`,混入 LoRA/KVConnector/ECConnector | `:158`,仅混入 LoRA |
| 输入载体 | 一个持久 `self.input_batch` + `SamplingMetadata`/`AttentionMetadata` | `InputBatch` dataclass + `InputBuffers` 缓冲池 |
| 请求状态 | 散落在各 numpy/torch 缓冲 | `RequestState`(`req_states`)+ `ModelState` 统一管理 |
| 采样 | 采样在 `execute_model` 流程内完成 | 拆成独立的 `sample_tokens`(`:1707`) |
| CUDA Graph | `_capture_cudagraphs`(`:7103`)/`_warmup_and_capture`(`:7055`),图挂在 CUDAGraphWrapper 上 | `ModelCudaGraphManager`(`cudagraph_utils.py:441`)统一管理 FULL/PIECEWISE 图 |
| PP 通信 | broadcast_pp_output(torchrun 特判,:558) | `PPHandler`(:292)走 side stream |
| `reload_weights`/`update_config` | 原生实现 | 临时 import V1 复用(`:462-478`) |

> 一句话:**V2 是 V1 的重写,把"每个张量一个临时对象"改成"预分配缓冲池 + 每步覆写",并让采样与 CUDA Graph 管理成为一等公民**。V2 尚不支持某些特性(由 `_get_v2_model_runner_unsupported_features` 判定)时会安全回退 V1。

### 3.3 V2 对 V1 的兼容壳

V2 目前还借用了 V1 的几处实现,读代码时会看到"V1 与 V2 混搭"的痕迹:

- `reload_weights` 与 `update_config` 直接 import V1 的 GPUModelRunner 复用其逻辑(`gpu/model_runner.py:462-478`),并在 `update_config` 末尾把改动同步回 `self.vllm_config`(:476-477)——因为 V2 读配置走 `vllm_config`,而 V1 的辅助函数改的是 `self.model_config` 属性。
- `cudagraph_utils.py` 顶部注释(`:1479-1481`)提到 MRV2 的 capture sizing 由它自己管,不需要 V1 的 `adjust_cudagraph_sizes_for_spec_decode`(`config/compilation.py:1482-1490` 只在 `use_v2_model_runner=False` 时启用)。

> 这意味着调试 V2 时若看到 `GPUModelRunnerV1.reload_weights`,不要惊讶——那是迁移期的显式复用,不是 bug。

---

## 4. CUDA Graph:为什么要,怎么捕获,为什么 shape 必须固定

### 4.1 为什么需要:启动开销

一次 LLM 前向会启动成百上千次 CUDA kernel;每次 kernel 启动约有 **2-10 微秒的 CPU 侧开销**(launch overhead,含 Python→C++→driver 调用链)。decode 阶段每 token 只有一次前向,这个开销可能占到延迟的 10-30%。CUDA Graph 的做法是:**把一串 kernel 录制成一张 DAG,回放时 CPU 只需一次 `graph.replay()` 调用**,GPU 侧按录好的依赖依次执行,启动开销被摊平到几乎为零。

vLLM 的落地形态是**一组固定 capture size 的图**,而不是一张万能图(见 §4.2/4.4)。

> 想关掉这一整套时,用 `--enforce-eager`:它把 `CompilationConfig.mode` 置 `NONE` 并跳过 `capture_model`(`gpu_worker.py:731` 的 `if not self.model_config.enforce_eager` 分支、`config/vllm.py:1909` 的捕获尺寸生成条件都看这个开关)。代价是牺牲吞吐,换来 0 捕获时间与最低显存占用,适合调试/显存极紧的场景。

### 4.2 capture_sizes:哪些 shape 要建图

默认值在 `VLLMConfig` 后置处理中生成(`config/vllm.py:1872-2031`):

```python
# max_graph_size 默认 = min(max_num_seqs * (1+num_spec_tokens) * 2, 512/1024)
# (config/vllm.py:1919-1926, Blackwell 数据中心为 1024, 否则 512)
cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max+1, 16))
# 最后再把 max_num_batched_tokens 补进去 (:1969-1973)
```

要点:
- **小批量加密(1/2/4)是为了低并发时也低延迟**;之后按 8/16 的步长稀疏覆盖,避免为每个批量各建一张图。
- `max_cudagraph_capture_size`(默认同 512/1024)被截断到 ≤ `max_num_batched_tokens`(:1928),最终写回 `compilation_config.cudagraph_capture_sizes`(:2026)。
- spec decode 时每张图的 token 数必须是 `1+num_speculative_tokens` 的整数倍,`adjust_cudagraph_sizes_for_spec_decode`(`config/compilation.py:1519`)负责向上取整。
- 用户可用 `--cudagraph-capture-sizes` 覆盖;若与 `--max-cudagraph-capture-size` 矛盾会直接报错(`config/vllm.py:1996-2002`)。

**为什么是"一组图"而不是"一张动态图"**:CUDA 的 dynamic shapes 捕获从 12.0 才支持且限制多(禁止某些控制流/分配),绝大多数部署仍是 CUDA 11/12 的静态捕获。vLLM 的工程取舍是——**用"数量有限、step 为 8/16 的离散尺寸"逼近连续批量分布**,让实际批量总能落在某个捕获点附近。这组 sizes 同时决定了两件事:捕获数量(启动时间)与 padding 浪费(运行延迟),所以它也是性能调优最常动的旋钮。

### 4.3 捕获流程:warmup → capture → replay

**入口** `capture_model`(`gpu/model_runner.py:848`)→ `ModelCudaGraphManager.capture`(`cudagraph_utils.py:307`):

1. 先 `gc.collect()` + `empty_cache()`,记录捕获前空闲显存,结束后相减得到图占用的真实内存(:868-911,V2;V1 在 `gpu_model_runner.py:7006-7028` 同样手法)。
2. 按 **PIECEWISE 先、FULL 后**的顺序捕获(:325 注释:PIECEWISE 激活更大,先捕获把大块内存留在 graph pool 里,小的 FULL 图可以复用)。
3. 对每个 descriptor:
   - `create_forward_fn(desc, warmup=True)` 准备输入,先 `forward_fn(CUDAGraphMode.NONE)` 跑一次 **warmup**(:334-337)——让注意力后端完成 lazy 初始化、让 Inductor 触发热编译,避免把这些开销录进图里。
   - FULL 模式再 `create_forward_fn(desc, warmup=False)` 拿"干净"的 forward,然后 `torch.cuda.CUDAGraph()` + `torch.cuda.graph(graph, pool)` 包一层执行(:357-366):所有 kernel 被录制。
   - PIECEWISE 模式的 forward_fn 内部其实是**在编译产物上按子图逐段 capture**(见 §5.2)。
4. 捕获期间 `set_graph_pool_id(self.pool)`(:361-364):让捕获时的所有临时分配都落进**专用 graph 内存池**,保证回放时地址不变。

**运行期回放** 见 `run_fullgraph`(`cudagraph_utils.py:412`):`self.graphs[desc].replay()`。回放前 `get_offloader().sync_prev_onload()` 确保上一轮 offload 拷贝结束(:418-424)。

**V1 的对应实现**:`capture_model`(`gpu_model_runner.py:6949`)→ `_capture_cudagraphs`(:7103)→ 对每个 desc 做 `num_warmups` 次 `_dummy_run` 预热(:7069-7080),再 `_dummy_run(is_graph_capturing=True)` 真正捕获(:7091-7101);这期间 `graph_capture()` 上下文(`distributed/parallel_state.py:1451`)会把 TP/PP 的通信原语也切到图捕获流上。

#### 4.3.1 warmup 到底在预热什么

捕获前那次 eager 运行不是"白跑",它的职责在 V2 注释里写得很清楚(`cudagraph_utils.py:316-319`):**FULL 与 breakable PIECEWISE 模式的 forward 工厂会被调用两次(warmup=True 与 warmup=False),因为注意力后端会在 warmup 期间变更或懒初始化元数据**。具体来说:

- 注意力后端(FA2/FA3/FlashInfer)首次见到某种 shape 会分配临时 buffer 或 build kernel cache,这些行为若是 capture 时首次发生,会产生**图捕获不允许的 API 调用**(如 `torch.cuda.malloc`)导致捕获失败。
- Triton/Inductor 的 lazy autotune 同理:把 autotune 的全部搜索放在 warmup 里,捕获到图里的就是**已定型的 kernel**。
- 二次调用 `create_forward_fn(desc, warmup=False)` 是为了拿到一份"无副作用、可安全录进图"的 forward。

#### 4.3.2 内存池(graph pool)如何工作

捕获时必须保证"录制时的内存地址 = 回放时的内存地址",否则回放会踩到别的张量。vLLM 的做法分两层:

1. **全局图池**:`current_platform.get_global_graph_pool()`(`platforms/interface.py:1150`)返回一个进程级 CUDA mempool(rocTX/CUDA 私有堆)。捕获时 `set_graph_pool_id`(`distributed/device_communicators/pynccl_allocator.py:63`)让 PyTorch 分配器把请求导向该池,并在 `torch.cuda.graph(graph, pool=self.graph_pool)` 里显式声明:图内所有分配都来自这个池(:313-316)。
2. **静态输入缓冲**:真正运行期还要保证"喂给图的输入张量地址"也是固定的——这由 `InputBuffers`(`gpu/input_batch.py:17`)在**捕获前就按最大形状分配好**、运行期只覆写不重分配来保证(§2.3)。V2 注释(`compilation/cuda_graph.py:161-167`)特别强调 wrapper 不做输入拷贝,正是因为这个职责被下沉到 `InputBuffers` 了。

因此 `capture_model` 结束时算出的 `cuda_graph_size`(= 捕获前后空闲显存差,:904)本质上就是**整个 graph 池被图 + 池内激活占据的体量**,它是后续 KV 预算扣减的依据(§6.1)。

### 4.4 为什么 shape 必须固定(以及 padding 怎么补)

CUDA Graph 录制的是**具体的 kernel 启动序列 + 具体的内存地址**。捕获时模型用 `dummy` 输入跑了一遍,所有中间激活的地址都固定了;回放时如果输入 shape 变了,要么 kernel 网格尺寸不对,要么 `cudagraph` 捕获的私有内存池被新分配打乱。所以:

- **捕获 = 用最大形状把输入缓冲池的地址钉死**(`prepare_inputs_to_capture`,`cudagraph_utils.py:606`,用 `InputBatch.make_dummy` + `get_dummy_block_tables/get_dummy_slot_mappings` 造同形状假输入)。
- **运行期 = 只能选"最接近且 ≥ 实际批量"的捕获尺寸**,多余的 slot 用 padding 填:query_start_loc 尾部填 `num_tokens`(:1178)、`is_padding` 标记尾部行(:1111-1116,VLLM_MOE_SKIP_PADDING 下 kernel 可跳过)。
- 映射关系由 `CudagraphDispatcher._compute_bs_to_padded_graph_size`(`cudagraph_dispatcher.py:72`)预算:批量 bs 被路由到"下一个捕获点",即 `padded_size[bs] = 下一个 capture_size`。
- 超过 `max_cudagraph_capture_size` 的批次**不建图**,退回 eager(`cudagraph_dispatcher.py:278`)。

> 这是 vLLM"批量取整"延迟与图加速之间的经典权衡:**小批量 padding 的浪费 vs 免去每次 kernel 启动开销**。interactivity 性能模式会把 `[1..32]` 每个批量都捕获(`config/vllm.py:1952-1953`),最大化低并发延迟。

### 4.5 LoRA 与图的组合爆炸

图是按 `(num_tokens, num_active_loras)` 两个维度 key 的,如果不加控制,LoRA 会带来"尺寸 × adapter 数"的组合爆炸。vLLM 用 `cudagraph_specialize_lora`(`config/compilation.py:660`,默认 True)限制:

- 捕获的 LoRA 变体只取**2 的幂子集**(`get_captured_lora_counts`,`cudagraph_dispatcher.py:123`),运行期 `num_active_loras` 用 `bisect_left` 找到"≥ 当前值的最小已捕获变体"(`cudagraph_dispatcher.py:291-293`)。
- 不特化时则只捕获"满 LoRA(max_loras+1)"一个变体,运行期一律按满配 dispatch(:294-300),图数量最少但 padding 最浪费。
- V2 侧 `gpu/lora_utils.py:20 get_lora_capture_cases` / `:76 LoraState` 与 `cudagraph_utils.py:147 _build_lora_dispatch_map` 维护同样的映射。

### 4.6 CUDAGraphWrapper:低层图载体

`compilation/cuda_graph.py` 提供**与运行时形状解耦的通用封装**(用于 V1 与 piecewise 场景):

- `CUDAGraphWrapper`(:145):构造时绑定一个 `runtime_mode`(FULL 或 PIECEWISE);`__call__`(:233)从 forward context 拿 `batch_descriptor` 与 `cudagraph_runtime_mode`,模式匹配则按 key 捕获(`torch.cuda.graph`+`pool`+`set_graph_pool_id`,:283-337)或 `entry.cudagraph.replay()`(:360)。
- `CUDAGraphEntry`(:128)保存 `{batch_descriptor → graph, output(weak ref)}`;`CUDAGraphOptions`(:139)控制调试日志/GC 禁用/输出弱引用。
- 注释(:161-167)明确:**wrapper 不负责持久缓冲与输入拷贝**——V2 在 runner 层用 `InputBuffers` 完成,保持 wrapper 与编译逻辑正交。
- 内存池:`self.graph_pool = current_platform.get_global_graph_pool()`(:200,平台实现见 `platforms/interface.py:1150`),capture 时 `set_graph_pool_id`(`distributed/device_communicators/pynccl_allocator.py:63`)把 PyTorch 分配导向该池;`torch.cuda.graph(pool=...)` 则保证池内地址跨 replay 不变。

---

## 5. torch.compile 与 vLLM 的编译体系

### 5.1 三个层级:CompilationMode / CUDAGraphMode / backend

`CompilationMode`(`config/compilation.py:37`,IntEnum):

| 值 | 含义 |
|---|---|
| `NONE`(0) | 完全不 torch.compile,纯 eager |
| `STOCK_TORCH_COMPILE`(1) | 标准 `torch.compile` 管线 |
| `DYNAMO_TRACE_ONCE`(2) | 单次 Dynamo trace,去 guard 避免重编译(要求无动态 shape 控制流) |
| `VLLM_COMPILE`(3) | **vLLM 自定义 Inductor 后端**(V1 默认),含缓存、piecewise 编译、shape 特化、自定义 fusion passes |

`CUDAGraphMode`(`config/compilation.py:53`)与之正交,管"**前向是否/如何进 CUDA Graph**",并在运行期作为派发依据:

- `NONE` / `PIECEWISE` / `FULL` 是三种**运行期形态**;`FULL_DECODE_ONLY=(FULL, NONE)`、`FULL_AND_PIECEWISE=(FULL, PIECEWISE)` 是"decode 用一种、prefill 用另一种"的复合模式,通过 `decode_mode()`/`mixed_mode()`(:65-69)拆分。

两者关系:PIECEWISE 运行期形态**要求** VLLM_COMPILE 编译(注意力在 splitting_ops 中或 breakable cudagraph 开启,`cudagraph_dispatcher.py:49-61` 有硬断言)。`CompilationConfig`(`config/compilation.py:398`)聚合 `mode`/`backend`/`splitting_ops`/`compile_sizes`/`cudagraph_*` 等全部字段。

### 5.2 VllmBackend:图拆分 + PostGrad passes + 缓存

`VllmBackend`(`compilation/backends.py:805`)是 VLLM_COMPILE 模式注册给 `torch.compile` 的 backend。它的核心工作:

1. **把整图拆成 piecewise 子图**(`split_graph`,:553):按 `splitting_ops`(注意力、RMSNorm+allreduce 融合点等)把计算图切成一段段,每段用 `PiecewiseBackend`(`compilation/piecewise_backend.py:86`)单独交给 Inductor 编译。拆出来的子图恰好是 PIECEWISE CUDA Graph 的**逐段捕获单位**。
2. **挂 PostGradPassManager**:把 vLLM 的融合 passes(如 `fuse_norm_quant`/`fuse_allreduce_rms`/`fuse_qk_norm_rope` 等,`PassConfig`,`config/compilation.py:107`)注入 Inductor 的 post-grad 阶段(:851-853,`configure_post_pass`:934)。
3. **编译缓存**:`__call__`(:1020)按 `env_hash + config_hash + code_hash(被 trace 的源文件内容) + compiler_hash` 生成 SHA-256 key(:1031-1066),缓存目录 `$VLLM_CACHE_ROOT/torch_compile_cache/<hash>/rank_{r}_{dp}/{prefix}`(:1067-1079),由 `CompilerManager`(`backends.py:124`)管理读写;可通过 `VLLM_DISABLE_COMPILE_CACHE` 关掉(:1082)。
4. **AOT 产物序列化**:`collect_standalone_compile_artifacts`(:872)把各 PiecewiseBackend 的编译产物(`VllmSerializableFunction`,`compilation/caching.py:174`)收成可落盘/跨 rank 复用的 mega artifact。

`gpu_worker.py` 里与编译相关的编排:VLLM_COMPILE 时把 `compile_sizes` 里不在 capture sizes 中的尺寸拿去 warmup(:697-718);warmup 后 `kernel_warmup(self)`(:728)调优 kernel;最后 `capture_model()`(:732)。Warmup 结束时对 Inductor 后端 `trigger_inductor_lazy_init`(:840-845)防运行期懒初始化毛刺,并激活 JIT monitor(:849-854)。

#### 5.2.1 PiecewiseBackend:一个子图、多个 shape range

`PiecewiseBackend`(`piecewise_backend.py:86`)管理**单独一个子图**在不同 shape 下的编译产物:

- 它的 `compile_all_ranges`(:245)按 `compile_ranges_endpoints`(`config/compilation.py:580`,如 `[256, 1024]` 表示 `(1..256]` 与 `(256..1024]` 两个区间)把 shape 空间切成若干 `Range`,**每个 range 对应一份 Inductor 编译**(同一 range 内 shape 共享一份代码,用动态 shape 满足)。
- `__call__`(:358)在运行期用 `_find_range_for_shape`(:343)按实际 token 数找到命中的 `RangeEntry` 执行——这就是"**general shape 编译一段、多种批量共用**"(`config/compilation.py:435-443` 注释)的落地,和 CUDA Graph 的"一 shape 一图"形成互补:Inductor 管 shape 泛化、CUDA Graph 管 shape 固定,两者叠加得到"每 shape 一份图、每 range 一份代码"。
- `to_bytes`(:209)把每份编译产物序列化成字节,配合 `VllmBackend.collect_standalone_compile_artifacts` 落进 mega artifact 缓存。

### 5.3 编译缓存与"编译一次,处处复用"

编译产物按 `aot_compile_hash_factors`(`compilation/caching.py:573`)提供的因素寻址,源文件内容参与哈希(`_compute_code_hash`,:608),保证**改一行 forward 代码就失效重编**;未改动时二次启动直接命中磁盘缓存,省去整机分钟级的 Inductor 编译。

### 5.4 三种运行形态的完整决策链(小结)

| 形态 | 谁编译 | 图载体 | 典型用途 | 代价 |
|---|---|---|---|---|
| `NONE`(eager) | 无 | 无 | profile、不支持编译的模型、`--enforce-eager` | 每次 kernel 全量启动开销 |
| `PIECEWISE` | VllmBackend 切图 + Inductor | CUDAGraphWrapper,每个 PiecewiseBackend 一段图 | prefill / 混合批量 | 段间切换有少量 Python 开销 |
| `FULL` | VllmBackend 整图 + Inductor | `ModelCudaGraphManager.graphs`,整张 `torch.cuda.CUDAGraph` | uniform decode(每请求同 token 数) | 每 shape 一张图,内存占用最大 |

默认配置为 `FULL_AND_PIECEWISE`(decode 走 FULL、prefill 走 PIECEWISE,`config/vllm.py:283/306`),它通过 `decode_mode()`/`mixed_mode()`(`config/compilation.py:65-69`)在两个运行期形态间切换;`FULL_DECODE_ONLY=(FULL, NONE)` 则让 prefill 完全 eager(省图内存,适合解码占比极高的场景)。

---

## 6. profile_run 与显存测定

### 6.1 流程

Worker 的 `determine_available_memory`(`gpu_worker.py:475`)在分配 KV cache 之前运行:

1. 若用户显式给了 `kv_cache_memory_bytes`,仍会跑一次 `profile_run()`(为了把模型按 `max_num_batched_tokens` 编译出来,见 :489-492),但跳过 memory_profiling 计算。
2. 否则在 `memory_profiling(...)` 上下文(`utils/mem_utils.py:234`)里调 `self.model_runner.profile_run()`(:515-519)。该上下文会 `reset_peak_memory_stats` 并测量 `allocated_bytes.all.peak` 与 `non_torch_memory`(:289-301),把显存分成"权重(weights_memory)+ 峰值激活 + 非 torch 组件"三块。
3. `profile_run()` 本体(`gpu/model_runner.py:790`):多模态先 profile 编码器缓存(:791-805);再 `_dummy_run(max_num_tokens, skip_attn=True, is_profile=True)`(:813)用**满批假输入**跑一次前向触到激活峰值;末位 PP rank 再 `_dummy_sampler_run`(:821)确保采样器最坏内存也被计入;最后 `synchronize()` + `gc.collect()`(:825-828)。
4. 回到 worker:若有 CUDA Graph 还会 `profile_cudagraph_memory()`(:532,`gpu/model_runner.py:843` 当前返回 0)预估图内存,并受 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`(`envs.py:308`)控制是否计入 KV 预算(:535-539)。
5. 产出 `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate`(:559-563),交给引擎决定 `num_gpu_blocks`。

### 6.2 为什么 profile 要和实际运行"同 shape"

因为 CUDA Graph 捕获(§4.3)与 Triton/Inductor 编译都是**按 shape 特化**的,profile 若不跑到 `max_num_batched_tokens`,就会低估峰值激活、高估可用 KV 内存,启动后 OOM。`profile_run` 里 `skip_attn=True`(:647-650 断言该参数仅 profile 可用)还能在首次 profile 时跳过注意力元数据构建,只量前向主体的内存。

> 提示:V1 的 `profile_run`(`gpu_model_runner.py:6558`)流程等价,但多模态 profile 更细致(`_get_mm_dummy_batch` + `embed_multimodal`,:6600-6615)。

### 6.3 启动期时序:把上面的流程串起来

`initialize_from_config`(`gpu_worker.py:665`)会按配置设定 KV 块数;整个 GPU 启动期由 `compile_or_warm_up_model`(`gpu_worker.py:694`)收尾,完整时序:

```
init_device / 建 NCCL 组          (gpu_worker.py:395 前后)
load_model: 权重载入 + 记录 model_memory_usage   (gpu_worker.py:450)
determine_available_memory:
  memory_profiling { profile_run() }  → peak_activation_memory
  profile_cudagraph_memory()          → cudagraph_memory_estimate(可被 env 关闭)
  available_kv_cache_memory_bytes     (gpu_worker.py:541-563)
  └─ 引擎据此定 num_gpu_blocks → initialize_from_config 分配 KV  (gpu_worker.py:665)
compile_or_warm_up_model:
  1. VLLM_COMPILE: compile_sizes 去重后逐个 _dummy_run(size)  (gpu_worker.py:697-723)
  2. kernel_warmup                                             (gpu_worker.py:728)
  3. capture_model() → 返回 cuda_graph_memory_bytes             (gpu_worker.py:730-732)
  4. V2: warmup_kernels(execute_model, sample_tokens)           (gpu_worker.py:808-810)
     V1: 末位 rank _dummy_run + _dummy_sampler_run 预热采样器     (gpu_worker.py:811-831)
  5. trigger_inductor_lazy_init + 激活 JIT monitor
  6. freeze_gc_heap / enable_gpu_sync_check / set_torch_threads_for_runtime  (gpu_worker.py:858-867)
```

注意第 4 步 V1 里"**采样器预热放在 capture_model 之后**"是刻意的(`gpu_worker.py:815-816` 注释):采样 buffer 要在图捕获之后再分配,避免被捕获前的 `empty_cache` 清掉。第 6 步的 `freeze_gc_heap` 等收尾意味着:**启动完成后运行时不再允许新的编译/分配毛刺**,任何运行期 JIT 都会被视为性能事故(由 JIT monitor 上报)。

---

## 7. 关键类 / 函数速查(文件:行号)

### ModelRunner(V2 与 V1)
- V2 `v1/worker/gpu/model_runner.py`:`GPUModelRunner`:158;`__init__`:159;`load_model`:350;`_dummy_run`:633;`profile_run`:790;`capture_model`:848;`finish_requests`:931;`add_requests`:955;`update_requests`:1013;`prepare_inputs`:1102;`prepare_attn`:1303;`sample`:1333;`postprocess_sampled`:1367;`execute_model`:1408;`sample_tokens`:1707;`ExecuteModelState`:1980
- V1 `v1/worker/gpu_model_runner.py`:`GPUModelRunner`:500;`__init__`:503;`_update_states`:1241;`_prepare_inputs`:2014;`_determine_batch_execution_and_padding`:4055;`execute_model`:4289;`load_model`:5414;`_dummy_run`:5940;`profile_run`:6558;`capture_model`:6949;`_warmup_and_capture`:7055;`_capture_cudagraphs`:7103
- 输入载体:`v1/worker/gpu/input_batch.py:17 InputBuffers`、`:42 InputBatch`;`gpu/block_table.py:17 BlockTables`、`gpu/cudagraph_utils.py:606 prepare_inputs_to_capture`

### Worker 侧编排(`v1/worker/gpu_worker.py`)
- `Worker`:142;`use_v2_model_runner`:185;runner 选择:424-438;`load_model`:450;`determine_available_memory`:475;`initialize_from_config`:665;`compile_or_warm_up_model`:694

### 配置(`config/`)
- `compilation.py`:`CompilationMode`:37;`CUDAGraphMode`:53;`PassConfig`:107;`CompilationConfig`:398;`adjust_cudagraph_sizes_for_spec_decode`:1519;`get_compile_ranges`:1566
- `vllm.py`:`default_v2_model_runner_architectures`:84;`use_v2_model_runner` property:615;`_is_default_v2_model_runner_model`:674;capture sizes 默认生成:1872-2031

### CUDA Graph 运行时
- `v1/cudagraph_dispatcher.py`:`CudagraphDispatcher`:15;`_compute_bs_to_padded_graph_size`:72;`dispatch`:235;`get_capture_descs`:326
- `v1/worker/gpu/cudagraph_utils.py`:`CudaGraphManager`:102;`capture`:307;`dispatch`:382;`run_fullgraph`:412;`ModelCudaGraphManager`:441;`prepare_inputs_to_capture`:606
- `compilation/cuda_graph.py`:`CUDAGraphEntry`:128;`CUDAGraphOptions`:139;`CUDAGraphWrapper`:145;`__call__`:233
- `distributed/parallel_state.py:1451 graph_capture`;`distributed/device_communicators/pynccl_allocator.py:63 set_graph_pool_id`;`platforms/interface.py:1150 get_global_graph_pool`

### torch.compile 体系(`compilation/`)
- `backends.py`:`CompilerManager`:124;`VllmBackend`:805;`__call__`:1020;`collect_standalone_compile_artifacts`:872;`split_graph`:553
- `piecewise_backend.py`:`PiecewiseBackend`:86;`caching.py`:`StandaloneCompiledArtifacts`:38;`VllmSerializableFunction`:174;`aot_compile_hash_factors`:573;`decorators.py:725 maybe_use_cudagraph_partition_wrapper`

### Metadata / 采样
- `v1/attention/backend.py`:`AttentionMetadata`:455;`CommonAttentionMetadata`:463;`AttentionMetadataBuilder`:678
- `v1/sample/metadata.py:15 SamplingMetadata`(V1 用);`utils/mem_utils.py:234 memory_profiling`

### 环境变量
- `VLLM_USE_V2_MODEL_RUNNER`(`envs.py:292`,解析 :2029):None→按架构自动;True/False 强制。
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`(`envs.py:308`,解析 :2093,默认 True):把 CUDA Graph 内存估算计入 KV 预算。
- `VLLM_DISABLE_COMPILE_CACHE`(见 `backends.py:1082`):关闭 torch.compile 磁盘缓存。
- `VLLM_MOE_SKIP_PADDING`(`gpu/model_runner.py:1111`):标记 graph padding 行让 kernel 跳过。

---

## 延伸阅读

- NVIDIA Developer Blog:《Getting Started with CUDA Graphs》 <https://developer.nvidia.com/blog/cuda-graphs/>
- NVIDIA Developer Blog:《Improving CUDA Graph Performance》(warmup 与内存池的最佳实践) <https://developer.nvidia.com/blog/improving-cuda-graph-performance/>
- PyTorch 官方文档:torch.compile 总览 <https://pytorch.org/docs/stable/torch.compiler.html>
- PyTorch 官方文档:torch.cuda.CUDAGraph 与 CUDA Graph 使用说明 <https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>
- PyTorch 官方文档:Dynamo 工作原理与 guard/编译缓存 <https://pytorch.org/docs/stable/torch.compiler_deepdive.html>
- vLLM 官方文档:性能调优(含 `--enforce-eager`、`--cudagraph-capture-sizes`、`--compile` 等开关) <https://docs.vllm.ai/en/latest/performance/performance_tuning.html>
- vLLM 官方文档:torch.compile 与编译配置 <https://docs.vllm.ai/en/latest/features/compilation/index.html>
- vLLM 官方文档:投机解码(与 capture sizes 的 `decode_query_len` 直接相关) <https://docs.vllm.ai/en/latest/features/spec_decode.html>
- 相关文档:本仓库 docs `01_system_architecture.md`(Worker 在进程拓扑中的位置)、`03_Scheduler.md`(SchedulerOutput 的来源与消费)、`02_*`(KV Cache Manager,`block_ids` 的来源)