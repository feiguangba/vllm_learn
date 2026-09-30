# 04 · vLLM V1 引擎前端:LLMEngine / AsyncLLM 与输入输出处理器

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `v1/engine/llm_engine.py:48` = 上游源码中的 `vllm/v1/engine/llm_engine.py` 第 48 行)。
> **一句话总结**:`LLMEngine`(同步)与 `AsyncLLM`(异步)是用户碰到的**唯一引擎门面**,它们自己不调度、不碰 GPU,只做三件事——用 `InputProcessor` 把输入渲染/分词成 `EngineCoreRequest`、用 `OutputProcessor` 把 `EngineCoreOutputs` 规整成用户可读的 `RequestOutput`,以及通过 `EngineCoreClient` 与后台的 EngineCore 进程互通;文档 01/03 讲的调度器、KV 管理、执行器全部在进程另一端。

---

## 一图看懂

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 前端进程(LLM 离线脚本 / vllm serve 的 uvicorn 进程)                           │
│  LLMEngine  (v1/engine/llm_engine.py:48)   同步封装(离线)                     │
│  AsyncLLM   (v1/engine/async_llm.py:72)    异步封装(在线), 两者同构          │
│                                                                             │
│  ① InputProcessor (v1/engine/input_processor.py:38)  process_inputs:255      │
│     renderer 渲染(chat template / 多模态) → 分词 → 校验 → EngineCoreRequest    │
│     (v1/engine/__init__.py:100)                                              │
│                                                                             │
│  ② OutputProcessor (v1/engine/output_processor.py:438)  process_outputs:598  │
│     按 request_id 找到 RequestState (output_processor.py:131)                │
│       ├─ IncrementalDetokenizer (detokenizer.py:31)    增量解码 + stop 词     │
│       ├─ LogprobsProcessor     (logprobs.py:29)        logprobs 组装          │
│       └─ make_request_output   (output_processor.py:279) → RequestOutput      │
│     RequestOutputCollector (output_processor.py:47)  每个请求一个队列         │
│                                                                             │
│  ③ engine_core = EngineCoreClient.make_client (core_client.py:90)            │
│                 make_async_mp_client   (core_client.py:116)                  │
│       └─ AsyncMPClient / SyncMPClient / InprocClient (文档 01)               │
└──────────────┬───────────────────────────────────────────────┬──────────────┘
               │ 请求通道:ADD / ABORT (ZMQ ROUTER↔DEALER, msgpack)              │
               │ 输出通道:PUSH→PULL, get_output()/get_output_async()            │
               ▼                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EngineCore 进程 (v1/engine/core.py) —— 文档 01 拓扑 / 03 调度器已详述          │
│   Scheduler.schedule() → execute_model/sample_tokens → update_from_output    │
│   产出 EngineCoreOutput(s) (v1/engine/__init__.py:189/242)                   │
└─────────────────────────────────────────────────────────────────────────────┘

请求生命周期(一镜到底):
  LLM.generate (llm.py:418) / AsyncLLM.generate (async_llm.py:550)
    ─► InputProcessor.process_inputs ─► OutputProcessor.add_request ─► engine_core.add_request
    ─► EngineCore 调度+采样 ─► EngineCoreOutputs ─► output_handler 拉取 ─► process_outputs
    ─► RequestState.make_request_output ─► RequestOutput(流式) ─► 客户端
    ─► 完成 → _finish_request (output_processor.py:726) → STREAM_FINISHED (outputs.py:216)
    ─► generate() 循环退出返回
```

---

## 1. 总体结论:前端进程里藏着一个"小引擎"

V1 三层进程模型(文档 01)里,**调度、KV、执行全部在 EngineCore 进程**,而用户代码直接打交道的 `LLMEngine` / `AsyncLLM` 反而是一个**轻量门面**。它内部的对象栈几乎同构(`llm_engine.py:51` 与 `async_llm.py:75`),三件套:

| 组件 | 职责 | 位置 |
|---|---|---|
| `InputProcessor` | 渲染 + 分词 + 校验 + 多模态 → `EngineCoreRequest` | `v1/engine/input_processor.py:38` |
| `OutputProcessor` | `EngineCoreOutputs` → `RequestOutput`,并持有每个请求的 `RequestState` | `v1/engine/output_processor.py:438` |
| `engine_core` | `EngineCoreClient` 句柄,跨进程通信 | `v1/engine/core_client.py:78`(`make_client`:90 / `make_async_mp_client`:116) |

为什么把"输入渲染"和"输出规格化"放在前端进程,而不是 EngineCore?

1. **CPU 密集、GPU 无关**:分词、detokenize、logprobs 组装、prompt 模板都是纯 Python/CPU 活,放前端可以**和 GPU 调度并行**(文档 01 §1)。
2. **按 request 记账天然属于前端**:`RequestState` 是按请求聚合状态的地方,它知道 prompt、detokenizer、logprobs 进度,而这些状态 EngineCore 一个都不需要——EngineCore 只认 token id 和采样参数。
3. **多进程消息最小化**:跨进程只传 `EngineCoreRequest`(进去)和 `EngineCoreOutputs`(出来),所有"给人看"的加工都在本地做,避免把大块文本塞进 ZMQ。

> **与 V0 的历史对照**:V0 时代 `LLMEngine` 里直接长着 `ModelRunner` 和 KV cache 管理器的引用,`step()` 里就是完整的一次"调度→执行→采样"循环。V1 里 `step()`(`llm_engine.py:298`)只是 `engine_core.get_output() + output_processor.process_outputs()` 的薄壳——真正的循环挪到了 EngineCore 的 busy loop(文档 01 §3)。

---

## 2. LLMEngine 与 AsyncLLM:同步 / 异步两个同构门面

### 2.1 `LLMEngine`(`llm_engine.py:48`)——离线同步封装

构造(`llm_engine.py:51`)时依次建立三件套:

- `renderer = renderer_from_config(vllm_config)`(:91,负责 chat template / 多模态渲染)。
- `self.input_processor = InputProcessor(vllm_config, renderer)`(:94)。
- `self.output_processor = OutputProcessor(renderer.tokenizer, stream_interval=...)`(:97-102)。
- `self.engine_core = EngineCoreClient.make_client(multiprocess_mode, asyncio_mode=False, ...)`(:105-111)——注意 `asyncio_mode=False`,得到同步客户端(`SyncMPClient` / `InprocClient`,文档 01 §2.4)。

三个入口:

- **`add_request`**(:218):分两条路。`n == 1` 时直接 `output_processor.add_request(request, prompt_text, None, 0)` + `engine_core.add_request(request)`(:274-279);`n > 1` 时走 `ParentRequest` 扇出(见 §7),把 `n` 个 child 各自 `add_request` 进 OutputProcessor 和 EngineCore(:281-296)。
- **`step`**(:298):从 `engine_core.get_output()` 拿到一批 `EngineCoreOutputs`(:306),交给 `output_processor.process_outputs(...)`(:313),然后 `engine_core.abort_requests(processed_outputs.reqs_to_abort)`(:322)把"前端检测到 stop 串但引擎还没停"的请求急停掉,最后返回 `processed_outputs.request_outputs`(:336)。**离线 `LLM.generate` 就是反复调这个 `step()`**。
- **`abort_request`**(:212):先让 `output_processor.abort_requests` 在本地清理 RequestState 并产出 `FINISHED_ABORTED` 的末帧,再把剩余的 internal id 传给 `engine_core.abort_requests`。

`LLMEngine` 还暴露了 `get_num_unfinished_requests`(:188)/`has_unfinished_requests`(:191)供离线循环判断退出,以及 sleep/wake_up/profile/LoRA/collective_rpc 等透传到 EngineCore 的 API(:363-438)。

### 2.2 `AsyncLLM`(`async_llm.py:72`)——在线异步封装

与 `LLMEngine` 几乎同构,差异在三点:

1. **engine_core 用异步客户端**:`EngineCoreClient.make_async_mp_client(...)`(`async_llm.py:149`),所有 `add_request_async` / `abort_requests_async` / `get_output_async` 都是可等待协程。
2. **原始 prompt 预处理不阻塞事件循环**:`add_request`(`async_llm.py:283`)对已渲染的 `EngineInput`(dict)走同步 `process_inputs`(:356),对**原始 prompt** 走 `process_inputs_async`(:372)——后者是 `make_async(process_inputs, executor=renderer._executor)`(`input_processor.py:80-82`)包出来的,跑到 renderer 的线程池里,避免分词卡住 uvicorn。
3. **输出不 return,而是推队列**:`add_request` 为每个请求建一个 `RequestOutputCollector`(`async_llm.py:400`),`generate()` 从这个 collector 里 `get`(见 §5.1)。

`generate`(`async_llm.py:550`)是 API server 的主入口:

```python
q = await self.add_request(request_id, prompt, sampling_params, ...)   # :586
finished = False
while not finished:
    out = q.get_nowait() or await q.get()        # :607 尽量不切任务直接排空
    assert isinstance(out, RequestOutput)
    finished = out.finished
    if out is not STREAM_FINISHED:
        yield out                                # :614 流式下发
```

出错处理(:616-663):客户端断开 / generator 被 GC(`asyncio.CancelledError`/`GeneratorExit`)→ `abort(q.request_id, internal=True)`;引擎死亡(`EngineDeadError`)→ 直接抛;其他异常 → `abort` 后包成 `EngineGenerateError`。`finally` 里 `q.close()`(:663)。

### 2.3 `output_handler` 后台任务(`async_llm.py:665`)

`AsyncLLM` 在事件循环里跑一个后台协程,持续从 EngineCore 拉输出并分发给各请求的 collector:

- 从 `engine_core.get_output_async()` 拿一批输出(:690)。
- **分片**:按 `envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`(默认 128,`envs.py:167`)切成小块(:700-703),逐片 `output_processor.process_outputs(...)`(:705),片与片之间 `await asyncio.sleep(0)`(:723)让出事件循环——这就是文档 01 §关键类里那个环境变量的真正用途。
- 处理 `processed_outputs.reqs_to_abort`(:726-729)。
- 结束后把 `scheduler_stats` 交给 `update_scheduler_stats`(:731)并记录日志(:736-742)。
- 任何异常 → `output_processor.propagate_error(e)`(:745),把所有 `generate()` 任务用同一个异常唤醒。

> **关键区分**:`LLMEngine.step()` 是"调用者主动拉取"(离线循环),`AsyncLLM` 是"后台任务推送"(在线服务)。两者共用同一个 `OutputProcessor`,只是消费方式不同。

---

## 3. InputProcessor:EngineInput → EngineCoreRequest(`input_processor.py:38`)

`EngineCoreRequest`(`v1/engine/__init__.py:100`)是跨进程请求载体——**只有 token id、采样参数、多模态特征指针**,没有原始文本。`InputProcessor` 负责把它从用户输入变出来。

### 3.1 `process_inputs` 主流程(`input_processor.py:255-400`)

```
校验 params (SamplingParams.verify / PoolingParams)     :270  _validate_params:91
校验 lora                                                :271  _validate_lora:159
data_parallel_rank 越界检查                              :277-281
│
├─ prompt 是已渲染 dict(EngineInput) → 直接用             :283-294
└─ prompt 是原始 prompt       → InputPreprocessor.preprocess :305  输入渲染
        │ (分词、多模态 token 展开、chat template 已在 renderer 阶段完成)
        ▼
split_enc_dec_input → encoder/decoder 输入                :312
_validate_model_inputs(长度 / 词表 / 多模态占位)           :313  _validate_prompt_len:402
│
├─ SamplingParams:clone + max_tokens 默认补满             :327-342
│     max_tokens=None → max_model_len - prompt_len        :331-335
│     并从 generation_config / tokenizer 补齐参数          :337-342
└─ PoolingParams:clone                                    :343-344
│
多模态:argsort_mm_positions 排序 → 组装 MultiModalFeatureSpec 列表 :347-382
│
EngineCoreRequest(全部字段打包)                           :384-400
```

要点:

- **`process_inputs_async`** = `make_async(self.process_inputs, executor=self.renderer._executor)`(:80-82)。这是 `AsyncLLM.add_request` 在原始 prompt 路径下的调用版本。
- **`assign_request_id`**(:235-253):把外部 `request_id` 存进 `external_req_id`,再把内部 id 改成 `"{external}-{8位随机}"` 保证唯一(受 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` 控制,:246)。之后 EngineCore 全程用内部 id,前端用 `external_req_id` 对外。
- **参数校验**在 `_validate_params`(:91):SamplingParams 要过 `params.verify(model_config, ...)`(:104),并检查模型是否支持 generation/pooling 任务、`return_sampling_mask` 对 temperature 的限制(:111-121)、`thinking_token_budget` 需要 reasoning 配置(:122-130)。
- **长度校验**在 `_validate_prompt_len`(:402):decoder prompt 不能为空、不能超过 `max_model_len`,且 `prompt_len == max_model_len` 时生成模型也要拒绝(没给输出 token 留位置,:438-447)。

### 3.2 谁调它?两个入口的差异

| 入口 | 调用方式 | 说明 |
|---|---|---|
| `LLMEngine.add_request`(`llm_engine.py:251`) | 同步 `process_inputs` | 离线,无事件循环可阻塞 |
| `AsyncLLM.add_request`(`async_llm.py:356/372`) | 已渲染 dict 走同步;原始 prompt 走 `process_inputs_async` | 在线,保持事件循环响应 |

---

## 4. OutputProcessor:EngineCoreOutputs → RequestOutput(`output_processor.py:438`)

### 4.1 数据结构与状态字典

OutputProcessor 持有三类按 id 索引的状态(:452-456):

- `request_states: dict[str, RequestState]`——**内部 request_id → 请求聚合状态**,`has_unfinished_requests()` 就是 `len(request_states) > 0`(:462)。
- `parent_requests: dict[str, ParentRequest]`——n>1 时的父请求聚合器。
- `external_req_ids: defaultdict[str, list[str]]`——外部 id → 内部 id 列表,abort 时按外部 id 批量命中(:563)。

### 4.2 `process_outputs` 主循环(`output_processor.py:598-724`)

这是**前端唯一遍历整批 `EngineCoreOutputs` 的函数**(docstring 里特意强调:全批次只应在这里循环一次,`output_processor.py:616-623`)。对每个 `EngineCoreOutput`:

```
1) 记录 stats (IterationStats)                          :636 _update_stats_from_output:805
2) 聚合 routed_experts / sampling_mask 分片             :646-649, 664-667
3) prefill 阶段:记 num_cached_tokens                   :651-659
4) 增量解码:detokenizer.update(new_token_ids)          :669-671
      返回 stop 串 → finish_reason=STOP + stop_reason   :672-674
5) logprobs:logprobs_processor.update_from_output       :678
6) make_request_output(...) → RequestOutput             :681-688
      └─ 有 queue → collector.put (AsyncLLM 流式)       :694
      └─ 无 queue → append 到返回列表 (LLMEngine)       :697
7) 结束处理:streaming 请求转下一段输入                    :700-706
   普通请求 → _finish_request + reqs_to_abort(如需要)    :708-712
```

**注意 stop 串的分工**:引擎侧 `check_stop`(文档 03 §7.2)只认 **eos 和 stop token id**;而**字符串形式的 stop** 由前端 detokenizer 发现(`detokenizer.py:672-674`)。一旦前端发现 stop 串,但 `engine_core_output.finished` 还是 False(:709),就把该 req 放进 `reqs_to_abort`,由 `LLMEngine.step()` / `output_handler` 回调 `engine_core.abort_requests` 把引擎侧的请求掐掉(`output_processor.py:709-712`)。

### 4.3 `abort_requests`(`output_processor.py:471-532`)

同时处理外部 id 和内部 id:`internal=True` 时按内部 id 逐个移除;`internal=False`(默认)时通过 `external_req_ids` 映射展开成所有内部 id。每个被移除的 RequestState 都会**产出一次 `finish_reason=ABORT` 的末帧**(:510-524)并 `queue.put`,让 `generate()` 循环能正常收尾返回;父请求则先递归 abort 所有 child 再移除自己(:525-531)。返回的 `request_ids_to_abort` 交给引擎侧执行真正的 abort。

---

## 5. RequestState:按请求聚合的"前端口袋"(`output_processor.py:131`)

### 5.1 创建:`from_new_request`(:214-277)

每个 `EngineCoreRequest` 进来时由 `add_request`(:534-563)创建:

- 若 `sampling_params.detokenize` 为 False,`tokenizer` 置 None(:227-228,不做解码)。
- 创建 `LogprobsProcessor`(:233)和 `IncrementalDetokenizer`(:237)。
- 记录 `output_kind`、`max_tokens_param`、`top_p`、`n`、`temperature`(:241-244)——这些会被 `do_tracing` 用于生成 OTel span 属性(:784-795)。
- `stream_input = request.resumable`(:276):流式输入请求会额外建 `input_chunk_queue`(:191-193)。

同一内部 id 再次 `add_request` 时,`output_processor.py:543-546` 判断 `req_state` 已存在 → 走 `_update_streaming_request_state`(:565):非 resumable 的"结束哨兵"请求直接 `_finish_request` 并 `put(STREAM_FINISHED)`(:573-577);resumable 的下一段输入则塞进 `input_chunk_queue`,等当前段完成后 `apply_streaming_update`(:592-596)再把 prompt 拼接起来。

### 5.2 消费:`make_request_output`(:279-343)

把"本步新 token + 结束原因"变成用户可读对象:

- **`FINAL_ONLY`**:未结束时直接 return None(:291-293),只有最后一步才产出——这是 OpenAI 非流式接口用的模式(`chat_completion/protocol.py:738`)。
- **`stream_interval > 1`**(默认 1,`config/scheduler.py:153`):只在 finished / 首 token / 累计达 interval 时才下发(:295-308);`DELTA` 模式下切片 `new_token_ids = output_token_ids[sent_tokens_offset:]` 只发增量(:310-316)。
- **pooling 请求**:直接返回 `PoolingRequestOutput`(:320-325)。
- **n>1**:交给 `parent_req.get_outputs(...)` 聚合(:329-335),最终用 `external_req_id` 对外(:335)。
- `_new_request_output`(:345-389)组装 `RequestOutput`;`DELTA` 时 `pop_prompt_logprobs`(:372)把 prefill 阶段累积的 prompt logprobs 一次性吐出。
- `_new_completion_output`(:391-432)组装单个 `CompletionOutput`:文本来自 `detokenizer.get_next_output_text(finished, delta)`(:403),token id 来自 `detokenizer.output_token_ids`(:405),logprobs 只保留增量部分(:409-410),结束时拼接 routed experts / sampling mask(:412-420)。

### 5.3 `RequestOutputCollector`(:47-108)——异步队列

每个请求一个 collector,生产者在 `process_outputs`/`output_handler`,消费者是 `generate()`:

- `put`(:64-78):`DELTA` 模式下游消费不及时时会 `self.output.add(output, aggregate=True)`(**合并增量**),避免积压;不同 request_index(n>1)的帧互不覆盖(:72-74)。
- `get`(:80)/`get_nowait`(:90):基于 `asyncio.Event` 的阻塞/非阻塞取,异常原样上抛(:86-87)。
- `STREAM_FINISHED`(`outputs.py:216`)作为"流结束哨兵"被 `_update_streaming_request_state` 放入(:577),`generate()` 里遇到它跳过 yield、循环退出(`async_llm.py:613`)。

### 5.4 收尾:`_finish_request`(:726-738)

从 `request_states`、`external_req_ids`、`parent_requests` 三处清理该请求;n>1 且最后一个 child 结束时父请求也从 `parent_requests` 移除(:736-738)。

---

## 6. Detokenizer:增量解码的原理(`detokenizer.py`)

解码是**逐步、增量**的:每个采样步只新产出一个 token,但 `tokenizers` 的 `decode()` 需要看完整序列才能做词条清理(如 SentencePiece 的 dummy prefix、字节回退),全量重算 O(n²)。V1 提供三条实现,由 `from_new_request`(:49-66)按 tokenizer 类型选择:

| 实现 | 触发条件 | 机制 |
|---|---|---|
| `IncrementalDetokenizer`(:31) | `tokenizer is None`(禁解码) | 空实现,只记账 `token_ids` |
| `FastIncrementalDetokenizer`(:168) | `tokenizers >= 0.22` 且 `TokenizersBackend` | HuggingFace `DecodeStream` 原生增量 API |
| `SlowIncrementalDetokenizer`(:251) | 其余情况 | `detokenize_incrementally`(`tokenizers/detokenizer_utils.py:176`),纯 Python 双指针 |

### 6.1 Fast 路径(`detokenizer.py:168-248`)

- 构造时用 `DecodeStream(ids=request.prompt_token_ids, ...)`(:184-187)做 **native prefill**——把 prompt token 一次性喂进流,之后只需 `stream.step(tokenizer, token_id)` 取增量文本。
- `decode_next`(:211)对特殊 token 抑制多余空格(`spaces_between_special_tokens`,:214-220)。
- `_protected_step`(:224-248)兜底两类异常:罕见 overflow(重置流、跳过);`Invalid prefix encountered`(UTF-8 非单调导致流状态损坏,`INVALID_PREFIX_ERR_MSG` :28)→ 重建空流后重试一步。

### 6.2 Slow 路径(`detokenizer.py:251-307` + `detokenizer_utils.py:176`)

核心是 `detokenize_incrementally`,维护 `(tokens, prefix_offset, read_offset)` 三个游标:

- `convert_prompt_ids_to_tokens`(:119)只转换 prompt 末尾一段(`INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET + 2` 个 token,:132-133),不用全量。
- 每次只对**最后一个新 token** 做 `convert_ids_to_tokens`(:221-223),拼进 `output_tokens`。
- **双 offset 的意义**(docstring :195-197):`decode()` 会根据前后文决定是否加空格,因此必须保留一段"已解码但可能被改写"的前缀文本(`prefix_offset`),解码后回调——保证前后两个 step 拼出来的文本与一次性 `decode(全量)` 完全一致。
- 首迭代(`is_first_iter`,`detokenizer_utils.py:211-216`)返回全部 tokens,后续只返回新增。

### 6.3 stop 串检测(`detokenizer.py:96-143` + `check_stop_strings`:310)

- `BaseIncrementalDetokenizer.update`(:96):若 `stop_terminated`(eos)且不要求包含 stop 串,先把最后一个 token 从解码序列里摘掉(:108-114)。
- 每个新 token `decode_next` 后追加到 `output_text`(:120),再对新增区段搜索 stop 串(:130-143)。
- `check_stop_strings`(:310):多个 stop 串竞争时选**最早完成**者(`best_end` 最小,:345-349),保证与"逐 token 追加"的结果一致(投机解码一步多 token 时尤为重要);命中后按 `include_in_output` 决定截断位置(:354-362)。
- `get_next_output_text`(:149-165):`DELTA` 模式用 `_last_output_text_offset` 只返回新增文本;未结束时把 `stop_buffer_length` 个字符留在缓冲里(避免 stop 串被提前吐给客户端,:154)。

---

## 7. 并行采样(n>1)与 logprobs

### 7.1 `ParentRequest`:n 个 child 的扇出与聚合(`parallel_sampling.py:13`)

`SamplingParams.n > 1` 时,一个请求被摊成 n 个独立采样流:

- **扇出**:`LLMEngine.add_request`(`llm_engine.py:281-296`)与 `AsyncLLM.add_request`(`async_llm.py:413-422`)各自构造 `ParentRequest`,对每个 `idx` 调 `get_child_info`(`parallel_sampling.py:83`):
  - child 内部 id = `f"{idx}_{parent_id}"`(:92);
  - child 采样参数:clone 父参数后置 `n=1`(:73-74);若父 `seed` 非 None,每个 child 拿到 `seed + index`(:79-80)——保证多采样可复现。
- **入队**:每个 child 独立 `output_processor.add_request` + `engine_core.add_request`,即 EngineCore 眼里它们就是 n 个普通请求。
- **聚合**:`RequestState.make_request_output`(`output_processor.py:329-335`)把每个 child 的 `CompletionOutput` 交给 `parent_req.get_outputs`(:100):
  - 流式(DELTA/CUMULATIVE):child 完成一个就即时返回该 child 的输出(:115-119);
  - `FINAL_ONLY`:n 个 child 全完成后一次性返回聚合列表(:121-123)。
  - `finished = not self.child_requests`(:125)表示最后一个 child 完成。
- **统计**:`observe_finished_request`(:134-149)在所有 child 结束后向 `IterationStats` 记 `max_num_generation_tokens` 与 `n_params`,供指标统计(如 TTFT/吞吐)使用。

### 7.2 `LogprobsProcessor`(`logprobs.py:29`)

EngineCore 回传的 logprobs 是**平铺张量**(`LogprobsTensors` / `LogprobsLists`),前端负责解包成 `Logprob` 容器:

- `from_new_request`(:43-67):按 `num_logprobs` / `prompt_logprobs` 开关建好样本/prompt 两套容器。
- `_update_sample_logprobs`(:69-119):每个输出位置 detokenize 候选 token(`convert_ids_list_to_tokens`,`detokenizer_utils.py:143`),并把采样 token 的 logprob 累进 `cumulative_logprob`(:108-109)。
- `_update_prompt_logprobs`(:121-187):prefill 阶段的 prompt logprobs,按 `(pos, rank)` 展开放进 `PromptLogprobs`。
- **UTF-8 字节回退修正**(:249-346):byte-fallback 分词把多字节字符拆进多个 token,单个 token 解码出 `�`。`_correct_decoded_token`(:249)用最近 4 个上下文 token 重新 decode,把残缺字节归位到"完成它的那个 token"上;`_verify_tokens`(:312)对每个候选做一遍这个修正。
- `pop_prompt_logprobs`(:189-206):`DELTA` 模式下 prefill 结束时一次性返回全部 prompt logprobs 并清空,保证增量语义正确。

---

## 8. 两个入口:离线 `LLM` 与在线 OpenAI Server

### 8.1 离线:`LLM`(`entrypoints/llm.py:67`)

构造(:179)把用户参数打包成 `EngineArgs`,再 `LLMEngine.from_engine_args(...)`(:343)得到同步引擎。`generate`(:418):

```
LLM.generate(prompts, sampling_params)   :418
  └─ sampling_params=None → get_default_sampling_params()  :469-470
  └─ _run_completion(...)                :472  (offline_utils.py:326)
       ├─ 渲染/分词每个 prompt,LLMEngine.add_request 逐个入队
       ├─ 循环 LLMEngine.step() 拉 RequestOutput
       └─ 全部 finished → 按输入顺序返回 list[RequestOutput]
```

批处理要点:单个 `generate` 调用内所有 prompt 共用一个 EngineCore,内部自动连续批处理;`enqueue()` + `wait_for_completion()`(:483/543)是显式拆分版本。离线 `LLM.generate` 会在每步调 `step()`,即"调度-执行"的节奏完全由前端循环驱动。

### 8.2 在线:`api_server.py` 启动链

```
vllm serve → run_server (api_server.py:726) → run_server_worker (:742)
  └─ build_async_engine_client (:105)
       └─ AsyncEngineArgs.from_cli_args (:122)
       └─ build_async_engine_client_from_engine_args (:136)
            └─ AsyncLLM.from_vllm_config (:163)   ← 真正实例化 AsyncLLM
  └─ build_and_serve (:633) → build_app (:184, 挂路由/中间件)
  └─ init_app_state (:330, 挂 serving 对象到 app.state)
  └─ serve_http (:659, uvicorn 启动)
```

每个 HTTP 请求最终落到 generate/chat 等 endpoint handler,内部调用 `engine_client.generate(...)`(`async_llm.py:550`)→ `add_request` → `generate()` 协程从 `RequestOutputCollector` 拉流式帧 → 转成 OpenAI SSE 或 JSON。`RequestOutputKind` 的选择(`sampling_params.py:198`,`CUMULATIVE=0` / `DELTA=1` / `FINAL_ONLY=2`)决定了接口层是流式增量、累计还是最终帧。

---

## 9. 请求生命周期全景(从 add_request 到 RequestOutput)

```
【前端】                                                          【EngineCore 进程】
LLM/AsyncLLM.generate
  │
  ├─ InputProcessor.process_inputs                    ──渲染/分词/校验──
  │     → EngineCoreRequest (v1/engine/__init__.py:100)
  ├─ assign_request_id → 内部 id
  ├─ OutputProcessor.add_request                      建 RequestState + (AsyncLLM: collector)
  │     → request_states[req_id] = RequestState       (output_processor.py:558)
  ├─ engine_core.add_request_async                    ──ZMQ ADD──►
  │                                                              Scheduler.add_request → WAITING
  │                                                              schedule() → RUNNING
  │                                                              execute_model + sample_tokens (每步)
  │                                                              update_from_output → EngineCoreOutputs
  │                                                              (v1/engine/__init__.py:242)
  │                    ◄─────────ZMQ PUSH─────────────
  ├─ output_handler 拉取 → 按 chunk 分片 (async_llm.py:701)
  │     └─ OutputProcessor.process_outputs
  │           ├─ detokenizer.update → 文本 + stop 串检测 (detokenizer.py:96)
  │           ├─ logprobs_processor.update_from_output
  │           ├─ RequestState.make_request_output → RequestOutput
  │           └─ collector.put / request_outputs.append
  ├─ generate() 循环 get → yield RequestOutput (流式, 多次)
  │
  └─ 结束:
        finish_reason 到达 (STOP/LENGTH/ABORT/ERROR)
        ├─ 普通: _finish_request 清理三张表 (output_processor.py:726)
        ├─ n>1:  parent_req.get_outputs 聚合到最后一个 child
        └─ 引擎未停但前端发现 stop 串 → reqs_to_abort → engine_core.abort_requests
        → STREAM_FINISHED (outputs.py:216) → generate() 返回
```

---

## 10. 关键类 / 函数速查(文件:行号)

### `v1/engine/llm_engine.py`(同步门面)
- `LLMEngine`:48;`__init__`:51;`from_vllm_config`:144;`from_engine_args`:161
- `add_request`:218;`step`:298;`abort_request`:212;`has_unfinished_requests`:191;`get_num_unfinished_requests`:188
- `collective_rpc`:421;`apply_model`:437;`sleep`:363;`wake_up`:371;`start_profile`:338

### `v1/engine/async_llm.py`(异步门面)
- `AsyncLLM`:72;`__init__`:75;`from_vllm_config`:206;`from_engine_args`:235
- `add_request`:283;`_add_request`:424;`_add_streaming_input_request`:441;`generate`:550;`encode`:843
- `_run_output_handler`:665(内嵌 `output_handler`:686);`abort`:749;`shutdown`:262;`check_health`:940
- `pause_generation`:790;`resume_generation`:835;`scale_elastic_ep`:1051

### `v1/engine/input_processor.py`
- `InputProcessor`:38;`__init__`:39;`process_inputs`:255;`process_inputs_async`:80;`assign_request_id`:235
- `_validate_params`:91;`_validate_lora`:159;`_validate_prompt_len`:402;`_validate_model_input`:449;`_validate_model_inputs`:513
- `inject_into_mm_cache`:196;`get_tokenizer`:88

### `v1/engine/output_processor.py`
- `RequestOutputCollector`:47(`put`:64 / `get`:80 / `get_nowait`:90 / `close`:100)
- `OutputProcessorOutput`:111;`StreamingUpdate`:117
- `RequestState`:131(`apply_streaming_update`:195 / `from_new_request`:214 / `make_request_output`:279 / `_new_request_output`:345 / `_new_completion_output`:391 / `_new_pooling_output`:434)
- `OutputProcessor`:438(`abort_requests`:471 / `add_request`:534 / `_update_streaming_request_state`:565 / `process_outputs`:598 / `_finish_request`:726 / `propagate_error`:464 / `do_tracing`:743 / `update_scheduler_stats`:740)

### `v1/engine/detokenizer.py`
- `IncrementalDetokenizer`:31(`from_new_request`:49);`BaseIncrementalDetokenizer`:69(`update`:96 / `get_next_output_text`:149)
- `FastIncrementalDetokenizer`:168(DecodeStream native prefill:184);`SlowIncrementalDetokenizer`:251(`decode_next`:292)
- `check_stop_strings`:310;`USE_FAST_DETOKENIZER`:25

### `v1/engine/parallel_sampling.py` / `logprobs.py`
- `ParentRequest`:13(`_get_child_sampling_params`:52 / `get_child_info`:83 / `get_outputs`:100 / `observe_finished_request`:134)
- `LogprobsProcessor`:29(`from_new_request`:43 / `_update_sample_logprobs`:69 / `_update_prompt_logprobs`:121 / `pop_prompt_logprobs`:189 / `_correct_decoded_token`:249 / `_verify_tokens`:312 / `update_from_output`:348)

### 入口层
- `entrypoints/llm.py:67 LLM`;`__init__`:179;`generate`:418;`enqueue`:483;`wait_for_completion`:543;`chat`:612
- `entrypoints/offline_utils.py`:`_add_completion_requests`:290;`_run_completion`:326;`_run_chat`:351;`_run_engine`:573
- `entrypoints/openai/api_server.py`:`build_async_engine_client`:105;`build_async_engine_client_from_engine_args`:136;`build_app`:184;`init_app_state`:330;`run_server_worker`:742

### 跨进程载体与周边
- `v1/engine/__init__.py`:`FinishReason`:43;`EngineCoreRequest`:100;`EngineCoreOutput`:189;`EngineCoreOutputs`:242;`EngineCoreRequestType`:273
- `v1/engine/core_client.py`:`EngineCoreClient`:78;`make_client`:90;`make_async_mp_client`:116
- `vllm/outputs.py`:`RequestOutput.add`:167(增量合并);`STREAM_FINISHED`:216
- `vllm/sampling_params.py`:`RequestOutputKind`:198(CUMULATIVE=0 / DELTA=1 / FINAL_ONLY=2);`output_kind`:317
- `vllm/config/scheduler.py:153` `stream_interval`(默认 1)

### 环境变量
- `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`(`envs.py:167`,默认 128):`output_handler` 单轮分片大小(`async_llm.py:684/701`)。
- `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`(`input_processor.py:246`):关闭内部 id 随机后缀,仅调试用。

---

## 延伸阅读

- vLLM 官方文档:EngineCore 与调度循环 <https://docs.vllm.ai/en/latest/design/arch_overview.html>
- vLLM 官方文档:离线推理 LLM 类 <https://docs.vllm.ai/en/latest/usage/offline_inference.html>
- vLLM 官方文档:OpenAI 兼容服务部署 <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- vLLM 官方文档:结构化输出 / 输出流式化配置 <https://docs.vllm.ai/en/latest/features/output_streaming.html>
- vLLM Blog:《Announcing vLLM V1: A Major Architectural Upgrade》(V1 前端/EngineCore 拆分动机) <https://blog.vllm.ai/2025/01/27/v1-alpha-release.html>
- 相关文档:本仓库 repowiki `01_system_architecture.md`(三层进程拓扑,LLMEngine/AsyncLLM 在顶层的位置)、`03_Scheduler.md`(EngineCore 内调度循环,`EngineCoreOutputs` 的产出端)