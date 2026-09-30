# 01 · vLLM V1 系统架构:多进程拓扑与事件循环

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `v1/engine/core.py:1375` = 上游源码中的 `vllm/v1/engine/core.py` 第 1375 行)。
> **一句话总结**:V1 把引擎拆成"**前端进程 → EngineCore 进程 → Worker 进程组**"三层,前端负责输入渲染与输出规格化,EngineCore 独占调度 + KV 管理 + 执行编排,Worker 只负责模型前向与采样;三层之间用 **ZMQ(msgpack)+ 共享内存广播队列 + 零拷贝张量 IPC** 连接。

---

## 一图看懂

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 进程①  前端进程  (uvicorn / vllm serve, 或离线 LLM 脚本)                        │
│  entrypoints/openai/api_server.py:105/136  ·  entrypoints/llm.py:67            │
│                                                                               │
│  AsyncLLM  (v1/engine/async_llm.py:72)  或  LLMEngine  (v1/engine/llm_engine.py:48)│
│   ├─ Renderer / InputProcessor  (v1/engine/input_processor.py:38)             │
│   │     文本+多模态渲染、分词 → EngineCoreRequest                              │
│   ├─ OutputProcessor  (v1/engine/output_processor.py:438)                      │
│   │     EngineCoreOutputs → RequestOutput(流式/规格化/detokenize/logprobs)     │
│   └─ engine_core = EngineCoreClient.make_client(...)  (v1/engine/core_client.py:90)│
│        └─ AsyncMPClient(977) / SyncMPClient(805) / InprocClient(306)          │
└──────────────┬───────────────────────────────────────────────┬────────────────┘
               │ ZMQ 消息流(请求/输出两条单向流)                 │ 握手 HELLO→INIT→READY
               │  请求:ROUTER(bind)──►DEALER(connect)           │  (v1/engine/core.py:1233)
               │  输出:PUSH(connect)──►PULL(bind)               │
               │  序列化:msgspec.msgpack, 零拷贝 encode_into     │
               ▼                                               ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 进程②  EngineCore 进程(每 DP rank 一个, 名称 "EngineCore[_DP{rank}]")          │
│  v1/engine/core.py:1271 run_engine_core()   ← multiprocessing 进程入口          │
│  EngineCoreProc (v1/engine/core.py:1007)                                       │
│  主线程 run_busy_loop (core.py:1375):                                          │
│     while _handle_shutdown():                                                  │
│        _process_input_queue()      ← 收 ADD/ABORT/UTILITY, 进 scheduler         │
│        _maybe_publish_request_counts()                                          │
│        _process_engine_step()      ← 调度→执行→产出 (核心一步)                  │
│  线程: input 线程(core.py:1651) / output 线程(core.py:1754) / 存活监控          │
│  持有: Scheduler + Executor + KV Cache + StructuredOutputManager               │
└──────────────┬────────────────────────────────────────────────────────────────┘
               │ Executor 抽象: execute_model / sample_tokens / collective_rpc
               ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 进程③  Worker 进程组(每 TP×PP rank 一个, 由 MultiprocExecutor 拉起)            │
│  MultiprocExecutor (v1/executor/multiproc_executor.py)  +  WorkerProc           │
│  通信: shm_broadcast.MessageQueue(共享内存广播) + multiprocessing.Pipe          │
│  Worker (v1/worker/gpu_worker.py:142) → GPUModelRunner (v1/worker/gpu/model_runner.py:158)│
│  职责: 模型前向(forward)、采样(sample_tokens)、KV cache 读写                    │
└───────────────────────────────────────────────────────────────────────────────┘

   DP>1 追加: DPCoordinator 进程 (v1/engine/coordinator.py:146, 负载均衡协调)
              MoE 用 DPEngineCoreProc (core.py:1963, wave 同步 + all-reduce)
```

---

## 1. 总体结论:三层进程模型

vLLM V1(2024 年起成为默认引擎)最重要的架构决定是:**把"引擎"从一个庞然大物拆成三个边界清晰的进程层**,各自可以用独立的 GIL、独立的内存、独立的错误隔离:

| 层 | 进程/线程 | 代码位置 | 是否碰 GPU | 是否碰调度 |
|---|---|---|---|---|
| 前端(Frontend) | API Server / `LLMEngine` / `AsyncLLM` | `entrypoints/`、`v1/engine/llm_engine.py:48`、`v1/engine/async_llm.py:72` | 否 | 否 |
| 引擎内核(EngineCore) | `EngineCoreProc` 后台进程 | `v1/engine/core.py:1007` | 间接(通过 Executor) | **是** |
| 执行器(Worker) | `WorkerProc` × (TP×PP) | `v1/executor/multiproc_executor.py`、`v1/worker/gpu_worker.py:142` | **是** | 否 |

这个设计带来的好处:

1. **CPU 密集的输入渲染 / 输出规格化**(分词、detokenize、logprobs 组装、prompt 模板)跑在前端进程,不会阻塞 GPU 调度。
2. **调度循环**(`Scheduler.schedule()`)独占一个进程,可以和 GPU 前向(`execute_model`)真正并行/流水。
3. **Worker 崩溃**可以被 Executor 捕获并通过 `EXECUTOR_FAILED` 通知引擎,而不是带着整个服务一起挂掉(`v1/engine/__init__.py:273` 中 `EngineCoreRequestType.EXECUTOR_FAILED`)。

> **历史对照**:V0 时代的 `AsyncLLMEngine` 是一个单进程的 asyncio 事件循环 + `ModelExecutor` 抽象;V1 直接把它拆成了真·多进程。这也解释了为什么你会在老资料里看到 `MultiProcIPCDriver` —— 它属于 V0,在 V1 已被 **ZMQ + shm_broadcast + torch.multiprocessing.Queue** 三条通道取代。

---

## 2. 进程拓扑与通信方式(逐层拆解)

### 2.1 前端进程

前端是用户代码直接交互的地方。离线推理走 `LLM`(`entrypoints/llm.py:67`,构造 :179,`generate`:418),在线推理走 OpenAI 兼容服务(`entrypoints/openai/api_server.py:105 build_async_engine_client`)。

前端进程内部的对象栈(`v1/engine/llm_engine.py:48` 与 `v1/engine/async_llm.py:72` 几乎同构,前者同步、后者异步):

- `InputProcessor`(`v1/engine/input_processor.py:38`):把 prompt 文本 + 多模态输入渲染成分词后的 `EngineCoreRequest`。
- `OutputProcessor`(`v1/engine/output_processor.py:438`):把引擎吐回的 `EngineCoreOutputs` 变成用户可读的 `RequestOutput`(见文档 04)。
- `engine_core`:一个 `EngineCoreClient` 句柄,负责跨进程通信。

前端与 EngineCore 之间是**两个单向 ZMQ 通道**(而非一对请求-应答):

- **请求通道**:前端 `ROUTER`(bind)←→ 引擎 `DEALER`(connect),`v1/engine/core_client.py:554/589`。
- **输出通道**:引擎 `PUSH`(connect)←→ 前端 `PULL`(bind),`v1/engine/core_client.py:596`。
- **握手通道**:引擎启动时先做一次 `startup_handshake`(`v1/engine/core.py:1233`),HELLO → 收 `EngineHandshakeMetadata` → 回 READY + `EngineCoreReadyResponse`(含 `max_model_len`/`num_gpu_blocks`/`block_size` 等,`v1/engine/core.py:1616`)。

线协议用 **msgspec.msgpack** 多帧编码:`[1 字节类型帧][数据帧...]`。`EngineCoreRequestType`(`v1/engine/__init__.py:273`)的取值直接以 `b"\x00"~b"\x05"` 表示 `ADD`/`ABORT`/`START_DP_WAVE`/`UTILITY`/`EXECUTOR_FAILED`/`WAKEUP`。

### 2.2 EngineCore 进程

`EngineCoreProc`(`v1/engine/core.py:1007`)是 `EngineCore`(`core.py:104`)的 IPC 包装。进程入口 `run_engine_core`(`core.py:1271`)由 `CoreEngineProcManager`(`v1/engine/utils.py:120`)通过 `multiprocessing.Process` 启动,`launch_core_engines`(`utils.py:1070`)负责拉起 DP 个引擎进程。

`EngineCore` 本体是一个**没有任何 IPC 的纯计算内核**,持有:

- `Scheduler`(`v1/core/sched/scheduler.py:69`,文档 03)
- `model_executor`(Executor 抽象,`v1/executor/abstract.py:38`)
- KV Cache 管理器(文档 02)
- `StructuredOutputManager`(结构化输出/语法约束)

`EngineCoreProc` 在其上叠加了:

- 两个后台线程:输入线程 `process_input_sockets`(`core.py:1651`,ZMQ DEALER 收包 → 解码 → `input_queue`)、输出线程 `process_output_sockets`(`core.py:1754`,`output_queue` → 编码 → ZMQ PUSH)。
- `run_busy_loop` 主循环(见 §3)。
- 内部 `queue.Queue`:`input_queue`/`output_queue`/`aborts_queue`(`core.py:1026/1027/239`)。

### 2.3 Worker 进程组

`EngineCore` 里的 Executor 是 `MultiprocExecutor`(默认,`v1/executor/multiproc_executor.py`)。它把每个 **TP×PP rank** 包成一个 `WorkerProc` 后台进程,进程内跑 `WorkerWrapperBase`(`v1/worker/worker_base.py:191`)→ `Worker`(`v1/worker/gpu_worker.py:142`)→ `GPUModelRunner`(`v1/worker/gpu/model_runner.py:158`)。

EngineCore → Worker 的通信是**集体 RPC(collective_rpc)**:

- 请求广播:共享内存广播队列 `shm_broadcast.MessageQueue`(`v1/executor/multiproc_executor.py:31` 导入;`rpc_broadcast_mq` 在 :813)。
- 结果回收:每个 worker 一条 `multiprocessing.Pipe`(`worker_response_mq`,:820),由 `FutureWrapper._wait_for_response` 等待。
- 父进程死亡检测:`monitor_death_pipe`(`multiproc_executor.py:824`)监听父进程 Pipe EOF,父进程一挂,worker 自动退出。

`execute_model`/`sample_tokens` 都走 `collective_rpc`(`multiproc_executor.py:372`):`rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))` → 各 worker 各自执行 → 结果回 `response_mqs`。

### 2.4 通信机制总表

| 链路 | 机制 | 关键类/函数 |
|---|---|---|
| 前端 ↔ EngineCore(请求) | ZMQ `ROUTER(bind)──DEALER(connect)` + msgpack | `core_client.py:554/589`、`core.py:1669` |
| 前端 ↔ EngineCore(输出) | ZMQ `PUSH──PULL`,零拷贝 `encode_into` | `core_client.py:596`、`core.py:1813` |
| 握手 | ZMQ DEALER 明文 msgpack,HELLO→INIT→READY | `core.py:1233` |
| EngineCore 内部线程间 | `queue.Queue` | `core.py:1026/1027/239` |
| EngineCore ↔ Worker(执行) | `shm_broadcast.MessageQueue` 共享内存广播 | `multiproc_executor.py:813` |
| Worker → EngineCore(结果) | `multiprocessing.Pipe` | `multiproc_executor.py:820` |
| 多模态张量 | 零拷贝共享内存 `TensorIpcSender/Receiver` | `v1/engine/tensor_ipc.py:45/114` |
| DP 引擎间 | `torch.distributed` all-reduce(同步 finished) | `core.py:2027` |
| 引擎 → DPCoordinator | ZMQ XSUB 订阅推送 `SchedulerStats` | `core.py:1678-1688` |

### 2.5 DP>1 时的额外进程

- `DPCoordinator`(`v1/engine/coordinator.py:23`)/`DPCoordinatorProc`(:146):收集各 DP 引擎的 waiting/running 队列长度,发布给前端做负载均衡,并广播 `START_DP_WAVE`。
- 仅 **MoE 模型** 用 `DPEngineCoreProc`(`core.py:1963`),实现 wave 同步 + 两阶段暂停;非 MoE 的 DP rank 完全独立(`reconfigure_for_independent_dp_rank`,`core.py:1314`)。

---

## 3. 事件循环:busy loop 与一步的完整流程

### 3.1 概念模型与真实实现

很多资料用 `INPUT_ADDED → SCHEDULER_ITERATION → EXECUTE_MODEL → SAMPLE → OUTPUT` 描述引擎状态机——那是 **V0 的 `EngineCoreEventType` 枚举**。在 V1,这套概念被**一个 busy loop + 两次调度回调**取代,但语义一一对应:

| V0 概念事件 | V1 真实落点 |
|---|---|
| INPUT_ADDED | `_process_input_queue()` 收到 `ADD` → `scheduler.add_request()`(`core.py:1401` → `scheduler.py:2301`) |
| SCHEDULER_ITERATION | `scheduler.schedule()`(`scheduler.py:476`) |
| EXECUTE_MODEL | `model_executor.execute_model(scheduler_output, non_block=True)`(`core.py:595`) |
| SAMPLE | `future.result()` 为空时 `sample_tokens`(`core.py:603`) |
| OUTPUT | `scheduler.update_from_output()` 产出 `EngineCoreOutputs` → 输出线程 PUSH(`core.py:608`) |

> 小提醒:V1 里 `EngineCoreEventType`(`v1/engine/__init__.py:162`)是**请求级事件**(`QUEUED=1`/`SCHEDULED=2`/`PREEMPTED=3`),不是引擎级状态,别和 V0 的引擎事件混淆。

### 3.2 `run_busy_loop`(core.py:1375-1386)

```python
def run_busy_loop(self):
    while self._handle_shutdown():          # 处理 SIGTERM/SIGINT:abort 或 drain
        self._process_input_queue()         # 1) 收客户端请求
        self._maybe_publish_request_counts()# 2) DP 负载均衡统计
        self._process_engine_step()         # 3) 一步:调度→执行→产出
        self._maybe_publish_request_counts()
    raise SystemExit                        # 退出进程
```

- `has_work()`(`core.py:1362`)= `engines_running or scheduler.has_requests() or bool(batch_queue)`,无请求时 `_process_input_queue` 阻塞在 `input_queue.get()`(`core.py:1417`),不空转。
- 退出协议 `_handle_shutdown`(`core.py:1456`):按 `shutdown_timeout` 决定立即 abort(timeout=0 → `finish_requests(None, FINISHED_ABORTED)`)还是 drain(先跑完在途请求)。

### 3.3 `_process_engine_step` 与 `step()`(core.py:1432 / 583)

```python
# step() 精简版 (core.py:583)
scheduler_output = scheduler.schedule(throttle_prefills)            # 调度
future = model_executor.execute_model(scheduler_output, non_block=True)  # 异步执行
grammar_output = scheduler.get_grammar_bitmask(scheduler_output)    # 语法约束
model_output = future.result()          # 阻塞等 GPU 前向
self._process_aborts_queue()            # 处理急停
engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)  # 产出
```

`step_with_batch_queue`(`core.py:624`,当 `max_concurrent_batches>1` 启用)进一步把"调度"和"执行"流水化:先把新批次 `batch_queue.appendleft(...)`,再 `batch_queue.pop()` 取最早批次的结果,PP 场景下消除流水线气泡。

### 3.4 输入/输出线程

- **输入线程**(`core.py:1651`):ZMQ DEALER 连接各前端;DP 时另连 coordinator 的 XSUB 订阅负载均衡统计;`poller.poll()` 循环收 `(type_frame, *data_frames)` → 解码 → `input_queue.put_nowait`。
- **输出线程**(`core.py:1754`):`output_queue.get()` → `ENGINE_CORE_DEAD` 特判(linger=4000 保证死讯送达,`core.py:1791`)→ 普通输出 `encode_into` 零拷贝发送。

### 3.5 一次请求的完整旅程

```
前端 add_request ─► ZMQ ADD ─► 输入线程 preprocess (core.py:1722-24)
  ─► input_queue ─► EngineCore.add_request (core.py:438)
  ─► scheduler.add_request (scheduler.py:2301, 进 waiting)
  ─► schedule() 分配 KV 块 (scheduler.py:631) ─► RUNNING
  ─► execute_model / sample_tokens (每步)
  ─► update_from_output: 产出 EngineCoreOutput(new_token_ids/logprobs)
  ─► 输出线程 PUSH ─► 前端 output_handler ─► process_outputs ─► RequestOutput 流式下发
  ─► 满足结束条件 ─► finish_requests (scheduler.py:2325) ─► FINISHED_* + 释放 KV
  ─► 前端 RequestOutputCollector 收到 STREAM_FINISHED ─► generate() 返回
```

---

## 4. 关键类 / 函数速查(文件:行号)

### `v1/engine/core.py`
- `EngineCore`:104;`__init__`:107;`add_request`:438;`abort_requests`:484;`step`:583;`post_step`:615;`step_with_batch_queue`:624;`pause_scheduler`:828;`resume_scheduler`:859;`sleep`:867;`wake_up`:905;`collective_rpc`:952;`preprocess_add_request`:968
- `EngineShutdownState`:1001;`EngineCoreProc`:1007;`_perform_handshakes`:1129;`startup_handshake`:1233;`run_engine_core`:1271;`has_work`:1362;`run_busy_loop`:1375;`_process_input_queue`:1401;`_process_engine_step`:1432;`_handle_shutdown`:1456;`_handle_client_request`:1504;`_invoke_utility_method`:1567;`_send_engine_dead`:1602;`_make_ready_response`:1616;`process_input_sockets`:1651;`process_output_sockets`:1754
- `DPEngineCoreProc`:1963(仅 MoE)

### `v1/engine/core_client.py`
- `EngineCoreClient`:78;`make_client`:90;`make_async_mp_client`:116;`InprocClient`:306;`MPClient`:503;`SyncMPClient`:805;`AsyncMPClient`:977;`DPAsyncMPClient`:1252;`DPLBAsyncMPClient`:1434;`get_core_engine_for_request`:1471

### `v1/engine/__init__.py`(跨进程消息类型)
- `PauseMode`:27;`FinishReason`:43;`EngineCoreReadyResponse`:69;`EngineCoreRequest`:100;`EngineCoreEventType`:162;`EngineCoreEvent`:170;`EngineCoreOutput`:189;`UtilityOutput`:230;`EngineCoreOutputs`:242;`EngineCoreRequestType`:273;`ReconfigureDistributedRequest`:289;`EngineStatusType`:308

### 前端 / Executor / Worker
- `v1/engine/llm_engine.py:48 LLMEngine`;`async_llm.py:72 AsyncLLM`;`input_processor.py:38 InputProcessor`;`output_processor.py:438 OutputProcessor`;`detokenizer.py:31 IncrementalDetokenizer`
- `v1/engine/utils.py:62 EngineZmqAddresses`、:120 `CoreEngineProcManager`、:1005 `get_engine_zmq_addresses`、:1070 `launch_core_engines`、:1216 `wait_for_engine_startup`
- `v1/executor/multiproc_executor.py:597 WorkerProc`、:372 `collective_rpc`、:824 `monitor_death_pipe`
- `v1/worker/worker_base.py:39 WorkerBase`;`gpu_worker.py:142 Worker`、:665 `initialize_from_config`、:1053 `execute_model`
- `v1/request.py:59 Request`、:351 `RequestStatus`

### 环境变量
- `VLLM_ENABLE_V1_MULTIPROCESSING`(`envs.py:156`,默认 **True**):置 0 才退回 `UniProcExecutor` 单进程模式(`uniproc_executor.py:180` 会 assert 该变量为 0)。
- `VLLM_ENGINE_READY_TIMEOUT_S`(`envs.py:27`,默认 600s):引擎启动握手超时。
- `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`(`envs.py:167`,默认 128):前端单轮事件循环内规格化输出的分片大小。

---

## 延伸阅读

- vLLM 官方文档:引擎与 API 服务架构 <https://docs.vllm.ai/en/latest/design/arch_overview.html>
- vLLM Blog:《vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention》 <https://blog.vllm.ai/2023/06/20/vllm.html>
- vLLM Blog:《Announcing vLLM V1: A Major Architectural Upgrade》 <https://blog.vllm.ai/2025/01/27/v1-alpha-release.html>
- 论文:SOSP'23 PagedAttention <https://arxiv.org/abs/2309.06180>
