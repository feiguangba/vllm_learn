# 07 · vLLM 张量/流水线/数据并行:进程组、层切分与通信器

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `distributed/parallel_state.py:1751` = 上游源码中的 `vllm/distributed/parallel_state.py` 第 1751 行)。
> **一句话总结**:vLLM 用 `initialize_model_parallel`(`distributed/parallel_state.py:1751`)把世界按 **DP×PP×PCP×TP** 五维排布切出若干 `GroupCoordinator`;TP 靠 `Column/RowParallelLinear` 把 GEMM 权重沿输出/输入维切开并用 all-gather / reduce-scatter 拼回;PP 靠 `make_layers` 把 transformer 层切成 `[start_layer, end_layer)` 段并只让首/尾 rank 持有 embed/norm/lm_head;底层通信交给一条 **symm-mem → CustomAllReduce(共享内存+IPC)→ PyNccl(直接 libnccl.so)** 的快速路径链。

---

## 一图看懂

```
  8 GPU, world rank 0..7, TP=2, PP=4 (无 DP/PCP)
  ────────────────────────────────────────────────────────────────
  rank:           0    1  |  2    3  |  4    5  |  6    7
  TP rank:        0    1  |  0    1  |  0    1  |  0    1   ← TP 最内层(相邻)
  PP rank:        0    0  |  1    1  |  2    2  |  3    3
  ────────────────────────────────────────────────────────────────
  TP 组(4个):   [0,1]  [2,3]  [4,5]  [6,7]     ← 张量并行,组内 all-gather/RS
  PP 组(2个):   [0,2,4,6]  [1,3,5,7]          ← 流水线并行,stage 间 send/recv
  DP 组(8个):   [0][1]…[7]                    ← 数据并行(本例 size=1)
  ────────────────────────────────────────────────────────────────
  建组流程  init_distributed_environment (:1586) ─ init_world_group(:1303) → _WORLD
             └─ initialize_model_parallel (:1751)
                  TP: all_ranks.view(-1,2)        → init_model_parallel_group("tp")
                  PP: all_ranks.transpose(2,4)    → init_model_parallel_group("pp")
                  DP: all_ranks.transpose(1,4)    → init_model_parallel_group("dp")
                 每个 GroupCoordinator(:380): torch.distributed.new_group(device=nccl)
                                             + new_group(cpu=gloo) + 可选 device_communicator

  TP 前向分工                    ColumnParallelLinear(:401)       RowParallelLinear(:1504)
                                 Y_i = X·A_i (A=[A_0|A_1])       Y_i = X_i·A_i (A=[A_0;A_1])
                                 gather_output → all-gather(:578)  reduce_results → all-reduce(:1653)
   ──────────┴──────────          qkv_proj: QKVParallelLinear(:965) 按 head 维切
                                 gate_up: MergedColumnParallelLinear(:639) 按块切

  PP 层切分  make_layers(models/utils.py:786) → get_pp_indices(distributed/utils.py:127)
             段外层 = PPMissingLayer(:773) 占位;embed 只首 rank(:374),norm/lm_head 只末 rank(:389/:485)

  all-reduce 快速路径  CudaCommunicator.all_reduce(cuda_communicator.py:275)
   symm_mem → quick(ROCm) → flashinfer → aiter → CustomAllreduce(:313) → symm_mem_comm
   → PyNccl(pynccl.py:166) → torch.distributed.all_reduce(最后兜底)
```

---

## 1. 总体结论:三种并行的角色分工

vLLM 的分布式模型来自 Megatron-LM(文件头部注释 `parallel_state.py:5-7` 声明"adapted from NVIDIA/Megatron-LM")。三种并行解决的是**不同维度的瓶颈**:

| 并行 | 切什么 | 动机 | vLLM 落点 |
|---|---|---|---|
| **TP(张量并行)** | 单个 GEMM 权重沿行/列切开,多卡算同一个矩阵乘法 | 单卡显存放不下权重/激活,且 NVLink 内带宽高 | `ColumnParallelLinear` / `RowParallelLinear`(`linear.py:401/:1504`) |
| **PP(流水线并行)** | 把 transformer 层沿纵深切成几段,每段算完后把中间激活传给下一段 | 权重总量超过单机多卡显存,PP 组间只传 hidden_states(比全量 all-reduce 轻) | `make_layers`(`models/utils.py:786`)+ `PPMissingLayer` 占位 |
| **DP(数据并行)** | 不同 GPU 跑**不同的请求**(完整模型副本) | 吞吐扩展:请求并行度随卡数线性增长 | 每 DP rank 一个独立 EngineCore(`v1/engine/core.py:1271`,见文档 01) |

三个关键认知:

1. **TP 是"每步都通信"、PP 是"每段一通信"、DP 是"几乎不通信"**。TP 组内的 all-reduce / all-gather 落在**每层前向的每步解码**上,是延迟敏感路径;PP 只在相邻 stage 间传一次中间激活(且 V1 已用异步调度把气泡基本抹平,见文档 03 的 `num_in_flight_tokens`);DP 之间只在调度负载均衡时交换统计(`v1/engine/coordinator.py:146`,文档 01)。
2. **维度顺序是硬编码的** `ExternalDP × DP × PP × PCP × TP`(`parallel_state.py:1817-1832` 注释)。TP 永远最内层,所以 TP rank 相邻(同一节点),PP 跨 TP 组大步长——这样 TP 通信优先落在 NVLink 上。
3. **并非三选一**:`parallel_config`(`config/parallel.py:119`)同时持有 `tensor_parallel_size`(:124)、`pipeline_parallel_size`(:122)、`data_parallel_size`(:129),三者可叠加。`world_size` = PP×TP×PCP(`parallel.py:833-837`),`world_size_across_dp` 再乘 DP(`parallel.py:549`)。

---

## 2. 进程组初始化:initialize_model_parallel

### 2.1 两步初始化流程

分布式初始化分两步(文件顶部 docstring `parallel_state.py:8-24`):

1. **`init_distributed_environment`**(`parallel_state.py:1586`):调用 `torch.distributed.init_process_group`(:1685)建立默认(世界)进程组,并据此构造 `_WORLD = init_world_group(...)`(:1722)。
2. **`initialize_model_parallel`**(`parallel_state.py:1751`):按维度切出 TP / PP / DP / EP 等子组,填入全局单例 `_TP` / `_PP` / `_DP` / `_EP`。

> 对外通常只调用 `ensure_model_parallel_initialized`(`parallel_state.py:1997`),它内部幂等地调用上述两步。

### 2.2 8 GPU、TP2 PP4 的切分实例

`initialize_model_parallel` 的 docstring(`parallel_state.py:1768-1779`)给的正是这个例子,我们把推导过程补全。设 8 个 rank g0..g7,`tensor_model_parallel_size=2`、`pipeline_model_parallel_size=4`:

```python
# parallel_state.py:1826-1832  五维排布,前两维(ExternalDP、DP)本例都是 1
all_ranks = torch.arange(8).reshape(-1, 1, 4, 1, 2)
#          shape = (ExternalDP, DP, PP, PCP, TP) = (1, 1, 4, 1, 2)

# TP 组: 展平最后一维 → 4 个组
all_ranks.view(-1, 2)              # [g0,g1] [g2,g3] [g4,g5] [g6,g7]   (:1837)
# PP 组: 把 PP 维(索引2)搬到 TP 维(索引4)再展平 → 2 个组
all_ranks.transpose(2, 4).reshape(-1, 4)
                                   # [g0,g2,g4,g6] [g1,g3,g5,g7]        (:1891-1893)
# DP 组: 把 DP 维(索引1)搬到 TP 维再展平 → 8 个单元素组
all_ranks.transpose(1, 4).reshape(-1, 1)
                                   # [g0] [g1] … [g7]                   (:1908-1909)
```

每个维度的 `group_ranks` 交给 `init_model_parallel_group`(`parallel_state.py:1315`),它只是 `GroupCoordinator` 的一个工厂函数:

| 组 | 工厂调用点 | 特殊参数 |
|---|---|---|
| `_TP` | `parallel_state.py:1843` | `use_message_queue_broadcaster=True`(共享内存广播,文档 01 的 worker 通信会用到) |
| `_PP` | `parallel_state.py:1902` | 默认参数 |
| `_DP` | `parallel_state.py:1919` | 默认参数 |
| `_EP` | `parallel_state.py:1949` | 仅 MoE 模型创建(`config.model_config.is_moe`,`:1926`) |
| `_WORLD` | `init_world_group`(`:1303`) | `use_device_communicator=False`(世界组不做通信优化) |

> **PP 的 rank 计算速记**:在 TP2 PP4 布局下,`pp_rank = global_rank // tp_size`、`tp_rank = global_rank % tp_size`。这条公式出现在 `config/model.py:1525-1527`(`pp_rank = (rank // tp_size) % pp_size`),模型初始化时用来判断"我这个 rank 是哪段流水线"。

### 2.3 GroupCoordinator 的结构(`parallel_state.py:380`)

`GroupCoordinator` 是 **PyTorch `ProcessGroup` 的封装**(docstring :382-387),它同时管 CPU 和 GPU 通信。构造(`__init__` :409)要点:

- **双 ProcessGroup**:对每个候选组调用 `torch.distributed.new_group(ranks, backend=device 后端)`,再建一个 `backend="gloo"` 的 CPU 组(:456-472)。前者走 NCCL 做 GPU 集体通信,后者做 CPU 侧的协调(建消息队列、对象广播等)。
- **三套 rank 编号**(:394-402):`rank`(全局)、`local_rank`(节点内)、`rank_in_group`(组内)。组内排列按 docstring 里的 4 卡 2 节点表格。
- **device_communicator**(:501-513):当 `use_device_communicator=True` 且 `world_size > 1` 时,由平台类实例化一个设备通信器(CUDA 上就是 `CudaCommunicator`,见 §7),它是快速 all-reduce 链的宿主。
- **mq_broadcaster**(:517-521):可选共享内存广播队列(`shm_broadcast.MessageQueue`),TP 组开了它。
- **常用成员**:`first_rank`/`last_rank`(:585/:590)、`is_first_rank`/`is_last_rank`(:595/:600,PP 首尾判断全靠它)、`next_rank`/`prev_rank`(:605/:612,PP 相邻 stage 的 send/recv 目标)、`device_group`/`cpu_group`(:480-481)。

**集体操作全部委托给 device_communicator**(以 `all_reduce` 为例,:662):`world_size == 1` 时直通返回(:678);`use_custom_op_call` 时走 `torch.ops.vllm.all_reduce(input_, group_name=...)` 自定义 op(:681-683);否则落到 `_all_reduce_out_place` → `device_communicator.all_reduce`(:684-689)。`all_gather`(:691)、`reduce_scatter`(:722)、`gather`(:750)、`broadcast`(:766)、`send`(:1209)、`recv`(:1216)、`barrier`(:1200)同构。`unique_name` 由 `_get_unique_name`(:110)生成并注册进全局 `_groups` 字典(:126),供 custom op 按名字反查组。

### 2.4 CUDA Graph 捕获与进程组(graph_capture)

`GroupCoordinator.graph_capture`(:619)是一个上下文管理器:进入时切到独立 CUDA stream、让 device_communicator 的 `ca_comm.capture()`(:642-644)进入自定义 all-reduce 的图捕获态,退出时恢复。模块级函数 `graph_capture`(:1450)把 **TP 组和 PP 组**的捕获叠在一起(:1474):

```python
with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
    yield context
```

这就是文档 05 里 CUDA Graph 捕获前向时,collective 也能被"冻结"进图的原因——捕获期间通信 kernel 不真正执行,重放时直接跑预录的 kernel 序列。

---

## 3. TP:张量并行层(linear.py)

TP 的核心单位是 `LinearBase`(:215)的两个子类。它们不重写数学本身,而是重写 **权重怎么切、输出怎么拼**。

### 3.1 ColumnParallelLinear(列切 + all-gather,`linear.py:401`)

定义 `Y = X·A + b`,`A` 沿**第二维(输出维)**切:`A = [A_0 | … | A_p]`(docstring :404-405)。

- **建参**:`output_size_per_partition = divide(output_size, tp_size)`(:461),即每 rank 只持 `输出/tp` 列;`tp_rank`/`tp_size` 取自全局 `get_tensor_model_parallel_rank/world_size()`(:452-459)。
- **前向**(`forward` :569):本地 GEMM(`self.quant_method.apply`,`:576`)后,若 `gather_output=True` 且 `tp_size > 1`,调 `tensor_model_parallel_all_gather(output_parallel)`(:578-580)把各 rank 的列拼回完整输出。

- **weight_loader**(:542):从 checkpoint 加载时,`loaded_weight.narrow(output_dim, tp_rank*shard_size, shard_size)`(:550-551)只把属于本 rank 的那一段权重拷进参数——这也是 `tp_rank` 决定 rank 拿哪一段的依据。

`tensor_model_parallel_all_gather` 只是 `get_tp_group().all_gather` 的别名(`communication_op.py:17-21`)。

### 3.2 RowParallelLinear(行切 + all-reduce,`linear.py:1504`)

`A` 沿**第一维(输入维)**切(`A = [A_0; …; A_p]`,docstring :1507-1515),输入 `X` 已被上游按列切开(`X = [X_0, …, X_p]`):

- **建参**:`input_size_per_partition = divide(input_size, tp_size)`(:1557),每 rank 持 `输入/tp` 行。
- **前向**(`forward` :1635):若 `input_is_parallel` 直接收上游切好的输入,否则自己 `split_tensor_along_last_dim` 再取第 `tp_rank` 片(:1639-1645);本地 GEMM 后,`reduce_results=True` 时 `tensor_model_parallel_all_reduce(output_parallel)`(:1653-1654)求和拼回完整输出。

- **bias 只在 rank 0 加一次**(:1648-1650):`bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias`,避免 TP>1 时 bias 被重复加。
- **weight_loader**(:1608):按 `input_dim` narrow 本 rank 的行段(:1613-1616)。

### 3.3 MergedColumnParallelLinear(按块切,`linear.py:639`)

把**多个列并行线性层压成一张大权重矩阵**(docstring :641-644),常用于 FFN 的 gate/up:`Y_gate = X·W_gate`、`Y_up = X·W_up` 合并成 `X·[W_gate|W_up]`。要点:

- `__init__`(:665):持 `output_sizes = [gate, up, …]`(:679),把 `output_size=sum(output_sizes)` 交给父类按列切(:684-695)。父类对每个 output_size 单独 `divide(…, tp_size)`(`linear.py:464-467`)。
- **weight_loader**(:724)按 `loaded_shard_id`(gate 为 0、up 为 1,对应 `llama.py:352-353` 的 `.gate_proj → .gate_up_proj` 映射)定位段偏移,再在段内按 `tp_rank` 切(:750-758+ 后续逻辑)。

### 3.4 QKVParallelLinear(按 head 维切,`linear.py:965`)

自注意力最常用的 QKV 投影。它把 Q、K、V 三张列并行权重合为一张,并沿 **head 维** 切分(docstring :968-973)。GQA/MQA 的关键是 **KV head 可以少于 Q head**:

```python
# linear.py:1017-1024  TP 下的 head 分配
self.num_heads = divide(self.total_num_heads, tp_size)
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1                              # KV head 不够分 → 每 rank 1 个
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)  # 其余 rank 复制
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

**实例**:Llama-2-7B(32 Q head、32 KV head、head_size 128、hidden 4096)在 TP=2 下:

```
total_num_heads=32, tp_size=2, num_heads=16
total_num_kv_heads=32 ≥ tp_size → num_kv_heads=16, replicas=1
output_sizes = [16*128*2, 16*128*2, 16*128*2] = [4096, 4096, 4096]  # 全量
每 rank 实持:  q=2048 列  k=2048 列  v=2048 列   (列切后 4096/2)
```

若换成 MQA(1 个 KV head)、TP=8:`num_kv_heads=1`、`num_kv_head_replicas=8`——**8 个 rank 各自持有同一个 KV head 的完整副本**,因为 KV 权重只有 1 组 head 的宽度,切无可切,只能复制。

权重加载时按 shard_id `"q"/"k"/"v"` 计算段偏移(`_get_shard_offset_mapping` :1054、`_get_shard_size_mapping` :1064),disk 上已融合的权重走 `_load_fused_module_from_checkpoint`(:1072)自动拆段。

### 3.5 Embedding 与 LM Head 的 TP

- **embedding**:`VocabParallelEmbedding`(`vocab_parallel_embedding.py:198`)沿**词表维**切(`num_embeddings_per_partition`),`forward`(:487)里把不在本 rank 词表区的 token 掩掉(:490-497),输出再做一次 `tensor_model_parallel_all_reduce`(:506)。
- **LM Head**:`ParallelLMHead`(:521)是 `VocabParallelEmbedding` 的子类,同样按词表切;它的 `forward` 直接抛 `RuntimeError`(:579-581),因为推理时 logits 由 `LogitsProcessor` 处理。`LogitsProcessor._gather_logits`(`logits_processor.py:84`)负责把各 rank 的局部 logits 用 `tensor_model_parallel_gather` / `all_gather` 拼回全词表(:92/:95),然后采样只发生在 rank 0 并广播结果。

---

## 4. PP:流水线并行与层切分

### 4.1 切层:make_layers(`model_executor/models/utils.py:786`)

```python
# utils.py:806-818
start_layer, end_layer = get_pp_indices(
    num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size)
modules = torch.nn.ModuleList(
    [PPMissingLayer() for _ in range(start_layer)]          # 段前占位
    + layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
    + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])  # 段后占位
```

- **段区间由 `get_pp_indices` 算**(`distributed/utils.py:127-172`):`start_layer = sum(partitions[:pp_rank])`、`end_layer = start_layer + partitions[pp_rank]`(:169-170)。默认均匀分,余数优先分给**中间段**(:159-161),因为首段有 embedding、末段有 norm/lm_head,算力本身更重;可用环境变量 `VLLM_PP_LAYER_PARTITION` 手工覆盖(:143)。
- **段外是 `PPMissingLayer`**(`utils.py:773`):一个 `nn.Identity`,`forward` 原样透传第一个参数(:781-783),保证 `ModuleList` 长度仍是全层数(便于权重加载时按名字对齐),但**不含任何参数、不会被调用到**。
- **LlamaModel 里的实际调用**(`models/llama.py:384`):`self.start_layer, self.end_layer, self.layers = make_layers(...)`,`forward` 里用 `islice(self.layers, self.start_layer, self.end_layer)` 只遍历本段(:421-426)。

### 4.2 首尾 rank 的 embed / norm / lm_head 处理

三个模块都只在本阶段需要时才真正实例化,否则放 `PPMissingLayer` 占位:

```python
# llama.py:374-383  embed_tokens: 首 rank 必有;末 rank 仅在 tie_word_embeddings 时也要
if get_pp_group().is_first_rank or (
    config.tie_word_embeddings and get_pp_group().is_last_rank):
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()

# llama.py:389-392  final norm: 只有末 rank 需要
if get_pp_group().is_last_rank:
    self.norm = RMSNorm(...)
else:
    self.norm = PPMissingLayer()

# llama.py:485-500  lm_head + LogitsProcessor: 只有末 rank 有
if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)
    self.logits_processor = LogitsProcessor(...)
else:
    self.lm_head = PPMissingLayer()
```

前向(`LlamaModel.forward`,`llama.py:401`)据此分三态:

1. **首 rank**(`is_first_rank` :409):做 embedding,`residual = None`。
2. **中间 rank**(:415-418):从上游收 `intermediate_tensors`,解出 `hidden_states` 和 `residual`;跑完本段层后**不输出 hidden_states,而是返回 `IntermediateTensors`**(:431-434),由执行层用 `send`/`recv` 传给下一 stage。
3. **末 rank**(:436):过 final norm(`self.norm`),产出真正的 `hidden_states`,随后 `LlamaForCausalLM.compute_logits`(:529)过 `lm_head`。

> `IntermediateTensors`(`models/intermediate_tensors.py`)就是 PP 的"包裹层"——它把 `hidden_states` 和 `residual` 打包成一个对象,`pipeline_model_parallel` 的 `send_tensor_dict`/`recv_tensor_dict`(`parallel_state.py:981/:1076`)沿 PP 组搬运。

### 4.3 PP 组内的通信原语

PP 组(`get_pp_group()`)上可用 `next_rank`/`prev_rank`(:605/:612)定位上下游,`send`(:1209)/`recv`(:1216)直接在 GPU 张量上做 point-to-point。因为是 stage 间的单次传输,PP 组**不**启用 custom all-reduce(§7 里 `CudaCommunicator.__init__` 的 `if "tp" not in unique_name:` 分支 `cuda_communicator.py:50-55` 直接关掉了所有快速路径)——PP 不需要 all-reduce,只需要带宽。

### 4.4 PP 运行时数据流:异步 send/recv 与采样广播

V1 的 PP 是**全异步流水线**(调度器用 `num_in_flight_tokens`/`num_output_placeholders` 记账,见文档 03 §5.3),stage 间的张量传输也用非阻塞原语叠在 GPU 前向后面(`gpu_worker.py:1134-1139` 注释 "launch non-blocking send")。执行路径拆成三步:

1. **收上游中间激活**(仅非首 rank,`gpu_worker.py:1098-1110`):`get_pp_group().irecv_tensor_dict(...)`(:1100)发起**异步接收**,并把 `all_gather_group=get_tp_group()` 传进去——`irecv_tensor_dict`/`isend_tensor_dict`(`parallel_state.py:1114/:1019`)在收/发 hidden 张量的同时,还能顺带对其中某些键做一次 TP all-gather(序列并行场景下 residual 的分片恢复,`gpu_worker.py:1092-1096`)。
2. **执行本段前向** `model_runner.execute_model`(:1113),非末 rank 拿到的是 `IntermediateTensors` 输出(:1127),立即 `isend_tensor_dict` 异步发给下一 stage(:1135)。
3. **采样 token 回广播**(末 rank → 所有 rank):真正采样只发生在末 rank,非末 rank 也要做下一轮的采样/状态推进。`PPHandler`(`v1/worker/gpu/pp_utils.py:52`)在一条**旁路 stream** 上用独立 NCCL 通信器广播:构造时 `make_sibling_device_group("pp_broadcast")`(:86,`parallel_state.py:535`)建一个与 PP `device_group` 同成员但**互不串线**的兄弟组;末 rank 调 `broadcast`(:170)发 `sampled_token_ids/num_sampled/num_rejected`,非末 rank 调 `receive`(:126)在旁路 stream 收,配 `torch.cuda.Event`(:152)延迟回主 stream 消费。队列按 PP 宽度预填占位(`pp_utils.py:76-78`),第 T 步的 recv 恰好到 T+pp_size 步才被消费,与调度器 `num_in_flight_tokens` 栅栏对齐。

> 一句话:PP 阶段间传 **hidden_states(一条 p2p 链路)+ sampled tokens(一条广播链路)**,两条都用非阻塞原语 + 独立 stream 叠在计算后面,才把流水线气泡压到接近零(配合文档 03 的 `max_concurrent_batches` 多批次并行)。

---

## 5. NCCL 通信器:PyNcclCommunicator

### 5.1 为什么绕开 torch.distributed 直接调 NCCL

`PyNcclCommunicator`(`pynccl.py:60`)直接通过 `ctypes` 封装 `libnccl.so`(`NCCLLibrary`,`pynccl_wrapper.py`)。动机写在构造里:

- **绕过 torch.distributed 的额外开销**。NCCL 2.19+ 提供 `ncclCommInitRank` + 通用 dtype/op,`PyNcclCommunicator` 直接用 `ncclCommInitRank`(:137)建组、在**当前 CUDA stream** 上提交操作,不给 PyTorch 的 PG 抽象留中间层。
- 它挂在一个**非 NCCL 后端**的 `ProcessGroup` 上(docstring :78-82 断言 `dist.get_backend(group) != NCCL`),即上一节说的 **gloo cpu_group**——用它做 `ncclUniqueId` 的同步广播(:118-127),真正的数据面完全走 libnccl。

### 5.2 构造与身份同步(`__init__` :61)

- rank 0 用 `ncclGetUniqueId` 生成 `unique_id`(:112),其余 rank 构造空 id(:116),再通过 gloo 组 `broadcast` 统一(:118-127)。
- 绑定一个**唯一设备**(docstring :76),`ncclCommInitRank(world_size, unique_id, rank)` 在 `torch.accelerator.device_index(device.index)` 上下文里建组(:136-139)。
- 结尾做一次 1 元素 `all_reduce` 热身并 `stream.synchronize()`(:141-146),确保通信器真正就绪。

### 5.3 核心集体操作

三个方法签名都是 `(输入, 输出, …)`,**调用方负责分配输出张量**,内部转成裸指针:

```python
# pynccl.py:166-197  all_reduce: 裸指针 + numel + dtype/op 枚举,提交到当前 CUDA stream
self.nccl.ncclAllReduce(buffer_type(in_tensor.data_ptr()), buffer_type(out_tensor.data_ptr()),
    in_tensor.numel(), ncclDataTypeEnum.from_torch(in_tensor.dtype),
    ncclRedOpTypeEnum.from_torch(op), self.comm, cudaStream_t(stream.cuda_stream))
```

- `all_reduce`(:166):`in_tensor → out_tensor`,`ReduceOp.SUM`(默认)。
- `all_gather`(:199):`output_tensor` 由调用方按 `世界大小×输入` 形状分配。
- `reduce_scatter`(:257):输入是拼接好的整张量,输出是 `1/世界` 的切片。
- 另有 `all_gatherv`(:222)/`reduce_scatterv`(:285):变长切片用 `ncclGroupStart/End` 包一组 `ncclBroadcast`/`ncclReduce` 循环。
- point-to-point:`send`(:322)/`recv`(:349)/`broadcast`(:376),FP8 张量会转成 `uint8` 走(**NCCL 本身不支持 fp8 规约**,:331-339/:358-366)。

所有方法都先断言 `in_tensor.device == self.device`(:178),因为 NCCL 通信器绑定设备,串了会 "illegal memory access"。

---

## 6. CustomAllReduce:共享内存 + IPC 的"零拷贝" all-reduce

### 6.1 思路

NCCL all-reduce 需要数据在卡间流动若干轮。CustomAllReduce 换一条路:**把所有 rank 的 GPU buffer 用 CUDA IPC 互相映射**,让每个 rank 直接读写别人的显存,再用一个自定义 kernel 在**一次 kernel 启动**里完成规约,中间结果留在共享临时 buffer(`custom_all_reduce.py:204-205` 注释)供所有 rank 读取。

### 6.2 前提检查:能不能用(`__init__` :70)

逐级把关,任一不满足就 `self.disabled = True` 退出:

1. **自定义库存在**(:107-114):`_custom_ops` 里没有 `meta_size` 就禁用(CPU 环境)。
2. **同节点**(:122-123):`in_the_same_node_as(group, source_rank=0)`;跨节点只能用 NCCL。
3. **world_size 受支持**(:132-140):`_SUPPORTED_WORLD_SIZES = [2,4,6,8,16]`(:57)。
4. **PCIe 拓扑**(:165-185):用 `is_fully_connected(physical_device_ids)`(:178)检查 GPU 是否全互连(NVLink/同 switch),超过 2 张 PCIe-only GPU 就禁用。
5. **P2P 能力实测**(`_can_p2p`,:36-50):对除自己外的每个 rank 调 `gpu_p2p_access_check`(:48,来自 `all_reduce_utils.py:336`),失败即禁用;`VLLM_SKIP_P2P_CHECK=1` 时降级为只信 driver 的 `torch.cuda.can_device_access_peer`(:44-47)。

### 6.3 共享 buffer 的建立:create_shared_buffer(`custom_all_reduce.py:521`)

```python
pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)  # 本卡分配
dist.all_gather_object(handles, handle, group=group)                    # 交换 IPC handle
pointers = [pointer if i == rank else ops.open_mem_handle(h)            # 映射别人家的 buffer
            for i, h in enumerate(handles)]
```

- `allocate_shared_buffer_and_handle` 返回本卡 CUDA IPC 内存的 `(指针, handle)`,`all_gather_object` 把 handle 广播给同组所有 rank,`open_mem_handle` 把别人的显存映射进自己的地址空间(:534-539)。
- 每个 rank 由此得到**指向所有 peer buffer 的指针数组** `buffer_ptrs`,配合 `ops.init_custom_ar`(:243)建内核句柄。映射的是**每个 rank 的同一块共享 buffer**,所以 all-reduce 的中间结果写到 buffer 后,所有 rank 都能读——这就是"共享内存"的含义。

### 6.4 何时走 custom 路径:should_custom_ar(:348)

```python
if self.disabled or self.world_size > 8: return False
if inp_size % 16 != 0: return False                 # 字节数须为 16 的倍数
if not is_weak_contiguous(inp): return False        # 必须弱连续
if self.world_size == 2 or self.fully_connected:
    return inp_size < self.max_size                 # 只在小张量上用
return False
```

**取舍**:
- 只有当 **TP=2 或 NVLink 全互连** 时才可能命中,且张量要小(`max_size` 默认 8MB,:74)。大张量/PCIe-only 多卡时 NCCL 反而更快(custom 是总线上的共享内存搬运,NCCL 有更成熟的拓扑感知分块)。
- CUDA Graph 捕获期间走 `registered=True` 的零拷贝路径(:387-393):输入直接指向已 IPC 注册的 buffer,不复制;非捕获时先 `cudaMemcpy` 进预注册 buffer(:398),注释说这笔拷贝开销约 ≤1% 延迟(:396-397)。
- 同时提供 `custom_all_gather`(:424)/`custom_reduce_scatter`(:477)及对应 `should_*` 判定(:400/:454),供 TP 里 all-gather / reduce-scatter 复用同一套 IPC buffer。

---

## 7. 通信优先级链:CudaCommunicator.all_reduce

`CudaCommunicator`(`cuda_communicator.py:29`)是所有快速路径通信器的**宿主**:构造时(`__init__` :30)按开关依次建好 `pynccl_comm`(:86)、`ca_comm`(:119)、`symm_mem_comm`(:100)、`fi_ar_comm`(:106)、`aiter_ar_comm`(:112)、`qr_comm`。

`all_reduce`(:275)从上到下逐级尝试,命中即返回,全部落空才用 `torch.distributed.all_reduce` 兜底:

| 优先级 | 后端 | 触发条件 | 行号 |
|---|---|---|---|
| 1 | **symm-mem**(torch symmetric memory) | `should_nccl_symm_mem_allreduce` 命中,走自定义 op `all_reduce_symmetric_with_copy` | :278-283 |
| 2 | **QuickReduce** | 仅 ROCm MI300 系,`qr_comm.should_quick_allreduce` | :286-294 |
| 3 | **FlashInfer AR** | `VLLM_ALLREDUCE_USE_FLASHINFER` 且 `should_use_fi_ar` | :295-303 |
| 4 | **AITER custom**(ROCm) | `should_custom_ar` | :304-312 |
| 5 | **CustomAllReduce** | `ca_comm.should_custom_ar`(§6.4) | :313-321 |
| 6 | **SymmMemCommunicator** | `should_use_symm_mem` | :322-326 |
| 7 | **PyNcclCommunicator** | 兜底主力,`pynccl_comm.all_reduce`;失败再退 `torch.distributed.all_reduce` | :327-341 |

> 除 TP 组外,其他组在构造时就把 2-6 全关了(`if "tp" not in unique_name` :50-55)——custom all-reduce 只对 TP 组有意义。
> 启动日志会打印实际启用的后端顺序(`cuda_communicator.py:266-273`),排查"为什么没走 custom all-reduce"先看这行。

`all_gather`(:355)与 `reduce_scatter`(:391)则主要委托 `pynccl_comm`,并在 NVLS 对称内存开启时改走 `_all_gather_symm_mem` / `_reduce_scatter_symm_mem`(:362/:407)。这些全部发生在 `GroupCoordinator._*_out_place`(:686-748)之下,所以**模型代码只认 `get_tp_group().all_reduce`**,具体用哪条路对上层透明。

---

## 8. 关键类 / 函数速查(文件:行号)

### `distributed/parallel_state.py`
- `GroupCoordinator`:380;`__init__`:409;`make_sibling_device_group`:535;`graph_capture`:619
- `first_rank`:585;`last_rank`:590;`is_first_rank`:595;`is_last_rank`:600;`next_rank`:605;`prev_rank`:612
- `all_reduce`:662;`_all_reduce_out_place`:686;`all_gather`:691;`_all_gather_out_place`:707;`all_gatherv`:712
- `reduce_scatter`:722;`_reduce_scatter_out_place`:745;`gather`:750;`broadcast`:766;`broadcast_object`:781;`broadcast_tensor_dict`:885
- `send_tensor_dict`:981;`recv_tensor_dict`:1076;`barrier`:1200;`send`:1209;`recv`:1216
- 模块级:`all_reduce`:152;`reduce_scatter`:164;`all_gather`:182;`get_world_group`:1293;`init_world_group`:1303;`init_model_parallel_group`:1315
- `get_tp_group`:1389;`get_pp_group`:1405;`get_dp_group`:1413;`get_ep_group`:1421
- 模块级 `graph_capture`:1450;`set_custom_all_reduce`:1483;`init_distributed_environment`:1586
- `initialize_model_parallel`:1751(TP :1843 / DCP :1861 / PCP :1884 / PP :1902 / DP :1919 / EP :1949)
- `ensure_model_parallel_initialized`:1997;`is_model_parallel_initialized`:2065;`destroy_model_parallel`:2084
- `in_the_same_node_as`:2175;`_node_count`:2323
- `_ENABLE_CUSTOM_ALL_REDUCE`:1480

### `distributed/communication_op.py`(模型层唯一入口)
- `tensor_model_parallel_all_reduce`:12;`tensor_model_parallel_all_gather`:17;`tensor_model_parallel_reduce_scatter`:24;`tensor_model_parallel_gather`:31;`broadcast_tensor_dict`:38

### `distributed/utils.py`
- `get_pp_indices`:127(`start_layer` :169 / `end_layer` :170);`split_tensor_along_last_dim`:99

### `distributed/device_communicators/pynccl.py`
- `PyNcclCommunicator`:60;`__init__`:61;`all_reduce`:166;`all_gather`:199;`all_gatherv`:222;`reduce_scatter`:257;`reduce_scatterv`:285;`send`:322;`recv`:349;`broadcast`:376;`destroy`:148

### `distributed/device_communicators/custom_all_reduce.py`
- `_can_p2p`:36;`CustomAllreduce`:56;`_SUPPORTED_WORLD_SIZES`:57;`__init__`:70
- `should_custom_ar`:348;`all_reduce`:363;`custom_all_reduce`:382
- `should_custom_all_gather`:400;`custom_all_gather`:424;`should_custom_reduce_scatter`:454;`custom_reduce_scatter`:477
- `create_shared_buffer`:521;`free_shared_buffer`:541

### `distributed/device_communicators/cuda_communicator.py`
- `CudaCommunicator`:29;`__init__`:30;`all_reduce`:275(优先级链 :278-341);`all_gather`:355;`reduce_scatter`:391;`reduce_scatterv`:418

### `model_executor/layers/linear.py`
- `LinearBase`:215;`ReplicatedLinear`:296
- `ColumnParallelLinear`:401;`__init__`:432;`weight_loader`:542;`forward`:569(gather_output :578)
- `MergedColumnParallelLinear`:639;`weight_loader`:724
- `QKVParallelLinear`:965;`__init__`:993(num_heads :1018 / num_kv_heads :1019-1024);`_get_shard_offset_mapping`:1054;`_get_shard_size_mapping`:1064;`_load_fused_module_from_checkpoint`:1072
- `RowParallelLinear`:1504;`__init__`:1539;`weight_loader`:1608;`forward`:1635(all-reduce :1653)

### `model_executor/models/llama.py`(PP 落点示例)
- `LlamaMLP`:80;`LlamaAttention`:123;`LlamaDecoderLayer`:249
- `LlamaModel`:345;`embed_tokens`(PP 首尾判断):374-383;`make_layers` 调用:384;`norm`(末 rank):389-392;`forward`:401(首 rank :409 / 中段 `IntermediateTensors` :431-434 / 末 rank norm :436)
- `LlamaForCausalLM`:447;`lm_head`+`logits_processor`(末 rank):485-500;`compute_logits`:529
- `hf_to_vllm_mapper`(q/k/v、gate/up 映射):346-355

### `model_executor/models/utils.py`
- `PPMissingLayer`:773;`make_layers`:786(占位 :810-816);`is_pp_missing_parameter`:843

### `model_executor/layers/vocab_parallel_embedding.py` 与 `logits_processor.py`
- `VocabParallelEmbedding`:198;`forward`:487(掩码+all-reduce :490-506)
- `ParallelLMHead`:521(forward 抛错 :579);`LogitsProcessor._gather_logits`:84

### `config/parallel.py`
- `ParallelConfig`:119;`pipeline_parallel_size`:122;`tensor_parallel_size`:124;`data_parallel_size`:129
- `world_size`:833(PP×TP×PCP);`world_size_across_dp`:549;`disable_custom_all_reduce`:205

### V1 运行时(P 的执行落点)
- `v1/worker/gpu/pp_utils.py:52 PPHandler`(采样广播:构造 :62 / `receive` :126 / `broadcast` :170)
- `v1/worker/gpu_worker.py:1098 irecv_tensor_dict`、`:1135 isend_tensor_dict`(PP 中间激活异步收发)
- `parallel_state.py:1019 isend_tensor_dict`、`:1114 irecv_tensor_dict`(异步收发原语)
- `v1/worker/gpu/model_runner.py:198/199 is_first_pp_rank/is_last_pp_rank`、`:440` 非首 rank 建 `intermediate_tensors` 持久 buffer

### 相关环境变量
- `VLLM_DISABLE_PYNCCL`(`pynccl.py:93`):关 PyNccl。
- `VLLM_SKIP_P2P_CHECK`(`custom_all_reduce.py:40`):跳过 P2P 实测,只信 driver。
- `VLLM_PP_LAYER_PARTITION`(`distributed/utils.py:143`):手工指定每段层数,如 `"8,8,8,8"`。
- `VLLM_ALLREDUCE_USE_FLASHINFER` / `VLLM_ALLREDUCE_USE_SYMM_MEM`(`cuda_communicator.py:60-61`):开/关对应 all-reduce 后端。
- `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`(`parallel_state.py:436`):切换 `split_group` 新路径(默认 False 走传统 `new_group`)。

---

## 延伸阅读

- vLLM 官方文档:分布式推理与 serving(TP/PP/DP 用法与限制)<https://docs.vllm.ai/en/latest/serving/distributed_serving.html>
- 论文:Megatron-LM —— 张量并行 + 流水线并行的 3D 并行划分 <https://arxiv.org/abs/1909.08053>
- 论文:Megatron-2(PipeDream 流水线调度的后续,PP 分段的动机)<https://arxiv.org/abs/2104.04473>
- NVIDIA NCCL 官方仓库(collective 语义、NVLink/PCIe 拓扑感知) <https://github.com/NVIDIA/nccl>
- 相关文档:本仓库 docs `01_system_architecture.md`(DP 时多 EngineCore 的拓扑)、`05_ModelRunner_CUDAGraph.md`(`graph_capture` 的调用方)、`03_Scheduler.md`(PP 异步调度如何借 `num_in_flight_tokens` 抹平流水线气泡)