# 第 11 章 · 华为昇腾与 MindSpore 全栈 · 参考答案

> 参考答案，建议先自己动手再看。对应 notebook：`exercises/ch11/71–100`，30 课。
> 章答案比 ch09/ch10 更简洁（每题 3–6 行思路），不逐行给完整代码；涉及真机/昇腾云的操作类练习给出“怎么做 + 预期结论”。代码假定 `import torch` 等按各课已定义变量对齐，需要时标注依赖。

---

## 第 71 课 · 华为 AI 生态全景：昇腾 + CANN + MindSpore + ModelArts

### 练习 1：把分层全景图的“应用层”再拆细，列出 5 个昇腾上跑的应用/推理引擎

- vLLM-Ascend、MindIE、MindSpore、ATB、MindFormers、ModelArts 推理服务、TensorRT-LLM（昇腾移植版）。应用层应细分为「推理引擎」「训练框架」「云服务」三类。

### 练习 2：给雷达图新增“开源生态”维度，给三大框架重新打分，说明昇腾最大短板

- 为 Ascend/CANN、MindSpore、PyTorch 各加一维（如菜谱/算子贡献者/issue 响应节奏），重新打分。预期：计算效率、算子覆盖接近；开源生态维度 MindSpore/CANN 明显低于 PyTorch 生态——社区规模与第三方算子数量是昇腾最大短板。

### 练习 3：查 Ascend 910B 与 A100/H100 的 FP16 算力与显存，做并排对比表（标年份）

- 表：维度 | 910B | A100 | H100（FP16 峰值 TFLOPS / 显存 / 显存带宽），标注资料年份与来源。预期 910B FP16 算力接近 A100、显存 64GB 同档但互联/软件生态差距明显。

### 练习 4：用 pyecharts 给生态组件图新增“边缘侧依赖度”第三个系列，重画柱状图

- 给每个组件加 `edge_deps`（如 Lite 高、MindSpore 中、ModelArts 低），用 `bar.add_yaxis("边缘侧依赖度", vals)` 成对显示，观察「云/边」组件定位差异。

---

## 第 72 课 · DaVinci 架构

### 练习 1：把 matmul_tiled 的 tile 改成 8/32，对比结果精度与浮点误差

- 思路：tile 越小分块越多、累加顺序变化越多，浮点误差随 tile 调整而小幅波动；tile=8 时块数与边界处理更多。对拍参考结果，误差量级仍 ~1e-6，但 tile 大小会影响访存效率。

### 练习 2：画一张 32 个 AI Core 共享 L2 与 HBM 的拓扑图

- 思路：画 4×8 网格的 AI Core，各自带局部 AI Memory（L0），分组共享 L2/low-bandwidth 缓存区，全部经 NoC 连到共享 HBM。标注「AI Core→L2→HBM」层级与带宽递减。

### 练习 3：查 Cube 单元处理 BF16 与 FP16 算力是否翻倍，解释低精度“一鱼两吃”

- 思路：低精度（BF16/FP16）在相同晶体管宽度下可做两倍 INT8/Cube 运算，半精度算力常为 FP32 的 2 倍。因降精度牺牲尾数精度换取吞吐，故在训练/推理低精度阶段“一个核心当两个用”——即一鱼两吃。

### 练习 4：用 time 对比 n=256 时分块与不分组 GEMM 的耗时，分析瓶颈

- 思路：n=256 时数据量小、访存几乎全命中，分块 vs 不分块速度差异小，瓶颈在启动/调度而非搬运。用比 `n256` 大得多的规模（如 4096）再测，辨析计算 vs 访存哪个主导。

---

## 第 73 课 · CANN 软件栈

### 练习 1：给 acl_inference_pipeline 增加 aclrtCreateStream / aclrtSynchronizeStream 两步，重跑打印

- 思路：在初始化与推理之间 `aclrtCreateStream(&stream)`，推理操作绑定该 stream，结束 `aclrtSynchronizeStream(stream)`。预期打印多了 stream 句柄、推理调用显式异步化，正确同步后结果仍一致。

### 练习 2：把 GE 账本的 iters 从 100 改成 1 和 1000，观察哪种模式逐算子启动反而更快

- 思路：`GE(graph mode)` 需要一次性构图/编译（固定大开销），但逐算子执行开销小；改写后：`iters=1` 时构图开销摊不开，逐算子模式反而快；`iters=1000` 时构图成本被摊薄，GE 图执行显著反超。即存在「构图盈亏平衡点」。

### 练习 3：在双栈对照图里新增“多租户/虚拟化”行

- 表格加一行：CANN 组件 = AscendCL 的多芯/资源切分（`aclrtSetDevice`、华为容器 NPU 隔离），CUDA 对应 = MIG / 时间切片 / CUDA Device 共享。二者都是“把算力按租户/任务隔离”的机制。

### 练习 4：查 CANN 文档，总结 ACL 与 C 语言 API 的命名规律

- 思路：查前缀分类：`aclmdl`（模型加载/执行）、`aclrt`（运行时 Runtime：流/事件/内存/设备）、`aclfv`（视频/图片编解码）、`aclblas`（BLAS 算子）、`aclnn`（算子接口）。总结：前缀 = 能力域，后缀动词 = 动作（Create/Destroy/Set/Get/Sync）。

---

## 第 74 课 · Ascend C

### 练习 1：把 vec_add_pipelined 改成“向量加再乘 2”（z=(a+b)*2），对比两次操作 vs 一次指令

- 思路：一个内核里先 add 再 mul，Vector 单元连续两次向量指令；与“拼成一次复合指令/预乘”对比。预期分两条向量指令耗时略高，但比拆成两个 kernel（两次搬移）快得多——体现“一个 kernel 内多指令”的价值。

### 练习 2：给数据流图加一条“流水重叠”标注

- 思路：在图里标「第 2 块 Load / 第 1 块 Compute / 第 0 块 Store」并行重叠段，用箭头/泳道标出三者同时进行。预期示意 double-buffer 下 Load(2)∥Compute(1)∥Store(0)，吞吐接近纯计算极限。

### 练习 3：用 torch 模拟多块流水重叠：提前预取下一块，统计加速比（体会 double-buffer）

- 思路：把分块计算写成“预取下一块再算当前块”的双缓冲循环，与“算完才取下一块”的朴素循环对拍耗时。预期双缓冲省去每次「等搬运→再算」的串行等待，加速比随块数增加趋于明显（模拟 1.5x–2x）。

### 练习 4：查阅 sync_all / set_flag / wait_flag 语义，画双缓冲同步时序图

- 思路：数据搬运在 `set_flag` 置位、计算侧 `wait_flag` 等齐后消费，`sync_all` 全场同步。时序图：Compute 核 wait 第 1 块 flag → 算 → set 消费 flag；Load 核搬第 2 块 → set 满 flag…… 两个 flag 交替乒乓，构成双缓冲握手。

---

## 第 75 课 · MindSpore 张量

### 练习 1：用 torch 手写三层线性网络 + loss，打印每层权重梯度，体会 autograd 的自动性

- 思路：`nn.Sequential(Linear(4,64),Linear(64,32),Linear(32,2))` 前向后 `loss.backward()`，打印 `list(param.grad)`。预期每层 `weight.grad` 自动累积非零，免手推反向——对照手工实现可见其开销由框架接管。

### 练习 2：把计算图改成 f=sin(x)+x²，重画 DAG 并标出反向路径导函数

- 思路：DAG：`x → sin(x)` 分支（导 cos(x)）与 `x → x²` 分支（导 2x），`+` 汇合（导分流按加法链式）。反向路径标 `df/dx = cos(x) + 2x`，体现乘法链式+加法分流。

### 练习 3：在映射表里补充 5 条熟悉的 API

- 思路：补 `MaxPool/CrossEntropyLoss/Adam/cat/stack` 的 MindSpore↔PyTorch 对照（如 `mindspore.ops.AdaptiveAvgPool` vs `torch.nn.MaxPool2d`），逐一验证等价。

### 练习 4：对比 mindspore.Tensor 与 torch.tensor 的默认设备，说明 MindSpore 为什么默认 NPU

- 思路：torch.tensor 默认 CPU、控 `to('cuda')`；MindSpore 建张量可指定 `device_target`，默认联动 `ASCEND` 之类。原因：MindSpore 定位昇腾硬件栈、默认 NPU 免显式搬运，直接面向昇腾推理生态。

---

## 第 76 课 · MindSpore 计算图与静态图模式

### 练习 1：给 TinyMLP 加一个 dropout 层再 symbolic_trace，观察图节点变化

- 思路：fx 对含随机 `ops.dropout` 需要把随机 mask 抽出；加后图节点增多（rand/mask/mul 系列）。预期 trace 后多出随机化节点，且需处理其非确定性写法才能单跑通过。

### 练习 2：把第 4 节曲线里 compile_ms 改成 200，重算交叉点，说明“编译成本高时该不该用静态图”

- 思路：编译成本从 xx 提到 200ms 后，图模式（graph）盈亏平衡的 `iters` 变大——只有运行次数超过更大阈值才划算。预期：少量调用该用动态图，海量复用/长会话才值得静态图。

### 练习 3：用 fx.GraphModule 修改图（如把 relu 换成 gelu），重跑推理验证“改图不改代码”

- 思路：`for n in gm.graph.nodes: if n.target is F.relu: n.target = F.gelu`，`gm.graph.lint(); gm.recompile()`，再跑前向。预期模型行为随图改而改变、原代码未动——体现计算图的可编程性。

### 练习 4：对比 MindIR 与 ONNX 的导出流程差异，各写 3 条要点

- 思路：MindIR：`mindspore.export` 统一产图、面向训练→推理全链路、多端复用但生态封闭；ONNX：`torch.onnx.export` 走 fx/torchscript 导出、第三方工具链丰富、算子方言多。要点围绕“导出入口/图特化/生态开放性”三点对比。

---

## 第 77 课 · 动态图 vs 静态图

### 练习 1：把 crossover 参数改成动态图每轮 0.5ms、静态图 0.01ms、编译 200ms，重算交叉点

- 思路：设动态图总耗时 `0.5×n`、静态图 `200 + 0.01×n`，解 `0.5n = 200+0.01n` → 交叉点约 `n≈408` 轮。静态图每轮快 50 倍但需 200ms 一次性编译，长期重复调用才更省。结论随相对速度改变交叉点右移/左移。

### 练习 2：用 torch.fx 打印一个带 if-else 的函数，观察分支如何体现在图中

- 思路：fx 对 Python `if/else` 在 trace 时按运行时真值只保留其中一支；否则需 `torch._dynamo` 的 guards/子图。预期看到图内容会随输入分支不同，体会静态图对动态控制流的「按输入特化子图」处理。

### 练习 3：给时间线补 2021 JAX jit 普及、2022 Dynamo 原型发布

- 思路：在时间线对应年份加节点：`2021 JAX jit / 2022 torch._dynamo 原型`，说明动态图→静态图加速的工程趋势（Dynamo 面向 PyTorch 默认图）。

### 练习 4：查 vLLM 文档中 CUDA Graph 的开启条件，写 3 条要点

- 思路：① CUDA Graph 需固定/静态 shape（序列长度与 batch padding 到固定档才可复用）；② 受 `max_num_seqs`/`max_model_len` 限制分配的捕获图池；③ 开启后首段捕获有开销，需预热，换取复用时的启动开销大幅下降。

---

## 第 78 课 · MindSpore 算子开发

### 练习 1：把 SwishFn 扩展成 swish + scale（乘一个常数），重做数值梯度校验

- 思路：前向 `y = swish(x)*scale`，解析梯度和 `scale` 与 swish 梯度运算结合；数值梯度（中心差分）与解析梯度在同一 scale 下对拍，`atol≈1e-5` 通过即正确。

### 练习 2：自定义 relu6 算子，与内置 relu6 对比结果

- 思路：`y = minmax(x, 0, 6)`（`tl.clamp` / torch `clamp_(0,6)`）；自定义实现与 `torch.nn.functional.hardtanh(x,0,6)` 或 MindSpore 内置 `relu6` 对拍，输出完全一致即体现 Primitive 通用套路（定义 func + grad）。

### 练习 3：把第 5 节融合例子加一个 bias 相加，对比 4 算子分离与 1 表达式融合的耗时

- 思路：融合表达式 `y = (x*w+b) 全链一个 kernel`，分离版每算子一次读写；加 bias 后融合仍 1 次、分离多 1 次。内存受限链上融合版本快若干倍，加 bias 拉大差距但量级不变。

### 练习 4：查 register_custom_op 流程，与 torch.autograd.Function 各写 3 条异同

- 思路：同：都需前向+定义反向/梯度、都用于自定义算子、都参与梯度图。异：MindSpore 需注册到算子库/prim 并在 `graph` 模式可用；torch.autograd 是纯 Python 类常驻 eager。围绕「注册入口、图/梯度融合方式、前端语言」对比。

---

## 第 79 课 · 图算融合：GE 如何把 N 个算子并成 1 个 kernel

### 练习 1：把 chain 改成 y = relu(x·w+b)，用 torch.nn.functional 对比融合与分离耗时

- 思路：融合 `torch.nn.functional.linear + relu`（`F.relu(F.linear(x,w,b))` 或 torch.compile），分离版 `x@w+b → relu`；融合省一次中间 HBM 读写，memory-bound 场景加速明显。

### 练习 2：画“融合收益 vs 算子个数”柱状图（横轴 n=2..10，纵轴加速比）

- 思路：n=2..10 各生成整条逐元素链，分「分离逐算子」与「融合 1 kernel」计时，`加速比 = t_sep / t_fusion`；柱状图显示加速比随 n 上升后趋平（省读写收益封顶）。

### 练习 3：查 vLLM 对 GEMM+act 融合的实现（PagedAttention 的 fused kernels），写 3 条要点

- 思路：① PagedAttention 把 QK^T/softmax/PV 融进一个 kernel、K/V 按块加载省搬运；② GEMM 输出直接接激活（如 GELU）写在 epilogue 省一次读写；③ 用 Triton/FlashAttention 模板实现、由启动参数驱动块切分。

### 练习 4：思考：为什么逐元素链最易融合，而 GEMM+softmax 这类“跨语义”算子融合更难

- 思路：逐元素链无归约、数据天然同 layout、极易向量化融合；跨语义（GEMM→softmax）涉及张量核产出的 layout 转换、归约跨行、精度更高的动态范围处理，融合需处理布局/归约/数值变化，复杂度陡增、收益却未必覆盖成本。

---

## 第 80 课 · MindSpore 分布式训练

### 练习 1：把 ring_comm 的 N 换成 128，对比每卡通信与朴素 AllReduce，体会 Ring 优势

- 思路：Ring 每卡 `2(N-1)/N * S` 通信量（N 越大越接近 2S），朴素 tree 每卡约 `2S`；N=128 时 Ring 接近朴素的一半。重跑并打印每卡累计通信量验证。

### 练习 2：用 matplotlib 画 7B/70B/405B 在 DP 下每卡显存随卡数的曲线，标注 16GB 卡能装哪几档

- 思路：`per_card = model_size_gb / num_gpus` 画对数曲线，水平线标 16GB；7B 卡上少数卡即可装、70B 需多卡、405B 即使 8 卡也超 16GB（需张量并行/量化）。标出各模型“可装”的卡数区间。

### 练习 3：查 MindSpore auto_parallel 的搜索维度（按维度切、recompute 等），写 3 条要点

- 思路：① 按张量各维切分策略、② 策略候选靠 `strategy` 与 cost model 组合搜、③ `recompute` 用重计算换显存、降低压不下时的重切需求。围绕「切分动作、搜索空间、显存-通信折衷」三点。

### 练习 4：结合第 35 课，把 vLLM 的 TP/PP/EP 与本章 MP/PP 做跨章对照表

- 思路：表列 维度 | vLLM(TP/PP/EP) | MindSpore(MP/PP)；行：切分对象（层维/算子维/专家维）、通信模式（ring AllReduce / P2P send-recv）、适用（TP 单层并、PP 分层流水、EP 分专家）。强调 EP≈“把专家维度切开、路由后各卡只算自己专家”。

---

## 第 81 课 · MindSpore 推理部署

### 练习 1：把 MiniLM 换成 2 层 Transformer，用 torch.jit.script 导出，观察图结构差异

- 思路：`torch.jit.script` 走 source-level 记录完整控制流（if/循环被具成图），比 `script` 差于 `trace` 的动态行为捕获，但能保留更多分支逻辑。导出后 `.code`/graph 能看到自注意力与 FFN 子图。

### 练习 2：给 MiniLM 加 batch=16 输入，估算并对比 prefill 与 decode 的 FLOPs

- 思路：prefill `≈ 2 × tokens × params`（并行处理整段）、decode 每 token `≈ 2 × params`（串行）；batch=16 时 prefill 一次处理 16 段、decode 每步出 16 token。算在同 seq 长度下 prefill 总量远大于 decode 单步，指导 throughput/时延取舍。

### 练习 3：搜索 mindspore.export 参数（file_format/input_names），写导出带动态轴 MindIR 的伪代码

- 思路：`ms.export(net, input1, input2, file_format="MINDIR", input_names=["x"], dynamic_shape=True)` 配合 `ms.Tensor` 声明动态轴，或用 `expand_dims` 指定动态范围；伪代码演示 `input_names + file_format + dynamic` 三参数，产出可多 batch 推理的 MindIR。

### 练习 4：对比 torch.jit 与 MindIR，说明 MindIR 为何强调“一次导出、多端复用”

- 思路：MindIR 是昇腾全栈统一中间表示，一次导出可在 MindSpore Lite/服务器/云侧多端通用，且携带算子级可优化信息；torch.jit 偏 PyTorch 局部。至少 2 理由：① 跨硬件的统一执行体、② 内置推理友好的算子优化入口，避免每端各导一次。

---

## 第 82 课 · MindIR 与模型转换

### 练习 1：给迷你 IR 增加一个 softmax 节点类型，并把注意力图的 JSON 描述跑通解释器

- 思路：给解释器注册 `"softmax"` 节点并实现 `y = softmax(x, dim)`；在注意力图 JSON 的 QK^T→V 之间插入 `"type":"softmax"` 节点，重跑解释器输出与手算 softmax 一致。

### 练习 2：把 run_graph 改成“先拓扑排序再执行”，验证有依赖的节点不会先于输入执行

- 思路：`tmp = dict(indeg)`，不断取 `indeg=0` 节点执行并减其出边入度（Kahn），保证每个节点在其所有输入产出后才运行；用一张 `a→b→c` 依赖图验证 b 不会先于 a。

### 练习 3：调研 ms.export 的 file_format 参数还能导出什么格式

- 思路：查文档：除 `MINDIR / ONNX` 外支持 `MindIR(Graph)`/`ckpt`/可选目标 `ASCEND` 的 `mindir` 变体。可借助 `file_format` 表梳理导出目标矩阵。

### 练习 4：对比 torch.jit.trace 与 script：哪个保留控制流更多？与 MindIR 子图机制异同

- 思路：`torch.jit.script` 保留的源级控制流（if 循环）比 `trace`（仅记录运行时一次路径）多；MindIR 的子图机制用独立子图承载动态/条件逻辑，与 script 的 `if` 组织类似，但 MindIR 面向昇腾多端优化而非仅 torch 图。

---

## 第 83 课 · MindSpore Lite 端侧推理

### 练习 1：把 symmetric_quantize 改成非对称量化（min/max 映射），对比误差差异

- 思路：对称 `scale = max_abs/127`，非对称再算 `zero_point`（`-min*(scale[...])`）——`q = round(x/scale)+zp`；对偏置分布明显的数据非对称误差更小，打印两种 `quant_mae` 对比。

### 练习 2：对不同 σ 的高斯权重做 INT8，画“σ vs 平均误差”曲线

- 思路：生成 σ∈(0.1..2) 的高斯 W，量化返回 `avg_err`；画 σ-x 曲线。预期 σ 越小分布越集中、落在像元附近误差越小；σ 大时离群值撑大 scale、均匀 quant 段误差上升。

### 练习 3：给 torch 的 MLP 前向加逐层 PTQ 校准循环，比较全局 vs 逐层 scale 误差

- 思路：校准一批样本统计每层激活 min/max，得到每层单独 scale（逐层）与整网单一范围（全局）；逐层 scale 因匹配各层分布误差更小。以 `mae`/`cos` 对比二者。

### 练习 4：调研手机 NPU（高通/苹果）与昇腾 310P 的 INT8 算力，讨论端侧 LLM 可行性

- 思路：列出 INT8 TOPS（如骁龙 Gen3 NPU、Apple Neural Engine ~35 TOPS、310P ~几十 TOPS）并换算：端侧把 7B 压到 ~3–4GB INT8 后，吞吐受算力与带宽限制、主要适合小模型/量化+蒸馏。结论：可行但受显存与 INT8 算力约束，端侧 LLM 以小模型+剪枝量化为主。

---

## 第 84 课 · 昇腾推理引擎 vs vLLM

### 练习 1：用 torch 模拟“两个请求共享物理块”的 block table，验证共享 KV 下 attention 结果不变

- 思路：构造 block table 使 req2 的前缀块复用 req1 的物理块（`bt[1] = [0,1] * 前两页 + 新块`），同一物理 KV 被两次读取；手写 decode attention 对比共享前后结果——共享只省显存不影响数学。

### 练习 2：调研 vllm-ascend 的 support matrix：哪些算子走 torch_npu、哪些走 CANN 融合算子

- 思路：查 support matrix：基础点乘/激活走 `torch_npu` 原生，注意力/PagedAttention、KV cache 读写走 CANN 融合算子（`npu_paged_attention` 等）。分类归纳「servable 公共算子 vs 昇腾专用融合」。

### 练习 3：对比 MindIE 与 TensorRT-LLM 架构，找 3 同 3 异

- 思路：同：① 都做编译期图优化+运行期引擎、② 都支持 KV cache/连续批处理、③ 都有量化/多级并行。异：① MindIE 绑定 CANN/昇腾、TensorRT-LLM 绑定 CUDA/NVIDIA、② MindIE 强耦合 MindSpore 生态、TensorRT-LLM 独立插件体系、③ 公开文档与社区规模差异。围绕「硬件耦合、生态定位、引擎复用性」展开。

### 练习 4：把第 45 课 paged_attention 函数改成“昇腾风格”：单算子封装（输入 block_table，输出 result）

- 思路：封装 `def npu_paged_attention(query, key_cache, value_cache, block_table, head_size)`：内部按 block_table 定位每个请求的页偏移、片上循环页加载 K/V、online softmax 累加，返回单一 result 张量——把“外部循环+offset 管理”收敛进算子内部。

---

## 第 85 课 · PagedAttention 在昇腾

### 练习 1：给 ascend_paged_attention 增加“块内 mask（只取前 k 个 token）”逻辑，模拟已生成部分

- 思路：打分后按 `token_idx < k` 生成块内掩码，`scores = tl.where(mask, scores, -inf)` 或填充 `other`，再归一；模拟“只关注已生成的前 k 个位置”。对拍期望序列一致。

### 练习 2：画“块表长度 vs 块大小”与碎片率同图的双轴曲线

- 思路：分块大小（如 16/32/64/128）→ 计算平均每序列所需块的“空闲格占比=碎片率”与平均块表项数；用双 y 轴（左块表长度、右碎片率）同画。预期块越大碎片率越低但浪费单块内空间上升，存在甜点。

### 练习 3：把 req_blocks 改成 lognormal 分布，观察碎片率曲线形状变化

- 思路：用 `np.random.lognormal` 生成长度分布再算碎片率。预期长尾分布比均匀分布碎片率更高、曲线整体上移，说明请求长度差异大时固定块策略碎片更明显（促发变长块/动态块机制）。

### 练习 4：调研 torch_npu 的 paged attention 接口，列出它与 CUDA 版参数异同

- 思路：查 `torch_npu.npu_paged_attention` 等参数：输入有 `cache_k/cache_v`、`block_table`、`seq_len`、`head` 等，与 CUDA PagedAttention 大多对应；异在昇腾接口把设备/流（`npu_stream`）与矩阵布局/尾处理参数做昇腾式定制。列出「同字段(paged/KV/attention) vs 昇腾专属偏移与 stream」两列。

---

## 第 86 课 · CANN 算子库与融合

### 练习 1：把 unfused_attn 的中间落地次数改成 3（QK^T、Softmax、P×V 各一次），重算对比

- 思路：naive 注意力把 QK^T、softmax、P×V 分别落盘 HBM=3 次中间读写，与 Flash 的 0 次对比；用 FLOPs/读写模型估算 naive 的额外带宽成本。预期 flash 相比 naive 在高 N 时省 O(N²) 级访存，显著提速。

### 练习 2：写一段 Conv→BN→ReLU 的“逐算子”与“融合” torch 代码，统计数值差异

- 思路：`y = F.relu(norm(conv(x)))`（逐算子）vs `torch.compile`/预融合 `relu(conv)`；对拍 `(融合-分离).abs().max()≈1e-5`。以耗时对比，融合省中间 tensor HBM 往返。

### 练习 3：调研 AscendCL 的 aclrtLaunch / aclrtSynchronizeStream 与 CUDA 流 API 对应

- 思路：对应 `cudaLaunchKernel` 与 `cudaStreamSynchronize`/`cudaDeviceSynchronize`；AscendCL 用 kernel 句柄 + stream 异步执行，语义与 CUDA stream 一致（提交-执行-同步），仅命名域 `aclrt` vs `cuda`。

### 练习 4：对比 Triton kernel 与 Ascend C kernel 的思维模型，写一段二者对照的伪代码

- 思路：Triton=`@triton.jit` + 块级 `tl.load/dot`（SPMD 声明式）；Ascend C=`__global__` 用 `Vector/Matrix` 单元显式搬进 Local 再算（命令式管搬移）。伪代码并排：Triton 写“声明一块怎么算”，Ascend C 写“把数据搬进 local→张量化指令→搬回 GM”。

---

## 第 87 课 · 华为量化方案

### 练习 1：给 quantize_chunk 实现非对称量化（scale+zero_point），与对称版对比误差

- 思路：对称 `scale=max_abs/127`（无 zp）；非对称 `scale=(max-min)/255, zp=round(-min/scale)`，`q=round(x/scale)+zp`。对偏置（非零偏）分布非对称误差更小，打印两版 `mae`。

### 练习 2：把 quant_mae 逐通道粒度从 32 段改到 64/128 段，画误差随段数曲线

- 思路：把逐通道/逐段量化从 32 段扫到 64/128，`avg_err = avg(mae)` 画 段数-误差。预期段数越多 per-channel 越细、误差越小后趋平（段数不足瓶颈），但代码量/复杂度上升。

### 练习 3：用 AWQ 思路（按激活幅度保护重要权重）重写量化，比较与普通量化误差

- 思路：AWQ 先根据激活统计找重要权重的每通道 scale 放大、再量化并复原缩放系数；普通量化不区分重要性。重写后对比 `mae`/`task_metric`，预期 AWQ 在低 bit 下误差更小。

### 练习 4：调研 MindIE 支持的低比特类型列表，与 vLLM quantization 选项逐一对照

- 思路：查 MindIE/昇腾低 bit：W8A8、W4A16、FP8/W8A8 等；vLLM 支持 `awq`/`gptq`/`squeezellm`/`fp8` 等。列对照表：昇腾低 bit ↔ vLLM 对应方法，标注哪类是 CANN 原生、哪类需量化算子支持。

---

## 第 88 课 · 昇腾大模型推理

### 练习 1：把 ContinuousBatch 加一个“等待队列”，槽位空出自动补位，打印占用曲线

- 思路：维护 `running` 与 `queue`；某序列完成即释放槽位，从队列头部 pop 新请求补位；逐 step 记录 `running` 数画占用曲线，体现连续批处理“槽位利用率满、吞吐高”。

### 练习 2：估算 70B、32 层、8 头在并发 256、序列 4096 时的 KV 显存，验证是否爆卡

- 思路：`KV_bytes = num_layers × 2 × seq × bootshead × dtype_bytes` 累加所有请求；并发 256×4096 的 KV 单独可能 60–120GB+，叠加 70B 权重（fp16 ≈ 140GB）与激活后必然爆单卡，需 TP+KV 量化或降并发。给出公式与量级判断。

### 练习 3：给 EP 画“通信量随专家数变化”曲线（专家越多单跳越小但 hops 越多）

- 思路：`每 token 跨卡通信 ≈ 路由量 × (hops)`，hops 随专家数增加、单跳 payload 随专家数下降，画两条线及其乘积式总通信曲线——存在最优专家数（甜点）。

### 练习 4：调研 DeepSeek MoE 的 EP 实践与 vllm-ascend 的 EP 支持，对比实现要点

- 思路：DeepSeek MoE 把 expert 切到多卡 EP、路由后每卡只算自己专家、用 All2All 交换 token；vllm-ascend 的 EP 在昇腾上把 expert 分到 NPU 并用 HCCL/自定义通信。对比：路由矩阵分发、all-to-all 实现（HCCL vs NCCL）、与 TP 的配合。

---

## 第 89 课 · MindSpore 自动并行

### 练习 1：实现“沿 K 切”的 2 路 GEMM：每卡算一部分 K，再用 torch.sum 模拟 AllReduce 验证结果

- 思路：`Ka = K//2`，卡 0 算 `A[:, :Ka]@B[:Ka, :]`、卡 1 算 `A[:, Ka:]@B[Ka:, :]`，`out = torch.sum(torch.stack([p0, p1]), 0)`（AllReduce 求和）；与整矩阵 `A@B` 对拍一致即验证 K 切分正确。

### 练习 2：给分块 GEMM 加一个 batch 维度，模拟 4 卡 DP+TP 组合切分

- 思路：DP 切 batch、TP 切 K（或 M），4 卡 = `[batch 2][K 2]`；每卡对各自 batch 子集做 K 分工 GEMM，DP 之后 no reduce、TP 之内 reduce。打印每卡分工与合并结果与全局一致。

### 练习 3：调研 mindspore.set_auto_parallel_context 的 search_mode 参数

- 思路：查文档：`search_mode` 取 `recursive_programming / dynamic_programming / sharding_propagation`（递归/动态规划/切分传播）等，控制自动并行搜索策略与搜索深度。

### 练习 4：对比 vLLM 的 TP 实现（vllm.distributed）与 MindSpore 自动并行：各自在哪一层切

- 思路：vLLM TP 在**推理图/算子层**显式做 `reduce`/列切（`column_parallel_linear` 风格，运行期切权重）；MindSpore 自动并行走**编译期构图+代价模型**搜索最优切分策略。前者手动在代码层、后者声明式自动在编译器层。

---

## 第 90 课 · 华为全栈 AI 框架演进

### 练习 1：把时间线图改造成横向分组甘特图，标 CANN 各版本发布时间

- 思路：用 `matplotlib`/`plotly` 把 CANN 5.0/6.0/7.0 等版本以横向 bar 按发布时间分段，x 轴年份；每 bar 标版本号与主特性（融合/LM 支持等）。

### 练习 2：调研昇腾 910C 的公开算力与显存，把估的爬坡曲线改成真实数据

- 思路：查公开规格（昇腾 910C FP16 TFLOPS、HBM 容量），把第 4 节“算力爬坡”估算图替换为真实点，标注数据来源年份，观察实际爬坡斜率。

### 练习 3：画一张“vLLM 术语 ↔ 昇腾术语”对照表

- 思路：表：`PagedAttention ↔ NPU PNA/npu_paged_attention`、`NCCL ↔ HCCL`、`CUDA Graph ↔ GE/图下沉`、`tensor parallel ↔ MP/TP(aclnn)`、`vLLM worker ↔ Ascend worker(npu)`、`KV cache ↔ 物理块 cache buffer`。逐行写出对应关系。

### 练习 4：写 500 字总结：昇腾生态对开源社区最大的贡献是什么？为什么

- 思路（要点）：昇腾最大贡献是**把“类 CUDA 的软硬件栈”在非 NVIDIA 上体系化开源**（CANN/Ascend C/MindSpore + vllm-ascend/torch_npu），理由：① 提供第二套不绑 NVIDIA 的高性能推理路径、促进双栈竞争与去单一化；② Ascend C 开源示算子编译/搬移模式、MindSpore 揭示从算子到图的国产化范式。展望其短板在社区规模与兼容面。

---

## 第 91 课 · 昇腾上跑 LLaMA：从 PyTorch 到昇腾的模型迁移

### 练习 1：给 MiniLLaMA 增加一个 LayerNorm 权重，补一条映射表规则，重跑权重转换

- 思路：`state = {".attention_norm.weight": "model.layers.{i}.input_layernorm.weight", ...}` 追加 `ln` 权重键；转换后用余弦/对拍确认新增层加载正确、其余权重未破坏。

### 练习 2：把模型换 fp16 后重算余弦相似度，试更大输入误差是否放大

- 思路：`model.half()` 后原权重余弦≈1；换更大输入/更长序列后用 `cos_sim` 对比 fp16 vs fp32 前向，误差在小规模 1e-3 量级、规模变大时累积放大。验证 fp16 在昇腾推理的可接受范围。

### 练习 3：用 torch.fx 列出模型全部算子，对照昇腾算子清单找出真正缺失的

- 思路：`fx.symbolic_trace(model)` 收集 `node.target` 集合，与 CANN 算子清单（查文档）取差集，标出缺的（如某些高级融合/自定义 op）；缺的算子需自定义 Ascend C 或降级到通用算子实现。

### 练习 4：查 vLLM-Ascend 支持 Qwen2-7B 所需 CANN 版本

- 思路：查 vllm-ascend support matrix：Qwen2-7B 通常在 CANN XX 版本 + torch_npu 对应版本可跑；写出版本号组合并说明满足的前置条件（CANN 版本 / 内核 / 量化选项）。

---

## 第 92 课 · Ascend C 高性能算子实战

### 练习 1：把第 2 节 N 改成 2_000_000，重跑三档对比，观察矢量并行加速比

- 思路：同一算法跑标量/矢量/矢量并行三档，N 更大时搬运/计算并行度更高。预期矢量并行相对标量的加速比随 N 增大而变大（更多块可供重叠），矢量未并行档居中。

### 练习 2：自己造一个“三缓冲”版本，画甘特图，看它与双缓冲差距

- 思路：三缓冲把同一 Cycle 内可重叠的段再细分，Load(i)/Compute(i-1)/Store(i-2) 各占一格；甘特图显示三段并行，空闲周期比双缓冲更少。预期三缓冲加速比相对双缓冲进一步收窄到接近纯计算极限。

### 练习 3：模拟一个 Elementwise 乘加算子（ax+b），重复三档优化并记录

- 思路：`y = a*x + b` 同样走标量→矢量→流水三档；记录各档耗时，与课程 add 版本对比，验证“访存受限逐元素算子优化套路可复用”。

### 练习 4：在真机/HiDevLab 上跑一次 msprof，量自定义算子搬移/计算占比

- 思路：`msprof --output=... python run_op.py` 导出 profiling；在 timeline 里找到自定义算子，读它 GM→local 搬移与 Vector/Matrix 计算时间占比，判断瓶颈在搬还是算，据此决定是否加大块/预取。

---

## 第 93 课 · ModelArts 与昇腾云

### 练习 1：改第 4 节随机种子与时长分布，再跑一次作业生命周期，观察排队时长波动

- 思路：把请求到达与执行时长的 `seed`/分布改动，重跑模拟；打印各作业 waiting/execution 时长，观察排队方差随分布（如泊松到达+偏斜时长）变化，体会云作业排队波动。

### 练习 2：给成本模型加一个“裸金属”档，把第 6 节三条曲线画全并比较拐点

- 思路：在实例/Serverless/NPU 共享之外加裸金属（固定租用价），画“耗时成本 vs 使用时长”三条线；裸金属有最低时长门槛（固定月费）拐点偏右，适合长稳任务，共享实例适合短任务，比较三线交叉点。

### 练习 3：模拟一次“失败重试”：RUNNING 后随机失败、自动重启，更新状态机并画时间线

- 思路：作业状态机 `QUEUED→RUNNING→(FAILED|SUCCESS)`，在 RUNNING 加随机失败分支；失败后重回 QUEUED 并 `retry++`，达上限则 FAILED。画时间线显示 `RUNNING→FAILED→QUEUED→RUNNING` 重试段。

### 练习 4：查 ModelArts 官网计费页，更新本课示意单价为当前真实价格

- 思路：访问 ModelArts 计费页，抄当前规格单价（如 NPU 实例 xx 元/时、存储 xx 元/GB 时），替换课内示意单价，重算成本曲线并标注“查价日期”。

---

## 第 94 课 · MindSpore ↔ PyTorch 迁移

### 练习 1：把 MsStyleMLP 改成三层，逐层验证与 torch 版本输出一致

- 思路：Ms 与 torch 各建 `Linear(→64→32→out)`，同权重初始化后逐层 `cos_sim`/`allclose(atol=1e-5)` 对拍；遇 API 差异（如 `reshape` vs `view`、init 方式）按映射修正，逐层确认。

### 练习 2：把第 4 节训练循环换成 Adam，对比两条损失曲线仍一致

- 思路：torch 用 `torch.optim.Adam`、MindSpore 用 `mindspore.nn.Adam`，同 lr 跑多步打印 `loss`；预期两条曲线几乎重合（同算法同参数），验证标识差异主要在 API 名称而非优化器语义。

### 练习 3：为第 5 节增加一个 Embedding 权重的映射规则，处理 padding_idx 特殊性

- 思路：`nn.Embedding(vocab, d, padding_idx=0)` 的映射注意 MindSpore 的 `nn.Embedding` 也有 `padding_idx` 概念；迁移时把 `padding_idx` 对齐，并确认 pad 行权重梯度不被更新（两框架行为一致）。补进映射表。

### 练习 4：读 MindSpore 迁移指南“动态 shape”一节，总结 3 种规避方案

- 思路：3 种：① 尽量固定 batch/序列、用静态 shape；② 动态范围 `ops.g`/`mindspore.ms` 声明 shape 区间供编译器预留；③ 用 `set_fixed_input`/图模式特化多档 shape、各自编译。各写一句话适用场景 → 落入规避表。

---

## 第 95 课 · 昇腾图引擎 GE

### 练习 1：给第 4 节 net 再加一个 dropout，重画图并统计节点数变化

- 思路：加 `nn.Dropout` 后图节点数上升（随机掩码/scale 节点）；`graph.print()` 统计 `get_op()` 数变化，说明结构化算子对图的展开。

### 练习 2：把第 5 节启动开销模型改成“计算占比 70%”，观察交叉点变化

- 思路：原模型假设启动开销占比固定，改计算 70%（启动 30%）后，图执行优于逐算子的交叉点（运行段数）提前——计算占比越高，图形化+少启动的收益越早体现。重画两条线。

### 练习 3：用 torch.compile 编译第 4 节 net，对比它与手工融合版的算子数（第 66/67 课方法）

- 思路：`torch.compile(net, mode="reduce-overhead")` 后打印生成 kernel（`get_triton_code`）计数，与手工（第 67 课把逐元素链并 1 kernel）对比；双向印证「编译器自动融合 ≈ 手工融合」的算子数，measure 融合收益。

### 练习 4：读 vLLM-Ascend 源码里的 graph runner，找出把哪些算子放进 CUDA-Graph 式图里

- 思路：查 graph runner 捕获的算子：多为固定 shape 的逐元素/attention/线性（无控制流）算子，condition 是 shape 静态可捕获；图中放置的算子与 CUDA Graph 捕获范围对应，动态分支留在图外。

---

## 第 96 课 · 昇腾 NPU 内存管理与 KV Cache

### 练习 1：把第 4 节 kv_cache_gb 改成返回字节数，核对 1e9 与 2^30 差别

- 思路：`kv_bytes = num_layers*2*seq*batch*head*dtype_bytes`；GB 用 `1e9`（商规）还是 `2^30`（GiB）差约 7%，打印两值对照，避免显存单位误判。

### 练习 2：用第 6 节模型算“1 个 32 层 7B 模型 + KV”所需显存，选一台装得下的昇腾

- 思路：权重 ≈ 7B×2B（fp16）≈14GB；KV 按并发/seq 公式累加（如并发 64×seq4k 约几十 GB，具体按 model config 代）；Σ 后对照昇腾单卡（310B 64GB/910B）判断是否需量化或降并发，选定机型。

### 练习 3：调研 vLLM-Ascend 的 kv-cache-dtype 参数，写 fp8 用法示例

- 思路：启动参数 `--kv-cache-dtype fp8`（或昇腾对应 `e4m3` 缩放到 KV 缓存）；示例命令行：`python -m vllm.entrypoints... --kv-cache-dtype fp8`；说明 fp8 KV 省一半内存、代价是 KV 精度损失。

### 练习 4：画“并发 vs 可支持上下文长度”权衡曲线，标注甜蜜点

- 思路：固定显存预算，随并发增加每请求可容 KV 变少→上下文长度下降，画负相关曲线；在“吞吐够高又不清空长度”处标甜点。曲线两端：张力-长上下文单请求 vs 高并发短上下文。

---

## 第 97 课 · 昇腾量化与稀疏化

### 练习 1：给 quantize_int8 加 per-channel scale（每行一个 scale），对比误差变化

- 思路：`scale = W.max(dim=1, keepdim=True).values/127`（每行 scale）替代全局单一 scale；per-channel 匹配行内分布、误差更小。打印全局 vs per-channel 的 `mae`。

### 练习 2：把 W 改成 4 的倍数不可整除的 K，观察 2:4 稀疏处理尾部

- 思路：N×K 中 `K % 4`，2:4 稀疏按 4 个连续列为一组必须整组处理；尾部不足 4 列时补 0 对齐或单独处理非结构化。观察掩码/补零后的误差与实现方式。

### 练习 3：调研 MindIE 的 W8A8 参数，写启用命令示例

- 思路：启用来例：`python vllm_chat.py --model ... --quantization DequantGptq/Amber/W8A8` 或 MindIE 的 `--lite_quant_mode W8A8`；写成命令行并注释各参数含义。

### 练习 4：设计小实验：量化+稀疏叠加，观察误差是相加还是相乘

- 思路：单独算量化误差 E_q 与稀疏误差 E_s，再算两者叠加误差 E_both；比较 `E_both` 与 `E_q+E_s`（相加假设）或 `E_q×E_s`（相乘假设）更接近。一般近似相加，若耦合偏大即证明二者相互作用，需联合调优。

---

## 第 98 课 · 昇腾集群通信：HCCL 与 AllReduce

### 练习 1：把第 3 节 N 改成 8、S 改成 16，重跑核对最大误差仍为 0

- 思路：Ring 在任意 N/S 下数学精确，改 N=8、S=16 跑多轮，`max_err = (ring - naive).abs().max()` 应为 0（浮点求和顺序仅轻微差异可忽略）。验证算法不依赖规模。

### 练习 2：给 ring_allreduce 加“统计每卡通信量”计数器，验证总量 = 2S(N-1)/N

- 思路：每个 Ring 阶段每卡进出各一块、共 `2(N-1)` 块，每块 S/N 余量？——每卡总通信 `2S(N-1)/N`；累加计数器对照公式，N 越大每卡通信越逼近 2S 下界。

### 练习 3：在第 5 节加一个 8MB 中间场景，找出 Ring 与 Tree 的交叉点

- 思路：Tree 大块带宽高但对数深度、Ring 在消息适中时每段带宽低但均衡；在 8MB 附近分别计时两算法，交点即“Ring/Tree 自适应切换”的场景规模（对应实际 NCCL/HCCL 按消息大小选算法）。

### 练习 4：查 HCCL 文档，列它与 NCCL 算法的对照表

- 思路：表：`HCCL_RING↔NCCL Ring`、`HCCL_TREE↔NCCL Tree`、`HCCL_P2P/SendRecv↔NCCL P2P`、`AllReduce/ReduceScatter/AllGather` 同名等价；列出各自支持的 collective 与拓扑感知/带宽选择机制。

---

## 第 99 课 · 昇腾推理性能调优

### 练习 1：把第 2 节 alpha 改成 0.05，观察甜蜜点位置如何右移

- 思路：alpha 控制时延 vs 吞吐的加权，0.05 更偏吞吐 → 最佳 batch/并发向右（更大吞吐）移动；重跑扫描画 α=0.05 与默认的曲线对比，拐点右移说明“目标函数变了，最优操作点不同”。

### 练习 2：给第 3 节瀑布图加一个“融合后”版本，对比总时长

- 思路：把逐算子各段时间合并成融合 kernel 的总时长，瀑布图并排两组（before/after fusion）；对比总时长下降，突出算子融合在时延上的贡献（省启动与 HBM 往返）。

### 练习 3：在真机/HiDevLab 上跑一次 msprof，导出分析结果并截图

- 思路：`msprof --output=result ./app` 采集，用 msprof 工具查看算子耗时/搬移占空比；截图关键时间线，定位最耗时算子与拷贝占比，据此提出下一个优化点。

### 练习 4：对照第 5 节表格，把 vLLM-Ascend 启动参数写成一份自己的“调优清单”

- 思路：清单示例：`--max-model-len`/`--max-num-seqs`/`--gpu-memory-utilization`/`--kv-cache-dtype`/`--enable-prefix-caching`/`--tensor-parallel-size`/线程与 block 相关参数；逐项注明“影响吞吐/时延/显存哪一项、调高调低建议”，形成可复用 checklist。

---

## 第 100 课 · 全书总结：vLLM 推理引擎全景与学习路线

### 练习 1：把第 2 节依赖图里第 11 章的入边单独画出来，说出它从哪几章“吸收”了什么

- 思路：第 11 章入边主要来自：第 5–8 章（attention/KV/paged 原理 → 昇腾 PagedAttention 印证）、第 9 章（Triton/算子开发 → Ascend C/CANN 对照）、第 10 章（编译/图优化 → GE/图下沉与算子融合），画这些"知识点→第 11 章"箭头并注明各吸收点。

### 练习 2：设计你自己的第 7 条主线（如内存优化）

- 思路：设计例：`内存主线：L8(KV)→L23(显存)→L63(内存规划)→L96(NPU KV)→L97(量化稀疏)`，标注每课对“把显存用得省”的贡献（KV 复用 → 规划 → NPU 分配 → 瘦身）。给出一条你自己的主线并编号。

### 练习 3：用 App“课号范围”筛选器圈出已掌握课程，制定 30 天补课计划

- 思路：在 dashboard 用筛选器（ch01–ch11 区间）勾选已完成课，用“剩余不掌握课”生成 30 天计划：每天 3–4 课，优先级按主线依赖排序（先基础 ch1–4 → 注意力 → 编译/昇腾），输出可执行表。

### 练习 4：写一篇 300 字毕业感想：书里哪一课改变了你对推理引擎的理解

- 思路（参考）：可写「第 9 章 Triton（或第 55 FlashAttention / 第 23 KV / 第 85 PagedAttention 昇腾）改变了理解」：它把“推理只是跑模型”升级为“推理 = 显存工程的系统设计”；PagedAttention 让显存按块动态分配、FlashAttention 让 attention 访存 O(N)，二者叠加让长序列/高并发成为可能——这本书把 vLLM 从黑盒变成可推理、可调优、可移植的体系。300 字完成。