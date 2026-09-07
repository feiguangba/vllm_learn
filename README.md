

# VLLM_learn · 图解 vLLM 推理引擎

> 用 **图解 + 动手实验** 的方式，从零到一理解 vLLM 推理引擎：KV Cache、PagedAttention、Continuous Batching、CUDA Graph、量化、分布式并行、Attention Kernel、Triton、AI 编译器，以及华为昇腾全栈。
>
> **100 课系统练习册 + 8 篇源码级架构文档 + 12 张顶会风格架构图。**

| 模块 | 路径 | 说明 |
|---|---|---|
| 🧪 **练习册（100 课）** | [`exercises/`](exercises/) | 11 章 100 课 Jupyter notebook，风格参照《鸢尾花书》：重图解、重直觉、循序渐进、动手实验 |
| 📚 **repowiki 架构文档** | [`repowiki/`](repowiki/) | 8 篇源码级架构文档（带 `文件:行号`），基于 `vendor/vllm`(v0.23.0-dev) 精读整理 |
| 🖼️ **架构图库** | [`exercises/figs/`](exercises/figs/) | 12 张 Visio 风格 SVG 架构图（已嵌入对应课程与第 100 课总画廊） |

---

## 🖼️ 核心架构图画廊

所有图遵循统一的顶会绘图规范（见 [`.claude/skills/paper-fig/SKILL.md`](.claude/skills/paper-fig/SKILL.md)）：
3 秒可懂、语义色板、每块标清所用网络/算子、结论脚注。点击即可查看 SVG（矢量无损缩放）。

| 架构图 | 讲的是什么 | 嵌入课程 |
|---|---|---|
| [Transformer 架构](exercises/figs/fig_01_transformer_block.svg) | 输入层 / 隐藏层 ×N / 输出层；QKV Linear、RoPE、SDPA、FFN/SwiGLU、lm_head 逐块标注 | [第 04 课](exercises/ch01/04_transformer_quickstart.ipynb) |
| [Prefill vs Decode](exercises/figs/fig_02_prefill_vs_decode.svg) | 两阶段推理：整段并行（算力瓶颈）vs 逐步循环（带宽瓶颈） | [第 05 课](exercises/ch01/05_prefill_vs_decode.ipynb) |
| [KV Cache](exercises/figs/fig_03_kv_cache.svg) | 历史 K/V 只算一次，O(T²) → O(T)，显存随 T 线性增长 | [第 07 课](exercises/ch02/07_kv_cache_principle.ipynb) |
| [PagedAttention](exercises/figs/fig_04_pagedattention.svg) | 逻辑块 → Block Table → 物理块，前缀共享 ref 计数 | [第 10 课](exercises/ch02/10_pagedattention_core.ipynb) |
| [Continuous Batching](exercises/figs/fig_05_continuous_batching.svg) | 静态批的 idle 空泡 vs 迭代粒度换入换出 | [第 15 课](exercises/ch03/15_continuous_batching.ipynb) |
| [请求状态机与调度循环](exercises/figs/fig_06_request_lifecycle.svg) | WAITING/RUNNING/PREEMPTED/FINISHED + Scheduler 单步迭代 | [第 16 课](exercises/ch03/16_request_state_machine.ipynb) |
| [vLLM 系统架构](exercises/figs/fig_07_vllm_arch.svg) | API 层 / EngineCore（Scheduler+KV 管理）/ GPU 执行层 | [第 48 课](exercises/ch08/48_serve_model.ipynb) |
| [端到端推理流水线](exercises/figs/fig_08_inference_pipeline.svg) | Prompt → tokenize → 组批 → forward → sample → SSE 流式 | [第 22 课](exercises/ch04/22_modelrunner_dataflow.ipynb) |
| [量化全景](exercises/figs/fig_09_quantization.svg) | scale/zero-point 仿射映射 + GPTQ / AWQ / FP8 三条路线 | [第 28 课](exercises/ch05/28_quantization_basics.ipynb) |
| [分布式并行](exercises/figs/fig_10_parallelism.svg) | TP 切权重、PP 切层、DP 切请求、EP 切专家 | [第 35 课](exercises/ch06/35_parallel_overview.ipynb) |
| [FlashAttention](exercises/figs/fig_11_flash_attention.svg) | 分块计算 + online softmax，显存 O(N²)→O(N) | [第 42 课](exercises/ch07/42_flash_attention_principle.ipynb) |
| [昇腾全栈对照](exercises/figs/fig_12_ascend_stack.svg) | NVIDIA ↔ 华为昇腾五层逐层对应：硬件/使能/框架/引擎/云 | [第 71 课](exercises/ch11/71_huawei_ecosystem.ipynb) |

> 第 [100 课](exercises/ch11/100_summary_roadmap.ipynb) 开头有全部 12 张图的总画廊。图审校记录见 [`exercises/figs/REVIEW.md`](exercises/figs/REVIEW.md)。

---

## 🧪 练习册总览（11 章 · 100 课）

> 每课 = 1 个 `NN_*.ipynb`（可运行、带实验）+ 多数课程配 `app_NN_*.py` 交互演示（ch01–ch08 全覆盖，ch09–ch11 部分）。

### 第 1 章 · LLM 推理基础（01–06）`exercises/ch01/`

| # | 课程 | 一句话 |
|---|---|---|
| [01](exercises/ch01/01_token_and_tokenizer.ipynb) | 认识 Token 与分词器 | 字符级到 BPE，手写迷你分词器，看懂 token→id |
| [02](exercises/ch01/02_autoregressive_generation.ipynb) | 自回归生成 | n-gram 玩具模型逐字接龙，生成长度 L 需 L 次前向 |
| [03](exercises/ch01/03_sampling_strategies.ipynb) | 采样策略 | greedy / temperature / top-k / top-p，连到 SamplingParams |
| [04](exercises/ch01/04_transformer_quickstart.ipynb) | 手写单层 Transformer 块 | CPU 手写 attention block，9 步打印形状 |
| [05](exercises/ch01/05_prefill_vs_decode.ipynb) | Prefill vs Decode | FLOPs 推导 + torch 实测两阶段耗时差 |
| [06](exercises/ch01/06_toy_inference_engine.ipynb) | 玩具推理引擎 | n-gram + 采样拼装批量引擎，实测吞吐曲线 |

### 第 2 章 · KV Cache 与 PagedAttention（07–13）`exercises/ch02/`

| # | 课程 | 一句话 |
|---|---|---|
| [07](exercises/ch02/07_kv_cache_principle.ipynb) | KV Cache 原理 | O(T²) 重算降成 O(T) 追加写，数值验证加速比 |
| [08](exercises/ch02/08_kv_cache_memory.ipynb) | KV Cache 内存账本 | 显存公式推导，MHA / GQA / MQA 对比 |
| [09](exercises/ch02/09_fragmentation_problem.ipynb) | 内存碎片化 | 内/外碎片，连续 vs 分页分配器模拟 |
| [10](exercises/ch02/10_pagedattention_core.ipynb) | PagedAttention 核心 | 用操作系统虚拟内存思想管理 KV |
| [11](exercises/ch02/11_block_table_slots.ipynb) | Block Table 与 Slot Mapping | block_ids / slot_mapping 实现，多请求并发 |
| [12](exercises/ch02/12_prefix_caching.ipynb) | 前缀缓存 | 块哈希复用，命中率对内存/延迟的影响 |
| [13](exercises/ch02/13_cow_copy_on_write.ipynb) | Copy-on-Write | 引用计数，共享块的写入时机 |

### 第 3 章 · Continuous Batching 与调度（14–20）`exercises/ch03/`

| # | 课程 | 一句话 |
|---|---|---|
| [14](exercises/ch03/14_static_batching_problem.ipynb) | 静态批处理的缺陷 | 队头阻塞、尾部浪费，甘特图模拟 |
| [15](exercises/ch03/15_continuous_batching.ipynb) | Continuous Batching | 事件驱动模拟器，每步 batch 成员变化 |
| [16](exercises/ch03/16_request_state_machine.ipynb) | 请求状态机 | WAITING / RUNNING / FINISHED 状态迁移日志 |
| [17](exercises/ch03/17_scheduler_design.ipynb) | Scheduler 设计 | token 预算 × FCFS / SJF / Priority |
| [18](exercises/ch03/18_preemption.ipynb) | 抢占 Preemption | recompute vs swap，延迟惩罚对比 |
| [19](exercises/ch03/19_iterative_scheduler_sim.ipynb) | 迭代级调度模拟器 | 完整 Simulator 与参数敏感性实验 |
| [20](exercises/ch03/20_batch_compare_experiment.ipynb) | 真实 GPU 对比 | 静态批 vs Continuous Batching 吞吐实测 |

### 第 4 章 · 模型执行与 CUDA Graph（21–27）`exercises/ch04/`

| # | 课程 | 一句话 |
|---|---|---|
| [21](exercises/ch04/21_metadata_assembly.ipynb) | 变长序列组批 | input_ids / positions / slot_mapping 组装 |
| [22](exercises/ch04/22_modelrunner_dataflow.ipynb) | ModelRunner 数据流 | 迷你 GPT 前向，从 embedding 到 lm_head |
| [23](exercises/ch04/23_kv_allocator.ipynb) | KV 块分配器 | free list、引用计数与碎片率 |
| [24](exercises/ch04/24_cuda_graph_principle.ipynb) | kernel 启动开销 | CPU 为什么会成为 GPU 的瓶颈 |
| [25](exercises/ch04/25_cudagraph_capture.ipynb) | CUDA Graph 捕获与重放 | 为什么形状必须固定 |
| [26](exercises/ch04/26_torch_compile.ipynb) | torch.compile | 把一串小算子冻成一颗大 kernel |
| [27](exercises/ch04/27_profiling_metrics.ipynb) | 性能剖析 | tokens/s、TTFT、TPOT 与延迟分布 |

### 第 5 章 · 量化（28–34）`exercises/ch05/`

| # | 课程 | 一句话 |
|---|---|---|
| [28](exercises/ch05/28_quantization_basics.ipynb) | 量化基础 | 位宽、scale 与 zero-point |
| [29](exercises/ch05/29_sym_asym_quant.ipynb) | 对称 vs 非对称 | per-tensor 与 per-channel |
| [30](exercises/ch05/30_quantization_error.ipynb) | 量化误差分析 | MSE、SNR 与校准 |
| [31](exercises/ch05/31_gptq_principle.ipynb) | GPTQ 思想 | 逐列量化与误差补偿，手写简化演示 |
| [32](exercises/ch05/32_awq_principle.ipynb) | AWQ 思想 | 激活感知的权重保护 |
| [33](exercises/ch05/33_fp8_kv_quant.ipynb) | FP8 与 KV 量化 | e4m3 / e5m2，显存节省计算 |
| [34](exercises/ch05/34_quant_tradeoff.ipynb) | 量化权衡实验 | 精度-速度-显存三角权衡 |

### 第 6 章 · 分布式并行（35–41）`exercises/ch06/`

| # | 课程 | 一句话 |
|---|---|---|
| [35](exercises/ch06/35_parallel_overview.ipynb) | 并行策略总览 | DP / TP / PP / EP，7B/70B 为什么必须并行 |
| [36](exercises/ch06/36_allreduce_nccl.ipynb) | AllReduce 与 NCCL | numpy 手写 ring / tree，带宽公式 |
| [37](exercises/ch06/37_tensor_parallel.ipynb) | 张量并行 | 列切 / 行切，Megatron QKV/FFN 切法 |
| [38](exercises/ch06/38_pipeline_parallel.ipynb) | 流水线并行 | GPipe vs 1F1B 甘特图，bubble 公式 |
| [39](exercises/ch06/39_data_parallel_dp.ipynb) | 数据并行 | 推理负载均衡 vs 训练梯度同步 |
| [40](exercises/ch06/40_custom_allreduce.ipynb) | CustomAllreduce | 共享内存替掉 NCCL，延迟模型 |
| [41](exercises/ch06/41_parallel_combination.ipynb) | 并行组合 | TP × PP × DP 估算公式与选型 |

### 第 7 章 · Attention Kernel 实战（42–47）`exercises/ch07/`

| # | 课程 | 一句话 |
|---|---|---|
| [42](exercises/ch07/42_flash_attention_principle.ipynb) | FlashAttention 原理 | O(N²) 瓶颈，分块 + 在线 softmax |
| [43](exercises/ch07/43_online_softmax.ipynb) | 在线 Softmax | running max 修正，numpy 分块验证 |
| [44](exercises/ch07/44_attention_kernel.ipynb) | Attention Kernel | naive vs 分块，kernel 逐行讲解 |
| [45](exercises/ch07/45_paged_attention.ipynb) | PagedAttention kernel | block table 收集非连续 KV |
| [46](exercises/ch07/46_attention_perf.ipynb) | Attention 性能 | naive / 分块 / SDPA 耗时实测 |
| [47](exercises/ch07/47_attn_backend_select.ipynb) | 后端选择机制 | FLASH_ATTN / TRITON / FLASHINFER 优先级 |

### 第 8 章 · 端到端部署与服务化（48–50）`exercises/ch08/`

| # | 课程 | 一句话 |
|---|---|---|
| [48](exercises/ch08/48_serve_model.ipynb) | 部署入门 | `vllm serve` 与 OpenAI 兼容接口，迷你 server |
| [49](exercises/ch08/49_openai_api.ipynb) | OpenAI 客户端实战 | chat/completions、stream、SSE 解析 |
| [50](exercises/ch08/50_metrics_monitor.ipynb) | 性能指标与监控 | TTFT / TPOT / 吞吐 / Prometheus 仪表盘 |

### 第 9 章 · Triton GPU 编程（51–60）`exercises/ch09/`

| # | 课程 | 一句话 |
|---|---|---|
| [51](exercises/ch09/51_triton_intro.ipynb) | Triton 是什么 | 用 Python 写 GPU kernel 的另一条路 |
| [52](exercises/ch09/52_triton_vectoradd.ipynb) | 第一个 Triton kernel | 向量加法，grid / block 直觉 |
| [53](exercises/ch09/53_triton_tile_model.ipynb) | tile 编程模型 | program_id + tl.arange + BLOCK 指针 |
| [54](exercises/ch09/54_triton_gemm.ipynb) | Triton GEMM | tl.dot 写矩阵乘，naive vs tile 对比 |
| [55](exercises/ch09/55_triton_flashattn.ipynb) | Triton FlashAttention | 分块 + 在线 softmax 的 kernel 实现 |
| [56](exercises/ch09/56_triton_compiler.ipynb) | Triton 编译器原理 | 从 Python 一路 lowered 到 PTX |
| [57](exercises/ch09/57_triton_tuning.ipynb) | 性能调优 | num_warps / num_stages / BLOCK 搜索 |
| [58](exercises/ch09/58_triton_vllm.ipynb) | Triton 与 vLLM | 为什么 vLLM 用 Triton 写 attention kernel |
| [59](exercises/ch09/59_triton_debug.ipynb) | 调试与验证 | 越界、掩码、精度：kernel bug 排查 |
| [60](exercises/ch09/60_triton_oplib.ipynb) | mini 算子库 | LayerNorm / GELU / Softmax / RMSNorm |

### 第 10 章 · AI 编译器（61–70）`exercises/ch10/`

| # | 课程 | 一句话 |
|---|---|---|
| [61](exercises/ch10/61_ai_compiler_overview.ipynb) | AI 编译器全景 | TVM / MLIR / XLA / Inductor 为什么存在 |
| [62](exercises/ch10/62_graph_optimization.ipynb) | 计算图优化 | 算子融合、常量折叠、死代码消除 |
| [63](exercises/ch10/63_memory_scheduling.ipynb) | 内存规划与生命周期 | 把显存用得抠门 |
| [64](exercises/ch10/64_mlir_intro.ipynb) | MLIR 与中间表示 | 多层级 IR 与 Dialect 思想 |
| [65](exercises/ch10/65_auto_tuning.ipynb) | 自动调优 | Ansor / auto-sched 搜索最优参数 |
| [66](exercises/ch10/66_torch_to_backend.ipynb) | 从 PyTorch 到后端 | 模型如何变成 Triton kernel |
| [67](exercises/ch10/67_kernel_fusion.ipynb) | Kernel 融合实战 | 把一串小 kernel 并成一个 |
| [68](exercises/ch10/68_codegen_schedule.ipynb) | 代码生成与调度 | 循环分块、向量化、tiling |
| [69](exercises/ch10/69_oplib_vs_compiler.ipynb) | 算子库 vs 编译器 | cuDNN/CUTLASS（专业厨师）与编译器（万能厨师） |
| [70](exercises/ch10/70_ai_compiler_trends.ipynb) | 发展趋势 | torch.compile 生态与 MLIR 动态 |

### 第 11 章 · 华为昇腾全栈（71–100）`exercises/ch11/`

| # | 课程 | 一句话 |
|---|---|---|
| [71](exercises/ch11/71_huawei_ecosystem.ipynb) | 华为 AI 生态全景 | 昇腾 + CANN + MindSpore + ModelArts |
| [72](exercises/ch11/72_davinci_arch.ipynb) | 达芬奇架构 | AI Core 三单元与分级缓存 |
| [73](exercises/ch11/73_cann_stack.ipynb) | CANN 软件栈 | 从应用接口到芯片执行的五层结构 |
| [74](exercises/ch11/74_ascend_c.ipynb) | Ascend C 算子编程 | 给 AI Core 排数据流水 |
| [75](exercises/ch11/75_mindspore_tensor.ipynb) | MindSpore 基础 | 张量、自动微分与计算图思维 |
| [76](exercises/ch11/76_mindspore_graph.ipynb) | 计算图与静态图 | 从 PyNative 到整图下沉 |
| [77](exercises/ch11/77_ms_dynamic_static.ipynb) | 动态图 vs 静态图 | 灵活、性能与调试的三方博弈 |
| [78](exercises/ch11/78_ms_operator_dev.ipynb) | MindSpore 算子开发 | Primitive、自定义算子与融合 |
| [79](exercises/ch11/79_graph_fusion.ipynb) | 图算融合 | GE 如何把 N 个算子并成 1 个 kernel |
| [80](exercises/ch11/80_ms_distributed.ipynb) | 分布式训练 | 数据并行、模型并行与自动并行 |
| [81](exercises/ch11/81_ms_inference.ipynb) | MindSpore 推理部署 | 从训练到服务的落地路径 |
| [82](exercises/ch11/82_mindir.ipynb) | MindIR 与模型转换 | 模型的「通用护照」 |
| [83](exercises/ch11/83_ms_lite.ipynb) | MindSpore Lite 端侧 | 手机上的大模型：转换、量化与硬件加速 |
| [84](exercises/ch11/84_ascend_vs_vllm.ipynb) | 昇腾推理引擎 vs vLLM | vLLM-Ascend 与 MindIE 同题对比 |
| [85](exercises/ch11/85_pagedattn_ascend.ipynb) | PagedAttention 在昇腾 | 分页思想换硬件的适配 |
| [86](exercises/ch11/86_cann_oplib.ipynb) | CANN 算子库与融合 | 昇腾的「cuDNN/cuBLAS」 |
| [87](exercises/ch11/87_ascend_quant.ipynb) | 华为量化方案 | 与 GPTQ / AWQ / FP8 同源的减精度思路 |
| [88](exercises/ch11/88_ascend_llm.ipynb) | 昇腾大模型推理 | MindIE 的 KV 管理、并行与长上下文 |
| [89](exercises/ch11/89_ms_autoparallel.ipynb) | MindSpore 自动并行 | 让框架替你选切分方式 |
| [90](exercises/ch11/90_ms_evolution.ipynb) | 全栈 AI 演进 | 芯片 → 框架 → 引擎的协同演进 |
| [91](exercises/ch11/91_ascend_llama.ipynb) | 昇腾上跑 LLaMA | 从 PyTorch 到昇腾的模型迁移 |
| [92](exercises/ch11/92_ascend_c_perf.ipynb) | Ascend C 性能实战 | 矢量优化与流水线 |
| [93](exercises/ch11/93_modelarts.ipynb) | ModelArts 与昇腾云 | 把训练和推理搬上云 |
| [94](exercises/ch11/94_ms_pytorch_migrate.ipynb) | MindSpore ↔ PyTorch 迁移 | API 对照与踩坑 |
| [95](exercises/ch11/95_ascend_graph_engine.ipynb) | 昇腾图引擎 GE | 让计算图整图下沉 |
| [96](exercises/ch11/96_npu_memory_kv.ipynb) | NPU 内存管理与 KV Cache | 内存池与复用策略 |
| [97](exercises/ch11/97_ascend_sparsity.ipynb) | 量化与稀疏化 | 让模型瘦身提速 |
| [98](exercises/ch11/98_ascend_cluster.ipynb) | 集群通信 | HCCL 与 AllReduce |
| [99](exercises/ch11/99_ascend_perf_tuning.ipynb) | 推理性能调优 | 吞吐、时延与 profiling |
| [100](exercises/ch11/100_summary_roadmap.ipynb) | 全书总结 | 100 课知识地图 + 12 张架构图总画廊 |

---

## 📚 repowiki 架构文档

基于 `vendor/vllm`(v0.23.0-dev) 源码精读，8 篇文档每篇都带真实 `文件:行号` 引用：

| # | 文档 | 主题 |
|---|---|---|
| 01 | [系统架构](repowiki/01_system_architecture.md) | 多进程拓扑(前端/EngineCore/Worker) + ZMQ 通信 + 事件循环 |
| 02 | [PagedAttention 与 KV Cache](repowiki/02_PagedAttention_KVCache.md) | 块管理/容量计算/前缀缓存/COW |
| 03 | [调度器](repowiki/03_Scheduler.md) | token 预算 / 抢占 / 连续批处理 |
| 04 | [LLMEngine](repowiki/04_LLMEngine.md) | 输入渲染 / 输出规格化 / detokenizer |
| 05 | [ModelRunner 与 CUDA Graph](repowiki/05_ModelRunner_CUDAGraph.md) | 执行器 / 图捕获 / torch.compile |
| 06 | [量化](repowiki/06_Quantization.md) | FP8 / GPTQ / AWQ 三层抽象 |
| 07 | [TP/PP/DP 并行](repowiki/07_TP_PP_DP.md) | 组切分 / 并行层 / NCCL / CustomAllReduce |
| 08 | [Attention 后端](repowiki/08_Attention_Kernels.md) | 后端选择 / FlashAttention / MLA |

> 📋 文档计划与源码依据见 [`repowiki/wiki_plan.yaml`](repowiki/wiki_plan.yaml)。

---

## 🚀 环境与使用方法

### 环境要求

- Python 3.12 + `uv`（或 venv）
- `torch`（CPU 即可跑通全部练习；GPU 可选）
- `pyecharts`、`plotly`、`streamlit`、`numpy`、`pandas`、`matplotlib`
- Windows 用户需设 `KMP_DUPLICATE_LIB_OK=TRUE`（避免 torch 与 Anaconda 的 OMP 库冲突）

安装（以 uv 为例）：

```bash
uv venv uv_cuda --python 3.12
uv pip install --python uv_cuda/Scripts/python.exe torch pyecharts streamlit plotly numpy pandas matplotlib ipykernel nbformat jupyterlab
```

### 运行 notebook

```powershell
# Windows (PowerShell)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
D:\uv_envs\uv_cuda\Scripts\python.exe -m jupyter lab   # 打开后选 uv_cuda 内核
```

每个 notebook 内部都自带「KMP 保护」开头 cell，可直接运行。

### 运行交互演示（Streamlit）

ch01–ch08 每课配有 `app_NN_*.py`，与 notebook 同目录：

```powershell
D:\uv_envs\uv_cuda\Scripts\python.exe -m streamlit run exercises/ch01/app_01_token_demo.py
# 浏览器打开 http://localhost:8501
```

app 源码也以 `%%writefile` cell 内嵌在对应 notebook 中，二者内容一致。

### 学习顺序建议

1. **入门主线**：ch01 → ch02 → ch03（推理基础 → KV Cache/PagedAttention → 调度），每课先跑 notebook 再玩 app；
2. **深入执行**：ch04 → ch05 → ch06（模型执行/CUDA Graph → 量化 → 并行）；
3. **底层与部署**：ch07 → ch08（Attention Kernel → 服务化）；
4. **扩展视野**：ch09（Triton）→ ch10（AI 编译器）→ ch11（昇腾全栈）；
5. 需要「源码级」理解时，对照 **repowiki 01–08** 与 `vendor/vllm` 源码；
6. 想快速建立全局直觉，先看第 [100 课](exercises/ch11/100_summary_roadmap.ipynb) 的 12 张架构图总画廊。

---

## 🗂️ 目录结构

```
VLLM_learn/
├── README.md                # 本文件
├── .claude/skills/paper-fig/ # 顶会级科研绘图规范（画新架构图时引用）
├── repowiki/                # 8 篇源码级架构文档
│   ├── 01_system_architecture.md .. 08_Attention_Kernels.md
│   └── wiki_plan.yaml
└── exercises/               # 练习册（11 章 100 课）
    ├── GUIDELINES.md        # 写作规范
    ├── nb_builder.py        # notebook 生成工具
    ├── figs/                # 12 张 SVG 架构图 + REVIEW.md 审校记录
    ├── ch01/ .. ch11/       # 11 章，每章 3–30 课
    │   ├── NN_*.ipynb       # 每课 notebook
    │   ├── app_NN_*.py      # 每课 streamlit 交互演示
    │   └── _build/          # 生成脚本（helpers + build_NN.py）
```

> 仓库外部依赖：`vendor/vllm`（vLLM 源码，repowiki 与 ch07/47 等课程引用）。

---

## ✍️ 版权与参考

- 本仓库为学习笔记，非官方文档；
- 架构图为本仓库原创绘制（SVG 源文件在 `exercises/figs/`），绘图规范提炼自公开的顶会绘图方法论；
- 参考：vLLM 官方文档 <https://docs.vllm.ai>、PagedAttention 论文 <https://arxiv.org/abs/2309.06180>、FlashAttention 论文 <https://arxiv.org/abs/2205.14135>、Orca 论文 <https://arxiv.org/abs/2208.14217>。
