# VLLM_learn · 图解 vLLM 推理引擎

> 用 **图解 + 动手实验** 的方式,从零到一理解 vLLM 推理引擎:KV Cache、PagedAttention、Continuous Batching、CUDA Graph、量化、分布式并行与 Attention Kernel。

本仓库包含两大块内容:

| 模块 | 路径 | 说明 |
|---|---|---|
| 📚 **repowiki 架构文档** | [`repowiki/`](repowiki/) | 8 篇源码级架构文档(带 `文件:行号`),基于 `vendor/vllm`(v0.23.0-dev)精读整理 |
| 🧪 **练习册(50 个 notebook)** | [`exercises/`](exercises/) | 50 课 Jupyter notebook + 每课配套 1 个 Streamlit 动态演示 app |

---

## 🧪 练习册总览(50 课)

> 风格参照《鸢尾花书》数据科学系列:中文讲解、图文并茂、形象比喻、循序渐进。
> 每课 = 1 个 `NN_*.ipynb` + 1 个 `app_NN_*.py`(Streamlit 交互演示)。

### 第 1 章 · LLM 推理基础 `exercises/ch01/`

| # | 主题 | 一句话 |
|---|---|---|
| 01 | 认识 Token 与分词器 🏷️ | 字符级到 BPE,手写迷你分词器,看懂 token→id |
| 02 | 自回归生成 🔁 | n-gram 玩具模型逐字接龙,证明"生成长度 L 需 L 次前向" |
| 03 | 采样策略 🎲 | greedy/temperature/top-k/top-p,连到 vLLM SamplingParams |
| 04 | Transformer 结构速览 🧠 | CPU 手写单层 Transformer block,9 步打印形状 |
| 05 | Prefill vs Decode ⚡ | FLOPs 推导 + torch 实测两阶段耗时差 |
| 06 | 玩具推理引擎 🛠️ | 拼装 n-gram+采样成批量引擎,实测吞吐曲线 |

### 第 2 章 · KV Cache 与 PagedAttention `exercises/ch02/`

| # | 主题 | 一句话 |
|---|---|---|
| 07 | KV Cache 原理 🔑 | 无缓存 O(T²) vs 有缓存 O(T),数值验证加速比 |
| 08 | KV Cache 内存账本 🧮 | 内存公式,MHA/GQA/MQA 对比 |
| 09 | 连续内存碎片问题 🧩 | 内/外碎片,连续 vs 分页分配器模拟 |
| 10 | PagedAttention 核心思想 📖 | 虚拟/物理块,手写页表,slot 映射 |
| 11 | Block Table 与 slot 映射 🔗 | block_ids/slot_mapping 实现,多请求并发模拟 |
| 12 | 前缀缓存 🌳 | 块哈希→复用,命中率对内存/延迟影响 |
| 13 | COW 写时复制 🖨️ | 引用计数,多请求共享块的复制时机 |

### 第 3 章 · Continuous Batching 与调度 `exercises/ch03/`

| # | 主题 | 一句话 |
|---|---|---|
| 14 | 静态批处理的缺陷 🐌 | 队头阻塞、尾部浪费,甘特图模拟 |
| 15 | Continuous Batching 🍳 | 事件驱动模拟器,每步 batch 成员变化 |
| 16 | 请求状态机 🚦 | WAITING/RUNNING/FINISHED,状态迁移日志 |
| 17 | 调度器设计 🎛️ | token 预算 × FCFS/SJF/Priority |
| 18 | 抢占 Preemption ⚔️ | recompute vs swap,延迟惩罚对比 |
| 19 | 迭代级调度模拟器 🛰️ | 完整 Simulator,参数敏感性实验 |
| 20 | 吞吐对比实验 📊 | 静态 vs 连续批吞吐实测 |

### 第 4 章 · 模型执行与 CUDA Graph `exercises/ch04/`

| # | 主题 | 一句话 |
|---|---|---|
| 21 | 从调度输出到 GPU 输入 🧩 | input_ids/position/slot_mapping/block_table 组装 |
| 22 | ModelRunner 数据流 🚰 | 迷你 GPT 前向数据流,8 步形状追踪 |
| 23 | KV 分配器模拟 🗃️ | free list/refcount/碎片率 |
| 24 | CUDA Graph 原理 📸 | kernel launch 开销,类比实测 |
| 25 | 图捕获与重放 🎬 | 录制-回放类比,shape 必须固定 |
| 26 | torch.compile 🧊 | inductor 融合,vLLM 编译层级概念 |
| 27 | 性能剖析 📈 | TTFT/TPOT/吞吐指标计算与可视化 |

### 第 5 章 · 量化 `exercises/ch05/`

| # | 主题 | 一句话 |
|---|---|---|
| 28 | 量化基础 🔢 | fp16/bf16/int8 位拆解,对称量化 |
| 29 | 对称 vs 非对称 ⚖️ | scale/zero-point,per-tensor/per-channel |
| 30 | 量化误差分析 📏 | MSE/SNR,重尾分布与离群值 |
| 31 | GPTQ 原理 🧮 | 逐列量化+误差补偿,手写简化演示 |
| 32 | AWQ 原理 🛡️ | 激活感知,缩放保护显著权重 |
| 33 | FP8 与 KV 量化 🚀 | e4m3/e5m2,显存节省计算 |
| 34 | 量化权衡实验 📊 | 精度-速度-显存三角权衡 |

### 第 6 章 · 分布式并行 `exercises/ch06/`

| # | 主题 | 一句话 |
|---|---|---|
| 35 | 并行策略总览 🗺️ | DP/TP/PP/EP,7B/70B 为什么必须并行 |
| 36 | AllReduce 与 NCCL 🔄 | numpy 手写 ring/tree,带宽公式 |
| 37 | 张量并行 ✂️ | 列切/行切,Megatron QKV/FFN 切法 |
| 38 | 流水并行 🏭 | GPipe vs 1F1B 甘特图,bubble 公式 |
| 39 | 数据并行 🧑‍🤝‍🧑 | 推理负载均衡 vs 训练梯度同步 |
| 40 | Custom AllReduce ⚡ | 共享内存+IPC 思路,延迟模型 |
| 41 | 并行组合 🧩 | TP×PP×DP 估算公式与选型 |

### 第 7 章 · Attention Kernel 实战 `exercises/ch07/`

| # | 主题 | 一句话 |
|---|---|---|
| 42 | FlashAttention 原理 ⚡ | O(N²) 瓶颈,分块+在线 softmax |
| 43 | 在线 softmax 🧮 | streaming max 修正,手写 numpy 分块验证 |
| 44 | Attention Kernel 🔬 | naive vs 分块,Triton 语法逐行讲解 |
| 45 | Paged Attention 📖 | block table 收集非连续 KV |
| 46 | Attention 性能对比 🚀 | naive/分块/SDPA 耗时实测 |
| 47 | 后端选择机制 🎛️ | FLASH_ATTN/TRITON/FLASHINFER 优先级 |

### 第 8 章 · 端到端 vLLM 部署 `exercises/ch08/`

| # | 主题 | 一句话 |
|---|---|---|
| 48 | 部署入门 🚀 | OpenAI 兼容 API,`vllm serve` 参数,迷你 server |
| 49 | OpenAI 客户端实战 🔌 | chat/completions,temperature/stream,SSE 解析 |
| 50 | 性能监控 📊 | TTFT/TPOT/吞吐,指标仪表盘 |

---

## 📚 repowiki 架构文档

基于 `vendor/vllm`(v0.23.0-dev)源码精读,8 篇文档每篇都带真实 `文件:行号` 引用:

| # | 文档 | 主题 |
|---|---|---|
| 01 | [系统架构](repowiki/01_system_architecture.md) | 多进程拓扑(前端/EngineCore/Worker)+ ZMQ 通信 + 事件循环 |
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

- Python 3.12 + `uv`(或 venv)
- `torch`(CPU 即可跑通全部练习;GPU 可选)
- `pyecharts`、`plotly`、`streamlit`、`numpy`、`pandas`、`matplotlib`
- Windows 用户需设 `KMP_DUPLICATE_LIB_OK=TRUE`(避免 torch 与 Anaconda 的 OMP 库冲突)

安装(以 uv 为例):

```bash
uv venv uv_cuda --python 3.12
uv pip install --python uv_cuda/Scripts/python.exe torch pyecharts streamlit plotly numpy pandas matplotlib ipykernel nbformat
```

### 运行 notebook

```bash
# Windows (PowerShell)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
D:\uv_envs\uv_cuda\Scripts\python.exe -m jupyter lab   # 打开后选 uv_cuda 内核
```

每个 notebook 内部都自带"KMP 保护"开头 cell,可直接运行。

### 运行 Streamlit 演示

每课的 app 与 notebook 同目录,用 `streamlit run` 启动:

```bash
D:\uv_envs\uv_cuda\Scripts\python.exe -m streamlit run exercises/ch01/app_01_token_demo.py
# 浏览器打开 http://localhost:8501
```

app 源码也以 `%%writefile` cell 内嵌在对应 notebook 中,二者内容一致。

### 学习顺序建议

1. **先读 repowiki 01**(系统架构),建立整体框架;
2. 按 **ch01 → ch02 → ch03** 顺序跑练习册(推理基础 → KV Cache → 调度),每课先看 notebook 再用 app 玩参数;
3. **ch04 → ch05 → ch06** 深入执行/量化/并行;
4. **ch07 → ch08** 收尾(Kernel 与部署);
5. 需要"源码级"理解时,对照 **repowiki 02-08** 与 `vendor/vllm` 源码。

---

## 🗂️ 目录结构

```
VLLM_learn/
├── README.md                # 本文件
├── repowiki/                # 8 篇源码级架构文档
│   ├── 01_system_architecture.md
│   ├── ... 08_Attention_Kernels.md
│   └── wiki_plan.yaml
└── exercises/               # 练习册(50 课)
    ├── GUIDELINES.md        # 写作规范(每课配 streamlit app)
    ├── nb_builder.py        # notebook 生成工具
    ├── ch01/ .. ch08/       # 8 章,每章 3-7 课
    │   ├── NN_*.ipynb       # 每课 notebook
    │   ├── app_NN_*.py      # 每课 streamlit 演示
    │   └── _build/          # 生成脚本(helpers + build_NN.py)
```

> 仓库外部依赖:`vendor/vllm`(vLLM 源码,repowiki 与 ch07/47 等课程引用)。

---

## ✍️ 版权与参考

- 本文档为学习笔记,非官方文档;
- 参考:vLLM 官方文档 <https://docs.vllm.ai>、PagedAttention 论文 <https://arxiv.org/abs/2309.06180>、FlashAttention 论文 <https://arxiv.org/abs/2205.14135>、Orca 论文 <https://arxiv.org/abs/2208.14217>。
