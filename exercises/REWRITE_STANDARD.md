# 全库重写标准 (REWRITE_STANDARD)

目标：把既有 100 课（重点 ch01-ch08 原 50 课）全部升级为「专业、细致、由浅入深、脉络清晰、每行注释」的教材级 notebook。

## 核心诉求（用户原话提炼）
1. 讲解太「通俗」→ 要 **专业细致**：引用真实论文/博客、给出严谨公式推导、讲清边界条件与中间量
2. 矩阵维度打印不清楚、零零散散没有脉络感 → 每个张量都**打印 shape + 标注每个维度的含义**，代码组织有清晰主线
3. 代码纯看看不懂 → **每一行都要写注释**（inline 注释，说明"这行在做什么、为什么"）
4. 要由浅入深 → 结构像教科书：直觉 → 定义 → 公式 → 小例子逐行推演 → 大规模实测 → 工程关联
5. 结合调研的论文 + 通用能力 → 参考文献必须真实、准确、贴题

## 参考风格
- 代码/维度风格参照 `aios` 仓库（已 clone 到 `VLLM_learn/aios/`）：每个 tensor 明确 shape 注释、
  docstring 写清维度含义、核心函数只做一件事。
- 讲解风格参照鸢尾花书：重图解、重直觉、循序渐进，但要比现在更「硬」——有公式、有推导、有数字。

## 每课 notebook 结构模板（由浅入深，脉络清晰）
1. **封面**：标题/副标题/学习目标/目录/参考链接（论文 arXiv + 官方博客）
2. **第 1 节 直觉与动机**：用最小例子/生活类比讲清「为什么需要」
3. **第 2 节 核心定义与公式**：给出精确公式，每个符号解释（含维度）
4. **第 3 节 最小实现 · 逐行推演**：手工构造小张量（如 batch=1, seq=4），**每行代码 inline 注释**，
   关键中间张量**打印 shape + 内容**，用 print 把脉络串起来
5. **第 4 节 数值验证/复杂度分析**：对比 naive vs 优化，画出曲线（matplotlib/seaborn 会议风格）
6. **第 5 节 真实规模数字**：代入主流模型参数（LLaMA/Qwen 配置表），算真实 FLOPs/内存/加速比
7. **第 6 节 与 vLLM 工程实现的关系**：引用真实代码路径（vendor/vllm 或 aios），讲清工程真相
8. **第 7 节 小结 + 练习 + 延伸阅读**（论文链接）

## 代码规范（最重要）
- **每一行可执行代码都要有注释**。`#` 注释写在行尾或上一行，说明"这行在做什么/为什么这么做/产出什么 shape"
- 变量命名用下划线语义化（`num_layers`, `head_dim`, `seq_len`），不用单字母凑
- 每个张量出现时，尽量伴随一行 `print(f"xxx shape: {xxx.shape}")` 并说明维度含义
- 核心函数带 docstring：参数、返回、每个维度的含义
- 数值示例用小而清晰：如 `batch=2, seq=4, num_heads=2, head_dim=8`，让读者能心算验证
- 一个 cell 只做一件事：导入、定义、小例子、大实验分开

## Markdown 规范
- 每节标题编号 `## 1. ...`, `## 2. ...`
- 公式用 LaTeX：`$$...$$`，关键符号用表格解释
- 每个概念先给直觉，再给严格定义
- 引用论文必须真实（见下方论文库），格式：`[作者, 年份](arXiv/url)`

## 论文/资料库（已调研，可直接引用）
- KV Cache 原理：TensorTonic "KV Cache in LLMs Explained" (tensortonic.com/llm-internals/kv-cache)；mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization（含完整 PyTorch 实现）
- PagedAttention：Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP'23, arXiv:2309.06180
- vLLM 官方：docs.vllm.ai（Paged Attention kernel 设计文档）；blog.vllm.ai (2023/06/20)
- GQA：Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245
- MQA：Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150
- FlashAttention：Dao et al., arXiv:2205.14135
- Orca（连续批处理）：Yu et al., arXiv:2208.14217
- CLA (Cross-Layer Attention, 再省 2×)：Brandon et al., NeurIPS 2024, arXiv:2405.12981
- 量化：GPTQ arXiv:2210.17323；AWQ arXiv:2306.00978
- KV 内存公式：`bytes = 2 × batch × seq_len × num_layers × num_kv_heads × head_dim × dtype_bytes`
  （参考 generalcompute.com、morphllm.com、tutorialq.com 的口径）
- aios（可运行的教学推理引擎，含 MHAKVCache 预分配 `(2, L, num_pages, page_size, kv_heads, head_dim)`）：`VLLM_learn/aios/python/aios/`

## 验证要求（每课必须）
1. 重写后必须 nbconvert 执行通过（`KMP_DUPLICATE_LIB_OK=TRUE`, kernel=uv_cuda, timeout=180）
2. 所有代码 cell 零 error 输出
3. app（如有）py_compile 通过 + headless 启动正常
4. 保留原有的 streamlit app 直跑入口结构（subprocess + st_runtime.exists() 保护）与 %%writefile 同步

## 节奏
- ch02-07 由主控亲自重写为金标准范例（必须先看，对齐后再批量）
- 其余每章派 1 个 agent，先用 2-3 个 websearch 调研该章主题论文，再按本规范重写