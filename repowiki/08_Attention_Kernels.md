# 08 · vLLM V1 Attention 内核:后端抽象、选择与 PagedAttention 元数据

> **版本**:基于 `vendor/vllm` 的 vLLM v0.23.0(dev/main)源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vendor/vllm/vllm/`(例如 `v1/attention/backend.py:55` = `vendor/vllm/vllm/v1/attention/backend.py` 第 55 行)。本仓库的 V1 attention 代码位于 `v1/attention/`(**不存在** `vllm/attention`);`v1/attention/backends/` 下每个子模块是一个后端。
> **一句话总结**:vLLM 把"注意力"抽象成**一对三**的插件体系——每个后端由一个 `AttentionBackend` 类(静态能力声明 + 校验)、一个 `AttentionMetadataBuilder`(把调度输出翻译成 kernel 需要的元数据)和一个 `AttentionImpl`(真正的 `forward` 实现)组成;引擎在启动时按 `AttentionSelectorConfig` 从 `AttentionBackendEnum` 全集里**按平台优先级逐个候选 validate**,选中最优后端,再由 `CommonAttentionMetadata` 统一承载跨后端的批次信息、由各后端 builder 派生各自的 kernel 元数据,最终在 GPU 前向里通过 **PagedAttention 的 block_table / slot_mapping 两级寻址**读写非连续的 KV cache。

---

## 一图看懂

```
模型层 (model_executor/layers/attention/attention.py)
   Attention.forward → 切 Q/K/V → 按 kv_sharing 与 forward_includes_kv_cache_update
                       决定是否先 unified_kv_cache_update (attention.py:538/558)
                                        │
                                        ▼
后端选择 (v1/attention/selector.py:103 get_attn_backend)
   AttentionSelectorConfig (selector.py:24, 14 个字段)
        │  current_platform.get_attn_backend_cls()  (platforms/cuda.py:401)
        ▼
   _get_backend_priorities (cuda.py:83)  →  候选列表 (SM100 vs 其它 / MLA vs 非 MLA)
        │  for 每个候选: backend_class.validate_configuration (cuda.py:385)
        ▼   ImportError 或返回非空 reasons 者淘汰;选 priority 最小的 (cuda.py:457)
   选定后端类 → 每个 KV cache group 实例化 (v1/worker/gpu/attn_utils.py:134)

┌────────────────────────── 后端三件套 (每个 AttentionGroup) ───────────────────────┐
│ AttentionGroup (v1/worker/utils.py:241)                                          │
│  ├─ backend           : type[AttentionBackend]     (静态能力 / 校验 / 形状)      │
│  ├─ metadata_builders : AttentionMetadataBuilder   (调度输出 → kernel 元数据)     │
│  │     (v1/worker/utils.py:253 create_metadata_builders, ubatch 可多个)          │
│  └─ impl              : AttentionImpl              (真正的 forward)              │
└──────────────────────────────────────────────────────────────────────────────────┘

元数据装配 (每步前向之前, gpu/attn_utils.py:591 build_attn_metadata)
   调度输出 ──► CommonAttentionMetadata (backend.py:463, 跨后端公共字段)
                  │  attn_utils.py:639 构造一次
                  ▼
             每个 group 的 builder.build(common_prefix_len, common_attn_metadata)
                  │  backend.py:750   → 各自后端私有 Metadata (e.g. FlashAttentionMetadata)
                  ▼
             以 layer_name → metadata 的 dict 存进 attn_metadata (attn_utils.py:679)

GPU 前向 (FlashAttentionImpl.forward, flash_attn.py:970)
   ┌─ 若 forward_includes_kv_cache_update=False: 先 do_kv_cache_update
   │      → reshape_and_cache_flash (flash_attn.py:1258)  写 K/V 进 cache
   │      → unified_kv_cache_update (attention.py:538/558 触发)
   └─ flash_attn_varlen_func(..., block_table=..., seqused_k=..., cu_seqlens_q=...)
         ↑ kernel 按 block_table 读不连续的页;按 cu_seqlens 做 varlen 注意力
```

---

## 1. 总体结论:为什么 V1 把 attention 拆成后端插件

V0 时代 attention 内核是"靠 `AttentionBackend` 三件套(**backend / metadata_builder / impl**)平铺在 `vllm/attention/` 下、层层 if-else 手写选择"。V1 把它彻底模块化,原因有三个:

1. **kernel 竞争太激烈**:FlashAttention / FlashInfer / FlashMLA / Triton / CUDA Graphs 各自对 head_size、block_size、dtype、KV cache 布局(NHD vs HND)有完全不同的约束。把这些约束收敛成**每个后端自己声明的一组 `supports_*` 能力**,校验逻辑就再也不用散落在各调用点。
2. **元数据是后端的延伸**:不同的 kernel 需要不同的批次元数据(FA3 要 `scheduler_metadata` 做 AOT 调度,MLA 要 `reorder_batch`,稀疏后端要 top-k 索引)。元数据怎么 build 理应跟着后端走,于是 `AttentionMetadataBuilder` 成为三件套之一。
3. **KV cache 形状是后端定义的**:不同后端对 K/V 的打包方式不同(FA 把 K/V 拼进 content 维成 `(B,H,N,2D)`,MLA 只存压缩后的 latent `(B,N,H_dim)`,Mamba 干脆存状态)。KV cache 管理器(文档 02)靠 `get_kv_cache_shape` / `get_kv_cache_stride_order` 拿到每个后端的物理布局,而不是硬编码。

三件套的职责边界(`v1/attention/backend.py`):

| 件 | 类 | 职责 |
|---|---|---|
| 声明 | `AttentionBackend`(`backend.py:55`) | 静态能力(`supports_*`)、`get_kv_cache_shape`、`get_name`、`get_impl_cls`/`get_builder_cls` |
| 构建 | `AttentionMetadataBuilder`(`backend.py:678`) | `build()` 把 `CommonAttentionMetadata` 转成后端私有 metadata;CUDA Graph 捕获 / 投机解码 / 级联注意力的变体入口 |
| 执行 | `AttentionImpl`(`backend.py:963`) | `forward()` 调 kernel 算注意力;`do_kv_cache_update()` 写 KV cache(当 `forward_includes_kv_cache_update=False` 时被单独调用) |

> **与 V0 的历史对照**:V0 的 `AttentionBackend`(`vllm/attention/backends/`)同样是三件套结构,但 V1 把它搬进 `v1/attention/`,并新增了两件 V0 没有的东西:①`AttentionBackendEnum` 注册表 + 平台级优先级选择(`platforms/cuda.py`);②`CommonAttentionMetadata` 这种"先跨后端公共、后按后端私有"的两级 metadata 体系。

---

## 2. AttentionBackend ABC:核心能力查询族

`AttentionBackend`(`backend.py:55`)是一个几乎全静态方法的 ABC。它不实例化、不带状态,纯靠**一组查询方法**描述"这个后端能干什么"。全部能力声明:`backend.py:58-452`。

### 2.1 特性开关族(逐个 is_mla / supports_*)

所有开关默认 `False`,后端按需覆写:

| 查询方法 | 行号 | 含义 |
|---|---|---|
| `is_mla()` | `backend.py:265` | 是否 Multi-Head Latent Attention(DeepSeek 系)。`MLACommonBackend` 覆写为 True(`mla_attention.py:1426`) |
| `is_sparse()` | `backend.py:281` | 是否 top-k 稀疏注意力 |
| `supports_sink()` | `backend.py:269` | 是否支持 attention sink(固定起始 token 始终参与注意力) |
| `supports_sliding_window()` | `backend.py:289` | 是否支持滑动窗口 |
| `supports_non_causal()` | `backend.py:293` | 是否支持 decoder 路径内的双向注意力(不同于 `ENCODER_ONLY` 的独立执行模型) |
| `supports_batch_invariance()` | `backend.py:304` | 是否批次不变(CUDA Graph 友好:结果与 batch 组成无关) |
| `supports_kv_connector()` | `backend.py:308` | 是否支持跨机 KV Connector(默认 True) |
| `supports_mm_prefix()` | `backend.py:277` | 是否支持多模态 PrefixLM 双向区间(FA 需 v4,`flash_attn.py:241`) |
| `supports_per_head_quant_scales()` | `backend.py:285` | 是否支持 per-head 量化 scale |
| `supports_alibi_sqrt()` | `backend.py:273` | ALiBi sqrt 变体 |
| `supports_pcp()` | `backend.py:327` | Prefill Context Parallelism(转发给 `impl.supports_pcp`) |
| `is_ssm()` | `backend.py:451` | 是否状态空间模型(Mamba 系) |
| `supports_device_cpu_query_lens_mismatch()` | `backend.py:312` | 是否容忍 GPU/CPU 的 query 长度不一致(自适应验证需要;SSM 后端主动拒绝,`backend.py:324`) |

### 2.2 形状 / 数据类型族

- `supports_dtype`(`backend.py:170`):dtype ∈ `supported_dtypes`。基类默认 `[float16, bfloat16]`(`backend.py:58`)。
- `supports_kv_cache_dtype`(`backend.py:174`):kv cache dtype ∈ `supported_kv_cache_dtypes`(默认 `auto / float16 / bfloat16`,`backend.py:59`)。
- `supports_head_size`(`backend.py:165`):由 `get_supported_head_sizes()`(`backend.py:161`,空列表 = 不限制)决定。
- `supports_block_size`(`backend.py:182`):对 `get_supported_kernel_block_sizes()`(`backend.py:69`,默认 `[MultipleOf(1)]`)逐个检查——`int` 要求精确相等,`MultipleOf(n)` 只要求整除(hybrid-blocks 下框架 block size 只需是 kernel 要求的倍数,`backend.py:193-196`)。
- `get_supported_kernel_block_sizes_for_config`(`backend.py:73`):给定具体 engine 配置(如 FA 的 SM90 FP8 路径)再算一次,让 block size 可以依赖运行时事实。
- `get_preferred_block_size`(`backend.py:215`):当默认 block size 不被支持时,返回支持列表里最小的那个。
- `get_kv_cache_shape`(`backend.py:96` 抽象):返回 cache 的逻辑 shape,后端自定义。
- `get_kv_cache_block_dim`(`backend.py:106`):用哨兵 `_S = 1234567` 塞进 shape 再 `shape.index(_S)`,**自动探测"哪个维度是 block 索引"**(各后端维序不同,不能硬编码)。
- `get_kv_cache_stride_order`(`backend.py:126` 抽象,默认抛 `NotImplementedError`):返回逻辑维到物理维的排列。逻辑 shape 是 `[num_blocks, num_heads, block_size, 2*head_size]`,返回 `(0,2,1,3)` 即物理布局为 `[num_blocks, block_size, num_heads, 2*head_size]`;`include_num_layers_dimension=True` 时在逻辑 shape 前追加 layers 维。若抛 `NotImplementedError`,物理布局 = 逻辑布局。
- `indexes_kv_by_block_stride`(`backend.py:233`):判断物理布局是否 block 最外层(`layered_stride_order[0] != 0`)——是则容忍非连续 block 维,从而允许页面填充(页面 padding)与跨层统一 KV 布局;否则(如 MLA 的恒等排列,`mla_attention.py:1419`)返回 False。
- `get_required_kv_cache_layout`(`backend.py:447`):后端需要强制 `NHD` / `HND` 布局时返回,选择逻辑会据此调用 `set_kv_cache_layout`(`selector.py:212-216`)。

### 2.3 校验总入口:validate_configuration(`backend.py:362`)

`validate_configuration(...)` 是选择器的**唯一裁决入口**,输入一组配置 + `device_capability`,输出**无效原因字符串列表**(空列表 = 通过)。它把 2.1/2.2 的所有查询串成 20 多个 check(`backend.py:382-444`):

```python
if not cls.supports_head_size(head_size):        invalid_reasons.append("head_size not supported")
if not cls.supports_dtype(dtype):                invalid_reasons.append("dtype not supported")
if not cls.supports_kv_cache_dtype(kv_cache_dtype): invalid_reasons.append("kv_cache_dtype not supported")
if not cls.supports_block_size(block_size):      invalid_reasons.append("block_size not supported")
if use_mm_prefix and not cls.supports_mm_prefix(): invalid_reasons.append("partial multimodal token full attention not supported")
if use_mla != cls.is_mla():                      # 双向校验:必须且只能 MLA 对 MLA
    invalid_reasons.append("MLA not supported" if use_mla else "non-MLA not supported")
if has_sink and not cls.supports_sink():         invalid_reasons.append("attention sinks not supported")
if use_sparse != cls.is_sparse():                # 稀疏同理,双向
...
```

几个值得注意的点:

- **`use_mla` / `use_sparse` 是双向校验**(`backend.py:395-406`):模型要求 MLA 而后端不是 MLA 报"MLA not supported",反过来模型非 MLA 而后端是 MLA 报 "non-MLA not supported"——防止稀疏/MLA 专用后端被误用于普通模型。
- **`supports_combination` 钩子**(`backend.py:347`,默认返回 None = 不拦):给跨维度的组合约束留后门。FA 用它拦截"sink 在 compute capability < 9.0"、"FP8 KV cache 需要 FA3(SM90)/FA4(SM100)"、"mm_prefix 需要 FA4"(`flash_attn.py:267-289`)。
- 返回值是 **`list[str]` 而非 bool**:既让选择器能打出每个候选的淘汰原因(日志),也让"恰好只有 block_size 不合适"这类情况能被单独识别——`get_attn_backend_cls` 用 `reasons == ["block_size not supported"]` 来对用户显式指定的 `--block-size` 发警告(`cuda.py:467-484`)。

### 2.4 实现侧基类:AttentionImplBase / AttentionImpl / MLAAttentionImpl

- `AttentionImplBase`(`backend.py:864`):不带 `forward` 的公共基类,把 **DCP / PCP 上下文**从 `__new__` 里统一塞给所有实现(`backend.py:931-957`,`dcp_world_size`/`dcp_rank` 等 6 个字段),并声明 `can_return_lse_for_decode`(`backend.py:887`)、`lse_base_on_e`(`backend.py:898`,DCP 合并 kernel 靠它区分自然对数/以 2 为底)等能力。
- `AttentionImpl`(`backend.py:963`):标准注意力实现。抽象 `forward`(`backend.py:990`)签名带 `output`/`output_scale`/`output_block_scale`;另有一组**算子融合钩子**:`fused_output_quant_supported`(`backend.py:1004`)、`fused_qk_norm_rope_kvcache_supported`(`backend.py:1018`)、`do_qk_norm_rope_kvcache_update`(`backend.py:1034`)、`do_rope_and_kv_cache_update`(`backend.py:1057`)——供编译 pass 把 QKNorm/RoPE/KV cache 写入融合进自定义算子。
- `MLAAttentionImpl`(`backend.py:1077`):MLA 专用,forward 拆成 `forward_mha`(prefill,`backend.py:1108`)与 `forward_mqa`(decode,`backend.py:1123`),并提供默认 `do_kv_cache_update`(`backend.py:1146`,内部调 `ops.concat_and_cache_mla`)。

---

## 3. 后端注册与选择

### 3.1 AttentionBackendEnum:注册表即全集(`backends/registry.py`)

`AttentionBackendEnum`(`registry.py:34`)的**枚举值本身就是默认类路径字符串**(如 `FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"`,`registry.py:44`)。枚举成员即全集,按用途分组:

| 类别 | 成员 | 枚举行号 |
|---|---|---|
| NVIDIA 标准 | `FLASH_ATTN` / `FLASH_ATTN_DIFFKV`(diff-KV) | `registry.py:44-47` |
| Triton | `TRITON_ATTN` / `TRITON_ATTN_DIFFKV` | `registry.py:48-51` |
| AMD(ROCm / AITER) | `ROCM_ATTN` / `ROCM_AITER_MLA` / `ROCM_AITER_TRITON_MLA` / `ROCM_AITER_FA` / `ROCM_AITER_MLA_SPARSE` / `ROCM_AITER_UNIFIED_ATTN` | `registry.py:52-62,120-123` |
| Intel XPU | `XPU_MLA_SPARSE` | `registry.py:63` |
| FlashInfer | `FLASHINFER` / `FLASHINFER_MLA` / `FLASHINFER_MLA_SPARSE` / `FLASHINFER_MLA_SPARSE_SM120` | `registry.py:65-79` |
| MLA 系 | `TOKENSPEED_MLA` / `TRITON_MLA` / `CUTLASS_MLA` / `FLASHMLA` / `FLASHMLA_SPARSE` / `FLASH_ATTN_MLA` / `FLASH_ATTN_MLA_SPARSE` / `CPU_MLA` | `registry.py:69-100,125` |
| DeepSeek V4 稀疏 | `FLASHMLA_SPARSE_DSV4` / `FLASHINFER_MLA_SPARSE_DSV4` / `ROCM_FLASHMLA_SPARSE_DSV4` | `registry.py:87-96` |
| 其它硬件/场景 | `TORCH_SDPA`(仅 ViT)、`NO_ATTENTION`、`FLEX_ATTENTION`、`HPC_ATTN`、`CPU_ATTN`、`TURBOQUANT` | `registry.py:64,112-119,124-126` |
| 第三方扩展 | `CUSTOM = None`(占位,使用前必须注册) | `registry.py:129` |

三个关键机制:

- **运行时覆写**:`register_backend(AttentionBackendEnum.FLASH_ATTN, ...)`(`registry.py:242`)可把任意枚举重定向到自定义类路径,写入 `_ATTN_OVERRIDES` 字典(`registry.py:238`);`get_path`(`registry.py:131`)/`get_class`(`registry.py:150`)都**尊重覆写**。`CUSTOM` 未注册就调用会直接抛 ValueError(`registry.py:141-145`)。
- **友好报错**:元类 `_AttentionBackendEnumMeta`(`registry.py:18`)在 `AttentionBackendEnum["xxx"]` 查不到时抛 ValueError 并列出全部合法选项——这就是你在 CLI 拼错 `--attention-backend` 时看到的错误来源。
- **Mamba 独立注册表**:`MambaAttentionBackendEnum`(`registry.py:175`)单独列 MAMBA1/MAMBA2/SHORT_CONV/LINEAR/GDN_ATTN,与注意力后端完全解耦。

### 3.2 平台优先级:_get_backend_priorities(`platforms/cuda.py:83`)

选择发生在 `_cached_get_attn_backend`(`selector.py:193`)→ `current_platform.get_attn_backend_cls`(`cuda.py:401`)。CUDA 平台的关键是**按 `device_capability`(计算能力)和 `use_mla` 分叉出有序候选列表**:

```python
if use_mla:
    if device_capability.major == 10:      # SM100 / Blackwell
        return [FLASHINFER_MLA, TOKENSPEED_MLA, CUTLASS_MLA,
                FLASH_ATTN_MLA, FLASHMLA, TRITON_MLA, *sparse_backends]   # cuda.py:118-129
    elif device_capability.major == 12:    # SM120
        return [TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]                  # cuda.py:130-134
    else:                                  # SM90 及以下
        return [FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA,
                FLASH_ATTN_MLA_SPARSE, FLASHMLA_SPARSE]                   # cuda.py:136-143
else:
    if device_capability.major == 10 and not use_non_causal:
        return [FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]  # cuda.py:148-155
    return [FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]      # cuda.py:156-163
```

要点:

- **SM100 是非 MLA 的分水岭**:默认路径 `FLASHINFER` 排第一(SM100f 上 TRTLLM causal 内核最快),但 `use_non_causal` 时**主动把它降级**——因为 SM100f 的 non-causal 走 FlashInfer 的 cutlass 路径"已知有问题",注释写得很直白(`cuda.py:145-147`)。
- **MLA 在 SM100 上还看 kv cache dtype 与 head 数**:`FLASHINFER_MLA_SPARSE` vs `FLASHMLA_SPARSE` 的顺序取决于 `kv_cache_dtype` 是否 FP8、`num_heads` 是否 ≤16(`cuda.py:96-116`);`TOKENSPEED_MLA` 排在第二位但注明"仅 R1 dims + FP8 KV",靠 `supports_combination` 淘汰。
- 候选里嵌入了**注释即文档**:每个取舍都写了性能依据(如 `cuda.py:120-122` 的 benchmark 结论"wins past bs≈8, regresses at bs≤2")。

### 3.3 逐候选 validate 与最终选择(`get_attn_backend_cls`,`cuda.py:401`)

```python
# 1) 用户显式指定 --attention-backend:只验这一个,不通过直接抛错
if selected_backend is not None:                    # cuda.py:411
    invalid_reasons = backend_class.validate_configuration(...)
    if invalid_reasons:
        raise ValueError(f"Selected backend {selected_backend} is not valid...")  # cuda.py:421

# 2) 未指定:get_valid_backends (cuda.py:363) 对优先级列表逐个 validate
for priority, backend in enumerate(backend_priorities):      # cuda.py:382
    try:
        invalid = backend_class.validate_configuration(...)
    except ImportError:
        invalid = ["ImportError"]                            # 没装该 kernel 库 → 淘汰
    if invalid: 记录 reasons
    else:       收进 valid_backends_priorities

# 3) 选 priority 最小的合法候选
selected_candidate = min(valid_backends_priorities, key=lambda c: c.priority)  # cuda.py:457
```

- 校验失败不等于报错:ImportError(没装 flashinfer 等可选依赖)只是普通淘汰原因(`cuda.py:389-390`),全部淘汰才抛 `"No valid attention backend found"`(`cuda.py:449-453`)。
- 结果通过 `_backend_cls_path`(`cuda.py:166`)以 `"module.ClassName"` 字符串返回,由 `selector.py:209` 用 `resolve_obj_by_qualname` 解析成类。
- 选择结果按 `(head_size, dtype, kv_cache_dtype, ...)` 组成的 `AttentionSelectorConfig` 做 **`functools.cache`**(`selector.py:192`),同一配置只选一次。
- 除全局后端外,还支持 `--attention-backend-per-kind` 按 KV cache 种类(`get_attn_spec_kind`,`selector.py:63`:`FULL_ATTENTION`/`SLIDING_WINDOW`/`MLA_ATTENTION`/`CROSS_ATTENTION` 等)逐组覆盖(`selector.py:176-183`)。

---

## 4. FlashAttention 后端(默认路径的解剖标本)

`FlashAttentionBackend`(`flash_attn.py:77`)是分析整个体系最好的样本——它覆盖了全部抽象方法、还带一堆运行时条件。逐一对照三件套:

### 4.1 能力声明(静态)

| 能力 | 值 | 行号 |
|---|---|---|
| `supported_dtypes` | `[float16, bfloat16]` | `flash_attn.py:78` |
| `supported_kv_cache_dtypes` | `auto / float16 / bfloat16 / fp8 / fp8_e4m3` | `flash_attn.py:79-85` |
| `get_name()` | `"FLASH_ATTN"` | `flash_attn.py:145` |
| `get_supported_kernel_block_sizes()` | 默认 `[MultipleOf(16)]`;**SM90 + FP8 KV + head_size=512 + FA4** 时改成精确 `[64]`(该 TMA kernel 的页契约) | `flash_attn.py:108-114` |
| `get_impl_cls()` / `get_builder_cls()` | `FlashAttentionImpl` / `FlashAttentionMetadataBuilder` | `flash_attn.py:176-181` |
| `supports_compute_capability` | 要求 **≥ 8.0**(Ampere 起) | `flash_attn.py:251` |
| `supports_sliding_window` / `supports_non_causal` / `supports_attn_type`(四种全支持) / `supports_batch_invariance` | 全部 True | `flash_attn.py:149-168` |

**head_size 限制**(`flash_attn.py:221-228`):`head_size % 8 != 0` 直接拒;≤256 全部接受;>256 只有装了 FA4 才接受且上限 512。这就是 512 头尺寸模型需要 FA4 的源码级依据。

**block size 约束的本质**:`MultipleOf(16)`(`flash_attn.py:114`)意味着框架只要保证 block_size 是 16 的倍数即可(默认 16 天然满足),而 SM90 FP8 路径下的精确 `64` 是**内核的固定页大小**(`flash_attn.py:103-104` 注释:64-token TMA tile)。框架据此通过 `get_preferred_block_size` 把默认 block size 顶到 ≥64(`flash_attn.py:127-132`)。

### 4.2 KV cache 形状与 stride(`get_kv_cache_shape` / `get_kv_cache_stride_order`)

```python
# flash_attn.py:184-194
if block_size % 16 != 0: raise ValueError("Block size must be a multiple of 16.")
return (num_blocks, num_kv_heads, block_size, 2 * head_size)   # K、V 拼进 content 维

# flash_attn.py:197-218  get_kv_cache_stride_order
# NHD 布局(默认):逻辑 (B,H,N,2D) → 物理 (B,N,H,2D)  → 返回 (0,2,1,3)
# NHD 布局 + layers:              物理 (B,L,N,H,2D)  → 返回 (1,0,3,2,4)
# HND 布局(VLLM_KV_CACHE_LAYOUT=HND):物理 = 逻辑        → 返回 (0,1,2,3)
# HND 布局 + layers:              物理 (B,H,L,N,2D)    → 返回 (1,2,0,3,4)
```

- **K 和 V 被压进同一个 content 维**:逻辑形状最后一维是 `2 * head_size`,前半段存 K、后半段存 V。`forward` 里靠 `kv_cache.transpose(1,2).split(self.head_size, dim=-1)` 拆开(`flash_attn.py:1037`)。
- 布局由 `VLLM_KV_CACHE_LAYOUT`(`envs.py:241`,`utils.py:154 get_kv_cache_layout`)或 KV Connector 决定;NHD 时 block 维不再是物理最高维,`get_kv_cache_block_dim` 会自动探测出正确下标。
- **stride 规范化**:拆出来的 K/V cache 若是退化 stride(如 `num_kv_heads=1` 的 TP 场景),FA3/4 的 TMA 需要 ≥16 字节对齐,`forward` 里用 `canonicalize_singleton_dim_strides` 修复(`flash_attn.py:1041-1054`)。

### 4.3 forward 是否内置 KV 更新?不内置

`forward_includes_kv_cache_update: bool = False`(`flash_attn.py:124`)。含义与后果:

- 对比基类默认 `True`(`backend.py:66`):基类假设"attention forward 顺带把 KV 写进 cache",而 **FlashAttention 的 `forward` 只读 cache、不写 cache**,写 K/V 是独立的一步。
- 后果链:Attention 层在 `forward` 里看到 `not forward_includes_kv_cache_update` 就先触发 `unified_kv_cache_update`(`attention.py:538/558`)→ 自定义算子分发到 `FlashAttentionImpl.do_kv_cache_update`(`flash_attn.py:1233`)→ `reshape_and_cache_flash`(`flash_attn.py:1258`)按 `slot_mapping` 做 scatter 写入。编码器注意力则直接跳过(`flash_attn.py:1241-1244`,encoder 用原生 Q/K/V 不缓存)。
- 哪些后端内置、哪些不内置?Triton(`triton_attn.py:318`)、FlashInfer(`flashinfer.py:569`)、ROCm(`rocm_attn.py:217`)、CPU(`cpu_attn.py:45`)、Flex(`flex_attention.py:98`)、TurboQuant(`turboquant_attn.py:127`)都是 False;唯一 True 的是 `hpc_attn.py:283`。这个开关决定 `forward` 是否要在入口处理 `kv_cache_dummy_dep` 依赖,以保 CUDA Graph 的依赖序正确。

### 4.4 forward 主路径(`flash_attn.py:970`)

非 cascade、非 DCP 的主路径是**单次 `flash_attn_varlen_func` 调用**(`flash_attn.py:1176-1201`),核心入参全部来自 metadata:

- `cu_seqlens_q` = `query_start_loc`(每个请求 query 的区间边界,`flash_attn.py:1062`)
- `seqused_k` = `seq_lens`(每个请求已算出的上下文长度,`flash_attn.py:1063`)
- `max_seqlen_q` / `max_seqlen_k` = `max_query_len` / `max_seq_len`(`flash_attn.py:1064-1065`)
- `block_table` = 非连续页表(`flash_attn.py:1066`,见 §7)
- `scheduler_metadata` = FA3 的 AOT 调度元数据(`flash_attn.py:1067`,CUDA Graph 时由 `_store_scheduler_metadata` 预分配并写回固定 buffer,`flash_attn.py:452-465`)
- 量化时:q/k/v_descale 按 `(num_reqs, num_kv_heads)` expand(`flash_attn.py:1069-1077`)

> **cascade 与 DCP 分支**:`common_prefix_len > 0` 时走 `cascade_attention`(`flash_attn.py:1205`,把公共前缀单独算一次避免重复 flops);`dcp_world_size > 1` 时走 `_forward_with_dcp`(`flash_attn.py:1269`,本地只算局部 KV + 跨 rank all-gather query 与 LSE 合并)。两条都是"分布式/优化"变体,主线不必深挖。

---

## 5. MLA 与其它后端简述

### 5.1 MLA(DeepSeek 系)家族

MLA 不再存 K/V 本身,而是存**压缩后的潜变量**。公共基类 `MLACommonBackend` 在 `model_executor/layers/attention/mla_attention.py`(注意不在 `v1/attention/backends/` 下):

- `get_kv_cache_shape` 返回 `(num_blocks, block_size, head_size)`(`mla_attention.py:1401`)——比 FA 少了 num_kv_heads 维(MLA 的 KV 头数恒为 1),也没有 K/V 拼接。
- `get_supported_head_sizes()` 返回 `[320, 576]`(`mla_attention.py:1422`),即 DeepSeek-V3/R1 与 V3.1+ 的典型潜变量维度;`is_mla()` 为 True(`mla_attention.py:1426`)。
- forward 由 `MLAAttentionImpl` 拆成 `forward_mha`(prefill)/ `forward_mqa`(decode)(`backend.py:1108/1123`),KV 写入走 `ops.concat_and_cache_mla`(`backend.py:1159`)。
- 具体实现按硬件/供应商分叉:FlashMLA(DeepSeek 官方内核)、FlashInfer MLA、Cutlass MLA、Triton MLA、FlashAttn MLA、Tokenspeed MLA,以及 SM100/120 上的稀疏变体(枚举全集见 §3.1 表)。

### 5.2 其它值得知道的后端

| 后端 | 类 / 行号 | 一句话 |
|---|---|---|
| FlashInfer | `FlashInferBackend`(`flashinfer.py:391`) | NVIDIA 生态第二主力,`FLASHINFER` 是 **SM100 非 MLA 默认首选**(`cuda.py:150`);内含 TRTLLM prefill / decode 等多套 kernel,`get_kv_cache_shape` 在 `flashinfer.py:468` |
| Triton | `TritonAttentionBackend`(`triton_attn.py:276`) | 纯 Triton 实现,无额外依赖,`get_kv_cache_shape` 在 `triton_attn.py:341` |
| FlexAttention | `FlexAttentionBackend`(`flex_attention.py:86`) | 基于 PyTorch FlexAttention,支持 mask_mod 自定义 mask |
| TurboQuant | `TurboQuantAttentionBackend`(`turboquant_attn.py:123`) | 量化场景专用 |
| ROCm / AITER | `RocmAttentionBackend`(`rocm_attn.py:165`) | AMD GPU;**是 V1 里唯一用 `PagedAttention` ops 封装的后端**(`rocm_attn.py:34`) |
| CPU | `CPUAttentionBackend`(`cpu_attn.py`) | 无 GPU 时兜底 |
| NoAttention | `NoAttentionBackend`(`no_attention.py`) | 某些模型层没有注意力时的空实现 |

> **历史对照**:V0 里只有 FLASH_ATTN / FLASHINFER / TRITON_ATTN / ROCM_ATTN 四个主力;V1 的枚举膨胀到 30+ 个成员,主要增量来自 MLA 生态(DeepSeek)与各硬件供应商的稀疏内核——这正是 §2 那套能力声明体系存在的意义:不把新内核写死进选择器,而是让每个新后端自己声明能力、自己参与 validate。

---

## 6. Metadata 体系:从公共到私有的两级结构

### 6.1 CommonAttentionMetadata(`backend.py:463`)

一个 `@dataclass`,承载**跨后端、跨层共享**的批次信息,每步在 `build_attn_metadata` 里按 KV cache group 构造一次(`attn_utils.py:639`),然后喂给该 group 的每个 builder(`attn_utils.py:659-678`)。核心字段:

| 字段 | 行号 | 含义 |
|---|---|---|
| `query_start_loc` / `query_start_loc_cpu` | `backend.py:471-472` | `(batch_size+1,)`,每个请求在拼接 query 张量里的起止位置 |
| `seq_lens` | `backend.py:475` | `(batch_size,)`,每请求已计算 token 数(device 端) |
| `num_reqs` / `num_actual_tokens` / `max_query_len` / `max_seq_len` | `backend.py:478-486` | 请求数 / 去 padding 的 token 总数 / 最长 query / 最长上下文 |
| `block_table_tensor` / `slot_mapping` | `backend.py:488-489` | 页表与槽位映射(§7 的主角) |
| `causal` | `backend.py:491` | 因果掩码,可 bool 或逐请求 Tensor |
| `positions` | `backend.py:506` | 每个 token 的位置号(供 DeepSeek V4 稀疏层预计算位置相关元数据) |
| `is_prefilling` | `backend.py:511` | 逐请求是否仍在 prefill 阶段 |
| `seq_lens_cpu_upper_bound` | `backend.py:516` | CPU 端 seq_lens 上界(异步 spec decode 下是乐观值,不适合要求精确上下文长度的 kernel) |
| `mm_req_doc_ranges` | `backend.py:522` | PrefixLM 多模态双向区间 |
| `rswa_prefix_lens` | `backend.py:528` | Reference Sliding Window Attention 的每请求前缀长度 |

配套方法:`naive_query_lens`(`backend.py:550`,用相邻 start_loc 相减推 query 长度)、`compute_num_computed_tokens`(`backend.py:587`,device 端算 `seq_lens - query_lens`)、`token_to_req_indices`(`backend.py:594`,用 `repeat_interleave` 建 token→请求 映射)、`unpadded`(`backend.py:622`,按实际 token 数切片,给非全量 CUDA Graph 的 spec-decode 用)。注意 `seq_lens_cpu` / `num_computed_tokens_cpu` 两个旧属性已被 `@deprecated`(`backend.py:557-585`):异步调度下隐式 H→D 同步会破坏性能。

### 6.2 AttentionMetadataBuilder(`backend.py:678`)

每个后端一个 builder 子类,核心抽象是 `build(common_prefix_len, common_attn_metadata, fast_build)`(`backend.py:750`)。围绕它衍生出一整套生命周期入口:

- `build_for_cudagraph_capture`(`backend.py:784`):CUDA Graph 捕获专用,默认直接 `build(common_prefix_len=0, ...)`,子类可覆写。
- `build_for_drafting`(`backend.py:796`):投机解码的 draft 模型入口,强制 `fast_build=True`(draft 的 metadata 只用几步,不值得慢建)。
- `update_block_table`(`backend.py:769`):KV cache 共享分组时,复用同一 metadata、只换 block table 的快速路径(FlashAttention 实现于 `flash_attn.py:829`)。
- `update_draft_decode_metadata`(`backend.py:818`):融合 draft 循环里**原地更新**每步变化的 draft 元数据,要求 emit capture-safe 操作(FA 实现于 `flash_attn.py:840`)。
- `use_cascade_attention`(`backend.py:828`):判断本批是否值得用级联注意力。
- 类级配置:`_cudagraph_support`(`backend.py:681`)、`reorder_batch_threshold`(`backend.py:685`,把特定 query 长度拉到 batch 前的阈值,`_init_reorder_batch_threshold` 按 spec 配置抬高,`backend.py:717-747`)、`supports_update_block_table`(`backend.py:688`)、`requires_block_table_width`(`backend.py:690`)、`supports_draft_decode_metadata_update`(`backend.py:693`)。

### 6.3 AttentionCGSupport(`backend.py:661`)

CUDA Graph 支持等级枚举,`AttentionMetadataBuilder._cudagraph_support` 声明、`get_cudagraph_support` 查询(`backend.py:709`):

```
ALWAYS = 3                      # 总是支持,含混合 prefill-decode
UNIFORM_BATCH = 2               # 仅 query 长度一致时(如 spec-decode 的 1+n 长度)
UNIFORM_SINGLE_TOKEN_DECODE = 1 # 仅纯 query_len==1 的 decode
NEVER  = 0                      # 不支持
```

FlashAttention 的取值(`flash_attn.py:401-405`):FA3 → `ALWAYS`(Hopper 的 scheduler_metadata 支持任意形状);FA2 → `UNIFORM_BATCH`(FA2 对 max_query_len=1 有特判 packed-GQA 优化,混合 prefill-decode 的图不可复用)。这个等级直接驱动 CUDA Graph 编译器选择 FULL_AND_PIECEWISE 等捕获策略。

### 6.4 装配流程(串起来)

`build_attn_metadata`(`attn_utils.py:591`)每步:遍历 KV cache group → 构造一份 `CommonAttentionMetadata`(`attn_utils.py:639`)→ 对 group 内每个 `AttentionGroup` 取其 builder(`get_metadata_builder`,`worker/utils.py:285`)→ `build`(`attn_utils.py:674`,CUDA Graph 捕获时走 `build_for_cudagraph_capture`,`attn_utils.py:662`)→ 以 `layer_name → metadata` 存进 dict(`attn_utils.py:679-680`)。**同一份 Common 元数据被多个 group 复用、同一 batch 的不同层共享同一份私有 metadata**——这就是两级结构省内存的关键。

`AttentionGroup`(`worker/utils.py:241`)把"后端类 + 层名列表 + kv_cache_spec + metadata_builders"打包成一个单元,`create_metadata_builders`(`worker/utils.py:253`)在初始化时按 kernel block size(虚拟分块时小于框架 block size,`prepare_kernel_block_sizes`,`worker/utils.py:370`)实例化 builder;ubatching 时每个 ubatch 一个独立 builder(`worker/utils.py:246-251` 注释),避免 CUDA Graph 持久 buffer 冲突。`select_common_block_size`(`worker/utils.py:301`)负责在所有共用同一 cache 的后端里挑一个公共 kernel block size。

---

## 7. PagedAttention ops:非连续 KV 如何寻址

V1 的 `v1/attention/ops/paged_attn.py` 只有 51 行,是 `_custom_ops` 的薄封装(`paged_attn.py:1-51`)。**真正的寻址逻辑在内核里**,封装暴露的是两级映射:`block_table` 和 `slot_mapping`。

### 7.1 两级寻址模型

```
第一级: block_table (每个请求一行, 宽 = 该请求的块数)
   request i 的第 j 个逻辑块 ──► 物理块号 block_table[i][j]
   (逻辑块连续, 物理块可以散落任意位置)

第二级: slot_mapping (每个 token 一个槽位)
   token t 在 (物理块, 块内偏移) 中的绝对槽位 slot_mapping[t]
   slot = 物理块号 * block_size + 块内偏移   ← 写入 K/V 时按此 scatter
```

- **读路径(注意力)**:kernel 拿 `block_table[i][j]` 找到物理页、按块内位置取对应 token 的 K/V;`seqused_k`(每请求上下文长度)告诉 kernel 每行要读多少页。
- **写路径(缓存)**:`write_to_paged_cache`(`paged_attn.py:32`)把新的 K/V 按 `slot_mapping.flatten()` scatter 进 cache——slot_mapping 的**形状决定实际写入的 token 数**(`flash_attn.py:1253-1257` 注释:key/value 是 padded 的,但 slot_mapping 不是)。

### 7.2 工具函数

- `split_kv_cache`(`paged_attn.py:17`):把 `(2, num_blocks, num_kv_heads, head_size, block_size)` 形状的 cache 拆成 K、V 两个 view,并按 16 字节对齐把 K 的 head_size 拆成 `(head_size//x, -1, x)` 让内核能用向量化访存(`paged_attn.py:22-26`)。V1 里只有 ROCm 后端在用(`rocm_attn.py:34`)——NVIDIA 主路径的 FA/FlashInfer 各有自己的 cache 读取内核,不再需要这个拆解。
- `write_to_paged_cache`(`paged_attn.py:32`):薄封装 `ops.reshape_and_cache`,签名透传 kv_cache_dtype + k/v_scale(量化 KV 的 per-tensor scale 也由它写入)。

### 7.3 与 V0 PagedAttention 的关系

V0 里 `PagedAttention` 是个完整实现(`vllm/attention/ops/paged_attn.py`,提供 `forward` 全套内核)。V1 把它拆解:

- **非连续 KV 的"页"概念没变**,但页的物理布局由后端 `get_kv_cache_shape`/`get_kv_cache_stride_order` 决定,V1 的 KV cache 管理器(文档 02)按后端形状分配,`KVBlockZeroer`(`worker/utils.py:100`)用 Triton kernel 按 block 地址清零新块。
- **每步清零新分配的块**:调度器分配物理块后,`KVBlockZeroer.zero_block_ids`(`worker/utils.py:211`)对块 ID 列表做一次 Triton 3 维 grid launch(`(block_id, seg, chunk)`,kernel 见 `worker/utils.py:54-97`),保证块内旧数据不会泄漏给新请求——这比 V0 的"首次写入覆盖"更严格,也是 PagedAttention 正确性的隐藏前提。

---

## 8. 关键类 / 函数速查(文件:行号)

### `v1/attention/backend.py`(抽象层)
- `AttentionType`:32(DECODER/ENCODER/ENCODER_ONLY/ENCODER_DECODER);`MultipleOf`:48
- `AttentionBackend`:55;`forward_includes_kv_cache_update`:66;`get_supported_kernel_block_sizes`:69;`get_kv_cache_shape`:96;`get_kv_cache_block_dim`:106;`get_kv_cache_stride_order`:126;`get_supported_head_sizes`:161;`supports_head_size`:165;`supports_dtype`:170;`supports_kv_cache_dtype`:174;`supports_block_size`:182;`get_preferred_block_size`:215;`indexes_kv_by_block_stride`:233;`is_mla`:265;`supports_sink`:269;`supports_mm_prefix`:277;`is_sparse`:281;`supports_sliding_window`:289;`supports_non_causal`:293;`supports_batch_invariance`:304;`supports_kv_connector`:308;`supports_attn_type`:334;`supports_compute_capability`:343;`supports_combination`:347;`validate_configuration`:362;`get_required_kv_cache_layout`:447;`is_ssm`:451
- `AttentionMetadata`:455;`CommonAttentionMetadata`:463;`AttentionCGSupport`:661;`AttentionMetadataBuilder`:678
- `AttentionImplBase`:864;`AttentionImpl`:963;`MLAAttentionImpl`:1077;`subclass_attention_backend`:1169

### `v1/attention/backends/registry.py`
- `AttentionBackendEnum`:34;`get_path`:131;`get_class`:150;`is_overridden`:162;`MambaAttentionBackendEnum`:175;`_ATTN_OVERRIDES`:238;`register_backend`:242

### `v1/attention/selector.py`
- `AttentionSelectorConfig`:24;`get_attn_spec_kind`:63;`get_attn_backend`:103;`_cached_get_attn_backend`:193

### `platforms/cuda.py`
- `_get_backend_priorities`:83;`_backend_cls_path`:166;`get_valid_backends`:363;`get_attn_backend_cls`:401;`get_supported_vit_attn_backends`:499

### `v1/attention/backends/flash_attn.py`
- `FlashAttentionBackend`:77;`get_supported_kernel_block_sizes`:108;`get_kv_cache_shape`:184;`get_kv_cache_stride_order`:197;`supports_head_size`:221;`supports_combination`:255
- `FlashAttentionMetadata`:293;`FlashAttentionMetadataBuilder`:382;`build`:581;`update_block_table`:829;`update_draft_decode_metadata`:840
- `FlashAttentionImpl`:866;`forward`:970;`do_kv_cache_update`:1233;`_forward_with_dcp`:1269

### `model_executor/layers/attention/mla_attention.py`(MLA 公共实现)
- `MLACommonBackend`:1377;`get_kv_cache_shape`:1401;`get_supported_head_sizes`:1422;`is_mla`:1426
- `MLACommonDecodeMetadata`:1486;`MLACommonMetadata`:1496;`MLACommonMetadataBuilder`:1968;`MLACommonImpl`:2911

### `v1/worker/utils.py` 与 `v1/worker/gpu/attn_utils.py`
- `KVBlockZeroer`:100;`zero_block_ids`:211;`AttentionGroup`:241;`create_metadata_builders`:253;`get_metadata_builder`:285;`select_common_block_size`:301;`prepare_kernel_block_sizes`:370
- `build_attn_metadata`:591;`CommonAttentionMetadata` 构造点:639;`build_slot_mappings_by_layer`:580

### `v1/attention/ops/paged_attn.py`
- `PagedAttention`:15;`split_kv_cache`:17;`write_to_paged_cache`:32

### 相关环境变量 / 配置
- `--attention-backend` / `--attention-backend-per-kind`(按 `KVCacheSpecKind` 逐组指定,`selector.py:176-183`)。
- `--block-size`:显式指定时若排挤了更高优先级的后端会打警告(`cuda.py:467-484`)。
- `VLLM_KV_CACHE_LAYOUT`(`envs.py:241`,NHD/HND):决定 FA 的物理布局;`VLLM_BATCH_INVARIANT`(`envs.py:90`):开启后强制 batch 不变并禁用 FA3 的 AOT 调度(`flash_attn.py:604`)。

---

## 延伸阅读

- 论文:FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>
- 论文:PagedAttention 与 vLLM 的块式 KV 管理 <https://arxiv.org/abs/2309.06180>
- FlashInfer 项目主页(FA3/FA4 之外的第二大 NVIDIA 后端) <https://flashinfer.ai/>
- vLLM 官方文档:Attention 后端选择机制 <https://docs.vllm.ai/en/latest/design/attention_backend.html>
- 相关文档:本仓库 repowiki `02_*`(KV Cache Manager,`get_kv_cache_shape` 的消费方)、`03_Scheduler.md`(`allocate_slots`/`block_table` 的产生方)、`01_system_architecture.md`(GPUModelRunner 里 attention 三件套的位置)