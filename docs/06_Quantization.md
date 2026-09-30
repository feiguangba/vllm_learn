# 06 · vLLM 量化子系统:三层抽象、配置注入与 kernel 选择链

> **版本**:基于 vLLM v0.23.0(dev/main @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)) 源码精读整理。
> **路径约定**:下文所有 `文件:行号` 均相对 vLLM 包根目录 `vllm/`(在线查证: https://github.com/vllm-project/vllm/blob/967e104/vllm/<路径>#L<行号> ;例如 `model_executor/layers/quantization/fp8.py:92` = 上游源码中的 `vllm/model_executor/layers/quantization/fp8.py` 第 92 行)。
> **一句话总结**:vLLM 的量化是"**配置(QuantizationConfig)→ 方法(QuantizeMethodBase)→ kernel(LinearKernel)**"三层抽象:配置类从 checkpoint 或 CLI 参数实例化,通过 `get_quant_method()` 为每个算子分派对应的 Linear/MoE 方法,方法在 `create_weights()` 里用 vLLM 自定义参数(打包权重/scale/zero-point)建权重,加载后由 `process_weights_after_loading()` 重排成特定 kernel 需要的布局,最终在 `apply()` 里委托给 `choose_scaled_mm_linear_kernel` / `choose_mp_linear_kernel` 按平台优先级挑选出的 kernel 完成计算。

---

## 一图看懂

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ CLI:  --quantization auto_gptq / fp8 / ...   +   --quantization-config      │
  │        │  config/quantization.py:116 _ONLINE_SHORTHANDS / :158 resolve_...  │
  │        ▼                                                                    │
  │  ModelConfig.quantization (config/model.py:224)                              │
  │   └─ _verify_quantization (model.py:1199):                                   │
  │        · 读 checkpoint 的 quantization_config["quant_method"] (model.py:1208)│
  │        · 按 overrides 顺序调 override_quantization_method 探测 (model.py:1245)│
  │        · GPTQ checkpoint "gptq" → auto_gptq  (auto_gptq.py:219)              │
  │        · AWQ  checkpoint "awq"   → auto_awq   (auto_awq.py:260)              │
  │        ▼                                                                    │
  │  VllmConfig._get_quantization_config (config/vllm.py:743)                    │
  │   └─ get_quant_config (model_loader/weight_utils.py:240)                     │
  │        · hf_config.quantization_config → quant_cls.from_config (:287)        │
  │        · 或 online 量化 → OnlineQuantizationConfig (:320-327)                │
  │        · 或按 get_config_filenames() 找 json (weight_utils.py:353)           │
  └────────────────────────────────────────┬───────────────────────────────────┘
                                           ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  layer 构造: LinearBase.__init__ (layers/linear.py:229)                      │
  │   quant_method = quant_config.get_quant_method(layer, prefix) (linear.py:260)│
  │        │                                                                     │
  │        ├─ FP8  : Fp8LinearMethod   (quantization/fp8.py:239)                 │
  │        ├─ GPTQ : AutoGPTQLinearMethod (quantization/auto_gptq.py:306)        │
  │        ├─ AWQ  : AutoAWQMarlinLinearMethod (auto_awq.py:388) / Triton 回退   │
  │        └─ 无量化: UnquantizedLinearMethod (linear.py:166)                     │
  │   quant_method.create_weights(layer, ...)   ← 注册 qweight/g_idx/scales/...  │
  │        ▼                                                                     │
  │  load_weights → process_weights_after_loading() ← 打包/转置/重排 kernel 布局  │
  │        ▼                                                                     │
  │  forward → quant_method.apply(layer, x) → kernel.apply_weights(...)          │
  └────────────────────────────────────────┬───────────────────────────────────┘
                                           ▼
  kernel 选择:  kernels/linear/__init__.py
    · init_fp8_linear_kernel (:665) → choose_scaled_mm_linear_kernel (:601)
         per-tensor: _POSSIBLE_FP8_KERNELS (:397)   Marlin→FlashInfer→Cutlass→B12x→Torch→Humming
         per-block : _POSSIBLE_FP8_BLOCK_KERNELS (:430) FlashInferDeepGEMM→DeepGemm→Cutlass→...
    · GPTQ/AWQ : choose_mp_linear_kernel (:774) → _POSSIBLE_KERNELS (:474)
         CutlassW4A8→Machete→AllSpark→Marlin→Conch→Exllama→TritonW4A16→Humming
```

---

## 1. 总体结论:配置 → 方法 → kernel 三层抽象

vLLM 把所有"权重用低比特存、计算时解码"的逻辑收敛到三个明确的层,`linear.py`、`fused_moe.py` 里的算子本身完全不感知量化细节:

| 层 | 职责 | 抽象基类 | 代表实现 |
|---|---|---|---|
| **配置层** | 从 checkpoint/CLI 解析量化格式、声明硬件/精度约束 | `QuantizationConfig`(`quantization/base_config.py:87`) | `Fp8Config`、`AutoGPTQConfig`、`AutoAWQConfig` |
| **方法层** | 为某一类算子(Linear/MoE/Attention)建权重、加载后重排、执行前向 | `QuantizeMethodBase`(`base_config.py:20`)→ `LinearMethodBase`(`linear.py:125`) | `Fp8LinearMethod`、`AutoGPTQLinearMethod`、`AutoAWQMarlinLinearMethod` |
| **kernel 层** | 真正跑 GEMM 的 CUDA/Triton kernel,按平台+量化 key 挑选 | `ScaledMMLinearKernel` / `MPLinearKernel` | `MarlinFP8ScaledMMLinearKernel`、`CutlassFP8ScaledMMLinearKernel`、`MarlinLinearKernel`… |

三个关键设计决定:

1. **分派发生在算子构造时,而不是前向时**。`LinearBase.__init__` 里就调用一次 `quant_config.get_quant_method(self, prefix)`,拿到方法对象存进 `self.quant_method`,`forward` 时直接 `quant_method.apply(...)`(`linear.py:576`),前向路径零分支。
2. **权重以"量化专用参数"形态存在**。每个方法用自己的 vLLM 自定义 `Parameter` 子类(`PackedvLLMParameter`、`GroupQuantScaleParameter` 等,定义在 `model_executor/parameter.py`)注册 `qweight`/`g_idx`/`scales`/`qzeros`,自带 `weight_loader` 负责 TP 切分与打包。
3. **加载后重排(layout conversion)统一放 `process_weights_after_loading`**。checkpoint 里是一套布局(GPTQ/AWQ 各有各的 int32 打包顺序),kernel 要的是另一套(Marlin 的 g_idx 排序、FP8 的 (N,K) 转置),全部在权重加载完成后一次性完成,前向时零转换开销。

> **历史对照**:V0 时代量化方法是"一坨 if-else + 一个 marlin_linear 模块"。V1(本分支)把它收敛为统一的 `QuantizeMethodBase` 抽象,并且把**kernel 选择**从方法里抽出去,交给 `kernels/linear/` 的"或acles"(`choose_scaled_mm_linear_kernel` / `choose_mp_linear_kernel`),让同一个 FP8 权重在 Ada/Blackwell/ROCm 上自动落到不同的 kernel。

---

## 2. 量化方法总览:QuantizationMethods 与注册表

### 2.1 `QuantizationMethods` 枚举(`quantization/__init__.py:12-46`)

它是一个 `typing.Literal`,把 vLLM 已知的量化方法名列出来,`QUANTIZATION_METHODS`(`__init__.py:47`)用 `get_args` 展开成实际列表供校验用。注意里面有**在线量化简写**(`fp8_per_tensor`、`fp8_per_block`、`fp8_per_channel`、`int8_per_channel_weight_only`、`nvfp4_per_token`、`mxfp8`),它们不是独立实现,而是 `--quantization` 的糖衣(见 §3.3)。

方法名 → 配置类 的映射表 `method_to_config`(`__init__.py:139-168`),几个重要别名(`__init__.py:140-152`):

| CLI 名称 | 配置类 | 说明 |
|---|---|---|
| `fp8` | `Fp8Config` | FP8 权重 + 静态/动态激活量化 |
| `auto_gptq` / `gptq` / `gptq_marlin` | `AutoGPTQConfig` | GPTQ checkpoint 的 Marlin 实现 |
| `awq` / `awq_marlin` / `auto_awq` | `AutoAWQConfig` | AWQ checkpoint(Marlin/Triton/XPU) |
| `compressed-tensors` | `CompressedTensorsConfig` | 兼容 llm-compressor 产出的通用格式 |
| `online` 及 `fp8_per_tensor` 等 | `OnlineQuantizationConfig` | 加载时在线量化 FP16/BF16 权重 |
| `modelopt*` / `mxfp4` / `mxfp8` | `ModelOpt*Config` 等 | 英伟达 ModelOpt / 新兴格式 |

### 2.2 `get_quantization_config`(`__init__.py:108`)

按方法名返回配置**类**(不是实例)。要点:

- 非法名字直接 `ValueError`(:109-110)。
- **lazy import**(:112-137):`method_to_config` 里的所有配置类都在函数内部才导入,避免早期 `torch.compile` 被提前触发。
- 在线简写用 `setdefault` 注入(:174-175),所以 checkpoint 标记 `quant_method: "mxfp8"` 的模型走 `ModelOptMxFp8Config`(checkpoint 方法优先),而用户 `--quantization mxfp8` 的在线量化路径仍可用。
- 用户自定义方法通过 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` 合并(:178)。

### 2.3 注册自定义量化方法:`register_quantization_config`(`__init__.py:58`)

装饰器工厂:把自定义配置类登记进 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`(:102),并**自动追加**到 `QUANTIZATION_METHODS` 与 `current_platform.supported_quantization`(:93-96),让 vLLM 平台校验接受它。前提是类必须是 `QuantizationConfig` 子类(:98-101)。

---

## 3. 量化配置注入链路:`--quantization` 如何一步步变成 layer 上的方法

### 3.1 ModelConfig 侧:校验与 checkpoint 探测(`config/model.py`)

`ModelConfig.quantization`(`model.py:224`)与 `quantization_config`(:229)是用户入口。真正的魔法在 `_verify_quantization`(`model.py:1199`):

1. 读 checkpoint 的 `quantization_config["quant_method"]`(:1208)。
2. 构建 `overrides` 优先级列表(:1212-1232),`auto_gptq`/`auto_awq` 排前面——它们重写了 `override_quantization_method`,用于"识别 checkpoint 并自动接管"。
3. 按优先级逐个调用 `method.override_quantization_method(quant_cfg, user_quant, hf_config)`(:1245-1266),第一个返回非 None 的胜出,并把 `self.quantization` 改写为该方法。
4. 用户显式指定的方法与 checkpoint 探测结果不一致时报错(:1272-1278);方法不在 `QUANTIZATION_METHODS` 里或平台不支持也报错(:1280-1286);废弃方法(`fbgemm_fp8`/`fp_quant`)默认拒绝,除非 `--allow-deprecated-quantization`(:1288-1301)。

### 3.2 `get_quant_config`:从哪实例化配置类(`model_loader/weight_utils.py:240`)

`VllmConfig._get_quantization_config`(`config/vllm.py:743`)在拿到 `model_config.quantization` 后调用它,并按顺序尝试:

1. **HF checkpoint 内嵌配置**:`hf_config.quantization_config` → `quant_cls.from_config(hf_quant_config)`(`weight_utils.py:287`)。
2. **hf_overrides 指定文件/JSON**:`quantization_config_file` / `quantization_config_dict_json`(:297-316)。
3. **在线量化**:用户给了 `quantization_config` 且方法是 online 简写 → `OnlineQuantizationConfig(args=...)`(:320-327)。
4. **配置文件兜底**:按 `quant_cls.get_config_filenames()`(如 GPTQ/AWQ 的 `quantize_config.json`)在模型目录搜 `*.json` 并用 `from_config` 解析(:353-369)。

拿到实例后 `_get_quantization_config` 再做两重硬校验(`vllm.py:753-770`):GPU 算力必须 ≥ `get_min_capability()`,模型 dtype 必须在 `get_supported_act_dtypes()` 里;然后调 `maybe_update_config` 让配置类有机会按权重元数据补充信息(`vllm.py:771-775`)。

### 3.3 在线量化的 CLI 糖衣(`config/quantization.py`)

这是 v0.23 新增的**在线量化**参数体系,和 §2 的 checkpoint 量化是两条平行路径:

- `QuantSpec`(`quantization.py:65`):描述某一类算子(linear / moe)的 weight/activation 量化 key,`None` 表示沿用默认。
- `QuantizationConfigArgs`(:81):`linear`/`moe`/`ignore` 三字段。
- `_ONLINE_SHORTHANDS`(:116):把 `fp8_per_tensor`、`fp8_per_block`、`fp8_per_channel`、`mxfp8`、`mxfp4`、`int8_per_channel_weight_only`、`nvfp4_per_token` 分别解糖为对应的 `QuantSpec`。
- `resolve_quantization_config`(:158):合并 shorthand 与 `--quantization-config` 里用户显式覆盖的字段。
- `QUANT_KEY_NAMES`(:26):把用户可读名字(`"fp8_per_tensor_static"` 等)映射到 `quant_utils.py` 里的 `QuantKey` 常量(`kFp8StaticTensorSym` 等在 `quant_utils.py:194-222`)。

`QuantKey`(`quant_utils.py:161`)是整个 kernel 选择系统的最小契约:一个 `(dtype, ScaleDesc, symmetric)` 元组,描述"权重/激活是什么 dtype、scale 是 per-tensor/per-token/per-channel/per-block"。

---

## 4. 抽象基类速写

### 4.1 `QuantizationConfig`(`quantization/base_config.py:87`)

配置层的抽象基类,每个方法必须实现(全部 `@abstractmethod`):

- `get_name()`(:108):返回方法名,例如 `"fp8"`。
- `get_supported_act_dtypes()`(:113):支持的激活 dtype 列表。
- `get_min_capability()`(:119):最低 GPU 算力(如 75 = Ada)。
- `get_config_filenames()`(:130):模型目录里要找的配置文件名,GPTQ/AWQ 返回 `["quantize_config.json"]`。
- `from_config()`(:136):从 checkpoint 的 quant dict 构造实例。
- `get_quant_method(layer, prefix)`(:180):**分派核心**,按 layer 类型返回对应 `QuantizeMethodBase`,不支持则返回 `None`。

还有两个有默认实现的可选钩子:

- `override_quantization_method()`(:141):默认返回 `None`;GPTQ/AWQ 重写它来自动识别 checkpoint(§7/§8)。
- `get_cache_scale_mapper()`(:195):返回一个 `WeightsMapper`,让 `AutoWeightsLoader` 自动把 KV-cache 的 scale 权重名映射到 vLLM 内部名(如 `.k_proj.output_scale → .attn.k_scale`)。

### 4.2 `QuantizeMethodBase`(`base_config.py:20`)

方法层抽象,五个接口:

- `create_weights(layer, *weight_args, **extra_weight_attrs)`(:31):为 layer 建量化权重参数并 `register_parameter`。
- `apply(layer, *args, **kwargs) → Tensor`(:40):把权重应用到输入,前向入口。
- `embedding()`(:47)/`tie_weights()`(:54):词嵌入特殊处理(默认共享张量)。
- `process_weights_after_loading(layer)`(:67):权重加载后的布局重排/转置。
- `uses_meta_device`(:23):在线量化时是否先在 meta device 建权重、逐层量化以省显存。

`LinearMethodBase`(`layers/linear.py:125`)就是它的 Linear 特化子类(只重声明 `create_weights`/`apply` 两个抽象方法)。

---

## 5. Linear 层分派:`quant_config → quant_method → create_weights/apply`

### 5.1 `LinearBase.__init__` 的分派点(`linear.py:229`)

所有 Linear 算子的公共基类(`linear.py:215`)。构造时最重要的三行(`linear.py:257-263`):

```python
self.quant_method: QuantizeMethodBase
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

也就是说:**量化方法由 layer 自己问配置类要**,配置类根据 `layer` 的类型(`LinearBase`/`RoutedExperts`/`Attention`)决定给谁。`UnquantizedLinearMethod`(`linear.py:166`)是"无量化"的实现,`create_weights` 建普通 `ModelWeightParameter`(:184),`apply` 走 `dispatch_unquantized_gemm()`(:212)。

### 5.2 三种主力 Linear 算子

| 算子 | 类定义 | create_weights 调用 | 前向 |
|---|---|---|---|
| 列并行(切输出维) | `ColumnParallelLinear`(`linear.py:401`) | :486 | `forward` :569 → `quant_method.apply` :576 + all-gather |
| 行并行(切输入维) | `RowParallelLinear`(`linear.py:1504`) | :1576 | `quant_method.apply` + all-reduce |
| 复制权重 | `ReplicatedLinear`(`linear.py:296`) | :345 | :385 |

`create_weights` 的入参协议统一为 `(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, weight_loader=...)`,量化方法据此计算本 rank 上的权重形状、以及 scale 是否需要在 TP 下重复或切分。`weight_loader` 由 layer 传入(`linear.py:493-497` / `1583-1587`),决定 scale/g_idx 这些参数在加载时怎么按 `tp_rank`/`tp_size` shard。

### 5.3 生命周期全流程

```
模型构造:  layer = ColumnParallelLinear(..., quant_config=...)      linear.py:432
           → LinearBase.__init__ 分派 quant_method                  linear.py:260
           → self.quant_method.create_weights(layer, ...)           linear.py:486
权重加载:  load_weights → weight_loader 把磁盘张量拷入 qweight/...  
           → quant_method.process_weights_after_loading(layer)      fp8.py:370 等
           → update_param_tp_status 校正子参数 TP 状态               linear.py:278
前向:      forward → quant_method.apply(layer, x, bias)             linear.py:576
           → kernel.apply_weights(layer, x, bias)
```

### 5.4 量化参数的载体:自定义 `Parameter` 家族(`model_executor/parameter.py`)

量化权重不是普通 `nn.Parameter`,而是一族带切分/打包语义的 `BasevLLMParameter`(`parameter.py:32`)。每个子类声明 `input_dim`/`output_dim`(TP 切分沿哪一维)、`packed_dim`/`packed_factor`(int32 打包)、以及自己的 `weight_loader`(加载时按 `tp_rank`/`tp_size` 切分或广播):

| 参数类 | 定义 | 用途 | 典型使用方 |
|---|---|---|---|
| `PackedvLLMParameter` | `parameter.py:353` | 打包权重:`qweight`,可指定沿 K 或沿 N 打包 | GPTQ(:381,packed_dim=0)/ AWQ(:453,packed_dim=1) |
| `RowvLLMParameter` | `parameter.py:204` | 行向只读张量:`g_idx`(K 维,不切分) | GPTQ `g_idx`(:395) |
| `GroupQuantScaleParameter` | `parameter.py:242` | 分组 scale:shape `(num_groups, N)`,沿 input_dim=0 切 | GPTQ/AWQ `scales`(auto_gptq.py:431、auto_awq.py:481) |
| `ChannelQuantScaleParameter` | `parameter.py:251` | per-channel scale,只有 output_dim | GPTQ desc_act=False 时的 `scales`(:422) |
| `PackedColumnParameter` | `parameter.py:313` | 沿 N 打包的 zero-point:`qzeros` | GPTQ `qzeros`(:423) |
| `PerTensorScaleParameter` | `parameter.py:260` | 标量/每逻辑权重一个的 FP8 scale | FP8 `weight_scale`(fp8.py:332) |
| `BlockQuantScaleParameter` | `parameter.py:397` | block-wise scale:`(N//block_n, K//block_k)` | FP8 `weight_scale_inv`(fp8.py:342) |

**切分语义的关键点**:scale 在 TP 下不一定跟着权重切。GPTQ 的 `marlin_repeat_scales_on_all_ranks`(`auto_gptq.py:367-378`)判断:当 `desc_act=True` 且行并行时,每组只覆盖本 rank 的 K 分片,`scales_and_zp_input_dim = 0` 正常切(:377);否则让 scale 在每张卡上都重复(置 `None`,:372),避免 MoE 里 gate/up/down 投影 scale 错位。FP8 的 `PerTensorScaleParameter` 对 fused QKV 则存 N 个标量,由 `weight_loader` 配合 `adjust_scalar_to_fused_array`(`linear.py:98`)按 `shard_id`(q/k/v)切片。

---

## 6. FP8:`Fp8Config` 与 `Fp8LinearMethod`

### 6.1 `Fp8Config`(`quantization/fp8.py:92`)

构造参数(`fp8.py:95-133`):

- `is_checkpoint_fp8_serialized`(:105):checkpoint 里权重**本来就是 FP8**(AutoFP8/llm-compressor 产出),还是 FP16/BF16 需要加载时现量。为 False 时 `get_quant_method` 直接给 `Fp8PerTensorOnlineLinearMethod`(`fp8.py:186-193`)。
- `activation_scheme`(:109):`"static"` / `"dynamic"`(`ACTIVATION_SCHEMES`,`fp8.py:87`)。static 需要 checkpoint 提供 `input_scale`;dynamic 前向时现算。
- `weight_block_size`(:132):非 None 表示 **block-wise(per-block)量化**,典型 `[128, 128]`;只支持 fp8-serialized checkpoint + dynamic activation(:116-131),weight 的 scale 参数改名为 `weight_scale_inv`(`fp8.py:350-351`)。
- `store_dtype`(:114):`"mxfp4"` 时 MoE 走 `Mxfp4MoEMethod`(`fp8.py:206-211`)。
- `use_deep_gemm`(:133):是否允许 DeepGEMM 后端。

`from_config`(:156)从 checkpoint 的 `quant_method` 判断是否 serialized(`"fp8" in quant_method`,:158),`ignored_layers` 兼容 `modules_to_not_convert`(:163-166)。

**FP8 格式要点**:vLLM 的 w8a8 FP8 只支持 `float8_e4m3fn`(4 位指数 3 位尾数,精度优先),因为 `torch._scaled_mm` 的限制(`fp8.py:244-246`);`e5m2`(范围优先)主要用于权重仅量化场景。scale 本身用 `float8_e8m0fnu`(纯指数,`fp8.py:348`,`is_scale_e8m0`)。

### 6.2 `get_quant_method` 分派(`fp8.py:175-222`)

按 layer 类型三路分派,`prefix` 参与 `ignored_layers` 匹配(`is_layer_skipped`,:179-185,被忽略的层返回 `UnquantizedLinearMethod`):

- `LinearBase` → `Fp8LinearMethod`(offline)/ `Fp8PerTensorOnlineLinearMethod`(online)
- `RoutedExperts` → `Fp8MoEMethod`(`fp8.py:464`)
- `Attention` → `Fp8KVCacheMethod`(FP8 KV cache 的 scale 参数管理)

### 6.3 `Fp8LinearMethod`(`fp8.py:239`)

**量化 key 决策**(`__init__`,`fp8.py:252-292`):这是整个 kernel 选择的输入。非 block 时,weight 固定 `kFp8StaticTensorSym`(:285);activation 在 static 时为 `kFp8StaticTensorSym`(:288),dynamic 时**能用 cutlass 就 per-token**(`kFp8DynamicTokenSym`,:289-290),否则退 per-tensor(:292)。block 时 activation 按 `weight_block_size[0]` 组对齐(`create_fp8_quant_key`,`fp8.py:277-283`)。

**create_weights**(`fp8.py:294-368`):

- 注册 `weight`(fp8 参数,:324-327)、`weight_scale`(per-tensor,:331-338)或 `weight_scale_inv`(block,:342-351)、static 时的 `input_scale`(:354-357)。
- 调用 `init_fp8_linear_kernel(...)`(:359-366)完成 kernel 选型并**记住 `use_marlin`**(:368)。

**process_weights_after_loading**(`fp8.py:370-416`):

- **Marlin 路径**(:371-380):权重转置为 (K,N),并把 `marlin_input_dtype` 传给 kernel(`fp8.py:377-378`)。
- **普通 w8a8 路径**(:382-416):若 checkpoint 非 per-tensor(如 fused QKV 有 N 个 scale),先按逻辑宽度合并再 per-tensor 重整(`process_fp8_weight_tensor_strategy`,:396-401);static 时取 `input_scale.max()`(:402-404);最后 `weight.t()`(:405)。

**apply**(`fp8.py:418-461`):`VLLM_BATCH_INVARIANT` 时走"BF16 反量化 + torch GEMM"的 batch-invariant 路径(:426-459);否则 `fp8_linear.apply_weights(layer, x, bias)`(:461)。

### 6.4 FP8 kernel 选择链(`kernels/linear/__init__.py`)

`init_fp8_linear_kernel`(`__init__.py:665`)用 activation key 是否 per-group 决定查哪张候选表:

- **per-block**(block quant):`_POSSIBLE_FP8_BLOCK_KERNELS`(`:430-455`,CUDA 顺序)FlashInferDeepGEMM → DeepGemm → Cutlass → B12x → Marlin → Triton → Humming → BlockWiseTorch。
- **per-tensor/per-token**:`_POSSIBLE_FP8_KERNELS`(`:397-426`,CUDA 顺序)**Marlin → FlashInfer → Cutlass → B12xTensor → PerTensorTorch → ChannelWiseTorch → Humming**。

选择过程在 `choose_scaled_mm_linear_kernel`(`__init__.py:601`):

1. 若指定 `force_kernel` 且它能实现配置,直接返回(:633-644)。
2. 取当前平台候选表(:646),应用 `--linear-backend` 过滤(`_resolve_backend_kernels`,:649)。
3. 按顺序找第一个同时通过 `is_supported(compute_capability)` 和 `can_implement(config)` 的 kernel(`is_supported_and_can_implement_kernel`,`__init__.py:576`;循环 :651-657)。
4. 全失败则聚合所有失败原因抛 `ValueError`(:659-662)。

示例:`MarlinFP8ScaledMMLinearKernel`(`kernels/linear/scaled_mm/marlin.py:29`)的 `is_supported`(:36)检查算力与 FP8 支持,`can_implement`(:59)检查 per-tensor/per-channel key,`process_weights_after_loading`(:70)做 Marlin 的权重排布,`apply_weights`(:86)真正调用 `apply_fp8_marlin_linear`。

> **环境变量钩子**:`VLLM_DISABLED_KERNELS`(`__init__.py:579`)可禁用特定 kernel;`--linear-backend` 可强制某一后端。这两个是所有 kernel 选择链通用的。

### 6.5 scaled_mm kernel 家族(`kernels/linear/scaled_mm/`)

`init_fp8_linear_kernel` 产出的对象都继承自 `FP8ScaledMMLinearKernel`(`ScaledMMLinearKernel.py:108`,基类 `ScaledMMLinearKernel`:59),其配置载体是 `FP8ScaledMMLinearLayerConfig`(`:33`),由 activation/weight 两个 `QuantKey` + 输入输出 dtype + 权重形状组成(`init_fp8_linear_kernel` 在 `__init__.py:674-679` 构造)。kernel 的统一接口只有两个:`process_weights_after_loading(layer)` 和 `apply_weights(layer, x, bias)`(`ScaledMMLinearKernel.py:90/94`)。

各平台候选(kernel 类 → 实现文件):

| kernel | 定义位置 | 适用 |
|---|---|---|
| `MarlinFP8ScaledMMLinearKernel` | `scaled_mm/marlin.py:29` | FP8 weight-only(无 FP8 算力的 GPU 也走它,`fp8.py:259-261`) |
| `FlashInferFP8ScaledMMLinearKernel` / `FlashInferFp8BlockScaledMMKernel` / `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | `scaled_mm/flashinfer.py:39/88/149` | Blackwell 上 FlashInfer 系 |
| `CutlassFP8ScaledMMLinearKernel` / `CutlassFp8BlockScaledMMKernel` | `scaled_mm/cutlass.py:156/275` | 通用 w8a8 |
| `B12xTensorFP8ScaledMMLinearKernel` / `B12xFp8BlockScaledMMKernel` | `scaled_mm/b12x_tensor.py:121` / `b12x_block.py:122` | ByteDance B12x |
| `PerTensorTorchFP8ScaledMMLinearKernel` / `RowWiseTorch...` / `ChannelWiseTorch...` / `BlockWiseTorch...` | `scaled_mm/pytorch.py:65/106/182/252` | torch 原生 `_scaled_mm` 兜底 |
| `DeepGemmFp8BlockScaledMMKernel` | `scaled_mm/deep_gemm.py:32` | block 量化 + DeepGEMM 可用时(`fp8.py:264-267`) |
| `ROCmFP8ScaledMMLinearKernel`、Aiter 系 | `scaled_mm/rocm.py:74`、`aiter.py` | ROCm |
| `XPUW8A8FP8LinearKernel` 等 | `scaled_mm/xpu.py:23/122` | XPU |

---

## 7. GPTQ:`AutoGPTQConfig` 与 Marlin 打包

### 7.1 `AutoGPTQConfig`(`quantization/auto_gptq.py:97`)

GPTQ(论文 arxiv 2210.17323)把权重逐组量化,`__init__`(`auto_gptq.py:106`)持有核心参数:

- `weight_bits` / `is_sym`(:148-149):4/8 bit,对称/非对称。`TYPE_MAP`(:101-104)决定 `quant_type`(如 `uint4b8`),非法组合直接抛错(:157-160)。
- `group_size`(:152):每组多少个 K 维度元素共享一个 scale,`-1` 表示整列一组(per-channel)。
- **`desc_act`(:153)**:"descending activation order",GPTQ 按激活值从大到小分组量化导致的行重排。为 True 时需要 `g_idx` 记录每行属于哪个组,并让 Marlin kernel 做 sort/permutation。`desc_act and group_size == -1` 时自动降为 False(:118-121)。
- `pack_factor = 32 // weight_bits`(:151):每个 int32 打包 8 个 4-bit 值。
- `dynamic`(:146):GPTQModel 的 per-module 覆盖规则(regex `+:...`/`-:...`)。
- `lm_head_quantized`(:154)。

`from_config`(:195)从 `quantize_config.json`(`get_config_filenames` :192)读取;`override_quantization_method`(:219)在 checkpoint 的 `quant_method == "gptq"` 且用户没指定冲突方法时自动接管。`maybe_update_config`(:278)用 safetensors 元数据自动推断哪些模块被量化(`modules_in_block_to_quantize`)。

### 7.2 `AutoGPTQLinearMethod`(`auto_gptq.py:306`)

`create_weights`(:326)完全由 kernel 选择驱动:

1. 构造 `MPLinearLayerConfig`(:341-352),关键字段:`weight_type=quant_type`、`group_size`、`zero_points=False`、**`has_g_idx=desc_act`**(:351)。
2. `choose_mp_linear_kernel(config)`(:354)按 `_POSSIBLE_KERNELS` 选 kernel(CUDA: CutlassW4A8 → Machete → AllSpark → Marlin → Conch → Exllama → TritonW4A16 → Humming,`__init__.py:474-501`)。
3. 注册四个参数(`auto_gptq.py:442-445`):
   - `qweight`:`PackedvLLMParameter`,`packed_dim=0`(沿 K 维打包),shape `(K//pack_factor, N)`(:381-392)。
   - `g_idx`:`RowvLLMParameter`,shape `(K,)`,`desc_act` 时才有意义(:395-402)。
   - `scales` / `qzeros`:按 TP 下 scale 是否重复决定用 `ChannelQuantScaleParameter` 还是 `GroupQuantScaleParameter`(`marlin_repeat_scales_on_all_ranks`,:367-440)。

`apply`(:458)直接 `self.kernel.apply_weights(layer, x, bias)`(:464),零 Python 层开销。MoE 走 `AutoGPTQMoEMethod`(:467),用 `select_wna16_moe_backend`(:489)选 WNA16 MoE 后端,`get_quant_method`(:240)里对不支持 Marlin 的层回退 `MoeWNA16Config`(:246-255)。

---

## 8. AWQ:`AutoAWQConfig` 与 Marlin

### 8.1 `AutoAWQConfig`(`quantization/auto_awq.py:171`)

AWQ(论文 arxiv 2306.00978)的关键是**按激活分布挑选 1% 的显著通道放大**,保护它们不被量化伤害,所以权重自带**per-channel/per-group 的 scale**(而不是 GPTQ 那种纯误差最小化)。`__init__`(:183):`weight_bits`(只支持 4,`TYPE_MAP` :179-181)、`group_size`、`zero_point`(AWQ 非对称量化带 zero-point)、`pack_factor = 32 // bits`(:193)。

`from_config`(:238)兼容两种字段名:`w_bit`/`bits`、`q_group_size`/`group_size`(:239-240),并把 `full_config["quant_method"]` 规范成 `"awq"` 供 MoE 回退用(:248-249)。`override_quantization_method`(:260)在 `quant_method == "awq"` 时接管(CPU 除外,:265-266,让位给 `cpu_awq`)。

### 8.2 `get_quant_method` 分派(`auto_awq.py:285-359`)

- `LinearBase`(:288):跳过层 → `UnquantizedLinearMethod`(:297);CPU/XPU → `AutoAWQMarlinLinearMethod`(:304-305);CUDA 且 Marlin 可用(`check_marlin_supported`,:309-315)→ `AutoAWQMarlinLinearMethod`(:328);层形状不兼容或 batch-invariant 模式 → Triton 版 `AutoAWQLinearMethod`(:319-332)。
- `RoutedExperts`(:334):不支持 Marlin 则回退 `MoeWNA16Config`,否则 `AutoAWQMoEMethod`(:357)。

### 8.3 AWQ 的非标准打包与转换(`auto_awq.py:73-168`)

AWQ checkpoint 的坑:它在 int32 里按**非标准位序**打包 4-bit 值——8 个值依次落在 bit 位 `[0,4,1,5,2,6,3,7]`,且 `qweight` 沿 **output 维**打包(`packed_dim=1`,:453-464)。Marlin/Exllama 等 kernel 要的是标准位序、沿 input 维打包。`_convert_awq_to_standard_format`(`auto_awq.py:93`)在 `process_weights_after_loading` 里做一次性转换:

1. 用 `_REVERSE_AWQ_PACK_ORDER`(:77)把位序反置换回标准序(:119-121)。
2. 沿 input 维重新打包成 `PackedvLLMParameter(packed_dim=0)`(:124-140)。
3. `qzeros` 同理,并转置成 kernel 期望的布局(:146-168)。

### 8.4 `AutoAWQMarlinLinearMethod`(`auto_awq.py:388`)

`create_weights`(:414)与 GPTQ 版同构:构造 `MPLinearLayerConfig`(:432-443,注意 `has_g_idx=False`,AWQ 无 act order)→ `choose_mp_linear_kernel`(:445)→ 注册 `qweight`(AWQ 布局)、`qzeros`、`scales`(:492-494)。`process_weights_after_loading`(:503)先做 §8.3 的格式转换再交给 kernel。`apply`(:513)委托 `kernel.apply_weights`。

Triton 回退版 `AutoAWQLinearMethod`(`auto_awq.py:901`)的 `apply`(:913)用启发式:token 数 ≥ 256 或 batch-invariant 时先 `ops.awq_dequantize` 反量化再 `torch.matmul`(:927-932),否则直接 `ops.awq_gemm`(:934)——这是 kernel 层 API(`_custom_ops`)直接进 `apply` 的唯一残留路径。

---

## 9. MoE 量化:同一抽象的第二个落点(`FusedMoEMethodBase`)

前面说的 `LinearMethodBase` 只管普通矩阵乘;MoE 专家层走的是另一棵子树 `FusedMoEMethodBase`(`layers/fused_moe/fused_moe_method_base.py:27`),它同样继承 `QuantizeMethodBase`,但接口不同——`create_weights(layer, num_experts, hidden_size, intermediate_size_per_partition, ...)`,`apply(layer, x, topk_weights, topk_ids, shared_experts, ...)`。

分派仍在配置类的 `get_quant_method` 里完成,`RoutedExperts` 分支返回对应 MoE 方法:

| 配置类 | MoE 方法 | 定义位置 | 后端选择 |
|---|---|---|---|
| `Fp8Config` | `Fp8MoEMethod` | `fp8.py:464` | `select_fp8_moe_backend`(`fp8.py:499`) |
| `AutoGPTQConfig` | `AutoGPTQMoEMethod` | `auto_gptq.py:467` | `select_wna16_moe_backend`(`auto_gptq.py:489`) |
| `AutoAWQConfig` | `AutoAWQMoEMethod` | `auto_awq.py:522` | `select_wna16_moe_backend`(`auto_awq.py:535`) |

MoE 量化与 Linear 量化几个本质差异:

1. **三组权重而非一组**:gate_up_proj(w13)与 down_proj(w2),且 gate 和 up 是 fused 的(`moe.w13_num_shards`,如 `auto_gptq.py:540-544`),scale/zero-point 也要按 w13/w2 分开建。
2. **TP 切分规则不同**:MoE 默认 EP(专家并行),intermediate 维由 TP 切,所以出现 `load_full_w2`(desc_act 时 w2 的 scale 不切,`auto_gptq.py:583`)和 `is_k_full`(`auto_gptq.py:517-519`)这类标记。
3. **权重布局转换更重**:`process_weights_after_loading` 通过 `convert_to_wna16_moe_kernel_format`(`auto_gptq.py:706`、`auto_awq.py:650`)把 checkpoint 布局统一转成 WNA16 后端格式,还要算 `g_idx` 的 sort 索引(`w13_g_idx_sort_indices`,`auto_gptq.py:632-651`)、Marlin 的 workspace(`marlin_make_workspace_new`,`auto_gptq.py:677`)。
4. **落地 kernel 不同**:MoE 的最终执行对象是 `FusedMoEKernel`(由 `make_wna16_moe_kernel` 构造,`auto_gptq.py:770`),由 `WNA16MoEBackend` 枚举(`WNA16MoEBackend.HUMMING` 等,`auto_gptq.py:784`)区分 CUDA/CPU/Humming 实现;FP8 侧则对应 `select_fp8_moe_backend` 选出的 `FusedMoEKernel`(`fp8.py:499-504`)。

FP8 MoE 的量化 key 决策在 `Fp8MoEMethod.__init__`(`fp8.py:487-496`):block 量化固定 `(weight=kFp8Static128BlockSym, activation=kFp8Dynamic128Sym)`,非 block 用 `kFp8StaticTensorSym` + static/dynamic 的 activation key,与 §6.3 Linear 侧的规则保持一致。

---

## 10. 关键类 / 函数速查(文件:行号)

### 配置入口
- `config/model.py:224 ModelConfig.quantization`;`:229 quantization_config`;`_verify_quantization`:1199;overrides 探测循环:1245;废弃方法检查:1288
- `config/vllm.py:743 VllmConfig._get_quantization_config`;`get_quantization_config`:780;`self.quant_config` 赋值:1102
- `model_executor/model_loader/weight_utils.py:240 get_quant_config`(from_config :287 / online :320 / 配置文件 :353)

### 方法注册与枚举
- `layers/quantization/__init__.py`:`QuantizationMethods`:12;`QUANTIZATION_METHODS`:47;`DEPRECATED_QUANTIZATION_METHODS`:49;`register_quantization_config`:58;`get_quantization_config`:108;`method_to_config`:139

### 抽象基类
- `layers/quantization/base_config.py`:`QuantizeMethodBase`:20(`create_weights`:31/`apply`:40/`embedding`:47/`tie_weights`:54/`process_weights_after_loading`:67);`QuantizationConfig`:87(`get_name`:108/`get_supported_act_dtypes`:113/`get_min_capability`:119/`get_config_filenames`:130/`from_config`:136/`override_quantization_method`:141/`get_quant_method`:180/`get_cache_scale_mapper`:195)
- `layers/linear.py`:`LinearMethodBase`:125;`UnquantizedLinearMethod`:166;`LinearBase`:215(分派 :257-263);`ReplicatedLinear`:296;`ColumnParallelLinear`:401(create_weights :486 / forward :569);`RowParallelLinear`:1504(create_weights :1576)

### FP8
- `layers/quantization/fp8.py`:`ACTIVATION_SCHEMES`:87;`Fp8Config`:92(`from_config`:156/`get_quant_method`:175);`Fp8LinearMethod`:239(`__init__`:252/`create_weights`:294/`process_weights_after_loading`:370/`apply`:418);`Fp8MoEMethod`:464
- `kernels/linear/__init__.py`:`_POSSIBLE_FP8_KERNELS`:397;`_POSSIBLE_FP8_BLOCK_KERNELS`:430;`choose_scaled_mm_linear_kernel`:601;`init_fp8_linear_kernel`:665;`is_supported_and_can_implement_kernel`:576
- `kernels/linear/scaled_mm/marlin.py`:`MarlinFP8ScaledMMLinearKernel`:29(`is_supported`:36/`can_implement`:59/`process_weights_after_loading`:70/`apply_weights`:86)

### GPTQ
- `layers/quantization/auto_gptq.py`:`AutoGPTQConfig`:97(`__init__`:106/`TYPE_MAP`:101/`from_config`:195/`override_quantization_method`:219/`get_quant_method`:240);`AutoGPTQLinearMethod`:306(`create_weights`:326/`choose_mp_linear_kernel`:354/`apply`:458);`AutoGPTQMoEMethod`:467
- `kernels/linear/__init__.py`:`_POSSIBLE_KERNELS`:474;`choose_mp_linear_kernel`:774

### AWQ
- `layers/quantization/auto_awq.py`:`_REVERSE_AWQ_PACK_ORDER`:77;`_convert_awq_to_standard_format`:93;`AutoAWQConfig`:171(`from_config`:238/`override_quantization_method`:260/`get_quant_method`:285);`AutoAWQMarlinLinearMethod`:388(`create_weights`:414/`process_weights_after_loading`:503/`apply`:513);`AutoAWQMoEMethod`:522;`BaseAWQLinearMethod`:819;`AutoAWQLinearMethod`:901(`apply`:913)

### 在线量化参数
- `config/quantization.py`:`QUANT_KEY_NAMES`:26;`QuantSpec`:65;`QuantizationConfigArgs`:81;`_ONLINE_SHORTHANDS`:116;`ONLINE_QUANT_SHORTHAND_NAMES`:152;`resolve_quantization_config`:158
- `layers/quantization/utils/quant_utils.py`:`GroupShape`:104;`QuantKey`:161;`kFp8StaticTensorSym`:194;`kFp8DynamicTokenSym`:206;kInt4/kInt8 常量:266-268

### 常用环境变量
- `VLLM_BATCH_INVARIANT`:启用 batch-invariant 路径,FP8 退到 BF16 反量化+matmul(`fp8.py:426`),AWQ 走 fp16 matmul(`auto_awq.py:930`)。
- `VLLM_DISABLED_KERNELS`(`kernels/linear/__init__.py:579`):禁用指定 kernel。
- `--linear-backend`:强制某一 GEMM 后端(`kernels/linear/__init__.py:649/805`)。

### MoE 与 scaled_mm kernel 补充
- `layers/fused_moe/fused_moe_method_base.py:27 FusedMoEMethodBase`;`RoutedExperts` 分派入口:`routed_experts.py:44 RoutedExperts`、`_get_quant_method`:189(调 `quant_config.get_quant_method`,:201);工厂 `layer.py:99 FusedMoEFactory`
- `kernels/linear/scaled_mm/ScaledMMLinearKernel.py`:`ScaledMMLinearKernel`:59;`FP8ScaledMMLinearKernel`:108;`FP8ScaledMMLinearLayerConfig`:33
- `kernels/linear/scaled_mm/marlin.py:29 MarlinFP8ScaledMMLinearKernel`;`cutlass.py:156 CutlassFP8ScaledMMLinearKernel`;`flashinfer.py:39/88/149`;`pytorch.py:65/252`;`deep_gemm.py:32`
- `model_executor/parameter.py`:`BasevLLMParameter`:32;`RowvLLMParameter`:204;`GroupQuantScaleParameter`:242;`ChannelQuantScaleParameter`:251;`PerTensorScaleParameter`:260;`PackedColumnParameter`:313;`PackedvLLMParameter`:353;`BlockQuantScaleParameter`:397

---

## 延伸阅读

- vLLM 官方文档:量化特性总览 <https://docs.vllm.ai/en/latest/features/quantization/index.html>
- 论文:GPTQ —— Accurate Post-Training Quantization for Generative Pre-trained Transformers <https://arxiv.org/abs/2210.17323>
- 论文:AWQ —— Activation-aware Weight Quantization for LLM Compression and Acceleration <https://arxiv.org/abs/2306.00978>
- vLLM Blog/文档:FP8(Meta FP8 与 per-block 扩展) <https://docs.vllm.ai/en/latest/features/quantization/fp8.html>
- Marlin:向量并行 GEMM 的 4-bit 量化 kernel(在 vLLM 中用于 GPTQ/AWQ/FP8) <https://github.com/IST-DASLab/marlin>
- ModelCloud/GPTQModel(dynamic 配置与 `quantize_config.json` 格式来源) <https://github.com/ModelCloud/GPTQModel>
- 相关文档:本仓库 docs `01_system_architecture.md`(量化方法在 Worker/GPUModelRunner 的模型构造阶段被实例化)、`03_Scheduler.md`(调度与执行,量化只影响 `execute_model` 里的 GEMM)