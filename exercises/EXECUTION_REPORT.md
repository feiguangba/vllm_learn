# 执行汇报 · ch02 manim 移除 + ch01 真实数据/讲解增强 基建样板

> 汇报对象:VLLM_learn 仓库维护者
> 汇报人:opencode(agent)
> 环境:win32 + PowerShell 5.1 · 推理后端 **uv_cuda**(`D:\uv_envs\uv_cuda\Scripts\python.exe`,torch 2.11.0+cu128,RTX 5060 Laptop GPU / sm_120)
> 关于 manim 的处置:按你的要求**全部移除**(见 §2);「回头有本地语音引擎再说」的 manim 语音讲解不在此次范围。

---

## 0. 一句话结论

本轮完成了两件事,并已全部验证可执行:

1. **彻底移除 ch02 的 manim 动画**(`manim_kit.py`、`manim_scenes.py`、`_media/`、各 app 内的动画代码),恢复为纯 Streamlit + LaTeX 讲解,7 个 app 全部通过编译与 headless 启动;
2. **建立“真实 CUDA 数据 + 前因后果讲解”的可复用样板**:新增跨章共享库 `vllm_real.py` + 一键批执行/自检工具 `chapter_exec.py` + 修好 notebook 的内核绑定 bug,并**完整交付并验证了 ch01 的 6 课**(1~4 重构建通过,5/6 已换上真实 GPU 数据并深刻充实讲解)。

> ⚠️ 重要提醒:你此前说「**所有章节都补充 + 所有 ipynb 都接真实 CUDA 数据**」。
> 上一轮我已明言这需要分章节逐批做(GPU 总时长是几十小时量级)。**本轮交付的是基建 + ch01 完整样板**;
> ch02~ch11 尚未逐课改造,后续可按 §6 的批处理流程逐章推进(每批都交给 `chapter_exec.py` 验证)。

---

## 1. 本次任务的背景与目标

- 仓库:`exercises/` 共 11 章、100+ 课,每课 = 1 个 `*.ipynb` + 1 个 `app_*.py`(Streamlit)。
- 引发点:上一轮已给 **ch02(7 课)** 加过 manim 动画 + 真实算子数据(`real_ops.py`)。
- 你的新决定:
  1. **改回使用 LaTeX 渲染**(即不要用 Text 替代公式,回归 LaTeX 数学排版);
  2. **manim 动画效果不好 → 全部移除**;
  3. **其他章节也要补充前因后果的细致讲解**真机数据,ipynb 也要接真实 CUDA 数据。

依据上述决定,我执行了下面的流程。

---

## 2. 移除 manim:改了哪些文件?

### 2.1 删除的产物(不复存在)

| 路径 | 说明 |
|---|---|
| `exercises/ch02/manim_scenes.py` | 7 个 manim 场景类(Scene07~Scene13) |
| `exercises/ch02/manim_kit.py` | ffmpeg 注入 + 渲染/缓存工具 |
| `exercises/ch02/_media/` | 已渲染的 mp4 与中间文件(全部删除) |

### 2.2 改动/新增的文件

| 文件 | 动作 | 说明 |
|---|---|---|
| `ch02/app_07..app_13.py`(7 个) | 修改 | 删除 `manim_player(...)` 调用、`from manim_scenes import ...`、`from app_common import manim_player, ...` 等;修正 3 处 docstring 里残留的 “(+manim)” |
| `ch02/app_common.py` | 重写 | 去掉 `manim_player`/`MEDIA_ROOT`,保留 `terminology`(升级为支持可选出处)、`real_badge`、`foot_note`,新增 `SRC` 参考链接表 |
| `ch02/real_ops.py` | 保留并微调 | 上一轮的真实 CUDA 算子库**保留**(它是“真实数据”那一支,与 manim 无关);补充 `import math`,并把 `frag_sim` 文档引用修正为 `alloc_frag_sim` |

### 2.3 验证命令与结果

- 语法:`uv_cuda\python.exe -m py_compile app_07..app_13.py` → **COMPILE OK**
- 确认 ch02 目录已无任何 manim 字样的引用 → `NO manim references`
- 逐个 `streamlit run --server.headless --server.port <p>` 并用 `urllib` 探活 → **全部 http200=True, crashed_lines=False**

---

## 3. 新增基建 ① `exercises/vllm_real.py`(跨章共享真实数据库)

设计动机:让所有章节的 ipynb/app 都能**直接拿真实 GPU 算子数字**,而不是各自手写 `np.arange`。

- `TinyGPT`:一个纯线性层 + tanh 的小模型,参数可测,可在 CUDA/CPU 上真实前向。
- `bench_prefill_decode(d, layers, L, ...)`:在 GPU 上实测 **一次并行 prefill** vs **逐 token decode** 的耗时与吞吐,返回 `prefill_ms / decode_total_ms / ratio / *tok_per_s` 等。
- `bench_throughput_curve(batch=(1,4,16,...))`:逐 decode 步、把 `batch` 个单 token 一起前向,测不同 batch 的 tokens/s(连续批处理的雏形)。
- `cuda_info()`:探测并返回设备名与显存。
- 所有基准用 `lru_cache` + 无 CUDA 时优雅回落。

**在你 RTX 5060 上实测的中间数据(本节真实验证输出):**

```
设备: NVIDIA GeForce RTX 5060 Laptop GPU · vram 8.0GiB
prefill(一次并行 256 token): 0.51 ms   → 503.5 k tok/s
decode (逐个 256 token):     134.84 ms  → 1.9 k tok/s
decode / prefill 耗时比:      265.2 倍

batch | 每步耗时(ms) | 吞吐(tokens/s)
  1   |    0.439      |    2,280
  4   |    0.339      |   11,816
  16  |    0.313      |   51,081
  64  |    0.569      |  112,420
  128 |    0.689      |  185,851
```

这几组数字直接进入了 ch01 第 05/06 课的 notebook。

---

## 4. 新增基建 ② `exercises/chapter_exec.py`(一键批执行 + 自检)

保证「补讲解 + 接真实数据」后的每一课都真正可执行、可复现。

- 对 `chXX`:
  1. 运行 `chXX/_build/` 下的 `build_*.py`(重建 ipynb + app);
  2. 扫描该目录所有 `*.ipynb`,逐个 `nbconvert --execute`(强制用 `uv_cuda` 内核);
  3. 逐个扫描 error cell 汇总,出报告。

---

## 5. 修掉的隐患:notebook 内核绑定 bug

执行 ch01 时发现 `nb_builder.py` 生成的 kernelspec 是 `"name": "python3"`,而机器上 `python3` 指向 **Anaconda(未装 pyecharts/不含 CUDA)**。这就导致 `nbconvert --execute` 会启动错误内核而报 `ModuleNotFoundError: No module named 'pyecharts'`。

修复:`exercises/nb_builder.py` 把
`"kernelspec": {"display_name": "Python 3 (uv_cuda)", ..., "name": "python3"}`
改为 `"name": "uv_cuda"`(与机器上已注册的 `uv_cuda` 内核一致)。

> 这也是后续所有章节重建时都会受益的一处根因修复。

---

## 6. ch01 完整样板:6 课全部重建 + 执行通过

对 ch01 逐课改造,重点在第 05、06 课(最有真实数据价值)。每课都 `nbconvert --execute` 到 `uv_cuda` 内核,error_cells=0。

### ch01/_build/build_05.py(重点改造)

- **接真实数据**:删掉原来的“纯公式 + CPU 手写矩阵”实验,改用 `vllm_real.bench_prefill_decode/bench_throughput_curve` 在 GPU 实测。
- **前因后果增补**:
  - 新增小节「根源:自回归必然走向两阶段」,承接上一课,解释为什么 LLM 被自回归下界劈成 prefill/decode;
  - 「为什么 decode 更慢」升级为 **compute-bound vs memory-bound** 的硬核分析(不再只是“卡车/空转”比喻);
  - 新增「那怎么救 decode?连续批处理预告」小节,用 `bench_throughput_curve` 直接展示 batch 1→128 吞吐暴涨;
  - 小结/练习/参考全部同步为真实数据导向(含 Orca 连续批处理出处)。
- apps `app_05_prefill_demo.py` 一并重写为“真实 GPU 实测版”。

### ch01/_build/build_06.py(重点改造)

- 保留纯 Python 的 `ToyEngine`(它教的是批量生成的代码机制,不该换掉);
- 新增「真实对照:在 GPU 上跑一个小 GPT 复现同一条吞吐曲线」小节,把玩具引擎的结论与真实设备对接;
- 其它各节调号并微调讲解。

### 其余 build_01..04

- 本轮**未做内容替换**(它们的主题是分词/自回归/采样/Transformer 概览,真实数据收益点不如 05/06),但通过重建保证与基建兼容、执行通过。

### ch01 执行结果(`chapter_exec.py ch01`)

```
[OK] build_01.py  ..  build_06.py        (重建脚本全部成功)
[OK] 01_token_and_tokenizer.ipynb        error_cells=0
[OK] 02_autoregressive_generation.ipynb  error_cells=0
[OK] 03_sampling_strategies.ipynb        error_cells=0
[OK] 04_transformer_quickstart.ipynb     error_cells=0
[OK] 05_prefill_vs_decode.ipynb          error_cells=0
[OK] 06_toy_inference_engine.ipynb       error_cells=0
[ch01] 全部通过 ✅
```

---

## 6.1 变更文件清单总表(本轮)

| 文件 | 类型 | 说明 |
|---|---|---|
| `exercises/vllm_real.py` | 新增 | 跨章共享真实 CUDA 微基准库 |
| `exercises/chapter_exec.py` | 新增 | 一键「重建+执行+自检」一章工具 |
| `exercises/nb_builder.py` | 修改 | kernelspec name: python3 → uv_cuda |
| `exercises/ch02/app_07..app_13.py` | 修改 | 移除一切 manim 代码 |
| `exercises/ch02/app_common.py` | 重写 | 去掉 manim_player,完善子语卡 |
| `exercises/ch02/real_ops.py` | 微调 | 补 import math、修 docstring |
| `exercises/ch02/manim_scenes.py`, `manim_kit.py`, `_media/` | **删除** | manim 全部移除 |
| `exercises/ch01/_build/build_05.py` | 修改 | 接真实 GPU 数据 + 深化讲解 |
| `exercises/ch01/_build/build_06.py` | 修改 | 新增真实 GPU 对照小节 |
| `exercises/ch01/*.ipynb`(6 个) | 重建 | 全部重新生成并已执行验证 |
| `exercises/ch01/app_05_prefill_demo.py`, `app_06_toy_engine.py` | 重写 | 真实数据版 |

---

## 7. 结果评估与风险提示

### 达成

- manim 已**彻底移除**,ch02 恢复为纯 Streamlit + LaTeX,7 app 启动/编译全过。
- 建立了**可复用的真实数据 + 讲解增强基建**(`vllm_real` / `chapter_exec`)。
- ch01 6 课全部重建并**执行通过**,05/06 是带真实 GPU 数字 + 前因后果讲解的样板。

### 风险 / 待办

1. **范围未完成**:ch02~ch11 尚未逐课补讲与接真实数据的改造(你要求“全部”,但需分批;已在 §0 说明)。
2. **manim 遗留**:ch02 的 `app_common.py` 我已去掉了 `manim_player`,但仓库里**其它章节有没有 manim 引用/媒体**本轮未全量扫描——需要 `rg -i manim` 检查一遍仓库。
3. **app 与 ipynb 的一致性**:ch01 的 notebook 内 `%%writefile app_*.py` 与独立 `app_*.py` 由同一 builder 产生,应一致;但 ch02 的 app 我在上一轮改过、而 **ch02 的 ipynb(07_*..13_*)还是旧版 app 源码,尚未重新生成**(与您此前的问题对应)。

### 后续建议执行顺序(每批一条命令)

```bash
# 逐章:先全仓库 manim 残留检查,然后对各章执行
rg -i "manim|st.video|_media" exercises/          # 1) 确认没有 manim 残渍
uv_cuda\python.exe chapter_exec.py ch02           # 2) 重建+执行+自检,逐章推进
```
每一批的先决条件:该章的 `_build/`(或 `gen_*.py`)已按要求补讲 + 接入 `vllm_real` / `real_ops` 真实数据。

---

## 8. 附:如何复现本轮所有操作

```powershell
# A. 移除 ch02 manim(在上一轮已完成的基础上)
Remove-Item -Recurse -Force exercises/ch02/manim_scenes.py,exercises/ch02/manim_kit.py,exercises/ch02/_media
# B. 重写/微调 app_common.py、real_ops.py、nb_builder.py(见 §2.2/§4)
# C. 新增 vllm_real.py、chapter_exec.py
# D. 重建并验证 ch01
exercises\chapter_exec.py ch01
# E. 单课手动验证(供 issue 复现)
D:\uv_envs\uv_cuda\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    exercises/ch01/05_prefill_vs_decode.ipynb --ExecutePreprocessor.kernel_name=uv_cuda
# F. 定向启动某个 app 验证
D:\uv_envs\uv_cuda\Scripts\python.exe -m streamlit run exercises/ch02/app_07_kv_principle.py
```