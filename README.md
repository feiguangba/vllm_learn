<div align="center">

<img src="assets/logo.png" alt="minivllm logo" width="320"/>

# minivllm

图解 vLLM 推理引擎：100 课动手练习册 + 一个能跑的迷你引擎
</br>
<em>An illustrated, hands-on course on how vLLM works — with a runnable mini inference engine.</em>

[![CI](https://github.com/feiguangba/minivllm/actions/workflows/sanity.yml/badge.svg)](./.github/workflows/sanity.yml)
[![GitHub Stars](https://img.shields.io/github/stars/feiguangba/minivllm?style=flat-square&color=DAA520)](https://github.com/feiguangba/minivllm/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/feiguangba/minivllm?style=flat-square)](https://github.com/feiguangba/minivllm/network)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square)](https://www.python.org/)
[![Docker Pulls](https://img.shields.io/docker/pulls/fuyunsi/vllm-learn-labs?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/fuyunsi/vllm-learn-labs)
[![Binder](https://img.shields.io/badge/Binder-launch-E77C35?style=flat-square&logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/feiguangba/minivllm/master)

[中文](./README.md) | [English](./README_en.md)

</div>

## ⚡ 一句话介绍

**minivllm** 是一套从零到一理解 LLM 推理系统的开源课程。围绕 vLLM 的核心技术——KV Cache、PagedAttention、Continuous Batching、CUDA Graph、量化、并行、Attention Kernel、Triton、AI 编译器、昇腾全栈——展开 **11 章 100 课** 的动手练习，并配一个 **~500 行、可运行、带测试** 的迷你推理引擎，把每节课的概念落成真实代码。

> 你只需要：跟着课程跑 notebook（有 GPU 用 Docker，没有用 Binder）</br>
> 你将得到：一条从「会用」到「能自己写出推理引擎」的完整学习路径

## ✨ 项目组成

| | 组件 | 说明 |
|---|---|---|
| 🧪 | **[练习册](exercises/)** | 11 章 100 课 Jupyter notebook，重图解、重直觉、每课自带实验 |
| 🚂 | **[迷你引擎](minivllm/)** | ~500 行复刻 vLLM 骨架：分词 → 采样 → 分页 KV → 连续批处理，10 个测试用例 |
| 📚 | **[架构文档](docs/)** | 8 篇 vLLM V1 源码级精读，全部带 `文件:行号` 可查证引用 |
| 🖼️ | **[架构图库](exercises/figs/)** | 12 张顶会风格 SVG 原创图，嵌入对应课程 |
| ✅ | **[参考答案](answers/)** | 全部 399 道课后练习题的参考答案 |

## 🔄 学习路径

1. **入门主线** · ch01–ch03：分词 → KV Cache / PagedAttention → 连续批处理
2. **写出引擎** · 对照 [minivllm](minivllm/) 五个模块自己实现一遍
3. **深入执行** · ch04–ch06：模型执行 / CUDA Graph → 量化 → 分布式并行
4. **底层与部署** · ch07–ch08：Attention Kernel → 服务化
5. **扩展视野** · ch09–ch11：Triton → AI 编译器 → 昇腾全栈

需要源码级细节时，随时对照 [docs/](docs/) 的 8 篇架构文档；想建立全局直觉，先看[第 100 课](exercises/ch11/100_summary_roadmap.ipynb)的 12 张架构图总画廊。

## 🚀 快速开始

```bash
git clone https://github.com/feiguangba/minivllm.git
cd minivllm
docker compose up --build          # GPU 教学镜像（torch 2.11.0+cu128）
```

浏览器打开 [http://localhost:8888](http://localhost:8888)（token `vllm_learn`）与 [http://localhost:8501](http://localhost:8501)（Streamlit 交互演示）。

不想本地构建？`docker pull fuyunsi/vllm-learn-labs:gpu` 后 `docker compose up -d labs`（镜像约 20GB）。

| 环境 | 适合 | 入口 |
|---|---|---|
| **Docker（推荐）** | 完整跑通 100 课，需 NVIDIA GPU | 上面的命令 |
| **Binder** | 无本地环境，浏览器体验 CPU 课程 | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/feiguangba/minivllm/master) |
| **Colab** | 免费 GPU 逐课打开 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/feiguangba/minivllm/blob/master/exercises/ch01/01_token_and_tokenizer.ipynb) |
| **本机 uv** | 不想用 Docker | `pip install -r requirements.txt` + `jupyter lab` |

没有 GPU 也能跑大部分课程——ch01–ch03 与迷你引擎 `minivllm` 纯 CPU 即可：

```bash
pip install -e .
python -m minivllm "the quick brown fox" --max-new 16 --seed 0 --verbose
```

## 🗺️ 课程地图

| 章 | 主题 | 课数 | 入口 |
|---|---|---|---|
| ch01 | LLM 推理基础（分词 / 自回归 / 采样 / Prefill vs Decode） | 01–06 | [`exercises/ch01/`](exercises/ch01/) |
| ch02 | KV Cache 与 PagedAttention | 07–13 | [`exercises/ch02/`](exercises/ch02/) |
| ch03 | Continuous Batching 与调度 | 14–20 | [`exercises/ch03/`](exercises/ch03/) |
| ch04 | 模型执行与 CUDA Graph | 21–27 | [`exercises/ch04/`](exercises/ch04/) |
| ch05 | 量化（GPTQ / AWQ / FP8） | 28–34 | [`exercises/ch05/`](exercises/ch05/) |
| ch06 | 分布式并行（TP / PP / DP / EP） | 35–41 | [`exercises/ch06/`](exercises/ch06/) |
| ch07 | Attention Kernel 实战 | 42–47 | [`exercises/ch07/`](exercises/ch07/) |
| ch08 | 端到端部署与服务化 | 48–50 | [`exercises/ch08/`](exercises/ch08/) |
| ch09 | Triton GPU 编程 | 51–60 | [`exercises/ch09/`](exercises/ch09/) |
| ch10 | AI 编译器 | 61–70 | [`exercises/ch10/`](exercises/ch10/) |
| ch11 | 华为昇腾全栈 | 71–100 | [`exercises/ch11/`](exercises/ch11/) |

每章目录内含逐课 notebook（`NN_*.ipynb`）与 Streamlit 交互演示（`app_NN_*.py`），配套答案在 [`answers/`](answers/)。

## 🖼️ 核心架构图（部分）

| | | |
|---|---|---|
| [Transformer 架构](exercises/figs/fig_01_transformer_block.svg) | [Prefill vs Decode](exercises/figs/fig_02_prefill_vs_decode.svg) | [KV Cache](exercises/figs/fig_03_kv_cache.svg) |
| [PagedAttention](exercises/figs/fig_04_pagedattention.svg) | [Continuous Batching](exercises/figs/fig_05_continuous_batching.svg) | [请求状态机](exercises/figs/fig_06_request_lifecycle.svg) |
| [vLLM 系统架构](exercises/figs/fig_07_vllm_arch.svg) | [推理流水线](exercises/figs/fig_08_inference_pipeline.svg) | [量化全景](exercises/figs/fig_09_quantization.svg) |
| [分布式并行](exercises/figs/fig_10_parallelism.svg) | [FlashAttention](exercises/figs/fig_11_flash_attention.svg) | [昇腾全栈对照](exercises/figs/fig_12_ascend_stack.svg) |

全部 12 张图集中在 [`exercises/figs/`](exercises/figs/)，并已嵌入对应课程。

## 🗂️ 目录结构

```
minivllm/
├── minivllm/                # ★ 迷你推理引擎包（~500 行，带测试）
├── exercises/               # 练习册：11 章 100 课 + 12 张架构图
├── docs/                    # 8 篇 vLLM V1 源码级架构文档 + 版本约定
├── answers/                 # 399 题参考答案
├── tests/                   # 迷你引擎测试
├── docker/                  # GPU 教学镜像（Dockerfile / entrypoint）
├── .binder/                 # Binder 云端环境（CPU）
└── assets/                  # LOGO
```
