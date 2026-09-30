<div align="center">

<img src="assets/logo.png" alt="minivllm logo" width="320"/>

# minivllm

An illustrated, hands-on course on how vLLM works — with a runnable mini inference engine.
</br>
<em>图解 vLLM 推理引擎：100 课动手练习册 + 一个能跑的迷你引擎</em>

[![CI](https://github.com/feiguangba/minivllm/actions/workflows/sanity.yml/badge.svg)](./.github/workflows/sanity.yml)
[![GitHub Stars](https://img.shields.io/github/stars/feiguangba/minivllm?style=flat-square&color=DAA520)](https://github.com/feiguangba/minivllm/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/feiguangba/minivllm?style=flat-square)](https://github.com/feiguangba/minivllm/network)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square)](https://www.python.org/)
[![Docker Pulls](https://img.shields.io/docker/pulls/fuyunsi/vllm-learn-labs?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/fuyunsi/vllm-learn-labs)
[![Binder](https://img.shields.io/badge/Binder-launch-E77C35?style=flat-square&logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/feiguangba/minivllm/master)

[English](./README_en.md) | [中文](./README.md)

</div>

## ⚡ Overview

**minivllm** is an open course that takes you from zero to understanding LLM inference systems. Around vLLM's core techniques — KV Cache, PagedAttention, continuous batching, CUDA Graph, quantization, parallelism, attention kernels, Triton, AI compilers, and the Huawei Ascend stack — it offers **11 chapters × 100 hands-on lessons**, plus a **~500-line, runnable, tested** mini inference engine that turns every concept into real code.

> All you need: follow the notebooks (Docker if you have a GPU, Binder if you don't)</br>
> What you get: a complete path from "using an inference engine" to "being able to write one"

## ✨ What's inside

| | Component | Description |
|---|---|---|
| 🧪 | **[Workbook](exercises/)** | 100 Jupyter lessons in 11 chapters — diagrams first, intuition first, experiments built in |
| 🚂 | **[Mini engine](minivllm/)** | ~500 lines replicating the vLLM skeleton: tokenizer → sampler → paged KV → continuous batching, with 10 tests |
| 📚 | **[Architecture notes](docs/)** | 8 source-level deep dives into vLLM V1, every claim backed by a verifiable `file:line` citation |
| 🖼️ | **[Figure gallery](exercises/figs/)** | 12 original top-conference-style SVG figures embedded in the lessons |
| ✅ | **[Answers](answers/)** | Reference answers to all 399 exercises |

## 🔄 Suggested path

1. **Core loop** · ch01–ch03: tokenization → KV Cache / PagedAttention → continuous batching
2. **Write the engine** · re-implement each of the five [minivllm](minivllm/) modules yourself
3. **Execution deep dive** · ch04–ch06: model execution / CUDA Graph → quantization → parallelism
4. **Kernels & serving** · ch07–ch08: attention kernels → end-to-end deployment
5. **Wider horizons** · ch09–ch11: Triton → AI compilers → Huawei Ascend

When you need source-level depth, open the matching note in [docs/](docs/); for a global intuition, start with the 12-figure gallery in [lesson 100](exercises/ch11/100_summary_roadmap.ipynb).

## 🚀 Quick start

```bash
git clone https://github.com/feiguangba/minivllm.git
cd minivllm
docker compose up --build          # GPU teaching image (torch 2.11.0+cu128)
```

Open [http://localhost:8888](http://localhost:8888) (token `vllm_learn`) and [http://localhost:8501](http://localhost:8501) (Streamlit demos). Or pull the prebuilt image: `docker pull fuyunsi/vllm-learn-labs:gpu` then `docker compose up -d labs` (~20GB).

| Environment | Best for | Link |
|---|---|---|
| **Docker (recommended)** | Full 100 lessons, needs an NVIDIA GPU | command above |
| **Binder** | Browser, CPU-friendly lessons | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/feiguangba/minivllm/master) |
| **Colab** | Free GPU, one notebook at a time | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/feiguangba/minivllm/blob/master/exercises/ch01/01_token_and_tokenizer.ipynb) |
| **Local uv** | Without Docker | `pip install -r requirements.txt` + `jupyter lab` |

Most content runs without a GPU — ch01–ch03 and the mini engine are pure CPU:

```bash
pip install -e .
python -m minivllm "the quick brown fox" --max-new 16 --seed 0 --verbose
```

## 🗺️ Course map

| Ch. | Topic | Lessons | Entry |
|---|---|---|---|
| ch01 | LLM inference basics (tokens / decoding / sampling / prefill vs decode) | 01–06 | [`exercises/ch01/`](exercises/ch01/) |
| ch02 | KV Cache & PagedAttention | 07–13 | [`exercises/ch02/`](exercises/ch02/) |
| ch03 | Continuous batching & scheduling | 14–20 | [`exercises/ch03/`](exercises/ch03/) |
| ch04 | Model execution & CUDA Graph | 21–27 | [`exercises/ch04/`](exercises/ch04/) |
| ch05 | Quantization (GPTQ / AWQ / FP8) | 28–34 | [`exercises/ch05/`](exercises/ch05/) |
| ch06 | Distributed parallelism (TP / PP / DP / EP) | 35–41 | [`exercises/ch06/`](exercises/ch06/) |
| ch07 | Attention kernels in practice | 42–47 | [`exercises/ch07/`](exercises/ch07/) |
| ch08 | End-to-end deployment & serving | 48–50 | [`exercises/ch08/`](exercises/ch08/) |
| ch09 | Triton GPU programming | 51–60 | [`exercises/ch09/`](exercises/ch09/) |
| ch10 | AI compilers | 61–70 | [`exercises/ch10/`](exercises/ch10/) |
| ch11 | Huawei Ascend full stack | 71–100 | [`exercises/ch11/`](exercises/ch11/) |

Each chapter folder contains the per-lesson notebooks (`NN_*.ipynb`) and Streamlit demos (`app_NN_*.py`); answers live in [`answers/`](answers/).

## 🖼️ Architecture figures (selection)

| | | |
|---|---|---|
| [Transformer block](exercises/figs/fig_01_transformer_block.svg) | [Prefill vs Decode](exercises/figs/fig_02_prefill_vs_decode.svg) | [KV Cache](exercises/figs/fig_03_kv_cache.svg) |
| [PagedAttention](exercises/figs/fig_04_pagedattention.svg) | [Continuous Batching](exercises/figs/fig_05_continuous_batching.svg) | [Request lifecycle](exercises/figs/fig_06_request_lifecycle.svg) |
| [vLLM system architecture](exercises/figs/fig_07_vllm_arch.svg) | [Inference pipeline](exercises/figs/fig_08_inference_pipeline.svg) | [Quantization](exercises/figs/fig_09_quantization.svg) |
| [Parallelism](exercises/figs/fig_10_parallelism.svg) | [FlashAttention](exercises/figs/fig_11_flash_attention.svg) | [Ascend stack](exercises/figs/fig_12_ascend_stack.svg) |

All 12 figures live in [`exercises/figs/`](exercises/figs/) and are embedded in their lessons.

## 🗂️ Repository layout

```
minivllm/
├── minivllm/                # ★ mini inference engine (~500 lines, tested)
├── exercises/               # workbook: 11 chapters, 100 lessons + 12 figures
├── docs/                    # 8 source-level vLLM V1 notes + version conventions
├── answers/                 # reference answers (399 exercises)
├── tests/                   # engine test suite
├── docker/                  # GPU teaching image (Dockerfile / entrypoint)
├── .binder/                 # cloud (CPU) environment for Binder
└── assets/                  # logo
```
