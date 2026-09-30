<div align="center">

<img src="assets/logo.png" alt="minivllm logo" width="340"/>

# minivllm · Illustrated vLLM Inference Engine

**Understand how a modern LLM inference engine works — with diagrams, hands-on experiments, and a runnable mini engine.**

KV Cache · PagedAttention · Continuous Batching · CUDA Graph · Quantization · Distributed Parallelism · Attention Kernels · Triton · AI Compilers · Huawei Ascend

[![CI](https://github.com/feiguangba/minivllm/actions/workflows/sanity.yml/badge.svg)](./.github/workflows/sanity.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/code-MIT-blue.svg)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Docker Pulls](https://img.shields.io/docker/pulls/fuyunsi/vllm-learn-labs.svg)](https://hub.docker.com/r/fuyunsi/vllm-learn-labs)

**100-lesson workbook · 8 source-level architecture notes · 12 top-conference-style figures · ~500-line runnable mini engine · 399 exercise answers**

> 🇨🇳 中文版 README：[README.md](README.md)（内容更全，含逐课清单）

</div>

---

## ⚡ Quick start

```bash
git clone https://github.com/feiguangba/minivllm.git
cd minivllm
docker compose up --build          # GPU teaching image (torch 2.11.0+cu128)
```

Open [http://localhost:8888](http://localhost:8888) (token: `vllm_learn`) and [http://localhost:8501](http://localhost:8501); pick the **Python 3 (vllm_learn)** kernel. No local build? `docker pull fuyunsi/vllm-learn-labs:gpu` then `docker compose up -d labs`.

No GPU? [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/feiguangba/minivllm/master) runs most of ch01–ch03 on CPU in your browser.

## 📦 What's inside

| | Module | Path | What it is |
|---|---|---|---|
| 🧪 | **Workbook (100 lessons)** | [`exercises/`](exercises/) | 11 chapters × 100 Jupyter notebooks — concept → minimal implementation → visualization → experiment; ch01–ch08 also ship Streamlit demos |
| 🚂 | **minivllm mini engine** | [`minivllm/`](minivllm/) | ~500 lines turning the core ideas into a runnable, tested Python package — the skeleton of vLLM in miniature |
| 📚 | **repowiki** | [`repowiki/`](repowiki/) | 8 source-level architecture notes with `file:line` citations, based on vLLM **V1** `v0.23.0-dev` |
| 🖼️ | **Figure gallery** | [`exercises/figs/`](exercises/figs/) | 12 original SVG architecture figures embedded in the lessons |
| ✅ | **Answers** | [`answers/`](answers/) | Reference answers to all 399 exercises across the 100 lessons |

## 🚂 minivllm: turn the course into a working engine

Each module maps to specific lessons:

| Module | Lessons | Covers |
|---|---|---|
| `tokenizer.py` | ch01 · L01 | mini BPE: merges → vocab → compression ratio |
| `sampler.py` | ch01 · L03 | greedy / temperature / top-k / top-p |
| `paged_kv.py` | ch02 · L10/11/13 | BlockAllocator, block table, prefix sharing (refcount), fragmentation |
| `model.py` | ch04 · L04/22 | prefill (parallel) and decode (incremental) paths |
| `engine.py` | ch03 · L14–18 | FCFS admission, in-batch rotation, completion, recompute preemption |

```bash
pip install -e .
python -m minivllm "the quick brown fox" --max-new 16 --seed 0 --verbose
```

The model is a randomly-initialized TinyGPT — the outputs are pseudo-text. The point is the **engine's data flow and scheduling**, not model quality. See [`minivllm/README.md`](minivllm/README.md).

## 🖼️ Architecture figures

12 original SVG figures drawn in a top-conference paper style live in [`exercises/figs/`](exercises/figs/) — overall architecture, PagedAttention block table, scheduler state machine, CUDA graph capture/replay, TP/PP/DP layouts, attention kernel anatomy, and more. The drawing spec is in `.claude/skills/paper-fig/`. Lesson [100](exercises/ch11/100_summary_roadmap.ipynb) opens with the full gallery.

## 📚 repowiki — source-level architecture notes

Eight deep-dive documents in [`repowiki/`](repowiki/) derived from reading the **vLLM V1** source (main branch @ commit [967e104](https://github.com/vllm-project/vllm/commit/967e104)):

1. [System architecture](repowiki/01_system_architecture.md) — processes, RPC, engine core loop
2. [PagedAttention & KV cache](repowiki/02_PagedAttention_KVCache.md) — block manager, allocator, prefix caching
3. [Scheduler](repowiki/03_Scheduler.md) — admission, batching, preemption policies
4. [LLMEngine](repowiki/04_LLMEngine.md) — sync/async frontends, output processing
5. [ModelRunner & CUDA Graph](repowiki/05_ModelRunner_CUDAGraph.md) — execute_model path, graph capture
6. [Quantization](repowiki/06_Quantization.md) — method registry, online/offline paths
7. [TP / PP / DP](repowiki/07_TP_PP_DP.md) — parallelism groups, communication schedules
8. [Attention kernels](repowiki/08_Attention_Kernels.md) — backend selection, FlashAttention integration

All `file:line` citations use relative paths into the upstream `vllm/` tree; you can verify each one online via the [permalink base](https://github.com/vllm-project/vllm/blob/967e104/vllm/v1/engine/core.py) — no local clone needed.

## ☁️ Run in the cloud

| Environment | Best for | Link |
|---|---|---|
| **Binder** | No local Python/GPU — browse most CPU-friendly lessons | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/feiguangba/minivllm/master) |
| **Colab** | Free GPU for a single notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/feiguangba/minivllm/blob/master/exercises/ch01/01_token_and_tokenizer.ipynb) |
| **Local Docker (recommended)** | The full 100 lessons on your own NVIDIA GPU | `docker compose up --build` |

## 🚀 Run it

### Docker (recommended — GPU teaching image)

Requires an NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The image ships **torch 2.11.0+cu128** (RTX 50-series / sm_120 compatible), Jupyter Lab and Streamlit.

```bash
docker compose up --build        # or: docker pull fuyunsi/vllm-learn-labs:gpu
```

- Jupyter Lab: <http://localhost:8888> — token `vllm_learn`, kernel **Python 3 (vllm_learn)**
- Streamlit hub: <http://localhost:8501> — switch demos via the `APP` env var:

```bash
docker compose run --rm -e APP=exercises/ch02/app_10_paged_demo.py -p 8501:8501 labs
```

Building in mainland China and pip keeps stalling? See the offline three-stage recipe in [`docker/Dockerfile.local`](docker/Dockerfile.local).

### Local uv / venv

Python 3.12; install `requirements.txt` plus CUDA torch (`pip install torch --index-url https://download.pytorch.org/whl/cu128`), then `jupyter lab` and pick the **Python 3 (vllm_learn)** kernel. On Windows set `KMP_DUPLICATE_LIB_OK=TRUE`.

### Optional: real vLLM server

```bash
docker compose --profile gpu up   # serves Qwen/Qwen3-0.6B on :8000 via vllm/vllm-openai
```

Not required for the 100 lessons.

### Suggested learning order

ch01 → ch02 → ch03 (core inference loop) → write/read the [minivllm](minivllm/) engine → ch04 → ch09 → ch05/ch06 → ch07 → ch08 → ch10/ch11. When you need source-level depth, open the matching repowiki note next to the chapter.

## 🗂️ Repository layout

```
minivllm/
├── README.md / README_en.md
├── VERSIONS.md              # version conventions for lessons vs vLLM V1
├── docker-compose.yml       # labs (GPU teaching image) + vllm-openai service
├── requirements.txt
├── minivllm/                # ★ mini inference engine package (~500 lines, tested)
├── tests/                   # 10 test cases for minivllm
├── answers/                 # reference answers for all 399 exercises
├── docker/                  # Dockerfile / entrypoint / Streamlit hub
├── .binder/                 # cloud (CPU) environment for Binder
├── assets/                  # logo
├── repowiki/                # 8 source-level architecture notes
└── exercises/               # 11 chapters, 100 lessons
    ├── GUIDELINES.md       # writing spec
    ├── nb_builder.py        # notebook generation tooling
    ├── figs/                # 12 SVG architecture figures
    └── ch01/ .. ch11/       # NN_*.ipynb lessons + app_NN_*.py demos
```

## Versions

See [VERSIONS.md](VERSIONS.md). Notes target **vLLM V1** (v0.23-dev @ 967e104). The notebooks do **not** import the `vllm` package — everything is implemented from scratch so you can see every moving part.

## ✍️ License & references

This repo is a study companion, not official documentation.

- **Code** (`.py` files, notebook code cells, `docker/`, `minivllm/`) is licensed under the [MIT License](./LICENSE);
- **Docs & figures** (`repowiki/*.md`, `exercises/figs/*.svg`, notebook prose) are additionally licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — share and adapt freely with attribution and a link back;
- References: [vLLM docs](https://docs.vllm.ai) · [PagedAttention](https://arxiv.org/abs/2309.06180) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [Orca](https://arxiv.org/abs/2208.14217).
