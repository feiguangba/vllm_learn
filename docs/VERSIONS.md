# 版本策略

教学仓库和 vLLM 运行时解耦。

| 层 | 版本 | 说明 |
|---|---|---|
| 练习册 / Streamlit | 不绑定 vLLM 包 | 用 `exercises/vllm_real.py`（TinyGPT），CPU 可跑完全部 100 课 |
| docs 架构文档 | vLLM **V1** · 标注为 v0.23.0-dev | 路径约定 `v1/engine/...`，见 `docs/01_system_architecture.md` 文首 |
| CPU 教学镜像 `vllm-learn-labs:cpu` | Python 3.12 + torch CPU | `docker compose up`，不含 vLLM |
| GPU profile `vllm` | 官方 `vllm/vllm-openai:latest` | `docker compose --profile gpu up`，默认 `Qwen/Qwen3-0.6B` |

## 和 V0 的关系

本仓库文档从一开始就按 V1 多进程拓扑写（前端 / EngineCore / Worker + ZMQ）。没有单独的 V0 源码树。若对照网上 2024 年博客里的 `LLMEngine` 单进程循环，以 `docs/` 为准。

## 本机 GPU 备注

作者环境曾用 RTX 5060 Laptop（sm_120）。官方 vLLM 镜像对 Blackwell 消费卡可能失败。失败不影响 CPU 教学路径。
