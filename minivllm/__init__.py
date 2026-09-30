# -*- coding: utf-8 -*-
"""minivllm — 一个 ~500 行的迷你 LLM 推理引擎。

把 exercises/ch01–ch04 的核心概念（分词、采样、分页 KV Cache、连续批处理）
落成一个能真跑的包，配套《minivllm》课程使用。

模块对应课程：
    tokenizer  → ch01 认识 Token 与分词器
    sampler    → ch01 采样策略
    paged_kv   → ch02 PagedAttention / Block Table / Copy-on-Write
    model      → ch04 ModelRunner 数据流
    engine     → ch03 Continuous Batching 与调度
"""
from minivllm.tokenizer import MiniBPETokenizer
from minivllm.sampler import Sampler
from minivllm.paged_kv import BlockAllocator, PagedKVCache
from minivllm.model import TinyGPT
from minivllm.engine import LLMEngine, Request

__all__ = [
    "MiniBPETokenizer",
    "Sampler",
    "BlockAllocator",
    "PagedKVCache",
    "TinyGPT",
    "LLMEngine",
    "Request",
]
__version__ = "0.1.0"
