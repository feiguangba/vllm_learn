# minivllm · 一个 ~500 行的迷你 LLM 推理引擎

`minivllm` 是《图解 vLLM 推理引擎》练习册的配套包：把 ch01–ch04 的核心概念
（分词 → 采样 → 分页 KV Cache → 连续批处理）落成一个能真跑、带测试的 Python 包。

> 模型是**随机权重**的 TinyGPT——输出是伪文本。本包教的是**推理引擎的数据流与调度**，
> 不是模型质量；想接真模型，把 `TinyGPT` 换成任意能逐步吐 logits 的 decoder-only 模型即可。

## 安装

```bash
pip install -e .          # 仓库根目录执行；依赖 torch>=2.0
```

## 快速上手

命令行：

```bash
python -m minivllm "the quick brown fox" --max-new 16 --seed 0 --verbose
```

Python：

```python
from minivllm import MiniBPETokenizer, Sampler, PagedKVCache, TinyGPT, LLMEngine

tok = MiniBPETokenizer(vocab_size=300).train(["the quick brown fox " * 8])
model = TinyGPT(vocab_size=tok.vocab_size)
cache = PagedKVCache(num_blocks=64, block_size=8,
                     n_heads=model.n_heads, head_dim=model.head_dim,
                     n_layers=model.n_layers)
engine = LLMEngine(model, cache, Sampler(temperature=0.8, top_k=20, seed=0))
engine.add_request(tok.encode("the quick brown fox"), max_new_tokens=16)
outputs = engine.run(verbose=True)
print(tok.decode(next(iter(outputs.values()))))
```

## 模块 ↔ 课程对照

| 模块 | 行数级 | 对应课程 | 学什么 |
|---|---|---|---|
| `tokenizer.py` | ~70 | ch01 第 01 课 | 迷你 BPE：合并对 → 词表 → 压缩率 |
| `sampler.py` | ~45 | ch01 第 03 课 | greedy / temperature / top-k / top-p |
| `paged_kv.py` | ~100 | ch02 第 10/11/13 课 | BlockAllocator、block table、fork 前缀共享（refcount）、内部碎片率 |
| `model.py` | ~120 | ch04 第 04/22 课 | prefill（并行）与 decode（增量）两条路径 |
| `engine.py` | ~110 | ch03 第 14–18 课 | FCFS 准入、批内轮转、完成回收、块耗尽抢占（recompute） |

## 设计取舍（教学优先）

- **逐 token 写缓存**：不做批量 `slot_mapping` 写入，一眼看清每个 token 落在哪个物理块；
- **gather 式注意力**：解码前把非连续物理块 `torch.cat` 回连续 K/V（即第 45 课的 gather 思路），
  不实现 paged kernel —— 那是 ch07/45 与 Triton 章（ch09）的课题；
- **抢占用 recompute**：丢弃缓存整段重放，与第 18 课的模拟口径一致。

## 测试

```bash
python -m pytest tests/test_minivllm.py -q   # 10 个用例：分词往返/采样约束/引用计数/前缀共享/引擎端到端
```

## License

MIT（随仓库根目录 LICENSE）。
