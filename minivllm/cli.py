# -*- coding: utf-8 -*-
"""minivllm 命令行入口。

用法：
    python -m minivllm "你好，世界" --max-new 16 --seed 0
"""
import argparse
import torch

from minivllm.tokenizer import MiniBPETokenizer
from minivllm.sampler import Sampler
from minivllm.paged_kv import PagedKVCache
from minivllm.model import TinyGPT
from minivllm.engine import LLMEngine


def main(argv=None):
    ap = argparse.ArgumentParser(prog="minivllm",
                                 description="迷你 LLM 推理引擎（教学用，随机权重输出伪文本）")
    ap.add_argument("prompt", help="输入 prompt")
    ap.add_argument("--max-new", type=int, default=16, help="最多生成的 token 数")
    ap.add_argument("--seed", type=int, default=0, help="采样随机种子")
    ap.add_argument("--verbose", action="store_true", help="打印调度日志")
    args = ap.parse_args(argv)

    # 1) 分词（用一段小语料训练出 300 词表的迷你 BPE）
    tok = MiniBPETokenizer(vocab_size=300).train(
        ["the quick brown fox jumps over the lazy dog " * 4 +
         "large language model inference engine kv cache paged attention " * 4])
    prompt_ids = tok.encode(args.prompt)

    # 2) 模型 + 分页 KV 缓存 + 采样器 + 引擎
    model = TinyGPT(vocab_size=tok.vocab_size)
    cache = PagedKVCache(num_blocks=64, block_size=8,
                         n_heads=model.n_heads, head_dim=model.head_dim,
                         n_layers=model.n_layers)
    engine = LLMEngine(model, cache, Sampler(temperature=0.8, top_k=20, seed=args.seed))
    engine.eos_id = None
    engine.add_request(tok.encode(args.prompt), max_new_tokens=args.max_new)

    # 3) 跑
    outputs = engine.run(verbose=args.verbose)
    out_ids = next(iter(outputs.values()))
    print(f"prompt  ({len(prompt_ids)} tok): {args.prompt}")
    print(f"output  ({len(out_ids)} tok): {tok.decode(out_ids)}")
    print(f"碎片率  : {cache.fragmentation():.1%}  (请求结束后归零属正常)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
