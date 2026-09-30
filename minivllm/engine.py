# -*- coding: utf-8 -*-
"""连续批处理引擎（对应第 14–18 课）。

调度循环每「一步」：
  1) admit   —— 从 WAITING 队列按 FCFS 放行请求：prefill 整段 prompt，
                立刻采出第 1 个 token（prefill 的产出就是它）；
  2) decode  —— RUNNING 请求轮转前进：把「上一轮新采的 token」喂进模型
                （它的 K/V 还没入缓存），采出下一个 token；
  3) finish  —— 生成满 / 命中 EOS 的请求归还全部物理块；
  4) preempt —— 物理块耗尽时抢占（recompute：丢弃缓存重新排队）。
"""
from collections import deque

from minivllm.paged_kv import OutOfBlocksError


class Request:
    _next_id = 0

    def __init__(self, prompt_ids, max_new_tokens=32):
        Request._next_id += 1
        self.seq_id = Request._next_id
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.finished = False

    @property
    def num_tokens(self):
        return len(self.prompt_ids) + len(self.output_ids)


class LLMEngine:
    def __init__(self, model, cache, sampler, max_running=4):
        self.model = model
        self.cache = cache
        self.sampler = sampler
        self.max_running = max_running
        self.waiting = deque()
        self.running = []                     # 保持到达顺序（FCFS）
        self.done = []                        # 已完成请求
        self.eos_id = None                    # 可选：命中即停
        self.verbose = False

    # ---------- 对外接口 ----------
    def add_request(self, prompt_ids, max_new_tokens=32):
        self.waiting.append(Request(prompt_ids, max_new_tokens))

    def run(self, verbose=False):
        """跑到所有请求完成，返回 {seq_id: 生成的 token ids}。"""
        self.verbose = verbose
        while self.waiting or self.running:
            self._admit(verbose)
            if self.running:
                self._decode_step(verbose)
        return {r.seq_id: r.output_ids for r in self.done}

    # ---------- 调度内部 ----------
    def _admit(self, verbose):
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            try:
                self.cache.new_seq(req.seq_id)
                logits = self.model.prefill(req.prompt_ids, self.cache, req.seq_id)
            except OutOfBlocksError:           # 显存不足以 prefill → 停止放行
                self.cache.drop_seq(req.seq_id)
                break
            self.waiting.popleft()
            self.running.append(req)
            self._sample_and_store(req, logits)
            if verbose:
                print(f"[admit ] seq {req.seq_id}  prompt {len(req.prompt_ids)} tok"
                      f"  first={req.output_ids[-1]}")

    def _decode_step(self, verbose):
        # 轮转调度：每步只让队首 RUNNING 请求前进一步 = 批内公平交错
        req = self.running.pop(0)
        try:
            logits = self.model.decode_step(req.output_ids[-1], self.cache, req.seq_id)
        except OutOfBlocksError:               # 块耗尽 → 抢占（recompute 策略）
            self._preempt(req, verbose)
            return
        self._sample_and_store(req, logits)
        if verbose:
            print(f"[decode] seq {req.seq_id}  {len(req.output_ids)}/{req.max_new_tokens} tok")

    def _sample_and_store(self, req, logits):
        token = int(self.sampler(logits))
        req.output_ids.append(token)
        if len(req.output_ids) >= req.max_new_tokens or token == self.eos_id:
            self.cache.drop_seq(req.seq_id)
            req.finished = True
            self.done.append(req)
            if req in self.running:
                self.running.remove(req)
            if self.verbose:
                print(f"[finish] seq {req.seq_id}  generated {len(req.output_ids)} tok")
        elif req not in self.running:
            self.running.append(req)           # 未完成 → 回到轮转队列继续解码

    def _preempt(self, req, verbose):
        """抢占：丢弃它的缓存，整个请求（含已生成部分）回 WAITING 等重算。"""
        self.cache.drop_seq(req.seq_id)
        req.output_ids = []                    # recompute：从 prompt 重放
        self.waiting.appendleft(req)
        if verbose:
            print(f"[preemp] seq {req.seq_id}  → 重新排队 (blocks free={self.cache.allocator.num_free})")
