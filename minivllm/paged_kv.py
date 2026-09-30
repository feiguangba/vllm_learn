# -*- coding: utf-8 -*-
"""分页 KV Cache（对应第 10/11/13 课）：BlockAllocator + PagedKVCache。

物理显存被切成固定大小的 block；每个序列通过 block_table 里的逻辑块号
找到物理块 —— 就是操作系统「虚拟内存页表」在 GPU 上的翻版。
"""
import torch


class OutOfBlocksError(RuntimeError):
    """物理块耗尽 → 引擎层应触发抢占（preemption）。"""


class BlockAllocator:
    """管理物理块的分配/回收/引用计数（前缀共享与 COW 的地基）。"""

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = list(range(num_blocks))     # free list
        self.refcount = [0] * num_blocks

    def allocate(self):
        if not self.free:
            raise OutOfBlocksError(f"no free block ({self.num_blocks} used)")
        b = self.free.pop(0)
        self.refcount[b] = 1
        return b

    def retain(self, b):
        self.refcount[b] += 1

    def release(self, b):
        self.refcount[b] -= 1
        if self.refcount[b] <= 0:
            self.refcount[b] = 0
            self.free.append(b)

    @property
    def num_free(self):
        return len(self.free)


class PagedKVCache:
    """按块存放 K/V：blocks[b] 形状 (block_size, n_heads, head_dim)。"""

    def __init__(self, num_blocks, block_size, n_heads, head_dim, n_layers=1, dtype=torch.float32):
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.k = [torch.zeros(num_blocks, block_size, n_heads, head_dim, dtype=dtype)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(num_blocks, block_size, n_heads, head_dim, dtype=dtype)
                  for _ in range(n_layers)]
        # seq_id -> block_table；seq_id -> 已写入的 token 数
        self.block_tables = {}
        self.seqlens = {}

    # ---------- 生命周期 ----------
    def new_seq(self, seq_id):
        self.block_tables[seq_id] = []
        self.seqlens[seq_id] = 0

    def drop_seq(self, seq_id):
        for b in self.block_tables.pop(seq_id, []):
            self.allocator.release(b)
        self.seqlens.pop(seq_id, None)

    def fork(self, src_id, dst_id):
        """前缀共享：复制 block table 并把引用计数 +1（第 12/13 课）。"""
        table = list(self.block_tables[src_id])
        for b in table:
            self.allocator.retain(b)
        self.block_tables[dst_id] = table
        self.seqlens[dst_id] = self.seqlens[src_id]

    def _append_slot(self, seq_id):
        """确保 seq 还有可写的槽位；不够就续块（可能触发 OutOfBlocksError）。"""
        length = self.seqlens[seq_id]
        if not self.block_tables[seq_id] or length % self.block_size == 0:
            self.block_tables[seq_id].append(self.allocator.allocate())
        table = self.block_tables[seq_id]
        return table[length // self.block_size], length % self.block_size

    def write(self, seq_id, layer, k, v):
        """追加写入一个 token 的 K/V：k, v 形状 (n_heads, head_dim)。"""
        b, off = self._append_slot(seq_id)
        self.k[layer][b, off] = k
        self.v[layer][b, off] = v
        self.seqlens[seq_id] += 1

    def gather(self, seq_id, layer):
        """把逻辑序列的非连续物理块拼回连续 KV（第 45 课的 gather 思路）。"""
        t = self.seqlens[seq_id]
        if t == 0:
            return None, None
        table = self.block_tables[seq_id]
        k = torch.cat([self.k[layer][b] for b in table])[:t]
        v = torch.cat([self.v[layer][b] for b in table])[:t]
        return k, v

    def fragmentation(self):
        """内部碎片率：已分配但尚未写满的槽位占比。"""
        used = sum(self.seqlens.values())
        allocated = sum(len(t) for t in self.block_tables.values()) * self.block_size
        return 1.0 - used / allocated if allocated else 0.0
