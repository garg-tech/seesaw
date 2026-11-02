"""KV storage and the block tables the Triton kernels read.

Full layers need every token, so each slot owns a fixed run of blocks. Sliding
layers only need the last `window` tokens, but a prefill chunk writes its whole
length before older blocks can go, so their blocks come from a shared pool:
every slot's window plus scratch for the chunks in flight. A block id means the
same thing in every sliding layer, so one table serves them all.
"""
import numpy as np
import torch

from .config import BLOCK_SIZE


class BlockTables:
    """CPU-side bookkeeping, one instance per GPU."""

    def __init__(self, slots, max_len, window, scratch_blocks):
        self.slots = slots
        self.window = window
        self.max_blocks = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.ring = (window + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
        self.num_full = slots * self.max_blocks
        self.num_sliding = slots * self.ring + scratch_blocks
        self.free = list(range(self.num_sliding - 1, -1, -1))
        # Released entries stay valid ids: kernel tiles align down past the window
        # and read them, then mask them out, so they must only be in bounds.
        self.sliding = np.zeros((slots, self.max_blocks), np.int32)
        self.full = (np.arange(slots, dtype=np.int32)[:, None] * self.max_blocks
                     + np.arange(self.max_blocks, dtype=np.int32)[None, :])
        self.head = np.zeros(slots, np.int64)     # first sliding logical block still held
        self.len = np.zeros(slots, np.int64)      # tokens allocated

    def extend(self, slot, new_len):
        """Back tokens up to new_len with sliding blocks."""
        have = (int(self.len[slot]) + BLOCK_SIZE - 1) // BLOCK_SIZE
        need = (new_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(have, need):
            self.sliding[slot, b] = self.free.pop()
        self.len[slot] = new_len

    def trim(self, slot, done):
        """Release sliding blocks no query after token `done` can see."""
        first_kept = max(done - self.window + 1, 0) // BLOCK_SIZE
        for b in range(int(self.head[slot]), first_kept):
            self.free.append(int(self.sliding[slot, b]))
        self.head[slot] = max(int(self.head[slot]), first_kept)

    def used_sliding(self):
        free = set(self.free)
        return [b for b in range(self.num_sliding) if b not in free]

    def used_full(self):
        """Blocks holding tokens, for the slots that have any."""
        out = []
        for slot in np.flatnonzero(self.len):
            n = (int(self.len[slot]) + BLOCK_SIZE - 1) // BLOCK_SIZE
            out.extend(self.full[slot, :n].tolist())
        return out

    def release(self, slot):
        have = (int(self.len[slot]) + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(int(self.head[slot]), have):
            self.free.append(int(self.sliding[slot, b]))
        self.head[slot] = self.len[slot] = 0


class Batch:
    """One forward pass: `lens[i]` new tokens for `slots[i]`, starting at `starts[i]`."""

    def __init__(self, tables, slots, starts, lens, device):
        slots, starts, lens = map(lambda a: np.asarray(a, np.int64), (slots, starts, lens))
        pos = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)])
        seq = np.repeat(np.arange(len(slots)), lens)
        blk, off = pos // BLOCK_SIZE, pos % BLOCK_SIZE
        row = slots[seq]
        as_tensor = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
        self.positions = as_tensor(pos)
        self.cu_seqlens = as_tensor(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
        self.seq_lens = as_tensor((starts + lens).astype(np.int32))
        self.max_q = int(lens.max())
        self.max_k = int((starts + lens).max())
        n = (self.max_k + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.table_full = as_tensor(tables.full[slots, :n].copy())
        self.table_sliding = as_tensor(tables.sliding[slots, :n].copy())
        self.map_full = as_tensor(tables.full[row, blk].astype(np.int64) * BLOCK_SIZE + off)
        self.map_sliding = as_tensor(tables.sliding[row, blk].astype(np.int64) * BLOCK_SIZE + off)
