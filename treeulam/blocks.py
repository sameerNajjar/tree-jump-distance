"""Common-block contraction and the pool of labels used by the approximation algorithms.

A common block is a maximal run of symbols that is contiguous and identically ordered
in every input permutation. Contracting each block to one symbol whose weight is the
block length preserves every distance among the inputs and among their medians.
"""
import numpy as np

from .distance import Metric, wlis


def common_blocks(seqs):
    """Maximal blocks that are contiguous and identically ordered in all seqs."""
    s0 = seqs[0]
    adj = {(s0[i], s0[i + 1]) for i in range(len(s0) - 1)}
    for s in seqs[1:]:
        adj &= {(s[i], s[i + 1]) for i in range(len(s) - 1)}
    blocks, cur = [], [s0[0]]
    for a, b in zip(s0, s0[1:]):
        if (a, b) in adj:
            cur.append(b)
        else:
            blocks.append(cur)
            cur = [b]
    blocks.append(cur)
    return blocks


class Space:
    """A set of permutations contracted by their common blocks (weight = block length)."""

    def __init__(self, seqs):
        self.blocks = common_blocks([list(s) for s in seqs])
        self.block_of = {}
        for b, B in enumerate(self.blocks):
            for s in B:
                self.block_of[s] = b
        self.w = [len(B) for B in self.blocks]
        self.k = len(self.blocks)

    def contract(self, seq):
        out, last = [], None
        for s in seq:
            b = self.block_of[s]
            if b != last:
                out.append(b)
                last = b
        if len(out) != self.k:
            raise ValueError("sequence splits a common block")
        return tuple(out)

    def expand(self, bseq):
        return tuple(s for b in bseq for s in self.blocks[b])


class Pool:
    """Labels of one contracted weighted space, indexed, with cached distances."""

    def __init__(self, weights):
        self.w = list(weights)
        self.W = sum(self.w)
        self.items, self.index, self.pos, self.wseq = [], {}, [], []
        self.cache = {}
        self.metric = Metric(self.w)

    def add(self, seq):
        seq = tuple(seq)
        i = self.index.get(seq)
        if i is None:
            i = len(self.items)
            self.items.append(seq)
            self.index[seq] = i
            p = [0] * len(seq)
            for r, s in enumerate(seq):
                p[s] = r
            self.pos.append(p)
            self.wseq.append([self.w[s] for s in seq])
        return i

    def d(self, i, j):
        if i == j:
            return 0
        key = (i, j) if i < j else (j, i)
        v = self.cache.get(key)
        if v is None:
            p = self.pos[i]
            v = self.W - wlis([p[s] for s in self.items[j]], self.wseq[j])
            self.cache[key] = v
        return v

    def dmat(self, idx, jdx, dtype=np.int64):
        """|idx| x |jdx| distance matrix, vectorised (no per-pair cache entries)."""
        idx, jdx = list(idx), list(jdx)
        out = np.empty((len(idx), len(jdx)), dtype=dtype)
        if not idx or not jdx:
            return out
        if len(idx) * len(jdx) <= 64:
            for a, i in enumerate(idx):
                for b, j in enumerate(jdx):
                    out[a, b] = self.d(i, j)
            return out
        C = np.array([self.items[j] for j in jdx], dtype=np.int64)
        Cp = self.metric.prepare(C)
        A = np.array([self.items[i] for i in idx], dtype=np.int64)
        B = max(1, 1_000_000 // len(jdx))
        for s0 in range(0, len(idx), B):
            out[s0:s0 + B] = self.metric.many_prepared(A[s0:s0 + B], Cp)
        return out
