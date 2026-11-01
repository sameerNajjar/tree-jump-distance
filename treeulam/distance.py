"""Weighted Ulam distance.

Every symbol s carries a weight w[s] >= 1 (all ones in the unweighted case).
For two permutations p, q of one alphabet, d_w(p, q) = W - (largest total weight of
a common subsequence), W = sum of all weights. A common subsequence of two
permutations is an increasing subsequence after relabelling, so the distance is a
weighted longest-increasing-subsequence computation.
"""
from bisect import bisect_left

import numpy as np


def wlis(xs, ws):
    """Largest total weight of a strictly increasing subsequence of xs (distinct values)."""
    ks, vs = [], []                       # staircase: keys increasing, values increasing
    for x, w in zip(xs, ws):
        i = bisect_left(ks, x)
        c = (vs[i - 1] if i else 0) + w
        j = i
        L = len(vs)
        while j < L and vs[j] <= c:
            j += 1
        ks[i:j] = [x]
        vs[i:j] = [c]
    return vs[-1] if vs else 0


def lis(xs):
    """Length of a longest strictly increasing subsequence."""
    t = []
    for x in xs:
        i = bisect_left(t, x)
        if i == len(t):
            t.append(x)
        else:
            t[i] = x
    return len(t)


def ulam(p, q):
    """Unweighted Ulam distance of two permutations of the same alphabet."""
    pos = {s: i for i, s in enumerate(p)}
    return len(p) - lis([pos[s] for s in q])


def wlis_batch(X, Wt):
    """Largest-weight increasing subsequence of every row of X (O(R k^2)).

    Column-major, so every step is a contiguous length-R vector operation."""
    R, k = X.shape
    if R == 0:
        return np.zeros(0, dtype=np.int32)
    Xt = np.ascontiguousarray(X.T)
    Wtt = np.ascontiguousarray(Wt.T, dtype=np.int32)
    best = np.zeros((k, R), dtype=np.int32)
    out = np.zeros(R, dtype=np.int32)
    cur = np.empty(R, dtype=np.int32)
    for j in range(k):
        cur.fill(0)
        xj = Xt[j]
        for i in range(j):
            np.maximum(cur, best[i] * (Xt[i] < xj), out=cur)
        np.add(cur, Wtt[j], out=best[j])
        np.maximum(out, best[j], out=out)
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int32)
    return _POP8[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def _lcs_bitpar_T(ref, Ct):
    """LCS(ref, column r of Ct) for every column; Ct holds the rows transposed (m x R).

    Bit-parallel LCS (Crochemore et al. 2001): V <- (V + (V & M_b)) | (V & ~M_b);
    the LCS is the number of zero bits of V. O(R m ceil(m/64)), in place."""
    ref = np.asarray(ref, dtype=np.int64)
    m = len(ref)
    pos = np.empty(m, dtype=np.int64)
    pos[ref] = np.arange(m)
    R = Ct.shape[1]
    if R == 0:
        return np.zeros(0, dtype=np.int32)
    ones = np.uint64(0xFFFFFFFFFFFFFFFF)
    if m <= 64:
        bits = np.left_shift(np.uint64(1), pos.astype(np.uint64))
        full = ones if m == 64 else np.uint64((1 << m) - 1)
        V = np.full(R, full, dtype=np.uint64)
        M = np.empty(R, dtype=np.uint64)
        U = np.empty(R, dtype=np.uint64)
        for j in range(Ct.shape[0]):
            np.take(bits, Ct[j], out=M)
            np.bitwise_and(V, M, out=U)
            np.add(V, U, out=U)
            np.invert(M, out=M)
            np.bitwise_and(V, M, out=V)
            np.bitwise_or(V, U, out=V)
            if m < 64:
                np.bitwise_and(V, full, out=V)
        return (m - _popcount64(V)).astype(np.int32)
    nw = (m + 63) // 64
    PM = np.zeros((nw, m), dtype=np.uint64)
    PM[pos // 64, np.arange(m)] = np.left_shift(np.uint64(1), (pos % 64).astype(np.uint64))
    fulls = np.full(nw, ones, dtype=np.uint64)
    if m % 64:
        fulls[-1] = np.uint64((1 << (m % 64)) - 1)
    V = np.repeat(fulls[:, None], R, axis=1)
    for j in range(Ct.shape[0]):
        M = PM[:, Ct[j]]
        U = V & M
        S = np.empty_like(V)
        carry = np.zeros(R, dtype=np.uint64)
        for w in range(nw):
            t = V[w] + U[w]
            c1 = t < V[w]
            s2 = t + carry
            carry = (c1 | (s2 < t)).astype(np.uint64)
            S[w] = s2
        V = (S | (V & ~M)) & fulls[:, None]
    return (m - _popcount64(V).sum(axis=0)).astype(np.int32)


def lcs_bitpar(ref, C):
    """LCS(ref, row) for every row of C (all permutations of one alphabet)."""
    if C.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    return _lcs_bitpar_T(ref, np.ascontiguousarray(C.T))


def lcs_bitpar_many(refs, Ct):
    """LCS of every row of refs against every column of Ct (len(refs) x R); m <= 64."""
    refs = np.asarray(refs, dtype=np.int64)
    B, m = refs.shape
    R = Ct.shape[1]
    pos = np.empty((B, m), dtype=np.int64)
    pos[np.arange(B)[:, None], refs] = np.arange(m)[None, :]
    bits = np.left_shift(np.uint64(1), pos.astype(np.uint64))
    full = np.uint64(0xFFFFFFFFFFFFFFFF) if m == 64 else np.uint64((1 << m) - 1)
    V = np.full((B, R), full, dtype=np.uint64)
    M = np.empty((B, R), dtype=np.uint64)
    U = np.empty((B, R), dtype=np.uint64)
    for j in range(Ct.shape[0]):
        np.take(bits, Ct[j], axis=1, out=M)
        np.bitwise_and(V, M, out=U)
        np.add(V, U, out=U)
        np.invert(M, out=M)
        np.bitwise_and(V, M, out=V)
        np.bitwise_or(V, U, out=V)
        if m < 64:
            np.bitwise_and(V, full, out=V)
    return (m - _popcount64(V)).astype(np.int32)


class Metric:
    """Weighted Ulam metric on permutations of the symbols 0..k-1."""

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=np.int32)
        self.W = int(self.w.sum())
        self.unit = bool((self.w == 1).all())
        k = len(self.w)
        # A weighted symbol stands for a contiguous block, so the weighted LCS equals
        # the plain LCS of the block-expanded rows; use bit-parallel LCS when cheaper.
        self.expand = (not self.unit) and self.W * ((self.W + 63) // 64) <= 2 * k * k
        if self.expand:
            self.start = np.concatenate([[0], np.cumsum(self.w)[:-1]]).astype(np.int64)

    def _expand(self, C):
        C = np.atleast_2d(C)
        lens = self.w[C].ravel().astype(np.int64)
        base = np.repeat(self.start[C].ravel(), lens)
        first = np.repeat(np.cumsum(lens) - lens, lens)
        return (base + np.arange(len(base)) - first).reshape(C.shape[0], self.W)

    def prepare(self, C):
        """Form of the rows C reused by many_prepared (transposed, block-expanded)."""
        if self.unit:
            return np.ascontiguousarray(C.T)
        if self.expand:
            return np.ascontiguousarray(self._expand(C).T)
        return C

    def batch_prepared(self, ref, Cp):
        if self.unit:
            return self.W - _lcs_bitpar_T(ref, Cp)
        if self.expand:
            return self.W - _lcs_bitpar_T(self._expand(np.asarray(ref))[0], Cp)
        return self.batch(ref, Cp)

    def many_prepared(self, refs, Cp):
        """Distance from each row of refs to every prepared row (len(refs) x R)."""
        refs = np.atleast_2d(refs)
        if self.unit and refs.shape[1] <= 64:
            return self.W - lcs_bitpar_many(refs, Cp)
        if self.expand and self.W <= 64:
            return self.W - lcs_bitpar_many(self._expand(refs), Cp)
        if not self.unit and not self.expand:
            k = refs.shape[1]
            R = len(Cp)
            inv = np.empty((len(refs), k), dtype=np.int64)
            inv[np.arange(len(refs))[:, None], refs] = np.arange(k)[None, :]
            X = inv[:, Cp].reshape(-1, k)
            Wt = np.broadcast_to(self.w[Cp], (len(refs), R, k)).reshape(-1, k)
            return (self.W - wlis_batch(X, Wt)).reshape(len(refs), R)
        return np.stack([self.batch_prepared(r, Cp) for r in refs])

    def batch(self, ref, C):
        """d(ref, row) for every row of the integer array C."""
        if self.unit:
            return self.W - lcs_bitpar(ref, C)
        if self.expand:
            return self.W - lcs_bitpar(self._expand(np.asarray(ref))[0], self._expand(C))
        inv = np.empty(len(ref), dtype=np.int32)
        inv[np.asarray(ref)] = np.arange(len(ref), dtype=np.int32)
        return self.W - wlis_batch(inv[C], self.w[C])
