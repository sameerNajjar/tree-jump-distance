"""Exact weighted median of three permutations under the Ulam distance.

min over tau of d(tau, a) + d(tau, b) + d(tau, c) equals 2W minus the largest value
of sum_s w(s) (|T(s)| - 1), where T(s) is the set of inputs whose chosen common
subsequence with tau contains s. The three-dimensional dynamic program below
computes that value and one optimal tau (cf. Chakraborty, Das and Krauthgamer, 2021).
"""
import time

import numpy as np

from .blocks import Space
from .distance import wlis


def _median3_dp(a, b, c, w):
    """Maximise sum_i wLCS(tau, input_i) over permutations tau.

    Some optimum puts every symbol in at least one LCS witness, so the value is
    W + max sum_{s in A} w(s)(|T(s)|-1) over symbols A used by >= 2 witnesses,
    T(s) being the witness set of s. Scanning tau left to right, the state is
    the last used position in each input; a symbol with |T| >= 2 advances the
    coordinates in T. G[p1,p2,p3] is the best value inside that box.
    Returns (tau as a list over 0..k-1, best value).
    """
    k = len(a)
    pos2 = [0] * k
    pos3 = [0] * k
    for i, s in enumerate(b):
        pos2[s] = i + 1
    for i, s in enumerate(c):
        pos3[s] = i + 1
    dtype = np.int32 if (k + 1) ** 3 * 4 <= 3e8 else np.int16
    if dtype is np.int16 and 2 * sum(w) > np.iinfo(np.int16).max:
        raise OverflowError("three-median values exceed the int16 table used for large alphabets")
    G = np.zeros((k + 1, k + 1, k + 1), dtype=dtype)
    r3 = np.array([pos3[b[p2]] for p2 in range(k)], dtype=np.int64)       # item (2,3) at row p2+1
    w23 = [w[b[p2]] for p2 in range(k)]
    rows1 = np.arange(1, k + 1)
    r3l = r3.tolist()
    for p1 in range(0, k + 1):
        if p1 == 0:                       # plane 0: only symbols with T = {2,3}
            B = np.zeros((k + 1, k + 1), dtype=dtype)
            C = B
        else:
            prev = G[p1 - 1]
            s = a[p1 - 1]
            ws = w[s]
            q2, q3 = pos2[s], pos3[s]
            B = prev.copy()
            np.maximum(B[q2, :], prev[q2 - 1, :] + ws, out=B[q2, :])      # T = {1,2}
            B[:, q3] = np.maximum(B[:, q3], prev[:, q3 - 1] + ws)          # T = {1,3}
            t = prev[q2 - 1, q3 - 1] + 2 * ws                              # T = {1,2,3}
            if t > B[q2, q3]:
                B[q2, q3] = t
            C = np.maximum.accumulate(np.maximum.accumulate(B, axis=0), axis=1)
        base = C[rows1 - 1, r3 - 1].tolist()
        fen = [0] * (k + 1)
        vals = [0] * k
        for p2 in range(k):                                                # T = {2,3}
            rr = r3l[p2]
            q, i = 0, rr - 1
            while i > 0:
                if fen[i] > q:
                    q = fen[i]
                i -= i & -i
            v = (base[p2] if base[p2] > q else q) + w23[p2]
            vals[p2] = v
            i = rr
            while i <= k:
                if fen[i] < v:
                    fen[i] = v
                i += i & -i
        B[rows1, r3] = np.maximum(B[rows1, r3], np.array(vals, dtype=B.dtype))
        G[p1] = np.maximum.accumulate(np.maximum.accumulate(B, axis=0), axis=1)
    best = int(G[k, k, k])
    # backtrack
    items = []
    p1 = p2 = p3 = k
    while G[p1, p2, p3] > 0:
        v = int(G[p1, p2, p3])
        if p1 and G[p1 - 1, p2, p3] == v:
            p1 -= 1
            continue
        if p2 and G[p1, p2 - 1, p3] == v:
            p2 -= 1
            continue
        if p3 and G[p1, p2, p3 - 1] == v:
            p3 -= 1
            continue
        s = a[p1 - 1] if p1 else None
        if s is not None and p2 and p3 and pos2[s] == p2 and pos3[s] == p3 \
                and v == 2 * w[s] + int(G[p1 - 1, p2 - 1, p3 - 1]):
            items.append((s, (1, 2, 3)))
            p1, p2, p3 = p1 - 1, p2 - 1, p3 - 1
        elif s is not None and p2 and pos2[s] == p2 and v == w[s] + int(G[p1 - 1, p2 - 1, p3]):
            items.append((s, (1, 2)))
            p1, p2 = p1 - 1, p2 - 1
        elif s is not None and p3 and pos3[s] == p3 and v == w[s] + int(G[p1 - 1, p2, p3 - 1]):
            items.append((s, (1, 3)))
            p1, p3 = p1 - 1, p3 - 1
        else:
            t2 = b[p2 - 1]
            assert pos3[t2] == p3 and v == w[t2] + int(G[p1, p2 - 1, p3 - 1]), "median backtrack"
            items.append((t2, (2, 3)))
            p2, p3 = p2 - 1, p3 - 1
    items.reverse()
    del G
    tau = [s for s, _ in items]
    in1 = {s for s, T in items if 1 in T}
    used = set(tau)
    # Insert every remaining symbol right after its predecessor (in input a) among
    # the witness-1 symbols, so witness 1 grows by every inserted symbol.
    last = None
    for s in a:
        if s in in1:
            last = s
        elif s not in used:
            if last is None:
                tau.insert(0, s)
            else:
                tau.insert(tau.index(last) + 1, s)
            used.add(s)
            last = s
    return tau, best


def _median3_value(a, b, c, w):
    """Value-only version of _median3_dp (two planes of memory, no backtrack)."""
    k = len(a)
    pos2 = [0] * k
    pos3 = [0] * k
    for i, s in enumerate(b):
        pos2[s] = i + 1
    for i, s in enumerate(c):
        pos3[s] = i + 1
    r3 = np.array([pos3[b[p2]] for p2 in range(k)], dtype=np.int64)
    w23 = [w[b[p2]] for p2 in range(k)]
    rows1 = np.arange(1, k + 1)
    r3l = r3.tolist()
    prev = None
    for p1 in range(0, k + 1):
        if p1 == 0:
            B = np.zeros((k + 1, k + 1), dtype=np.int64)
            C = B
        else:
            s = a[p1 - 1]
            ws = w[s]
            q2, q3 = pos2[s], pos3[s]
            B = prev.copy()
            np.maximum(B[q2, :], prev[q2 - 1, :] + ws, out=B[q2, :])
            B[:, q3] = np.maximum(B[:, q3], prev[:, q3 - 1] + ws)
            t = prev[q2 - 1, q3 - 1] + 2 * ws
            if t > B[q2, q3]:
                B[q2, q3] = t
            C = np.maximum.accumulate(np.maximum.accumulate(B, axis=0), axis=1)
        base = C[rows1 - 1, r3 - 1].tolist()
        fen = [0] * (k + 1)
        vals = [0] * k
        for p2 in range(k):
            rr = r3l[p2]
            q, i = 0, rr - 1
            while i > 0:
                if fen[i] > q:
                    q = fen[i]
                i -= i & -i
            v = (base[p2] if base[p2] > q else q) + w23[p2]
            vals[p2] = v
            i = rr
            while i <= k:
                if fen[i] < v:
                    fen[i] = v
                i += i & -i
        B[rows1, r3] = np.maximum(B[rows1, r3], np.array(vals, dtype=B.dtype))
        prev = np.maximum.accumulate(np.maximum.accumulate(B, axis=0), axis=1)
    return int(prev[k, k])


def median3_cost(A, B, C, w):
    """Exact min_tau d(tau,A) + d(tau,B) + d(tau,C) (weighted Ulam), value only."""
    sp = Space([list(A), list(B), list(C)])
    if sp.k == 1:
        return 0
    a, b, c = sp.contract(A), sp.contract(B), sp.contract(C)
    ww = [sum(w[s] for s in blk) for blk in sp.blocks]
    return 2 * sum(ww) - _median3_value(list(a), list(b), list(c), ww)


def median3_seq(A, B, C, w):
    """Exact weighted median of three permutations A, B, C of 0..k-1 (tuples)."""
    sp = Space([A, B, C])                      # contract the blocks common to the triple
    a, b, c = sp.contract(A), sp.contract(B), sp.contract(C)
    if sp.k == 1:
        return tuple(A)
    ww = [sum(w[s] for s in blk) for blk in sp.blocks]
    tau, best = _median3_dp(list(a), list(b), list(c), ww)
    Wt = sum(ww)
    pos = {s: i for i, s in enumerate(tau)}
    got = sum(Wt - wlis([pos[s] for s in x], [ww[s] for s in x]) for x in (a, b, c))
    assert got == 2 * Wt - best, "3-median reconstruction does not attain the DP value"
    return sp.expand(tau)


class MedianOracle:
    """Cached exact three-medians of pool labels, with metric shortcuts."""

    def __init__(self, pool):
        self.pool = pool
        self.cache = {}
        self.calls = 0
        self.dp_calls = 0
        self.time = 0.0

    def __call__(self, i, j, k):
        key = tuple(sorted((i, j, k)))
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        t0 = time.time()
        P = self.pool
        x, y, z = key
        res = None
        if x == y or x == z:
            res = x
        elif y == z:
            res = y
        else:
            # An input lying between the other two is an optimal median.
            dxy, dyz, dxz = P.d(x, y), P.d(y, z), P.d(x, z)
            if dxy + dyz == dxz:
                res = y
            elif dxy + dxz == dyz:
                res = x
            elif dxz + dyz == dxy:
                res = z
        if res is None:
            self.dp_calls += 1
            tau = median3_seq(P.items[x], P.items[y], P.items[z], P.w)
            res = P.add(tau)
            got = P.d(res, x) + P.d(res, y) + P.d(res, z)
            for cand in key:                   # every input is a feasible centre
                if sum(P.d(cand, t) for t in key) < got:
                    raise AssertionError("3-median DP returned a non-optimal centre")
        self.cache[key] = res
        self.time += time.time() - t0
        return res

    def cost(self, i, j, k):
        m = self(i, j, k)
        return self.pool.d(m, i) + self.pool.d(m, j) + self.pool.d(m, k)
