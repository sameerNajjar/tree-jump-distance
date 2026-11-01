"""Polynomial-time algorithms and the contour lower bound.

All functions work in one contracted Pool: leaf_lab maps each leaf vertex to a pool
index, and a labeling maps every vertex to a pool index.

lifting       best labeling whose internal labels are leaf labels (factor 2)
approx_74     best labeling whose internal labels are leaf labels or exact medians
              of three distinct leaf labels (factor 7/4)
local_search  replace the label of a degree-3 internal vertex by the exact median
              of its neighbours while that strictly lowers the cost
contour_bound a depth-first leaf tour crosses every edge twice, so half its length
              is a lower bound on every labeling
"""
import itertools
import random

import numpy as np


def labeling_cost(t, lab, d):
    return sum(d(lab[u], lab[v]) for u, v in t.edges())


def _minplus(Dm, f, Dmax):
    """vals[c] = min_c' Dm[c, c'] + f[c'] and the first minimising c'.

    Exact pruning: a column with f[c'] > min(f) + Dmax can never beat argmin(f), and
    every minimising column survives, so the first argmin is unchanged."""
    keep = np.flatnonzero(f <= f.min() + Dmax)
    fk = f[keep]
    vals = np.empty(len(Dm), dtype=np.int64)
    arg = np.empty(len(Dm), dtype=np.int64)
    B = max(1, 2_000_000 // max(1, len(keep)))
    for s0 in range(0, len(Dm), B):
        M = Dm[s0:s0 + B][:, keep].astype(np.int64) + fk[None, :]
        a = M.argmin(axis=1)
        arg[s0:s0 + B] = keep[a]
        vals[s0:s0 + B] = M[np.arange(len(a)), a]
    return vals, arg


def tree_dp_pool(t, leaf_lab, cand, pool):
    """Cheapest labeling with every internal label in cand (pool indices).

    Returns (cost, labeling dict vertex -> pool index)."""
    internal = t.internal()
    lab = dict(leaf_lab)
    if not internal:
        return labeling_cost(t, lab, pool.d), lab
    cand = list(dict.fromkeys(cand))
    Dm = pool.dmat(cand, cand, dtype=np.int16 if pool.W < 32767 else np.int32)
    Dmax = int(Dm.max()) if Dm.size else 0
    leafset = sorted(set(leaf_lab.values()))
    LC = pool.dmat(cand, leafset)
    leafcols = {L: LC[:, a].astype(np.int64) for a, L in enumerate(leafset)}
    root = internal[0]
    parent, order, stack = {root: None}, [], [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in t.adj[u]:
            if w not in parent:
                parent[w] = u
                stack.append(w)
    F, choice = {}, {}
    for u in reversed(order):
        if u in leaf_lab:
            continue
        f = np.zeros(len(cand), dtype=np.int64)
        for w in t.adj[u]:
            if parent.get(w) != u:
                continue
            if w in leaf_lab:
                f += leafcols[leaf_lab[w]]
            else:
                vals, arg = _minplus(Dm, F[w], Dmax)
                choice[w] = arg
                f += vals
        F[u] = f
    r = int(F[root].argmin())
    sel = {root: r}
    for u in order:
        if u in leaf_lab or u == root:
            continue
        sel[u] = int(choice[u][sel[parent[u]]])
    for u, ci in sel.items():
        lab[u] = cand[ci]
    return int(F[root][r]), lab


def contour_bound(t, leaf_lab, d, tries=64, seed=0):
    """Best of several depth-first leaf orders: ceil(cyclic tour length / 2).

    Returns (bound, leaf order attaining it). The first order is the deterministic
    DFS; the others shuffle the children with a seeded generator."""
    rng = random.Random(seed)
    best, witness = 0, []
    for k in range(tries):
        order = t.dfs_leaf_order(rng if k else None)
        if len(order) < 2:
            return 0, order
        tour = sum(d(leaf_lab[order[i]], leaf_lab[order[(i + 1) % len(order)]])
                   for i in range(len(order)))
        if not witness or (tour + 1) // 2 > best:
            best, witness = (tour + 1) // 2, order
    return best, witness


def lifting(t, leaf_lab, pool):
    return tree_dp_pool(t, leaf_lab, sorted(set(leaf_lab.values())), pool)


def approx_74(t, leaf_lab, pool, med):
    L = sorted(set(leaf_lab.values()))
    cand = set(L)
    for x, y, z in itertools.combinations(L, 3):
        cand.add(med(x, y, z))
    return tree_dp_pool(t, leaf_lab, sorted(cand), pool)


def local_search(t, lab, leaf_lab, pool, med, max_rounds=1000):
    """Vertexwise improvement by exact three-medians (degree-3 internal vertices)."""
    lab = dict(lab)
    free = [v for v in t.internal() if v not in leaf_lab and len(t.adj[v]) == 3]
    for _ in range(max_rounds):
        improved = False
        for v in free:
            nb = [lab[u] for u in t.adj[v]]
            cur = sum(pool.d(lab[v], x) for x in nb)
            m = med(*nb)
            new = sum(pool.d(m, x) for x in nb)
            if new < cur:
                lab[v] = m
                improved = True
        if not improved:
            break
    return labeling_cost(t, lab, pool.d), lab
