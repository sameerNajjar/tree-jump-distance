"""Lower bound from edge-disjoint leaf paths and leaf tripods, with a witness.

Claim. Choose edge-disjoint pieces of the tree, each either the path between two
leaves (labels a, b) or a tripod: the subtree spanned by three leaves (labels a, b, c)
with one branch vertex. In every labeling the edges of a path cost at least d(a, b)
and the edges of a tripod cost at least med(a, b, c) = min_t d(t,a)+d(t,b)+d(t,c)
(triangle inequality along each arm). The pieces share no edge, so the sum of their
values is at most the cost of every labeling, hence at most OPT.

The dynamic program finds a packing of largest value. Root the tree at an internal
vertex; for a vertex x and the edges below it:
  C[x]        best packing with nothing left open;
  P1[x][a]    best packing of the other edges while one open path runs from x down
              to a leaf labelled a;
  P2[x][a,b]  best packing while an open tripod has two arms (to leaves a and b)
              below x and its third arm still has to continue upward from x.
A piece is valued when it is closed. The traceback returns the pieces as leaf
vertices, so the bound can be checked without trusting the DP.
"""
import itertools

import numpy as np

NEG = -(1 << 60)          # "no such open piece"; any sum involving it stays below NEG // 2


def packing_bound(adj, leaf_label, dist, median_cost, root=None):
    """adj: vertex -> neighbours; leaf_label: leaf vertex -> hashable label;
    dist(a, b) and median_cost(a, b, c) on labels.

    Returns (value, pieces); each piece is ('path', (u, v), d) or ('tripod', (u, v, w), m)
    with leaf vertices u, v, w and the piece value."""
    labels = sorted(set(leaf_label.values()))
    q = len(labels)
    if q <= 1:
        return 0, []
    li = {a: i for i, a in enumerate(labels)}
    Dm = np.array([[dist(a, b) for b in labels] for a in labels], dtype=np.int64)
    M = np.zeros((q, q, q), dtype=np.int64)
    for i, j, k in itertools.combinations_with_replacement(range(q), 3):
        v = median_cost(labels[i], labels[j], labels[k])
        for p in set(itertools.permutations((i, j, k))):
            M[p] = v
    W = int(Dm.max()) * max(1, sum(len(n) for n in adj.values()))
    assert 4 * W < -NEG // 4, "packing values too large for the sentinel"
    if root is None:
        internal = [v for v in adj if v not in leaf_label]
        if not internal:                    # a single edge between two leaves
            u, v = sorted(leaf_label)
            val = int(Dm[li[leaf_label[u]], li[leaf_label[v]]])
            return val, [('path', (u, v), val)]
        root = internal[0]
    order, parent, stack = [], {root: None}, [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in adj[u]:
            if w not in parent:
                parent[w] = u
                stack.append(w)
    kids_of = {x: [z for z in adj[x] if parent.get(z) == x] for x in order}
    C, P1, P2 = {}, {}, {}
    for x in reversed(order):
        kids = kids_of[x]
        if x in leaf_label:
            C[x] = 0
            p = np.full(q, NEG, dtype=np.int64)
            p[li[leaf_label[x]]] = 0
            P1[x], P2[x] = p, np.full((q, q), NEG, dtype=np.int64)
            continue
        S = sum(C[z] for z in kids)
        best = S
        p1 = np.full(q, NEG, dtype=np.int64)
        p2 = np.full((q, q), NEG, dtype=np.int64)
        for z in kids:
            base = S - C[z]
            p1 = np.maximum(p1, P1[z] + base)              # open path continues upward
            p2 = np.maximum(p2, P2[z] + base)              # open tripod's third arm continues upward
        for z, y in itertools.permutations(kids, 2):
            base = S - C[z] - C[y]
            A, B = P1[z], P1[y]
            best = max(best, int((A[:, None] + B[None, :] + Dm).max()) + base)   # close a path at x
            p2 = np.maximum(p2, A[:, None] + B[None, :] + base)                # start a tripod at x
            T2 = P2[z]
            if T2.max() > NEG // 2:                        # close a tripod: third arm down through y
                best = max(best, int((T2[:, :, None] + B[None, None, :] + M).max()) + base)
        for z, y, w in itertools.combinations(kids, 3):    # tripod centred at x
            base = S - C[z] - C[y] - C[w]
            val = (P1[z][:, None, None] + P1[y][None, :, None] + P1[w][None, None, :] + M).max()
            best = max(best, int(val) + base)
        C[x] = best
        p1[p1 < NEG // 2] = NEG
        p2[p2 < NEG // 2] = NEG
        P1[x], P2[x] = p1, p2

    # Traceback: find, for every stored value, one transition that attains it.
    def closed(x):
        kids = kids_of[x]
        if x in leaf_label:
            return []
        target, S = C[x], sum(C[z] for z in kids)

        def rest(*used):
            return [p for z in kids if z not in used for p in closed(z)]
        if target == S:
            return rest()
        for z, y in itertools.permutations(kids, 2):
            base = S - C[z] - C[y]
            for a in range(q):
                if P1[z][a] <= NEG // 2:
                    continue
                for b in range(q):
                    if P1[y][b] > NEG // 2 and P1[z][a] + P1[y][b] + Dm[a, b] + base == target:
                        pz, u = open1(z, a)
                        py, v = open1(y, b)
                        return pz + py + rest(z, y) + [('path', (u, v), int(Dm[a, b]))]
                    for c in range(q):
                        if P2[z][a, c] > NEG // 2 and P1[y][b] > NEG // 2 and \
                                P2[z][a, c] + P1[y][b] + M[a, c, b] + base == target:
                            pz, u, w = open2(z, a, c)
                            py, v = open1(y, b)
                            return pz + py + rest(z, y) + [('tripod', (u, w, v), int(M[a, c, b]))]
        for z, y, w in itertools.combinations(kids, 3):
            base = S - C[z] - C[y] - C[w]
            for a, b, c in itertools.product(range(q), repeat=3):
                if min(P1[z][a], P1[y][b], P1[w][c]) > NEG // 2 and \
                        P1[z][a] + P1[y][b] + P1[w][c] + M[a, b, c] + base == target:
                    pz, u = open1(z, a)
                    py, v = open1(y, b)
                    pw, s = open1(w, c)
                    return pz + py + pw + rest(z, y, w) + [('tripod', (u, v, s), int(M[a, b, c]))]
        raise AssertionError("packing traceback failed")

    def open1(x, a):
        if x in leaf_label:
            return [], x
        kids, target = kids_of[x], P1[x][a]
        S = sum(C[z] for z in kids)
        for z in kids:
            if P1[z][a] > NEG // 2 and P1[z][a] + S - C[z] == target:
                pz, u = open1(z, a)
                return pz + [p for o in kids if o != z for p in closed(o)], u
        raise AssertionError("packing traceback failed (open path)")

    def open2(x, a, b):
        kids, target = kids_of[x], P2[x][a, b]
        S = sum(C[z] for z in kids)
        for z in kids:
            if P2[z][a, b] > NEG // 2 and P2[z][a, b] + S - C[z] == target:
                pz, u, v = open2(z, a, b)
                return pz + [p for o in kids if o != z for p in closed(o)], u, v
        for z, y in itertools.permutations(kids, 2):
            if P1[z][a] > NEG // 2 and P1[y][b] > NEG // 2 and \
                    P1[z][a] + P1[y][b] + S - C[z] - C[y] == target:
                pz, u = open1(z, a)
                py, v = open1(y, b)
                return pz + py + [p for o in kids if o not in (z, y) for p in closed(o)], u, v
        raise AssertionError("packing traceback failed (open tripod)")

    value = int(C[root])
    pieces = closed(root)
    assert sum(p[2] for p in pieces) == value, "packing witness does not attain the DP value"
    return value, pieces
