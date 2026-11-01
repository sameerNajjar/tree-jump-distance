#!/usr/bin/env python3
"""Independent checker for result records (Python standard library only).

  python scripts/check.py results/ATGC0093 [results/ATGC0006 ...] [--tripod-limit 150]

It shares no code with the solver. For every result it re-reads the instance and
  * checks the instance (tree, leaves, permutations) and its SHA-256;
  * recomputes the cost of every recorded labeling (each algorithm, the exact
    search's YES labeling, the best labeling) with its own Ulam distance;
  * checks the contour certificate: consecutive leaves of the recorded cyclic order
    are joined by tree paths that use every edge at most twice, and half the tour
    length (rounded up) equals the recorded value;
  * checks the packing certificate: the pieces are edge-disjoint leaf paths and
    tripods, each path is valued by the Ulam distance and each tripod by an exact
    three-median recomputed here (an independent O(k^3) recurrence on the alphabet
    contracted by the blocks common to the three orders, when k <= --tripod-limit);
  * checks that lower, upper, status and certificate agree with these values.
An exact NO answer cannot be re-derived here; certificates that need one are counted.
"""
import argparse
import bisect
import collections
import hashlib
import json
import sys
from pathlib import Path


def lis_len(xs):
    t = []
    for x in xs:
        i = bisect.bisect_left(t, x)
        if i == len(t):
            t.append(x)
        else:
            t[i] = x
    return len(t)


def ulam(p, q):
    pos = {s: i for i, s in enumerate(p)}
    return len(p) - lis_len([pos[s] for s in q])


def contracted(a, b, c):
    """The three orders over the blocks common to all of them, with block weights."""
    pairs = None
    for s in (a, b, c):
        here = set(zip(s, s[1:]))
        pairs = here if pairs is None else pairs & here
    block, w, cur = {}, [], -1
    for i, s in enumerate(a):
        if i == 0 or (a[i - 1], s) not in pairs:
            cur += 1
            w.append(0)
        block[s] = cur
        w[cur] += 1

    def squeeze(s):
        out = []
        for x in s:
            if not out or out[-1] != block[x]:
                out.append(block[x])
        return out
    return squeeze(a), squeeze(b), squeeze(c), w


def median3(a, b, c, limit):
    """Exact min_t d(t,a)+d(t,b)+d(t,c), or None when the contracted alphabet exceeds limit.

    G[i][j][l]: largest sum of w(s)(|T(s)|-1) over chains of matches inside the box,
    a match of s using the inputs T(s) at their positions of s (|T(s)| >= 2)."""
    a, b, c, w = contracted(a, b, c)
    k = len(a)
    if k == 1:
        return 0
    if k > limit:
        return None
    W = sum(w)
    G = [[[0] * (k + 1) for _ in range(k + 1)] for _ in range(k + 1)]
    for i in range(k + 1):
        Gi, Gp = G[i], G[i - 1] if i else None
        ai = a[i - 1] if i else None
        for j in range(k + 1):
            bj = b[j - 1] if j else None
            row, rowp = Gi[j], Gi[j - 1] if j else None
            for l in range(k + 1):
                v = 0
                if i and Gp[j][l] > v:
                    v = Gp[j][l]
                if j and rowp[l] > v:
                    v = rowp[l]
                if l and row[l - 1] > v:
                    v = row[l - 1]
                cl = c[l - 1] if l else None
                if i and j and ai == bj:
                    x = Gp[j - 1][l] + w[ai]
                    if x > v:
                        v = x
                    if l and cl == ai:
                        x = Gp[j - 1][l - 1] + 2 * w[ai]
                        if x > v:
                            v = x
                if i and l and ai == cl:
                    x = Gp[j][l - 1] + w[ai]
                    if x > v:
                        v = x
                if j and l and bj == cl:
                    x = rowp[l - 1] + w[bj]
                    if x > v:
                        v = x
                row[l] = v
    return 2 * W - G[k][k][k]


class Checked:
    def __init__(self, inst):
        self.inst = inst
        self.n = inst["n_genes"]
        self.verts = inst["tree"]["vertices"]
        self.adj = {v: [] for v in self.verts}
        for u, v in inst["tree"]["edges"]:
            self.adj[u].append(v)
            self.adj[v].append(u)
        self.leaves = {g: tuple(p) for g, p in inst["leaves"].items()}
        self.dist_cache = {}

    def d(self, p, q):
        key = (p, q) if p <= q else (q, p)
        if key not in self.dist_cache:
            self.dist_cache[key] = ulam(p, q)
        return self.dist_cache[key]

    def path_edges(self, u, v):
        prev, stack = {u: None}, [u]
        while stack:
            x = stack.pop()
            if x == v:
                break
            for y in self.adj[x]:
                if y not in prev:
                    prev[y] = x
                    stack.append(y)
        out, x = [], v
        while prev[x] is not None:
            out.append(frozenset((x, prev[x])))
            x = prev[x]
        return out

    def labeling_cost(self, internal):
        lab = dict(self.leaves)
        full = tuple(range(1, self.n + 1))
        for v, p in internal.items():
            if v in self.leaves or v not in self.adj:
                raise ValueError(f"labels a vertex that is not internal: {v}")
            if tuple(sorted(p)) != full:
                raise ValueError(f"label of {v} is not a permutation of 1..n")
            lab[v] = tuple(p)
        if set(lab) != set(self.verts):
            raise ValueError("labeling misses vertices")
        return sum(self.d(lab[u], lab[v]) for u, v in self.inst["tree"]["edges"])


def check_instance(inst):
    n = inst["n_genes"]
    verts, edges = inst["tree"]["vertices"], inst["tree"]["edges"]
    assert len(edges) == len(verts) - 1 and len(set(verts)) == len(verts)
    deg = collections.Counter(x for e in edges for x in e)
    assert {v for v in verts if deg[v] <= 1} == set(inst["leaves"]), "leaves"
    assert not any(deg[v] == 2 for v in verts), "degree-2 vertex"
    for p in inst["leaves"].values():
        assert sorted(p) == list(range(1, n + 1)), "leaf order is not a permutation"


ROOT = Path(__file__).resolve().parents[1]


def check_result(res, inst_path, tripod_limit, med_cache, stats):
    path = Path(inst_path)
    raw = (path if path.is_absolute() else ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != res["instance_sha256"]:
        raise ValueError("instance hash differs")
    inst = json.loads(raw)
    check_instance(inst)
    C = Checked(inst)
    status = res["status"]
    if status == "ERROR":
        raise ValueError(f"solver error: {res.get('error')}")
    distinct = len(set(C.leaves.values()))
    assert res["distinct_orders"] == distinct, "distinct_orders"
    if status == "TRIVIAL":
        assert distinct == 1 and res["lower"] == res["upper"] == 0
        assert C.labeling_cost(res["best"]["labels"]) == 0
        return
    costs = {}
    for name, a in res["algorithms"].items():
        got = C.labeling_cost(a["labels"])
        if got != a["cost"]:
            raise ValueError(f"{name}: recorded {a['cost']}, recomputed {got}")
        costs[name] = got
    best = C.labeling_cost(res["best"]["labels"])
    assert best == res["upper"] == min(costs.values()), "upper bound"
    # contour
    con = res["bounds"]["contour"]
    order = con["leaf_order"]
    assert all(v in C.leaves for v in order) and len(set(order)) == len(order)
    use = collections.Counter()
    tour = 0
    for i, u in enumerate(order):
        v = order[(i + 1) % len(order)]
        use.update(C.path_edges(u, v))
        tour += C.d(C.leaves[u], C.leaves[v])
    assert max(use.values()) <= 2, "contour order uses an edge more than twice"
    assert (tour + 1) // 2 == con["value"], "contour value"
    # packing
    pk = res["bounds"]["packing"]
    used, total = set(), 0
    for piece in pk["pieces"]:
        ls = piece["leaves"]
        assert all(v in C.leaves for v in ls) and len(set(ls)) == len(ls)
        if piece["kind"] == "path":
            assert len(ls) == 2
            edges = set(C.path_edges(ls[0], ls[1]))
            value = C.d(C.leaves[ls[0]], C.leaves[ls[1]])
        else:
            assert piece["kind"] == "tripod" and len(ls) == 3
            edges = set(C.path_edges(ls[0], ls[1])) | set(C.path_edges(ls[0], ls[2])) \
                | set(C.path_edges(ls[1], ls[2]))
            key = tuple(sorted(C.leaves[x] for x in ls))
            if key not in med_cache:
                med_cache[key] = median3(*key, tripod_limit)
            value = med_cache[key]
            if value is None:
                stats["tripods_not_rechecked"] += 1
                value = piece["value"]
                lo = (sum(C.d(C.leaves[x], C.leaves[y]) for x, y in ((ls[0], ls[1]), (ls[0], ls[2]), (ls[1], ls[2]))) + 1) // 2
                assert value >= lo, "tripod value below the pair bound"
            else:
                stats["tripods_rechecked"] += 1
        assert value == piece["value"], f"piece value {piece['value']} != {value}"
        assert not (edges & used), "packing pieces share an edge"
        used |= edges
        total += value
    assert total == pk["value"], "packing value"
    lower = max(con["value"], pk["value"])
    cert = set(res["certificate"])
    calls = res["exact"]["calls"]
    for call in calls:
        if call["result"] == "YES":
            assert call["found_cost"] <= call["D"]
    for name, v in (("contour", con["value"]), ("packing", pk["value"])):
        assert (name in cert) == (v == res["upper"]), f"certificate {name}"
    exact_no = bool(calls) and calls[-1]["result"] == "NO" and calls[-1]["D"] == res["upper"] - 1
    assert ("exact NO" in cert) == exact_no
    if cert:
        assert status == "OPTIMAL" and res["lower"] == res["upper"]
        if cert == {"exact NO"}:
            stats["optimal_only_by_exact_NO"] += 1
    else:
        assert status == "OPEN" and res["lower"] == lower < res["upper"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tripod-limit", type=int, default=150)
    args = ap.parse_args()
    stats = collections.Counter()
    failures = []
    med_cache = {}
    for d in args.dirs:
        for f in sorted(Path(d).rglob("*.json")):
            res = json.loads(f.read_text(encoding="utf-8"))
            if res.get("schema") != "tree-ulam-result/1":
                continue
            stats["results"] += 1
            stats[res["status"]] += 1
            try:
                check_result(res, res["instance_file"], args.tripod_limit, med_cache, stats)
                stats["passed"] += 1
            except Exception as e:                       # report every failure
                failures.append(f"{f}: {type(e).__name__}: {e}")
    for line in failures:
        print("FAIL", line)
    print(json.dumps(dict(stats), indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
