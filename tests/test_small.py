#!/usr/bin/env python3
"""Brute-force tests on small random instances.

  python tests/test_small.py [--cases 300] [--seed 1]

For alphabets of at most 6 genes the true optimum is a tree DP over all n!
permutations. Checked on every case:
  * the three-median DP (package) and the checker's independent recurrence equal
    brute force over all permutations;
  * contour <= OPT, packing <= OPT, and both witnesses pass the independent checks;
  * lifting <= 2 OPT, 7/4 <= 1.75 OPT, local search never worsens;
  * the exact decision answers YES at OPT (with a labeling of cost <= OPT) and NO at
    OPT - 1.
"""
import argparse
import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check                                                         # noqa: E402
from treeulam import exact                                           # noqa: E402
from treeulam.approx import approx_74, contour_bound, lifting, local_search  # noqa: E402
from treeulam.blocks import Pool, Space                              # noqa: E402
from treeulam.distance import ulam                                   # noqa: E402
from treeulam.instance import Instance, make_instance               # noqa: E402
from treeulam.median import MedianOracle, median3_cost              # noqa: E402
from treeulam.packing import packing_bound                           # noqa: E402
from treeulam.solve import solve                                     # noqa: E402
from treeulam.tree import Tree, normalise                            # noqa: E402


def random_tree(rng, leaves):
    """Random unrooted tree on the given leaf names, internal degree 3 or 4."""
    t = Tree()
    a, b = t.new(), t.new()
    t.edge(a, b)
    t.name[a], t.name[b] = leaves[0], leaves[1]
    for name in leaves[2:]:
        u, v = rng.choice(t.edges())
        if rng.random() < 0.25 and t.internal():
            m = rng.choice(t.internal())          # attach to an existing internal vertex
        else:
            m = t.new()
            t.adj[u].discard(v)
            t.adj[v].discard(u)
            t.edge(u, m)
            t.edge(m, v)
        x = t.new()
        t.edge(m, x)
        t.name[x] = name
    return normalise(t)


def brute_opt(t, leaf_seq, n):
    return exact.label_tree_from(t, leaf_seq, list(itertools.permutations(range(n))))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for case in range(args.cases):
        n = rng.randint(3, 6)
        k = rng.randint(3, 8)
        base = list(range(n))
        orders = [tuple(rng.sample(base, n)) for _ in range(rng.randint(2, 4))]
        names = [f"g{i}" for i in range(k)]
        t = random_tree(rng, names)
        leaves = {g: rng.choice(orders) for g in names}
        if len(set(leaves.values())) == 1:
            continue
        rec = make_instance(f"test/{case}", "test", "subtree", [f"x{i}" for i in range(n)], t, leaves)
        inst = Instance(rec)
        t, leaf_seq = inst.tree, inst.leaf_seq
        opt = brute_opt(t, leaf_seq, n)
        # medians
        perms = list(itertools.permutations(base))
        for _ in range(3):
            a, b, c = (tuple(rng.sample(base, n)) for _ in range(3))
            want = min(ulam(x, a) + ulam(x, b) + ulam(x, c) for x in perms)
            assert median3_cost(a, b, c, [1] * n) == want, "package median"
            assert check.median3(tuple(s + 1 for s in a), tuple(s + 1 for s in b),
                                 tuple(s + 1 for s in c), 100) == want, "checker median"
        # approximations and bounds
        space = Space(list(dict.fromkeys(leaf_seq.values())))
        pool = Pool(space.w)
        med = MedianOracle(pool)
        ll = {v: pool.add(space.contract(s)) for v, s in leaf_seq.items()}
        c2, lab = lifting(t, ll, pool)
        c2ls = local_search(t, lab, ll, pool, med)[0]
        c74, lab = approx_74(t, ll, pool, med)
        c74ls = local_search(t, lab, ll, pool, med)[0]
        assert opt <= c2ls <= c2 <= 2 * opt and opt <= c74ls <= c74 and 4 * c74 <= 7 * opt, (opt, c2, c74)
        assert contour_bound(t, ll, pool.d)[0] <= opt
        assert packing_bound(t.adj, ll, pool.d, med.cost)[0] <= opt
        # exact decision
        r = exact.decide(t, leaf_seq, opt, time_limit=30)
        assert r["result"] == "YES", r
        cost, _ = exact.label_tree_from(t, leaf_seq, r["labels_used"])
        assert cost == opt
        if opt:
            assert exact.decide(t, leaf_seq, opt - 1, time_limit=30)["result"] == "NO"
        # full pipeline + independent checker
        res = solve(inst, 30)
        assert res["status"] == "OPTIMAL" and res["upper"] == opt, res["status"]
        stats = __import__("collections").Counter()
        res["instance_sha256"] = None
        check.check_instance(rec)
        check_result_inline(res, rec, stats)
        if case % 50 == 0:
            print(f"case {case}: n={n}, leaves={k}, OPT={opt} ok", flush=True)
    print("all small cases passed")


def check_result_inline(res, rec, stats):
    """check.check_result without the file and hash lookup."""
    import hashlib
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "inst.json"
        p.write_text(json.dumps(rec))
        res = dict(res, instance_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        check.check_result(res, str(p), 100, {}, stats)


if __name__ == "__main__":
    main()
