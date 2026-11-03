#!/usr/bin/env python3
"""Run MGR 2.01 and SCJ small parsimony on ATGC full-tree instances and score them in Ulam cost.

  python baselines/run_atgc.py --data <atgc-25 folder> --inputs inputs --records results/full \
      --mgr path/to/MGR --out results/baselines ATGC0327 ATGC0835 ...

--data holds one folder per cluster with its atgc.pty (from data.ATGCnnnn.tgz);
--inputs holds the instances made by scripts/prepare_atgc.py --full-only; --records
holds our solver's results (scripts/run_all.py), used for the comparison only.

Every method gets the same input as our solver: the cluster's own tree and the
core-gene orders of inputs/<cluster>. MGR and SCJ need signed orders, so each gene
takes its strand from atgc.pty (mgr_input.py checks that dropping the signs gives
back the instance permutation). Leaves are single linear chromosomes.

  MGR 2.01  reversal parsimony, fixed-topology mode (-t) with -L -c -H2. Build it with
            mgr_fixed_topology.patch: without it MGR renumbers merged vertices off by
            one and returns a different tree. The output tree is compared split by
            split with the cluster's tree, and a run whose tree differs is rejected.
  SCJ       exact SCJ small parsimony (Feijao and Meidanis), genomes.py. Ancestors can
            split into several pieces; they are joined by genomes.linear_order.

Scoring. Each baseline ancestor is dropped to an unsigned order and then given the
rotation and direction closest to the reference genome, the rule the leaves were
normalised with (treeulam.atgc.linearise); this can only lower the baselines' cost.
The Ulam cost is summed over the edges of the cluster's tree.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import genomes as R                                 # noqa: E402
from mgr_input import signed_orders, topology       # noqa: E402
from treeulam.atgc import linearise                 # noqa: E402
from treeulam.distance import ulam                  # noqa: E402


def normalise(order):
    """Unsigned order (genes 1..n) -> rotation and direction closest to the identity."""
    return tuple(s + 1 for s in linearise([abs(g) - 1 for g in order])[3])


def splits(edges, leaves):
    """Leaf bipartitions of an unrooted tree, each as the side without the first leaf."""
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    first = min(leaves)
    out = set()
    for u, v in edges:
        seen, stack = {u, v}, [v]
        side = set()
        while stack:
            x = stack.pop()
            if x in leaves:
                side.add(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if first in side:
            side = set(leaves) - side
        if 1 < len(side) < len(leaves) - 1:
            out.add(frozenset(side))
    return out


def parse_newick_edges(text):
    """Edges of a Newick tree with named internal vertices (branch lengths dropped)."""
    text = re.sub(r":[0-9.eE+-]+", "", text.strip().rstrip(";"))
    edges, pos, fresh = [], 0, [0]

    def node():
        nonlocal pos
        kids = []
        if text[pos] == "(":
            pos += 1
            kids.append(node())
            while text[pos] == ",":
                pos += 1
                kids.append(node())
            pos += 1
        m = re.compile(r"[^,();]*").match(text, pos)
        pos = m.end()
        name = m.group().strip()
        if not name:
            fresh[0] += 1
            name = f"_u{fresh[0]}"
        edges.extend((name, k) for k in kids)
        return name

    node()
    return edges


def parse_mgr(text):
    """MGR's output: the Newick line of the recovered tree and every genome (name -> signed genes)."""
    newick = next(line for line in text.splitlines() if line.startswith("(") and line.rstrip().endswith(";"))
    found, name = {}, None
    for line in text.splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            found[name] = []
        elif name is not None:
            toks = line.replace("$", " ").split()
            if toks and all(re.fullmatch(r"-?\d+", t) for t in toks):
                found[name] += [int(t) for t in toks]
            elif not line.strip():
                name = None
    score = re.search(r"Score of tree recovered:\s*(\d+)", text)
    return newick, found, int(score.group(1)) if score else None


def run_mgr(mgr, dataset, orders, tree_edges, leaves, out, seconds):
    number = {g: i + 1 for i, g in enumerate(dataset["genomes"])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "genomes.txt").write_text(
        "".join(f">{g}\n{' '.join(map(str, orders[g]))} $\n\n" for g in dataset["genomes"]), encoding="ascii")
    (out / "topology.txt").write_text(topology(dataset["tree"], number) + "\n", encoding="ascii")
    began = time.time()
    try:
        p = subprocess.run([str(mgr), "-f", "genomes.txt", "-t", "topology.txt", "-L", "-c", "-H2"],
                           cwd=out, capture_output=True, text=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        return dict(status="TIMEOUT", seconds=seconds)
    secs = round(time.time() - began, 2)
    (out / "mgr_stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        return dict(status="FAILED", seconds=secs, exit=p.returncode, error=(p.stderr or p.stdout)[-300:])
    newick, found, reported = parse_mgr(p.stdout)
    mgr_edges = parse_newick_edges(newick)
    if splits(mgr_edges, leaves) != splits(tree_edges, leaves):
        return dict(status="WRONG TREE", seconds=secs, newick=newick)
    for g in leaves:
        if found.get(g) != orders[g]:
            return dict(status="LEAF CHANGED", seconds=secs, genome=g)
    n = dataset["n_genes"]
    A = {v: R.adjacencies([(False, g)]) for v, g in found.items()}
    lin = {v: (tuple(dataset["permutations"][v]) if v in leaves else normalise(found[v])) for v in found}
    return dict(status="OK", seconds=secs, reversals_reported=reported,
                dcj=sum(R.dcj_distance(A[u], A[v], n) for u, v in mgr_edges),
                ulam=sum(ulam(lin[u], lin[v]) for u, v in mgr_edges),
                ancestors={v: list(o) for v, o in lin.items() if v not in leaves})


def run_scj(dataset, orders, tree_edges, leaves):
    adj = collections.defaultdict(list)
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    root = next(v for v in adj if v not in leaves)
    parent, order, stack = {root: None}, [], [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in adj[u]:
            if w not in parent:
                parent[w] = u
                stack.append(w)
    edges = [(parent[v], v) for v in order if parent[v] is not None]
    n = dataset["n_genes"]
    began = time.time()
    leaf_adj = {g: R.adjacencies([(False, orders[g])]) for g in leaves}
    anc = R.scj_small_parsimony(edges, root, leaf_adj)
    found = {v: R.genome_from_adjacencies(A, n) for v, A in anc.items()}
    secs = round(time.time() - began, 2)
    lin = {g: tuple(dataset["permutations"][g]) for g in leaves}
    lin.update({v: normalise(R.linear_order(G, n)) for v, G in found.items()})
    allA = dict(leaf_adj, **anc)
    return dict(status="OK", seconds=secs,
                scj=sum(R.scj_distance(allA[p], allA[c]) for p, c in edges),
                ulam=sum(ulam(lin[p], lin[c]) for p, c in edges),
                pieces={v: len(G) for v, G in found.items()},
                fragmented_ancestors=sum(1 for G in found.values() if len(G) > 1),
                ancestors={v: list(lin[v]) for v in found})


def ours(records, cluster):
    path = Path(records) / cluster / "F.json"
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    algs = rec.get("algorithms", {})
    return dict(status=rec["status"], lower=rec.get("lower"), upper=rec.get("upper"),
                **{m: algs[m]["cost"] for m in ("lifting", "lifting+ls", "7/4", "7/4+ls", "exact") if m in algs})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clusters", nargs="+")
    ap.add_argument("--data", required=True, help="folder with one atgc-25 cluster folder each")
    ap.add_argument("--inputs", required=True, help="instances from prepare_atgc.py --full-only")
    ap.add_argument("--records", required=True, help="our solver's results")
    ap.add_argument("--mgr", required=True, help="MGR 2.01 binary built with mgr_fixed_topology.patch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mgr-seconds", type=float, default=3600)
    args = ap.parse_args()
    out = Path(args.out)
    for cluster in args.clusters:
        dataset = json.loads((Path(args.inputs) / cluster / "dataset.json").read_text(encoding="utf-8"))
        orders = signed_orders(Path(args.data) / cluster / "atgc.pty", dataset)
        tree_edges = [tuple(e) for e in dataset["tree"]["edges"]]
        leaves = set(dataset["genomes"])
        rec = dict(cluster=cluster, genomes=len(leaves), genes=dataset["n_genes"],
                   distinct_orders=dataset["distinct_orders"],
                   mgr=run_mgr(Path(args.mgr).resolve(), dataset, orders, tree_edges, leaves,
                               out / cluster / "mgr", args.mgr_seconds),
                   scj=run_scj(dataset, orders, tree_edges, leaves),
                   ours=ours(args.records, cluster))
        (out / cluster).mkdir(parents=True, exist_ok=True)
        (out / cluster / "baselines.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        o = rec["ours"] or {}
        print(f"{cluster}: ours {o.get('upper')} (lower {o.get('lower')}, {o.get('status')}), "
              f"MGR {rec['mgr']['status']} {rec['mgr'].get('ulam')} ({rec['mgr'].get('reversals_reported')} rev), "
              f"SCJ {rec['scj']['ulam']} ({rec['scj']['fragmented_ancestors']} fragmented)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
