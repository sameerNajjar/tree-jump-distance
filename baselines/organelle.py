#!/usr/bin/env python3
"""Ulam cost of our methods and of MGR, SCJ and saved SPP-DCJ ancestors on the organelle datasets.

  python baselines/organelle.py meta28 --data <benchmark folder> --write-mgr-input meta28_mgr_input.txt
  MGR -f meta28_mgr_input.txt -C -c -H2 > meta28_mgr.out
  python baselines/organelle.py meta28 --data <benchmark folder> --mgr-out meta28_mgr.out \
      [--saved sppdcj.json ...] [--exact-seconds 60]

--write-mgr-input writes the dataset in MGR's format and exits. Campanulaceae and
metazoa11 are MGR files already, so MGR reads them as they are.

Datasets (not redistributed; see baselines/README.md): Campanulaceae chloroplasts
(Cosner et al. 2000) and metazoan mitochondria (Blanchette, Kunisawa and Sankoff
1999), both from the MGR 2.01 sample data, and meta28 / meta87 from Larget et al.
(2005), supplementary files 1 and 3.

The tree is the one MGR 2.01 reports with its ancestors (run as MGR -C -c -H2 on the
dataset), read from --mgr-out. Leaves are circular genomes written from gene 1 on its
forward strand; dropping the signs gives the unsigned linear orders our solver uses.

  ours      treeulam.solve on that tree: lifting and 7/4, each with local search, the
            bounds, and the exact decision for --exact-seconds.
  MGR       the ancestors in --mgr-out.
  SCJ       exact SCJ small parsimony on the same tree (genomes.py).
  --saved   any JSON with "signed_genomes" (vertex -> list of [is_circular, genes]),
            for example the SPP-DCJ results of the October benchmark.

Every baseline ancestor is turned into one unsigned order by genomes.linear_order:
the chromosome holding gene 1 first, from +1, then the others by smallest gene.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import genomes as R                                       # noqa: E402
from treeulam.instance import Instance, make_instance     # noqa: E402
from treeulam.solve import solve                          # noqa: E402
from treeulam.tree import Tree, normalise                 # noqa: E402

DECIDE = dict(max_cands=8_000_000, candidate_memory_mib=1024, probe_seconds=0.05, subtree_feedback=True)
FILES = {"campanulaceae": ("mgr", "campanulaceae_cosner2000_13x105.txt"),
         "metazoa11": ("mgr", "metazoa_blanchette1999_11x36.txt"),
         "meta28": ("larget", "meta28_larget2005_28x15.txt", "m"),
         "meta87": ("larget", "meta87_larget2005_87x37.txt", "g")}


def load(name, data):
    """Signed genomes (lists of signed ints 1..n), keyed as in the MGR runs."""
    kind, fn, *label = FILES[name]
    G = {}
    if kind == "mgr":
        cur = None
        for line in open(Path(data) / fn, encoding="latin-1"):
            line = line.split("#")[0].strip()
            if line.startswith(">"):
                cur = line[1:].strip()
                G[cur] = []
            elif line and cur:
                G[cur] += [int(x) for x in line.replace("$", " ").split()]
    else:
        rows = [p for p in (line.strip().split(",") for line in open(Path(data) / fn, encoding="latin-1"))
                if len(p) > 1]
        gid = {g.lstrip("-"): i + 1 for i, g in enumerate(rows[0][1:])}
        for i, (_, *gs) in enumerate(rows):
            G[f"{label[0]}{i + 1:02d}"] = [(-1 if g.startswith("-") else 1) * gid[g.lstrip("-")] for g in gs]
    return G


def mgr_output(path):
    """Newick tree and ancestors (name -> signed genes) from an MGR output file."""
    txt = Path(path).read_text(encoding="latin-1")
    nwk = txt.split("Tree recovered (Newick Standard):")[1].strip().splitlines()[0]
    found, name = {}, None
    for line in txt.split(nwk, 1)[1].splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].strip()
        elif name and re.fullmatch(r"[-0-9 ]+", line or "x"):
            found[name] = [int(x) for x in line.split()]
            name = None
    return nwk, found


def parse_newick(text):
    """(root, edges as (parent, child), leaves), keeping MGR's internal names."""
    s = re.sub(r"\[[^\]]*\]", "", text.strip()).rstrip(";").strip()
    pos, edges, leaves, fresh = 0, [], [], [0]

    def node():
        nonlocal pos
        kids = []
        if s[pos] == "(":
            pos += 1
            while True:
                kids.append(node())
                pos += 1
                if s[pos - 1] == ")":
                    break
        j = pos
        while pos < len(s) and s[pos] not in ",()":
            pos += 1
        name = s[j:pos].split(":")[0].strip()
        if not name:
            fresh[0] += 1
            name = f"_N{fresh[0]}"
        if not kids:
            leaves.append(name)
        edges.extend((name, k) for k in kids)
        return name

    return node(), edges, leaves


def ulam_cost(edges, lin):
    return sum(R.ulam(lin[p], lin[c]) for p, c in edges)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=list(FILES))
    ap.add_argument("--data", required=True, help="folder with the four benchmark files")
    ap.add_argument("--mgr-out", help="MGR 2.01 output with tree and ancestors")
    ap.add_argument("--write-mgr-input", help="write the dataset as MGR input to this file and exit")
    ap.add_argument("--saved", nargs="*", default=[], help="saved results with signed_genomes")
    ap.add_argument("--exact-seconds", type=float, default=60)
    ap.add_argument("--out")
    args = ap.parse_args()
    G = load(args.dataset, args.data)
    n = len(next(iter(G.values())))
    if args.write_mgr_input:
        Path(args.write_mgr_input).write_text(
            "".join(f">{v}\n{' '.join(map(str, g))} $\n\n" for v, g in G.items()), encoding="ascii", newline="\n")
        return 0
    if not args.mgr_out:
        ap.error("--mgr-out is required unless --write-mgr-input is given")
    nwk, mgr_anc = mgr_output(args.mgr_out)
    root, edges, leaves = parse_newick(nwk)
    if sorted(leaves) != sorted(G):
        sys.exit("the MGR tree's leaves differ from the dataset's genomes")
    lin = {v: R.linear_order(R.circular(G[v]), n) for v in leaves}
    out = dict(dataset=args.dataset, genomes=len(leaves), genes=n)

    t, vid = Tree(), {}
    for u, v in edges:
        for x in (u, v):
            vid.setdefault(x, t.new())
        t.edge(vid[u], vid[v])
    for v in leaves:
        t.name[vid[v]] = v
    rec = solve(Instance(make_instance(f"organelle/{args.dataset}", args.dataset, "user",
                                       [f"g{i + 1}" for i in range(n)], normalise(t),
                                       {v: [x - 1 for x in lin[v]] for v in leaves})),
                args.exact_seconds, decide_options=DECIDE)
    out["ours"] = dict(status=rec["status"], lower=rec["lower"], upper=rec["upper"],
                       **{m: a["cost"] for m, a in rec["algorithms"].items()})

    anc = {v: R.linear_order(R.circular(g), n) for v, g in mgr_anc.items()}
    out["mgr"] = dict(ulam=ulam_cost(edges, dict(lin, **anc)))
    leaf_adj = {v: R.adjacencies(R.circular(G[v])) for v in leaves}
    scj = R.scj_small_parsimony(edges, root, leaf_adj)
    found = {v: R.genome_from_adjacencies(A, n) for v, A in scj.items()}
    out["scj"] = dict(ulam=ulam_cost(edges, dict(lin, **{v: R.linear_order(g, n) for v, g in found.items()})),
                      fragmented_ancestors=sum(1 for g in found.values() if len(g) > 1))
    for path in args.saved:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        sg = {v: g for v, g in saved["signed_genomes"].items() if v not in lin}
        out[saved.get("method", Path(path).stem)] = dict(
            ulam=ulam_cost(edges, dict(lin, **{v: R.linear_order(g, n) for v, g in sg.items()})))
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
