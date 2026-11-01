#!/usr/bin/env python3
"""Make an instance file from your own tree and gene orders.

  python scripts/make_instance.py examples/example.tre examples/example.txt my_instance.json

Tree: a Newick file. Leaf names must equal the genome names in the gene-order file
(quotes around names are ignored). Branch lengths and internal labels are ignored;
degree-2 vertices are suppressed. Tree leaves without a gene order are dropped.

Gene orders: GRIMM-style text. A line ">name" starts a genome; the following lines
list its genes, separated by spaces or commas. The chromosome markers "$" and "@"
are allowed and dropped, so several chromosomes are joined in file order. "#"
starts a comment. Gene names are arbitrary tokens. The model is unsigned: a
leading "+" or "-" on a gene is removed (the count is reported).

Every genome must contain the same genes, each exactly once. With --core, genes
that are missing from some genome or repeated in one are dropped instead.

The first genome in the file (or --reference) defines the gene numbering 1..n.
With --linearise, each genome is first rotated and possibly reversed to the order
closest to the reference in Ulam distance, as for circular chromosomes.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeulam.atgc import linearise                            # noqa: E402
from treeulam.distance import ulam                             # noqa: E402
from treeulam.instance import make_instance                    # noqa: E402
from treeulam.tree import induced, parse_newick                # noqa: E402


def read_orders(path):
    orders, name, signs = {}, None, 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].strip()
            if not name or name in orders:
                sys.exit(f"empty or repeated genome name: {line!r}")
            orders[name] = []
            continue
        if name is None:
            sys.exit("the gene-order file must start with a '>name' line")
        for tok in line.replace(",", " ").split():
            if tok in ("$", "@"):
                continue
            if tok[0] in "+-" and len(tok) > 1:
                tok = tok[1:]
                signs += 1
            orders[name].append(tok)
    return orders, signs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", help="Newick file")
    ap.add_argument("orders", help="gene-order file (GRIMM-style)")
    ap.add_argument("out", help="instance file to write (.json)")
    ap.add_argument("--core", action="store_true", help="keep only genes present exactly once in every genome")
    ap.add_argument("--linearise", action="store_true", help="rotate/reverse each genome towards the reference")
    ap.add_argument("--reference", help="genome that defines the numbering (default: first in the file)")
    ap.add_argument("--dataset", default="user", help="dataset name used in the instance id")
    args = ap.parse_args()

    orders, signs = read_orders(args.orders)
    if len(orders) < 2:
        sys.exit("need at least two genomes")
    if signs:
        print(f"note: removed {signs} gene signs (the model is unsigned)")
    counts = {g: collections.Counter(seq) for g, seq in orders.items()}
    if args.core:
        keep = set.intersection(*({x for x, c in cnt.items() if c == 1} for cnt in counts.values()))
        orders = {g: [x for x in seq if x in keep] for g, seq in orders.items()}
        print(f"--core: {len(keep)} genes kept")
    else:
        genes = set(next(iter(orders.values())))
        for g, cnt in counts.items():
            rep = sorted(x for x, c in cnt.items() if c > 1)
            if rep or set(cnt) != genes:
                sys.exit(f"genome {g}: repeated genes {rep[:5]} or a different gene set; "
                         "use --core to keep only the genes shared once by all genomes")
    ref = args.reference or next(iter(orders))
    if ref not in orders:
        sys.exit(f"reference {ref} is not in the gene-order file")
    gene_ids = orders[ref]
    if not gene_ids:
        sys.exit("no genes left")
    sym = {x: i for i, x in enumerate(gene_ids)}
    perms = {}
    for g, seq in orders.items():
        p = [sym[x] for x in seq]
        perms[g] = tuple(linearise(p)[3] if args.linearise else p)

    phylo = parse_newick(Path(args.tree).read_text(encoding="utf-8"))
    phylo.name = {v: nm.strip("'\"") for v, nm in phylo.name.items()}
    in_tree = set(phylo.name.values())
    missing = sorted(set(orders) - in_tree)
    if missing:
        sys.exit(f"genomes without a tree leaf: {missing[:10]}")
    dropped = sorted(in_tree - set(orders))
    if dropped:
        print(f"note: {len(dropped)} tree leaves have no gene order and are dropped")
    tree = induced(phylo, list(orders))
    iid = f"{args.dataset}/{Path(args.out).stem}"
    rec = make_instance(iid, args.dataset, "user", gene_ids, tree, perms)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
    distinct = len(set(perms.values()))
    far = max(ulam(perms[ref], p) for p in perms.values())
    print(f"{iid}: {len(perms)} genomes, {len(gene_ids)} genes, {distinct} distinct orders, "
          f"largest distance to the reference {far} -> {args.out}")


if __name__ == "__main__":
    main()
