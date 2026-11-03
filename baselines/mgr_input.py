#!/usr/bin/env python3
"""MGR 2.01 input for one ATGC cluster: signed gene orders and the fixed tree.

  python mgr_input.py data/atgc25/ATGC0327 results/atgc25_20261102/inputs/ATGC0327 out/ATGC0327

Writes genomes.txt (one unichromosomal signed order per genome, in MGR's format) and
topology.txt (the cluster's normalised tree as a Newick topology over genome numbers
1..m, for MGR -t). The unsigned orders are the instance's own permutations: the core
genes, contig order, rotation and direction are taken from dataset.json, and each
gene's sign is its strand in atgc.pty, flipped when the genome was read reversed.
The script checks that dropping the signs gives back the instance's permutation.
"""
import collections
import json
import re
import sys
from pathlib import Path


def signed_orders(pty, dataset):
    core = set(dataset["gene_ids"])
    sym = {c: i + 1 for i, c in enumerate(dataset["gene_ids"])}
    genes, contigs = collections.defaultdict(list), collections.defaultdict(list)
    with open(pty, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] == "#":
                contigs[f[3]].append(f[4])
            elif len(f) > 7:
                m = re.match(r"(\d+)", f[1])
                genes[f[3]].append((f[4], int(m.group(1)) if m else 0, f[7], f[2]))
    out = {}
    for g in dataset["genomes"]:
        v = genes[g]
        seq = []
        for ct in contigs[g] or sorted({x[0] for x in v}):
            seq += [(sym[c], 1 if s == "+" else -1)
                    for _, _, c, s in sorted((x for x in v if x[0] == ct), key=lambda x: x[1]) if c in core]
        lin = dataset["linearisation"][g]
        if lin["reversed"]:
            seq = [(a, -s) for a, s in reversed(seq)]
        k = lin["rotation"]
        seq = seq[k:] + seq[:k]
        if [a for a, _ in seq] != dataset["permutations"][g]:
            raise AssertionError(f"{g}: signed order does not match the instance permutation")
        out[g] = [a * s for a, s in seq]
    return out


def topology(tree, number):
    """Binary Newick over genome numbers, rooted on the edge of genome 1.

    MGR's fixed-topology mode takes every "(x,y" of two genome numbers as a cherry, so
    the tree must be rooted on an edge: an unrooted trifurcation "(1,5,(...))" would be
    read as a cherry (1,5)."""
    adj = collections.defaultdict(list)
    for u, v in tree["edges"]:
        adj[u].append(v)
        adj[v].append(u)
    first = next(g for g, i in number.items() if i == 1)

    def nw(v, parent):
        kids = [w for w in adj[v] if w != parent]
        return str(number[v]) if not kids else "(" + ",".join(nw(w, v) for w in kids) + ")"

    if len(adj[first]) != 1:
        raise ValueError("genome 1 is not a leaf")
    return f"(1,{nw(adj[first][0], first)});"


def main():
    folder, inst, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dataset = json.loads((inst / "dataset.json").read_text(encoding="utf-8"))
    orders = signed_orders(folder / "atgc.pty", dataset)
    number = {g: i + 1 for i, g in enumerate(dataset["genomes"])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "genomes.txt").write_text(
        "".join(f">{g}\n{' '.join(map(str, orders[g]))} $\n\n" for g in dataset["genomes"]), encoding="ascii")
    (out / "topology.txt").write_text(topology(dataset["tree"], number) + "\n", encoding="ascii")
    (out / "numbers.json").write_text(json.dumps(number, indent=1), encoding="utf-8")
    print(f"{dataset['dataset']}: {len(orders)} genomes, {dataset['n_genes']} genes, "
          f"{len(set(map(tuple, orders.values())))} distinct signed orders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
