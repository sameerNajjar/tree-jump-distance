#!/usr/bin/env python3
"""Build the instance files of one ATGC dataset.

  python scripts/prepare_atgc.py ATGC0093 data/ATGC0093.pty "data/atgc0093 original.tre" inputs/ATGC0093 \
      --subset-sizes 4 8 16

writes inputs/ATGC0093/dataset.json (genomes, linearisation, gene ids, full tree)
and one instance file per case under F/, W10/, W20/, W50/ and S<k>/. It then
compares the files with inputs.sha256, the checksums of the inputs used in the
experiments, and exits with status 1 if any file differs.

  python scripts/prepare_atgc.py ATGC0327 <cluster>/atgc.pty <cluster>/atgc.iq.r.lab.tre inputs/ATGC0327 --full-only

--full-only writes dataset.json and F.json only: the cluster's whole tree on all its
core genes, with no windows or subtrees (the full-tree evaluation). Only F.json is
compared with inputs.sha256 in this mode.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeulam import __version__, atgc                    # noqa: E402
from treeulam.instance import sha256, tree_record          # noqa: E402

WINDOWS = ((10, 5), (20, 10), (50, 25))                    # (length, step)
SEEDS = (1, 2, 3)
ATGC_SOURCE = "https://ftp.ncbi.nih.gov/pub/wolf/ATGC/atgc-25/"
CHECKSUMS = Path(__file__).resolve().parents[1] / "inputs.sha256"


def verify(name, out, only=None):
    """Compare the written files with the published checksums of the experiments' inputs.
    only: restrict the comparison to these relative paths (F.json for --full-only)."""
    if not CHECKSUMS.is_file():
        return True
    want = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        if rel.startswith(name + "/") and (only is None or rel[len(name) + 1:] in only):
            want[rel[len(name) + 1:]] = digest
    if not want:
        print(f"{name}: no published checksums for this dataset name")
        return True
    bad = [rel for rel, digest in want.items()
           if not (out / rel).is_file() or sha256(out / rel) != digest]
    print(f"{name}: {len(want) - len(bad)} of {len(want)} files match inputs.sha256")
    for rel in bad[:5]:
        print(f"  differs or missing: {rel}")
    return not bad


def file_record(rec):
    """File name and SHA-256 only: no local paths in the shipped records."""
    return dict(file=Path(rec["path"]).name, sha256=rec["sha256"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("pty")
    ap.add_argument("tre")
    ap.add_argument("out")
    ap.add_argument("--subset-sizes", type=int, nargs="+")
    ap.add_argument("--full-only", action="store_true", help="whole tree on all core genes only")
    args = ap.parse_args()
    if not args.full_only and not args.subset_sizes:
        ap.error("--subset-sizes is required unless --full-only is given")
    windows, sizes, seeds = ((), [], ()) if args.full_only else (WINDOWS, args.subset_sizes, SEEDS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    data = atgc.load_dataset(args.name, args.pty, args.tre)
    dataset = dict(
        schema="tree-ulam-dataset/1", dataset=args.name, package_version=__version__,
        source=ATGC_SOURCE, pty=file_record(data["pty"]), tre=file_record(data["tre"]),
        cluster=data["cluster"], reference=data["reference"],
        n_genes=data["n"], gene_ids=data["gene_ids"], genomes=data["names"],
        missing_from_tree=data["missing_from_tree"],
        linearisation={g: data["meta"][g] for g in data["names"]},
        permutations={g: [s + 1 for s in data["perms"][g]] for g in data["names"]},
        distinct_orders=len(set(data["perms"].values())),
        tree=tree_record(data["phylo"]),
        design=dict(windows=[dict(length=w, step=s) for w, s in windows],
                    subset_sizes=sizes, subset_seeds=list(seeds)))
    (out / "dataset.json").write_text(json.dumps(dataset, indent=1), encoding="utf-8")
    instances = atgc.build_instances(data, windows, sizes, seeds)
    counts = {}
    for rec in instances:
        rel = Path(*rec["id"].split("/")[1:])
        path = out / rel.with_suffix(".json") if len(rel.parts) > 1 else out / f"{rel}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
        key = rel.parts[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"{args.name}: {len(instances)} instances {counts} in {time.time() - began:.1f} s")
    return 0 if verify(args.name, out, only=("F.json",) if args.full_only else None) else 1


if __name__ == "__main__":
    sys.exit(main())
