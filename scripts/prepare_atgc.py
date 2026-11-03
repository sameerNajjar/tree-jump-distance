#!/usr/bin/env python3
"""Build the full-tree instance of one ATGC cluster.

  python scripts/prepare_atgc.py ATGC0327 <cluster>/atgc.pty <cluster>/atgc.iq.r.lab.tre inputs/ATGC0327

writes inputs/ATGC0327/dataset.json (genomes, linearisation, gene ids, full tree)
and inputs/ATGC0327/F.json: the cluster's whole tree on all its core genes. It then
compares F.json with inputs.sha256, the checksums of the instances used in the
experiments, and exits with status 1 if it differs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeulam import __version__, atgc                    # noqa: E402
from treeulam.instance import sha256, tree_record          # noqa: E402

ATGC_SOURCE = "https://ftp.ncbi.nih.gov/pub/wolf/ATGC/atgc-25/"
CHECKSUMS = Path(__file__).resolve().parents[1] / "inputs.sha256"


def verify(name, path):
    """Compare the written F.json with the published checksum of the experiments' instance."""
    if not CHECKSUMS.is_file():
        return True
    want = None
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        if rel == f"{name}/F.json":
            want = digest
    if want is None:
        print(f"{name}: no published checksum for this dataset name")
        return True
    ok = sha256(path) == want
    print(f"{name}: F.json {'matches' if ok else 'differs from'} inputs.sha256")
    return ok


def file_record(rec):
    """File name and SHA-256 only: no local paths in the shipped records."""
    return dict(file=Path(rec["path"]).name, sha256=rec["sha256"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("pty")
    ap.add_argument("tre")
    ap.add_argument("out")
    args = ap.parse_args()
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
        tree=tree_record(data["phylo"]))
    (out / "dataset.json").write_text(json.dumps(dataset, indent=1), encoding="utf-8")
    path = out / "F.json"
    path.write_text(json.dumps(atgc.full_instance(data), separators=(",", ":")), encoding="utf-8")
    print(f"{args.name}: F.json in {time.time() - began:.1f} s")
    return 0 if verify(args.name, path) else 1


if __name__ == "__main__":
    sys.exit(main())
