#!/usr/bin/env python3
"""Solve one instance file and write its result record.

  python scripts/solve.py inputs/ATGC0093/W20/0011.json results/ATGC0093/W20/0011.json --exact-seconds 120
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeulam import __version__, instance               # noqa: E402
from treeulam.solve import solve                          # noqa: E402

DECIDE = dict(max_cands=8_000_000, candidate_memory_mib=1024, probe_seconds=0.05, subtree_feedback=True)


def write_atomic(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance")
    ap.add_argument("result")
    ap.add_argument("--exact-seconds", type=float, required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    began = time.time()
    path = Path(args.instance)
    inst = instance.load(path)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    log(f"{inst.id}: {len(inst.leaf_seq)} leaves, {inst.n} genes, {inst.distinct_orders} distinct orders")
    try:
        rec = solve(inst, args.exact_seconds, decide_options=DECIDE, log=log)
    except Exception as e:                               # recorded, never silently dropped
        rec = dict(schema="tree-ulam-result/1", id=inst.id, status="ERROR",
                   error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec["instance_file"] = str(path).replace("\\", "/")
    rec["instance_sha256"] = instance.sha256(path)
    rec["settings"] = dict(exact_seconds=args.exact_seconds, decide=DECIDE)
    rec["environment"] = dict(package=__version__, python=platform.python_version(),
                              numpy=__import__("numpy").__version__, platform=platform.platform())
    rec["wall_seconds"] = round(time.time() - began, 3)
    write_atomic(Path(args.result), rec)
    log(f"{inst.id}: {rec['status']} lower {rec.get('lower')} upper {rec.get('upper')} "
        f"({rec['wall_seconds']} s)")
    return 0 if rec["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
