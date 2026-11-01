#!/usr/bin/env python3
"""Exercise the exact search's YES path on certified optima.

  python tests/test_yes_path.py results/campaign --out results/yes_path --seconds 120 --workers 4

For every OPTIMAL record of a campaign, the exact decision procedure runs at
D = OPT. The answer must be YES (or an honest TIMEOUT / CAP), never NO. A YES
labeling is carried back to the full tree with label_tree_from, checked by the
package, and its cost recomputed by the independent checker (scripts/check.py).
Its cost must equal OPT exactly: at most D by the YES, at least OPT by the
certificate. Each instance runs in its own process.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECIDE = dict(max_cands=8_000_000, candidate_memory_mib=1024, probe_seconds=0.05, subtree_feedback=True)


def one(instance_path, opt, out, seconds):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import check                                         # independent checker
    from treeulam import exact, instance
    from treeulam.solve import check_labeling
    inst = instance.load(instance_path)
    began = time.time()
    r = exact.decide(inst.tree, inst.leaf_seq, opt, time_limit=seconds, **DECIDE)
    rec = dict(id=inst.id, D=opt, result=r["result"], reason=r["reason"],
               decide_seconds=round(r["t_total"], 3), distinct_orders=inst.distinct_orders,
               **{k: r.get(k) for k in ("V", "V_red", "q", "n", "n_red", "cherries", "cherry_exact",
                                        "cands", "t_cands", "t_dp", "tree_lb")})
    if r["result"] == "NO":
        rec["status"] = "CONTRADICTION"                   # NO at a certified optimum
    elif r["result"] == "YES":
        cost, lab = exact.label_tree_from(inst.tree, inst.leaf_seq, r["labels_used"])
        check_labeling(inst, lab, int(cost))
        internal = {inst.vertices[v]: [s + 1 for s in seq] for v, seq in lab.items() if v not in inst.leaf_seq}
        independent = check.Checked(inst.rec).labeling_cost(internal)
        rec.update(cost=int(cost), independent_cost=independent, labels=internal,
                   status="YES_OPT" if cost == independent == opt else "WRONG_COST")
    else:
        rec["status"] = r["result"]
    rec["wall_seconds"] = round(time.time() - began, 3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--one", nargs=3, metavar=("INSTANCE", "OPT", "OUT"))
    args = ap.parse_args()
    if args.one:
        one(args.one[0], int(args.one[1]), args.one[2], args.seconds)
        return 0
    out = Path(args.out).resolve()
    jobs = []
    for f in sorted(Path(args.campaign).rglob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("schema") == "tree-ulam-result/1" and r["status"] == "OPTIMAL":
            jobs.append((r["id"], r["instance_file"], r["upper"], out / (r["id"].replace("/", "_") + ".json")))
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a", encoding="utf-8")

    def run(job):
        iid, inst, opt, res = job
        if res.exists():
            return json.loads(res.read_text())["status"]
        cmd = [sys.executable, "-E", "-s", __file__, "--one", inst, str(opt), str(res), "--seconds", str(args.seconds)]
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                               capture_output=True, text=True, timeout=args.seconds + 600)
            status = json.loads(res.read_text())["status"] if res.exists() else f"ERROR: {p.stderr[-400:]}"
        except subprocess.TimeoutExpired:
            status = "KILLED"
        line = f"{time.strftime('%H:%M:%S')} {iid}: {status}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        return status

    with ThreadPoolExecutor(args.workers) as ex:
        statuses = list(ex.map(run, jobs))
    counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    print(json.dumps(dict(instances=len(jobs), outcomes=counts), indent=1))
    return 0 if not any(s in ("CONTRADICTION", "WRONG_COST", "KILLED") or s.startswith("ERROR") for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
