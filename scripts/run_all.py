#!/usr/bin/env python3
"""Run a campaign: every instance under the input folders, in parallel processes.

  python scripts/run_all.py inputs/ATGC0006 inputs/ATGC0093 --out results/campaign \
      --workers 4 --exact-seconds 120 --exact-seconds-full 3600

Each instance runs in its own process (scripts/solve.py), so memory is returned
after every instance. A process that exceeds its exact allowance plus --grace
seconds is stopped and recorded as KILLED. Finished results are kept on a rerun
(resume); --retry reruns ERROR and KILLED records.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs_for(dirs, out, exact_seconds, exact_full):
    jobs = []
    for d in dirs:
        d = Path(d)
        for f in sorted(d.rglob("*.json")):
            if f.name == "dataset.json":
                continue
            family = json.loads(f.read_text(encoding="utf-8"))["family"]
            rel = f.relative_to(d.parent)
            jobs.append(dict(instance=f, result=out / rel, family=family,
                             exact=exact_full if family == "full" else exact_seconds))
    # long jobs first, so the full trees overlap with the many short ones
    order = {"full": 0, "subtree": 1, "window": 2}
    return sorted(jobs, key=lambda j: order[j["family"]])


def finished(path, retry):
    if not path.exists():
        return False
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    return not (retry and status in ("ERROR", "KILLED"))


def run(job, grace, log):
    began = time.time()
    try:
        inst = os.path.relpath(job["instance"], ROOT)
    except ValueError:
        inst = str(job["instance"])
    cmd = [sys.executable, "-E", "-s", str(ROOT / "scripts" / "solve.py"), inst, str(job["result"]),
           "--exact-seconds", str(job["exact"]), "--quiet"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=job["exact"] + grace)
        status = json.loads(Path(job["result"]).read_text(encoding="utf-8"))["status"] \
            if Path(job["result"]).exists() else f"NO RESULT (exit {p.returncode})"
        if p.returncode not in (0, 1):
            log(f"  {inst}: exit {p.returncode}: {p.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        status = "KILLED"
        rec = dict(schema="tree-ulam-result/1", id=json.loads(Path(job["instance"]).read_text())["id"],
                   status="KILLED", error=f"process exceeded {job['exact'] + grace} s",
                   instance_file=inst.replace("\\", "/"), instance_sha256=sha256(job["instance"]))
        Path(job["result"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["result"]).write_text(json.dumps(rec), encoding="utf-8")
    log(f"{inst}: {status} ({time.time() - began:.1f} s)")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exact-seconds", type=float, default=120)
    ap.add_argument("--exact-seconds-full", type=float, default=3600)
    ap.add_argument("--grace", type=float, default=1800,
                    help="seconds allowed for the polynomial methods and bookkeeping")
    ap.add_argument("--retry", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    jobs = [j for j in jobs_for([Path(d).resolve() for d in args.inputs], out,
                                args.exact_seconds, args.exact_seconds_full)
            if not finished(j["result"], args.retry)]
    code = sorted(p for p in (ROOT / "treeulam").glob("*.py")) + \
        sorted(p for p in (ROOT / "native").glob("*") if p.suffix in (".cpp", ".dll", ".so")) + \
        [ROOT / "scripts" / "solve.py", ROOT / "scripts" / "run_all.py"]
    manifest = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), inputs=[str(d) for d in args.inputs],
                    settings=dict(workers=args.workers, exact_seconds=args.exact_seconds,
                                  exact_seconds_full=args.exact_seconds_full, grace=args.grace),
                    code={str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in code},
                    python=platform.python_version(), platform=platform.platform(),
                    cpu_count=os.cpu_count(), pending=len(jobs))
    (out / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(manifest, indent=1))
    log(f"campaign: {len(jobs)} instances to run with {args.workers} workers")
    with ThreadPoolExecutor(args.workers) as ex:
        statuses = list(ex.map(lambda j: run(j, args.grace, log), jobs))
    counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    log(f"campaign finished: {counts}")


if __name__ == "__main__":
    main()
