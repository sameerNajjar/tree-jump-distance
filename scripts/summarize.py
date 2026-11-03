#!/usr/bin/env python3
"""Tables from a campaign's result records.

  python scripts/summarize.py results/full

writes results/full/instances.csv (one row per tree) and results/full/summary.md
(status counts, tables by number of distinct leaf orders q, exact-search outcomes
and one row per tree). Only OPTIMAL trees enter the "attains OPT" counts.
"""
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

ALGS = ("lifting", "lifting+ls", "7/4", "7/4+ls")


def qclass(q):
    return "1" if q == 1 else "2" if q == 2 else "3" if q == 3 else "4+"


def rows(folder):
    out = []
    for f in sorted(Path(folder).rglob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("schema") != "tree-ulam-result/1":
            continue
        row = dict(id=r["id"], status=r["status"],
                   n_genes=r.get("n_genes"), leaves=r.get("leaves"), q=r.get("distinct_orders"),
                   lower=r.get("lower"), upper=r.get("upper"), gap=r.get("gap"),
                   certificate="+".join(r.get("certificate") or []),
                   seconds=r.get("wall_seconds"))
        if r["status"] in ("OPTIMAL", "OPEN"):
            row.update(contour=r["bounds"]["contour"]["value"], packing=r["bounds"]["packing"]["value"])
            for a in ALGS:
                row[a] = r["algorithms"][a]["cost"]
            row["exact_cost"] = r["algorithms"].get("exact", {}).get("cost")
            calls = r["exact"]["calls"]
            row["exact_calls"] = len(calls)
            row["exact_last"] = calls[-1]["result"] if calls else "NOT_RUN"
            row["exact_reason"] = calls[-1]["reason"] if calls else ""
            row["exact_seconds"] = round(sum(c["seconds"] for c in calls), 2)
            row["exact_D"] = ";".join(str(c["D"]) for c in calls)
        out.append(row)
    return out


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in body]
    return "\n".join(lines)


def main():
    folder = Path(sys.argv[1])
    data = rows(folder)
    if not data:
        sys.exit("no results")
    keys = list(dict.fromkeys(k for r in data for k in r))
    with open(folder / "instances.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(data)
    md = [f"# Campaign summary ({len(data)} trees)", ""]
    # 1. status
    c = collections.Counter(r["status"] for r in data)
    md += ["## Status", "",
           table(["trees", "trivial", "optimal", "open", "error/killed"],
                 [[len(data), c["TRIVIAL"], c["OPTIMAL"], c["OPEN"], c["ERROR"] + c["KILLED"]]]), ""]
    # 2. by q among nontrivial
    body = []
    nontriv = [r for r in data if r["status"] in ("OPTIMAL", "OPEN")]
    for qc in ("2", "3", "4+"):
        rs = [r for r in nontriv if qclass(r["q"]) == qc]
        if not rs:
            continue
        opt = [r for r in rs if r["status"] == "OPTIMAL"]
        body.append([qc, len(rs), len(opt), len(rs) - len(opt)]
                    + [sum(1 for r in opt if r[a] == r["upper"]) for a in ALGS]
                    + [sum(1 for r in opt if r["contour"] == r["upper"]),
                       sum(1 for r in opt if r["packing"] == r["upper"]),
                       sum(1 for r in opt if r["certificate"] == "exact NO"),
                       sum(1 for r in opt if r["exact_cost"] is not None and r["exact_cost"] == r["upper"]
                           and min(r[a] for a in ALGS) > r["upper"])])
    md += ["## Nontrivial trees by number of distinct leaf orders q", "",
           "Algorithm columns count OPTIMAL trees where the method attains OPT. "
           "'only exact NO' = optimality certified by the exact search alone; "
           "'exact improves' = the exact search found a cheaper labeling than every approximation.", "",
           table(["q", "trees", "optimal", "open"] + list(ALGS)
                 + ["contour = OPT", "packing = OPT", "only exact NO", "exact improves"], body), ""]
    # 3. exact search outcomes
    if nontriv:
        ex = collections.Counter(r["exact_last"] for r in nontriv)
        reasons = collections.Counter(r["exact_reason"] for r in nontriv if r["exact_last"] == "NO")
        secs = [r["exact_seconds"] for r in nontriv]
        md += ["## Exact search (last call per tree)", "",
               table(["nontrivial", "NO", "YES", "TIMEOUT", "CAP", "NO reasons", "median s", "max s"],
                     [[len(nontriv), ex["NO"], ex["YES"], ex["TIMEOUT"], ex["CAP"],
                       "; ".join(f"{k}: {v}" for k, v in reasons.most_common()),
                       round(statistics.median(secs), 2), round(max(secs), 1)]]), ""]
    # 4. every tree
    body = [[r["id"], r["leaves"], r["n_genes"], r["q"], r["status"], r["lower"], r["upper"], r["gap"],
             r["certificate"] or "-"] + [r[a] for a in ALGS] + [r["exact_last"], r["exact_D"]]
            for r in sorted(nontriv, key=lambda r: r["id"])]
    md += ["## Trees", "", table(["id", "leaves", "genes", "q", "status", "lower", "upper", "gap",
                                  "certificate"] + list(ALGS) + ["exact", "D tried"], body), ""]
    bad = [r["id"] for r in data if r["status"] in ("ERROR", "KILLED")]
    if bad:
        md += ["## Errors", ""] + [f"- {b}" for b in bad]
    (folder / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
