"""Run every method on one instance and return a checked result record.

Order of work:
  1. lifting and the 7/4 algorithm, each followed by local search;
  2. the contour and packing lower bounds (each with a witness);
  3. the exact decision procedure at D = U - 1, where U is the best cost found so
     far, repeated at the new U - 1 after every YES while the time allowance lasts.
Every labeling is expanded to the original genes and its cost recomputed before it
is recorded. OPTIMAL means lower = upper; the certificate says which method closed
the gap. TIMEOUT and CAP of the exact search never raise the lower bound.
"""
import time

from . import exact
from .approx import approx_74, contour_bound, lifting, local_search
from .blocks import Pool, Space
from .distance import ulam
from .median import MedianOracle
from .packing import packing_bound

RESULT_SCHEMA = "tree-ulam-result/1"


def labeling_cost(tree, labeling):
    return sum(ulam(labeling[u], labeling[v]) for u, v in tree.edges())


def check_labeling(inst, labeling, claimed):
    """A labeling must keep every leaf order, label every vertex by a permutation of
    the genes, and cost exactly what is claimed."""
    t = inst.tree
    if set(labeling) != set(t.adj):
        raise AssertionError("labeling does not cover the vertices")
    for v, seq in inst.leaf_seq.items():
        if tuple(labeling[v]) != seq:
            raise AssertionError("labeling changes a leaf")
    full = tuple(range(inst.n))
    for seq in labeling.values():
        if tuple(sorted(seq)) != full:
            raise AssertionError("a label is not a permutation of the genes")
    cost = labeling_cost(t, labeling)
    if cost != claimed:
        raise AssertionError(f"labeling costs {cost}, claimed {claimed}")
    return cost


def solve(inst, exact_seconds, decide_options=None, log=lambda *a: None):
    t_start = time.time()
    t, leaf_seq = inst.tree, inst.leaf_seq
    internal = [v for v in t.adj if v not in leaf_seq]
    rec = dict(schema=RESULT_SCHEMA, id=inst.id, family=inst.rec["family"], n_genes=inst.n,
               leaves=len(leaf_seq), internal_vertices=len(internal),
               distinct_orders=inst.distinct_orders, algorithms={}, bounds={}, exact=None)

    def internal_labels(labeling):
        return inst.named({v: labeling[v] for v in internal})

    if inst.distinct_orders == 1:
        only = next(iter(leaf_seq.values()))
        rec.update(status="TRIVIAL", lower=0, upper=0, certificate="all leaves equal",
                   best=dict(algorithm="lifting", labels=internal_labels({v: only for v in t.adj})))
        rec["seconds"] = round(time.time() - t_start, 3)
        return rec

    space = Space(list(dict.fromkeys(leaf_seq.values())))
    pool = Pool(space.w)
    med = MedianOracle(pool)
    leaf_lab = {v: pool.add(space.contract(seq)) for v, seq in leaf_seq.items()}
    rec["blocks"] = space.k
    best = dict(cost=None)

    def record(name, cost, lab_idx, seconds):
        labeling = {v: space.expand(pool.items[i]) for v, i in lab_idx.items()}
        check_labeling(inst, labeling, int(cost))
        rec["algorithms"][name] = dict(cost=int(cost), seconds=round(seconds, 3),
                                       labels=internal_labels(labeling))
        if best["cost"] is None or cost < best["cost"]:
            best.update(cost=int(cost), algorithm=name, labeling=labeling)
        log(f"  {name}: {cost} ({seconds:.2f} s)")

    s = time.time()
    cost, lab = lifting(t, leaf_lab, pool)
    record("lifting", cost, lab, time.time() - s)
    s = time.time()
    cost, lab_ls = local_search(t, lab, leaf_lab, pool, med)
    record("lifting+ls", cost, lab_ls, time.time() - s)
    s = time.time()
    cost, lab = approx_74(t, leaf_lab, pool, med)
    record("7/4", cost, lab, time.time() - s)
    s = time.time()
    cost, lab_ls = local_search(t, lab, leaf_lab, pool, med)
    record("7/4+ls", cost, lab_ls, time.time() - s)
    rec["median_calls"] = dict(distinct=med.calls, dynamic_program=med.dp_calls)

    s = time.time()
    value, order = contour_bound(t, leaf_lab, pool.d)
    rec["bounds"]["contour"] = dict(value=int(value), seconds=round(time.time() - s, 3),
                                    leaf_order=[inst.vertices[v] for v in order])
    s = time.time()
    value, pieces = packing_bound(t.adj, leaf_lab, pool.d, med.cost)
    rec["bounds"]["packing"] = dict(
        value=int(value), seconds=round(time.time() - s, 3),
        pieces=[dict(kind=k, leaves=[inst.vertices[v] for v in vs], value=int(val))
                for k, vs, val in pieces])
    contour, packing = rec["bounds"]["contour"]["value"], rec["bounds"]["packing"]["value"]
    lower, upper = max(contour, packing), best["cost"]
    if lower > upper:
        raise AssertionError(f"lower bound {lower} exceeds a feasible cost {upper}")
    log(f"  bounds: contour {contour}, packing {packing}")

    # exact decision procedure; runs even when the bounds already match, as a check
    opts = dict(decide_options or {})
    calls = []
    deadline = time.time() + exact_seconds
    D = upper - 1
    while D >= 0:
        remaining = deadline - time.time()
        if remaining <= 0.1:
            break
        s = time.time()
        r = exact.decide(t, leaf_seq, D, time_limit=remaining, **opts)
        call = dict(D=D, result=r["result"], reason=r["reason"], seconds=round(time.time() - s, 3),
                    **{k: r.get(k) for k in ("q", "V_red", "n_red", "cherries", "cands", "tree_lb")})
        calls.append(call)
        log(f"  exact D={D}: {r['result']} ({r['reason']}, {call['seconds']} s)")
        if r["result"] == "YES":
            if lower == upper:
                raise AssertionError(f"exact search found cost <= {D} below a certified optimum {upper}")
            cost, labeling = exact.label_tree_from(t, leaf_seq, r["labels_used"])
            check_labeling(inst, labeling, int(cost))
            if cost > D or cost < lower:
                raise AssertionError(f"exact YES labeling costs {cost} (D={D}, lower={lower})")
            call["found_cost"] = int(cost)
            rec["algorithms"]["exact"] = dict(cost=int(cost), seconds=call["seconds"],
                                              labels=internal_labels(labeling))
            best.update(cost=int(cost), algorithm="exact", labeling=labeling)
            upper = int(cost)
            if lower == upper:
                break
            D = upper - 1
            continue
        break
    # Certificates of optimality for the final upper bound; D = upper - 1 holds for
    # the last call unless it was a YES, so a final NO proves OPT >= upper.
    certificate = [name for name, v in (("contour", contour), ("packing", packing)) if v == upper]
    if calls and calls[-1]["result"] == "NO" and calls[-1]["D"] == upper - 1:
        certificate.append("exact NO")
    if certificate:
        lower = upper
    rec["exact"] = dict(budget_seconds=exact_seconds, calls=calls,
                        outcome=calls[-1]["result"] if calls else "NOT_RUN")
    rec.update(lower=int(lower), upper=int(upper), gap=int(upper - lower),
               status="OPTIMAL" if certificate else "OPEN", certificate=certificate,
               best=dict(algorithm=best["algorithm"], cost=best["cost"],
                         labels=internal_labels(best["labeling"])))
    rec["seconds"] = round(time.time() - t_start, 3)
    return rec
