"""Exact decision procedure: is there a labeling of cost at most D?

decide() is the FPT algorithm in the total cost D, as implemented for the
experiments:
  1. kernel rules 1-4 (monochromatic sides, degree-2 vertices, repeated sibling
     leaves, uniform paths) shrink the tree without changing the answer;
  2. quick NO certificates: more than D+1 distinct leaf labels, a leaf pair farther
     than D, a minimum spanning tree of the labels longer than 2D, or a contour
     bound above D;
  3. every common block of the leaf labels becomes one weighted symbol;
  4. each internal vertex receives every permutation that passes exact branch lower
     bounds (native prefix branch-and-bound); a vertex with two leaf neighbours and
     one other neighbour is solved exactly by three-medians instead;
  5. a tree dynamic program over these candidates, with values capped at D + 1.
YES comes with a labeling that is rebuilt in the original alphabet and checked.
TIMEOUT and CAP mean the computation stopped; neither is a NO.
"""
import collections
import random
import time

import numpy as np

from . import native
from .approx import contour_bound
from .blocks import common_blocks
from .distance import Metric, ulam
from .errors import CapExceeded, Timeout
from .median import median3_cost, median3_seq


# ----------------------------------------------------------------------------- kernel

def _rule1(t, lab):
    """Monochromatic side: a side of >= 2 vertices whose leaves share one label pi
    is replaced by one pi-leaf."""
    if len(t.adj) <= 2:
        return False
    for p in list(t.adj):
        for r in list(t.adj[p]):
            C, st = {r}, [r]
            while st:
                a = st.pop()
                for x in t.adj[a]:
                    if x != p and x not in C:
                        C.add(x)
                        st.append(x)
            if len(C) < 2:
                continue
            labs = {lab[v] for v in C if v in lab}
            if len(labs) != 1:
                continue
            pi = next(iter(labs))
            for v in C:
                t.remove(v)
                lab.pop(v, None)
            x = t.new()
            t.edge(p, x)
            lab[x] = pi
            return True
    return False


def _rule2(t, lab):
    """An unlabelled vertex of degree 2 is suppressed."""
    for v in list(t.adj):
        if v not in lab and len(t.adj[v]) == 2:
            a, b = t.adj[v]
            t.remove(v)
            t.edge(a, b)
            return True
    return False


def _rule3(t, lab, D):
    """More than D + 1 leaves with one label on one vertex: keep D + 1 of them."""
    if len(t.adj) <= 2:
        return False
    changed = False
    for v in list(t.adj):
        if v not in t.adj:
            continue
        groups = collections.defaultdict(list)
        for u in t.adj[v]:
            if u in lab and len(t.adj[u]) == 1:
                groups[lab[u]].append(u)
        for us in groups.values():
            if len(us) > D + 1:
                for u in sorted(us)[D + 1:]:
                    t.remove(u)
                    lab.pop(u, None)
                changed = True
    return changed


def _rule4(t, lab):
    """Uniform path: >= 3 consecutive skeleton-degree-2 vertices whose pendant
    leaves all carry one label pi shrink to the two end vertices."""
    leafset = {v for v in t.adj if v in lab and len(t.adj[v]) == 1}
    skel = {v: {u for u in t.adj[v] if u not in leafset} for v in t.adj if v not in leafset}
    pend = {v: [lab[u] for u in t.adj[v] if u in leafset] for v in skel}
    cand = {v for v in skel if v not in lab and len(skel[v]) == 2 and pend[v]
            and len(set(pend[v])) == 1}
    seen = set()
    for v in cand:
        if v in seen:
            continue
        pi = pend[v][0]
        run, st = {v}, [v]
        while st:
            a = st.pop()
            for x in skel[a]:
                if x in cand and x not in run and pend[x][0] == pi:
                    run.add(x)
                    st.append(x)
        seen |= run
        if len(run) < 3:
            continue
        ends = [u for u in run if any(x not in run for x in skel[u])]
        if len(ends) != 2:
            continue
        for u in run - set(ends):
            for x in list(t.adj[u]):
                if x in leafset:
                    t.remove(x)
                    lab.pop(x, None)
            t.remove(u)
        t.edge(ends[0], ends[1])
        return True
    return False


def kernelize(t, lab, D):
    t = t.copy()
    lab = dict(lab)
    changed = True
    while changed:
        changed = _rule1(t, lab) or _rule2(t, lab) or _rule3(t, lab, D) or _rule4(t, lab)
    return t, lab


# ----------------------------------------------------------------------------- bounds

def _off_path_bounds(t, lab, v, lid, dm):
    """Cost forced off a v-to-leaf path, as a constant for each anchor.

    Closed subtrees use edge-disjoint leaf-pair packings (triangle inequality).
    Deleting a path leaves disjoint side subtrees: add their closed bounds.
    For degree > 3 this remains a valid, possibly weaker packing bound.
    """
    par, order, stack = {v: None}, [], [v]
    while stack:
        u = stack.pop()
        order.append(u)
        for z in t.adj[u]:
            if z not in par:
                par[z] = u
                stack.append(z)
    closed, opened = {}, {}
    for u in reversed(order):
        if u in lab:
            closed[u], opened[u] = 0, {lid[lab[u]]: 0}
            continue
        kids = [z for z in t.adj[u] if par.get(z) == u]
        total = sum(closed[z] for z in kids)
        best, op = total, {}
        for z in kids:
            for a, value in opened[z].items():
                op[a] = max(op.get(a, -1), value + total - closed[z])
        for i, z in enumerate(kids):
            for w in kids[i + 1:]:
                for a, av in opened[z].items():
                    for b, bv in opened[w].items():
                        best = max(best, total - closed[z] - closed[w] + av + bv + dm[a][b])
        closed[u], opened[u] = best, op
    result = {}
    for branch in t.adj[v]:
        stack = [(branch, 0)]
        while stack:
            u, extra = stack.pop()
            if u in lab:
                key = (branch, lid[lab[u]])
                result[key] = max(result.get(key, 0), extra)
                continue
            kids = [z for z in t.adj[u] if par.get(z) == u]
            total = sum(closed[z] for z in kids)
            for z in kids:
                stack.append((z, extra + total - closed[z]))
    return result


def _mst(dm):
    q = len(dm)
    edges = []
    best = {j: (dm[0][j], 0) for j in range(1, q)}
    while best:
        j = min(best, key=lambda x: best[x][0])
        w, i = best.pop(j)
        edges.append((i, j, w))
        for x in best:
            if dm[j][x] < best[x][0]:
                best[x] = (dm[j][x], j)
    return edges


def _branches(t, lab, v, lid, rng, orders=8):
    """For each neighbour u of v: (u, distinct label ids in that branch, several DFS
    leaf orders of the branch as label-id sequences)."""
    out = []
    for u in t.adj[v]:
        S = set()
        seqs = []
        for k in range(orders):
            seq, st, seen = [], [u], {v, u}
            while st:
                x = st.pop()
                if x in lab:
                    li = lid[lab[x]]
                    S.add(li)
                    if not seq or seq[-1] != li:
                        seq.append(li)
                nb = [y for y in t.adj[x] if y not in seen]
                if k:
                    rng.shuffle(nb)
                for y in nb:
                    seen.add(y)
                    st.append(y)
            seqs.append(seq)
        out.append((u, sorted(S), seqs))
    return out


def _branch_lb(dist, labs, seqs, dm):
    """Lower bound on the cost of one branch (with its connecting edge) when the
    branch root is labelled tau: the farthest leaf, and half of every closed DFS walk
    from tau through the branch's leaves."""
    lb = dist[:, labs].max(axis=1)
    for seq in seqs:
        inner = sum(dm[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
        walk = dist[:, seq[0]] + dist[:, seq[-1]] + inner
        lb = np.maximum(lb, (walk + 1) // 2)
    return lb


# ----------------------------------------------------------------------------- decision

def decide(t, lab, D, time_limit=60.0, max_cands=8_000_000, candidate_memory_mib=1024,
           probe_seconds=0.05, subtree_feedback=True):
    """Decide OPT <= D for the tree t with leaf labels lab (vertex -> tuple).

    Returns a dict with result in {YES, NO, TIMEOUT, CAP}, the reason, statistics and,
    for YES, 'labels_used': the labels of a checked solution on the kernel tree."""
    st = dict(D=D, V=len(t.adj), result=None, reason="",
              n=len(next(iter(lab.values()))), n_red=None, cands=0, t_total=0.0)
    t0 = time.time()
    deadline = t0 + time_limit
    metric = None
    try:
        t, lab = kernelize(t, lab, D)
        st["V_red"] = len(t.adj)
        labels = list(dict.fromkeys(lab.values()))
        q = st["q"] = len(labels)
        if q == 1:
            st.update(result="YES", reason="one label", cost=0, labels_used=labels)
            return st
        if D == 0:
            st.update(result="NO", reason="D=0, q>1")
            return st
        if q > D + 1:
            st.update(result="NO", reason="q>D+1")
            return st
        dm = [[ulam(labels[i], labels[j]) for j in range(q)] for i in range(q)]
        if max(map(max, dm)) > D:
            st.update(result="NO", reason="pair distance > D")
            return st
        if sum(w for _, _, w in _mst(dm)) > 2 * D:
            st.update(result="NO", reason="MST > 2D")
            return st
        lid = {L: i for i, L in enumerate(labels)}
        tree_lb = contour_bound(t, lab, lambda a, b: dm[lid[a]][lid[b]])[0]
        st["tree_lb"] = tree_lb
        if tree_lb > D:
            st.update(result="NO", reason="DFS tree lower bound > D")
            return st
        # weighted common-block contraction
        blocks = common_blocks([list(x) for x in labels])
        wts = [len(B) for B in blocks]
        st.update(blocks=len(blocks), n_red=len(blocks))
        sym_id = {s: b for b, B in enumerate(blocks) for s in B}
        red = {L: np.array(_contract_by(L, sym_id), dtype=np.int16) for L in labels}
        metric = Metric(wts)
        redl = [red[L] for L in labels]
        internal = [v for v in t.adj if v not in lab]
        st["t_compress"] = time.time() - t0
        if not internal:
            c = sum(ulam(lab[u], lab[v]) for u, v in t.edges())
            st.update(result="YES" if c <= D else "NO", reason="no free vertex", cost=c)
            if c <= D:
                st["labels_used"] = labels
            return st
        # A cherry (free vertex with two leaf neighbours and one other neighbour) is
        # solved exactly by three-medians for every label of that neighbour.
        explicit = {}
        st["cherries"] = 0
        st["cherry_exact"] = 0
        leafn = {v: [u for u in t.adj[v] if u in lab] for v in internal}
        intn = {v: [u for u in t.adj[v] if u not in lab] for v in internal}
        if len(internal) == 1 and len(leafn[internal[0]]) == 3:
            v0 = internal[0]
            x, y, z = (lid[lab[u]] for u in leafn[v0])
            val = median3_cost(redl[x], redl[y], redl[z], metric.w)
            st.update(reason="single free vertex: exact 3-median", cost_red=val,
                      t_cands=0.0, t_dp=time.time() - t0)
            if val > D:
                st["result"] = "NO"
                return st
            st["result"] = "YES"
            tau = np.array(median3_seq(redl[x], redl[y], redl[z], metric.w), dtype=np.int16)
            full_lab = _expand_solution(t, lab, {}, {}, red, blocks, explicit={v0: tau})
            _accept(st, t, lab, full_lab, val)
            return st
        cherries = {v for v in internal if len(leafn[v]) == 2 and len(intn[v]) == 1}
        non_ch = [v for v in internal if v not in cherries]
        root = non_ch[0] if non_ch else internal[0]
        elim = cherries - {root}
        st["cherries"] = len(elim)
        parent, order, stk = {root: None}, [], [root]
        while stk:
            u = stk.pop()
            order.append(u)
            for w in t.adj[u]:
                if w not in parent:
                    parent[w] = u
                    stk.append(w)
        # candidate labels
        t1 = time.time()
        C, dist, lbout = {}, {}, {}
        F, alive, choice = {}, {}, {}
        rng = random.Random(D)
        total, resident = 0, 0
        memory_budget = int(candidate_memory_mib * 1024 * 1024)
        if memory_budget <= 0 or probe_seconds < 0:
            raise ValueError("candidate_memory_mib must be positive; probe_seconds nonnegative")
        search_stats = []
        tasks = []
        for v in internal:
            if v in elim:
                continue
            br = _branches(t, lab, v, lid, rng)
            tasks.append((v, br, _off_path_bounds(t, lab, v, lid, dm)))
        task_bounds = {v: (br, off) for v, br, off in tasks}
        task_map = {task[0]: task for task in tasks}
        st["subtree_bound_updates"] = 0

        def generate(task, short=False):
            nonlocal total, resident
            v, br, off_path = task
            if subtree_feedback and not short:
                # completed child DP tables give stronger off-path constants
                off_path = dict(off_path)
                for w, labs, seqs in br:
                    if parent.get(w) != v or w not in F:
                        continue
                    indices = alive[w]
                    for s_ in labs:
                        bound = int(np.min(F[w][indices] - dist[w][indices, s_]))
                        if bound > off_path.get((w, s_), 0):
                            off_path[w, s_] = bound
                            st["subtree_bound_updates"] += 1
                task_bounds[v] = (br, off_path)
            available = memory_budget - resident
            if available <= 0:
                raise CapExceeded("total candidate-storage budget exceeded")
            local_deadline = min(deadline, time.time() + probe_seconds) if short else deadline
            before = len(search_stats)
            try:
                reg, rd = native.prefix_domain(
                    redl, metric.w, br, off_path, dm, D,
                    min(max_cands, 10000) if short else max_cands, local_deadline,
                    search_stats, Timeout, CapExceeded,
                    byte_cap=min(available, 16 * 1024 * 1024) if short else available)
            finally:
                for entry in search_stats[before:]:
                    entry.update(vertex=int(v), phase="probe" if short else "full")
            if not len(reg):
                return False
            C[v], dist[v] = reg, rd
            resident += reg.nbytes + rd.nbytes
            if parent[v] is not None:
                u, labs, seqs = next(b for b in br if b[0] == parent[v])
                out = _branch_lb(rd, labs, seqs, dm)
                for s_ in labs:
                    out = np.maximum(out, rd[:, s_] + off_path.get((u, s_), 0))
                lbout[v] = out
                resident += out.nbytes
            total += len(reg)
            st["cands"] = total
            st["candidate_bytes"] = resident
            if resident > memory_budget:
                raise CapExceeded("total candidate-storage budget exceeded")
            return True

        if probe_seconds:
            pending = []
            for task in tasks:
                try:
                    ok = generate(task, short=True)
                except (Timeout, CapExceeded):
                    if time.time() >= deadline:
                        raise Timeout()
                    # A probe that stops is discarded; partial domains never prove NO.
                    count = search_stats[-1].get("candidates", 0) if search_stats else 0
                    pending.append((count, task))
                    continue
                if not ok:
                    st.update(result="NO", reason="empty prefix domain", cands=total,
                              t_cands=time.time() - t1)
                    return st
            tasks = [task for _, task in sorted(pending, key=lambda x: x[0])]
        if not subtree_feedback:
            for task in tasks:
                if not generate(task):
                    st.update(result="NO", reason="empty prefix domain", cands=total,
                              t_cands=time.time() - t1)
                    return st
        st["cands"] = total
        st["t_cands"] = time.time() - t1
        # tree DP over the candidates (values capped at D + 1)
        t2 = time.time()
        st["dp_stats"] = []
        deferred_seconds = 0.0
        for u in reversed(order):
            if u in lab or u in elim:
                continue
            if u not in C:
                began = time.time()
                try:
                    ok = generate(task_map[u])
                finally:
                    elapsed = time.time() - began
                    deferred_seconds += elapsed
                    st["t_cands"] += elapsed
                if not ok:
                    st.update(result="NO", reason="empty subtree-informed prefix domain",
                              t_dp=time.time() - t2 - deferred_seconds)
                    return st
            f = np.zeros(len(C[u]), dtype=np.int64)
            kids = [w for w in t.adj[u] if parent.get(w) == u]
            outside = lbout.get(u, np.zeros(len(C[u]), dtype=np.int64))
            child_lb = {}
            for w in kids:
                if w in lab:
                    child_lb[w] = dist[u][:, lid[lab[w]]]
                elif w in elim:
                    x, y = (lid[lab[z]] for z in t.adj[w] if z in lab)
                    child_lb[w] = (dm[x][y] + dist[u][:, x] + dist[u][:, y] + 1) // 2
                else:
                    br, off = task_bounds[u]
                    _, labs, seqs = next(b for b in br if b[0] == w)
                    bound = _branch_lb(dist[u], labs, seqs, dm)
                    for s_ in labs:
                        bound = np.maximum(bound, dist[u][:, s_] + off.get((w, s_), 0))
                    child_lb[w] = np.maximum(bound, int(F[w][alive[w]].min()))
            remaining = sum(child_lb.values(), np.zeros(len(C[u]), dtype=np.int64))
            for w in sorted(kids, key=lambda w: (w in elim, w not in lab)):
                remaining -= child_lb[w]   # leaves first, cherries last
                if w in lab:
                    f += dist[u][:, lid[lab[w]]]
                    continue
                if w in elim:
                    x, y = (lid[lab[z]] for z in t.adj[w] if z in lab)
                    da, db, dxy = dist[u][:, x], dist[u][:, y], dm[x][y]
                    lo = (dxy + da + db + 1) // 2                       # triangle inequality
                    hi = np.minimum(da + db, dxy + np.minimum(da, db))  # centre = parent, x or y
                    val = hi.copy()
                    slack = D - f - outside - remaining
                    need = np.flatnonzero((lo < hi) & (lo <= slack))
                    if len(need):
                        val[need] = native.median_batch(redl[x], redl[y], C[u][need], metric.w,
                                                        deadline, Timeout, CapExceeded)
                    late = lo > slack
                    val[late] = lo[late]                               # already over budget
                    st["cherry_exact"] += len(need)
                    f += val
                    continue
                limits = D - f - outside - remaining
                rows = np.flatnonzero(child_lb[w] <= limits)
                A = alive[w]
                g, arg = native.min_plus(metric, C[u], rows, C[w][A], F[w][A], D, deadline,
                                         Timeout, CapExceeded, limits=limits[rows],
                                         stats=st["dp_stats"])
                full = np.full(len(C[u]), D + 1, dtype=np.int64)
                full[rows] = g
                f += full
                ch = np.full(len(C[u]), -1, dtype=np.int64)
                ok = arg >= 0
                ch[rows[ok]] = A[arg[ok]]
                choice[w] = ch
            f = np.minimum(f, D + 1)
            F[u] = f
            lim = f + lbout[u] if (u in lbout) else f
            alive[u] = np.flatnonzero(lim <= D)
            if not len(alive[u]):
                st.update(result="NO", reason="DP (no feasible label)",
                          t_dp=time.time() - t2 - deferred_seconds)
                return st
            if time.time() > deadline:
                raise Timeout()
        r = int(F[root].argmin())
        val = int(F[root][r])
        st["t_dp"] = time.time() - t2 - deferred_seconds
        if val > D:
            st.update(result="NO", reason="DP")
            return st
        st.update(result="YES", reason="DP", cost_red=val)
        sel = {root: r}
        for u in order:
            if u not in lab and u != root and u not in elim:
                sel[u] = int(choice[u][sel[parent[u]]])
        for w in elim:
            x, y = (lid[lab[z]] for z in t.adj[w] if z in lab)
            pu = parent[w]
            explicit[w] = np.array(median3_seq(redl[x], redl[y], C[pu][sel[pu]], metric.w),
                                   dtype=np.int16)
        full_lab = _expand_solution(t, lab, C, sel, red, blocks, explicit=explicit)
        _accept(st, t, lab, full_lab, val)
    except Timeout:
        st.update(result="TIMEOUT", reason="time limit")
    except CapExceeded as e:
        st.update(result="CAP", reason=str(e) or "cap")
    finally:
        st["t_total"] = time.time() - t0
        if metric is not None and "search_stats" in locals():
            st["search_stats"] = search_stats
    return st


def _accept(st, t, lab, full_lab, val):
    """Check a YES labeling on the kernel tree; a failed check is an error, not a NO."""
    c = sum(ulam(full_lab[u], full_lab[v]) for u, v in t.edges())
    if not (c <= val and all(full_lab[v] == lab[v] for v in lab)):
        raise AssertionError(f"exact search returned YES with an unverified labeling ({c} > {val})")
    st["cost"] = c
    st["labels_used"] = sorted(set(full_lab.values()))


def _contract_by(L, sym_id):
    out, last = [], None
    for s in L:
        b = sym_id[s]
        if b != last:
            out.append(b)
            last = b
    return out


def _expand_solution(t, lab, C, sel, red, blocks, explicit=None):
    """Rebuild a contracted labeling in the original alphabet."""
    explicit = explicit or {}
    out = {}
    for v in t.adj:
        if v in lab:
            seq = red[lab[v]]
        elif v in explicit:
            seq = explicit[v]
        else:
            seq = C[v][sel[v]]
        out[v] = tuple(s for b in seq for s in blocks[int(b)])
    return out


def label_tree_from(t, leaf_seq, cands):
    """Best labeling of t whose internal labels come from cands (explicit gene orders).

    Leaves keep leaf_seq. Exact tree DP over the candidates. Used to carry a solution
    found on the kernel back to the full tree: every kernel rule labels the removed
    vertices with a leaf label or a neighbouring solution label, so the leaf labels and
    the kernel solution's labels always contain an optimal full labeling."""
    cands = list(dict.fromkeys(list(cands) + list(leaf_seq.values())))
    k = len(cands)
    Dm = np.zeros((k, k), dtype=np.int64)
    for i in range(k):
        for j in range(i + 1, k):
            Dm[i, j] = Dm[j, i] = ulam(cands[i], cands[j])
    internal = [v for v in t.adj if v not in leaf_seq]
    if not internal:
        return sum(ulam(leaf_seq[u], leaf_seq[v]) for u, v in t.edges()), dict(leaf_seq)
    root = internal[0]
    parent, order, stack = {root: None}, [], [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in t.adj[u]:
            if w not in parent:
                parent[w] = u
                stack.append(w)
    col = {}
    F, choice = {}, {}
    for u in reversed(order):
        if u in leaf_seq:
            continue
        f = np.zeros(k, dtype=np.int64)
        for w in t.adj[u]:
            if parent.get(w) != u:
                continue
            if w in leaf_seq:
                L = leaf_seq[w]
                if L not in col:
                    col[L] = np.array([ulam(c, L) for c in cands], dtype=np.int64)
                f += col[L]
            else:
                M = Dm + F[w][None, :]
                choice[w] = M.argmin(axis=1)
                f += M.min(axis=1)
        F[u] = f
    sel = {root: int(F[root].argmin())}
    for u in order:
        if u in leaf_seq or u == root:
            continue
        sel[u] = int(choice[u][sel[parent[u]]])
    lab = dict(leaf_seq)
    for u, i in sel.items():
        lab[u] = cands[i]
    return sum(ulam(lab[u], lab[v]) for u, v in t.edges()), lab
