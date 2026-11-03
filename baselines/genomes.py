"""Genome utilities shared by all baselines.

A genome is a list of chromosomes; a chromosome is (is_circular, [signed gene ids]).
Extremities: gene g read forward goes from (g,'t') to (g,'h').
"""
import bisect
from collections import defaultdict


def right_ext(x):
    return (abs(x), 'h') if x > 0 else (abs(x), 't')


def left_ext(x):
    return (abs(x), 't') if x > 0 else (abs(x), 'h')


def norm_adj(a, b):
    return (a, b) if a <= b else (b, a)


def adjacencies(genome):
    A = set()
    for circ, genes in genome:
        for i in range(len(genes) - 1):
            A.add(norm_adj(right_ext(genes[i]), left_ext(genes[i + 1])))
        if circ and genes:
            A.add(norm_adj(right_ext(genes[-1]), left_ext(genes[0])))
    return A


def circular(genes):
    return [(True, list(genes))]


def genome_from_adjacencies(A, n):
    """Rebuild chromosomes (paths and cycles) from an adjacency set over genes 1..n."""
    partner = {}
    for a, b in A:
        assert a not in partner and b not in partner, ('extremity used twice', a, b)
        partner[a] = b
        partner[b] = a
    seen, chroms = set(), []

    def walk(g, sign):
        out = []
        while True:
            out.append(sign * g)
            seen.add(g)
            nxt = partner.get(right_ext(sign * g))
            if nxt is None:
                return out, False
            h, end = nxt
            s = 1 if end == 't' else -1
            if h in seen:
                return out, True
            g, sign = h, s

    # linear chromosomes: start at telomeres (extremities without partner)
    for g in range(1, n + 1):
        if g in seen:
            continue
        if (g, 't') not in partner:
            chroms.append((False, walk(g, 1)[0]))
        elif (g, 'h') not in partner:
            chroms.append((False, walk(g, -1)[0]))
    for g in range(1, n + 1):
        if g not in seen:
            genes, closed = walk(g, 1)
            chroms.append((True, genes))
    assert sorted(abs(x) for _, c in chroms for x in c) == list(range(1, n + 1))
    return chroms


def dcj_distance(A, B, n):
    """DCJ distance N - (C + I/2) between two genomes on genes 1..n given as adjacency sets."""
    pa, pb = {}, {}
    for x, y in A:
        pa[x] = y; pa[y] = x
    for x, y in B:
        pb[x] = y; pb[y] = x
    exts = [(g, e) for g in range(1, n + 1) for e in ('t', 'h')]
    seen, C, I = set(), 0, 0
    # paths start at an extremity that is a telomere in A or in B
    for start in exts:
        if start in seen:
            continue
        if start in pa and start in pb:
            continue
        # walk the path
        x, use_a = start, start in pa
        length = 0
        seen.add(x)
        while True:
            p = pa if use_a else pb
            if x not in p:
                break
            y = p[x]
            seen.add(y)
            length += 1
            x, use_a = y, not use_a
        # a walk with an even number of adjacency steps joins a telomere of A to a telomere of B:
        # an odd path in the adjacency graph
        if length % 2 == 0:
            I += 1
    for start in exts:
        if start in seen:
            continue
        x, use_a = start, True
        while x not in seen:
            seen.add(x)
            y = (pa if use_a else pb)[x]
            seen.add(y)
            x = (pb if use_a else pa)[y]
        C += 1
    return n - (C + I // 2)


def scj_distance(A, B):
    return len(A ^ B)


def linear_order(genome, n):
    """Unsigned linear gene order used for the Ulam objective.

    Leaves are circular genomes written from gene 1 on its forward strand. A single circular chromosome is
    rotated and, if needed, read on the other strand so that it starts with +1. A genome with several
    chromosomes or fragments is completed by a fixed rule: the chromosome containing gene 1 first (oriented
    with +1, cut before gene 1 if circular), then the others in order of their smallest gene, each oriented
    so that its smallest gene is forward."""
    def orient(c, anchor):
        circ, genes = c[0], list(c[1])          # copy: never modify the genome being read
        if -anchor in genes:
            genes = [-x for x in reversed(genes)]
        if circ:
            i = genes.index(anchor)
            genes = genes[i:] + genes[:i]
        return genes
    first = next(c for c in genome if 1 in c[1] or -1 in c[1])
    out = orient(first, 1)
    rest = sorted((c for c in genome if c is not first), key=lambda c: min(abs(x) for x in c[1]))
    for c in rest:
        out += orient(c, min(abs(x) for x in c[1]))
    assert sorted(abs(x) for x in out) == list(range(1, n + 1))
    return tuple(abs(x) for x in out)


def ulam(a, b):
    pos = {g: i for i, g in enumerate(b)}
    tails = []
    for x in (pos[g] for g in a):
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
    return len(a) - len(tails)


def scj_small_parsimony(edges, root, leaf_adj):
    """Exact SCJ small parsimony (Feijao & Meidanis 2011): Sankoff on presence/absence of every adjacency seen at a
    leaf, ties broken towards absence, which makes every ancestral adjacency set a valid genome."""
    children = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
    order, stack = [], [root]
    while stack:
        u = stack.pop(); order.append(u); stack.extend(children[u])
    cand = set().union(*leaf_adj.values())
    anc = {u: set() for u in order if u not in leaf_adj}
    for a in cand:
        cost = {}
        for u in reversed(order):
            if u in leaf_adj:
                s = 1 if a in leaf_adj[u] else 0
                cost[u] = [0 if s == 0 else 10**9, 0 if s == 1 else 10**9]
            else:
                c0 = c1 = 0
                for w in children[u]:
                    c0 += min(cost[w][0], cost[w][1] + 1)
                    c1 += min(cost[w][1], cost[w][0] + 1)
                cost[u] = [c0, c1]
        state = {root: 0 if cost[root][0] <= cost[root][1] else 1}
        for u in order:
            for w in children[u]:
                s = state[u]
                if w in leaf_adj:
                    state[w] = 1 if a in leaf_adj[w] else 0
                    continue
                stay, flip = cost[w][s], cost[w][1 - s] + 1
                if stay < flip or (stay == flip and s == 0):
                    state[w] = s
                elif stay == flip:
                    state[w] = 0
                else:
                    state[w] = 1 - s
        for u in anc:
            if state[u]:
                anc[u].add(a)
    return anc
