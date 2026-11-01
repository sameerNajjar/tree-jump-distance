"""Unrooted trees: Newick parsing, normalisation and induced subtrees.

Vertices are integers. Leaves carry genome names in Tree.name. Branch lengths and
internal-node labels in the Newick input are ignored.
"""
import collections
import re


class Tree:
    def __init__(self):
        self.adj = collections.defaultdict(set)
        self.name = {}                         # leaf vertex -> genome name
        self._next = 0

    def new(self):
        v = self._next
        self._next += 1
        self.adj[v]
        return v

    def edge(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)

    def copy(self):
        t = Tree()
        for u, nb in self.adj.items():
            t.adj[u] = set(nb)
        t.name = dict(self.name)
        t._next = self._next
        return t

    def leaves(self):
        return [v for v in self.adj if len(self.adj[v]) <= 1]

    def internal(self):
        return [v for v in self.adj if len(self.adj[v]) > 1]

    def edges(self):
        return [(u, v) for u in self.adj for v in self.adj[u] if u < v]

    def remove(self, v):
        for u in self.adj[v]:
            self.adj[u].discard(v)
        del self.adj[v]
        self.name.pop(v, None)

    def dfs_leaf_order(self, rng=None):
        """Leaves in depth-first order (children shuffled when rng is given)."""
        if not self.adj:
            return []
        start = next(iter(self.internal()), next(iter(self.adj)))
        order, seen, stack = [], {start}, [start]
        while stack:
            u = stack.pop()
            if len(self.adj[u]) <= 1:
                order.append(u)
            nb = sorted(self.adj[u], reverse=True)
            if rng is not None:
                rng.shuffle(nb)
            for w in nb:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return order


def parse_newick(text):
    s = re.sub(r"\[[^\]]*\]", "", text.strip()).rstrip(";").strip()
    t = Tree()
    i = 0

    def node():
        nonlocal i
        v = t.new()
        if s[i] == "(":
            i += 1
            while True:
                c = node()
                t.edge(v, c)
                if s[i] == ",":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    break
        j = i
        while i < len(s) and s[i] not in ",()":
            i += 1
        tok = s[j:i].split(":")[0].strip()
        if tok and len(t.adj[v]) == 0:
            t.name[v] = tok
        return v

    node()
    return t


def normalise(t):
    """Drop unnamed leaves and suppress unnamed degree-2 vertices (both cost-preserving)."""
    t = t.copy()
    changed = True
    while changed:
        changed = False
        for v in list(t.adj):
            if v not in t.adj:
                continue
            d = len(t.adj[v])
            if v not in t.name and d <= 1 and len(t.adj) > 1:
                t.remove(v)
                changed = True
            elif v not in t.name and d == 2:
                a, b = t.adj[v]
                t.remove(v)
                t.edge(a, b)
                changed = True
    return t


def induced(t, names):
    """The subtree spanned by the named leaves, normalised."""
    keep = set(names)
    t = t.copy()
    for v in list(t.adj):
        if v in t.name and t.name[v] not in keep:
            t.remove(v)
    return normalise(t)
