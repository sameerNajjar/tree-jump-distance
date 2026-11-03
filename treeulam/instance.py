"""Instance files (schema tree-ulam-instance/1).

An instance is one tree with a gene order at every leaf:

  schema      "tree-ulam-instance/1"
  id          "<dataset>/F" for an ATGC cluster, "<dataset>/<file stem>" from make_instance.py
  dataset     source dataset name
  family      "full" (an ATGC cluster's whole tree) or "user" (make_instance.py)
  n_genes     n, the alphabet size; leaf orders are permutations of 1..n
  gene_ids    the ATGC cluster id of symbol 1..n
  window      always null (kept so that F.json files match inputs.sha256)
  subset      always null (likewise)
  tree        {"vertices": [...], "edges": [[u, v], ...]}; leaves are genome names,
              internal vertices "i<k>"; unrooted, no degree-2 vertices, no lengths
  leaves      genome -> permutation of 1..n

Labelings (witnesses) use the same vertex names and 1-based permutations.
"""
import hashlib
import json
from pathlib import Path

from .tree import Tree

SCHEMA = "tree-ulam-instance/1"


def tree_record(t):
    """Vertices in increasing id order (leaves by name, internal as i<id>) and edges."""
    name = {v: t.name.get(v, f"i{v}") for v in sorted(t.adj)}
    return dict(vertices=[name[v] for v in sorted(t.adj)],
                edges=sorted([name[u], name[v]] for u, v in t.edges()))


def make_instance(iid, dataset, family, gene_ids, tree, leaves, source=None):
    """leaves: genome -> 0-based permutation of range(len(gene_ids))."""
    n = len(gene_ids)
    rec = dict(schema=SCHEMA, id=iid, dataset=dataset, family=family, n_genes=n,
               gene_ids=list(gene_ids), window=None, subset=None, tree=tree_record(tree),
               leaves={g: [s + 1 for s in leaves[g]] for g in sorted(leaves)})
    if source:
        rec["source"] = source
    validate(rec)
    return rec


def validate(rec):
    if rec.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} record")
    n = rec["n_genes"]
    if len(rec["gene_ids"]) != n or len(set(rec["gene_ids"])) != n:
        raise ValueError("gene_ids must list n distinct identifiers")
    verts = rec["tree"]["vertices"]
    if len(set(verts)) != len(verts):
        raise ValueError("repeated vertex name")
    vs = set(verts)
    deg = {v: 0 for v in verts}
    for u, v in rec["tree"]["edges"]:
        if u not in vs or v not in vs or u == v:
            raise ValueError(f"bad edge {u}-{v}")
        deg[u] += 1
        deg[v] += 1
    if len(rec["tree"]["edges"]) != len(verts) - 1:
        raise ValueError("a tree on V vertices has V - 1 edges")
    leaves = {v for v in verts if deg[v] <= 1}
    if leaves != set(rec["leaves"]):
        raise ValueError("the labelled vertices must be exactly the leaves")
    if any(deg[v] == 2 for v in verts):
        raise ValueError("degree-2 vertices must be suppressed")
    for g, p in rec["leaves"].items():
        if sorted(p) != list(range(1, n + 1)):
            raise ValueError(f"leaf {g} is not a permutation of 1..n")
    # connectivity
    adj = {v: [] for v in verts}
    for u, v in rec["tree"]["edges"]:
        adj[u].append(v)
        adj[v].append(u)
    seen, stack = {verts[0]}, [verts[0]]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    if len(seen) != len(verts):
        raise ValueError("the tree is not connected")


class Instance:
    """A loaded instance: Tree with integer vertices, leaf orders as 0-based tuples."""

    def __init__(self, rec):
        validate(rec)
        self.rec = rec
        self.id = rec["id"]
        self.n = rec["n_genes"]
        self.vertices = list(rec["tree"]["vertices"])
        self.vid = {v: i for i, v in enumerate(self.vertices)}
        t = Tree()
        for v in self.vertices:
            t.new()
        for u, v in rec["tree"]["edges"]:
            t.edge(self.vid[u], self.vid[v])
        for g in rec["leaves"]:
            t.name[self.vid[g]] = g
        self.tree = t
        self.leaf_seq = {self.vid[g]: tuple(s - 1 for s in p) for g, p in rec["leaves"].items()}
        self.distinct_orders = len(set(self.leaf_seq.values()))

    def named(self, labeling):
        """vertex id -> 0-based tuple  ==>  vertex name -> 1-based list."""
        return {self.vertices[v]: [s + 1 for s in seq] for v, seq in sorted(labeling.items())}


def load(path):
    return Instance(json.loads(Path(path).read_text(encoding="utf-8")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
