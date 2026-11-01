"""From ATGC files to gene-order permutations and instances.

Input files (ATGC database, Kristensen et al.):
  .pty  tab-separated gene table. A gene line has the gene id (col 0), "start..end"
        (col 1), strand (2), genome accession (3), contig (4) and the ATGC cluster
        ids cls (7) and cog (8). A line starting with "#" is a contig record
        (#, coordinates, strand, genome, contig).
  .tre  Newick tree whose leaves are genome accessions.

Rules (identical to the rc1 package, so earlier results stay comparable):
  core genes   cls clusters present exactly once in every genome of the .pty file;
  gene order   per genome, contigs in the order of their "#" records, genes on a
               contig by start coordinate, non-core genes dropped;
  reference    the alphabetically first genome; its order defines symbols 1..n;
  linearised   each genome takes the rotation and direction of its order closest to
               the reference in Ulam distance (first such: forward rotations
               0..n-1, then reversed rotations); the reference is the identity;
  tree         Newick leaves restricted to the genomes, unnamed leaves dropped,
               degree-2 vertices suppressed, branch lengths ignored.
"""
import collections
import random
import re

from .distance import lis
from .instance import make_instance, sha256
from .tree import induced, normalise, parse_newick


def read_pty(path, cluster="cls"):
    col = 7 if cluster == "cls" else 8
    genes = collections.defaultdict(list)
    contigs = collections.defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] == "#":
                contigs[f[3]].append(f[4])
                continue
            if len(f) <= col:
                continue
            m = re.match(r"(\d+)", f[1])
            genes[f[3]].append((f[4], int(m.group(1)) if m else 0, f[col]))
    return genes, contigs


def linearise(seq):
    """Rotation and direction of seq (symbols 0..n-1) closest to the identity."""
    best = None
    for rev in (False, True):
        u0 = seq[::-1] if rev else seq
        for k in range(len(u0)):
            u = u0[k:] + u0[:k]
            d = len(u) - lis(u)
            if best is None or d < best[0]:
                best = (d, rev, k, u)
    return best


def load_dataset(name, pty, tre, cluster="cls", log=print):
    genes, contigs = read_pty(pty, cluster)
    counts = {g: collections.Counter(c for _, _, c in v) for g, v in genes.items()}
    core = set.intersection(*({c for c, k in cnt.items() if k == 1} for cnt in counts.values()))
    raw = {}
    for g, v in genes.items():
        order = []
        for ct in contigs[g] or sorted({x[0] for x in v}):
            order += [c for _, _, c in sorted((x for x in v if x[0] == ct), key=lambda x: x[1])
                      if c in core]
        raw[g] = order
    names = sorted(raw)
    ref = names[0]
    gene_ids = raw[ref]
    sym = {c: i for i, c in enumerate(gene_ids)}
    perms, meta = {}, {}
    for g in names:
        d, rev, k, u = linearise([sym[c] for c in raw[g]])
        if sorted(u) != list(range(len(core))):
            raise ValueError(f"{g}: core gene order is not a permutation")
        perms[g] = tuple(u)
        meta[g] = dict(genes=len(genes[g]), contigs=len(contigs[g]) or len({x[0] for x in genes[g]}),
                       reversed=rev, rotation=k, ulam_to_reference=d)
    with open(tre, encoding="utf-8", errors="replace") as fh:
        phylo = normalise(induced(parse_newick(fh.read()), names))
    in_tree = set(phylo.name.values())
    missing = sorted(set(names) - in_tree)
    if missing:
        log(f"  genomes missing from the tree are dropped: {missing}")
    names = [g for g in names if g in in_tree]
    log(f"  {name}: {len(names)} genomes, {len(core)} single-copy core {cluster} clusters, "
        f"reference {ref}, {len(set(perms[g] for g in names))} distinct orders")
    return dict(name=name, names=names, perms={g: perms[g] for g in names}, n=len(core),
                gene_ids=gene_ids, reference=ref, meta={g: meta[g] for g in names},
                missing_from_tree=missing, phylo=phylo, cluster=cluster,
                pty=dict(path=str(pty), sha256=sha256(pty)),
                tre=dict(path=str(tre), sha256=sha256(tre)))


def window_starts(n, w, step):
    """0-based starts; the last window ends at position n even when step does not divide."""
    starts = list(range(0, n - w + 1, step))
    if starts[-1] != n - w:
        starts.append(n - w)
    return starts


def restrict(perms, lo, hi):
    """Every genome restricted to reference positions lo..hi-1, relabelled 0..hi-lo-1."""
    return {g: tuple(s - lo for s in p if lo <= s < hi) for g, p in perms.items()}


def build_instances(data, windows, subset_sizes, seeds):
    """The three families: F (full tree, all genes), W (windows, full tree), S (subtrees)."""
    name, n, phylo = data["name"], data["n"], data["phylo"]
    out = [make_instance(f"{name}/F", name, "full", data["gene_ids"], phylo, data["perms"])]
    for w, step in windows:
        for lo in window_starts(n, w, step):
            out.append(make_instance(
                f"{name}/W{w}/{lo + 1:04d}", name, "window", data["gene_ids"][lo:lo + w], phylo,
                restrict(data["perms"], lo, lo + w), window=dict(start=lo + 1, end=lo + w)))
    for seed in seeds:
        order = list(data["names"])
        random.Random(seed).shuffle(order)
        for k in subset_sizes:
            chosen = sorted(order[:k])
            out.append(make_instance(
                f"{name}/S{k}/seed{seed}", name, "subtree", data["gene_ids"], induced(phylo, chosen),
                {g: data["perms"][g] for g in chosen}, subset=dict(seed=seed, size=k)))
    return out
