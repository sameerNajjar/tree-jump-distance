# The Tree Jump Distance: Hardness, Parameterized Algorithms, and Approximation

Code for the paper of this title by Sameer Najjar: approximation algorithms, lower
bounds and an exact FPT solver for the Tree Ulam Median problem. It runs on your own
tree and gene orders, and includes the scripts that rebuild and rerun the
experiments on two ATGC bacterial datasets.

**Problem.** The input is an unrooted tree whose leaves carry permutations of the same
n genes. The task is to label every internal vertex with a permutation so that the
sum of Ulam distances over the edges is smallest. The Ulam distance is
d(p, q) = n − LCS(p, q). Gene orders are unsigned and linear.


License: MIT (see `LICENSE`). Requirements: Python ≥ 3.9 and NumPy (`requirements.txt`).

## Quick start

```
pip install -r requirements.txt
python scripts/make_instance.py examples/example.tre examples/example.txt results/example.json
python scripts/solve.py results/example.json results/solved/example.json --exact-seconds 30
python scripts/check.py results/solved
```

- The commands run on the shipped example: 6 genomes and 8 genes, in `examples/`.
- `make_instance.py` turns the tree and the gene orders into an instance file.
- `solve.py` runs every method (see [Methods](#methods-treeulamsolvepy)) and writes a
  result record.
- `check.py` checks that record independently.
- The example's optimum is 4. Both lower bounds prove it, and so does the exact search.
- With the native libraries for your platform, this takes a few seconds (see Platforms below).

## Your own data

You need two text files.

**1. A tree** in Newick format, for example `((A,B),(C,(D,E)),F);`.

- Its leaves are your genomes.
- Branch lengths and internal labels are ignored, so the tree is treated as
  unrooted.
- Leaves without a gene order are dropped.

**2. Gene orders** in GRIMM-style text:

```
>A
g1 g2 g3 g4 g5 $
>B
g1 g3 g2 g4 g5 $
```

- A line `>name` starts a genome. Its name must match a tree leaf.
- The following lines list the genes in order, separated by spaces or commas.
- `$` and `@` (end of chromosome) are dropped, so the chromosomes of a genome are
  joined in file order.
- `#` starts a comment.
- Gene names are arbitrary words.
- The model is unsigned, so a leading `+` or `-` is removed (the script reports how
  many).

Then run:

```
python scripts/make_instance.py my_tree.nwk my_orders.txt my_instance.json
python scripts/solve.py my_instance.json my_result.json --exact-seconds 120
python scripts/check.py .
```

`make_instance.py` options:

| Option | Effect |
|---|---|
| (default) | Every genome must contain the same genes, each exactly once. Otherwise the script stops and names the first genome that differs. |
| `--core` | Keep only the genes present exactly once in every genome and drop the rest. Use this for real genomes with gene gains, losses or duplications. |
| `--linearise` | For circular chromosomes: rotate, and possibly reverse, each genome to the order closest to the reference in Ulam distance, as done for the ATGC data. Without it, each order is used exactly as written. |
| `--reference NAME` | Genome whose order defines the gene numbering 1..n (default: the first genome in the file). It also serves as the target of `--linearise`. |
| `--dataset NAME` | Prefix of the instance id (default `user`). |

**Reading the result** (`my_result.json`):

| Field | Meaning |
|---|---|
| `upper` | Cost of the best labeling found. The labeling is in `best.labels`, which gives a gene order for every internal vertex. |
| `lower` | Proven lower bound. |
| `status` | `OPTIMAL` when lower = upper; `certificate` says which bound or exact NO proves it. `OPEN` otherwise. |
| `algorithms` | Cost, time and labeling of each method. |
| `exact.calls` | Every exact decision, with its answer (YES, NO, TIMEOUT or CAP). |

In the labels, gene i is `gene_ids[i-1]` of the instance file.

**Limits.**

- The polynomial methods are fast on hundreds of genes and genomes.
- The time of the exact search grows quickly with the optimal cost. Our experiments
  settled most instances with a cost up to about 80 within 120 s; with a few hundred
  genes it usually stops on TIMEOUT. A TIMEOUT is never reported as NO, and the
  record then holds the best labeling and the proven lower bound.
- The 7/4 guarantee applies to binary trees; on other trees the method still returns
  a valid labeling.
- For many instances, put the instance files in one folder and use
  `scripts/run_all.py` (parallel, resumable) and `scripts/summarize.py`.

## Contents

| Path | Purpose |
|---|---|
| `treeulam/distance.py` | weighted Ulam distance (LIS, bit-parallel LCS) |
| `treeulam/blocks.py` | common-block contraction, label pool |
| `treeulam/median.py` | exact weighted three-median (3D dynamic program) |
| `treeulam/tree.py` | Newick parsing, normalisation, induced subtrees |
| `treeulam/approx.py` | lifting, 7/4 algorithm, local search, contour bound |
| `treeulam/packing.py` | path/tripod packing bound with witness |
| `treeulam/exact.py` | exact decision procedure (kernel + weighted search), solution lifting |
| `treeulam/native.py` | ctypes interface to `native/` |
| `treeulam/atgc.py` | ATGC files → permutations and instances |
| `treeulam/instance.py` | instance schema, loading, validation |
| `treeulam/solve.py` | all methods on one instance → checked result record |
| `native/` | C++ sources of the two native libraries, with Windows x64 builds |
| `scripts/make_instance.py` | your own Newick tree + gene orders → instance file |
| `scripts/prepare_atgc.py` | build `inputs/<dataset>/` from the raw ATGC files and check it against `inputs.sha256` |
| `scripts/solve.py` | solve one instance |
| `scripts/run_all.py` | parallel campaign with resume |
| `scripts/check.py` | independent checker (standard library only) |
| `scripts/summarize.py` | CSV and Markdown tables |
| `scripts/build_native.py` | rebuild the native libraries |
| `tests/test_small.py` | brute-force tests on small random instances |
| `tests/test_yes_path.py` | exact search at D = OPT on certified optima; every answer must be YES with cost OPT |
| `examples/` | a small example tree and gene-order file |
| `inputs.sha256` | checksums of the 608 ATGC input files used in the experiments; the inputs themselves are not in the repository |

**Platforms.** Tested on 64-bit Windows with Python 3.13.15 and NumPy 2.5.2, using
both the shipped DLLs and DLLs rebuilt by `scripts/build_native.py` with Visual Studio.
On Linux or macOS, run `scripts/build_native.py` first: it builds `.so` files with g++
or clang++. Those platforms have not been tested.

## Data

The raw data come from the ATGC database (Alignable Tight Genomic Clusters),
release atgc-25: <https://ftp.ncbi.nih.gov/pub/wolf/ATGC/atgc-25/>. Two clusters are
used. From each cluster folder you need the gene table (`.pty`) and the tree (`.tre`).

| Dataset | File used here | SHA-256 |
|---|---|---|
| ATGC0006 | `atgc.pty` | `b539520464e72e04ffd47b34cafc4ee52b9515f345cf5d99101ca41cce0343a3` |
| ATGC0006 | `atgc.iq.r.lab.tre` | `441ceccb49e7dacfec20e2d89ef157fb271b749fe99f44a2482a26b4b94e8338` |
| ATGC0093 | `ATGC0093.pty` (the cluster's gene table, renamed) | `2760a109c11eed8f94a87ae49b8f59bcab6f2b3dc8a2401b6cee4ee6bde54c1e` |
| ATGC0093 | `atgc0093 original.tre` (the cluster's tree, renamed) | `53791b070f06bd1f0c39ea01c7fefc2c85f3e851db593c79e506243ab927e175` |

Check a download against these hashes.

- `scripts/prepare_atgc.py` turns these four files into the experiments' instances
  under `inputs/` (see [Reproduction](#reproduction)).
- `inputs/<dataset>/dataset.json` records the file names and hashes, every genome's
  linearisation and the full permutations.
- The generated files are not part of the repository. `inputs.sha256` lists their
  checksums, and the script compares against it.

The MIT license covers the code. The instances are derived from the public ATGC
database, release atgc-25. If you use them, please cite ATGC:

- D. M. Kristensen, Y. I. Wolf and E. V. Koonin. ATGC database and ATGC-COGs: an
  updated resource for micro- and macro-evolutionary studies of prokaryotic genomes
  and protein family annotation. *Nucleic Acids Research* 45(D1):D210–D218, 2017.
  <https://doi.org/10.1093/nar/gkw934>
- P. S. Novichkov, I. Ratnere, Y. I. Wolf, E. V. Koonin and I. Dubchak. ATGC: a
  database of orthologous genes from closely related prokaryotic genomes and a
  research platform for microevolution of prokaryotes. *Nucleic Acids Research*
  37(Database issue):D448–D454, 2009. <https://doi.org/10.1093/nar/gkn684>

### From gene table to permutations (`treeulam/atgc.py`)

1. **Core genes.** The core genes are the `cls` clusters (column 8 of the `.pty`
   file) that occur exactly once in every genome of the gene table.
2. **Gene order of a genome.**
   - Contigs follow the order of the genome's `#` contig records.
   - Genes on a contig are sorted by start coordinate.
   - Non-core genes are dropped.
3. **Symbols.** The reference genome is the alphabetically first accession:

   | Dataset | Reference genome |
   |---|---|
   | ATGC0006 | GCF_000007265.1 |
   | ATGC0093 | GCF_000022385.1 |

   Its gene order defines symbols 1..n. `gene_ids[i]` is the cluster id of symbol i+1.
4. **Linearisation.** Chromosomes are circular. Each genome is cut at the rotation and
   read in the direction whose order is closest to the reference in Ulam distance.
   - The candidates are tried in this order: forward rotations 0..n−1, then reversed
     rotations.
   - On a tie, the first candidate wins.
   - The per-genome choice (rotation, reversed, distance) is in `dataset.json`.
5. **Tree.** The tree is built from the Newick file as follows:
   - Leaves are restricted to the genomes and unnamed leaves are dropped.
   - Degree-2 vertices (including the root) are suppressed.
   - Branch lengths are ignored.
   - Both trees are binary after this step.

| Dataset | Genomes | Core genes n | Distinct orders | Multi-contig genomes |
|---|---|---|---|---|
| ATGC0006 | 146 | 810 | 11 (136 genomes share one) | 8 |
| ATGC0093 | 21 | 918 | 15 | 4 |

### Instance families (`scripts/prepare_atgc.py`)

The same design is used for both datasets:

| Family | Id | Tree | Genes | Count (0006 / 0093) |
|---|---|---|---|---|
| F | `<ds>/F` | full tree | all n | 1 / 1 |
| W | `<ds>/W<w>/<start>` | full tree | window of w consecutive reference positions, w = 10, 20, 50 with step 5, 10, 25 | 273 / 310 |
| S | `<ds>/S<k>/seed<r>` | subtree induced by k genomes | all n | 12 / 9 |

- **Windows (W).**
  - Window starts are 1, 1+step, … .
  - A final window ending at position n is added when the step does not reach it.
  - Each genome keeps the window's genes in its own order. They are relabelled
    1..w in reference order.
- **Subtrees (S).**
  - For seed r, the genomes are shuffled with Python's `random.Random(r)`.
  - The first k of the shuffled list are kept: k = 8, 16, 32, 64 for ATGC0006 and
    k = 4, 8, 16 for ATGC0093.
  - The subsets are nested per seed.

Instance file (`tree-ulam-instance/1`, see `treeulam/instance.py`):

- the id, dataset and family;
- `n_genes` and `gene_ids`;
- `window` or `subset`;
- `tree` as vertex names (leaves = accessions, internal = `i<k>`) and edges;
- `leaves`: accession → permutation of 1..n.

## Methods (`treeulam/solve.py`)

On every instance, in this order:

1. **Trivial instances.** If all leaves carry the same order, the instance is
   TRIVIAL with cost 0.
2. **Polynomial methods.** Labels are computed on the alphabet contracted by the
   blocks common to all leaves; contraction preserves all distances.

   | Method | What it does |
   |---|---|
   | lifting | Best labeling whose internal labels are leaf labels (factor 2). |
   | 7/4 | Best labeling whose internal labels are leaf labels or exact medians of three distinct leaf labels (factor 7/4). |
   | local search | Applied after each of the two above. Replaces a degree-3 internal label by the exact median of its neighbours while that strictly lowers the cost. |

3. **Lower bounds.**
   - **Contour.** Take the best of 64 depth-first leaf orders (the first deterministic,
     the others seeded). The bound is ⌈cyclic tour length / 2⌉. The witness is the
     leaf order.
   - **Packing.** Choose edge-disjoint leaf paths, valued by the leaf distance, and leaf
     tripods, valued by the exact three-median. The bound is the maximum total value,
     found by a tree DP. The witness is the list of pieces.
4. **Exact search.** The exact decision procedure is called at D = U − 1, where U is the
   best cost so far. It runs even when the bounds already meet U, as a cross-check.
   - **YES.** The labeling is carried back to the full tree, checked, and becomes the
     new U. The procedure repeats at the new U − 1 while time remains.
   - **NO.** OPT ≥ U.
   - **TIMEOUT / CAP.** Nothing is concluded.
   - **Time allowance (defaults).** 120 s in total per window or subtree instance;
     3600 s per full tree.
   - **Settings.** At most 8,000,000 candidates, 1024 MiB of candidate storage, subtree
     feedback on.
5. **Status.**
   - OPTIMAL when the contour bound, the packing bound or an exact NO at U − 1 meets
     U. The record lists every certificate that holds.
   - Otherwise OPEN, with lower = max(contour, packing).

Every labeling is expanded to the original genes and its cost recomputed before it
is stored (`algorithms.<name>.labels`, internal vertices only).

### Exact decision procedure (`treeulam/exact.py`)

1. **Kernel rules.**
   - Collapse monochromatic sides.
   - Suppress degree-2 vertices.
   - Keep at most D + 1 equal sibling leaves.
   - Shorten uniform paths.
2. **Immediate NO.** Any of the following answers NO:
   - more than D + 1 distinct labels;
   - a label pair farther apart than D;
   - a minimum spanning tree of the labels longer than 2D;
   - a contour bound above D.
3. **Weighted contraction.** Every common block becomes one weighted symbol.
4. **Candidates.** Each internal vertex gets every permutation that passes exact
   branch lower bounds. They are enumerated by native prefix branch-and-bound.
   - Vertices with two leaf neighbours are solved by exact three-medians instead.
   - Bounds from finished child tables tighten later enumerations.
5. **Tree DP.** Values are capped at D + 1.
6. **YES labeling.** It is rebuilt in the original alphabet and checked. A failed
   check is an error, not an answer.

## Reproduction

Download the four raw files listed under Data, then:

```
python scripts/prepare_atgc.py ATGC0006 <atgc.pty> <atgc.iq.r.lab.tre> inputs/ATGC0006 --subset-sizes 8 16 32 64
python scripts/prepare_atgc.py ATGC0093 <ATGC0093 gene table .pty> <ATGC0093 tree .tre> inputs/ATGC0093 --subset-sizes 4 8 16
python tests/test_small.py
python scripts/run_all.py inputs/ATGC0006 inputs/ATGC0093 --out results/campaign --workers 4
python scripts/check.py results/campaign
python scripts/summarize.py results/campaign
python tests/test_yes_path.py results/campaign --out results/yes_path
```

- The first two commands build `inputs/` and compare it with `inputs.sha256`. They
  should report 287 of 287 and 321 of 321 matching files: the experiments' inputs,
  byte for byte. A mismatch means the downloaded files differ from the ones used
  here (compare their hashes with the table under Data).
- On the reference machine, the campaign takes about an hour with 4 workers; most of
  that is the two full-tree exact searches.

Result records follow the `tree-ulam-result/1` format and contain:

- the instance hash and settings;
- every algorithm's cost, time and labeling;
- both bounds with witnesses;
- every exact call (D, answer, reason, time);
- lower, upper, status and certificate.

Exact answers depend on the time allowance and the machine. TIMEOUT and CAP are
never counted as NO.

## Checking (`scripts/check.py`)

The checker uses only the standard library and shares no code with the solver.

| Item | What is checked |
|---|---|
| Instance | Re-read; its hash is compared. |
| Labelings | Every labeling cost is recomputed. |
| Contour | The leaf order uses each tree edge at most twice, and the value is recomputed. |
| Packing | The pieces are edge-disjoint. Path values are recomputed. Tripod values are recomputed by an independent three-median recurrence when the alphabet contracted by the three orders' common blocks has at most `--tripod-limit` symbols (default 150). Larger tripods are counted, and only their pair bound is checked. |
| Status and certificate | Must agree with the recomputed values. |

An exact NO cannot be re-derived independently. Certificates that rest on one alone
are counted.
