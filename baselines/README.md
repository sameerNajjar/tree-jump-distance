# Baselines

Other ancestral-reconstruction methods, run on the same fixed trees as our solver and
scored in Ulam cost. Each method optimises its own distance (reversals, DCJ or SCJ), so
these tables compare Ulam cost only; they do not rank the methods on their own measures.

| File | Contents |
|---|---|
| `run_atgc.py` | MGR 2.01 and SCJ on ATGC full-tree instances |
| `organelle.py` | ours, MGR 2.01, SCJ and saved SPP-DCJ ancestors on four organelle datasets |
| `mgr_input.py` | signed gene orders from `atgc.pty`, and the MGR fixed-topology file |
| `genomes.py` | adjacencies, DCJ and SCJ distances, exact SCJ small parsimony, linear orders |
| `mgr_fixed_topology.patch` | one-line fix to MGR 2.01's fixed-topology mode |
| `spp_dcj_construct_ilp.patch` | one-line fix to SPP-DCJ 2.0 for circular genomes |

## Software

- **MGR 2.01** (Bourque and Pevzner, Genome Research 12(1):26–36, 2002), from
  <https://mathweb.ucsd.edu/~gptesler/grimm/DIST/>. It is GPLv2 and is not included
  here. Apply the patch from the folder that holds `MGR-2.01/`, then build:

  ```
  patch -p0 < mgr_fixed_topology.patch
  cd MGR-2.01 && make
  ```

  Without the patch, `-t` numbers merged vertices off by one and MGR returns a
  different tree from the one it was given. `run_atgc.py` compares MGR's tree with the
  input tree split by split and rejects any run whose tree differs. On Windows, MGR was
  built with MinGW gcc 6.3 and a 512 MiB stack (add `-Wl,--stack,536870912` to the
  link line); with the default stack it crashes on 1,505 genes.
- **SCJ small parsimony** (Feijão and Meidanis 2011) is exact and needs nothing
  external (`genomes.py`). Its ancestors can fall into several pieces; they are joined
  into one order for scoring.
- **SPP-DCJ 2.0** (Bohnenkämper, Stoye and Doerr) is not run from here. The organelle
  results came from `spp-dcj ilp -t -a 1`, warm-started with `spp-dcj heuristic`, and
  solved with HiGHS 1.15.1 for 1,200 s (Gurobi needs a licence). SPP-DCJ 2.0 raises a
  `NameError` on circular genomes; `spp_dcj_construct_ilp.patch` fixes it. `organelle.py
  --saved` rescores any JSON file with `signed_genomes` (vertex → list of
  `[is_circular, genes]`).

## Data (not redistributed)

- **ATGC.** The 20 clusters in [../atgc25_clusters.tsv](../atgc25_clusters.tsv). Put
  each cluster's `atgc.pty` in `<atgc25>/ATGCnnnn/`.
- **Organelle.** Four permutation datasets:

  | Dataset | Genomes × genes | Source | SHA-256 of the file used |
  |---|---|---|---|
  | campanulaceae | 13 × 105 | Cosner et al. 2000; MGR 2.01 `data/allplants_data.txt` | `5afd4559e9c23d019b6b803b3ce0fcdf0f5ec8a2b5374ce3fe7b55f893077231` |
  | metazoa11 | 11 × 36 | Blanchette, Kunisawa and Sankoff 1999; MGR 2.01 `data/metazoan_data.txt` | `1a5324bf156af1c2a4d585f25615df386c864fed6afbb9f4d0a46778ddcf6bdc` |
  | meta28 | 28 × 15 | Larget et al., Mol. Biol. Evol. 22(3):486–495, 2005, supplementary file `msi032supp1.txt` | `9ded3928f3b9adde4c32bf198de28ee9b50375818c9f015e0ea09007da2de6e1` |
  | meta87 | 87 × 37 | the same article, `msi032supp3.txt` | `52415e9bfe547f02687a8f555d531544d494ee58f33b21515ef1acc42c676e86` |

  `organelle.py` expects them in one folder as
  `campanulaceae_cosner2000_13x105.txt`, `metazoa_blanchette1999_11x36.txt`,
  `meta28_larget2005_28x15.txt` and `meta87_larget2005_87x37.txt`.

## Running

ATGC (after `scripts/prepare_atgc.py` and `scripts/run_all.py`; see the
main README):

```
python baselines/run_atgc.py --data <atgc25> --inputs inputs --records results/full \
    --mgr <path to patched MGR> --out results/baselines \
    ATGC0835 ATGC0327 ATGC0868 ATGC0625 ATGC0737 ATGC0421 ATGC0874
```

Organelle (the tree is the one MGR 2.01 infers, with `-C -c -H2`):

```
python baselines/organelle.py meta28 --data <benchmarks> --write-mgr-input meta28_mgr_input.txt
MGR -f meta28_mgr_input.txt -C -c -H2 > meta28_mgr.out
python baselines/organelle.py meta28 --data <benchmarks> --mgr-out meta28_mgr.out \
    --exact-seconds 30 [--saved meta28_sppdcj.json] --out meta28.json
```

For campanulaceae and metazoa11, run MGR on the data file itself.

## Scoring

- Leaves get exactly the orders our solver uses.
- **ATGC.** Each gene takes its strand from `atgc.pty`, and leaves are single linear
  chromosomes. A baseline ancestor is dropped to an unsigned order and given the
  rotation and direction closest to the reference genome, the rule the leaves were
  normalised with. This can only lower the baselines' cost.
- **Organelle.** Genomes are circular and are written from gene 1 on its forward
  strand. An ancestor's chromosome holding gene 1 comes first, from +1, then the other
  chromosomes by smallest gene.
- The Ulam cost is summed over the edges of the fixed tree.

## Results on the reference machine

ATGC, the seven trees with at most 10 genomes:

| Cluster | Genomes | Genes | Ours (lower–upper) | MGR | SCJ |
|---|---:|---:|---|---:|---:|
| ATGC0835 | 5 | 1505 | 26 (optimal) | 26 | 26 |
| ATGC0327 | 7 | 504 | 64 (optimal) | 64 | 64 |
| ATGC0868 | 5 | 1558 | 581–601 | 630 | 4141 |
| ATGC0625 | 10 | 1350 | 1774–1989 | 4606 | 6361 |
| ATGC0737 | 7 | 1490 | 1441–1660 | 4045 | 5234 |
| ATGC0421 | 8 | 2050 | 4138–5607 | 10605 | 10151 |
| ATGC0874 | 5 | 2732 | 179–188 | 4299 | 4705 |

On ATGC0874, MGR's 241 reversals were confirmed by recomputing the DCJ distance of
its ancestors. Its ancestors are close in reversals and far in gene moves.

Organelle, with 30 s of exact search for ours:

| Dataset | Ours (lower–upper) | MGR | SPP-DCJ | SCJ |
|---|---|---:|---:|---:|
| campanulaceae | 208–250 | 343 | 283 | 853 |
| metazoa11 | 131–156 | 286 | 276 | 270 |
| meta28 | 82–96 | 166 | 252 | 216 |
| meta87 | 351–418 | 841 | did not finish | 1520 |

The MGR inputs `organelle.py --write-mgr-input` writes for meta28 and meta87 are
byte-identical to the ones these MGR runs used.
