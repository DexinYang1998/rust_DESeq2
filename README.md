# rust_deseq2

A from-scratch **Rust reimplementation of the DESeq2 differential-expression
workflow** (`DESeq()` + `results()`, Wald test), engineered to reproduce
Bioconductor DESeq2 numerically — not just the same method, but the same
algorithms, the same iteration paths, and the same edge-case rules — with
support for **multi-factor designs with covariates**
(`~ batch + age + condition`).

The crate is dependency-free (pure `std` Rust); a thin R wrapper
(`R/rustDESeq2.R`) drives the CLI and returns a `data.frame`.

## What it implements

Each stage is a faithful port of the corresponding DESeq2 1.42.x code path
(R/C++ sources cited in the module docs):

1. **Size factors** — median-of-ratios (`estimateSizeFactorsForMatrix`).
2. **Model matrix** — additive design formulas `~ a + b + cond` with factor
   columns (treatment coding, R's sorted level order, first level =
   reference) and numeric covariate columns, as `model.matrix` builds them.
   Interactions are not supported.
3. **Gene-wise dispersions** (`estimateDispersionsGeneEst`): rough+moments
   initial value; expected counts via the linear model (group designs) or the
   NB GLM; a single Cox-Reid adjusted fit by DESeq2's Armijo backtracking
   line search (`fitDisp`), with the no-increase revert, the iteration-1
   convention, and the coarse+fine grid fallback (`fitDispGrid`).
4. **Dispersion trend** — `parametricDispersionFit`'s iteratively re-gated
   Gamma GLM (identity link, R `glm.fit` semantics with step-halving), and
   the log-normal prior variance from R's `mad` (with its literal 1.4826
   constant) minus `trigamma((m-p)/2)`, floored at 0.25.
5. **MAP dispersions** (`estimateDispersionsMAP`) with DESeq2's start rule,
   grid fallback and the 2-SD dispersion-outlier carve-out.
6. **Wald GLM** — `fitBeta`'s ridge-penalised IRLS via the QR path
   (Householder factorization of the weighted, ridge-augmented design),
   `minmu=0.5`, the deviance convergence test computed with R's exact
   saddle-point `dnbinom`, the sandwich covariance, hat diagonals, and —
   for rows that do not converge — **a full port of R 4.3.3's L-BFGS-B**
   (`optim`'s `lbfgsb.c`, including the bounds-aware `ndeps=1e-3` numerical
   gradient and R's long-double sums), reproducing even the genes whose
   estimates are optimizer-path dependent.
7. **Contrasts** as `results(dds, contrast=c(var, case, control))`: direct
   coefficient when `control` is the reference level, negated coefficient
   when `case` is, and the numeric-contrast path `c'β / sqrt(c'Σc)`
   otherwise; the all-zero-contrast rule (LFC=0, stat=0, p=1).
8. **Outlier handling** — Cook's distances from the robust method-of-moments
   dispersion (cells = distinct model-matrix rows), `qf(.99, p, m-p)` cutoff,
   trimmed-mean **replacement** for cells with ≥ 7 replicates followed by a
   refit of only the replaced genes against the stored trend/prior (as
   `refitWithoutOutliers`), maxCooks **p-value flagging** for the rest,
   including the two-level-design heuristic and the post-replacement
   flagging rules.
9. **Independent filtering** — the exact `results()` procedure with a port
   of R's `lowess` (delta/robustness iterations included) and `p.adjust`'s
   BH.

Ported special functions match R bitwise or to a few ulp: `pnorm` (Cody),
`lgamma` (nmath Chebyshev/`lgammacor`), `digamma`/`trigamma` (nmath
`dpsifn`), `dnbinom` (saddle-point: `dbinom_raw`, `bd0`, `stirlerr`),
`qf`, `lowess`, type-7 quantiles, BH. `cargo test` checks them against
R-generated reference values.

## Numerical agreement with DESeq2

Validated head-to-head against Bioconductor DESeq2 1.42.1 (R 4.3.3) with
`examples/parity_check.R`, over six design/contrast configurations
(single-factor balanced/unbalanced 400–20k genes, batch + 3-level condition,
batch + continuous covariate + condition; contrasts against the reference
level, against a non-reference level, and reversed):

* **Identical on every dataset:** DE call sets at any threshold, padj NA
  patterns (independent filtering + Cook's flagging), outlier replacement
  flags, dispersion-outlier flags, size factors and baseMeans (≤ 2e-15).
* **Median relative difference** across genes: `1e-11 – 1e-15` for every
  reported column (LFC, lfcSE, stat, pvalue, padj) — i.e. all printed
  digits.
* **Worst case:** a handful of genes per dataset (≈ 0.1%) whose IRLS does
  not converge or whose dispersion line search sits on an accept/reject
  boundary differ by `1e-6 – 1e-3` (relative). This is the reproducibility
  floor of DESeq2 itself: those genes' values change by the same magnitude
  when R is run against a different BLAS (e.g. `FLEXIBLAS=NETLIB` vs
  OpenBLAS), because they are optimizer-path dependent. rust_deseq2's
  isolated L-BFGS-B reproduces R's `optim` on identical inputs to ≤ 1e-10
  with a bit-identical objective.

For designs with residual df ≤ 3 (e.g. 3 paired samples), DESeq2 estimates
the dispersion prior variance with a seeded Monte-Carlo procedure; this is
reproduced exactly via ports of R's Mersenne-Twister/`set.seed`, `qnorm`
(AS 241), `rgamma`/`rchisq`, and a 1-D port of R's loess with
`surface="interpolate"` (the kd-tree + Hermite interpolation), giving the
bit-identical prior variance.

Caveats (documented divergences):

* If the parametric dispersion trend fails, DESeq2 switches to a local
  (locfit) fit, which is not implemented — a flat median trend is used and
  a warning printed.
* Uncentered continuous covariates make the GLM ill-conditioned; DESeq2
  itself warns and recommends centering/scaling. Parity is tightest with
  centered covariates (as is R's own cross-BLAS reproducibility).
* `betaPrior=TRUE`, LRT, `lfcShrink` (apeglm/ashr), weights, and
  interaction designs are not implemented.

## Build

```bash
cd rust_deseq2
cargo build --release          # binary: target/release/rust_deseq2
cargo test --release           # (needs RUST_DESEQ2_REF_DIR with R reference files)
```

## Usage

```bash
./target/release/rust_deseq2 \
  --counts   examples/cov_counts.tsv \
  --coldata  examples/cov_metadata.tsv \
  --design   "~ batch + condition" \
  --contrast-var condition --contrast-case trtA --contrast-control ctrl \
  --threads  16 \
  --out      results.tsv
```

- **counts TSV**: first column = gene id, header row = sample ids
  (integer counts, as DESeq2 requires).
- **colData TSV**: first column = sample id, plus design columns. Rows may
  be in any order; they are aligned to the count matrix by id.
- **--design**: additive formula; numeric-looking columns become continuous
  covariates (`--factor <col>` forces a factor). Factor levels are ordered
  as R's `factor()` orders them (sorted), first level = reference.
- **--contrast-***: any pair of levels of a design factor, as in
  `results(dds, contrast=c(var, case, control))`.
- **--dump-prefix**: writes per-gene stage diagnostics
  (dispGeneEst/dispFit/dispMAP/dispersion/maxCooks/...) and size factors,
  for stage-level comparison against `mcols(dds)`.

Output columns: `gene baseMean log2FoldChange lfcSE stat pvalue padj`
(full precision; `NA` where DESeq2 reports NA).

### R wrapper

```r
source("R/rustDESeq2.R")
res <- rustDESeq2(countData = "counts.tsv", colData = "meta.tsv",
                  design = ~ batch + condition,
                  contrast = c("condition", "trtA", "ctrl"))
```

## Validation & examples

```bash
# simulate datasets
/opt/R/4.3.3/bin/Rscript examples/make_example.R             # 400 x 12
/opt/R/4.3.3/bin/Rscript examples/make_example_large.R       # 20k x 16
/opt/R/4.3.3/bin/Rscript examples/make_example_4v12.R        # 15k, 4 vs 12
/opt/R/4.3.3/bin/Rscript examples/make_example_covariates.R  # 8k x 24, batch+age+condition

# stage-level + results-level parity vs Bioconductor DESeq2
/opt/R/4.3.3/bin/Rscript examples/parity_check.R \
  examples/cov_counts.tsv examples/cov_metadata.tsv \
  "~ batch + condition" condition trtA ctrl target/release/rust_deseq2
```

## Performance

Wall-clock on the validation datasets (single machine, `--threads` auto;
DESeq2 single-threaded, timings include I/O):

| dataset | DESeq2 1.42.1 | rust_deseq2 | speedup |
|---|---|---|---|
| 400 × 12, ~group | 0.8 s | 0.02 s | ~40× |
| 20k × 16, ~group (with outlier refit) | 6.6 s | 1.3 s | ~5× |
| 15k × 16, 4 vs 12 | 5.0 s | 1.0 s | ~5× |
| 8k × 24, ~batch+condition | 14.8 s | 1.1 s | ~13× |
| 8k × 24, ~batch+age+condition | 9.1 s | 1.3 s | ~7× |

## Layout

```
src/
├── main.rs     # CLI
├── io.rs       # TSV I/O
├── design.rs   # formula parsing, model matrix, contrasts, cells
├── deseq.rs    # the staged workflow (dispersions, trend, MAP, Wald,
│               #  outlier replacement/flagging, filtering)
├── glm.rs      # fitBeta (QR-IRLS), fitDisp (Armijo), fitDispGrid, optim path
├── lbfgsb.rs   # port of R 4.3.3 L-BFGS-B + optim's numerical gradient
├── linalg.rs   # small dense solve/inverse/log-det, Householder QR
├── rrand.rs    # R RNG ports: Mersenne-Twister/set.seed, norm_rand
│               #  (inversion + AS241 qnorm), exp_rand, rgamma, rchisq
├── rloess.rs   # 1-D port of R loess surface="interpolate" (kd-tree,
│               #  vertex fits, cubic Hermite evaluation)
└── mathx.rs    # nmath ports: pnorm, lgamma, digamma/trigamma, dnbinom
                #  (saddle-point), qf, lowess, quantiles, BH, R-style sums
```

## Real-data validation (GSE144269, 49,579 genes)

Tumor/normal pairs subsampled from the GSE144269 HCC RSEM counts; unpaired
(`~ tissue`) and paired (`~ patient + tissue`) designs vs DESeq2 1.42.1:

| config | DE calls (both tools) | discordant | max rel diff (padj) | DESeq2 | rust (1 thr) | rust (auto) |
|---|---|---|---|---|---|---|
| 3 pairs, unpaired | 676 | 0 | 2e-10 | 9.4 s | 2.1 s | 0.4 s |
| 3 pairs, paired   | 1217 | 0 | 6e-7 | 10.9 s | 3.0 s | 0.7 s |
| 6 pairs, unpaired | 2896 | 0 | 1e-5 | 12.0 s | 3.5 s | 0.5 s |
| 6 pairs, paired   | 3956 | 0 | 4e-5 | 50.7 s | 5.4 s | 0.6 s |
| 10 pairs, unpaired| 4380 | 0 | 5e-6 | 17.0 s | 5.3 s | 0.7 s |
| 10 pairs, paired  | 5288 | 0 | 3e-4 | 147.9 s | 10.9 s | 1.0 s |

## References

- Love, Huber & Anders (2014), *Moderated estimation of fold change and
  dispersion for RNA-seq data with DESeq2*, Genome Biology 15:550.
- Bioconductor DESeq2 1.42.1 (R and C++ sources; validation target).
- R 4.3.3 sources for `pnorm`, `lgamma`, `dpsifn`, `dnbinom`/`dbinom_raw`/
  `bd0`/`stirlerr`, `lowess`, `p.adjust`, `optim`/`lbfgsb`.
- Byrd, Lu, Nocedal & Zhu (1995), *A limited memory algorithm for bound
  constrained optimization* (L-BFGS-B).
