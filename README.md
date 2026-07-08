# rust_deseq2

A from-scratch **Rust reimplementation of the core DESeq2 differential-expression
workflow**, plus a thin R wrapper (`R/rustDESeq2.R`) that drives the compiled
CLI and returns a `data.frame`.

The crate is intentionally **dependency-free** — all numerics (GLM/IRLS, linear
algebra, special functions, statistics) are implemented in pure `std` Rust, so
it builds offline with no external footprint.

## What it implements

The pipeline mirrors the essential steps of DESeq2:

1. **Median-of-ratios size factors** (`estimateSizeFactors`) from genes with
   all-positive counts.
2. A **design matrix** from the condition factor, with the *control* level as
   the reference so the *case* coefficient is directly the log fold change.
3. **Cox–Reid adjusted gene-wise dispersion** by negative-binomial maximum
   likelihood — seeded by DESeq2's **rough + moments initialiser**
   (`min(roughDispEstimate, momentsDispEstimate)`) and bounded to
   `[minDisp, maxDisp]` with `maxDisp = max(10, n)` — a parametric
   **mean–dispersion trend** (`disp = a0 + a1/mean`) fit by DESeq2's
   **iteratively reweighted Gamma GLM** (identity link, residual gating), an
   **estimated log-normal prior variance** (robust MAD spread minus the expected
   sampling variance, `trigamma((m − p)/2)`), and **MAP shrinkage**
   of each gene's dispersion toward the trend — with a **dispersion-outlier
   carve-out** (gene-wise estimates > 2 raw-residual SDs, `sqrt(varLogDispEsts)`,
   above the trend are left unshrunk).
4. A **negative-binomial GLM** (log link) fit per gene by iteratively reweighted
   least squares (IRLS), converged on **relative deviance change**
   (`|dev − dev_old|/(|dev|+0.1) < 1e-8`, DESeq2's criterion) with a `|β| > 30`
   divergence guard. DESeq2's numerical stabilisers are included: a weak **ridge
   prior** (`λ = 1e-6/ln2²`) on the coefficients and a **fitted-mean floor**
   (`minmu = 0.5`), which keep separated / all-zero-in-a-group genes finite and
   bounded. The coefficient covariance is the **ridge sandwich**
   `(XᵀWX+λ)⁻¹ · XᵀWX · (XᵀWX+λ)⁻¹`, not the plain ridged inverse.
5. A **Wald test** on the case-vs-control contrast, giving `log2FoldChange`,
   `lfcSE`, `stat`, `pvalue`. Tail probabilities use an **`erfc`-based** normal
   survival function (relative accuracy ~1e-7), so extreme statistics yield
   correct tiny p-values instead of underflowing to 0.
6. **Benjamini–Hochberg** adjusted p-values (`padj`) with **independent
   filtering** on `baseMean`, reproducing DESeq2's procedure: 50 quantile
   thresholds, a **LOWESS (f = 1/5)** smooth of the rejection-count curve, the
   smallest threshold within one residual SD of the smoothed peak, and filtering
   disabled when the peak rejection count is ≤ 10.

Output columns match DESeq2's `results()` table:
`gene  baseMean  log2FoldChange  lfcSE  stat  pvalue  padj`.

### Scope / simplifications

This is a faithful *core* reimplementation, not a drop-in replacement. It
implements size factors, Cox–Reid dispersion with a fitted trend + estimated
prior + MAP shrinkage, the NB-GLM Wald test, and independent filtering. It still
omits a few of DESeq2's refinements: Cook's-distance **outlier replacement**, the
exact `parametricDispersionFit` gamma-GLM trend (a least-squares fit is used
instead), and `apeglm`/`ashr` LFC shrinkage. Estimates track DESeq2 to 3–4
significant figures on any adequately expressed gene (see validation); the only
material divergence is in near-zero-count genes, where fold-change estimation is
inherently unstable for both tools.

## Build

```bash
cd rust_deseq2
cargo build --release
# binary: target/release/rust_deseq2
```

## Command-line usage

```bash
./target/release/rust_deseq2 \
  --counts   examples/gene_counts.tsv \
  --coldata  examples/sample_metadata.tsv \
  --design   group \
  --contrast-var group --contrast-case tumor --contrast-control normal \
  --threads  16 \
  --out      results.tsv
```

- **counts TSV**: first column = gene id, header row = sample ids.
- **colData TSV**: first column = sample id (matching the count header), plus a
  condition column (e.g. `group`). Sample rows may be in any order — they are
  aligned to the count matrix by id.
- **`--threads`**: worker threads for the per-gene fits (default: cores, capped
  at 16). See [Performance](#performance).

## R wrapper

```r
source("R/rustDESeq2.R")

res <- rustDESeq2(
  countData = "gene_counts.tsv",
  colData   = "sample_metadata.tsv",
  design    = "group",
  contrast  = c("group", "tumor", "normal")   # c(column, case, control)
)
head(res[order(res$padj), ])
```

The wrapper calls the binary with `system2()` and reads the output back with
`read.delim()`, returning a `data.frame`.

> **Binary path.** The wrapper's default `binary=` is
> `/data/dyang11/software/rust_deseq2/target/release/rust_deseq2`. This project
> lives at `.../rewrite_package/rust_deseq2`, so either build there and point to
> that path, or pass `binary=` explicitly, e.g.
> `binary = normalizePath("target/release/rust_deseq2")`. To make the default
> work as-is you can symlink:
> `ln -s /data/dyang11/software/rewrite_package/rust_deseq2 /data/dyang11/software/rust_deseq2`.

## Validation

`examples/make_example.R` simulates 400 genes × 12 samples (2 groups) with 60
truly DE genes, known fold changes, per-sample depth differences and NB
overdispersion:

```bash
Rscript examples/make_example.R      # writes examples/*.tsv
```

On this dataset rust_deseq2 recovers:

- log2FoldChange vs truth: **Pearson r ≈ 0.92**
- DE detection at `padj < 0.05`: **sensitivity ≈ 0.97, specificity ≈ 0.98**

### Head-to-head vs. real DESeq2

`examples/compare_deseq2.R` runs Bioconductor **DESeq2 1.42.1** and rust_deseq2
on the identical dataset and compares them gene-by-gene:

```bash
/opt/R/4.3.3/bin/Rscript examples/compare_deseq2.R
```

**Small set — 400 genes × 12 samples:**

| quantity | agreement with DESeq2 |
|----------|-----------------------|
| `baseMean` | r = 1.000 (identical) |
| `log2FoldChange` | r = 1.000, RMSE ≈ 0.0008 |
| `lfcSE` | r = 0.998 |
| `stat` (Wald) | r = 0.9999, Spearman = 1.000 |
| `-log10(pvalue)` | r = 1.000 |
| LFC vs known truth | DESeq2 r = 0.9220 · rust r = 0.9220 |

DE calls at `padj < 0.05` agree on **all 400 genes** (Jaccard 1.00).

**Large set — 20,000 genes × 16 samples, ~8× depth range** (`examples/make_example_large.R`,
runs in ~7 s single-threaded):

| quantity | agreement with DESeq2 |
|----------|-----------------------|
| `baseMean` | r = 1.000 |
| `log2FoldChange`, all 19.9k genes | r = 0.985 |
| `log2FoldChange`, `baseMean > 5` (16.9k genes) | r = 0.9999, RMSE ≈ 0.009 |
| `stat` (Wald) | **r = 0.9997**, Spearman = 0.996 |
| `-log10(pvalue)` | **r = 0.9999** |
| `lfcSE` (all genes) | r = 0.9945 |
| `lfcSE`, `baseMean > 5` | r = 0.9999 |
| `pvalue = NA` count | 71 (identical to DESeq2) |
| LFC vs known truth | DESeq2 r = 0.7701 · rust r = 0.7699 |

DE calls at `padj < 0.05` agree on **99.97 %** of genes (Jaccard 0.997; 2 FP,
4 FN out of ~2096). The Gamma-GLM dispersion trend keeps the low-mean
extrapolation in line with DESeq2 (Wald-statistic r = 0.9997); the `erfc` tail
lifts `-log10(pvalue)` agreement from ~0.7 to 0.9999; the rough/moments
dispersion initialiser with the `maxDisp = max(10, n)` cap plus the ridge
sandwich covariance and `minmu` floor bring `lfcSE` agreement to 0.994 (0.9999
for `baseMean > 5`). The only residual `lfcSE` differences are a handful of
genes with `baseMean ≈ 0` that DESeq2 emits as exactly `0`.

**Unbalanced set — 15,000 genes, 4 vs 12 samples** (`examples/make_example_4v12.R`),
stressing the small-group regime:

| quantity | agreement with DESeq2 |
|----------|-----------------------|
| `log2FoldChange` | r = 0.996 |
| `lfcSE` | r = 0.998 |
| `stat` (Wald) | r = 0.9998 |
| `-log10(pvalue)` | r = 0.9997 |
| DE calls @ padj<0.05 | Jaccard 0.994 |
| LFC vs known truth | DESeq2 r = 0.7333 · rust r = 0.7334 |

Reproduce with:

```bash
/opt/R/4.3.3/bin/Rscript examples/make_example_large.R
/opt/R/4.3.3/bin/Rscript examples/compare_deseq2.R \
  examples/large_counts.tsv examples/large_metadata.tsv examples/large_truth.tsv
```

## Performance

The per-gene fits (pass 1 dispersion, pass 2 Wald test) are embarrassingly
parallel and run across `--threads` scoped worker threads (pure `std`, no
dependencies). Wall-clock, best of 3, vs. single-threaded Bioconductor DESeq2
(`DESeq()` + `results()`), on one machine; rust timings include TSV read/write:

| dataset | DESeq2 | rust (16 thr) | speedup |
|---------|--------|---------------|---------|
| 400 × 12    | 0.87 s | 0.01 s | ~67× |
| 15k × 16 (4v12) | 4.70 s | 0.42 s | ~11× |
| 20k × 16    | 6.01 s | 0.56 s | ~11× |

Thread scaling on the 20k set (near-linear to ~16 threads):

| threads | 1 | 4 | 8 | 16 | 32 |
|---------|-----|-----|-----|-----|-----|
| time (s)| 7.4 | 1.9 | 1.0 | 0.55 | 0.38 |

Note that single-threaded (`--threads 1`) rust is ~on par with DESeq2's
optimised C++/Armadillo (7.4 s vs 6 s); the speedup comes from parallelism, since
`DESeq()` is single-threaded by default. Reproduce with
`examples/benchmark_time.R`.

## Layout

```
rust_deseq2/
├── Cargo.toml
├── src/
│   ├── main.rs      # CLI argument parsing + driver
│   ├── io.rs        # TSV read/write
│   ├── deseq.rs     # workflow: size factors, design, dispersion trend, Wald
│   ├── glm.rs       # NB GLM (IRLS) + dispersion MLE/MAP
│   ├── linalg.rs    # small dense matrix inverse + log-determinant
│   └── mathx.rs     # ln-gamma, trigamma, NB log-pmf, erfc/normal, BH, median
├── R/
│   └── rustDESeq2.R # R wrapper (system2 -> read.delim -> data.frame)
└── examples/
    ├── make_example.R        # small synthetic dataset (400 x 12)
    ├── make_example_large.R  # 20k-gene dataset, varying depth
    ├── make_example_4v12.R   # 15k-gene unbalanced dataset (4 vs 12)
    ├── compare_deseq2.R      # head-to-head vs Bioconductor DESeq2
    └── benchmark_time.R      # wall-clock timing vs DESeq2
```

## References & acknowledgements

- Love, Huber & Anders (2014), *Moderated estimation of fold change and
  dispersion for RNA-seq data with DESeq2*, **Genome Biology** 15:550 — the
  method this reimplements.
- The Bioconductor **DESeq2** package (validated here against v1.42.1).
- [`necoli1822/rust_deseq2`](https://github.com/necoli1822/rust_deseq2), a fuller
  Rust DESeq2 port (MIT). Studying it informed two numerical-stability choices
  used here for separated / low-count genes — the weak coefficient **ridge
  prior** (`λ = 1e-6/ln2²`) and the fitted-mean floor (**`minmu = 0.5`**), both
  DESeq2 defaults — which raised the low-count fold-change agreement and matched
  DESeq2's set of untested genes exactly. This implementation is independent and
  dependency-free (that project uses `ndarray` and adds Cook's-distance
  outliers, apeglm/ashr shrinkage, LRT, VST/rlog and multi-factor designs, which
  are out of scope here).
