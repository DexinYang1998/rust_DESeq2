#!/usr/bin/env Rscript
# Wall-clock comparison of rust_deseq2 vs. Bioconductor DESeq2 on each dataset.
# Measures the core compute (DESeq() + results() for R; the binary for rust).
suppressMessages(library(DESeq2))

proj <- "/data/dyang11/software/rewrite_package/rust_deseq2"
bin  <- file.path(proj, "target/release/rust_deseq2")

datasets <- list(
  c("small (400x12)",  "gene_counts.tsv",  "sample_metadata.tsv"),
  c("large (20kx16)",  "large_counts.tsv", "large_metadata.tsv"),
  c("u4v12 (15kx16)",  "u4v12_counts.tsv", "u4v12_metadata.tsv")
)

reps <- 3
cat(sprintf("%-18s %8s %10s %10s %9s\n", "dataset", "genes", "DESeq2(s)", "rust(s)", "speedup"))
cat(strrep("-", 60), "\n")

for (ds in datasets) {
  name <- ds[1]
  cf <- file.path(proj, "examples", ds[2])
  mf <- file.path(proj, "examples", ds[3])
  if (!file.exists(cf)) { cat(sprintf("%-18s  (missing)\n", name)); next }

  cts  <- as.matrix(read.delim(cf, row.names = 1, check.names = FALSE))
  meta <- read.delim(mf, row.names = 1); meta <- meta[colnames(cts), , drop = FALSE]
  meta$group <- factor(meta$group, levels = c("normal", "tumor"))

  # DESeq2 core compute (exclude file reading), best of `reps`.
  ds_t <- Inf
  for (r in seq_len(reps)) {
    t <- system.time({
      dds <- DESeqDataSetFromMatrix(cts, meta, ~ group)
      dds <- DESeq(dds, quiet = TRUE)
      invisible(results(dds, contrast = c("group", "tumor", "normal")))
    })["elapsed"]
    ds_t <- min(ds_t, as.numeric(t))
  }

  # rust full pipeline (includes TSV read/write), best of `reps`.
  out <- tempfile(fileext = ".tsv")
  rust_t <- Inf
  for (r in seq_len(reps)) {
    t <- system.time({
      system2(bin, c("--counts", cf, "--coldata", mf, "--design", "group",
                     "--contrast-case", "tumor", "--contrast-control", "normal",
                     "--out", out), stdout = FALSE, stderr = FALSE)
    })["elapsed"]
    rust_t <- min(rust_t, as.numeric(t))
  }

  cat(sprintf("%-18s %8d %10.2f %10.2f %8.1fx\n",
              name, nrow(cts), ds_t, rust_t, ds_t / rust_t))
}
