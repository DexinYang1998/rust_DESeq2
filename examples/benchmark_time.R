#!/usr/bin/env Rscript
# Wall-clock comparison of rust_deseq2 vs. Bioconductor DESeq2 on each dataset.
# Measures DESeq2 core compute (DESeq() + results()) and rust full CLI runtime
# (TSV read + compute + TSV write). Reports best-of-N elapsed seconds.
suppressMessages(library(DESeq2))

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
script_dir <- if (!is.na(script_file)) dirname(normalizePath(script_file)) else getwd()
proj <- normalizePath(file.path(script_dir, ".."))
bin <- Sys.getenv("RUST_DESEQ2_BIN", file.path(proj, "target/release/rust_deseq2"))
if (!file.exists(bin)) {
  stop("rust_deseq2 binary not found: ", bin, "\nRun `cargo build --release` first.")
}

datasets <- list(
  c("small (400x12)",  "gene_counts.tsv",  "sample_metadata.tsv"),
  c("large (20kx16)",  "large_counts.tsv", "large_metadata.tsv"),
  c("u4v12 (15kx16)",  "u4v12_counts.tsv", "u4v12_metadata.tsv")
)

args <- commandArgs(trailingOnly = TRUE)
reps <- if (length(args) >= 1) as.integer(args[1]) else 3L
if (is.na(reps) || reps < 1L) stop("first argument, if supplied, must be repetitions >= 1")

bench_rust <- function(cf, mf, threads, reps) {
  out <- tempfile(fileext = ".tsv")
  on.exit(unlink(out), add = TRUE)
  best <- Inf
  for (r in seq_len(reps)) {
    cli <- c("--counts", cf, "--coldata", mf, "--design", "group",
             "--contrast-case", "tumor", "--contrast-control", "normal",
             "--threads", as.character(threads),
             "--out", out)
    elapsed <- system.time({
      status <- system2(bin, cli, stdout = FALSE, stderr = FALSE)
      if (!identical(status, 0L)) stop("rust_deseq2 failed with status ", status)
    })["elapsed"]
    best <- min(best, as.numeric(elapsed))
  }
  best
}

cat(sprintf("project: %s\nbinary:  %s\nreps:    %d (best elapsed seconds)\n\n",
            proj, bin, reps))
cat(sprintf("%-18s %8s %10s %11s %11s %10s\n",
            "dataset", "genes", "DESeq2", "rust_1thr", "rust_auto", "auto_speedup"))
cat(strrep("-", 76), "\n")

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
    elapsed <- system.time({
      dds <- DESeqDataSetFromMatrix(cts, meta, ~ group)
      dds <- DESeq(dds, quiet = TRUE)
      invisible(results(dds, contrast = c("group", "tumor", "normal")))
    })["elapsed"]
    ds_t <- min(ds_t, as.numeric(elapsed))
  }

  rust_1 <- bench_rust(cf, mf, threads = 1L, reps = reps)
  rust_auto <- bench_rust(cf, mf, threads = 0L, reps = reps)

  cat(sprintf("%-18s %8d %10.2f %11.2f %11.2f %9.1fx\n",
              name, nrow(cts), ds_t, rust_1, rust_auto, ds_t / rust_auto))
}
