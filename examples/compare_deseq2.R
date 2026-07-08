#!/usr/bin/env Rscript
# Head-to-head: rust_deseq2 vs. real DESeq2 on the same example dataset.
suppressMessages(library(DESeq2))

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
script_dir <- if (!is.na(script_file)) dirname(normalizePath(script_file)) else getwd()
proj <- normalizePath(file.path(script_dir, ".."))
argv <- commandArgs(trailingOnly = TRUE)
counts_f <- if (length(argv) >= 1) argv[1] else file.path(proj, "examples/gene_counts.tsv")
meta_f   <- if (length(argv) >= 2) argv[2] else file.path(proj, "examples/sample_metadata.tsv")
truth_f  <- if (length(argv) >= 3) argv[3] else file.path(proj, "examples/ground_truth.tsv")
cat("counts:", counts_f, "\nmeta:  ", meta_f, "\n")

## ---- reference DESeq2 run ------------------------------------------------
cts  <- as.matrix(read.delim(counts_f, row.names = 1, check.names = FALSE))
meta <- read.delim(meta_f, row.names = 1, stringsAsFactors = FALSE)
meta <- meta[colnames(cts), , drop = FALSE]
meta$group <- factor(meta$group, levels = c("normal", "tumor"))  # normal = reference

dds <- DESeqDataSetFromMatrix(cts, colData = meta, design = ~ group)
dds <- DESeq(dds, quiet = TRUE)
ref <- as.data.frame(results(dds, contrast = c("group", "tumor", "normal")))
ref$gene <- rownames(ref)

## ---- rust_deseq2 run -----------------------------------------------------
source(file.path(proj, "R/rustDESeq2.R"))
rust <- rustDESeq2(
  countData = counts_f,
  colData   = meta_f,
  design    = "group",
  contrast  = c("group", "tumor", "normal"),
  binary    = file.path(proj, "target/release/rust_deseq2")
)

## ---- align & compare -----------------------------------------------------
m <- merge(ref, rust, by = "gene", suffixes = c(".ds", ".rust"))
ok <- is.finite(m$log2FoldChange.ds) & is.finite(m$log2FoldChange.rust)
m <- m[ok, ]

cat("\n================ rust_deseq2  vs  DESeq2  ================\n")
cat(sprintf("genes compared: %d\n\n", nrow(m)))

cmp <- function(a, b, name) {
  ok <- is.finite(a) & is.finite(b)   # drop NA/Inf pairs (e.g. Cook's-outlier NAs)
  a <- a[ok]; b <- b[ok]
  cat(sprintf("%-16s  Pearson r=%.4f  Spearman=%.4f  RMSE=%.4g\n",
      name, cor(a, b), cor(a, b, method = "spearman"),
      sqrt(mean((a - b)^2))))
}
cmp(m$baseMean.ds,       m$baseMean.rust,       "baseMean")
cmp(m$log2FoldChange.ds, m$log2FoldChange.rust, "log2FoldChange")
cmp(m$lfcSE.ds,          m$lfcSE.rust,          "lfcSE")
cmp(m$stat.ds,           m$stat.rust,           "stat (Wald)")

# p-value agreement on the -log10 scale (robust to tiny values).
pl_ds   <- -log10(pmax(m$pvalue.ds,   1e-300))
pl_rust <- -log10(pmax(m$pvalue.rust, 1e-300))
cmp(pl_ds, pl_rust, "-log10(pvalue)")

## ---- DE-call concordance at padj < 0.05 ----------------------------------
ds_call   <- !is.na(m$padj.ds)   & m$padj.ds   < 0.05
rust_call <- !is.na(m$padj.rust) & m$padj.rust < 0.05
tab <- table(DESeq2 = ds_call, rust = rust_call)
cat("\nDE calls at padj < 0.05 (contingency):\n"); print(tab)
agree <- mean(ds_call == rust_call)
jacc  <- sum(ds_call & rust_call) / sum(ds_call | rust_call)
cat(sprintf("\nagreement=%.3f   Jaccard(DE sets)=%.3f\n", agree, jacc))

## ---- vs ground truth (both) ----------------------------------------------
truth <- read.delim(truth_f)
mt <- merge(m, truth, by = "gene")
cat("\nLFC accuracy vs known truth:\n")
cat(sprintf("  DESeq2       r=%.4f\n", cor(mt$log2FoldChange.ds,   mt$true_lfc)))
cat(sprintf("  rust_deseq2  r=%.4f\n", cor(mt$log2FoldChange.rust, mt$true_lfc)))
