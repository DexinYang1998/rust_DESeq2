#!/usr/bin/env Rscript
# Stage-level parity check: export DESeq2 internals and compare them with
# rust_deseq2's `--dump-prefix` diagnostics on the same dataset.
suppressMessages(library(DESeq2))

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
script_dir <- if (!is.na(script_file)) dirname(normalizePath(script_file)) else getwd()
proj <- normalizePath(file.path(script_dir, ".."))
argv <- commandArgs(trailingOnly = TRUE)
counts_f <- if (length(argv) >= 1) argv[1] else file.path(proj, "examples/gene_counts.tsv")
meta_f <- if (length(argv) >= 2) argv[2] else file.path(proj, "examples/sample_metadata.tsv")
out_dir <- if (length(argv) >= 3) argv[3] else tempfile("rust_deseq2_parity_")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

bin <- Sys.getenv("RUST_DESEQ2_BIN", file.path(proj, "target/release/rust_deseq2"))
if (!file.exists(bin)) {
  stop("rust_deseq2 binary not found: ", bin, "\nRun `cargo build --release` first.")
}

cts <- as.matrix(read.delim(counts_f, row.names = 1, check.names = FALSE))
meta <- read.delim(meta_f, row.names = 1, stringsAsFactors = FALSE)
meta <- meta[colnames(cts), , drop = FALSE]
meta$group <- factor(meta$group, levels = c("normal", "tumor"))

dds <- DESeqDataSetFromMatrix(cts, colData = meta, design = ~ group)
dds <- DESeq(dds, quiet = TRUE)
res <- as.data.frame(results(dds, contrast = c("group", "tumor", "normal")))
mc <- as.data.frame(mcols(dds))

sf <- data.frame(sample = names(sizeFactors(dds)), sizeFactor = unname(sizeFactors(dds)))
write.table(sf, file.path(out_dir, "deseq2.size_factors.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

wald_col <- "group_tumor_vs_normal"
se_col <- paste0("SE_", wald_col)
stat_col <- paste0("WaldStatistic_", wald_col)
p_col <- paste0("WaldPvalue_", wald_col)
ds_genes <- data.frame(
  gene = rownames(mc),
  baseMean = mc$baseMean,
  dispGeneEst = mc$dispGeneEst,
  dispFit = mc$dispFit,
  dispMAP = mc$dispMAP,
  dispersion = dispersions(dds),
  dispOutlier = mc$dispOutlier,
  betaConv = mc$betaConv,
  log2FoldChange = mc[[wald_col]],
  lfcSE = mc[[se_col]],
  stat = mc[[stat_col]],
  pvalue = mc[[p_col]],
  padj = res$padj
)
write.table(ds_genes, file.path(out_dir, "deseq2.genes.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

rust_out <- file.path(out_dir, "rust.results.tsv")
rust_prefix <- file.path(out_dir, "rust")
status <- system2(
  bin,
  c("--counts", counts_f, "--coldata", meta_f, "--design", "group",
    "--contrast-case", "tumor", "--contrast-control", "normal",
    "--threads", "1", "--dump-prefix", rust_prefix, "--out", rust_out),
  stdout = FALSE,
  stderr = FALSE
)
if (!identical(status, 0L)) stop("rust_deseq2 failed with status ", status)

rust_sf <- read.delim(paste0(rust_prefix, ".size_factors.tsv"), check.names = FALSE)
rust_genes <- read.delim(paste0(rust_prefix, ".genes.tsv"), check.names = FALSE, na.strings = "NA")

cat("counts:", counts_f, "\n")
cat("meta:  ", meta_f, "\n")
cat("out:   ", out_dir, "\n\n")

sf_m <- merge(sf, rust_sf, by = "sample", suffixes = c(".ds", ".rust"))
cat(sprintf("sizeFactor RMSE=%.4g max_abs=%.4g\n\n",
            sqrt(mean((sf_m$sizeFactor.ds - sf_m$sizeFactor.rust)^2)),
            max(abs(sf_m$sizeFactor.ds - sf_m$sizeFactor.rust))))

g <- merge(ds_genes, rust_genes, by = "gene", suffixes = c(".ds", ".rust"))
cmp <- function(field) {
  a <- g[[paste0(field, ".ds")]]
  b <- g[[paste0(field, ".rust")]]
  ok <- is.finite(a) & is.finite(b)
  if (sum(ok) < 3) {
    cat(sprintf("%-16s n=%5d\n", field, sum(ok)))
    return(invisible(NULL))
  }
  cat(sprintf("%-16s n=%5d Pearson=%.4f Spearman=%.4f RMSE=%.4g max_abs=%.4g\n",
              field, sum(ok), cor(a[ok], b[ok]), cor(a[ok], b[ok], method = "spearman"),
              sqrt(mean((a[ok] - b[ok])^2)), max(abs(a[ok] - b[ok]))))
}

for (field in c("baseMean", "dispGeneEst", "dispFit", "dispMAP", "dispersion",
                "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")) {
  cmp(field)
}

cat("\nBoolean agreement:\n")
for (field in c("dispOutlier", "betaConv")) {
  a <- toupper(as.character(g[[paste0(field, ".ds")]]))
  b <- toupper(as.character(g[[paste0(field, ".rust")]]))
  ok <- !is.na(a) & !is.na(b)
  cat(sprintf("%-16s %.4f (%d/%d)\n", field, mean(a[ok] == b[ok]), sum(a[ok] == b[ok]), sum(ok)))
}
