#!/usr/bin/env Rscript
# Stage-level and results-level parity check: Bioconductor DESeq2 vs rust_deseq2.
#
# Usage:
#   Rscript parity_check.R <counts.tsv> <coldata.tsv> "<design formula>" \
#       <contrast_var> <case> <control> [rust_binary]
suppressMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) stop("usage: parity_check.R counts coldata design var case control [binary]")
countsPath <- args[1]; coldataPath <- args[2]; designStr <- args[3]
cvar <- args[4]; ccase <- args[5]; cctrl <- args[6]
bin <- if (length(args) >= 7) args[7] else file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value=TRUE))), "..", "target", "release", "rust_deseq2")

cts <- as.matrix(read.delim(countsPath, row.names = 1, check.names = FALSE))
cd  <- read.delim(coldataPath, row.names = 1, check.names = FALSE)
cd  <- cd[colnames(cts), , drop = FALSE]
for (v in all.vars(as.formula(designStr))) {
  if (is.character(cd[[v]])) cd[[v]] <- factor(cd[[v]])
}
mode(cts) <- "integer"

dds <- DESeqDataSetFromMatrix(cts, cd, design = as.formula(designStr))
t_r <- system.time(dds <- DESeq(dds, quiet = TRUE))
res <- results(dds, contrast = c(cvar, ccase, cctrl))

out <- tempfile(fileext = ".tsv"); dump <- tempfile()
t_rust <- system.time(status <- system2(bin, c(
  "--counts", countsPath, "--coldata", coldataPath,
  "--design", shQuote(designStr),
  "--contrast-var", cvar, "--contrast-case", ccase, "--contrast-control", cctrl,
  "--threads", "1", "--out", out, "--dump-prefix", dump), stdout = FALSE, stderr = ""))
stopifnot(status == 0)
rust <- read.delim(out, row.names = 1)
rust <- rust[rownames(res), ]
diag <- read.delim(paste0(dump, ".genes.tsv"), row.names = 1)
diag <- diag[rownames(res), ]
sfr  <- read.delim(paste0(dump, ".size_factors.tsv"), row.names = 1)

cmp <- function(name, a, b, rel = TRUE) {
  na_a <- is.na(a); na_b <- is.na(b)
  na_mismatch <- sum(na_a != na_b)
  ok <- !na_a & !na_b
  if (rel) {
    d <- abs(a[ok] - b[ok]) / pmax(abs(a[ok]), 1e-8)
  } else {
    d <- abs(a[ok] - b[ok])
  }
  cat(sprintf("%-16s  max %sdiff %.3e   median %.3e   NA-mismatch %d\n",
              name, if (rel) "rel-" else "abs-",
              if (length(d)) max(d) else 0, if (length(d)) median(d) else 0, na_mismatch))
  invisible(max(c(d, 0)))
}

cat("== stage parity ==\n")
cmp("sizeFactor", sizeFactors(dds), sfr[colnames(cts), "sizeFactor"])
mc <- mcols(dds)
cmp("baseMean",    mc$baseMean,    diag$baseMean)
cmp("dispGeneEst", mc$dispGeneEst, diag$dispGeneEst)
cmp("dispFit",     mc$dispFit,     diag$dispFit)
cmp("dispMAP",     mc$dispMAP,     diag$dispMAP)
cmp("dispersion",  mc$dispersion,  diag$dispersion)
cmp("maxCooks",    mc$maxCooks,    diag$maxCooks)
dfun <- dispersionFunction(dds)
cat(sprintf("trend coefs (R): %s ; dispPriorVar: %.10g ; varLogDispEsts: %.10g\n",
    paste(sprintf("%.10g", attr(dfun, "coefficients")), collapse=", "),
    attr(dfun, "dispPriorVar"), attr(dfun, "varLogDispEsts")))
if (!is.null(mc$replace)) {
  rep_r <- ifelse(is.na(mc$replace), FALSE, mc$replace)
  rep_x <- diag$replace == "true"
  cat(sprintf("replace flags: R %d, rust %d, mismatch %d\n",
      sum(rep_r), sum(rep_x), sum(rep_r != rep_x)))
}
cat(sprintf("dispOutlier: R %d, rust %d, mismatch %d\n",
    sum(mc$dispOutlier, na.rm=TRUE), sum(diag$dispOutlier == "true"),
    sum(ifelse(is.na(mc$dispOutlier), FALSE, mc$dispOutlier) != (diag$dispOutlier == "true"))))

cat("\n== results parity ==\n")
cmp("baseMean",       res$baseMean,       rust$baseMean)
cmp("log2FoldChange", res$log2FoldChange, rust$log2FoldChange)
cmp("lfcSE",          res$lfcSE,          rust$lfcSE)
cmp("stat",           res$stat,           rust$stat)
cmp("pvalue",         res$pvalue,         rust$pvalue)
cmp("padj",           res$padj,           rust$padj)

sig_r <- !is.na(res$padj) & res$padj < 0.05
sig_x <- !is.na(rust$padj) & rust$padj < 0.05
cat(sprintf("\nDE calls @ padj<0.05: R %d, rust %d, discordant %d\n",
            sum(sig_r), sum(sig_x), sum(sig_r != sig_x)))
cat(sprintf("timing: DESeq2 %.2fs, rust %.2fs\n", t_r["elapsed"], t_rust["elapsed"]))
