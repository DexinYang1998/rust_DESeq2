#!/usr/bin/env Rscript
# Simulate a multi-factor dataset for validating covariate designs:
#   * condition: 3 levels (ctrl, trtA, trtB), the variable of interest
#   * batch:     3 levels (b1, b2, b3) with real batch effects on many genes
#   * age:       a continuous covariate with a mild expression effect
# 8000 genes x 24 samples, NB counts with mean-dependent dispersion and
# per-sample depth differences.
set.seed(42)

n_genes <- 8000
cond  <- factor(rep(c("ctrl", "trtA", "trtB"), each = 8))
batch <- factor(rep(rep(c("b1", "b2", "b3"), times = c(3, 3, 2)), 3))
age   <- round(runif(24, 30, 70))
samples <- paste0("s", seq_len(24))

base <- exp(rnorm(n_genes, log(120), 1.6))
disp <- 0.02 + 4 / base + rexp(n_genes, 8)

lfcA <- rep(0, n_genes); lfcB <- rep(0, n_genes)
deA <- sample(n_genes, 600); lfcA[deA] <- rnorm(600, 0, 1.2)
deB <- sample(n_genes, 600); lfcB[deB] <- rnorm(600, 0, 1.2)
bfx2 <- rnorm(n_genes, 0, 0.25) * rbinom(n_genes, 1, 0.5)
bfx3 <- rnorm(n_genes, 0, 0.25) * rbinom(n_genes, 1, 0.5)
agefx <- rnorm(n_genes, 0, 0.01) * rbinom(n_genes, 1, 0.3)

depth <- exp(rnorm(24, 0, 0.35))
mu <- matrix(0, n_genes, 24)
for (j in seq_len(24)) {
  lc <- log2(base) +
    (cond[j] == "trtA") * lfcA + (cond[j] == "trtB") * lfcB +
    (batch[j] == "b2") * bfx2 + (batch[j] == "b3") * bfx3 +
    agefx * (age[j] - 50)
  mu[, j] <- 2^lc * depth[j]
}
cts <- matrix(rnbinom(n_genes * 24, mu = mu, size = 1 / disp), n_genes, 24)
rownames(cts) <- paste0("gene", seq_len(n_genes))
colnames(cts) <- samples

dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
write.table(data.frame(gene = rownames(cts), cts, check.names = FALSE),
            file.path(dir, "cov_counts.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(sample = samples, condition = as.character(cond),
                       batch = as.character(batch), age = age),
            file.path(dir, "cov_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
truth <- data.frame(gene = rownames(cts), lfcA = lfcA, lfcB = lfcB)
write.table(truth, file.path(dir, "cov_truth.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("wrote cov_counts.tsv / cov_metadata.tsv / cov_truth.tsv\n")
