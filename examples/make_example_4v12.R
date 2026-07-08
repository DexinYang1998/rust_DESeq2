#!/usr/bin/env Rscript
# Unbalanced design: 4 vs 12 samples, ~15k genes, varying sequencing depth.
# Exercises the small-group-size regime (control n=4).

set.seed(7)

n_case    <- 12   # tumor
n_control <- 4    # normal
groups  <- c(rep("normal", n_control), rep("tumor", n_case))
samples <- paste0(groups, "_", ave(seq_along(groups), groups, FUN = seq_along))
ns      <- length(samples)

n_genes <- 15000
n_de    <- 1800

base_mu <- pmax(rlnorm(n_genes, meanlog = 4.0, sdlog = 2.2), 0.2)

true_lfc <- rep(0, n_genes)
de_idx <- sample(seq_len(n_genes), n_de)
true_lfc[de_idx] <- sample(c(-1, 1), n_de, TRUE) * runif(n_de, 0.75, 3.0)

# Varying sequencing depth (~6x span), geometric mean 1.
sf <- 2^runif(ns, -1.3, 1.3)
sf <- sf / exp(mean(log(sf)))

disp <- 0.02 + 4 / base_mu

counts <- matrix(0L, nrow = n_genes, ncol = ns,
                 dimnames = list(paste0("gene", seq_len(n_genes)), samples))
for (g in seq_len(n_genes)) {
  size_g <- 1 / disp[g]
  for (j in seq_len(ns)) {
    lfc <- if (groups[j] == "tumor") true_lfc[g] else 0
    mu  <- base_mu[g] * 2^lfc * sf[j]
    counts[g, j] <- rnbinom(1, mu = mu, size = size_g)
  }
}

dir <- file.path("/data/dyang11/software/rewrite_package/rust_deseq2", "examples")

cm <- data.frame(gene = rownames(counts), counts, check.names = FALSE)
write.table(cm, file.path(dir, "u4v12_counts.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

meta <- data.frame(sample = samples, group = groups, stringsAsFactors = FALSE)
write.table(meta, file.path(dir, "u4v12_metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

truth <- data.frame(gene = rownames(counts), true_lfc = true_lfc,
                    is_de = seq_len(n_genes) %in% de_idx)
write.table(truth, file.path(dir, "u4v12_truth.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat(sprintf("Wrote %d genes, %d vs %d samples (DE=%d)\n",
            n_genes, n_control, n_case, n_de))
cat(sprintf("size factors range: %.2f .. %.2f\n", min(sf), max(sf)))
