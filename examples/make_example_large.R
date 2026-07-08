#!/usr/bin/env Rscript
# Larger, more realistic synthetic RNA-seq dataset: ~20k genes across two
# groups with widely varying per-sample sequencing depth. Written for stress-
# testing rust_deseq2 against DESeq2.

set.seed(2024)

n_per_group <- 8
groups  <- rep(c("normal", "tumor"), each = n_per_group)
samples <- paste0(groups, "_", ave(seq_along(groups), groups, FUN = seq_along))
ns      <- length(samples)

n_genes <- 20000
n_de    <- 2500

# Baseline mean expression: heavy-tailed (log-normal), broad dynamic range,
# including many low-count genes.
base_mu <- pmax(rlnorm(n_genes, meanlog = 4.0, sdlog = 2.3), 0.2)

# True log2 fold changes for a subset of genes.
true_lfc <- rep(0, n_genes)
de_idx <- sample(seq_len(n_genes), n_de)
true_lfc[de_idx] <- sample(c(-1, 1), n_de, TRUE) * runif(n_de, 0.75, 3.0)

# WIDELY varying sequencing depth: per-sample size factors spanning ~10x.
sf <- 2^runif(ns, -1.6, 1.6)          # ~0.33x .. ~3x
sf <- sf / exp(mean(log(sf)))          # centre the geometric mean at 1

# Mean-dependent overdispersion (DESeq2-like trend: high disp at low counts).
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
write.table(cm, file.path(dir, "large_counts.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

meta <- data.frame(sample = samples, group = groups, stringsAsFactors = FALSE)
write.table(meta, file.path(dir, "large_metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

truth <- data.frame(gene = rownames(counts), true_lfc = true_lfc,
                    is_de = seq_len(n_genes) %in% de_idx)
write.table(truth, file.path(dir, "large_truth.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat(sprintf("Wrote %d genes x %d samples (DE=%d) to %s\n", n_genes, ns, n_de, dir))
cat(sprintf("size factors range: %.2f .. %.2f (%.1fx)\n",
            min(sf), max(sf), max(sf)/min(sf)))
cat(sprintf("library sizes (M reads): %.1f .. %.1f\n",
            min(colSums(counts))/1e6, max(colSums(counts))/1e6))
