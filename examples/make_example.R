#!/usr/bin/env Rscript
# Generate a small synthetic RNA-seq dataset with a known set of
# differentially expressed genes, for testing rust_deseq2.

set.seed(42)

n_per_group <- 6
groups <- rep(c("normal", "tumor"), each = n_per_group)
samples <- paste0(groups, "_", ave(seq_along(groups), groups, FUN = seq_along))

n_genes <- 400
n_de    <- 60

# Baseline mean expression per gene (broad dynamic range).
base_mu <- 2^runif(n_genes, 2, 12)

# True log2 fold changes: 0 for null genes, +/- signal for DE genes.
true_lfc <- rep(0, n_genes)
de_idx <- sample(seq_len(n_genes), n_de)
true_lfc[de_idx] <- sample(c(-1, 1), n_de, TRUE) * runif(n_de, 1.0, 3.0)

# Per-sample size factors (sequencing-depth differences).
sf <- runif(length(samples), 0.6, 1.6)

# Modest per-gene overdispersion.
disp <- 0.01 + 3 / base_mu

counts <- matrix(0L, nrow = n_genes, ncol = length(samples),
                 dimnames = list(paste0("gene", seq_len(n_genes)), samples))

for (g in seq_len(n_genes)) {
  for (j in seq_along(samples)) {
    lfc <- if (groups[j] == "tumor") true_lfc[g] else 0
    mu  <- base_mu[g] * 2^lfc * sf[j]
    size <- 1 / disp[g]                       # NB size = 1/dispersion
    counts[g, j] <- rnbinom(1, mu = mu, size = size)
  }
}

dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
if (length(dir) == 0 || dir == "") dir <- "."

# Write count matrix TSV (first column = gene id).
cm <- data.frame(gene = rownames(counts), counts, check.names = FALSE)
write.table(cm, file.path(dir, "gene_counts.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Write sample metadata TSV (first column = sample id).
meta <- data.frame(sample = samples, group = groups, stringsAsFactors = FALSE)
write.table(meta, file.path(dir, "sample_metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Also save ground truth for validation.
truth <- data.frame(gene = rownames(counts), true_lfc = true_lfc,
                    is_de = seq_len(n_genes) %in% de_idx)
write.table(truth, file.path(dir, "ground_truth.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("Wrote example data to", dir, "\n")
cat("  genes:", n_genes, " DE genes:", n_de, " samples:", length(samples), "\n")
