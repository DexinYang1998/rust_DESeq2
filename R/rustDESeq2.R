#' Run the Rust reimplementation of the core DESeq2 workflow
#'
#' A thin R wrapper around the `rust_deseq2` command-line binary. It shells out
#' to the compiled Rust program via [system2()], which performs median-of-ratios
#' size-factor estimation, negative-binomial dispersion estimation with
#' shrinkage toward a fitted mean-dispersion trend, and a negative-binomial GLM
#' Wald test for the requested contrast. The result table is read back into R
#' with [read.delim()].
#'
#' @param countData Path to a TSV count matrix (genes x samples). The first
#'   column holds gene ids and the header row holds sample ids.
#' @param colData Path to a TSV sample-metadata table. The first column holds
#'   sample ids (matching the count-matrix header) and other columns hold
#'   sample attributes such as the condition column.
#' @param design Name of the condition column in `colData`, e.g. `"group"`.
#' @param contrast A character vector of length 3 following DESeq2 ordering:
#'   `c(condition_column, case_level, control_level)`. The log2 fold change is
#'   reported for `case_level` relative to `control_level`.
#' @param out Path for the output TSV. Defaults to a temporary file.
#' @param binary Path to the compiled `rust_deseq2` binary.
#'
#' @return A data.frame with columns `gene`, `baseMean`, `log2FoldChange`,
#'   `lfcSE`, `stat`, `pvalue`, `padj`.
#'
#' @examples
#' \dontrun{
#' res <- rustDESeq2(
#'   countData = "gene_counts.tsv",
#'   colData   = "sample_metadata.tsv",
#'   design    = "group",
#'   contrast  = c("group", "tumor", "normal")
#' )
#' head(res[order(res$padj), ])
#' }
#'
#' @export
rustDESeq2 <- function(
    countData,
    colData,
    design,
    contrast,
    out = tempfile(fileext = ".tsv"),
    binary = "/data/dyang11/software/rust_deseq2/target/release/rust_deseq2"
) {
  ## ---- validate inputs -----------------------------------------------------
  if (!file.exists(countData)) {
    stop("countData file does not exist: ", countData)
  }
  if (!file.exists(colData)) {
    stop("colData file does not exist: ", colData)
  }
  if (!is.character(design) || length(design) != 1L) {
    stop("`design` must be a single column name, e.g. \"group\".")
  }
  if (!is.character(contrast) || length(contrast) != 3L) {
    stop(
      "`contrast` must be a character vector of length 3: ",
      "c(condition_column, case_level, control_level)."
    )
  }
  if (!file.exists(binary)) {
    stop(
      "rust_deseq2 binary not found at: ", binary,
      "\nBuild it with `cargo build --release` in the rust_deseq2 project, ",
      "or pass a different `binary=` path."
    )
  }

  contrast_var     <- contrast[1]
  contrast_case    <- contrast[2]
  contrast_control <- contrast[3]

  ## ---- assemble CLI arguments ---------------------------------------------
  args <- c(
    "--counts",           countData,
    "--coldata",          colData,
    "--design",           design,
    "--contrast-var",     contrast_var,
    "--contrast-case",    contrast_case,
    "--contrast-control", contrast_control,
    "--out",              out
  )

  ## ---- call the Rust CLI ---------------------------------------------------
  status <- system2(binary, args = shQuote(args), stdout = "", stderr = "")
  if (!identical(status, 0L)) {
    stop("rust_deseq2 exited with non-zero status: ", status)
  }
  if (!file.exists(out)) {
    stop("rust_deseq2 did not produce an output file at: ", out)
  }

  ## ---- read results back ---------------------------------------------------
  res <- read.delim(out, header = TRUE, stringsAsFactors = FALSE, na.strings = "NA")
  res
}
