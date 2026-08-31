#' rustDESeq2: run the rust_deseq2 binary and return a DESeq2-style data.frame
#'
#' A thin wrapper around the compiled `rust_deseq2` CLI. Results columns match
#' `DESeq2::results(dds, contrast = c(var, case, control))`:
#' baseMean, log2FoldChange, lfcSE, stat, pvalue, padj.
#'
#' @param countData path to a TSV count matrix (genes x samples, first column
#'   gene ids) OR a matrix/data.frame of counts (written to a temp file).
#' @param colData path to a TSV of sample metadata (first column sample ids)
#'   OR a data.frame (written to a temp file; rownames become sample ids).
#' @param design a design formula such as `~ batch + condition` (also accepts
#'   a formula object or a single column name). Additive factors and numeric
#'   covariates are supported; interactions are not.
#' @param contrast c(variable, caseLevel, controlLevel), as in DESeq2.
#' @param factorCols character vector of colData columns to force to factors
#'   even when their values look numeric.
#' @param threads worker threads (0 = auto: available cores, capped at 12).
#' @param binary path to the rust_deseq2 executable. Defaults to
#'   RUST_DESEQ2_BIN if set, else the binary next to this script's crate
#'   (found automatically at source() time), so the wrapper works from any
#'   working directory.
#' @return data.frame with rownames = gene ids.

# Locate this script at source() time so the compiled binary can be found
# regardless of the caller's working directory.
.rustDESeq2.binary.default <- local({
  path <- NULL
  for (i in seq_len(sys.nframe())) {
    of <- tryCatch(sys.frame(i)$ofile, error = function(e) NULL)
    if (!is.null(of)) path <- of
  }
  if (!is.null(path)) {
    normalizePath(file.path(dirname(normalizePath(path)),
                            "..", "target", "release", "rust_deseq2"),
                  mustWork = FALSE)
  } else {
    "target/release/rust_deseq2"
  }
})

rustDESeq2 <- function(countData,
                       colData,
                       design,
                       contrast,
                       factorCols = character(),
                       threads = 0,
                       binary = Sys.getenv("RUST_DESEQ2_BIN",
                                           .rustDESeq2.binary.default)) {
  stopifnot(length(contrast) == 3)
  if (!file.exists(binary)) {
    stop("rust_deseq2 binary not found at '", binary,
         "'; build it with 'cargo build --release' or set RUST_DESEQ2_BIN")
  }

  as_path <- function(obj, what) {
    if (is.character(obj) && length(obj) == 1 && file.exists(obj)) {
      return(obj)
    }
    f <- tempfile(fileext = ".tsv")
    df <- as.data.frame(obj)
    if (what == "counts" && any(as.matrix(df) != floor(as.matrix(df)), na.rm = TRUE)) {
      warning("countData contains non-integer values; DESeq2 requires integer ",
              "counts - round them (e.g. floor()) for exact parity")
    }
    id_col <- if (what == "counts") "gene" else "sample"
    out <- cbind(setNames(data.frame(rownames(df)), id_col), df)
    write.table(out, f, sep = "\t", quote = FALSE, row.names = FALSE)
    f
  }

  countsPath <- as_path(countData, "counts")
  colPath <- as_path(colData, "coldata")
  designStr <- if (inherits(design, "formula")) {
    paste(deparse(design), collapse = " ")
  } else {
    as.character(design)
  }

  out <- tempfile(fileext = ".tsv")
  args <- c(
    "--counts", shQuote(countsPath),
    "--coldata", shQuote(colPath),
    "--design", shQuote(designStr),
    "--contrast-var", shQuote(contrast[1]),
    "--contrast-case", shQuote(contrast[2]),
    "--contrast-control", shQuote(contrast[3]),
    "--threads", as.character(threads),
    "--out", shQuote(out)
  )
  for (fc in factorCols) args <- c(args, "--factor", shQuote(fc))

  status <- system2(binary, args, stdout = FALSE, stderr = "")
  if (status != 0) {
    stop("rust_deseq2 failed with exit status ", status)
  }
  res <- read.delim(out, row.names = 1, check.names = FALSE)
  res
}
