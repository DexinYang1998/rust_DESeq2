//! rust_deseq2 — a from-scratch Rust reimplementation of the DESeq2
//! differential-expression workflow (Wald test).
//!
//! Usage:
//!   rust_deseq2 \
//!     --counts   gene_counts.tsv \
//!     --coldata  sample_metadata.tsv \
//!     --design   "~ batch + group" \
//!     --contrast-var group --contrast-case tumor --contrast-control normal \
//!     --out      results.tsv
//!
//! The output TSV has columns:
//!   gene  baseMean  log2FoldChange  lfcSE  stat  pvalue  padj

mod deseq;
mod design;
mod glm;
mod lbfgsb;
mod io;
mod linalg;
mod mathx;
mod rloess;
mod rrand;

use deseq::Options;

fn usage() -> String {
    "\
rust_deseq2 — DESeq2 workflow in Rust

Required arguments:
  --counts   <path>   TSV count matrix (genes x samples; first col = gene id)
  --coldata  <path>   TSV sample metadata (first col = sample id + columns)
  --design   <spec>   design: a formula like \"~ batch + group\" (additive
                      covariates + condition), or a single column name
  --contrast-case    <level>   case (numerator) level, e.g. \"tumor\"
  --contrast-control <level>   control (denominator) level

Optional arguments:
  --contrast-var <col>   contrast variable (default: last design variable)
  --factor       <col>   force a numeric-looking colData column to be treated
                         as a factor (repeatable)
  --sample-col   <col>   colData column holding sample ids (default: 1st col)
  --out          <path>  output TSV (default: results.tsv)
  --threads      <n>     worker threads (default: auto = cores, capped at 12)
  --dump-prefix  <path>  write intermediate diagnostics to <path>.*.tsv
  -h, --help             show this help

Notes:
  * factor columns follow R's level ordering (sorted levels, first level is
    the reference); numeric columns enter the model as continuous covariates
  * the contrast is handled as DESeq2's results(dds, contrast=c(var, case,
    control)): direct coefficient, negated coefficient, or a numeric contrast
    when neither level is the reference
"
    .to_string()
}

fn take(args: &[String], i: &mut usize) -> Result<String, String> {
    let flag = args[*i].clone();
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_args() -> Result<(String, String, String, Options), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut counts = None;
    let mut coldata = None;
    let mut design = None;
    let mut contrast_var = None;
    let mut case_level = None;
    let mut control_level = None;
    let mut sample_col = None;
    let mut factor_cols = Vec::new();
    let mut out = "results.tsv".to_string();
    let mut threads: usize = 0;
    let mut dump_prefix = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Err(usage()),
            "--counts" => counts = Some(take(&args, &mut i)?),
            "--coldata" => coldata = Some(take(&args, &mut i)?),
            "--design" => design = Some(take(&args, &mut i)?),
            "--contrast-var" => contrast_var = Some(take(&args, &mut i)?),
            "--contrast-case" => case_level = Some(take(&args, &mut i)?),
            "--contrast-control" => control_level = Some(take(&args, &mut i)?),
            "--sample-col" => sample_col = Some(take(&args, &mut i)?),
            "--factor" => factor_cols.push(take(&args, &mut i)?),
            "--out" => out = take(&args, &mut i)?,
            "--dump-prefix" => dump_prefix = Some(take(&args, &mut i)?),
            "--threads" => {
                threads = take(&args, &mut i)?
                    .parse()
                    .map_err(|_| "--threads must be a non-negative integer".to_string())?
            }
            other => return Err(format!("unknown argument: {other}\n\n{}", usage())),
        }
        i += 1;
    }

    let counts = counts.ok_or("--counts is required")?;
    let coldata = coldata.ok_or("--coldata is required")?;
    let design = design.ok_or("--design is required")?;
    let case_level = case_level.ok_or("--contrast-case is required")?;
    let control_level = control_level.ok_or("--contrast-control is required")?;
    let contrast_var = match contrast_var {
        Some(v) => v,
        None => design::parse_design_formula(&design)?
            .last()
            .cloned()
            .ok_or("empty design")?,
    };

    Ok((
        counts,
        coldata,
        out,
        Options {
            design,
            contrast_var,
            case_level,
            control_level,
            sample_col,
            factor_cols,
            threads,
            dump_prefix,
        },
    ))
}

fn run() -> Result<(), String> {
    let (counts_path, coldata_path, out_path, opts) = parse_args()?;

    let counts = io::read_counts(&counts_path)?;
    let coldata = io::read_coldata(&coldata_path)?;

    eprintln!(
        "[rust_deseq2] {} genes x {} samples; design {}; contrast {} = {} vs {}",
        counts.n_genes(),
        counts.n_samples(),
        opts.design,
        opts.contrast_var,
        opts.case_level,
        opts.control_level
    );

    let results = deseq::run(&counts, &coldata, &opts)?;
    io::write_results(&out_path, &results)?;
    eprintln!(
        "[rust_deseq2] wrote {} results to {}",
        results.len(),
        out_path
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
