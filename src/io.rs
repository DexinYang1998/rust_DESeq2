//! Tab-separated I/O for the count matrix, sample metadata and results table.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// A gene-by-sample count matrix.
#[derive(Clone)]
pub struct CountMatrix {
    pub genes: Vec<String>,
    pub samples: Vec<String>,
    /// Row-major: `counts[g * n_samples + s]`.
    pub counts: Vec<f64>,
}

impl CountMatrix {
    pub fn n_genes(&self) -> usize {
        self.genes.len()
    }
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }
    #[inline]
    pub fn row(&self, g: usize) -> &[f64] {
        let n = self.samples.len();
        &self.counts[g * n..(g + 1) * n]
    }
}

fn split_tab(line: &str) -> Vec<&str> {
    line.trim_end_matches(['\r', '\n']).split('\t').collect()
}

/// Read a TSV count matrix. First row = header (first cell is the gene-id
/// column label, remaining cells are sample ids). Each subsequent row starts
/// with a gene id followed by integer/float counts.
pub fn read_counts(path: &str) -> Result<CountMatrix, String> {
    let f = File::open(path).map_err(|e| format!("cannot open counts '{path}': {e}"))?;
    let mut reader = BufReader::new(f);
    let mut header = String::new();
    reader
        .read_line(&mut header)
        .map_err(|e| format!("reading counts header: {e}"))?;
    let hcells = split_tab(&header);
    if hcells.len() < 2 {
        return Err("count matrix header needs a gene column and >=1 sample".into());
    }
    let samples: Vec<String> = hcells[1..].iter().map(|s| s.to_string()).collect();
    let n = samples.len();

    let mut genes = Vec::new();
    let mut counts = Vec::new();
    for (li, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("reading counts line {}: {e}", li + 2))?;
        if line.trim().is_empty() {
            continue;
        }
        let cells = split_tab(&line);
        if cells.len() != n + 1 {
            return Err(format!(
                "counts line {} has {} fields, expected {}",
                li + 2,
                cells.len(),
                n + 1
            ));
        }
        genes.push(cells[0].to_string());
        for c in &cells[1..] {
            let v: f64 = c
                .trim()
                .parse()
                .map_err(|_| format!("non-numeric count '{c}' on line {}", li + 2))?;
            counts.push(v);
        }
    }
    Ok(CountMatrix {
        genes,
        samples,
        counts,
    })
}

/// Sample metadata: a header row of column names and one row per sample. The
/// first column holds sample ids.
pub struct ColData {
    pub columns: Vec<String>,
    pub sample_ids: Vec<String>,
    /// `values[col][sample]`.
    pub values: Vec<Vec<String>>,
}

impl ColData {
    /// Look up the values of a named column, indexed by sample position.
    pub fn column(&self, name: &str) -> Result<&Vec<String>, String> {
        let idx = self
            .columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| format!("design/contrast column '{name}' not found in colData"))?;
        Ok(&self.values[idx])
    }
}

pub fn read_coldata(path: &str) -> Result<ColData, String> {
    let f = File::open(path).map_err(|e| format!("cannot open colData '{path}': {e}"))?;
    let reader = BufReader::new(f);
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or("empty colData")?
        .map_err(|e| format!("reading colData header: {e}"))?;
    let columns: Vec<String> = split_tab(&header).iter().map(|s| s.to_string()).collect();
    let ncol = columns.len();

    let mut sample_ids = Vec::new();
    let mut values: Vec<Vec<String>> = vec![Vec::new(); ncol];
    for (li, line) in lines.enumerate() {
        let line = line.map_err(|e| format!("reading colData line {}: {e}", li + 2))?;
        if line.trim().is_empty() {
            continue;
        }
        let cells = split_tab(&line);
        if cells.len() != ncol {
            return Err(format!(
                "colData line {} has {} fields, expected {}",
                li + 2,
                cells.len(),
                ncol
            ));
        }
        sample_ids.push(cells[0].to_string());
        for (c, cell) in cells.iter().enumerate() {
            values[c].push(cell.to_string());
        }
    }
    Ok(ColData {
        columns,
        sample_ids,
        values,
    })
}

/// A single gene's differential-expression result.
pub struct GeneResult {
    pub gene: String,
    pub base_mean: f64,
    pub log2_fold_change: f64,
    pub lfc_se: f64,
    pub stat: f64,
    pub pvalue: f64,
    pub padj: f64,
}

fn fmt(x: f64) -> String {
    if x.is_nan() {
        "NA".to_string()
    } else if x == 0.0 {
        "0".to_string()
    } else {
        // Full round-trippable precision in scientific notation. Fixed 6-decimal
        // output silently collapsed tiny p-values and SEs to 0, which both loses
        // information and makes validation against DESeq2 look worse than it is;
        // `read.delim()` in R parses this form without any loss.
        format!("{x:.17e}")
    }
}

/// Write the results as a DESeq2-style TSV. The first column header is empty so
/// that `read.delim(row.names = 1)` in R treats it as row names, but we also
/// name it `gene` for clarity when read without row.names.
pub fn write_results(path: &str, results: &[GeneResult]) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("cannot write '{path}': {e}"))?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "gene\tbaseMean\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj"
    )
    .map_err(|e| e.to_string())?;
    for r in results {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.gene,
            fmt(r.base_mean),
            fmt(r.log2_fold_change),
            fmt(r.lfc_se),
            fmt(r.stat),
            fmt(r.pvalue),
            fmt(r.padj),
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}
