//! The core DESeq2 workflow, wired together:
//!   1. median-of-ratios size factors
//!   2. a design matrix from the condition factor (control level = reference)
//!   3. Cox-Reid gene-wise dispersion + a parametric mean-dispersion trend
//!   4. an estimated log-normal prior and MAP (shrunken) dispersion per gene,
//!      with a dispersion-outlier carve-out
//!   5. a negative-binomial GLM Wald test for the case-vs-control contrast
//!   6. Benjamini-Hochberg adjusted p-values with independent filtering

use crate::glm::{estimate_dispersion, fit_nb_glm};
use crate::io::{ColData, CountMatrix, GeneResult};
use crate::linalg::{invert, Mat};
use crate::mathx::{benjamini_hochberg, mad, median, norm_two_sided_p, quantile, trigamma};
use std::fs::File;
use std::io::{BufWriter, Write};

/// Floor on the estimated dispersion-prior variance (natural-log scale),
/// matching DESeq2's default of 0.25.
const DISP_PRIOR_VAR_FLOOR: f64 = 0.25;

/// Target FDR used to optimise the independent-filtering threshold, matching
/// DESeq2's `results()` default `alpha = 0.1`.
const FILTER_ALPHA: f64 = 0.1;

/// Minimum dispersion (DESeq2's `minDisp`).
const MIN_DISP: f64 = 1e-8;

const LN2: f64 = std::f64::consts::LN_2;

pub struct Options {
    pub design_col: String,
    pub contrast_var: String,
    pub case_level: String,
    pub control_level: String,
    /// Column in colData holding sample ids (defaults to the first column).
    pub sample_col: Option<String>,
    /// Worker threads for the per-gene loops; 0 = auto (cores, capped at 16).
    pub threads: usize,
    /// Optional file prefix for intermediate parity diagnostics.
    pub dump_prefix: Option<String>,
}

/// Map `f` over gene indices `0..g`, splitting the work across `nthreads`
/// scoped worker threads (the per-gene fits are independent). Results are
/// returned in gene order. Uses only `std` — no external dependencies.
fn parallel_map<T, F>(g: usize, nthreads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if nthreads <= 1 || g <= 1 {
        return (0..g).map(f).collect();
    }
    let nthreads = nthreads.min(g);
    let chunk = g.div_ceil(nthreads);
    let parts: Vec<Vec<T>> = std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = (0..nthreads)
            .map(|t| {
                let start = t * chunk;
                let end = ((t + 1) * chunk).min(g);
                s.spawn(move || (start..end).map(f).collect::<Vec<T>>())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = Vec::with_capacity(g);
    for part in parts {
        out.extend(part);
    }
    out
}

/// Per-gene output of pass 1.
struct Pass1 {
    base_mean: f64,
    disp: f64,
    all_zero: bool,
    mu: Vec<f64>,
}

/// Per-gene diagnostics from pass 2, used only by `--dump-prefix`.
struct Pass2 {
    result: GeneResult,
    mu: Vec<f64>,
    disp_fit: f64,
    disp_map: f64,
    dispersion: f64,
    disp_outlier: bool,
    beta_conv: bool,
}

/// Pass 1 for a single gene: baseMean, fitted means and the Cox-Reid gene-wise
/// dispersion (seeded by the rough/moments initialiser).
#[allow(clippy::too_many_arguments)]
fn pass1_gene(
    y: &[f64],
    x: &[f64],
    offset: &[f64],
    sf: &[f64],
    inv_xtx: &Mat,
    p: usize,
    xim: f64,
    max_disp: f64,
) -> Pass1 {
    let n = y.len();
    let m = n as f64;
    let y_norm: Vec<f64> = (0..n).map(|i| y[i] / sf[i]).collect();
    let base_mean = y_norm.iter().sum::<f64>() / m;

    if y.iter().sum::<f64>() <= 0.0 {
        return Pass1 {
            base_mean,
            disp: f64::NAN,
            all_zero: true,
            mu: Vec::new(),
        };
    }
    let alpha_init = init_dispersion(&y_norm, x, inv_xtx, p, base_mean, xim, MIN_DISP, max_disp);
    let mut fit = fit_nb_glm(y, x, offset, p, alpha_init, 100);
    let mut disp = estimate_dispersion(y, &fit.mu, x, p, max_disp, None);
    for _ in 0..2 {
        fit = fit_nb_glm(y, x, offset, p, disp, 100);
        disp = estimate_dispersion(y, &fit.mu, x, p, max_disp, None);
    }
    Pass1 {
        base_mean,
        disp,
        all_zero: false,
        mu: fit.mu,
    }
}

/// Pass 2 for a single gene: MAP dispersion (with outlier carve-out) and the
/// negative-binomial GLM Wald test for the case-vs-control contrast.
#[allow(clippy::too_many_arguments)]
fn pass2_gene(
    gene: String,
    y: &[f64],
    x: &[f64],
    offset: &[f64],
    mu_init: &[f64],
    base_mean: f64,
    genewise_disp: f64,
    all_zero: bool,
    trend: f64,
    prior_var: f64,
    disp_outlier_sd: f64,
    p: usize,
    case_col: usize,
    max_disp: f64,
) -> Pass2 {
    if all_zero {
        return Pass2 {
            result: GeneResult {
                gene,
                base_mean: 0.0,
                log2_fold_change: f64::NAN,
                lfc_se: f64::NAN,
                stat: f64::NAN,
                pvalue: f64::NAN,
                padj: f64::NAN,
            },
            mu: Vec::new(),
            disp_fit: trend,
            disp_map: f64::NAN,
            dispersion: f64::NAN,
            disp_outlier: false,
            beta_conv: false,
        };
    }
    // MAP (shrunken) dispersion; gene-wise estimates > 2 raw residual SDs above
    // the trend are left unshrunk (DESeq2 dispersion-outlier rule).
    let map_disp = estimate_dispersion(y, mu_init, x, p, max_disp, Some((trend, prior_var)));
    let disp_outlier =
        genewise_disp.is_finite() && genewise_disp.ln() - trend.ln() > 2.0 * disp_outlier_sd;
    let final_disp = if disp_outlier {
        genewise_disp
    } else {
        map_disp
    };

    let fit = fit_nb_glm(y, x, offset, p, final_disp, 100);
    let var = fit.cov.get(case_col, case_col);
    let (lfc2, lfcse2, stat, pval) = if var.is_finite() && var > 0.0 {
        let beta = fit.beta[case_col];
        let se = var.sqrt();
        let stat = beta / se;
        (beta / LN2, se / LN2, stat, norm_two_sided_p(stat))
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };
    Pass2 {
        result: GeneResult {
            gene,
            base_mean,
            log2_fold_change: lfc2,
            lfc_se: lfcse2,
            stat,
            pvalue: pval,
            padj: f64::NAN,
        },
        mu: fit.mu,
        disp_fit: trend,
        disp_map: map_disp,
        dispersion: final_disp,
        disp_outlier,
        beta_conv: fit.converged,
    }
}

fn fmt_num(x: f64) -> String {
    if x.is_finite() {
        format!("{:.17e}", x)
    } else {
        "NA".to_string()
    }
}

fn write_diagnostics(
    prefix: &str,
    counts: &CountMatrix,
    sf: &[f64],
    base_means: &[f64],
    genewise_disp: &[f64],
    pass2: &[Pass2],
) -> Result<(), String> {
    let sf_path = format!("{prefix}.size_factors.tsv");
    let mut sfw = BufWriter::new(
        File::create(&sf_path).map_err(|e| format!("cannot write '{sf_path}': {e}"))?,
    );
    writeln!(sfw, "sample\tsizeFactor").map_err(|e| e.to_string())?;
    for (sample, size_factor) in counts.samples.iter().zip(sf) {
        writeln!(sfw, "{}\t{}", sample, fmt_num(*size_factor)).map_err(|e| e.to_string())?;
    }

    let gene_path = format!("{prefix}.genes.tsv");
    let mut gw = BufWriter::new(
        File::create(&gene_path).map_err(|e| format!("cannot write '{gene_path}': {e}"))?,
    );
    writeln!(
        gw,
        "gene\tbaseMean\tdispGeneEst\tdispFit\tdispMAP\tdispersion\tdispOutlier\tbetaConv\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj"
    )
    .map_err(|e| e.to_string())?;
    for gi in 0..counts.n_genes() {
        let p2 = &pass2[gi];
        let r = &p2.result;
        writeln!(
            gw,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.gene,
            fmt_num(base_means[gi]),
            fmt_num(genewise_disp[gi]),
            fmt_num(p2.disp_fit),
            fmt_num(p2.disp_map),
            fmt_num(p2.dispersion),
            p2.disp_outlier,
            p2.beta_conv,
            fmt_num(r.log2_fold_change),
            fmt_num(r.lfc_se),
            fmt_num(r.stat),
            fmt_num(r.pvalue),
            fmt_num(r.padj)
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn trimmed_mean(values: &[f64], trim: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cut = (v.len() as f64 * trim).floor() as usize;
    let lo = cut.min(v.len());
    let hi = v.len().saturating_sub(cut);
    if lo >= hi {
        return v.iter().sum::<f64>() / v.len() as f64;
    }
    v[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
}

fn trim_rule(n: usize) -> (f64, f64) {
    if n <= 3 {
        (1.0 / 3.0, 2.04)
    } else if n <= 23 {
        (1.0 / 4.0, 1.86)
    } else {
        (1.0 / 8.0, 1.51)
    }
}

fn hat_diagonal(x: &[f64], mu: &[f64], dispersion: f64, p: usize) -> Option<Vec<f64>> {
    let n = mu.len();
    let mut xtwx = Mat::zeros(p, p);
    let mut w = vec![0.0; n];
    for i in 0..n {
        w[i] = mu[i] / (1.0 + dispersion * mu[i]);
        for a in 0..p {
            let xa = x[i * p + a];
            for b in a..p {
                xtwx.add(a, b, w[i] * xa * x[i * p + b]);
            }
        }
    }
    for a in 0..p {
        for b in (a + 1)..p {
            let v = xtwx.get(a, b);
            xtwx.set(b, a, v);
        }
    }
    let inv = invert(&xtwx)?;
    let mut h = vec![0.0; n];
    for i in 0..n {
        let mut q = 0.0;
        for a in 0..p {
            for b in 0..p {
                q += x[i * p + a] * inv.get(a, b) * x[i * p + b];
            }
        }
        h[i] = w[i] * q;
    }
    Some(h)
}

fn cooks_cutoff_99(p: usize, residual_df: usize) -> Option<f64> {
    // The supported CLI design has two coefficients. For F(2, v),
    // CDF(x) = 1 - (v / (v + 2x))^(v/2), so the 0.99 quantile is analytic.
    if p != 2 || residual_df == 0 {
        return None;
    }
    let v = residual_df as f64;
    Some(v * (0.01_f64.powf(-2.0 / v) - 1.0) / 2.0)
}

fn replace_cooks_outliers(
    counts: &CountMatrix,
    sf: &[f64],
    x: &[f64],
    p: usize,
    levels_per_sample: &[String],
    pass2: &[Pass2],
) -> Option<CountMatrix> {
    let n = counts.n_samples();
    let g = counts.n_genes();
    let cutoff = cooks_cutoff_99(p, n.checked_sub(p)?)?;

    let mut group_sizes = std::collections::HashMap::<&str, usize>::new();
    for level in levels_per_sample {
        *group_sizes.entry(level.as_str()).or_insert(0) += 1;
    }
    let replaceable: Vec<bool> = levels_per_sample
        .iter()
        .map(|level| group_sizes.get(level.as_str()).copied().unwrap_or(0) >= 7)
        .collect();
    if !replaceable.iter().any(|&x| x) {
        return None;
    }

    let mut replaced = counts.clone();
    let mut changed = false;
    for (gi, p2) in pass2.iter().enumerate().take(g) {
        let row = counts.row(gi);
        if p2.mu.is_empty() || !p2.dispersion.is_finite() {
            continue;
        }

        let norm: Vec<f64> = (0..n).map(|i| row[i] / sf[i]).collect();
        let base_mean = norm.iter().sum::<f64>() / n as f64;
        if base_mean <= 0.0 {
            continue;
        }

        // DESeq2 robustMethodOfMomentsDisp for designs with >=3 samples per cell.
        let mut cell_vars = Vec::new();
        for level in group_sizes.keys() {
            let idx: Vec<usize> = levels_per_sample
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if v == level { Some(i) } else { None })
                .collect();
            if idx.len() < 3 {
                continue;
            }
            let (trim, scale) = trim_rule(idx.len());
            let vals: Vec<f64> = idx.iter().map(|&i| norm[i]).collect();
            let cell_mean = trimmed_mean(&vals, trim);
            let sqerr: Vec<f64> = idx.iter().map(|&i| (norm[i] - cell_mean).powi(2)).collect();
            cell_vars.push(scale * trimmed_mean(&sqerr, trim));
        }
        if cell_vars.is_empty() {
            continue;
        }
        let robust_var = cell_vars.into_iter().fold(0.0_f64, f64::max);
        let robust_disp = ((robust_var - base_mean) / (base_mean * base_mean)).max(0.04);

        let h = match hat_diagonal(x, &p2.mu, p2.dispersion, p) {
            Some(v) => v,
            None => continue,
        };
        let trim_base_mean = trimmed_mean(&norm, 0.2);
        for i in 0..n {
            if !replaceable[i] {
                continue;
            }
            let v = p2.mu[i] + robust_disp * p2.mu[i] * p2.mu[i];
            let denom = (1.0 - h[i]).powi(2);
            if v <= 0.0 || denom <= 0.0 {
                continue;
            }
            let pearson_sq = (row[i] - p2.mu[i]).powi(2) / v;
            let cooks = pearson_sq / p as f64 * h[i] / denom;
            if cooks > cutoff {
                let new_count = (trim_base_mean * sf[i]).trunc();
                let pos = gi * n + i;
                if (replaced.counts[pos] - new_count).abs() > 0.0 {
                    replaced.counts[pos] = new_count;
                    changed = true;
                }
            }
        }
    }

    if changed {
        Some(replaced)
    } else {
        None
    }
}

/// DESeq2 median-of-ratios size factors.
fn size_factors(counts: &CountMatrix) -> Vec<f64> {
    let g = counts.n_genes();
    let n = counts.n_samples();
    // Per-gene log geometric mean, using only genes with all-positive counts.
    let mut loggeom = vec![f64::NEG_INFINITY; g];
    for (gi, loggeom_i) in loggeom.iter_mut().enumerate() {
        let row = counts.row(gi);
        let mut sum_log = 0.0;
        let mut ok = true;
        for &c in row {
            if c <= 0.0 {
                ok = false;
                break;
            }
            sum_log += c.ln();
        }
        if ok {
            *loggeom_i = sum_log / n as f64;
        }
    }

    let mut sf = vec![1.0; n];
    for (s, sf_s) in sf.iter_mut().enumerate() {
        let mut ratios = Vec::with_capacity(g);
        for (gi, &loggeom_i) in loggeom.iter().enumerate() {
            if loggeom_i.is_finite() {
                let c = counts.counts[gi * n + s];
                if c > 0.0 {
                    ratios.push(c.ln() - loggeom_i);
                }
            }
        }
        *sf_s = if ratios.is_empty() {
            1.0
        } else {
            median(&ratios).exp()
        };
    }
    sf
}

/// Build the model matrix. Reference level = the control level, so the case
/// level's coefficient is directly the log fold change of case vs control.
/// Returns (matrix row-major n x p, p, case coefficient index, level order).
fn build_design(
    levels_per_sample: &[String],
    control: &str,
    case: &str,
) -> Result<(Vec<f64>, usize, usize), String> {
    // Unique levels present, control first (reference), rest sorted.
    let mut uniq: Vec<String> = Vec::new();
    for l in levels_per_sample {
        if !uniq.contains(l) {
            uniq.push(l.clone());
        }
    }
    if !uniq.iter().any(|l| l == control) {
        return Err(format!(
            "control level '{control}' not present in design column"
        ));
    }
    if !uniq.iter().any(|l| l == case) {
        return Err(format!("case level '{case}' not present in design column"));
    }
    if case == control {
        return Err("case and control levels must differ".into());
    }
    // Order: control (reference) first, then the remaining levels sorted.
    let mut rest: Vec<String> = uniq
        .iter()
        .filter(|l| l.as_str() != control)
        .cloned()
        .collect();
    rest.sort();
    let mut order = vec![control.to_string()];
    order.extend(rest);

    let n = levels_per_sample.len();
    let p = order.len(); // intercept + (n_levels - 1) dummies
                         // Coefficient index for a level: reference -> intercept only (no own col);
                         // non-reference level k (1-based in `order`) -> column k.
    let mut level_col = std::collections::HashMap::new();
    for (k, lv) in order.iter().enumerate() {
        level_col.insert(lv.clone(), k); // k==0 is the reference (intercept)
    }
    let case_col = *level_col.get(case).unwrap();

    let mut x = vec![0.0; n * p];
    for i in 0..n {
        x[i * p] = 1.0; // intercept
        let k = *level_col.get(&levels_per_sample[i]).unwrap();
        if k > 0 {
            x[i * p + k] = 1.0;
        }
    }
    Ok((x, p, case_col))
}

/// DESeq2-style gene-wise dispersion initialization used to seed the Cox-Reid
/// optimization. It combines two crude estimators and takes the smaller:
///
/// * **rough** (`roughDispEstimate`): from an ordinary *linear* (not log-link)
///   least-squares fit `mu_ols` of the normalized counts on the design (floored
///   at 1), the moment estimate `sum((y-mu)^2 - mu)/mu^2 / (m-p)`.
/// * **moments** (`momentsDispEstimate`): `(baseVar - mean(1/sf)*baseMean)/baseMean^2`.
///
/// `min(rough, moments)` is then bounded to `[minDisp, maxDisp]`. A better start
/// makes the optimizer land on DESeq2's estimate for hard/high-dispersion genes.
#[allow(clippy::too_many_arguments)]
fn init_dispersion(
    y_norm: &[f64],
    x: &[f64],
    inv_xtx: &Mat,
    p: usize,
    base_mean: f64,
    xim: f64,
    min_disp: f64,
    max_disp: f64,
) -> f64 {
    let n = y_norm.len();
    let (m, pf) = (n as f64, p as f64);

    // OLS coefficients: beta = (X'X)^-1 X' y_norm.
    let mut xty = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            xty[j] += x[i * p + j] * y_norm[i];
        }
    }
    let mut beta = vec![0.0; p];
    for (j, beta_j) in beta.iter_mut().enumerate().take(p) {
        let mut s = 0.0;
        for (k, &xty_k) in xty.iter().enumerate().take(p) {
            s += inv_xtx.get(j, k) * xty_k;
        }
        *beta_j = s;
    }

    // rough dispersion from the linear-model residuals (mu floored at 1).
    let rough = if m > pf {
        let mut acc = 0.0;
        for i in 0..n {
            let mut mu = 0.0;
            for j in 0..p {
                mu += x[i * p + j] * beta[j];
            }
            let mu = mu.max(1.0);
            acc += ((y_norm[i] - mu).powi(2) - mu) / (mu * mu);
        }
        (acc / (m - pf)).max(0.0)
    } else {
        min_disp
    };

    // moments dispersion from the per-gene variance of normalized counts.
    let base_var = if n > 1 {
        let ss: f64 = y_norm.iter().map(|&v| (v - base_mean).powi(2)).sum();
        ss / (m - 1.0)
    } else {
        0.0
    };
    let moments = if base_mean > 0.0 {
        (base_var - xim * base_mean) / (base_mean * base_mean)
    } else {
        f64::INFINITY
    };

    // min(rough, moments); a negative moments collapses to minDisp via the clamp.
    rough.min(moments).clamp(min_disp, max_disp)
}

/// Fit a parametric dispersion trend `disp = a0 + a1 / mean` (a0 = asymptotic
/// dispersion, a1 = extra-Poisson term), reproducing DESeq2's
/// `parametricDispersionFit`: an iteratively reweighted Gamma GLM with an
/// identity link, refit until the coefficients converge, rejecting points whose
/// ratio of observed to fitted dispersion falls outside (1e-4, 15).
///
/// The Gamma family's constant-CV weighting (`w = 1/mu^2`) is what makes this
/// differ from ordinary least squares: OLS is dominated by the large-dispersion
/// low-mean genes and over-estimates the trend as `mean -> 0`, which in turn
/// inflates the shrunken dispersion (and hence the standard error) of very
/// low-count genes. The Gamma fit keeps that extrapolation in line with DESeq2.
fn fit_dispersion_trend(base_means: &[f64], disps: &[f64]) -> (f64, f64) {
    // Points (1/mean, disp) over informative genes. DESeq2 selects genes whose
    // gene-wise dispersion sits clearly above the minimum (dispGeneEst >
    // 100 * minDisp, with minDisp = 1e-8), i.e. > 1e-6 — it does NOT impose a
    // baseMean cutoff, so low-count genes still inform the extrapolation.
    let pts: Vec<(f64, f64)> = (0..base_means.len())
        .filter(|&i| base_means[i] > 0.0 && disps[i].is_finite() && disps[i] > 1e-6)
        .map(|i| (1.0 / base_means[i], disps[i]))
        .collect();

    let median_fallback = || -> (f64, f64) {
        let med = median(disps);
        (if med.is_finite() { med.max(1e-4) } else { 0.1 }, 0.0)
    };
    if pts.len() < 3 {
        return median_fallback();
    }

    // Iteratively reweighted Gamma-GLM (identity link): mu = a0 + a1*x,
    // weights w = 1/mu^2, working response = disp (identity link).
    let (mut a0, mut a1) = (0.1, 1.0);
    let mut ok = false; // at least one valid IRLS update produced positive coefs
    for _iter in 0..10 {
        let (mut s00, mut s01, mut s11, mut s0y, mut s1y) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut used = 0;
        for &(x, d) in &pts {
            let mu = a0 + a1 * x;
            if mu <= 0.0 {
                continue;
            }
            // Reject genes far from the current fit (DESeq2's residual gate).
            let ratio = d / mu;
            if !(ratio > 1e-4 && ratio < 15.0) {
                continue;
            }
            let w = 1.0 / (mu * mu);
            s00 += w;
            s01 += w * x;
            s11 += w * x * x;
            s0y += w * d;
            s1y += w * x * d;
            used += 1;
        }
        if used < 3 {
            break;
        }
        let det = s00 * s11 - s01 * s01;
        if det.abs() < 1e-30 {
            break;
        }
        let na0 = (s0y * s11 - s1y * s01) / det;
        let na1 = (s00 * s1y - s01 * s0y) / det;
        // Coefficients must stay positive; otherwise keep the last valid fit.
        if !(na0 > 0.0 && na1 > 0.0 && na0.is_finite() && na1.is_finite()) {
            break;
        }
        let conv = (na0 / a0).ln().powi(2) + (na1 / a1).ln().powi(2);
        a0 = na0;
        a1 = na1;
        ok = true;
        if conv < 1e-6 {
            break;
        }
    }
    // DESeq2 declares the parametric fit failed if coefficients go non-positive
    // and switches to a local (locfit) dispersion fit. We don't implement the
    // local fit; if no valid parametric update was obtained we fall back to a
    // flat trend at the median dispersion (a documented, coarser substitute).
    if ok && a0 > 0.0 && a1 > 0.0 {
        (a0, a1)
    } else {
        median_fallback()
    }
}

/// Run the full workflow, returning one result per gene (input gene order).
pub fn run(
    counts: &CountMatrix,
    coldata: &ColData,
    opts: &Options,
) -> Result<Vec<GeneResult>, String> {
    run_inner(counts, coldata, opts, None, true)
}

fn run_inner(
    counts: &CountMatrix,
    coldata: &ColData,
    opts: &Options,
    fixed_size_factors: Option<&[f64]>,
    allow_outlier_replacement: bool,
) -> Result<Vec<GeneResult>, String> {
    let n = counts.n_samples();

    // --- align samples: count-matrix column order is authoritative ---
    let sample_ids = match &opts.sample_col {
        Some(name) => coldata.column(name)?.clone(),
        None => coldata.sample_ids.clone(),
    };
    let design_vals = coldata.column(&opts.design_col)?;

    let mut levels_per_sample = Vec::with_capacity(n);
    for s in &counts.samples {
        let idx = sample_ids
            .iter()
            .position(|id| id == s)
            .ok_or_else(|| format!("sample '{s}' from counts not found in colData"))?;
        levels_per_sample.push(design_vals[idx].clone());
    }

    if opts.contrast_var != opts.design_col {
        eprintln!(
            "[rust_deseq2] note: contrast variable '{}' differs from design column '{}'; \
             using '{}' as the model factor.",
            opts.contrast_var, opts.design_col, opts.design_col
        );
    }

    let (x, p, case_col) = build_design(&levels_per_sample, &opts.control_level, &opts.case_level)?;

    // --- size factors and offsets ---
    let sf = fixed_size_factors
        .map(|v| v.to_vec())
        .unwrap_or_else(|| size_factors(counts));
    let offset: Vec<f64> = sf.iter().map(|s| s.ln()).collect();

    let g = counts.n_genes();
    let m = n as f64; // samples
    let pf = p as f64; // coefficients

    // Dispersion bounds (DESeq2): minDisp = 1e-8, maxDisp = max(10, n_samples).
    let max_disp = 10.0_f64.max(m);
    // Worker threads for the per-gene loops.
    let nthreads = if opts.threads > 0 {
        opts.threads
    } else {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .min(16)
    };
    // Mean reciprocal size factor, for the moments dispersion initialiser.
    let xim = sf.iter().map(|s| 1.0 / s).sum::<f64>() / m;
    // Inverse of X'X for the linear-model (OLS) rough dispersion initialiser.
    let mut xtx = Mat::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                xtx.add(a, b, x[i * p + a] * x[i * p + b]);
            }
        }
    }
    let inv_xtx = invert(&xtx).ok_or("design matrix X'X is singular")?;

    // --- pass 1 (parallel): base means, fitted means, gene-wise dispersion ---
    let p1 = parallel_map(g, nthreads, |gi| {
        pass1_gene(counts.row(gi), &x, &offset, &sf, &inv_xtx, p, xim, max_disp)
    });
    let base_means: Vec<f64> = p1.iter().map(|r| r.base_mean).collect();
    let genewise_disp: Vec<f64> = p1.iter().map(|r| r.disp).collect();
    let all_zero: Vec<bool> = p1.iter().map(|r| r.all_zero).collect();
    let mu_cache: Vec<Vec<f64>> = p1.into_iter().map(|r| r.mu).collect();

    // --- fit the mean-dispersion trend ---
    let (a0, a1) = fit_dispersion_trend(&base_means, &genewise_disp);
    let trend_at = |bm: f64| -> f64 { (a0 + a1 / bm.max(1e-8)).max(1e-8) };

    // --- estimate the log-normal dispersion-prior variance ---
    // Residuals of log gene-wise dispersion around the trend, over genes whose
    // gene-wise estimate is above the minimum (dispGeneEst >= 100*minDisp = 1e-6;
    // DESeq2's `aboveMinDisp`, keyed to dispersion, NOT baseMean). The robust
    // (MAD) spread minus the expected sampling variance gives the prior variance.
    let mut resid = Vec::new();
    for gi in 0..g {
        let d = genewise_disp[gi];
        if d.is_finite() && d >= 1e-6 {
            resid.push(d.ln() - trend_at(base_means[gi]).ln());
        }
    }
    let var_log_disp = if resid.len() >= 3 {
        let s = mad(&resid);
        if s.is_finite() {
            s * s
        } else {
            0.0
        }
    } else {
        0.0
    };
    let exp_var = if m - pf > 0.0 {
        trigamma((m - pf) / 2.0)
    } else {
        1.0
    };
    let prior_var = (var_log_disp - exp_var).max(DISP_PRIOR_VAR_FLOOR);
    // Dispersion-outlier cutoff uses the *raw* spread of the log-dispersion
    // estimates (varLogDispEsts), before subtracting the expected sampling
    // variance and flooring — DESeq2 flags outliers with 2 * sqrt(varLogDispEsts),
    // which is a wider band than the shrinkage prior SD.
    let disp_outlier_sd = if var_log_disp > 0.0 {
        var_log_disp.sqrt()
    } else {
        f64::INFINITY // too few genes to judge outliers: don't flag any
    };

    // --- pass 2 (parallel): MAP dispersion + NB GLM Wald test ---
    let mut pass2 = parallel_map(g, nthreads, |gi| {
        pass2_gene(
            counts.genes[gi].clone(),
            counts.row(gi),
            &x,
            &offset,
            &mu_cache[gi],
            base_means[gi],
            genewise_disp[gi],
            all_zero[gi],
            trend_at(base_means[gi]),
            prior_var,
            disp_outlier_sd,
            p,
            case_col,
            max_disp,
        )
    });

    if allow_outlier_replacement {
        if let Some(replaced_counts) =
            replace_cooks_outliers(counts, &sf, &x, p, &levels_per_sample, &pass2)
        {
            eprintln!("[rust_deseq2] Cook's outlier counts replaced; refitting");
            return run_inner(&replaced_counts, coldata, opts, Some(&sf), false);
        }
    }

    let mut results: Vec<GeneResult> = pass2
        .iter()
        .map(|p2| GeneResult {
            gene: p2.result.gene.clone(),
            base_mean: p2.result.base_mean,
            log2_fold_change: p2.result.log2_fold_change,
            lfc_se: p2.result.lfc_se,
            stat: p2.result.stat,
            pvalue: p2.result.pvalue,
            padj: p2.result.padj,
        })
        .collect();

    // --- BH adjustment with independent filtering on baseMean ---
    let pvals: Vec<f64> = results.iter().map(|r| r.pvalue).collect();
    let padj = independent_filter(&base_means, &pvals, FILTER_ALPHA);
    for gi in 0..g {
        results[gi].padj = padj[gi];
        pass2[gi].result.padj = padj[gi];
    }

    if let Some(prefix) = &opts.dump_prefix {
        write_diagnostics(prefix, counts, &sf, &base_means, &genewise_disp, &pass2)?;
    }

    Ok(results)
}

/// DESeq2-style independent filtering: low-count genes carry little power, so
/// filtering them out before multiple-testing correction raises overall power.
///
/// This follows DESeq2's `results()` procedure exactly: 50 candidate `theta`
/// from `mean(filter == 0)` to 0.95 (or 1), a baseMean cutoff at each quantile,
/// the rejection count at `alpha`, a LOWESS (f = 1/5) smooth of that curve to
/// set the RMSE threshold `maxFit - rmse`, then the first `theta` whose *raw*
/// rejection count exceeds that threshold (backing off to 0.9*maxFit then
/// 0.8*maxFit). Filtering is disabled when `max(numRej) <= 10`. Filtered-out
/// genes get `padj = NA`.
fn independent_filter(base_means: &[f64], pvals: &[f64], alpha: f64) -> Vec<f64> {
    let g = pvals.len();
    if !(0..g).any(|i| pvals[i].is_finite()) {
        return vec![f64::NAN; g];
    }
    // Filter statistic quantiles are taken over ALL genes (DESeq2 uses the full
    // baseMean vector); genes with NA p-values simply stay NA after masking.
    let mut all_bm: Vec<f64> = base_means.to_vec();
    all_bm.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_q = base_means.iter().filter(|&&v| v == 0.0).count() as f64 / g as f64;
    let upper_q = if lower_q < 0.95 { 0.95 } else { 1.0 };
    const NTHETA: usize = 50;
    let thetas: Vec<f64> = (0..NTHETA)
        .map(|k| lower_q + (upper_q - lower_q) * (k as f64) / (NTHETA as f64 - 1.0))
        .collect();

    // Rejection count and the padj vector at each theta.
    let mut num_rej = vec![0.0; NTHETA];
    let mut padj_by_theta: Vec<Vec<f64>> = Vec::with_capacity(NTHETA);
    for (ti, &theta) in thetas.iter().enumerate() {
        let cut = quantile(&all_bm, theta);
        let masked: Vec<f64> = (0..g)
            .map(|i| {
                if pvals[i].is_finite() && base_means[i] >= cut {
                    pvals[i]
                } else {
                    f64::NAN
                }
            })
            .collect();
        let padj = benjamini_hochberg(&masked);
        num_rej[ti] = padj.iter().filter(|v| v.is_finite() && **v < alpha).count() as f64;
        padj_by_theta.push(padj);
    }

    let max_rej = num_rej.iter().cloned().fold(0.0_f64, f64::max);
    let j = if max_rej <= 10.0 {
        // Too few rejections to justify filtering: keep the lowest threshold.
        0
    } else {
        let smoothed = lowess(&thetas, &num_rej, 0.2, 3);
        let max_fit = smoothed.iter().cloned().fold(f64::MIN, f64::max);
        // RMSE of the raw counts around the smoothed curve, over thetas with at
        // least one rejection (DESeq2's noise estimate).
        let (mut ss, mut c) = (0.0, 0.0);
        for k in 0..NTHETA {
            if num_rej[k] > 0.0 {
                ss += (num_rej[k] - smoothed[k]).powi(2);
                c += 1.0;
            }
        }
        let rmse = if c > 0.0 { (ss / c).sqrt() } else { 0.0 };
        let thresh = max_fit - rmse;
        // DESeq2 selects on the RAW rejection counts (not the smoothed curve):
        // first theta with numRej > thresh, backing off to 0.9*maxFit then
        // 0.8*maxFit if a late uptick with little variation defeats the RMSE
        // threshold. Falls back to the lowest threshold if none qualify.
        (0..NTHETA)
            .find(|&k| num_rej[k] > thresh)
            .or_else(|| (0..NTHETA).find(|&k| num_rej[k] > 0.9 * max_fit))
            .or_else(|| (0..NTHETA).find(|&k| num_rej[k] > 0.8 * max_fit))
            .unwrap_or(0)
    };

    padj_by_theta.into_iter().nth(j).unwrap()
}

/// LOWESS scatterplot smoother (Cleveland 1979), matching R's `lowess`:
/// local linear regression over the nearest `f * n` points with tricube
/// distance weights, refined by `iter` bisquare robustness passes. `x` must be
/// sorted ascending. Used to smooth the independent-filtering rejection curve.
fn lowess(x: &[f64], y: &[f64], f: f64, iter: usize) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return y.to_vec();
    }
    let r = ((f * n as f64).ceil() as usize).clamp(2, n);
    let mut fitted = vec![0.0; n];
    let mut rw = vec![1.0; n]; // robustness weights

    for pass in 0..=iter {
        for i in 0..n {
            // r nearest neighbours of x[i] (indices), by absolute distance.
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                (x[a] - x[i])
                    .abs()
                    .partial_cmp(&(x[b] - x[i]).abs())
                    .unwrap()
            });
            let nb = &idx[0..r];
            let h = nb.iter().map(|&j| (x[j] - x[i]).abs()).fold(0.0, f64::max);

            let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &j in nb {
                let d = if h > 0.0 {
                    (x[j] - x[i]).abs() / h
                } else {
                    0.0
                };
                let tri = if d < 1.0 {
                    let t = 1.0 - d * d * d;
                    t * t * t
                } else {
                    0.0
                };
                let w = tri * rw[j];
                sw += w;
                swx += w * x[j];
                swy += w * y[j];
                swxx += w * x[j] * x[j];
                swxy += w * x[j] * y[j];
            }
            let denom = sw * swxx - swx * swx;
            fitted[i] = if sw <= 0.0 {
                y[i]
            } else if denom.abs() < 1e-12 {
                swy / sw // degenerate neighbourhood: weighted mean
            } else {
                let b = (sw * swxy - swx * swy) / denom;
                let a = (swy - b * swx) / sw;
                a + b * x[i]
            };
        }

        if pass < iter {
            // Bisquare robustness weights from the current residuals.
            let res: Vec<f64> = (0..n).map(|k| (y[k] - fitted[k]).abs()).collect();
            let s = median(&res);
            if s <= 0.0 {
                break;
            }
            for k in 0..n {
                let u = (y[k] - fitted[k]).abs() / (6.0 * s);
                rw[k] = if u < 1.0 {
                    let t = 1.0 - u * u;
                    t * t
                } else {
                    0.0
                };
            }
        }
    }
    fitted
}
