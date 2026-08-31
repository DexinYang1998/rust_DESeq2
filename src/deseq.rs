//! The DESeq2 workflow, staged exactly as `DESeq()` runs it:
//!
//!   1. median-of-ratios size factors
//!   2. model matrix from the design formula (factors + numeric covariates)
//!   3. gene-wise Cox-Reid dispersion (`estimateDispersionsGeneEst`):
//!      rough/moments initial value, mu via linear model or NB GLM, one
//!      Armijo line-search fit with the no-increase revert and the grid
//!      fallback for non-converged genes
//!   4. parametric mean-dispersion trend (Gamma GLM, identity link) and the
//!      log-normal prior variance
//!   5. MAP dispersions with the outlier carve-out
//!   6. NB GLM Wald test with contrast handling as in `results()`
//!   7. Cook's-distance outlier replacement (refit of the replaced genes
//!      only) and Cook's p-value flagging
//!   8. independent filtering (exact R `lowess`) + Benjamini-Hochberg

use crate::design::{build_design, parse_design_formula, Contrast, Design};
use crate::glm::{fit_disp, fit_disp_grid, fit_nb_glm, optim_beta};
use crate::io::{ColData, CountMatrix, GeneResult};
use crate::linalg::{invert, Mat};
use crate::mathx::{
    benjamini_hochberg, mad, median, qf, quantile, r_lowess, trigamma, trimmed_mean, wald_pvalue,
};
use crate::rloess::Loess1D;
use crate::rrand::RRng;
use std::fs::File;
use std::io::{BufWriter, Write};

const MIN_DISP: f64 = 1e-8;
const MINMU: f64 = 0.5;
const KAPPA_0: f64 = 1.0;
const DISP_TOL: f64 = 1e-6;
const BETA_TOL: f64 = 1e-8;
const MAX_IT: usize = 100;
const FILTER_ALPHA: f64 = 0.1;
const MIN_REPLICATES_FOR_REPLACE: usize = 7;
const LN2: f64 = std::f64::consts::LN_2;
/// log2(e)
const LOG2E: f64 = std::f64::consts::LOG2_E;

pub struct Options {
    /// Design: a formula "~ a + b + cond" or a single column name.
    pub design: String,
    pub contrast_var: String,
    pub case_level: String,
    pub control_level: String,
    pub sample_col: Option<String>,
    /// Columns to force as factors even if their values are numeric.
    pub factor_cols: Vec<String>,
    pub threads: usize,
    pub dump_prefix: Option<String>,
}

/// Map `f` over `items`, splitting across `nthreads` scoped threads.
fn parallel_map<I, T, F>(items: Vec<I>, nthreads: usize, f: F) -> Vec<T>
where
    I: Send + Sync + Copy,
    T: Send,
    F: Fn(I) -> T + Sync,
{
    let g = items.len();
    if nthreads <= 1 || g <= 1 {
        return items.into_iter().map(f).collect();
    }
    let nthreads = nthreads.min(g);
    let chunk = g.div_ceil(nthreads);
    let parts: Vec<Vec<T>> = std::thread::scope(|s| {
        let f = &f;
        let items = &items;
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let start = t * chunk;
                if start >= g { return None; }
                let end = ((t + 1) * chunk).min(g);
                Some(s.spawn(move || items[start..end].iter().map(|&it| f(it)).collect::<Vec<T>>()))
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

/// DESeq2 median-of-ratios size factors.
fn size_factors(counts: &CountMatrix) -> Result<Vec<f64>, String> {
    let g = counts.n_genes();
    let n = counts.n_samples();
    let mut loggeom = vec![f64::NEG_INFINITY; g];
    for (gi, lg) in loggeom.iter_mut().enumerate() {
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
            *lg = sum_log / n as f64;
        }
    }
    if loggeom.iter().all(|v| !v.is_finite()) {
        return Err("every gene contains at least one zero, cannot compute log geometric means".into());
    }
    let mut sf = vec![1.0; n];
    for (s, sf_s) in sf.iter_mut().enumerate() {
        let mut ratios = Vec::new();
        for (gi, &lg) in loggeom.iter().enumerate() {
            if lg.is_finite() {
                let c = counts.counts[gi * n + s];
                if c > 0.0 {
                    ratios.push(c.ln() - lg);
                }
            }
        }
        *sf_s = median(&ratios).exp();
    }
    Ok(sf)
}

/// Per-gene output of the gene-wise dispersion stage.
struct GeneEst {
    alpha_init: f64,
    /// mu floored at minmu — reused for the MAP fit (assays[["mu"]]).
    mu: Vec<f64>,
    disp_gene_est: f64,
}

/// Per-gene output of the Wald stage.
struct WaldOut {
    /// log2 coefficients and their SEs for the full model.
    lfc_log2: f64,
    se_log2: f64,
    stat: f64,
    pvalue: f64,
    cooks: Vec<f64>,
    max_cooks: f64,
    beta_conv: bool,
    disp_map: f64,
    dispersion: f64,
    disp_outlier: bool,
}

struct Shared<'a> {
    x: &'a [f64],
    p: usize,
    n: usize,
    sf: &'a [f64],
    inv_xtx: &'a Mat,
    linear_mu: bool,
    max_disp: f64,
    xim: f64,
    lambda_nat: Vec<f64>,
    lambda_log2: Vec<f64>,
}

/// OLS coefficients (X'X)^-1 X' v.
fn ols_coef(shared: &Shared, v: &[f64]) -> Vec<f64> {
    let (x, p, n) = (shared.x, shared.p, shared.n);
    let mut xtv = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            xtv[j] += x[i * p + j] * v[i];
        }
    }
    let mut coef = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for k in 0..p {
            s += shared.inv_xtx.get(j, k) * xtv[k];
        }
        coef[j] = s;
    }
    coef
}

fn ols_fitted(shared: &Shared, coef: &[f64]) -> Vec<f64> {
    let (x, p, n) = (shared.x, shared.p, shared.n);
    (0..n)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..p {
                s += x[i * p + j] * coef[j];
            }
            s
        })
        .collect()
}

/// Initial betas: OLS of log(normalized counts + 0.1) on X (natural log).
fn beta_ols_init(shared: &Shared, y: &[f64]) -> Vec<f64> {
    let logy: Vec<f64> = (0..shared.n)
        .map(|i| (y[i] / shared.sf[i] + 0.1).ln())
        .collect();
    ols_coef(shared, &logy)
}

/// One gene of `estimateDispersionsGeneEst` (niter = 1).
fn gene_est_one(shared: &Shared, y: &[f64]) -> GeneEst {
    let (n, p) = (shared.n, shared.p);
    let m = n as f64;
    let pf = p as f64;
    let ynorm: Vec<f64> = (0..n).map(|i| y[i] / shared.sf[i]).collect();
    let base_mean = ynorm.iter().sum::<f64>() / m;
    let base_var = if n > 1 {
        ynorm.iter().map(|&v| (v - base_mean).powi(2)).sum::<f64>() / (m - 1.0)
    } else {
        0.0
    };

    // rough dispersion from the OLS fit of normalized counts (mu floored at 1)
    let coef = ols_coef(shared, &ynorm);
    let fitted = ols_fitted(shared, &coef);
    let mut acc = 0.0;
    for i in 0..n {
        let mu = fitted[i].max(1.0);
        acc += ((ynorm[i] - mu).powi(2) - mu) / (mu * mu);
    }
    let rough = (acc / (m - pf)).max(0.0);
    let moments = (base_var - shared.xim * base_mean) / (base_mean * base_mean);
    let alpha_init = rough.min(moments).clamp(MIN_DISP, shared.max_disp);

    // expected counts: linear model when the design is group-like, else NB GLM
    let mut mu: Vec<f64> = if shared.linear_mu {
        (0..n).map(|i| fitted[i] * shared.sf[i]).collect()
    } else {
        let beta0 = beta_ols_init(shared, y);
        let fit = fit_nb_glm(
            y,
            shared.x,
            shared.sf,
            p,
            alpha_init,
            &beta0,
            &shared.lambda_nat,
            BETA_TOL,
            MAX_IT,
            MINMU,
        );
        let bad = !fit.converged
            || fit.beta.iter().any(|b| !b.is_finite())
            || (0..p).any(|j| fit.sigma.get(j, j) <= 0.0);
        if bad {
            // fitNbinomGLMs falls back to optim for these rows
            let large = 30.0;
            let stable = fit.beta.iter().all(|b| b.is_finite());
            let beta_log2: Vec<f64> = fit.beta.iter().map(|b| b * LOG2E).collect();
            let start: Vec<f64> = if stable && beta_log2.iter().all(|b| b.abs() < large) {
                beta_log2
            } else {
                beta0.clone()
            };
            let o = optim_beta(
                y,
                shared.x,
                shared.sf,
                p,
                alpha_init,
                &start,
                &shared.lambda_log2,
                MINMU,
            );
            o.mu
        } else {
            fit.mu
        }
    };
    for v in mu.iter_mut() {
        if *v < MINMU {
            *v = MINMU;
        }
    }

    // single fitDisp round + noIncrease revert + grid fallback
    let d = fit_disp(
        y,
        &mu,
        shared.x,
        p,
        alpha_init.ln(),
        None,
        (MIN_DISP / 10.0).ln(),
        KAPPA_0,
        DISP_TOL,
        MAX_IT,
    );
    let mut disp = d.log_alpha.exp().min(shared.max_disp);
    if d.last_lp < d.initial_lp + d.initial_lp.abs() / 1e6 {
        disp = alpha_init;
    }
    let converged = d.iter < MAX_IT && d.iter != 1;
    if !converged && disp > MIN_DISP * 10.0 {
        disp = fit_disp_grid(y, &mu, shared.x, p, shared.max_disp, None).exp();
    }
    let disp_gene_est = disp.clamp(MIN_DISP, shared.max_disp);

    GeneEst {
        alpha_init,
        mu,
        disp_gene_est,
    }
}

/// Gamma GLM with identity link for `disps ~ 1 + 1/means`, replicating R's
/// `glm(..., family=Gamma(link="identity"), start=coefs)` closely enough to
/// land on the same coefficients: IRLS with weights 1/mu^2, deviance-based
/// convergence (eps 1e-8, maxit 25), step-halving on invalid (non-positive)
/// fitted values. Returns (a0, a1, converged) or None on failure.
fn gamma_glm_identity(xv: &[f64], yv: &[f64], start: (f64, f64)) -> Option<(f64, f64, bool)> {
    let n = xv.len();
    if n < 2 {
        return None;
    }
    let dev_resids = |a0: f64, a1: f64| -> f64 {
        let mut dev = 0.0;
        for i in 0..n {
            let mu = a0 + a1 * xv[i];
            if mu <= 0.0 {
                return f64::NAN;
            }
            dev += -2.0 * ((yv[i] / mu).ln() - (yv[i] - mu) / mu);
        }
        dev
    };
    let (mut a0, mut a1) = start;
    if (0..n).any(|i| a0 + a1 * xv[i] <= 0.0) {
        return None; // invalid starting mu — R's glm would error
    }
    let mut dev_old = dev_resids(a0, a1);
    let mut coef_old = (a0, a1);
    let mut converged = false;
    for iter in 0..25 {
        // weighted LS: weights 1/mu^2, response z = y (identity link working
        // response reduces to y since z = eta + (y - mu) and eta = mu)
        let (mut s00, mut s01, mut s11, mut s0y, mut s1y) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let mu = a0 + a1 * xv[i];
            let w = 1.0 / (mu * mu);
            s00 += w;
            s01 += w * xv[i];
            s11 += w * xv[i] * xv[i];
            s0y += w * yv[i];
            s1y += w * xv[i] * yv[i];
        }
        let det = s00 * s11 - s01 * s01;
        if det.abs() < 1e-300 || !det.is_finite() {
            return None;
        }
        let mut na0 = (s0y * s11 - s1y * s01) / det;
        let mut na1 = (s00 * s1y - s01 * s0y) / det;
        if !(na0.is_finite() && na1.is_finite()) {
            return None;
        }
        // step-halving on invalid mu or non-finite deviance
        let mut dev = dev_resids(na0, na1);
        let mut halvings = 0;
        while (!dev.is_finite() || (0..n).any(|i| na0 + na1 * xv[i] <= 0.0)) && halvings < 25 {
            if iter == 0 && halvings == 0 && coef_old == start {
                // R would still halve toward start
            }
            na0 = (na0 + coef_old.0) / 2.0;
            na1 = (na1 + coef_old.1) / 2.0;
            dev = dev_resids(na0, na1);
            halvings += 1;
        }
        if !dev.is_finite() {
            return None;
        }
        a0 = na0;
        a1 = na1;
        if (dev - dev_old).abs() / (dev.abs() + 0.1) < 1e-8 {
            converged = true;
            break;
        }
        dev_old = dev;
        coef_old = (a0, a1);
    }
    Some((a0, a1, converged))
}

/// `parametricDispersionFit`: outer loop re-gating residuals and refitting the
/// Gamma GLM until the coefficients stabilise. Errors mirror R's, in which
/// case DESeq2 would switch to a local (locfit) trend that we do not
/// implement.
fn parametric_dispersion_fit(means: &[f64], disps: &[f64]) -> Result<(f64, f64), String> {
    let mut coefs = (0.1_f64, 1.0_f64);
    let mut iter = 0;
    loop {
        let mut xg = Vec::new();
        let mut yg = Vec::new();
        for i in 0..means.len() {
            let mu = coefs.0 + coefs.1 / means[i];
            let resid = disps[i] / mu;
            if resid > 1e-4 && resid < 15.0 {
                xg.push(1.0 / means[i]);
                yg.push(disps[i]);
            }
        }
        let (a0, a1, glm_conv) = gamma_glm_identity(&xg, &yg, coefs)
            .ok_or("parametric dispersion fit failed (glm error)")?;
        let old = coefs;
        coefs = (a0, a1);
        if !(coefs.0 > 0.0 && coefs.1 > 0.0) {
            return Err("parametric dispersion fit failed".into());
        }
        if ((coefs.0 / old.0).ln().powi(2) + (coefs.1 / old.1).ln().powi(2)) < 1e-6 && glm_conv {
            return Ok(coefs);
        }
        iter += 1;
        if iter > 10 {
            return Err("dispersion fit did not converge".into());
        }
    }
}

/// MAP dispersion for one gene (`estimateDispersionsMAP`).
fn map_disp_one(
    shared: &Shared,
    y: &[f64],
    mu: &[f64],
    disp_gene_est: f64,
    disp_fit: f64,
    prior_var: f64,
    var_log_disp_ests: f64,
) -> (f64, f64, bool) {
    let disp_init = if disp_gene_est.is_finite() {
        if disp_gene_est > 0.1 * disp_fit {
            disp_gene_est
        } else {
            disp_fit
        }
    } else {
        disp_fit
    };
    let prior = Some((disp_fit.ln(), prior_var));
    let d = fit_disp(
        y,
        mu,
        shared.x,
        shared.p,
        disp_init.ln(),
        prior,
        (MIN_DISP / 10.0).ln(),
        KAPPA_0,
        DISP_TOL,
        MAX_IT,
    );
    let mut disp_map = d.log_alpha.exp();
    if d.iter >= MAX_IT {
        disp_map = fit_disp_grid(y, mu, shared.x, shared.p, shared.max_disp, prior).exp();
    }
    disp_map = disp_map.clamp(MIN_DISP, shared.max_disp);
    let outlier = disp_gene_est.is_finite()
        && disp_gene_est.ln() > disp_fit.ln() + 2.0 * var_log_disp_ests.sqrt();
    let final_disp = if outlier { disp_gene_est } else { disp_map };
    (disp_map, final_disp, outlier)
}

/// Robust method-of-moments dispersion for Cook's distance, one gene.
fn robust_mom_disp(ynorm: &[f64], base_mean: f64, cells_three_or_more: &[Vec<usize>]) -> f64 {
    let v = if !cells_three_or_more.is_empty() {
        let mut vmax = f64::NEG_INFINITY;
        for idx in cells_three_or_more {
            let nn = idx.len();
            let (trim, scale) = trim_rule(nn);
            let vals: Vec<f64> = idx.iter().map(|&i| ynorm[i]).collect();
            let cm = trimmed_mean(&vals, trim);
            let sq: Vec<f64> = idx.iter().map(|&i| (ynorm[i] - cm).powi(2)).collect();
            let ve = scale * trimmed_mean(&sq, trim);
            if ve > vmax {
                vmax = ve;
            }
        }
        vmax
    } else {
        let rm = trimmed_mean(ynorm, 1.0 / 8.0);
        let sq: Vec<f64> = ynorm.iter().map(|&v| (v - rm).powi(2)).collect();
        1.51 * trimmed_mean(&sq, 1.0 / 8.0)
    };
    (((v - base_mean) / (base_mean * base_mean)).max(0.04)) as f64
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

/// The Wald stage for one gene: final GLM fit, contrast, Cook's distances.
#[allow(clippy::too_many_arguments)]
fn wald_one(
    shared: &Shared,
    y: &[f64],
    dispersion: f64,
    contrast: &Contrast,
    cells_three_or_more: &[Vec<usize>],
    samples_for_cooks: &[bool],
    disp_map: f64,
    disp_outlier: bool,
) -> WaldOut {
    let (n, p) = (shared.n, shared.p);
    let beta0 = beta_ols_init(shared, y);
    let fit = fit_nb_glm(
        y,
        shared.x,
        shared.sf,
        p,
        dispersion,
        &beta0,
        &shared.lambda_nat,
        BETA_TOL,
        MAX_IT,
        MINMU,
    );
    let mut beta_conv = fit.converged;
    let stable = fit.beta.iter().all(|b| b.is_finite());
    let var_positive = (0..p).all(|j| fit.sigma.get(j, j) > 0.0);

    // log2-scale coefficients / covariance actually used for reporting
    let mut beta_log2: Vec<f64> = fit.beta.iter().map(|b| b * LOG2E).collect();
    let mut sigma_log2 = Mat::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            sigma_log2.set(i, j, LOG2E * LOG2E * fit.sigma.get(i, j));
        }
    }
    let mut mu = fit.mu.clone();
    let hat = fit.hat.clone();

    if !beta_conv || !stable || !var_positive {
        // optim fallback, replicating fitNbinomGLMsOptim's starting rule
        let large = 30.0;
        let start: Vec<f64> = if stable && beta_log2.iter().all(|b| b.abs() < large) {
            beta_log2.clone()
        } else {
            beta0.clone()
        };
        let o = optim_beta(
            y,
            shared.x,
            shared.sf,
            p,
            dispersion,
            &start,
            &shared.lambda_log2,
            MINMU,
        );
        beta_log2 = o.beta_log2;
        sigma_log2 = o.sigma_log2;
        mu = o.mu;
        if o.converged {
            beta_conv = true;
        }
    }

    // betaSE = log2(e) * sqrt(pmax(var, 0))
    let se_of = |j: usize| -> f64 { sigma_log2.get(j, j).max(0.0).sqrt() };
    let (lfc_log2, se_log2, stat) = match contrast {
        Contrast::Coef(j) => {
            let lfc = beta_log2[*j];
            let se = se_of(*j);
            (lfc, se, lfc / se)
        }
        Contrast::NegCoef(j) => {
            let lfc = beta_log2[*j];
            let se = se_of(*j);
            let stat = lfc / se;
            (-lfc, se, -stat)
        }
        Contrast::Vector(c) => {
            let mut num = 0.0;
            for j in 0..p {
                num += c[j] * beta_log2[j] * LN2; // natural-scale c'beta
            }
            let mut q = 0.0;
            for a in 0..p {
                for b in 0..p {
                    q += c[a] * (LN2 * LN2 * sigma_log2.get(a, b)) * c[b];
                }
            }
            let est = LOG2E * num;
            let se = LOG2E * q.max(0.0).sqrt();
            (est, se, est / se)
        }
    };
    let pvalue = wald_pvalue(stat);

    // Cook's distances (using the unfloored mu, hat with ridge)
    let ynorm: Vec<f64> = (0..n).map(|i| y[i] / shared.sf[i]).collect();
    let base_mean = ynorm.iter().sum::<f64>() / n as f64;
    let rob_disp = robust_mom_disp(&ynorm, base_mean, cells_three_or_more);
    let mut cooks = vec![0.0; n];
    for i in 0..n {
        let v = mu[i] + rob_disp * mu[i] * mu[i];
        let pr2 = (y[i] - mu[i]).powi(2) / v;
        cooks[i] = pr2 / p as f64 * hat[i] / (1.0 - hat[i]).powi(2);
    }
    let max_cooks = if n > p && samples_for_cooks.iter().any(|&b| b) {
        let mut mx = f64::NEG_INFINITY;
        for i in 0..n {
            if samples_for_cooks[i] && cooks[i] > mx {
                mx = cooks[i];
            }
        }
        mx
    } else {
        f64::NAN
    };

    WaldOut {
        lfc_log2,
        se_log2,
        stat,
        pvalue,
        cooks,
        max_cooks,
        beta_conv,
        disp_map,
        dispersion,
        disp_outlier,
    }
}

/// All per-gene state from one full fit over a set of genes.
struct FitState {
    base_means: Vec<f64>,
    disp_gene_est: Vec<f64>,
    disp_fit: Vec<f64>,
    wald: Vec<Option<WaldOut>>, // None for all-zero genes
    trend: (f64, f64),
    prior_var: f64,
    var_log_disp_ests: f64,
}

/// Run gene-est + trend + MAP + Wald over all genes of `counts`.
/// When `fixed` is provided (refit after outlier replacement), the trend
/// coefficients, prior variance and varLogDispEsts are reused and only the
/// genes in `subset` are fit.
#[allow(clippy::too_many_arguments)]
fn fit_all(
    counts: &CountMatrix,
    shared: &Shared,
    design: &Design,
    contrast: &Contrast,
    nthreads: usize,
    fixed: Option<((f64, f64), f64, f64)>,
    subset: Option<&[usize]>,
) -> Result<FitState, String> {
    let g = counts.n_genes();
    let n = shared.n;
    let m = n as f64;
    let pf = shared.p as f64;

    let gene_idx: Vec<usize> = match subset {
        Some(s) => s.to_vec(),
        None => (0..g).collect(),
    };

    let mut base_means = vec![f64::NAN; g];
    let mut all_zero = vec![true; g];
    for &gi in &gene_idx {
        let row = counts.row(gi);
        let bm = (0..n).map(|i| row[i] / shared.sf[i]).sum::<f64>() / m;
        base_means[gi] = bm;
        all_zero[gi] = row.iter().sum::<f64>() == 0.0;
    }

    // --- gene-wise dispersions ---
    let nonzero: Vec<usize> = gene_idx.iter().copied().filter(|&gi| !all_zero[gi]).collect();
    let ge: Vec<GeneEst> = parallel_map(nonzero.clone(), nthreads, |gi| {
        gene_est_one(shared, counts.row(gi))
    });
    let mut disp_gene_est = vec![f64::NAN; g];
    let mut alpha_init = vec![f64::NAN; g];
    let mut mu_store: Vec<Vec<f64>> = vec![Vec::new(); g];
    for (k, &gi) in nonzero.iter().enumerate() {
        disp_gene_est[gi] = ge[k].disp_gene_est;
        alpha_init[gi] = ge[k].alpha_init;
        mu_store[gi] = ge[k].mu.clone();
    }
    drop(ge);

    // --- trend + prior variance (or reuse the stored ones on refit) ---
    let (trend, prior_var, var_log_disp_ests) = match fixed {
        Some((t, pv, vld)) => (t, pv, vld),
        None => {
            let mut fit_means = Vec::new();
            let mut fit_disps = Vec::new();
            for &gi in &nonzero {
                if disp_gene_est[gi] > 100.0 * MIN_DISP {
                    fit_means.push(base_means[gi]);
                    fit_disps.push(disp_gene_est[gi]);
                }
            }
            if fit_means.is_empty() {
                return Err(
                    "all gene-wise dispersion estimates are within 2 orders of magnitude of the minimum"
                        .into(),
                );
            }
            let trend = match parametric_dispersion_fit(&fit_means, &fit_disps) {
                Ok(t) => t,
                Err(e) => {
                    let med = median(&fit_disps).max(1e-4);
                    eprintln!(
                        "[rust_deseq2] WARNING: {e}; DESeq2 would switch to a local (locfit) \
                         trend here. Falling back to a flat median trend ({med:.4}); results \
                         will NOT match DESeq2 on this dataset."
                    );
                    (med, 0.0)
                }
            };
            // varLogDispEsts over genes with dispGeneEst >= 100*minDisp
            let mut resid = Vec::new();
            for &gi in &nonzero {
                if disp_gene_est[gi] >= 100.0 * MIN_DISP {
                    let fit = trend.0 + trend.1 / base_means[gi];
                    resid.push(disp_gene_est[gi].ln() - fit.ln());
                }
            }
            let s = mad(&resid);
            let vld = if s.is_finite() { s * s } else { 0.0 };
            let prior_var = if m - pf > 3.0 {
                (vld - trigamma((m - pf) / 2.0)).max(0.25)
            } else if m > pf {
                // DESeq2's seeded Monte-Carlo estimate for residual df <= 3
                mc_disp_prior_var(&resid, m - pf)
            } else {
                vld
            };
            (trend, prior_var, vld)
        }
    };

    let mut disp_fit = vec![f64::NAN; g];
    for &gi in &nonzero {
        disp_fit[gi] = trend.0 + trend.1 / base_means[gi];
    }

    // --- MAP dispersions ---
    let map_out: Vec<(f64, f64, bool)> = parallel_map(nonzero.clone(), nthreads, |gi| {
        map_disp_one(
            shared,
            counts.row(gi),
            &mu_store[gi],
            disp_gene_est[gi],
            disp_fit[gi],
            prior_var,
            var_log_disp_ests,
        )
    });

    // --- Wald stage ---
    let cells_str = design.row_cells();
    let mut cell_members: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, cell) in cells_str.iter().enumerate() {
        match cell_members.iter_mut().find(|(c, _)| c == cell) {
            Some((_, mems)) => mems.push(i),
            None => cell_members.push((cell.clone(), vec![i])),
        }
    }
    let cells_three_or_more: Vec<Vec<usize>> = cell_members
        .iter()
        .filter(|(_, mems)| mems.len() >= 3)
        .map(|(_, mems)| mems.clone())
        .collect();
    let samples_for_cooks = design.n_or_more_in_cell(3);

    let wald_res: Vec<WaldOut> = {
        let items: Vec<usize> = (0..nonzero.len()).collect();
        parallel_map(items, nthreads, |k| {
            let gi = nonzero[k];
            let (dmap, dfinal, doutlier) = map_out[k];
            wald_one(
                shared,
                counts.row(gi),
                dfinal,
                contrast,
                &cells_three_or_more,
                &samples_for_cooks,
                dmap,
                doutlier,
            )
        })
    };

    let mut wald: Vec<Option<WaldOut>> = (0..g).map(|_| None).collect();
    for (w, &gi) in wald_res.into_iter().zip(nonzero.iter()) {
        wald[gi] = Some(w);
    }
    let _ = alpha_init;

    Ok(FitState {
        base_means,
        disp_gene_est,
        disp_fit,
        wald,
        trend,
        prior_var,
        var_log_disp_ests,
    })
}

/// Run the full workflow, returning one result row per gene (input order).
pub fn run(
    counts: &CountMatrix,
    coldata: &ColData,
    opts: &Options,
) -> Result<Vec<GeneResult>, String> {
    let n = counts.n_samples();
    let g = counts.n_genes();
    let m = n as f64;

    // --- design & contrast ---
    let sample_ids = match &opts.sample_col {
        Some(name) => coldata.column(name)?.clone(),
        None => coldata.sample_ids.clone(),
    };
    let var_names = parse_design_formula(&opts.design)?;
    if !var_names.iter().any(|v| v == &opts.contrast_var) {
        return Err(format!(
            "contrast variable '{}' is not part of the design ({})",
            opts.contrast_var,
            var_names.join(" + ")
        ));
    }
    let design = build_design(
        coldata,
        &sample_ids,
        &counts.samples,
        &var_names,
        &opts.factor_cols,
    )?;
    let resolved =
        design.resolve_contrast(&opts.contrast_var, &opts.case_level, &opts.control_level)?;
    let p = design.p;
    let pf = p as f64;

    // --- size factors ---
    let sf = size_factors(counts)?;

    let mut xtx = Mat::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                xtx.add(a, b, design.x[i * p + a] * design.x[i * p + b]);
            }
        }
    }
    let inv_xtx = invert(&xtx).ok_or("the model matrix is not full rank")?;

    let lambda_log2 = vec![1e-6; p];
    let lambda_nat: Vec<f64> = lambda_log2.iter().map(|l| l / (LN2 * LN2)).collect();
    let shared = Shared {
        x: &design.x,
        p,
        n,
        sf: &sf,
        inv_xtx: &inv_xtx,
        linear_mu: design.use_linear_mu(),
        max_disp: 10.0_f64.max(m),
        xim: sf.iter().map(|s| 1.0 / s).sum::<f64>() / m,
        lambda_nat,
        lambda_log2,
    };

    let nthreads = if opts.threads > 0 {
        opts.threads
    } else {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .min(12)
    };

    // --- full fit ---
    let mut state = fit_all(counts, &shared, &design, &resolved.kind, nthreads, None, None)?;
    eprintln!(
        "[rust_deseq2] trend coefs: {:.10}, {:.10} ; dispPriorVar: {:.10} ; varLogDispEsts: {:.10}",
        state.trend.0, state.trend.1, state.prior_var, state.var_log_disp_ests
    );

    // --- Cook's outlier replacement (minReplicatesForReplace = 7) ---
    let mut replace_flag = vec![false; g];
    let cooks_cutoff = if m > pf {
        qf(0.99, pf, m - pf)
    } else {
        f64::NAN
    };
    let replaceable = design.n_or_more_in_cell(MIN_REPLICATES_FOR_REPLACE);
    let any_replaceable = replaceable.iter().any(|&b| b);
    let mut max_cooks: Vec<f64> = (0..g)
        .map(|gi| state.wald[gi].as_ref().map_or(f64::NAN, |w| w.max_cooks))
        .collect();
    // Snapshot of the ORIGINAL fit's Cook's distances: results() keeps using
    // these for flagging even after the refit replaces the per-gene fits.
    let orig_cooks: Vec<Option<Vec<f64>>> = (0..g)
        .map(|gi| state.wald[gi].as_ref().map(|w| w.cooks.clone()))
        .collect();

    if m > pf && any_replaceable {
        for gi in 0..g {
            if let Some(w) = &state.wald[gi] {
                replace_flag[gi] = w.cooks.iter().any(|&c| c > cooks_cutoff);
            }
        }
        let n_replace = replace_flag.iter().filter(|&&b| b).count();
        if n_replace > 0 {
            // build replacement counts
            let mut replaced = counts.clone();
            for gi in 0..g {
                if !replace_flag[gi] {
                    continue;
                }
                let w = state.wald[gi].as_ref().unwrap();
                let row = counts.row(gi);
                let ynorm: Vec<f64> = (0..n).map(|i| row[i] / sf[i]).collect();
                let trim_bm = trimmed_mean(&ynorm, 0.2);
                for i in 0..n {
                    if replaceable[i] && w.cooks[i] > cooks_cutoff {
                        replaced.counts[gi * n + i] = (trim_bm * sf[i]).trunc();
                    }
                }
            }
            // refit only the replaced genes, reusing trend/prior
            let new_all_zero: Vec<bool> = (0..g)
                .map(|gi| replace_flag[gi] && replaced.row(gi).iter().sum::<f64>() == 0.0)
                .collect();
            let refit_genes: Vec<usize> = (0..g)
                .filter(|&gi| replace_flag[gi] && !new_all_zero[gi])
                .collect();
            eprintln!(
                "[rust_deseq2] replacing outliers and refitting for {} genes",
                n_replace
            );
            if !refit_genes.is_empty() {
                let mut sub = fit_all(
                    &replaced,
                    &shared,
                    &design,
                    &resolved.kind,
                    nthreads,
                    Some((state.trend, state.prior_var, state.var_log_disp_ests)),
                    Some(&refit_genes),
                )?;
                for &gi in &refit_genes {
                    state.base_means[gi] = sub.base_means[gi];
                    state.disp_gene_est[gi] = sub.disp_gene_est[gi];
                    state.disp_fit[gi] = sub.disp_fit[gi];
                    state.wald[gi] = sub.wald[gi].take();
                }
            }
            // genes replaced into all-zero are NOT refit; their intermediate
            // columns keep the original-fit values (as in DESeq2) and their
            // results are zeroed by the nowZero rule below via baseMean == 0
            for gi in 0..g {
                if new_all_zero[gi] {
                    state.base_means[gi] = 0.0;
                }
            }
            // baseMean of every gene is recomputed from the replaced counts
            for gi in 0..g {
                if !replace_flag[gi] {
                    // unchanged rows: same counts, same baseMean
                    continue;
                }
                let row = replaced.row(gi);
                state.base_means[gi] = (0..n).map(|i| row[i] / sf[i]).sum::<f64>() / m;
            }
            // post-refit Cook's flagging: with every sample replaceable there
            // is no flagging at all; otherwise recompute maxCooks from the
            // ORIGINAL cooks with the replaceable columns zeroed.
            if replaceable.iter().all(|&b| b) {
                for mc in max_cooks.iter_mut() {
                    *mc = f64::NAN;
                }
            } else {
                let samples_for_cooks = design.n_or_more_in_cell(3);
                let any_cooks_sample = samples_for_cooks.iter().any(|&b| b);
                // R zeroes the replaceable COLUMNS of the (NA-filled) cooks
                // matrix before taking row maxima, so an all-zero gene gets 0
                // when every Cook's-eligible sample is replaceable, and NA
                // (max with NA) otherwise.
                let any_nonreplaceable_cooks_sample = (0..n)
                    .any(|i| samples_for_cooks[i] && !replaceable[i]);
                for gi in 0..g {
                    max_cooks[gi] = if !(n > p && any_cooks_sample) {
                        f64::NAN
                    } else {
                        match &orig_cooks[gi] {
                            Some(cooks) => {
                                let mut mx = f64::NEG_INFINITY;
                                for i in 0..n {
                                    if samples_for_cooks[i] {
                                        let c = if replaceable[i] { 0.0 } else { cooks[i] };
                                        if c > mx {
                                            mx = c;
                                        }
                                    }
                                }
                                mx
                            }
                            None if any_nonreplaceable_cooks_sample => f64::NAN,
                            None => 0.0,
                        }
                    };
                }
            }
        }
    }

    // --- assemble results ---
    let mut lfc = vec![f64::NAN; g];
    let mut se = vec![f64::NAN; g];
    let mut stat = vec![f64::NAN; g];
    let mut pval = vec![f64::NAN; g];
    for gi in 0..g {
        if let Some(w) = &state.wald[gi] {
            lfc[gi] = w.lfc_log2;
            se[gi] = w.se_log2;
            stat[gi] = w.stat;
            pval[gi] = w.pvalue;
        }
    }

    // contrast-all-zero rule (character contrasts): computed on the ORIGINAL
    // counts over the samples in the case/control levels; skipped for genes
    // whose (post-replacement) baseMean is zero, which stay NA / nowZero.
    for gi in 0..g {
        if !(state.base_means[gi] > 0.0) {
            continue;
        }
        let row = counts.row(gi);
        let mut all0 = true;
        let mut any_sample = false;
        for i in 0..n {
            if resolved.in_contrast_levels[i] {
                any_sample = true;
                if row[i] != 0.0 {
                    all0 = false;
                    break;
                }
            }
        }
        if any_sample && all0 {
            lfc[gi] = 0.0;
            stat[gi] = 0.0;
            pval[gi] = 1.0;
        }
    }

    // Cook's p-value flagging
    let two_level_heuristic = design.single_two_level_factor();
    if m > pf {
        for gi in 0..g {
            if !(max_cooks[gi] > cooks_cutoff) {
                continue;
            }
            if two_level_heuristic {
                if let Some(cooks) = &orig_cooks[gi] {
                    // index of the max-Cook's sample over ALL samples
                    let mut imax = 0;
                    let mut cmax = f64::NEG_INFINITY;
                    for i in 0..n {
                        if cooks[i] > cmax {
                            cmax = cooks[i];
                            imax = i;
                        }
                    }
                    let out_count = counts.row(gi)[imax];
                    let bigger = counts.row(gi).iter().filter(|&&c| c > out_count).count();
                    if bigger >= 3 {
                        continue; // don't filter
                    }
                }
            }
            pval[gi] = f64::NAN;
        }
    }

    // nowZero rule: replaced genes whose baseMean became zero
    for gi in 0..g {
        if replace_flag[gi] && state.base_means[gi] == 0.0 {
            lfc[gi] = 0.0;
            se[gi] = 0.0;
            stat[gi] = 0.0;
            pval[gi] = 1.0;
        }
    }

    // --- independent filtering + BH ---
    let padj = pvalue_adjustment(&state.base_means, &pval, FILTER_ALPHA);

    let results: Vec<GeneResult> = (0..g)
        .map(|gi| GeneResult {
            gene: counts.genes[gi].clone(),
            base_mean: state.base_means[gi],
            log2_fold_change: lfc[gi],
            lfc_se: se[gi],
            stat: stat[gi],
            pvalue: pval[gi],
            padj: padj[gi],
        })
        .collect();

    if let Some(prefix) = &opts.dump_prefix {
        write_diagnostics(prefix, counts, &sf, &state, &max_cooks, &replace_flag, &results)?;
    }

    Ok(results)
}


/// DESeq2's Monte-Carlo estimate of the dispersion prior variance for
/// designs with residual df <= 3 (`estimateDispersionsPriorVar`): with
/// `set.seed(2)`, match the distribution of the observed log-dispersion
/// residuals against `log(rchisq(1e4, df)) + rnorm(1e4, 0, sqrt(v)) - log(df)`
/// over a grid of prior variances by KL divergence of binned densities,
/// smooth with `loess(span = 0.2)`, and take the fine-grid argmin
/// (floored at 0.25). RNG streams and histogram counts match R bitwise.
fn mc_disp_prior_var(resid: &[f64], df: f64) -> f64 {
    const NDRAW: usize = 10_000;
    let brks: Vec<f64> = (-20..=20).map(|i| i as f64 / 2.0).collect();
    let nb = brks.len() - 1; // 40 bins of width 0.5, right-closed

    let density = |vals: &[f64]| -> Vec<f64> {
        let mut counts = vec![0usize; nb];
        for &x in vals {
            // R hist: breaks[i] < x <= breaks[i+1]
            for j in 0..nb {
                if x > brks[j] && x <= brks[j + 1] {
                    counts[j] += 1;
                    break;
                }
            }
        }
        let n = vals.len() as f64;
        counts.iter().map(|&c| c as f64 / (n * 0.5)).collect()
    };

    let obs: Vec<f64> = resid
        .iter()
        .copied()
        .filter(|&v| v > brks[0] && v < brks[nb])
        .collect();
    let obs_dens = density(&obs);
    let ln_df = df.ln();

    let mut rng = RRng::new(2);
    let by = 8.0 / 199.0;
    let mut grid = vec![0.0; 200];
    for (k, g) in grid.iter_mut().enumerate() {
        *g = k as f64 * by;
    }
    grid[199] = 8.0; // R's seq() pins the endpoint

    let mut chis = vec![0.0; NDRAW];
    let mut norms = vec![0.0; NDRAW];
    let mut kl = vec![0.0; 200];
    for (k, &x) in grid.iter().enumerate() {
        let sd = x.sqrt();
        for c in chis.iter_mut() {
            *c = rng.rchisq(df);
        }
        for nrm in norms.iter_mut() {
            // R's rnorm(0, 0) returns 0 WITHOUT consuming the stream
            *nrm = if sd == 0.0 { 0.0 } else { sd * rng.norm_rand() };
        }
        let rand: Vec<f64> = (0..NDRAW)
            .map(|i| chis[i].ln() + norms[i] - ln_df)
            .filter(|&v| v > brks[0] && v < brks[nb])
            .collect();
        let rand_dens = density(&rand);
        let mut small = f64::INFINITY;
        for &z in obs_dens.iter().chain(rand_dens.iter()) {
            if z > 0.0 && z < small {
                small = z;
            }
        }
        let mut s = 0.0;
        for j in 0..nb {
            s += obs_dens[j] * ((obs_dens[j] + small).ln() - (rand_dens[j] + small).ln());
        }
        kl[k] = s;
    }

    let lofit = Loess1D::fit(&grid, &kl, 0.2);
    let byf = 8.0 / 999.0;
    let mut best_v = f64::INFINITY;
    let mut best = 0.0;
    for k in 0..1000 {
        let z = if k == 999 { 8.0 } else { k as f64 * byf };
        let f = lofit.predict(z);
        if f < best_v {
            best_v = f;
            best = z;
        }
    }
    best.max(0.25)
}

/// results()' p-value adjustment: independent filtering over 50 baseMean
/// quantiles with R's lowess smoother, then BH at the chosen threshold.
fn pvalue_adjustment(base_means: &[f64], pvals: &[f64], alpha: f64) -> Vec<f64> {
    let g = pvals.len();
    if !(0..g).any(|i| pvals[i].is_finite()) {
        return vec![f64::NAN; g];
    }
    let mut all_bm: Vec<f64> = base_means.to_vec();
    all_bm.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_q = base_means.iter().filter(|&&v| v == 0.0).count() as f64 / g as f64;
    let upper_q = if lower_q < 0.95 { 0.95 } else { 1.0 };
    const NTHETA: usize = 50;
    let thetas: Vec<f64> = (0..NTHETA)
        .map(|k| lower_q + (upper_q - lower_q) * (k as f64) / (NTHETA as f64 - 1.0))
        .collect();

    let mut num_rej = vec![0.0; NTHETA];
    let mut padj_by_theta: Vec<Vec<f64>> = Vec::with_capacity(NTHETA);
    for (ti, &theta) in thetas.iter().enumerate() {
        let cut = quantile(&all_bm, theta);
        let masked: Vec<f64> = (0..g)
            .map(|i| {
                if base_means[i] >= cut {
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
        0
    } else {
        let delta = 0.01 * (thetas[NTHETA - 1] - thetas[0]);
        let smoothed = r_lowess(&thetas, &num_rej, 0.2, 3, delta);
        let max_fit = smoothed.iter().cloned().fold(f64::MIN, f64::max);
        let (mut ss, mut c) = (0.0, 0.0);
        for k in 0..NTHETA {
            if num_rej[k] > 0.0 {
                ss += (num_rej[k] - smoothed[k]).powi(2);
                c += 1.0;
            }
        }
        let rmse = if c > 0.0 { (ss / c).sqrt() } else { 0.0 };
        let thresh = max_fit - rmse;
        (0..NTHETA)
            .find(|&k| num_rej[k] > thresh)
            .or_else(|| (0..NTHETA).find(|&k| num_rej[k] > 0.9 * max_fit))
            .or_else(|| (0..NTHETA).find(|&k| num_rej[k] > 0.8 * max_fit))
            .unwrap_or(0)
    };

    padj_by_theta.into_iter().nth(j).unwrap()
}

fn fmt_num(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.17e}")
    } else {
        "NA".to_string()
    }
}

fn write_diagnostics(
    prefix: &str,
    counts: &CountMatrix,
    sf: &[f64],
    state: &FitState,
    max_cooks: &[f64],
    replace_flag: &[bool],
    results: &[GeneResult],
) -> Result<(), String> {
    let sf_path = format!("{prefix}.size_factors.tsv");
    let mut sfw = BufWriter::new(
        File::create(&sf_path).map_err(|e| format!("cannot write '{sf_path}': {e}"))?,
    );
    writeln!(sfw, "sample\tsizeFactor").map_err(|e| e.to_string())?;
    for (sample, s) in counts.samples.iter().zip(sf) {
        writeln!(sfw, "{}\t{}", sample, fmt_num(*s)).map_err(|e| e.to_string())?;
    }

    let gene_path = format!("{prefix}.genes.tsv");
    let mut gw = BufWriter::new(
        File::create(&gene_path).map_err(|e| format!("cannot write '{gene_path}': {e}"))?,
    );
    writeln!(
        gw,
        "gene\tbaseMean\tdispGeneEst\tdispFit\tdispMAP\tdispersion\tdispOutlier\tbetaConv\tmaxCooks\treplace\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj"
    )
    .map_err(|e| e.to_string())?;
    for gi in 0..counts.n_genes() {
        let r = &results[gi];
        let (dmap, dfinal, doutlier, dconv) = match &state.wald[gi] {
            Some(w) => (w.disp_map, w.dispersion, w.disp_outlier, w.beta_conv),
            None => (f64::NAN, f64::NAN, false, false),
        };
        writeln!(
            gw,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.gene,
            fmt_num(state.base_means[gi]),
            fmt_num(state.disp_gene_est[gi]),
            fmt_num(state.disp_fit[gi]),
            fmt_num(dmap),
            fmt_num(dfinal),
            doutlier,
            dconv,
            fmt_num(max_cooks[gi]),
            replace_flag[gi],
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
