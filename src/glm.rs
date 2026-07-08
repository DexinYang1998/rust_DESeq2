//! Negative-binomial GLM with a log link, fit by iteratively reweighted least
//! squares (IRLS), plus maximum-likelihood dispersion estimation. This is the
//! numerical heart of the DESeq2 workflow: for a fixed dispersion `alpha` the
//! per-gene coefficients are found by IRLS, and the dispersion itself is found
//! by maximising the negative-binomial likelihood (optionally with a
//! log-normal shrinkage prior toward a fitted mean-dispersion trend).

use crate::linalg::{invert, log_abs_det, Mat};
use crate::mathx::nb_log_pmf;

/// Result of a per-gene GLM fit.
pub struct GlmFit {
    pub beta: Vec<f64>,
    /// Coefficient covariance: the ridge sandwich
    /// (X'WX + ridge)^{-1} · X'WX · (X'WX + ridge)^{-1}.
    pub cov: Mat,
    pub mu: Vec<f64>,
    /// Whether IRLS reached the convergence tolerance (informational; low-count
    /// genes may be reported even without it as long as `cov` is valid).
    #[allow(dead_code)]
    pub converged: bool,
}

const ETA_MAX: f64 = 30.0;

/// Floor on the fitted mean (DESeq2's `minmu`, default 0.5). Flooring mu bounds
/// the intercept for genes where a whole group is zero, so their fold change
/// and standard error stay finite and comparable to DESeq2 instead of running
/// off to the eta clamp.
const MIN_MU: f64 = 0.5;

/// Tiny ridge added to the diagonal of X'WX, matching DESeq2's default weak
/// prior (`lambda = 1e-6` on the log2 scale) expressed here on the natural-log
/// scale: `1e-6 / ln(2)^2`. It is negligible for well-estimated genes but keeps
/// the Hessian invertible for separated / all-zero-in-a-group genes, so they
/// yield a finite, bounded coefficient and standard error instead of NA.
const RIDGE: f64 = 1e-6 / (std::f64::consts::LN_2 * std::f64::consts::LN_2);

/// Sandwich covariance `inv · H · inv` for a symmetric `inv` and `H` (p x p).
fn sandwich(inv: &Mat, h: &Mat, p: usize) -> Mat {
    // T = H · inv
    let mut t = Mat::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                s += h.get(i, k) * inv.get(k, j);
            }
            t.set(i, j, s);
        }
    }
    // cov = inv · T
    let mut cov = Mat::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                s += inv.get(i, k) * t.get(k, j);
            }
            cov.set(i, j, s);
        }
    }
    cov
}

/// Fit a negative-binomial GLM (log link) with dispersion `alpha` held fixed.
///
/// * `y`      – counts, length n
/// * `x`      – design matrix, row-major n x p
/// * `offset` – per-sample offset (log size factor), length n
pub fn fit_nb_glm(
    y: &[f64],
    x: &[f64],
    offset: &[f64],
    p: usize,
    alpha: f64,
    max_iter: usize,
) -> GlmFit {
    const TOL: f64 = 1e-8; // DESeq2 betaTol
    const LARGE: f64 = 30.0; // DESeq2 coefficient bound for divergence

    let n = y.len();
    let mut beta = vec![0.0; p];

    // Warm start: intercept at log(mean of y/exp(offset)); others at 0.
    let mut m0 = 0.0;
    for i in 0..n {
        m0 += y[i] / offset[i].exp();
    }
    m0 = (m0 / n as f64).max(0.1);
    beta[0] = m0.ln();

    // Fitted means for the current beta, floored at MIN_MU.
    let eval_mu = |beta: &[f64], mu: &mut [f64]| {
        for i in 0..n {
            let mut eta = offset[i];
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            mu[i] = eta.clamp(-ETA_MAX, ETA_MAX).exp().max(MIN_MU);
        }
    };
    // NB deviance = -2 * sum log NB(y; 1/alpha, mu), DESeq2's convergence metric.
    let deviance = |mu: &[f64]| -> f64 {
        let mut dev = 0.0;
        for i in 0..n {
            dev += -2.0 * nb_log_pmf(y[i], mu[i], alpha);
        }
        dev
    };

    let mut mu = vec![0.0; n];
    let mut converged = false;
    let mut dev_old = 0.0;

    for t in 0..max_iter {
        // Means and weights at the current beta.
        eval_mu(&beta, &mut mu);
        let mut xtwx = Mat::zeros(p, p);
        let mut xtwu = vec![0.0; p];
        for i in 0..n {
            let mui = mu[i];
            // NB working weight for a log link: w = mu / (1 + alpha*mu).
            let w = mui / (1.0 + alpha * mui);
            // Working response (offset removed); ln(mu) keeps it consistent when
            // mu is floored:  u = (ln mu - offset) + (y - mu)/mu
            let u = (mui.ln() - offset[i]) + (y[i] - mui) / mui;
            for a in 0..p {
                let xa = x[i * p + a];
                xtwu[a] += w * xa * u;
                for b in a..p {
                    xtwx.add(a, b, w * xa * x[i * p + b]);
                }
            }
        }
        for a in 0..p {
            for b in (a + 1)..p {
                let v = xtwx.get(a, b);
                xtwx.set(b, a, v);
            }
        }
        // Weak ridge prior on all coefficients (DESeq2's numerical stabiliser).
        for j in 0..p {
            xtwx.add(j, j, RIDGE);
        }
        let inv = match invert(&xtwx) {
            Some(m) => m,
            None => {
                return GlmFit {
                    beta,
                    cov: Mat::zeros(p, p),
                    mu,
                    converged: false,
                }
            }
        };

        // Penalised (ridge) IRLS update: beta = (X'WX + ridge)^-1 X'Wu.
        let mut new_beta = vec![0.0; p];
        for (a, new_beta_a) in new_beta.iter_mut().enumerate().take(p) {
            let mut s = 0.0;
            for (b, &xtwu_b) in xtwu.iter().enumerate().take(p) {
                s += inv.get(a, b) * xtwu_b;
            }
            *new_beta_a = s;
        }
        // Divergence guard (DESeq2's `large`): leave beta and stop as unconverged.
        if new_beta.iter().any(|b| b.abs() > LARGE || !b.is_finite()) {
            break;
        }
        beta = new_beta;

        // Convergence on relative deviance change (DESeq2), evaluated at the
        // updated mu, checked only after the first iteration.
        eval_mu(&beta, &mut mu);
        let dev = deviance(&mu);
        let conv_test = (dev - dev_old).abs() / (dev.abs() + 0.1);
        if conv_test.is_nan() {
            break;
        }
        if t > 0 && conv_test < TOL {
            converged = true;
            break;
        }
        dev_old = dev;
    }

    // Final covariance from the weights at the fitted mu (DESeq2's `sigma`):
    //   (X'WX + ridge)^-1 · X'WX · (X'WX + ridge)^-1.
    eval_mu(&beta, &mut mu);
    let mut h = Mat::zeros(p, p);
    for i in 0..n {
        let w = mu[i] / (1.0 + alpha * mu[i]);
        for a in 0..p {
            let xa = x[i * p + a];
            for b in a..p {
                h.add(a, b, w * xa * x[i * p + b]);
            }
        }
    }
    for a in 0..p {
        for b in (a + 1)..p {
            let v = h.get(a, b);
            h.set(b, a, v);
        }
    }
    let mut hr = h.clone();
    for j in 0..p {
        hr.add(j, j, RIDGE);
    }
    let cov = match invert(&hr) {
        Some(inv) => sandwich(&inv, &h, p),
        None => Mat::zeros(p, p),
    };

    GlmFit {
        beta,
        cov,
        mu,
        converged,
    }
}

/// Log-determinant of the Cox-Reid adjustment matrix `X' W X`, where
/// `W_ii = mu_i / (1 + alpha * mu_i)` are the NB GLM weights at dispersion
/// `alpha`. This is the term that turns the plain NB likelihood into DESeq2's
/// Cox-Reid *adjusted profile likelihood* for the dispersion.
fn cox_reid_logdet(x: &[f64], mu: &[f64], alpha: f64, p: usize) -> f64 {
    let n = mu.len();
    let mut a = Mat::zeros(p, p);
    for i in 0..n {
        let w = mu[i] / (1.0 + alpha * mu[i]);
        for r in 0..p {
            let xr = x[i * p + r];
            if xr == 0.0 {
                continue;
            }
            for c in r..p {
                a.add(r, c, w * xr * x[i * p + c]);
            }
        }
    }
    for r in 0..p {
        for c in (r + 1)..p {
            let v = a.get(r, c);
            a.set(c, r, v);
        }
    }
    log_abs_det(&a)
}

/// Cox-Reid adjusted (optionally MAP-shrunken) dispersion estimate given fixed
/// means `mu` and the design `x` (row-major n x p).
///
/// Maximises  sum_i logNB(y_i; mu_i, alpha)  -  0.5 * log det(X' W(alpha) X)
/// and, if a `prior = (trend, var)` is supplied, adds a log-normal shrinkage
/// term  -(ln alpha - ln trend)^2 / (2 var)  on the natural-log scale. This
/// mirrors DESeq2's Cox-Reid gene-wise estimate (no prior) and MAP estimate
/// (with prior). The result is bounded to `[minDisp, max_disp]` with
/// `minDisp = 1e-8` and `max_disp = max(10, n_samples)`, matching DESeq2.
pub fn estimate_dispersion(
    y: &[f64],
    mu: &[f64],
    x: &[f64],
    p: usize,
    max_disp: f64,
    prior: Option<(f64, f64)>,
) -> f64 {
    let objective = |alpha: f64| -> f64 {
        let mut ll = 0.0;
        for i in 0..y.len() {
            ll += nb_log_pmf(y[i], mu[i], alpha);
        }
        // Cox-Reid adjustment.
        ll -= 0.5 * cox_reid_logdet(x, mu, alpha, p);
        if let Some((trend, pv)) = prior {
            if trend > 0.0 && pv > 0.0 {
                let d = alpha.ln() - trend.ln();
                ll -= d * d / (2.0 * pv);
            }
        }
        ll
    };

    // Coarse grid over log10(alpha) in [log10(minDisp), log10(max_disp)].
    let lo = -8.0_f64;
    let hi = max_disp.log10();
    let steps = 60;
    let mut best_l = lo;
    let mut best_val = f64::NEG_INFINITY;
    for k in 0..=steps {
        let la = lo + (hi - lo) * (k as f64) / (steps as f64);
        let v = objective(10f64.powf(la));
        if v > best_val {
            best_val = v;
            best_l = la;
        }
    }

    // Golden-section refine in the bracketing interval around best_l.
    let width = (hi - lo) / steps as f64;
    let mut a = best_l - width;
    let mut b = best_l + width;
    const GR: f64 = 0.618_033_988_749_895;
    let mut c = b - GR * (b - a);
    let mut d = a + GR * (b - a);
    let mut fc = objective(10f64.powf(c));
    let mut fd = objective(10f64.powf(d));
    for _ in 0..40 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - GR * (b - a);
            fc = objective(10f64.powf(c));
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + GR * (b - a);
            fd = objective(10f64.powf(d));
        }
        if (b - a).abs() < 1e-4 {
            break;
        }
    }
    let best = 10f64.powf(0.5 * (a + b));
    best.clamp(1e-8, max_disp)
}
