//! Ports of DESeq2's C++ numerical core (src/DESeq2.cpp):
//!
//! * `fit_nb_glm`  — `fitBeta`: negative-binomial IRLS with a ridge prior,
//!   deviance-based convergence, hat diagonals and the sandwich covariance;
//! * `fit_disp`    — `fitDisp`: Cox-Reid adjusted profile (or posterior)
//!   maximisation over log(alpha) by an Armijo backtracking line search;
//! * `fit_disp_grid` — `fitDispGrid`: the coarse+fine grid fallback;
//! * `optim_beta`  — the `fitNbinomGLMsOptim` fallback for rows whose IRLS
//!   did not converge (R uses L-BFGS-B; here a damped Newton with the same
//!   objective, bounds and starting rules).

use crate::lbfgsb::optim_lbfgsb;
use crate::linalg::{invert, log_abs_det, qr_solve, Mat};
use crate::mathx::{digamma, dnbinom_mu_log, dnorm_log, ln_gamma, r_sum};

const LN2: f64 = std::f64::consts::LN_2;

/// Result of a per-gene GLM fit (natural-log coefficient scale).
pub struct GlmFit {
    pub beta: Vec<f64>,
    /// Sandwich covariance (X'WX + ridge)^-1 X'WX (X'WX + ridge)^-1.
    pub sigma: Mat,
    /// Final fitted means WITHOUT the minmu floor (recomputed from beta as in
    /// R: `nf * exp(X beta)`), matching the `mu` DESeq2 stores and uses for
    /// Cook's distances.
    pub mu: Vec<f64>,
    /// Hat-matrix diagonals (with the ridge), from the floored working mu.
    pub hat: Vec<f64>,
    #[allow(dead_code)]
    pub iter: usize,
    pub converged: bool,
}

/// Fit the NB GLM with dispersion `alpha` fixed; port of `fitBeta` for one row.
///
/// * `y`  counts, `x` row-major n x p, `nf` size factors (normalization),
/// * `beta_init` starting coefficients (natural log),
/// * `lambda` ridge diagonal on the natural-log scale.
#[allow(clippy::too_many_arguments)]
pub fn fit_nb_glm(
    y: &[f64],
    x: &[f64],
    nf: &[f64],
    p: usize,
    alpha: f64,
    beta_init: &[f64],
    lambda: &[f64],
    tol: f64,
    max_iter: usize,
    minmu: f64,
) -> GlmFit {
    const LARGE: f64 = 30.0;
    let n = y.len();
    let mut beta = beta_init.to_vec();

    let eval_mu_floored = |beta: &[f64], mu: &mut [f64]| {
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            mu[i] = (nf[i] * eta.exp()).max(minmu);
        }
    };

    let mut mu = vec![0.0; n];
    eval_mu_floored(&beta, &mut mu);

    let mut iter = 0usize;
    let mut dev;
    let mut dev_old = 0.0;

    // Scratch for the QR solve of the ridge-augmented weighted system
    // (fitBeta's useQR=TRUE path): rows = [X * sqrt(w); sqrt(ridge)].
    let mm = n + p;
    let mut aq = vec![0.0; mm * p];
    let mut bq = vec![0.0; mm];
    let sqrt_lambda: Vec<f64> = lambda.iter().map(|l| l.sqrt()).collect();

    for t in 0..max_iter {
        iter += 1;
        // weights and working response at the current (floored) mu
        for v in aq.iter_mut() {
            *v = 0.0;
        }
        for v in bq.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n {
            let mui = mu[i];
            let w_sqrt = (mui / (1.0 + alpha * mui)).sqrt();
            let z = (mui / nf[i]).ln() + (y[i] - mui) / mui;
            for j in 0..p {
                aq[i * p + j] = x[i * p + j] * w_sqrt;
            }
            bq[i] = z * w_sqrt;
        }
        for j in 0..p {
            aq[(n + j) * p + j] = sqrt_lambda[j];
        }
        match qr_solve(&mut aq, mm, p, &mut bq) {
            Some(nb) => beta = nb,
            None => {
                iter = max_iter;
                break;
            }
        }
        // Divergence: keep the diverged beta, do NOT update mu (C++ behavior).
        if beta.iter().any(|b| b.abs() > LARGE || !b.is_finite()) {
            iter = max_iter;
            break;
        }
        eval_mu_floored(&beta, &mut mu);
        dev = 0.0;
        for i in 0..n {
            dev += -2.0 * dnbinom_mu_log(y[i], 1.0 / alpha, mu[i]);
        }
        let conv_test = (dev - dev_old).abs() / (dev.abs() + 0.1);
        if conv_test.is_nan() {
            iter = max_iter;
            break;
        }
        if t > 0 && conv_test < tol {
            break;
        }
        dev_old = dev;
    }

    // Final weights at the last working mu (floored), hat diagonals with the
    // ridge, and the sandwich covariance — exactly as at the end of fitBeta.
    let mut xtwx = Mat::zeros(p, p);
    for i in 0..n {
        let w = mu[i] / (1.0 + alpha * mu[i]);
        for a in 0..p {
            let xa = x[i * p + a];
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
    let mut xtwxr = xtwx.clone();
    for j in 0..p {
        xtwxr.add(j, j, lambda[j]);
    }
    let (hat, sigma) = match invert(&xtwxr) {
        Some(inv) => {
            let mut hat = vec![0.0; n];
            for i in 0..n {
                let w = mu[i] / (1.0 + alpha * mu[i]);
                let mut q = 0.0;
                for a in 0..p {
                    for b in 0..p {
                        q += x[i * p + a] * inv.get(a, b) * x[i * p + b];
                    }
                }
                hat[i] = w * q;
            }
            // sigma = inv * xtwx * inv
            let mut t = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    let mut s = 0.0;
                    for k in 0..p {
                        s += xtwx.get(i, k) * inv.get(k, j);
                    }
                    t.set(i, j, s);
                }
            }
            let mut sigma = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    let mut s = 0.0;
                    for k in 0..p {
                        s += inv.get(i, k) * t.get(k, j);
                    }
                    sigma.set(i, j, s);
                }
            }
            (hat, sigma)
        }
        None => (vec![0.0; n], Mat::zeros(p, p)),
    };

    // The stored mu is recomputed WITHOUT the floor (R-side recompute).
    let mut mu_out = vec![0.0; n];
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += x[i * p + j] * beta[j];
        }
        mu_out[i] = nf[i] * eta.exp();
    }

    GlmFit {
        beta,
        sigma,
        mu: mu_out,
        hat,
        iter,
        converged: iter < max_iter,
    }
}

/// The `optim` fallback of `fitNbinomGLMs` for rows whose IRLS did not
/// converge (or produced NaN betas / non-positive variances): R's
/// `optim(betaRow, objectiveFn, method="L-BFGS-B", lower=-30, upper=30)`,
/// reproduced with the ported L-BFGS-B and R's exact objective
/// (saddle-point dnbinom + dnorm prior, long-double sums).
pub struct OptimResult {
    pub beta_log2: Vec<f64>,
    pub sigma_log2: Mat,
    pub mu: Vec<f64>,
    pub converged: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn optim_beta(
    y: &[f64],
    x: &[f64],
    nf: &[f64],
    p: usize,
    alpha: f64,
    start_log2: &[f64],
    lambda_log2: &[f64],
    minmu: f64,
) -> OptimResult {
    const BOUND: f64 = 30.0;
    let n = y.len();
    let size = 1.0 / alpha;
    let sd: Vec<f64> = lambda_log2.iter().map(|l| (1.0 / l).sqrt()).collect();

    let mut ll_buf = vec![0.0; n];
    let mut lp_buf = vec![0.0; p];
    let mut objective = |b: &[f64]| -> f64 {
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * b[j];
            }
            let mu = nf[i] * 2f64.powf(eta);
            ll_buf[i] = dnbinom_mu_log(y[i], size, mu);
        }
        let log_like = r_sum(&ll_buf);
        for j in 0..p {
            lp_buf[j] = dnorm_log(b[j], 0.0, sd[j]);
        }
        let log_prior = r_sum(&lp_buf);
        let neg = -(log_like + log_prior);
        if neg.is_finite() {
            neg
        } else {
            1e300
        }
    };

    let lower = vec![-BOUND; p];
    let upper = vec![BOUND; p];
    let o = optim_lbfgsb(start_log2, &lower, &upper, &mut objective);

    let b = o.par;
    // standard errors: sandwich at the optimum, minmu-floored mu for weights
    let mut mu = vec![0.0; n];
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += x[i * p + j] * b[j];
        }
        mu[i] = nf[i] * 2f64.powf(eta);
    }
    let mut xtwx = Mat::zeros(p, p);
    for i in 0..n {
        let m = mu[i].max(minmu);
        let w = 1.0 / (1.0 / m + alpha);
        for a in 0..p {
            let xa = x[i * p + a];
            for c in a..p {
                xtwx.add(a, c, w * xa * x[i * p + c]);
            }
        }
    }
    for a in 0..p {
        for c in (a + 1)..p {
            let v = xtwx.get(a, c);
            xtwx.set(c, a, v);
        }
    }
    // ridge on the natural-log scale (lambdaNatLogScale), as in R
    let mut xtwxr = xtwx.clone();
    for j in 0..p {
        xtwxr.add(j, j, lambda_log2[j] / (LN2 * LN2));
    }
    let sigma_nat = match invert(&xtwxr) {
        Some(inv) => {
            let mut t = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    let mut s = 0.0;
                    for k in 0..p {
                        s += xtwx.get(i, k) * inv.get(k, j);
                    }
                    t.set(i, j, s);
                }
            }
            let mut sig = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    let mut s = 0.0;
                    for k in 0..p {
                        s += inv.get(i, k) * t.get(k, j);
                    }
                    sig.set(i, j, s);
                }
            }
            sig
        }
        None => Mat::zeros(p, p),
    };
    let scale = 1.0 / (LN2 * LN2);
    let mut sigma_log2 = Mat::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            sigma_log2.set(i, j, scale * sigma_nat.get(i, j));
        }
    }

    OptimResult {
        beta_log2: b,
        sigma_log2,
        mu,
        converged: o.convergence == 0,
    }
}

// ---------------------------------------------------------------------------
// Dispersion fitting (ports of log_posterior / dlog_posterior / fitDisp /
// fitDispGrid, without observation weights)
// ---------------------------------------------------------------------------

/// DESeq2's log posterior for log(alpha): NB log likelihood (dropping terms
/// constant in alpha), the Cox-Reid term, and optionally the log-normal prior.
pub fn disp_log_posterior(
    log_alpha: f64,
    y: &[f64],
    mu: &[f64],
    x: &[f64],
    p: usize,
    prior: Option<(f64, f64)>, // (log_alpha_prior_mean, sigmasq)
) -> f64 {
    let n = y.len();
    let alpha = log_alpha.exp();
    // Cox-Reid: -0.5 * log det(X' W X), W = 1/(1/mu + alpha)
    let mut b = Mat::zeros(p, p);
    for i in 0..n {
        let w = 1.0 / (1.0 / mu[i] + alpha);
        for r in 0..p {
            let xr = x[i * p + r];
            if xr == 0.0 {
                continue;
            }
            for c in r..p {
                b.add(r, c, w * xr * x[i * p + c]);
            }
        }
    }
    for r in 0..p {
        for c in (r + 1)..p {
            let v = b.get(r, c);
            b.set(c, r, v);
        }
    }
    let cr_term = -0.5 * log_abs_det(&b);

    let r_inv = 1.0 / alpha;
    let mut ll = 0.0;
    for i in 0..n {
        ll += ln_gamma(y[i] + r_inv) - ln_gamma(r_inv)
            - y[i] * (mu[i] + r_inv).ln()
            - r_inv * (1.0 + mu[i] * alpha).ln();
    }
    let prior_part = match prior {
        Some((mean, sigmasq)) => -0.5 * (log_alpha - mean).powi(2) / sigmasq,
        None => 0.0,
    };
    ll + prior_part + cr_term
}

/// Derivative of the log posterior w.r.t. log(alpha).
fn disp_dlog_posterior(
    log_alpha: f64,
    y: &[f64],
    mu: &[f64],
    x: &[f64],
    p: usize,
    prior: Option<(f64, f64)>,
) -> f64 {
    let n = y.len();
    let alpha = log_alpha.exp();
    // b = X'WX, db = X' dW X with dW/dalpha = -W^2
    let mut b = Mat::zeros(p, p);
    let mut db = Mat::zeros(p, p);
    for i in 0..n {
        let w = 1.0 / (1.0 / mu[i] + alpha);
        let dw = -w * w;
        for r in 0..p {
            let xr = x[i * p + r];
            if xr == 0.0 {
                continue;
            }
            for c in r..p {
                b.add(r, c, w * xr * x[i * p + c]);
                db.add(r, c, dw * xr * x[i * p + c]);
            }
        }
    }
    for r in 0..p {
        for c in (r + 1)..p {
            let v = b.get(r, c);
            b.set(c, r, v);
            let v = db.get(r, c);
            db.set(c, r, v);
        }
    }
    // cr_term = -0.5 * trace(b^-1 db)
    let cr_term = match invert(&b) {
        Some(binv) => {
            let mut tr = 0.0;
            for i in 0..p {
                for j in 0..p {
                    tr += binv.get(i, j) * db.get(j, i);
                }
            }
            -0.5 * tr
        }
        None => 0.0,
    };

    let a1 = 1.0 / alpha;
    let a2 = a1 * a1;
    let mut s = 0.0;
    for i in 0..n {
        s += digamma(a1) + (1.0 + mu[i] * alpha).ln()
            - mu[i] * alpha / (1.0 + mu[i] * alpha)
            - digamma(y[i] + a1)
            + y[i] / (mu[i] + a1);
    }
    let ll_part = a2 * s;
    let prior_part = match prior {
        Some((mean, sigmasq)) => -(log_alpha - mean) / sigmasq,
        None => 0.0,
    };
    (ll_part + cr_term) * alpha + prior_part
}

pub struct DispFit {
    pub log_alpha: f64,
    pub iter: usize,
    pub initial_lp: f64,
    pub last_lp: f64,
}

/// Port of `fitDisp` (one row): Armijo backtracking line search on log(alpha).
#[allow(clippy::too_many_arguments)]
pub fn fit_disp(
    y: &[f64],
    mu: &[f64],
    x: &[f64],
    p: usize,
    log_alpha_init: f64,
    prior: Option<(f64, f64)>,
    min_log_alpha: f64, // log(minDisp/10)
    kappa_0: f64,       // 1.0
    tol: f64,           // 1e-6
    maxit: usize,       // 100
) -> DispFit {
    const EPSILON: f64 = 1.0e-4;
    let mut a = log_alpha_init;
    let mut lp = disp_log_posterior(a, y, mu, x, p, prior);
    let mut dlp = disp_dlog_posterior(a, y, mu, x, p, prior);
    let initial_lp = lp;
    let mut kappa = kappa_0;
    let mut iter = 0usize;
    let mut iter_accept = 0usize;

    for _t in 0..maxit {
        iter += 1;
        let mut a_propose = a + kappa * dlp;
        if a_propose < -30.0 {
            kappa = (-30.0 - a) / dlp;
        }
        if a_propose > 10.0 {
            kappa = (10.0 - a) / dlp;
        }
        a_propose = a + kappa * dlp;
        let theta_kappa = -disp_log_posterior(a_propose, y, mu, x, p, prior);
        let theta_hat_kappa = -lp - kappa * EPSILON * dlp * dlp;
        if theta_kappa <= theta_hat_kappa {
            iter_accept += 1;
            a += kappa * dlp;
            let lpnew = disp_log_posterior(a, y, mu, x, p, prior);
            let change = lpnew - lp;
            if change < tol {
                lp = lpnew;
                break;
            }
            if a < min_log_alpha {
                break;
            }
            lp = lpnew;
            dlp = disp_dlog_posterior(a, y, mu, x, p, prior);
            kappa = (kappa * 1.1).min(kappa_0);
            if iter_accept % 5 == 0 {
                kappa /= 2.0;
            }
        } else {
            kappa /= 2.0;
        }
    }

    DispFit {
        log_alpha: a,
        iter,
        initial_lp,
        last_lp: lp,
    }
}

/// Port of `fitDispGrid` (one row): coarse grid over log(alpha) in
/// [log(minDisp), log(maxDisp)] with `grid_n` points, then a fine grid of the
/// same size spanning +/- one coarse step around the best point. Returns
/// log(alpha). Ties resolve to the FIRST maximum (as Armadillo's index_max).
pub fn fit_disp_grid(
    y: &[f64],
    mu: &[f64],
    x: &[f64],
    p: usize,
    max_disp: f64,
    prior: Option<(f64, f64)>,
) -> f64 {
    const GRID_N: usize = 20;
    let min_log_alpha = (1e-8_f64).ln();
    let max_log_alpha = max_disp.ln();
    let delta = (max_log_alpha - min_log_alpha) / (GRID_N as f64 - 1.0);

    let eval_grid = |lo: f64, hi: f64| -> f64 {
        let mut best_a = lo;
        let mut best_v = f64::NEG_INFINITY;
        for t in 0..GRID_N {
            let a = lo + (hi - lo) * (t as f64) / (GRID_N as f64 - 1.0);
            let v = disp_log_posterior(a, y, mu, x, p, prior);
            if v > best_v {
                best_v = v;
                best_a = a;
            }
        }
        best_a
    };

    let a_hat = eval_grid(min_log_alpha, max_log_alpha);
    eval_grid(a_hat - delta, a_hat + delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optim_matches_r_lbfgsb() {
        let dir = env!("RUST_DESEQ2_REF_DIR");
        let case = std::fs::read_to_string(format!("{dir}/optim_case.txt")).unwrap();
        let lines: Vec<&str> = case.lines().collect();
        let parse = |s: &str| -> Vec<f64> {
            s.split('\t').map(|v| v.trim().parse().unwrap()).collect()
        };
        let y = parse(lines[0]);
        let nf = parse(lines[1]);
        let alpha: f64 = lines[2].trim().parse().unwrap();
        let beta_row = parse(lines[3]);
        let n = y.len();
        let p = beta_row.len();
        let mut x = vec![0.0; n * p];
        for i in 0..n {
            let row = parse(lines[4 + i]);
            x[i * p..(i + 1) * p].copy_from_slice(&row);
        }
        let expect = std::fs::read_to_string(format!("{dir}/optim_expect.txt")).unwrap();
        let elines: Vec<&str> = expect.lines().collect();
        let epar = parse(elines[0]);
        let evalue: f64 = elines[1].trim().parse().unwrap();
        let econv: i32 = elines[2].trim().parse().unwrap();

        let lambda_log2 = vec![1e-6; p];
        // objective value at R's endpoint, for bit-level comparison
        {
            let size = 1.0 / alpha;
            let sd: Vec<f64> = lambda_log2.iter().map(|l| (1.0_f64 / l).sqrt()).collect();
            let mut ll = vec![0.0; n];
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..p {
                    eta += x[i * p + j] * epar[j];
                }
                let mu = nf[i] * 2f64.powf(eta);
                ll[i] = crate::mathx::dnbinom_mu_log(y[i], size, mu);
            }
            let log_like = crate::mathx::r_sum(&ll);
            let lp: Vec<f64> = (0..p).map(|j| crate::mathx::dnorm_log(epar[j], 0.0, sd[j])).collect();
            let f = -(log_like + crate::mathx::r_sum(&lp));
            eprintln!("rust f(epar) = {:.17e}\nR    value   = {:.17e}\ndiff = {:e}", f, evalue, (f - evalue).abs());
        }
        let o = optim_beta(&y, &x, &nf, p, alpha, &beta_row, &lambda_log2, 0.5);
        eprintln!("rust par: {:?}", o.beta_log2);
        eprintln!("R    par: {:?}", epar);
        assert_eq!(o.converged, econv == 0, "convergence flag");
        for j in 0..p {
            let d = (o.beta_log2[j] - epar[j]).abs();
            assert!(
                d <= 1e-10 * epar[j].abs().max(1.0),
                "par[{j}]: rust {} vs R {} (diff {d:e})",
                o.beta_log2[j],
                epar[j]
            );
        }
        let _ = evalue;
    }
}
