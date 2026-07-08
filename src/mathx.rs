//! Special functions and elementary statistics implemented in pure std Rust.

/// Natural log of the gamma function via the Lanczos approximation (g=7, n=9).
/// Accurate to ~1e-13 relative error for x > 0, which is more than enough here.
pub fn ln_gamma(x: f64) -> f64 {
    // Coefficients for the Lanczos approximation.
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Gamma(x)Gamma(1-x) = pi / sin(pi x)
        use std::f64::consts::PI;
        PI.ln() - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Complementary error function, `erfc(x) = 1 - erf(x)`, via the Numerical
/// Recipes `erfcc` rational-times-Gaussian approximation. Its fractional
/// (relative) error is < 1.2e-7 *everywhere*, so unlike a fixed-absolute-error
/// erf it stays accurate deep into the tail (down to underflow near 1e-300)
/// instead of collapsing to 0 below ~1e-7. The `exp` carries the magnitude, so
/// there is no catastrophic cancellation for large arguments.
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let tau = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let ans = t * tau.exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Standard normal cumulative distribution function, `P(Z <= x)`.
#[allow(dead_code)]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Two-sided standard normal tail probability, `P(|Z| > |x|) = erfc(|x|/sqrt2)`.
/// Routed through `erfc` so extreme Wald statistics give accurate tiny p-values
/// (not 0), fixing the `-log10(pvalue)` agreement without changing the ranking.
pub fn norm_two_sided_p(x: f64) -> f64 {
    erfc(x.abs() / std::f64::consts::SQRT_2)
}

/// Negative-binomial log-likelihood (single observation) with mean `mu` and
/// dispersion `alpha`, using the DESeq2 parameterisation Var = mu + alpha*mu^2.
/// Here r = 1/alpha is the "size" parameter.
pub fn nb_log_pmf(y: f64, mu: f64, alpha: f64) -> f64 {
    let mu = mu.max(1e-10);
    if alpha <= 1e-12 {
        // Poisson limit.
        return y * mu.ln() - mu - ln_gamma(y + 1.0);
    }
    let r = 1.0 / alpha;
    ln_gamma(y + r) - ln_gamma(r) - ln_gamma(y + 1.0)
        + r * (r / (r + mu)).ln()
        + y * (mu / (r + mu)).ln()
}

/// Benjamini-Hochberg FDR adjustment. `NaN` p-values are passed through as
/// `NaN` and excluded from the multiple-testing correction (matching the way
/// DESeq2 reports padj only for tested genes).
pub fn benjamini_hochberg(pvals: &[f64]) -> Vec<f64> {
    let idx: Vec<usize> = (0..pvals.len()).filter(|&i| pvals[i].is_finite()).collect();
    let m = idx.len();
    let mut out = vec![f64::NAN; pvals.len()];
    if m == 0 {
        return out;
    }
    // Sort tested indices by p-value ascending.
    let mut order = idx.clone();
    order.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap());

    // Enforce monotonicity from the largest p-value downwards.
    let mut prev = 1.0_f64;
    for k in (0..m).rev() {
        let i = order[k];
        let rank = (k + 1) as f64;
        let adj = (pvals[i] * m as f64 / rank).min(prev);
        out[i] = adj.min(1.0);
        prev = out[i];
    }
    out
}

/// Trigamma function psi'(x) = d^2/dx^2 ln Gamma(x). Used for the expected
/// sampling variance of the log-dispersion MLE when estimating the prior
/// variance (matches DESeq2's `trigamma((m - p)/2)`).
pub fn trigamma(x: f64) -> f64 {
    // Recurrence to push the argument above 6, then an asymptotic series.
    let mut x = x;
    let mut result = 0.0;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - 1/(30x^9)
    result
        + inv
            * (1.0
                + inv
                    * (0.5
                        + inv
                            * (1.0 / 6.0
                                + inv2 * (-1.0 / 30.0 + inv2 * (1.0 / 42.0 - inv2 / 30.0)))))
}

/// Median absolute deviation, scaled by 1.4826 to be a consistent estimator of
/// the standard deviation for normal data (R's `mad`).
pub fn mad(values: &[f64]) -> f64 {
    let med = median(values);
    if !med.is_finite() {
        return f64::NAN;
    }
    let dev: Vec<f64> = values
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (x - med).abs())
        .collect();
    1.482_602_218_505_602 * median(&dev)
}

/// Empirical quantile of `values` (type-7, matching R's default `quantile`).
pub fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Median of a slice (ignores NaN). Returns NaN for an empty input.
pub fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
