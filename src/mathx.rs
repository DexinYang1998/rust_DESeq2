//! Special functions and elementary statistics implemented in pure std Rust.
//! Where a function feeds directly into reported numbers (pnorm, lowess,
//! quantile, BH), it is a faithful port of the corresponding R routine so the
//! output matches R to double precision.

/// Clenshaw evaluation of a Chebyshev series (port of R nmath chebyshev_eval).
fn chebyshev_eval(x: f64, a: &[f64], n: usize) -> f64 {
    let twox = x * 2.0;
    let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

const M_LN_SQRT_2PI: f64 = 0.918_938_533_204_672_741_780_329_736_406;

/// Port of R nmath `lgammacor(x)` for x >= 10.
fn lgammacor(x: f64) -> f64 {
    const ALGMCS: [f64; 15] = [
        0.1666389480451863247205729650822e+0,
        -0.1384948176067563840732986059135e-4,
        0.9810825646924729426157171547487e-8,
        -0.1809129475572494194263306266719e-10,
        0.6221098041892605227126015543416e-13,
        -0.3399615005417721944303330599666e-15,
        0.2683181998482698748957538846666e-17,
        -0.2868042435334643284144622399999e-19,
        0.3962837061046434803679306666666e-21,
        -0.6831888753985766870111999999999e-23,
        0.1429227355942498147573333333333e-24,
        -0.3547598158101070547199999999999e-26,
        0.1025680058010470912e-27,
        -0.3401102254316748799999999999999e-29,
        0.1276642195630062933333333333333e-30,
    ];
    const NALGM: usize = 5;
    const XBIG: f64 = 94906265.62425156;
    if x < XBIG {
        let tmp = 10.0 / x;
        chebyshev_eval(tmp * tmp * 2.0 - 1.0, &ALGMCS, NALGM) / x
    } else {
        1.0 / (x * 12.0)
    }
}

/// Port of R nmath `gammafn(x)` restricted to 0 < x <= 10 (the only range
/// `lgammafn` uses it for).
fn gammafn_small(x: f64) -> f64 {
    const GAMCS: [f64; 42] = [
        0.8571195590989331421920062399942e-2,
        0.4415381324841006757191315771652e-2,
        0.5685043681599363378632664588789e-1,
        -0.4219835396418560501012500186624e-2,
        0.1326808181212460220584006796352e-2,
        -0.1893024529798880432523947023886e-3,
        0.3606925327441245256578082217225e-4,
        -0.6056761904460864218485548290365e-5,
        0.1055829546302283344731823509093e-5,
        -0.1811967365542384048291855891166e-6,
        0.3117724964715322277790254593169e-7,
        -0.5354219639019687140874081024347e-8,
        0.9193275519859588946887786825940e-9,
        -0.1577941280288339761767423273953e-9,
        0.2707980622934954543266540433089e-10,
        -0.4646818653825730144081661058933e-11,
        0.7973350192007419656460767175359e-12,
        -0.1368078209830916025799499172309e-12,
        0.2347319486563800657233471771688e-13,
        -0.4027432614949066932766570534699e-14,
        0.6910051747372100912138336975257e-15,
        -0.1185584500221992907052387126192e-15,
        0.2034148542496373955201026051932e-16,
        -0.3490054341717405849274012949108e-17,
        0.5987993856485305567135051066026e-18,
        -0.1027378057872228074490069778431e-18,
        0.1762702816060529824942759660748e-19,
        -0.3024320653735306260958772112042e-20,
        0.5188914660218397839717833550506e-21,
        -0.8902770842456576692449251601066e-22,
        0.1527474068493342602274596891306e-22,
        -0.2620731256187362900257328332799e-23,
        0.4496464047830538670331046570666e-24,
        -0.7714712731336877911703901525333e-25,
        0.1323635453126044036486572714666e-25,
        -0.2270999412942928816702313813333e-26,
        0.3896418998003991449320816639999e-27,
        -0.6685198115125953327792127999999e-28,
        0.1146998663140024384347613866666e-28,
        -0.1967938586345134677295103999999e-29,
        0.3376448816585338090334890666666e-30,
        -0.5793070335782135784625493333333e-31,
    ];
    const NGAM: usize = 22;
    const XSML: f64 = 2.2474362225598545e-308;

    let mut n = x as i64; // x > 0 here
    let y = x - n as f64;
    n -= 1;
    let mut value = chebyshev_eval(y * 2.0 - 1.0, &GAMCS, NGAM) + 0.9375;
    if n == 0 {
        return value;
    }
    if n < 0 {
        // 0 < x < 1
        if y.abs() < XSML {
            return f64::INFINITY;
        }
        let n = -n;
        for i in 0..n {
            value /= x + i as f64;
        }
        value
    } else {
        for i in 1..=n {
            value *= y + i as f64;
        }
        value
    }
}

/// Natural log of the gamma function for x > 0: an exact port of R nmath
/// `lgammafn` (positive-argument branches), bitwise-matching R's `lgamma`.
pub fn ln_gamma(x: f64) -> f64 {
    const XMAX: f64 = 2.5327372760800758e305;
    if x.is_nan() {
        return x;
    }
    debug_assert!(x > 0.0, "ln_gamma requires x > 0 (got {x})");
    let y = x;
    if y < 1e-306 {
        return -y.ln();
    }
    if y <= 10.0 {
        return gammafn_small(x).abs().ln();
    }
    if y > XMAX {
        return f64::INFINITY;
    }
    if x > 1e17 {
        x * (x.ln() - 1.0)
    } else if x > 4934720.0 {
        M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x
    } else {
        M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x + lgammacor(x)
    }
}

/// Port of R nmath `dpsifn(x, n, kode=1, m=1)` for x > 0 (Amos TOMS 610):
/// computes (-1)^(n+1)/gamma(n+1) * psi_n(x). Backbone of digamma/trigamma.
fn dpsifn(x: f64, n: usize) -> f64 {
    const BVALUES: [f64; 22] = [
        1.00000000000000000e+00,
        -5.00000000000000000e-01,
        1.66666666666666667e-01,
        -3.33333333333333333e-02,
        2.38095238095238095e-02,
        -3.33333333333333333e-02,
        7.57575757575757576e-02,
        -2.53113553113553114e-01,
        1.16666666666666667e+00,
        -7.09215686274509804e+00,
        5.49711779448621554e+01,
        -5.29124242424242424e+02,
        6.19212318840579710e+03,
        -8.65802531135531136e+04,
        1.42551716666666667e+06,
        -2.72982310678160920e+07,
        6.01580873900642368e+08,
        -1.51163157670921569e+10,
        4.29614643061166667e+11,
        -1.37116552050883328e+13,
        4.88332318973593167e+14,
        -1.92965793419400681e+16,
    ];
    debug_assert!(x > 0.0);
    let xln = x.ln();
    // R case shortcut for very large x
    let lrg = 1.0 / (2.0 * f64::EPSILON);
    if n == 0 && x * xln > lrg {
        return -xln;
    } else if n >= 1 && x > n as f64 * lrg {
        return (-(n as f64) * xln).exp() / n as f64;
    }

    let r1m5 = std::f64::consts::LOG10_2;
    let r1m4 = f64::EPSILON * 0.5;
    let wdtol = r1m4.max(0.5e-18);
    let elim = 2.302 * (1021.0 * r1m5 - 3.0);
    let rln = (r1m5 * 53.0).min(18.06);
    let mut fln = rln.max(3.0) - 3.0;
    let yint = 3.50 + 0.40 * fln;
    let slope = 0.21 + fln * (0.0006038 * fln + 0.008677);

    let nn = n; // m = 1
    let fn_ = nn as f64;
    let t = (fn_ + 1.0) * xln;
    if t.abs() > elim {
        if t <= 0.0 {
            return f64::NAN; // ierr = 2
        }
        return 0.0; // underflow: ans set to 0
    }
    if x < wdtol {
        // x^(-n-1)
        return x.powi(-(n as i32) - 1);
    }
    // compute xmin and the number of terms of the series
    let mut xm = yint + slope * fn_;
    let mx = xm as i32 + 1;
    let xmin = mx as f64;
    let mut use_series = false;
    if n != 0 {
        xm = -2.302 * rln - 0.0_f64.min(xln);
        let arg = 0.0_f64.min(xm / n as f64);
        let eps = arg.exp();
        let xm2 = if arg.abs() < 1.0e-3 { -arg } else { 1.0 - eps };
        fln = x * xm2 / eps;
        let xm3 = xmin - x;
        if xm3 > 7.0 && fln < 15.0 {
            use_series = true;
        }
    }
    if use_series {
        // series: sum over (x+k)^(-n-1)
        let nterms = fln as i32 + 1;
        let np = n as i32 + 1;
        let t1 = (n as f64 + 1.0) * xln;
        let t = (-t1).exp();
        let mut s = t;
        let mut den = x;
        for _ in 1..=nterms {
            den += 1.0;
            s += den.powi(-np);
        }
        return s;
    }

    let (xdmy, xdmln, xinc) = if x < xmin {
        let nx = x as i32;
        let xinc = xmin - nx as f64;
        let xd = x + xinc;
        (xd, xd.ln(), xinc)
    } else {
        (x, xln, 0.0)
    };

    // asymptotic expansion at xdmy
    let t = fn_ * xdmln;
    let t1 = xdmln + xdmln;
    let t2 = t + xdmln;
    let tk = t.abs().max(t1.abs()).max(t2.abs());
    if tk > elim {
        return 0.0; // underflow path (nz++)
    }
    let tss = (-t).exp();
    let tt = 0.5 / xdmy;
    let mut t1 = tt;
    let tst = wdtol * tt;
    if nn != 0 {
        t1 = tt + 1.0 / fn_;
    }
    let rxsq = 1.0 / (xdmy * xdmy);
    let ta = 0.5 * rxsq;
    let mut t = (fn_ + 1.0) * ta;
    let mut s = t * BVALUES[2];
    if s.abs() >= tst {
        let mut tk = 2.0;
        for k in 4..=22usize {
            t = t * ((tk + fn_ + 1.0) / (tk + 1.0)) * ((tk + fn_) / (tk + 2.0)) * rxsq;
            let trm_k = t * BVALUES[k - 1];
            if trm_k.abs() < tst {
                break;
            }
            s += trm_k;
            tk += 2.0;
        }
    }
    s = (s + t1) * tss;

    if xinc != 0.0 {
        let nx = xinc as i32;
        if nn == 0 {
            // L20: digamma backward recursion
            for i in 1..=nx {
                s += 1.0 / (x + (nx - i) as f64);
            }
            // L30
            return s - xdmln;
        } else {
            let np = nn as i32 + 1;
            let mut xm = xinc - 1.0;
            let mut fx = x + xm;
            for _ in 1..=nx {
                s += fx.powi(-np);
                xm -= 1.0;
                fx = x + xm;
            }
        }
    }
    if fn_ == 0.0 {
        // L30 for kode == 1
        return s - xdmln;
    }
    s
}

/// R's `digamma(x)` for x > 0 (exact nmath port).
pub fn digamma(x: f64) -> f64 {
    -dpsifn(x, 0)
}

/// R's `trigamma(x)` for x > 0 (exact nmath port).
pub fn trigamma(x: f64) -> f64 {
    dpsifn(x, 1)
}

const M_1_SQRT_2PI: f64 = 0.398_942_280_401_432_677_939_946_059_934;
const M_SQRT_32: f64 = 5.656_854_249_492_380_195_206_754_896_84;
const SIXTEN: f64 = 16.0;

/// Standard normal distribution function, a faithful port of R's `pnorm`
/// (Cody's algorithm, as in R's pnorm.c). Returns (cum, ccum) =
/// (P(Z <= x), P(Z > x)) with full double precision including far tails.
fn pnorm_both(x: f64) -> (f64, f64) {
    const A: [f64; 5] = [
        2.2352520354606839287,
        161.02823106855587881,
        1067.6894854603709582,
        18154.981253343561249,
        0.065682337918207449113,
    ];
    const B: [f64; 4] = [
        47.20258190468824187,
        976.09855173777669322,
        10260.932208618978205,
        45507.789335026729956,
    ];
    const C: [f64; 9] = [
        0.39894151208813466764,
        8.8831497943883759412,
        93.506656132177855979,
        597.27027639480026226,
        2494.5375852903726711,
        6848.1904505362823326,
        11602.651437647350124,
        9842.7148383839780218,
        1.0765576773720192317e-8,
    ];
    const D: [f64; 8] = [
        22.266688044328115691,
        235.38790178262499861,
        1519.377599407554805,
        6485.558298266760755,
        18615.571640885098091,
        34900.952721145977266,
        38912.003286093271411,
        19685.429676859990727,
    ];
    const P: [f64; 6] = [
        0.21589853405795699,
        0.1274011611602473639,
        0.022235277870649807,
        0.001421619193227893466,
        2.9112874951168792e-5,
        0.02307344176494017303,
    ];
    const Q: [f64; 5] = [
        1.28426009614491121,
        0.468238212480865118,
        0.0659881378689285515,
        0.00378239633202758244,
        7.29751555083966205e-5,
    ];

    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let eps = f64::EPSILON * 0.5;
    let y = x.abs();

    let (cum, ccum);
    if y <= 0.67448975 {
        // qnorm(3/4)
        let mut xnum = 0.0;
        let mut xden = 0.0;
        if y > eps {
            let xsq = x * x;
            xnum = A[4] * xsq;
            xden = xsq;
            for i in 0..3 {
                xnum = (xnum + A[i]) * xsq;
                xden = (xden + B[i]) * xsq;
            }
        }
        let temp = x * (xnum + A[3]) / (xden + B[3]);
        cum = 0.5 + temp;
        ccum = 0.5 - temp;
        (cum, ccum)
    } else if y <= M_SQRT_32 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let temp = (xnum + C[7]) / (xden + D[7]);
        // do_del(y)
        let xsq = (y * SIXTEN).trunc() / SIXTEN;
        let del = (y - xsq) * (y + xsq);
        let mut cum = (-xsq * xsq * 0.5).exp() * (-del * 0.5).exp() * temp;
        let mut ccum = 1.0 - cum;
        if x > 0.0 {
            std::mem::swap(&mut cum, &mut ccum);
        }
        (cum, ccum)
    } else if (-37.5193 < x && x < 8.2924) || (-8.2924 < x && x < 37.5193) {
        let xsq = 1.0 / (x * x);
        let mut xnum = P[5] * xsq;
        let mut xden = xsq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * xsq;
            xden = (xden + Q[i]) * xsq;
        }
        let mut temp = xsq * (xnum + P[4]) / (xden + Q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;
        // do_del(x)
        let xsq2 = (x * SIXTEN).trunc() / SIXTEN;
        let del = (x - xsq2) * (x + xsq2);
        let mut cum = (-xsq2 * xsq2 * 0.5).exp() * (-del * 0.5).exp() * temp;
        let mut ccum = 1.0 - cum;
        if x > 0.0 {
            std::mem::swap(&mut cum, &mut ccum);
        }
        (cum, ccum)
    } else if x > 0.0 {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    }
}

/// P(Z > x): R's `pnorm(x, lower.tail=FALSE)`.
pub fn pnorm_upper(x: f64) -> f64 {
    pnorm_both(x).1
}

/// Two-sided Wald p-value as DESeq2 computes it: `2*pnorm(|stat|, lower=FALSE)`.
pub fn wald_pvalue(stat: f64) -> f64 {
    if !stat.is_finite() {
        return f64::NAN;
    }
    2.0 * pnorm_upper(stat.abs())
}

/// Regularized incomplete beta function I_x(a, b) via the continued fraction
/// (Numerical Recipes `betai`/`betacf`), accurate to ~1e-14.
fn pbeta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// F distribution CDF, P(F(df1, df2) <= x).
fn pf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    pbeta(df1 * x / (df1 * x + df2), df1 / 2.0, df2 / 2.0)
}

/// F distribution quantile, R's `qf(p, df1, df2)`, via bisection + Newton on
/// the CDF; agrees with R to ~1e-12 relative. Used for the Cook's cutoff
/// `qf(0.99, p, m - p)`.
pub fn qf(p: f64, df1: f64, df2: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0 && df1 > 0.0 && df2 > 0.0);
    // Bracket the quantile.
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while pf(hi, df1, df2) < p {
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    // Bisection to a decent width, then Newton refinement.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if pf(mid, df1, df2) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) / hi.max(1e-300) < 1e-13 {
            break;
        }
    }
    let mut x = 0.5 * (lo + hi);
    // Newton polish with the analytic density.
    for _ in 0..8 {
        let cdf = pf(x, df1, df2);
        let ldens = (df1 / 2.0) * (df1 / df2).ln() + (df1 / 2.0 - 1.0) * x.ln()
            - ((df1 + df2) / 2.0) * (1.0 + df1 * x / df2).ln()
            - (ln_gamma(df1 / 2.0) + ln_gamma(df2 / 2.0) - ln_gamma((df1 + df2) / 2.0));
        let dens = ldens.exp();
        if dens <= 0.0 || !dens.is_finite() {
            break;
        }
        let step = (cdf - p) / dens;
        let xn = x - step;
        if xn <= lo || xn >= hi {
            break;
        }
        x = xn;
        if step.abs() < 1e-14 * x {
            break;
        }
    }
    x
}

/// Benjamini-Hochberg FDR adjustment, matching R's `p.adjust(p, "BH")`:
/// `NaN` p-values are passed through as `NaN` and excluded (m = number of
/// non-NA p-values, matching p.adjust's lazily-evaluated default n).
pub fn benjamini_hochberg(pvals: &[f64]) -> Vec<f64> {
    let idx: Vec<usize> = (0..pvals.len()).filter(|&i| pvals[i].is_finite()).collect();
    let m = idx.len();
    let mut out = vec![f64::NAN; pvals.len()];
    if m == 0 {
        return out;
    }
    let mut order = idx.clone();
    // R's order() breaks p-value ties by index (stable sort on a stable key).
    order.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap().then(a.cmp(&b)));

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

/// Median absolute deviation. R's `mad` default scales by the LITERAL
/// constant 1.4826 (not the exact 1/qnorm(3/4) = 1.48260221850560...), and
/// matching that literal matters for reproducing DESeq2's dispersion prior.
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
    1.4826 * median(&dev)
}

/// Empirical quantile of a sorted slice (type-7, matching R's default).
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

/// R's trimmed mean, `mean(x, trim=t)`: drop floor(n*t) values from each end
/// of the sorted vector and average the rest.
pub fn trimmed_mean(values: &[f64], trim: f64) -> f64 {
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

/// Faithful port of R's `lowess` (src/library/stats/src/lowess.c, `clowess`):
/// local linear regression over the nearest `f*n` points with tricube weights,
/// `nsteps` bisquare robustness iterations, and the `delta` speedup that skips
/// nearby points and linearly interpolates. `x` must be sorted ascending.
/// R defaults: f as given, nsteps = 3 (R's `iter`), delta = 0.01 * range(x).
pub fn r_lowess(x: &[f64], y: &[f64], f: f64, nsteps: usize, delta: f64) -> Vec<f64> {
    let n = x.len();
    let mut ys = vec![0.0; n];
    if n < 2 {
        if n == 1 {
            ys[0] = y[0];
        }
        return ys;
    }
    let ns = ((f * n as f64 + 1e-7) as usize).clamp(2, n);
    let mut rw = vec![0.0; n];
    let mut res = vec![0.0; n];

    let mut iter = 1usize;
    while iter <= nsteps + 1 {
        let mut nleft = 0usize; // 0-based window [nleft, nright]
        let mut nright = ns - 1;
        let mut last: i64 = -1; // index of previously estimated point
        let mut i = 0usize; // current point

        loop {
            if nright < n - 1 {
                // move window right if the radius decreases
                let d1 = x[i] - x[nleft];
                let d2 = x[nright + 1] - x[i];
                if d1 > d2 {
                    nleft += 1;
                    nright += 1;
                    continue;
                }
            }
            let ok = lowest(x, y, x[i], &mut ys[i], nleft, nright, &mut res, iter > 1, &rw);
            if !ok {
                ys[i] = y[i];
            }
            // interpolate skipped points
            if last < i as i64 - 1 {
                let lastu = last as usize; // last >= 0 here whenever points were skipped
                let denom = x[i] - x[lastu];
                for j in (last + 1) as usize..i {
                    let alpha = (x[j] - x[lastu]) / denom;
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[lastu];
                }
            }
            last = i as i64;
            let cut = x[last as usize] + delta;
            i = last as usize + 1;
            while i < n {
                if x[i] > cut {
                    break;
                }
                if x[i] == x[last as usize] {
                    ys[i] = ys[last as usize];
                    last = i as i64;
                }
                i += 1;
            }
            i = ((last + 1) as usize).max(i.saturating_sub(1));
            if last as usize >= n - 1 {
                break;
            }
        }

        for k in 0..n {
            res[k] = y[k] - ys[k];
        }
        if iter > nsteps {
            break;
        }
        let mut sc = 0.0;
        for k in 0..n {
            rw[k] = res[k].abs();
            sc += rw[k];
        }
        let m1 = n / 2;
        let mut sorted = rw.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cmad = if n % 2 == 0 {
            let m2 = n - m1 - 1;
            3.0 * (sorted[m1] + sorted[m2])
        } else {
            6.0 * sorted[m1]
        };
        if cmad < 1e-7 * sc {
            // effectively zero: finish with a plain pass already done
            break;
        }
        let c9 = 0.999 * cmad;
        let c1 = 0.001 * cmad;
        for k in 0..n {
            let r = res[k].abs();
            rw[k] = if r <= c1 {
                1.0
            } else if r <= c9 {
                let t = 1.0 - (r / cmad) * (r / cmad);
                t * t
            } else {
                0.0
            };
        }
        iter += 1;
    }
    ys
}

/// Port of R's `lowest()` helper: weighted local linear fit at `xs` over the
/// window [nleft, nright] (extended over right-ties), tricube weights times
/// optional robustness weights. Returns false when all weights are zero.
#[allow(clippy::too_many_arguments)]
fn lowest(
    x: &[f64],
    y: &[f64],
    xs: f64,
    ys: &mut f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    userw: bool,
    rw: &[f64],
) -> bool {
    let n = x.len();
    let range = x[n - 1] - x[0];
    let h = (xs - x[nleft]).max(x[nright] - xs);
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;

    let mut a = 0.0; // sum of weights
    let mut j = nleft;
    while j < n {
        w[j] = 0.0;
        let r = (x[j] - xs).abs();
        if r <= h9 {
            if r <= h1 {
                w[j] = 1.0;
            } else if h > 0.0 {
                let t = 1.0 - (r / h) * (r / h) * (r / h);
                w[j] = t * t * t;
            }
            if userw {
                w[j] *= rw[j];
            }
            a += w[j];
        } else if x[j] > xs {
            break;
        }
        j += 1;
    }
    let nrt = j.saturating_sub(1); // rightmost point (may exceed nright due to ties)
    if a <= 0.0 {
        return false;
    }
    for wj in w.iter_mut().take(nrt + 1).skip(nleft) {
        *wj /= a;
    }
    if h > 0.0 {
        // weighted center of x
        let mut a = 0.0;
        for j in nleft..=nrt {
            a += w[j] * x[j];
        }
        let mut b = xs - a;
        let mut c = 0.0;
        for j in nleft..=nrt {
            c += w[j] * (x[j] - a) * (x[j] - a);
        }
        if c.sqrt() > 0.001 * range {
            b /= c;
            for j in nleft..=nrt {
                w[j] *= b * (x[j] - a) + 1.0;
            }
        }
    }
    *ys = 0.0;
    for j in nleft..=nrt {
        *ys += w[j] * y[j];
    }
    true
}


// ---------------------------------------------------------------------------
// R 4.3.3 dnbinom(mu) via the saddle-point method (nmath dnbinom.c,
// dbinom.c, bd0.c, stirlerr.c) — needed to reproduce fitBeta's deviance and
// the optim objective bit-for-bit.
// ---------------------------------------------------------------------------

const M_LN_2PI: f64 = 1.837_877_066_409_345_483_560_659_472_811;

/// R 4.3.3 `stirlerr(n)`: log(n!) - log(sqrt(2*pi*n)*(n/e)^n).
fn stirlerr433(n: f64) -> f64 {
    const S0: f64 = 0.083333333333333333333;
    const S1: f64 = 0.00277777777777777777778;
    const S2: f64 = 0.00079365079365079365079365;
    const S3: f64 = 0.000595238095238095238095238;
    const S4: f64 = 0.0008417508417508417508417508;
    const SFERR_HALVES: [f64; 31] = [
        0.0,
        0.1534264097200273452913848,
        0.0810614667953272582196702,
        0.0548141210519176538961390,
        0.0413406959554092940938221,
        0.03316287351993628748511048,
        0.02767792568499833914878929,
        0.02374616365629749597132920,
        0.02079067210376509311152277,
        0.01848845053267318523077934,
        0.01664469118982119216319487,
        0.01513497322191737887351255,
        0.01387612882307074799874573,
        0.01281046524292022692424986,
        0.01189670994589177009505572,
        0.01110455975820691732662991,
        0.010411265261972096497478567,
        0.009799416126158803298389475,
        0.009255462182712732917728637,
        0.008768700134139385462952823,
        0.008330563433362871256469318,
        0.007934114564314020547248100,
        0.007573675487951840794972024,
        0.007244554301320383179543912,
        0.006942840107209529865664152,
        0.006665247032707682442354394,
        0.006408994188004207068439631,
        0.006171712263039457647532867,
        0.005951370112758847735624416,
        0.005746216513010115682023589,
        0.005554733551962801371038690,
    ];
    if n <= 15.0 {
        let nn = n + n;
        if nn == (nn as i64) as f64 {
            return SFERR_HALVES[nn as usize];
        }
        return ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
    }
    let nn = n * n;
    if n > 500.0 {
        (S0 - S1 / nn) / n
    } else if n > 80.0 {
        (S0 - (S1 - S2 / nn) / nn) / n
    } else if n > 35.0 {
        (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    } else {
        (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
    }
}

/// R 4.3.3 `bd0(x, np)`: x*log(x/np) + np - x, computed stably.
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        let v2 = v * v;
        for j in 1..1000 {
            ej *= v2;
            let s_ = s;
            s += ej / ((2 * j + 1) as f64);
            if s == s_ {
                return s;
            }
        }
    }
    x * (x / np).ln() + np - x
}

/// R 4.3.3 `dbinom_raw(x, n, p, q, log=TRUE)`.
fn dbinom_raw_log(x: f64, n: f64, p: f64, q: f64) -> f64 {
    if p == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if q == 0.0 {
        return if x == n { 0.0 } else { f64::NEG_INFINITY };
    }
    if x == 0.0 {
        if n == 0.0 {
            return 0.0;
        }
        let lc = if p < 0.1 {
            -bd0(n, n * q) - n * p
        } else {
            n * q.ln()
        };
        return lc;
    }
    if x == n {
        let lc = if q < 0.1 {
            -bd0(n, n * p) - n * q
        } else {
            n * p.ln()
        };
        return lc;
    }
    if x < 0.0 || x > n {
        return f64::NEG_INFINITY;
    }
    let lc = stirlerr433(n) - stirlerr433(x) - stirlerr433(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    let lf = M_LN_2PI + x.ln() + (-x / n).ln_1p();
    lc - 0.5 * lf
}

/// R 4.3.3 `dnbinom_mu(x, size, mu, log=TRUE)` for finite size and integer
/// x >= 0 — matches R's `dnbinom(x, mu=mu, size=size, log=TRUE)` bitwise.
pub fn dnbinom_mu_log(x: f64, size: f64, mu: f64) -> f64 {
    if mu < 0.0 || size < 0.0 {
        return f64::NAN;
    }
    if x < 0.0 || !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 && size == 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return size
            * (if size < mu {
                (size / (size + mu)).ln()
            } else {
                (-mu / (size + mu)).ln_1p()
            });
    }
    if x < 1e-10 * size {
        let p = if size < mu {
            (size / (1.0 + size / mu)).ln()
        } else {
            (mu / (1.0 + mu / size)).ln()
        };
        // lgamma1p(x) = ln_gamma(x + 1) at these (integer, >= 1) x
        return x * p - mu - ln_gamma(x + 1.0) + (x * (x - 1.0) / (2.0 * size)).ln_1p();
    }
    let p = if x < size {
        (-x / (size + x)).ln_1p()
    } else {
        (size / (size + x)).ln()
    };
    let ans = dbinom_raw_log(size, x + size, size / (size + mu), mu / (size + mu));
    p + ans
}

/// R's `dnorm(x, mu, sigma, log=TRUE)` (dnorm4, log branch).
pub fn dnorm_log(x: f64, mu: f64, sigma: f64) -> f64 {
    let x = ((x - mu) / sigma).abs();
    -(M_LN_SQRT_2PI + 0.5 * x * x + sigma.ln())
}

/// R's `sum()` for doubles: extended-precision (long double) accumulation,
/// emulated with double-double summation and rounded once at the end.
pub fn r_sum(values: &[f64]) -> f64 {
    let mut hi = 0.0_f64;
    let mut lo = 0.0_f64;
    for &v in values {
        // two-sum
        let s = hi + v;
        let bb = s - hi;
        let err = (hi - (s - bb)) + (v - bb);
        hi = s;
        lo += err;
    }
    hi + lo
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_ref(path: &str) -> Vec<f64> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().unwrap())
            .collect()
    }

    const REF_DIR: &str = env!("RUST_DESEQ2_REF_DIR");

    #[test]
    fn gamma_family_matches_r() {
        let x = [
            1e-8,
            1e-4,
            0.01,
            0.1,
            0.30000000000000004,
            0.5,
            0.9375,
            1.0,
            1.5,
            2.0,
            2.5,
            3.7,
            5.0,
            8.99,
            9.5,
            10.0,
            10.5,
            12.0,
            47.3,
            100.0,
            1234.5678,
            4934720.0,
            4934721.0,
            1e8,
            1e12,
            1e17,
            2e17,
        ];
        for (name, f, reff) in [
            ("lgamma", ln_gamma as fn(f64) -> f64, "lgamma_ref.txt"),
            ("digamma", digamma as fn(f64) -> f64, "digamma_ref.txt"),
            ("trigamma", trigamma as fn(f64) -> f64, "trigamma_ref.txt"),
        ] {
            let expect = read_ref(&format!("{REF_DIR}/{reff}"));
            for (i, &xi) in x.iter().enumerate() {
                let got = f(xi);
                let e = expect[i];
                let rel = if e == 0.0 {
                    got.abs()
                } else {
                    ((got - e) / e).abs()
                };
                assert!(
                    rel < 4.0 * f64::EPSILON,
                    "{name}({xi}) = {got:.17e}, R = {e:.17e}, rel = {rel:e}"
                );
            }
        }
    }

    #[test]
    fn pnorm_matches_r() {
        let z = [
            0.0, 0.1, 0.5, 0.67448975, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 5.65685424949238, 6.0, 8.0,
            10.0, 15.0, 20.0, 30.0, 37.0, 37.6, 40.0,
        ];
        let expect = read_ref(&format!("{REF_DIR}/pnorm_ref.txt"));
        for (i, &zi) in z.iter().enumerate() {
            let got = pnorm_upper(zi);
            let e = expect[i];
            let rel = if e == 0.0 {
                got.abs()
            } else {
                ((got - e) / e).abs()
            };
            assert!(rel < 1e-14, "pnorm_upper({zi}) = {got:e}, R = {e:e}, rel = {rel:e}");
        }
    }

    #[test]
    fn qf_matches_r() {
        let dfs = [(2.0, 10.0), (3.0, 13.0), (4.0, 4.0), (2.0, 14.0), (5.0, 27.0), (3.0, 1.0)];
        let expect = read_ref(&format!("{REF_DIR}/qf_ref.txt"));
        for (i, &(d1, d2)) in dfs.iter().enumerate() {
            let got = qf(0.99, d1, d2);
            let e = expect[i];
            let rel = ((got - e) / e).abs();
            assert!(rel < 1e-10, "qf(.99,{d1},{d2}) = {got}, R = {e}, rel = {rel:e}");
        }
    }

    #[test]
    fn lowess_matches_r() {
        // same x/y as refvals.R
        let n = 50;
        let x: Vec<f64> = (0..n)
            .map(|k| 0.05 + (0.95 - 0.05) * k as f64 / (n as f64 - 1.0))
            .collect();
        // y values regenerated in R with set.seed(1); read them from the file
        // alongside: the test data file stores y then expected? Simpler: R's
        // sorted rpois values are deterministic; embed them here.
        let y: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 13.0, 13.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 17.0,
            17.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 19.0, 20.0, 21.0, 21.0, 21.0, 21.0, 22.0,
            22.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 25.0, 25.0, 25.0,
            26.0, 27.0, 120.0, 120.0, 120.0, 120.0, 120.0,
        ];
        let expect = read_ref(&format!("{REF_DIR}/lowess_ref.txt"));
        let delta = 0.01 * (x[n - 1] - x[0]);
        let got = r_lowess(&x, &y, 0.2, 3, delta);
        for i in 0..n {
            let d = (got[i] - expect[i]).abs();
            let tol = 1e-9 * expect[i].abs().max(1.0);
            assert!(d < tol, "lowess[{i}] = {}, R = {}, diff = {d:e}", got[i], expect[i]);
        }
    }
}
