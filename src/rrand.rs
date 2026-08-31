//! Exact ports of R 4.3.3's random number machinery, needed to reproduce
//! DESeq2's seeded Monte-Carlo estimate of the dispersion prior variance for
//! designs with residual df <= 3 (`estimateDispersionsPriorVar`):
//!
//! * Mersenne-Twister `unif_rand` with R's `set.seed` scrambling (RNG.c)
//! * `norm_rand` with the INVERSION method and `qnorm` (AS 241, qnorm.c)
//! * `exp_rand` (sexp.c) and `rgamma` (rgamma.c, Ahrens-Dieter GD/GS)
//! * `rchisq(df) = rgamma(df/2, 2)`
//!
//! The streams are bit-identical to R's defaults (Mersenne-Twister +
//! Inversion), which makes the downstream histogram counts exact.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

pub struct RRng {
    mt: [u32; N],
    mti: usize,
    // rgamma's static caches (aa/aaa blocks)
    g_aa: f64,
    g_s: f64,
    g_s2: f64,
    g_d: f64,
    g_aaa: f64,
    g_q0: f64,
    g_b: f64,
    g_si: f64,
    g_c: f64,
}

impl RRng {
    /// R's `set.seed(seed)` with the default Mersenne-Twister kind.
    pub fn new(seed: i32) -> RRng {
        let mut s = seed as u32;
        // initial scrambling
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        // i_seed[0] is mti (overwritten by FixupSeeds), then mt[0..624]
        let mut i_seed = [0u32; N + 1];
        for v in i_seed.iter_mut() {
            s = s.wrapping_mul(69069).wrapping_add(1);
            *v = s;
        }
        let mut mt = [0u32; N];
        mt.copy_from_slice(&i_seed[1..]);
        RRng {
            mt,
            mti: N, // FixupSeeds: I1 = 624 -> full reload on first draw
            g_aa: 0.0,
            g_s: 0.0,
            g_s2: 0.0,
            g_d: 0.0,
            g_aaa: 0.0,
            g_q0: 0.0,
            g_b: 0.0,
            g_si: 0.0,
            g_c: 0.0,
        }
    }

    fn mt_genrand(&mut self) -> f64 {
        let mag01 = [0u32, MATRIX_A];
        if self.mti >= N {
            let mt = &mut self.mt;
            for kk in 0..(N - M) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y as f64 * 2.3283064365386963e-10 // in [0,1)
    }

    /// R's `unif_rand()` (Mersenne-Twister kind, with fixup).
    pub fn unif_rand(&mut self) -> f64 {
        const I2_32M1: f64 = 2.328306437080797e-10; // 1/(2^32 - 1)
        let x = self.mt_genrand();
        if x <= 0.0 {
            return 0.5 * I2_32M1;
        }
        if 1.0 - x <= 0.0 {
            return 1.0 - 0.5 * I2_32M1;
        }
        x
    }

    /// R's `norm_rand()` with the default INVERSION method.
    pub fn norm_rand(&mut self) -> f64 {
        const BIG: f64 = 134217728.0; // 2^27
        let mut u1 = self.unif_rand();
        u1 = (BIG * u1) as i64 as f64 + self.unif_rand();
        qnorm(u1 / BIG)
    }

    /// R's `exp_rand()`.
    pub fn exp_rand(&mut self) -> f64 {
        const Q: [f64; 16] = [
            0.6931471805599453,
            0.9333736875190459,
            0.9888777961838675,
            0.9984959252914960,
            0.9998292811061389,
            0.9999833164100727,
            0.9999985691438767,
            0.9999998906925558,
            0.9999999924734159,
            0.9999999995283275,
            0.9999999999728814,
            0.9999999999985598,
            0.9999999999999289,
            0.9999999999999968,
            0.9999999999999999,
            1.0000000000000000,
        ];
        let mut a = 0.0;
        let mut u = self.unif_rand();
        while u <= 0.0 || u >= 1.0 {
            u = self.unif_rand();
        }
        loop {
            u += u;
            if u > 1.0 {
                break;
            }
            a += Q[0];
        }
        u -= 1.0;
        if u <= Q[0] {
            return a + u;
        }
        let mut i = 0usize;
        let mut ustar = self.unif_rand();
        let mut umin = ustar;
        loop {
            ustar = self.unif_rand();
            if umin > ustar {
                umin = ustar;
            }
            i += 1;
            if u <= Q[i] {
                break;
            }
        }
        a + umin * Q[0]
    }

    /// R's `rgamma(a, scale)` (Ahrens-Dieter GD for a >= 1, GS for a < 1),
    /// including the static caches keyed on `a`.
    pub fn rgamma(&mut self, a: f64, scale: f64) -> f64 {
        const SQRT32: f64 = 5.656854;
        const EXP_M1: f64 = 0.36787944117144232159;
        const Q1: f64 = 0.04166669;
        const Q2: f64 = 0.02083148;
        const Q3: f64 = 0.00801191;
        const Q4: f64 = 0.00144121;
        const Q5: f64 = -7.388e-5;
        const Q6: f64 = 2.4511e-4;
        const Q7: f64 = 2.424e-4;
        const A1: f64 = 0.3333333;
        const A2: f64 = -0.250003;
        const A3: f64 = 0.2000062;
        const A4: f64 = -0.1662921;
        const A5: f64 = 0.1423657;
        const A6: f64 = -0.1367177;
        const A7: f64 = 0.1233795;

        debug_assert!(a > 0.0 && scale > 0.0);
        if a < 1.0 {
            // GS algorithm
            let e = 1.0 + EXP_M1 * a;
            loop {
                let p = e * self.unif_rand();
                if p >= 1.0 {
                    let x = -((e - p) / a).ln();
                    if self.exp_rand() >= (1.0 - a) * x.ln() {
                        return scale * x;
                    }
                } else {
                    let x = (p.ln() / a).exp();
                    if self.exp_rand() >= x {
                        return scale * x;
                    }
                }
            }
        }
        // GD algorithm
        if a != self.g_aa {
            self.g_aa = a;
            self.g_s2 = a - 0.5;
            self.g_s = self.g_s2.sqrt();
            self.g_d = SQRT32 - self.g_s * 12.0;
        }
        let (s, s2, d) = (self.g_s, self.g_s2, self.g_d);
        let mut t = self.norm_rand();
        let x = s + 0.5 * t;
        let ret_val = x * x;
        if t >= 0.0 {
            return scale * ret_val;
        }
        let u = self.unif_rand();
        if d * u <= t * t * t {
            return scale * ret_val;
        }
        if a != self.g_aaa {
            self.g_aaa = a;
            let r = 1.0 / a;
            self.g_q0 =
                ((((((Q7 * r + Q6) * r + Q5) * r + Q4) * r + Q3) * r + Q2) * r + Q1) * r;
            if a <= 3.686 {
                self.g_b = 0.463 + s + 0.178 * s2;
                self.g_si = 1.235;
                self.g_c = 0.195 / s - 0.079 + 0.16 * s;
            } else if a <= 13.022 {
                self.g_b = 1.654 + 0.0076 * s2;
                self.g_si = 1.68 / s + 0.275;
                self.g_c = 0.062 / s + 0.024;
            } else {
                self.g_b = 1.77;
                self.g_si = 0.75;
                self.g_c = 0.1515 / s;
            }
        }
        let (q0, b, si, c) = (self.g_q0, self.g_b, self.g_si, self.g_c);
        let mut q;
        if x > 0.0 {
            let v = t / (s + s);
            if v.abs() <= 0.25 {
                q = q0
                    + 0.5
                        * t
                        * t
                        * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1)
                        * v;
            } else {
                q = q0 - s * t + 0.25 * t * t + (s2 + s2) * (1.0 + v).ln();
            }
            if (1.0 - u).ln() <= q {
                return scale * ret_val;
            }
        }
        loop {
            let e = self.exp_rand();
            let mut u = self.unif_rand();
            u = u + u - 1.0;
            t = if u < 0.0 { b - si * e } else { b + si * e };
            if t >= -0.71874483771719 {
                let v = t / (s + s);
                if v.abs() <= 0.25 {
                    q = q0
                        + 0.5
                            * t
                            * t
                            * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v
                                + A1)
                            * v;
                } else {
                    q = q0 - s * t + 0.25 * t * t + (s2 + s2) * (1.0 + v).ln();
                }
                if q > 0.0 {
                    let w = q.exp_m1();
                    if c * u.abs() <= w * (e - 0.5 * t * t).exp() {
                        break;
                    }
                }
            }
        }
        let x = s + 0.5 * t;
        scale * x * x
    }

    /// R's `rchisq(df)`.
    pub fn rchisq(&mut self, df: f64) -> f64 {
        self.rgamma(df / 2.0, 2.0)
    }

    /// R's `rnorm(mu, sd)`.
    pub fn rnorm(&mut self, mu: f64, sd: f64) -> f64 {
        mu + sd * self.norm_rand()
    }
}

/// R's `qnorm(p, 0, 1, lower.tail=TRUE, log.p=FALSE)`: AS 241 (Wichura),
/// port of nmath qnorm.c.
pub fn qnorm(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    let val;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        val = q
            * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
        return val;
    }
    let lp = if q > 0.0 { (1.0 - p).ln() } else { p.ln() };
    let mut r = (-lp).sqrt();
    let mut v;
    if r <= 5.0 {
        r -= 1.6;
        v = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                + 0.0151986665636164571966)
                * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0);
    } else if r <= 27.0 {
        r -= 5.0;
        v = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                + 1.8463183175100546818e-5)
                * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0);
    } else {
        // unreachable for norm_rand's input range; crude far-tail form
        v = r * std::f64::consts::SQRT_2;
    }
    if q < 0.0 {
        v = -v;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_r() {
        let dir = env!("RUST_DESEQ2_REF_DIR");
        let expect: Vec<f64> = std::fs::read_to_string(format!("{dir}/rng_ref.txt"))
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().unwrap())
            .collect();
        // set.seed(2); runif(5), rchisq(5,2), rnorm(5), rchisq(3,1), rchisq(3,3), rnorm(3,0,2)
        let mut r = RRng::new(2);
        let mut got = Vec::new();
        for _ in 0..5 {
            got.push(r.unif_rand());
        }
        for _ in 0..5 {
            got.push(r.rchisq(2.0));
        }
        for _ in 0..5 {
            got.push(r.rnorm(0.0, 1.0));
        }
        for _ in 0..3 {
            got.push(r.rchisq(1.0));
        }
        for _ in 0..3 {
            got.push(r.rchisq(3.0));
        }
        for _ in 0..3 {
            got.push(r.rnorm(0.0, 2.0));
        }
        assert_eq!(got.len(), expect.len());
        for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
            assert!(
                g == e || (g - e).abs() < 1e-15 * e.abs(),
                "draw {i}: rust {g:.17e} vs R {e:.17e}"
            );
        }
    }
}
