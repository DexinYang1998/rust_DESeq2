//! Minimal dense linear algebra for small (p x p) systems arising in the GLM
//! fit. `p` is the number of model coefficients (typically 2-5), so a plain
//! Gauss-Jordan routine is perfectly adequate.

/// A small dense, row-major matrix of f64.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    #[inline]
    pub fn add(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] += v;
    }
}

/// Invert a square matrix in place via Gauss-Jordan elimination with partial
/// pivoting. Returns `None` if the matrix is singular (pivot ~ 0), which the
/// caller treats as a non-convergent gene.
pub fn invert(a: &Mat) -> Option<Mat> {
    let n = a.rows;
    debug_assert_eq!(a.rows, a.cols);
    // Augmented [A | I].
    let mut m = vec![0.0f64; n * 2 * n];
    let w = 2 * n;
    for i in 0..n {
        for j in 0..n {
            m[i * w + j] = a.get(i, j);
        }
        m[i * w + n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: find the row with the largest absolute value.
        let mut piv = col;
        let mut best = m[col * w + col].abs();
        for r in (col + 1)..n {
            let v = m[r * w + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            for j in 0..w {
                m.swap(col * w + j, piv * w + j);
            }
        }
        // Normalise pivot row.
        let d = m[col * w + col];
        for j in 0..w {
            m[col * w + j] /= d;
        }
        // Eliminate other rows.
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * w + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..w {
                m[r * w + j] -= f * m[col * w + j];
            }
        }
    }

    let mut inv = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv.set(i, j, m[i * w + n + j]);
        }
    }
    Some(inv)
}

/// Least-squares solve of `A beta ~= b` via Householder QR without pivoting
/// (the factorization used by `fitBeta`'s `useQR=TRUE` path through
/// Armadillo/LAPACK). `a` is row-major m x p (consumed), `b` has length m.
/// Returns None when a diagonal of R is numerically zero.
pub fn qr_solve(a: &mut [f64], m: usize, p: usize, b: &mut [f64]) -> Option<Vec<f64>> {
    debug_assert!(m >= p);
    for k in 0..p {
        // Householder vector for column k, rows k..m
        let mut norm = 0.0;
        for i in k..m {
            let v = a[i * p + k];
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            return None;
        }
        let akk = a[k * p + k];
        let alpha = if akk >= 0.0 { -norm } else { norm };
        // v = x - alpha e1, normalized so v[0] = 1
        let v0 = akk - alpha;
        if v0 == 0.0 {
            // x is already alpha*e1; nothing to reflect
            a[k * p + k] = alpha;
            continue;
        }
        // tau = -v0 / alpha (LAPACK's beta = 2/(v'v) with v0-normalized v)
        // apply H = I - tau v v' with v = (1, x[k+1..]/v0)
        let tau = -v0 / alpha;
        let inv_v0 = 1.0 / v0;
        // update remaining columns
        for j in (k + 1)..p {
            let mut s = a[k * p + j];
            for i in (k + 1)..m {
                s += a[i * p + k] * inv_v0 * a[i * p + j];
            }
            s *= tau;
            a[k * p + j] -= s;
            for i in (k + 1)..m {
                a[i * p + j] -= a[i * p + k] * inv_v0 * s;
            }
        }
        // update b
        let mut s = b[k];
        for i in (k + 1)..m {
            s += a[i * p + k] * inv_v0 * b[i];
        }
        s *= tau;
        b[k] -= s;
        for i in (k + 1)..m {
            b[i] -= a[i * p + k] * inv_v0 * s;
        }
        a[k * p + k] = alpha;
    }
    // back substitution on the p x p upper triangle
    let mut beta = vec![0.0; p];
    for k in (0..p).rev() {
        let d = a[k * p + k];
        if d.abs() < 1e-300 {
            return None;
        }
        let mut s = b[k];
        for j in (k + 1)..p {
            s -= a[k * p + j] * beta[j];
        }
        beta[k] = s / d;
    }
    Some(beta)
}

/// Natural log of |det(A)| for a small square matrix, via Gaussian elimination
/// with partial pivoting. Returns `f64::NEG_INFINITY` for a singular matrix.
/// Used for the Cox-Reid adjustment term `0.5 * log det(X' W X)`.
pub fn log_abs_det(a: &Mat) -> f64 {
    let n = a.rows;
    let mut m = a.data.clone();
    let mut logdet = 0.0;
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return f64::NEG_INFINITY;
        }
        if piv != col {
            for j in 0..n {
                m.swap(col * n + j, piv * n + j);
            }
        }
        let d = m[col * n + col];
        logdet += d.abs().ln();
        for r in (col + 1)..n {
            let f = m[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                m[r * n + j] -= f * m[col * n + j];
            }
        }
    }
    logdet
}
