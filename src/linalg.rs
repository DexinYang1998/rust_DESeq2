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
        Mat { rows, cols, data: vec![0.0; rows * cols] }
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
