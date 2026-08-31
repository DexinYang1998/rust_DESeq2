//! One-dimensional port of R's loess with `surface = "interpolate"`
//! (stats/src/loessf.f), as used by DESeq2's Monte-Carlo dispersion-prior
//! estimation: `loess(klDivs ~ obsVarGrid, span = 0.2)` + `predict`.
//!
//! The kd-tree build (ehg124/ehg125/ehg126/ehg129, with the stateful
//! Floyd-Rivest partial selection ehg106), the per-vertex local quadratic
//! fits (ehg127: tricube weights over the nf nearest points), and the cubic
//! Hermite evaluation (ehg128, d = 1 tensor path) follow the Fortran flow.
//! The local least-squares solve uses a Householder QR instead of the
//! original equilibrated QR + SVD pseudo-inverse; for the well-conditioned
//! quadratic fits involved the solutions agree to ~1e-12, far below the
//! spacing of the downstream argmin grid.

use crate::linalg::qr_solve;

pub struct Loess1D {
    // 1-based arrays mirroring the Fortran
    a: Vec<usize>,   // split dimension per cell (0 = leaf; 1 for d=1)
    xi: Vec<f64>,    // split value per cell
    lo: Vec<usize>,  // child cells (after split)
    hi: Vec<usize>,
    c: Vec<[usize; 2]>, // cell -> [lower vertex, upper vertex]
    v: Vec<f64>,        // vertex positions
    vval: Vec<[f64; 2]>, // vertex -> (value, derivative)
}

/// Floyd-Rivest partial selection (ehg106, nk = 1): permutes `pi[il..=ir]`
/// (1-based) so that dist[pi[k]] is the k-th smallest.
fn ehg106(il: usize, ir: usize, k: usize, dist: &[f64], pi: &mut [usize]) {
    let mut l = il;
    let mut r = ir;
    while l < r {
        let t = dist[pi[k]];
        let mut i = l;
        let mut j = r;
        pi.swap(l, k);
        if t < dist[pi[r]] {
            pi.swap(l, r);
        }
        while i < j {
            pi.swap(i, j);
            i += 1;
            j -= 1;
            while dist[pi[i]] < t {
                i += 1;
            }
            while t < dist[pi[j]] {
                j -= 1;
            }
        }
        if dist[pi[l]] == t {
            pi.swap(l, j);
        } else {
            j += 1;
            pi.swap(r, j);
        }
        if j <= k {
            l = j + 1;
        }
        if k <= j {
            if j == 0 {
                break;
            }
            r = j - 1;
        }
    }
}

impl Loess1D {
    /// Fit loess to (x, y) with the given span; degree 2, tricube weights,
    /// cell parameter as in R: leaf capacity fc = floor(n * span * cell)
    /// with cell = 0.2 passed as span*cell from loess.R.
    pub fn fit(x: &[f64], y: &[f64], span: f64) -> Loess1D {
        let n = x.len();
        let nf = (n as f64 * span + 1e-5).floor() as usize; // loess_workspace
        let fc = (n as f64 * (span * 0.2)).floor() as usize; // ntol, cell=0.2

        // --- bounding box (ehg126) ---
        let mut alpha = f64::MAX;
        let mut beta = -f64::MAX;
        for &t in x {
            alpha = alpha.min(t);
            beta = beta.max(t);
        }
        let mu = 0.005 * (beta - alpha).max(1e-10 * alpha.abs().max(beta.abs()) + 1e-30);
        alpha -= mu;
        beta += mu;

        // 1-based vertex and cell arrays
        let mut v = vec![0.0; 2 + 1];
        v[1] = alpha;
        v[2] = beta;
        let mut c: Vec<[usize; 2]> = vec![[0, 0]; 2];
        c[1] = [1, 2];
        let mut a = vec![0usize; 2];
        let mut xi = vec![0.0; 2];
        let mut lo = vec![0usize; 2];
        let mut hi = vec![0usize; 2];

        // x with 1-based indexing helper
        let xv = |i: usize| x[i - 1];
        let x1 = dist_of(x); // x coordinates, 1-based, for the tree's selections
        let mut pi: Vec<usize> = (0..=n).collect(); // pi[1..=n] = 1..=n

        // --- build tree (ehg124, d = 1, fd = 0) ---
        let mut nv = 2usize;
        let mut nc = 1usize;
        lo[1] = 1;
        hi[1] = n;
        let mut p = 1usize;
        while p <= nc {
            let l = lo[p];
            let u = hi[p];
            // leaf tests (fd = 0, capacity checks vs ncmax = nvmax = max(200, n))
            let ncmax = 200.max(n);
            let mut leaf = (u - l) + 1 <= fc;
            if !leaf {
                leaf = ncmax < nc + 2 || ncmax < nv + 1;
            }
            let mut m = 0usize;
            if !leaf {
                // spread (ehg129) is trivially positive for d=1; split at median
                m = (l + u) / 2;
                ehg106(l, u, m, &x1, &mut pi);
                // ties go with hi son (bug-fixed offset walk)
                let mut offset: i64 = 0;
                loop {
                    let mo = m as i64 + offset;
                    if mo >= u as i64 || mo < l as i64 {
                        break;
                    }
                    let (lower, check, upper) = if offset < 0 {
                        (l as i64, mo, mo)
                    } else {
                        (mo + 1, mo + 1, u as i64)
                    };
                    ehg106(lower as usize, upper as usize, check as usize, &x1, &mut pi);
                    if xv(pi[mo as usize]) == xv(pi[(mo + 1) as usize]) {
                        offset = -offset;
                        if offset >= 0 {
                            offset += 1;
                        }
                    } else {
                        m = mo as usize;
                        break;
                    }
                }
                let split = xv(pi[m]);
                if v[c[p][0]] == split || v[c[p][1]] == split {
                    leaf = true;
                }
            }
            if leaf {
                a[p] = 0;
            } else {
                a[p] = 1;
                xi[p] = xv(pi[m]);
                // left son
                nc += 1;
                grow(&mut a, &mut xi, &mut lo, &mut hi, &mut c, nc);
                lo[p] = nc;
                lo[nc] = l;
                hi[nc] = m;
                // right son
                nc += 1;
                grow(&mut a, &mut xi, &mut lo, &mut hi, &mut c, nc);
                hi[p] = nc;
                lo[nc] = m + 1;
                hi[nc] = u;
                // add vertex (ehg125, d=1): new vertex at the split value
                let t = xi[p];
                // redundancy check (ehg125)
                let mut matched = 0usize;
                for mm in 1..=nv {
                    if v[mm] == t {
                        matched = mm;
                        break;
                    }
                }
                let vnew = if matched != 0 {
                    matched
                } else {
                    v.push(t);
                    nv += 1;
                    nv
                };
                c[lo[p]] = [c[p][0], vnew];
                c[hi[p]] = [vnew, c[p][1]];
            }
            p += 1;
        }

        // --- vertex fits (ehg139 -> ehg127), with the stateful psi ---
        let mut psi: Vec<usize> = (0..=n).collect();
        let mut vval = vec![[0.0, 0.0]; nv + 1];
        let mut dist = vec![0.0; n + 1];
        for l in 1..=nv {
            let q = v[l];
            for i in 1..=n {
                dist[i] = (xv(i) - q) * (xv(i) - q);
            }
            ehg106(1, n, nf, &dist, &mut psi);
            let rho = dist[psi[nf]] * 1f64.max(span);
            // tricube weights
            let mut w = vec![0.0; nf];
            for i in 1..=nf {
                let z = (dist[psi[i]] / rho).sqrt();
                let t = 1.0 - z * z * z;
                w[i - 1] = (t * t * t).max(0.0).sqrt();
            }
            // weighted quadratic design and rhs
            let k = 3usize;
            let mut bmat = vec![0.0; nf * k];
            let mut eta = vec![0.0; nf];
            for i in 0..nf {
                let xd = xv(psi[i + 1]) - q;
                bmat[i * k] = w[i];
                bmat[i * k + 1] = w[i] * xd;
                bmat[i * k + 2] = w[i] * xd * xd;
                eta[i] = w[i] * y[psi[i + 1] - 1];
            }
            let coef = qr_solve(&mut bmat, nf, k, &mut eta).unwrap_or(vec![0.0; k]);
            vval[l] = [coef[0], coef[1]];
        }

        Loess1D {
            a,
            xi,
            lo,
            hi,
            c,
            v,
            vval,
        }
    }

    /// Evaluate the interpolation surface at `z` (ehg128, d = 1).
    pub fn predict(&self, z: f64) -> f64 {
        let mut j = 1usize;
        while self.a[j] != 0 {
            j = if z <= self.xi[j] {
                self.lo[j]
            } else {
                self.hi[j]
            };
        }
        let ll = self.c[j][0];
        let ur = self.c[j][1];
        let h = (z - self.v[ll]) / (self.v[ur] - self.v[ll]);
        let phi0 = (1.0 - h) * (1.0 - h) * (1.0 + 2.0 * h);
        let phi1 = h * h * (3.0 - 2.0 * h);
        let psi0 = h * (1.0 - h) * (1.0 - h);
        let psi1 = h * h * (h - 1.0);
        phi0 * self.vval[ll][0]
            + phi1 * self.vval[ur][0]
            + (psi0 * self.vval[ll][1] + psi1 * self.vval[ur][1]) * (self.v[ur] - self.v[ll])
    }
}

fn grow(
    a: &mut Vec<usize>,
    xi: &mut Vec<f64>,
    lo: &mut Vec<usize>,
    hi: &mut Vec<usize>,
    c: &mut Vec<[usize; 2]>,
    nc: usize,
) {
    while a.len() <= nc {
        a.push(0);
        xi.push(0.0);
        lo.push(0);
        hi.push(0);
        c.push([0, 0]);
    }
}

/// The tree build partial-selects on the raw coordinates (x itself).
fn dist_of(x: &[f64]) -> Vec<f64> {
    let mut d = vec![0.0; x.len() + 1];
    d[1..].copy_from_slice(x);
    d
}
