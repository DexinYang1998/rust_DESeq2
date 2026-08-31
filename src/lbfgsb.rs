//! Faithful port of R 4.3.3's L-BFGS-B implementation (src/appl/lbfgsb.c,
//! itself the Byrd-Lu-Nocedal-Zhu code translated to C) together with the
//! `optim()` driver loop and its bounds-aware numerical gradient
//! (src/library/stats/src/optim.c, `fmingr` with `ndeps`). DESeq2 falls back
//! to `optim(method="L-BFGS-B")` for genes whose IRLS does not converge, so
//! reproducing those genes exactly requires reproducing this optimizer's
//! path, not just its limit.
//!
//! Translation notes: arrays keep the original 1-based Fortran indexing
//! (element 0 unused) and matrices the original column-major layout, so the
//! code can be compared line by line with the C source. BLAS/LINPACK helpers
//! (ddot/dpofa/dtrsl) are the reference implementations, whose accumulation
//! order the original relies on.

const FTOL: f64 = 0.001;
const GTOL: f64 = 0.9;
const XTOL: f64 = 0.1;
const STPMIN: f64 = 0.0;

#[inline]
fn ix(i: usize, j: usize, dim1: usize) -> usize {
    (i - 1) + (j - 1) * dim1
}

/// Reference-BLAS ddot for unit strides (sequential accumulation).
fn ddot(n: usize, x: &[f64], sx: usize, y: &[f64], sy: usize) -> f64 {
    let mut t = 0.0;
    for i in 0..n {
        t += x[sx + i] * y[sy + i];
    }
    t
}

/// LINPACK dpofa: Cholesky of the leading n x n block (upper triangle),
/// stored at `base` with leading dimension `lda`. Returns 0 or the failing
/// column.
fn dpofa(a: &mut [f64], base: usize, lda: usize, n: usize) -> i32 {
    for j in 1..=n {
        let mut s = 0.0;
        for k in 1..(j) {
            let dot = ddot(k - 1, a, base + ix(1, k, lda), a, base + ix(1, j, lda));
            let mut t = a[base + ix(k, j, lda)] - dot;
            t /= a[base + ix(k, k, lda)];
            a[base + ix(k, j, lda)] = t;
            s += t * t;
        }
        let s = a[base + ix(j, j, lda)] - s;
        if s <= 0.0 {
            return j as i32;
        }
        a[base + ix(j, j, lda)] = s.sqrt();
    }
    0
}

/// LINPACK dtrsl for the two jobs used here, operating within one buffer:
/// the triangle starts at `t_base` (leading dim `ldt`), the right-hand side
/// at `b_base` (contiguous). job 01: solve T x = b, T upper. job 11: solve
/// T' x = b, T upper. Returns 0, or j when T[j][j] == 0.
fn dtrsl(buf: &mut [f64], t_base: usize, ldt: usize, n: usize, b_base: usize, job: i32) -> i32 {
    for j in 1..=n {
        if buf[t_base + ix(j, j, ldt)] == 0.0 {
            return j as i32;
        }
    }
    match job {
        1 => {
            // solve T*x=b, T upper
            buf[b_base + n - 1] /= buf[t_base + ix(n, n, ldt)];
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -buf[b_base + j]; // b[j+1]
                // daxpy(j, temp, t(1, j+1), b(1))
                for i in 0..j {
                    buf[b_base + i] += temp * buf[t_base + ix(1, j + 1, ldt) + i];
                }
                buf[b_base + j - 1] /= buf[t_base + ix(j, j, ldt)];
            }
        }
        11 => {
            // solve T'*x=b, T upper
            buf[b_base] /= buf[t_base + ix(1, 1, ldt)];
            for j in 2..=n {
                let dot = ddot(j - 1, buf, t_base + ix(1, j, ldt), buf, b_base);
                buf[b_base + j - 1] = (buf[b_base + j - 1] - dot) / buf[t_base + ix(j, j, ldt)];
            }
        }
        _ => unreachable!(),
    }
    0
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[allow(dead_code)]
enum Task {
    Start,
    FgStart,
    FgLnsrch,
    NewX,
    Convergence,
    Warning,
    Error,
}

#[derive(Clone, Copy, PartialEq)]
enum DcTask {
    Start,
    Fg,
    Conv,
    Warn,
    Error,
}

#[derive(Default)]
struct Dcsrch {
    stage: i32,
    brackt: bool,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

struct State {
    n: usize,
    m: usize,
    // persistent scalars (the C statics)
    prjctd: bool,
    cnstnd: bool,
    boxed: bool,
    updatd: bool,
    iback: i32,
    nskip: i32,
    head: usize,
    col: usize,
    itail: usize,
    iter: i32,
    iupdat: i32,
    nint: i32,
    nintol: i32,
    nfgv: i32,
    info: i32,
    ifun: i32,
    iword: i32,
    nfree: usize,
    ileave: usize,
    nenter: usize,
    theta: f64,
    fold: f64,
    tol: f64,
    dnorm: f64,
    epsmch: f64,
    gd: f64,
    stpmx: f64,
    sbgnrm: f64,
    stp: f64,
    gdold: f64,
    dtd: f64,
    // workspaces (1-based)
    ws: Vec<f64>,   // n x m
    wy: Vec<f64>,   // n x m
    sy: Vec<f64>,   // m x m
    ss: Vec<f64>,   // m x m
    wt: Vec<f64>,   // m x m
    wn: Vec<f64>,   // 2m x 2m
    snd: Vec<f64>,  // 2m x 2m
    z: Vec<f64>,    // n (1-based)
    r: Vec<f64>,
    d: Vec<f64>,
    t: Vec<f64>,
    wa: Vec<f64>, // 8m (1-based)
    indx: Vec<usize>,
    iwhere: Vec<i32>,
    indx2: Vec<usize>,
    dcs: Dcsrch,
    csave: DcTask,
    wrk_flag: bool,
}

fn active(n: usize, l: &[f64], u: &[f64], nbd: &[i32], x: &mut [f64], st: &mut State) {
    st.prjctd = false;
    st.cnstnd = false;
    st.boxed = true;
    for i in 1..=n {
        if nbd[i] > 0 {
            if nbd[i] <= 2 && x[i] <= l[i] {
                if x[i] < l[i] {
                    st.prjctd = true;
                    x[i] = l[i];
                }
            } else if nbd[i] >= 2 && x[i] >= u[i] && x[i] > u[i] {
                st.prjctd = true;
                x[i] = u[i];
            }
        }
    }
    for i in 1..=n {
        if nbd[i] != 2 {
            st.boxed = false;
        }
        if nbd[i] == 0 {
            st.iwhere[i] = -1;
        } else {
            st.cnstnd = true;
            if nbd[i] == 2 && u[i] - l[i] <= 0.0 {
                st.iwhere[i] = 3;
            } else {
                st.iwhere[i] = 0;
            }
        }
    }
}

fn projgr(n: usize, l: &[f64], u: &[f64], nbd: &[i32], x: &[f64], g: &[f64]) -> f64 {
    let mut sbgnrm = 0.0_f64;
    for i in 1..=n {
        let mut gi = g[i];
        if nbd[i] != 0 {
            if gi < 0.0 {
                if nbd[i] >= 2 && gi < x[i] - u[i] {
                    gi = x[i] - u[i];
                }
            } else if nbd[i] <= 2 && gi > x[i] - l[i] {
                gi = x[i] - l[i];
            }
        }
        if sbgnrm < gi.abs() {
            sbgnrm = gi.abs();
        }
    }
    sbgnrm
}

/// dtrsl where the triangle and rhs live in different buffers; rhs starts at
/// b[b1] (1-based length n).
fn dtrsl_sep(t: &[f64], ldt: usize, n: usize, b: &mut [f64], b1: usize, job: i32) -> i32 {
    for j in 1..=n {
        if t[ix(j, j, ldt)] == 0.0 {
            return j as i32;
        }
    }
    match job {
        1 => {
            b[b1 + n - 1] /= t[ix(n, n, ldt)];
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -b[b1 + j];
                for i in 0..j {
                    b[b1 + i] += temp * t[ix(1, j + 1, ldt) + i];
                }
                b[b1 + j - 1] /= t[ix(j, j, ldt)];
            }
        }
        11 => {
            b[b1] /= t[ix(1, 1, ldt)];
            for j in 2..=n {
                let mut dot = 0.0;
                for k in 0..(j - 1) {
                    dot += t[ix(1, j, ldt) + k] * b[b1 + k];
                }
                b[b1 + j - 1] = (b[b1 + j - 1] - dot) / t[ix(j, j, ldt)];
            }
        }
        _ => unreachable!(),
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn cauchy(
    st: &mut State,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
) -> i32 {
    let n = st.n;
    let m = st.m;
    let col = st.col;
    let head = st.head;
    let theta = st.theta;
    let epsmch = st.epsmch;
    // wa split: p [1..2m], c [2m+1..4m], wbp [4m+1..6m], v [6m+1..8m]
    // handled through explicit offsets into st.wa
    let (p_off, c_off, wbp_off, v_off) = (0usize, 2 * m, 4 * m, 6 * m);

    if st.sbgnrm <= 0.0 {
        for i in 1..=n {
            st.z[i] = x[i];
        }
        return 0;
    }
    let mut bnded = true;
    let mut nfree = n + 1;
    let mut nbreak = 0usize;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.0;
    let col2 = 2 * col;
    let mut f1 = 0.0;
    for i in 1..=col2 {
        st.wa[p_off + i] = 0.0;
    }
    let mut tl = 0.0;
    let mut tu = 0.0;
    for i in 1..=n {
        let neggi = -g[i];
        if st.iwhere[i] != 3 && st.iwhere[i] != -1 {
            if nbd[i] <= 2 {
                tl = x[i] - l[i];
            }
            if nbd[i] >= 2 {
                tu = u[i] - x[i];
            }
            let xlower = nbd[i] <= 2 && tl <= 0.0;
            let xupper = nbd[i] >= 2 && tu <= 0.0;
            st.iwhere[i] = 0;
            if xlower {
                if neggi <= 0.0 {
                    st.iwhere[i] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    st.iwhere[i] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                st.iwhere[i] = -3;
            }
        }
        let mut pointr = head;
        if st.iwhere[i] != 0 && st.iwhere[i] != -1 {
            st.d[i] = 0.0;
        } else {
            st.d[i] = neggi;
            f1 -= neggi * neggi;
            for j in 1..=col {
                st.wa[p_off + j] += st.wy[ix(i, pointr, n)] * neggi;
                st.wa[p_off + col + j] += st.ws[ix(i, pointr, n)] * neggi;
                pointr = pointr % m + 1;
            }
            if nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.0 {
                nbreak += 1;
                st.indx2[nbreak] = i; // iorder
                st.t[nbreak] = tl / (-neggi);
                if nbreak == 1 || st.t[nbreak] < bkmin {
                    bkmin = st.t[nbreak];
                    ibkmin = nbreak;
                }
            } else if nbd[i] >= 2 && neggi > 0.0 {
                nbreak += 1;
                st.indx2[nbreak] = i;
                st.t[nbreak] = tu / neggi;
                if nbreak == 1 || st.t[nbreak] < bkmin {
                    bkmin = st.t[nbreak];
                    ibkmin = nbreak;
                }
            } else {
                nfree -= 1;
                st.indx2[nfree] = i;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }
    if theta != 1.0 {
        // dscal(col, theta, p[col+1..])
        for j in 1..=col {
            st.wa[p_off + col + j] *= theta;
        }
    }
    for i in 1..=n {
        st.z[i] = x[i];
    }
    if nbreak == 0 && nfree == n + 1 {
        return 0;
    }
    for j in 1..=col2 {
        st.wa[c_off + j] = 0.0;
    }
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        // bmv(v = p-slice, p = v-slice): copy the input, write in place
        let vin: Vec<f64> = st.wa[p_off..p_off + 2 * m + 1].to_vec();
        let info = bmv_split(m, &st.sy, &st.wt, col, &vin, 0, &mut st.wa, v_off);
        if info != 0 {
            return info;
        }
        f2 -= ddot(col2, &st.wa, v_off + 1, &st.wa, p_off + 1);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    st.nint = 1;
    if nbreak != 0 {
        let mut nleft = nbreak;
        let mut iter = 1usize;
        let mut tj = 0.0;
        loop {
            // L777
            let tj0 = tj;
            let ibp;
            if iter == 1 {
                tj = bkmin;
                ibp = st.indx2[ibkmin];
            } else {
                if iter == 2 {
                    if ibkmin != nbreak {
                        st.t[ibkmin] = st.t[nbreak];
                        st.indx2[ibkmin] = st.indx2[nbreak];
                    }
                }
                hpsolb(nleft, &mut st.t, &mut st.indx2, iter as i32 - 2);
                tj = st.t[nleft];
                ibp = st.indx2[nleft];
            }
            let dt = tj - tj0;
            if dtm < dt {
                break; // L888
            }
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = st.d[ibp];
            st.d[ibp] = 0.0;
            let zibp;
            if dibp > 0.0 {
                zibp = u[ibp] - x[ibp];
                st.z[ibp] = u[ibp];
                st.iwhere[ibp] = 2;
            } else {
                zibp = l[ibp] - x[ibp];
                st.z[ibp] = l[ibp];
                st.iwhere[ibp] = 1;
            }
            if nleft == 0 && nbreak == n {
                #[allow(unused_assignments)]
                {
                    dtm = dt;
                }
                // L999
                if col > 0 {
                    for j in 1..=col2 {
                        st.wa[c_off + j] += dtm * st.wa[p_off + j];
                    }
                }
                return 0;
            }
            st.nint += 1;
            let dibp2 = dibp * dibp;
            f1 += dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if col > 0 {
                for j in 1..=col2 {
                    st.wa[c_off + j] += dt * st.wa[p_off + j];
                }
                let mut pointr = head;
                for j in 1..=col {
                    st.wa[wbp_off + j] = st.wy[ix(ibp, pointr, n)];
                    st.wa[wbp_off + col + j] = theta * st.ws[ix(ibp, pointr, n)];
                    pointr = pointr % m + 1;
                }
                let vin: Vec<f64> = st.wa[wbp_off..wbp_off + 2 * m + 1].to_vec();
                let info = bmv_split(m, &st.sy, &st.wt, col, &vin, 0, &mut st.wa, v_off);
                if info != 0 {
                    return info;
                }
                let wmc = ddot(col2, &st.wa, c_off + 1, &st.wa, v_off + 1);
                let wmp = ddot(col2, &st.wa, p_off + 1, &st.wa, v_off + 1);
                let wmw = ddot(col2, &st.wa, wbp_off + 1, &st.wa, v_off + 1);
                for j in 1..=col2 {
                    st.wa[p_off + j] += -dibp * st.wa[wbp_off + j];
                }
                f1 += dibp * wmc;
                f2 += 2.0 * dibp * wmp - dibp2 * wmw;
            }
            let floor2 = epsmch * f2_org;
            if f2 < floor2 {
                f2 = floor2;
            }
            if nleft > 0 {
                dtm = -f1 / f2;
                continue;
            } else if bnded {
                f1 = 0.0;
                f2 = 0.0;
                dtm = 0.0;
            } else {
                dtm = -f1 / f2;
            }
            break;
        }
    }
    // L888
    if dtm <= 0.0 {
        dtm = 0.0;
    }
    tsum += dtm;
    for i in 1..=n {
        st.z[i] += tsum * st.d[i];
    }
    // L999
    if col > 0 {
        for j in 1..=col2 {
            st.wa[c_off + j] += dtm * st.wa[p_off + j];
        }
    }
    0
}

/// bmv where v and p come from two mutable splits of `wa`.
#[allow(clippy::too_many_arguments)]
fn bmv_split(
    m: usize,
    sy: &[f64],
    wt: &[f64],
    col: usize,
    v_buf: &[f64],
    v_off: usize,
    p_buf: &mut [f64],
    p_off: usize,
) -> i32 {
    if col == 0 {
        return 0;
    }
    p_buf[p_off + col + 1] = v_buf[v_off + col + 1];
    for i in 2..=col {
        let i2 = col + i;
        let mut sum = 0.0;
        for k in 1..=(i - 1) {
            sum += sy[ix(i, k, m)] * v_buf[v_off + k] / sy[ix(k, k, m)];
        }
        p_buf[p_off + i2] = v_buf[v_off + i2] + sum;
    }
    let info = dtrsl_sep(wt, m, col, p_buf, p_off + col + 1, 11);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        p_buf[p_off + i] = v_buf[v_off + i] / sy[ix(i, i, m)].sqrt();
    }
    let info = dtrsl_sep(wt, m, col, p_buf, p_off + col + 1, 1);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        p_buf[p_off + i] = -p_buf[p_off + i] / sy[ix(i, i, m)].sqrt();
    }
    for i in 1..=col {
        let mut sum = 0.0;
        for k in (i + 1)..=col {
            sum += sy[ix(k, i, m)] * p_buf[p_off + col + k] / sy[ix(i, i, m)];
        }
        p_buf[p_off + i] += sum;
    }
    0
}

fn hpsolb(n: usize, t: &mut [f64], iorder: &mut [usize], iheap: i32) {
    if iheap == 0 {
        for k in 2..=n {
            let ddum = t[k];
            let indxin = iorder[k];
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j] {
                    t[i] = t[j];
                    iorder[i] = iorder[j];
                    i = j;
                } else {
                    break;
                }
            }
            t[i] = ddum;
            iorder[i] = indxin;
        }
    }
    if n > 1 {
        let mut i = 1;
        let out = t[1];
        let indxou = iorder[1];
        let ddum = t[n];
        let indxin = iorder[n];
        loop {
            let mut j = i + i;
            if j <= n - 1 {
                if t[j + 1] < t[j] {
                    j += 1;
                }
                if t[j] < ddum {
                    t[i] = t[j];
                    iorder[i] = iorder[j];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i] = ddum;
        iorder[i] = indxin;
        t[n] = out;
        iorder[n] = indxou;
    }
}

fn freev(st: &mut State) -> bool {
    let n = st.n;
    st.nenter = 0;
    st.ileave = n + 1;
    if st.iter > 0 && st.cnstnd {
        for i in 1..=st.nfree {
            let k = st.indx[i];
            if st.iwhere[k] > 0 {
                st.ileave -= 1;
                st.indx2[st.ileave] = k;
            }
        }
        for i in (st.nfree + 1)..=n {
            let k = st.indx[i];
            if st.iwhere[k] <= 0 {
                st.nenter += 1;
                st.indx2[st.nenter] = k;
            }
        }
    }
    let wrk = st.ileave < n + 1 || st.nenter > 0 || st.updatd;
    st.nfree = 0;
    let mut iact = n + 1;
    for i in 1..=n {
        if st.iwhere[i] <= 0 {
            st.nfree += 1;
            st.indx[st.nfree] = i;
        } else {
            iact -= 1;
            st.indx[iact] = i;
        }
    }
    wrk
}

fn formk(st: &mut State) -> i32 {
    let n = st.n;
    let m = st.m;
    let m2 = 2 * m;
    let col = st.col;
    let nsub = st.nfree;
    let upcl;
    if st.updatd {
        if st.iupdat as usize > m {
            // shift old part of wn1 (snd)
            for jy in 1..=(m - 1) {
                let js = m + jy;
                let i2 = m - jy;
                dcopy_within(&mut st.snd, i2, ix(jy + 1, jy + 1, m2), ix(jy, jy, m2));
                dcopy_within(&mut st.snd, i2, ix(js + 1, js + 1, m2), ix(js, js, m2));
                dcopy_within(&mut st.snd, m - 1, ix(m + 2, jy + 1, m2), ix(m + 1, jy, m2));
            }
        }
        let pbegin = 1usize;
        let pend = nsub;
        let dbegin = nsub + 1;
        let dend = n;
        let iy = col;
        let is = m + col;
        let mut ipntr = st.head + col - 1;
        if ipntr > m {
            ipntr -= m;
        }
        let mut jpntr = st.head;
        for jy in 1..=col {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            for k in pbegin..=pend {
                let k1 = st.indx[k];
                temp1 += st.wy[ix(k1, ipntr, n)] * st.wy[ix(k1, jpntr, n)];
            }
            for k in dbegin..=dend {
                let k1 = st.indx[k];
                temp2 += st.ws[ix(k1, ipntr, n)] * st.ws[ix(k1, jpntr, n)];
                temp3 += st.ws[ix(k1, ipntr, n)] * st.wy[ix(k1, jpntr, n)];
            }
            st.snd[ix(iy, jy, m2)] = temp1;
            st.snd[ix(is, js, m2)] = temp2;
            st.snd[ix(is, jy, m2)] = temp3;
            jpntr = jpntr % m + 1;
        }
        let jy = col;
        let mut jpntr2 = st.head + col - 1;
        if jpntr2 > m {
            jpntr2 -= m;
        }
        let mut ipntr2 = st.head;
        for i in 1..=col {
            let is = m + i;
            let mut temp3 = 0.0;
            for k in pbegin..=pend {
                let k1 = st.indx[k];
                temp3 += st.ws[ix(k1, ipntr2, n)] * st.wy[ix(k1, jpntr2, n)];
            }
            ipntr2 = ipntr2 % m + 1;
            st.snd[ix(is, jy, m2)] = temp3;
        }
        upcl = col - 1;
    } else {
        upcl = col;
    }

    let mut ipntr = st.head;
    for iy in 1..=upcl {
        let is = m + iy;
        let mut jpntr = st.head;
        for jy in 1..=iy {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            let mut temp4 = 0.0;
            for k in 1..=st.nenter {
                let k1 = st.indx2[k];
                temp1 += st.wy[ix(k1, ipntr, n)] * st.wy[ix(k1, jpntr, n)];
                temp2 += st.ws[ix(k1, ipntr, n)] * st.ws[ix(k1, jpntr, n)];
            }
            for k in st.ileave..=n {
                let k1 = st.indx2[k];
                temp3 += st.wy[ix(k1, ipntr, n)] * st.wy[ix(k1, jpntr, n)];
                temp4 += st.ws[ix(k1, ipntr, n)] * st.ws[ix(k1, jpntr, n)];
            }
            st.snd[ix(iy, jy, m2)] = st.snd[ix(iy, jy, m2)] + temp1 - temp3;
            st.snd[ix(is, js, m2)] = st.snd[ix(is, js, m2)] - temp2 + temp4;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    let mut ipntr = st.head;
    for is in (m + 1)..=(m + upcl) {
        let mut jpntr = st.head;
        for jy in 1..=upcl {
            let mut temp1 = 0.0;
            let mut temp3 = 0.0;
            for k in 1..=st.nenter {
                let k1 = st.indx2[k];
                temp1 += st.ws[ix(k1, ipntr, n)] * st.wy[ix(k1, jpntr, n)];
            }
            for k in st.ileave..=n {
                let k1 = st.indx2[k];
                temp3 += st.ws[ix(k1, ipntr, n)] * st.wy[ix(k1, jpntr, n)];
            }
            if is <= jy + m {
                st.snd[ix(is, jy, m2)] += temp1 - temp3;
            } else {
                st.snd[ix(is, jy, m2)] += -temp1 + temp3;
            }
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    // form wn
    for iy in 1..=col {
        let is = col + iy;
        let is1 = m + iy;
        for jy in 1..=iy {
            let js = col + jy;
            let js1 = m + jy;
            st.wn[ix(jy, iy, m2)] = st.snd[ix(iy, jy, m2)] / st.theta;
            st.wn[ix(js, is, m2)] = st.snd[ix(is1, js1, m2)] * st.theta;
        }
        for jy in 1..=(iy - 1) {
            st.wn[ix(jy, is, m2)] = -st.snd[ix(is1, jy, m2)];
        }
        for jy in iy..=col {
            st.wn[ix(jy, is, m2)] = st.snd[ix(is1, jy, m2)];
        }
        st.wn[ix(iy, iy, m2)] += st.sy[ix(iy, iy, m)];
    }
    let info = dpofa(&mut st.wn, 0, m2, col);
    if info != 0 {
        return -1;
    }
    let col2 = 2 * col;
    for js in (col + 1)..=col2 {
        // dtrsl on wn's own column js against its leading col x col triangle
        let info = dtrsl(&mut st.wn, 0, m2, col, ix(1, js, m2), 11);
        if info != 0 {
            return -1;
        }
    }
    for is in (col + 1)..=col2 {
        for js in is..=col2 {
            st.wn[ix(is, js, m2)] += ddot(col, &st.wn, ix(1, is, m2), &st.wn, ix(1, js, m2));
        }
    }
    let info = dpofa(&mut st.wn, ix(col + 1, col + 1, m2), m2, col);
    if info != 0 {
        return -2;
    }
    0
}

fn dcopy_within(buf: &mut [f64], n: usize, src: usize, dst: usize) {
    for i in 0..n {
        buf[dst + i] = buf[src + i];
    }
}

fn formt(st: &mut State) -> i32 {
    let m = st.m;
    let col = st.col;
    for j in 1..=col {
        st.wt[ix(1, j, m)] = st.theta * st.ss[ix(1, j, m)];
    }
    for i in 2..=col {
        for j in i..=col {
            let k1 = i.min(j) - 1;
            let mut ddum = 0.0;
            for k in 1..=k1 {
                ddum += st.sy[ix(i, k, m)] * st.sy[ix(j, k, m)] / st.sy[ix(k, k, m)];
            }
            st.wt[ix(i, j, m)] = ddum + st.theta * st.ss[ix(i, j, m)];
        }
    }
    let info = dpofa(&mut st.wt, 0, m, col);
    if info != 0 {
        return -3;
    }
    0
}

fn cmprlb(st: &mut State, x: &[f64], g: &[f64]) -> i32 {
    let n = st.n;
    let m = st.m;
    let col = st.col;
    if !st.cnstnd && col > 0 {
        for i in 1..=n {
            st.r[i] = -g[i];
        }
    } else {
        let nf = st.nfree;
        for i in 1..=nf {
            let k = st.indx[i];
            st.r[i] = -st.theta * (st.z[k] - x[k]) - g[k];
        }
        // bmv with v = wa[2m+1..4m], p = wa[1..2m]
        let vin: Vec<f64> = st.wa[2 * m..4 * m + 1].to_vec();
        let info = bmv_split(m, &st.sy, &st.wt, col, &vin, 0, &mut st.wa, 0);
        if info != 0 {
            return -8;
        }
        let mut pointr = st.head;
        for j in 1..=col {
            let a1 = st.wa[j];
            let a2 = st.theta * st.wa[col + j];
            for i in 1..=nf {
                let k = st.indx[i];
                st.r[i] += st.wy[ix(k, pointr, n)] * a1 + st.ws[ix(k, pointr, n)] * a2;
            }
            pointr = pointr % m + 1;
        }
    }
    0
}

fn subsm(st: &mut State, l: &[f64], u: &[f64], nbd: &[i32]) -> i32 {
    let n = st.n;
    let m = st.m;
    let m2 = 2 * m;
    let col = st.col;
    let ns = st.nfree;
    if ns == 0 {
        return 0;
    }
    let mut pointr = st.head;
    for i in 1..=col {
        let mut temp1 = 0.0;
        let mut temp2 = 0.0;
        for j in 1..=ns {
            let k = st.indx[j];
            temp1 += st.wy[ix(k, pointr, n)] * st.r[j];
            temp2 += st.ws[ix(k, pointr, n)] * st.r[j];
        }
        st.wa[i] = temp1;
        st.wa[col + i] = st.theta * temp2;
        pointr = pointr % m + 1;
    }
    let col2 = 2 * col;
    // dtrsl with wn triangle vs wa rhs (separate buffers)
    let info = dtrsl_sep(&st.wn, m2, col2, &mut st.wa, 1, 11);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        st.wa[i] = -st.wa[i];
    }
    let info = dtrsl_sep(&st.wn, m2, col2, &mut st.wa, 1, 1);
    if info != 0 {
        return info;
    }
    let mut pointr = st.head;
    for jy in 1..=col {
        let js = col + jy;
        for i in 1..=ns {
            let k = st.indx[i];
            st.r[i] += st.wy[ix(k, pointr, n)] * st.wa[jy] / st.theta
                + st.ws[ix(k, pointr, n)] * st.wa[js];
        }
        pointr = pointr % m + 1;
    }
    for i in 1..=ns {
        st.r[i] /= st.theta;
    }
    let mut alpha = 1.0_f64;
    let mut temp1 = alpha;
    let mut ibd = 0usize;
    for i in 1..=ns {
        let k = st.indx[i];
        let dk = st.r[i];
        if nbd[k] != 0 {
            if dk < 0.0 && nbd[k] <= 2 {
                let temp2 = l[k] - st.z[k];
                if temp2 >= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0.0 && nbd[k] >= 2 {
                let temp2 = u[k] - st.z[k];
                if temp2 <= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if alpha < 1.0 {
        let dk = st.r[ibd];
        let k = st.indx[ibd];
        if dk > 0.0 {
            st.z[k] = u[k];
            st.r[ibd] = 0.0;
        } else if dk < 0.0 {
            st.z[k] = l[k];
            st.r[ibd] = 0.0;
        }
    }
    for i in 1..=ns {
        let k = st.indx[i];
        st.z[k] += alpha * st.r[i];
    }
    st.iword = if alpha < 1.0 { 1 } else { 0 };
    0
}

fn matupd(st: &mut State, stp: f64, dtd: f64, rr: f64, dr: f64) {
    let n = st.n;
    let m = st.m;
    if st.iupdat as usize <= m {
        st.col = st.iupdat as usize;
        st.itail = (st.head + st.iupdat as usize - 2) % m + 1;
    } else {
        st.itail = st.itail % m + 1;
        st.head = st.head % m + 1;
    }
    for i in 1..=n {
        st.ws[ix(i, st.itail, n)] = st.d[i];
        st.wy[ix(i, st.itail, n)] = st.r[i];
    }
    st.theta = rr / dr;
    if st.iupdat as usize > m {
        for j in 1..=(st.col - 1) {
            dcopy_within(&mut st.ss, j, ix(2, j + 1, m), ix(1, j, m));
            let i2 = st.col - j;
            dcopy_within(&mut st.sy, i2, ix(j + 1, j + 1, m), ix(j, j, m));
        }
    }
    let mut pointr = st.head;
    for j in 1..=(st.col - 1) {
        st.sy[ix(st.col, j, m)] = ddot(n, &st.d, 1, &st.wy, ix(1, pointr, n));
        st.ss[ix(j, st.col, m)] = ddot(n, &st.ws, ix(1, pointr, n), &st.d, 1);
        pointr = pointr % m + 1;
    }
    if stp == 1.0 {
        st.ss[ix(st.col, st.col, m)] = dtd;
    } else {
        st.ss[ix(st.col, st.col, m)] = stp * stp * dtd;
    }
    st.sy[ix(st.col, st.col, m)] = dr;
}

enum LnsResult {
    NeedFg,
    NewX,
    Error, // info = -4
}

#[allow(clippy::too_many_arguments)]
fn lnsrlb(
    st: &mut State,
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    f: &mut f64,
    g: &[f64],
    entering_fg: bool,
) -> LnsResult {
    let n = st.n;
    if !entering_fg {
        st.dtd = ddot(n, &st.d, 1, &st.d, 1);
        st.dnorm = st.dtd.sqrt();
        st.stpmx = 1e10;
        if st.cnstnd {
            if st.iter == 0 {
                st.stpmx = 1.0;
            } else {
                for i in 1..=n {
                    let a1 = st.d[i];
                    if nbd[i] != 0 {
                        if a1 < 0.0 && nbd[i] <= 2 {
                            let a2 = l[i] - x[i];
                            if a2 >= 0.0 {
                                st.stpmx = 0.0;
                            } else if a1 * st.stpmx < a2 {
                                st.stpmx = a2 / a1;
                            }
                        } else if a1 > 0.0 && nbd[i] >= 2 {
                            let a2 = u[i] - x[i];
                            if a2 <= 0.0 {
                                st.stpmx = 0.0;
                            } else if a1 * st.stpmx > a2 {
                                st.stpmx = a2 / a1;
                            }
                        }
                    }
                }
            }
        }
        if st.iter == 0 && !st.boxed {
            st.stp = (1.0 / st.dnorm).min(st.stpmx);
        } else {
            st.stp = 1.0;
        }
        for i in 1..=n {
            st.t[i] = x[i];
            st.r[i] = g[i];
        }
        st.fold = *f;
        st.ifun = 0;
        st.iback = 0;
        st.csave = DcTask::Start;
    }
    // L556
    st.gd = ddot(n, g, 1, &st.d, 1);
    if st.ifun == 0 {
        st.gdold = st.gd;
        if st.gd >= 0.0 {
            st.info = -4;
            return LnsResult::Error;
        }
    }
    dcsrch(st, *f);
    if st.csave != DcTask::Conv && st.csave != DcTask::Warn {
        st.ifun += 1;
        st.nfgv += 1;
        st.iback = st.ifun - 1;
        if st.stp == 1.0 {
            for i in 1..=n {
                x[i] = st.z[i];
            }
        } else {
            for i in 1..=n {
                x[i] = st.stp * st.d[i] + st.t[i];
            }
        }
        LnsResult::NeedFg
    } else {
        LnsResult::NewX
    }
}

fn dcsrch(st: &mut State, f: f64) {
    let stpmin = STPMIN;
    let stpmax = st.stpmx;
    let ftol = FTOL;
    let gtol = GTOL;
    let xtol = XTOL;
    let g = st.gd;
    let dc = &mut st.dcs;
    if st.csave == DcTask::Start {
        // error checks omitted: inputs are valid by construction except
        // possibly stp > stpmax when stpmax == 0
        if st.stp < stpmin || st.stp > stpmax || g >= 0.0 {
            st.csave = DcTask::Error;
            return;
        }
        dc.brackt = false;
        dc.stage = 1;
        dc.finit = f;
        dc.ginit = g;
        dc.gtest = ftol * dc.ginit;
        dc.width = stpmax - stpmin;
        dc.width1 = dc.width / 0.5;
        dc.stx = 0.0;
        dc.fx = dc.finit;
        dc.gx = dc.ginit;
        dc.sty = 0.0;
        dc.fy = dc.finit;
        dc.gy = dc.ginit;
        dc.stmin = 0.0;
        dc.stmax = st.stp + st.stp * 4.0;
        st.csave = DcTask::Fg;
        return;
    }
    let ftest = dc.finit + st.stp * dc.gtest;
    if dc.stage == 1 && f <= ftest && g >= 0.0 {
        dc.stage = 2;
    }
    let mut task = DcTask::Fg;
    if dc.brackt && (st.stp <= dc.stmin || st.stp >= dc.stmax) {
        task = DcTask::Warn;
    }
    if dc.brackt && dc.stmax - dc.stmin <= xtol * dc.stmax {
        task = DcTask::Warn;
    }
    if st.stp == stpmax && f <= ftest && g <= dc.gtest {
        task = DcTask::Warn;
    }
    if st.stp == stpmin && (f > ftest || g >= dc.gtest) {
        task = DcTask::Warn;
    }
    if f <= ftest && g.abs() <= gtol * (-dc.ginit) {
        task = DcTask::Conv;
    }
    if task == DcTask::Warn || task == DcTask::Conv {
        st.csave = task;
        return;
    }
    if dc.stage == 1 && f <= dc.fx && f > ftest {
        let fm = f - st.stp * dc.gtest;
        let mut fxm = dc.fx - dc.stx * dc.gtest;
        let mut fym = dc.fy - dc.sty * dc.gtest;
        let gm = g - dc.gtest;
        let mut gxm = dc.gx - dc.gtest;
        let mut gym = dc.gy - dc.gtest;
        dcstep(
            &mut dc.stx,
            &mut fxm,
            &mut gxm,
            &mut dc.sty,
            &mut fym,
            &mut gym,
            &mut st.stp,
            fm,
            gm,
            &mut dc.brackt,
            dc.stmin,
            dc.stmax,
        );
        dc.fx = fxm + dc.stx * dc.gtest;
        dc.fy = fym + dc.sty * dc.gtest;
        dc.gx = gxm + dc.gtest;
        dc.gy = gym + dc.gtest;
    } else {
        dcstep(
            &mut dc.stx,
            &mut dc.fx,
            &mut dc.gx,
            &mut dc.sty,
            &mut dc.fy,
            &mut dc.gy,
            &mut st.stp,
            f,
            g,
            &mut dc.brackt,
            dc.stmin,
            dc.stmax,
        );
    }
    if dc.brackt {
        if (dc.sty - dc.stx).abs() >= dc.width1 * 0.66 {
            st.stp = dc.stx + (dc.sty - dc.stx) * 0.5;
        }
        dc.width1 = dc.width;
        dc.width = (dc.sty - dc.stx).abs();
    }
    if dc.brackt {
        dc.stmin = dc.stx.min(dc.sty);
        dc.stmax = dc.stx.max(dc.sty);
    } else {
        dc.stmin = st.stp + (st.stp - dc.stx) * 1.1;
        dc.stmax = st.stp + (st.stp - dc.stx) * 4.0;
    }
    if st.stp < stpmin {
        st.stp = stpmin;
    }
    if st.stp > stpmax {
        st.stp = stpmax;
    }
    if (dc.brackt && (st.stp <= dc.stmin || st.stp >= dc.stmax))
        || (dc.brackt && dc.stmax - dc.stmin <= xtol * dc.stmax)
    {
        st.stp = dc.stx;
    }
    st.csave = DcTask::Fg;
}

#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamm = s * ((theta / s) * (theta / s) - *dx / s * (dp / s)).sqrt();
        if *stp < *stx {
            gamm = -gamm;
        }
        let p = gamm - *dx + theta;
        let q = gamm - *dx + gamm + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2.0 * (*stp - *stx);
        if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.0;
        }
        *brackt = true;
    } else if sgnd < 0.0 {
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamm = s * ((theta / s) * (theta / s) - *dx / s * (dp / s)).sqrt();
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let d1 = theta / s;
        let d1 = d1 * d1 - *dx / s * (dp / s);
        let mut gamm = if d1 < 0.0 { 0.0 } else { s * d1.sqrt() };
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm + (*dx - dp) + gamm;
        let r = p / q;
        let stpc;
        if r < 0.0 && gamm != 0.0 {
            stpc = *stp + r * (*stx - *stp);
        } else if *stp > *stx {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if *brackt {
            let mut f = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            let d1 = *stp + (*sty - *stp) * 0.66;
            if *stp > *stx {
                f = d1.min(f);
            } else {
                f = d1.max(f);
            }
            stpf = f;
        } else {
            let mut f = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            f = stpmax.min(f);
            f = stpmin.max(f);
            stpf = f;
        }
    } else if *brackt {
        let theta = (fp - *fy) * 3.0 / (*sty - *stp) + *dy + dp;
        let s = theta.abs().max(dy.abs()).max(dp.abs());
        let mut gamm = s * ((theta / s) * (theta / s) - *dy / s * (dp / s)).sqrt();
        if *stp > *sty {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dy;
        let r = p / q;
        let stpc = *stp + r * (*sty - *stp);
        stpf = stpc;
    } else if *stp > *stx {
        stpf = stpmax;
    } else {
        stpf = stpmin;
    }
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

/// One reverse-communication step of mainlb. `x`, `f`, `g` are 1-based.
#[allow(clippy::too_many_arguments)]
fn mainlb_step(
    st: &mut State,
    x: &mut [f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    f: &mut f64,
    g: &mut [f64],
    pgtol: f64,
    task: &mut Task,
) {
    let n = st.n;

    enum Wh {
        L111,
        L222,
        L333,
        L555,
        L666(bool), // entering with FG (line-search re-entry)?
        L777,
    }

    let mut wh = match *task {
        Task::Start => {
            st.tol = 1e7 * st.epsmch; // factr * epsmch, factr = 1e7
            active(n, l, u, nbd, x, st);
            *task = Task::FgStart;
            return;
        }
        Task::FgStart => Wh::L111,
        Task::FgLnsrch => Wh::L666(true),
        Task::NewX => Wh::L777,
        _ => return,
    };

    loop {
        match wh {
            Wh::L111 => {
                st.nfgv = 1;
                st.sbgnrm = projgr(n, l, u, nbd, x, g);
                if st.sbgnrm <= pgtol {
                    *task = Task::Convergence;
                    return;
                }
                wh = Wh::L222;
            }
            Wh::L222 => {
                st.iword = -1;
                if !st.cnstnd && st.col > 0 {
                    for i in 1..=n {
                        st.z[i] = x[i];
                    }
                    st.wrk_flag = st.updatd;
                    st.nint = 0;
                    wh = Wh::L333;
                    continue;
                }
                let info = cauchy(st, x, l, u, nbd, g);
                if info != 0 {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                    wh = Wh::L222;
                    continue;
                }
                st.nintol += st.nint;
                st.wrk_flag = freev(st);
                wh = Wh::L333;
            }
            Wh::L333 => {
                if st.nfree == 0 || st.col == 0 {
                    wh = Wh::L555;
                    continue;
                }
                let mut failed = false;
                if st.wrk_flag {
                    let info = formk(st);
                    if info != 0 {
                        failed = true;
                    }
                }
                if !failed {
                    let info = cmprlb(st, x, g);
                    if info == 0 {
                        let info2 = subsm(st, l, u, nbd);
                        if info2 != 0 {
                            failed = true;
                        }
                    } else {
                        failed = true;
                    }
                }
                if failed {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                    wh = Wh::L222;
                    continue;
                }
                wh = Wh::L555;
            }
            Wh::L555 => {
                for i in 1..=n {
                    st.d[i] = st.z[i] - x[i];
                }
                wh = Wh::L666(false);
            }
            Wh::L666(entering_fg) => {
                st.info = 0;
                let res = lnsrlb(st, l, u, nbd, x, f, g, entering_fg);
                let err = matches!(res, LnsResult::Error) || st.iback >= 20;
                if err {
                    // restore x, g and f, exactly as the C code does in place
                    for i in 1..=n {
                        x[i] = st.t[i];
                        g[i] = st.r[i];
                    }
                    *f = st.fold;
                    if st.col == 0 {
                        if st.info == 0 {
                            st.nfgv -= 1;
                            st.ifun -= 1;
                            st.iback -= 1;
                        }
                        st.iter += 1;
                        *task = Task::Error;
                        return;
                    } else {
                        if st.info == 0 {
                            st.nfgv -= 1;
                        }
                        st.info = 0;
                        st.col = 0;
                        st.head = 1;
                        st.theta = 1.0;
                        st.iupdat = 0;
                        st.updatd = false;
                        wh = Wh::L222;
                        continue;
                    }
                }
                match res {
                    LnsResult::NeedFg => {
                        *task = Task::FgLnsrch;
                        return;
                    }
                    LnsResult::NewX => {
                        st.iter += 1;
                        st.sbgnrm = projgr(n, l, u, nbd, x, g);
                        *task = Task::NewX;
                        return;
                    }
                    LnsResult::Error => unreachable!(),
                }
            }
            Wh::L777 => {
                if st.sbgnrm <= pgtol {
                    *task = Task::Convergence;
                    return;
                }
                let ddum = st.fold.abs().max(f.abs()).max(1.0);
                if st.fold - *f <= st.tol * ddum {
                    *task = Task::Convergence;
                    return;
                }
                for i in 1..=n {
                    st.r[i] = g[i] - st.r[i];
                }
                let rr = ddot(n, &st.r, 1, &st.r, 1);
                let dr;
                let ddum2;
                if st.stp == 1.0 {
                    dr = st.gd - st.gdold;
                    ddum2 = -st.gdold;
                } else {
                    dr = (st.gd - st.gdold) * st.stp;
                    for i in 1..=n {
                        st.d[i] *= st.stp;
                    }
                    ddum2 = -st.gdold * st.stp;
                }
                if dr <= st.epsmch * ddum2 {
                    st.nskip += 1;
                    st.updatd = false;
                    wh = Wh::L222;
                    continue;
                }
                st.updatd = true;
                st.iupdat += 1;
                matupd(st, st.stp, st.dtd, rr, dr);
                let info = formt(st);
                if info != 0 {
                    st.info = 0;
                    st.col = 0;
                    st.head = 1;
                    st.theta = 1.0;
                    st.iupdat = 0;
                    st.updatd = false;
                }
                wh = Wh::L222;
            }
        }
    }
}

// State needs two extra fields used across the goto structure:
impl State {
    fn new(n: usize, m: usize) -> State {
        State {
            n,
            m,
            prjctd: false,
            cnstnd: false,
            boxed: false,
            updatd: false,
            iback: 0,
            nskip: 0,
            head: 1,
            col: 0,
            itail: 0,
            iter: 0,
            iupdat: 0,
            nint: 0,
            nintol: 0,
            nfgv: 0,
            info: 0,
            ifun: 0,
            iword: 0,
            nfree: n,
            ileave: 0,
            nenter: 0,
            theta: 1.0,
            fold: 0.0,
            tol: 0.0,
            dnorm: 0.0,
            epsmch: f64::EPSILON,
            gd: 0.0,
            stpmx: 0.0,
            sbgnrm: 0.0,
            stp: 0.0,
            gdold: 0.0,
            dtd: 0.0,
            ws: vec![0.0; n * m],
            wy: vec![0.0; n * m],
            sy: vec![0.0; m * m],
            ss: vec![0.0; m * m],
            wt: vec![0.0; m * m],
            wn: vec![0.0; 4 * m * m],
            snd: vec![0.0; 4 * m * m],
            z: vec![0.0; n + 1],
            r: vec![0.0; n + 1],
            d: vec![0.0; n + 1],
            t: vec![0.0; n + 1],
            wa: vec![0.0; 8 * m + 1],
            indx: vec![0; n + 1],
            iwhere: vec![0; n + 1],
            indx2: vec![0; n + 1],
            dcs: Dcsrch::default(),
            csave: DcTask::Start,
            wrk_flag: false,
        }
    }
}

pub struct OptimResult {
    pub par: Vec<f64>,
    #[allow(dead_code)]
    pub value: f64,
    /// 0 = converged (factr test), 1 = maxit, 51 = warning, 52 = error.
    pub convergence: i32,
}

/// R's `optim(par, fn, method="L-BFGS-B", lower, upper)` with the default
/// control parameters DESeq2 uses: factr=1e7, pgtol=0, lmm=5, maxit=100,
/// parscale=1, fnscale=1, and the numerical gradient with ndeps=1e-3
/// (bounds-aware central differences).
pub fn optim_lbfgsb<F: FnMut(&[f64]) -> f64>(
    par: &[f64],
    lower: &[f64],
    upper: &[f64],
    mut fun: F,
) -> OptimResult {
    let n = par.len();
    let m = 5usize;
    let maxit = 100;
    let pgtol = 0.0;
    let ndeps = 1e-3;

    // 1-based buffers
    let mut x = vec![0.0; n + 1];
    let mut l = vec![0.0; n + 1];
    let mut u = vec![0.0; n + 1];
    let mut nbd = vec![0i32; n + 1];
    for i in 0..n {
        x[i + 1] = par[i];
        l[i + 1] = lower[i];
        u[i + 1] = upper[i];
        nbd[i + 1] = 2; // both bounds finite in DESeq2's call
    }
    let mut g = vec![0.0; n + 1];
    let mut f = 0.0;
    let mut st = State::new(n, m);
    let mut task = Task::Start;
    let mut iter = 0;
    let fail;

    let mut p0 = vec![0.0; n];
    loop {
        mainlb_step(&mut st, &mut x, &l, &u, &nbd, &mut f, &mut g, pgtol, &mut task);
        match task {
            Task::FgStart | Task::FgLnsrch => {
                p0.copy_from_slice(&x[1..=n]);
                f = fun(&p0);
                if !f.is_finite() {
                    // R would error(); our objective returns 1e300 instead
                    fail = 52;
                    break;
                }
                // numerical gradient, usebounds branch of fmingr
                for i in 0..n {
                    let eps0 = ndeps;
                    let mut epsused = eps0;
                    let mut tmp = p0[i] + eps0;
                    if tmp > upper[i] {
                        tmp = upper[i];
                        epsused = tmp - p0[i];
                    }
                    let saved = p0[i];
                    p0[i] = tmp;
                    let val1 = fun(&p0);
                    let mut eps = eps0;
                    tmp = saved - eps0;
                    if tmp < lower[i] {
                        tmp = lower[i];
                        eps = saved - tmp;
                    }
                    p0[i] = tmp;
                    let val2 = fun(&p0);
                    g[i + 1] = (val1 - val2) / (epsused + eps);
                    p0[i] = saved;
                }
            }
            Task::NewX => {
                iter += 1;
                if iter > maxit {
                    fail = 1;
                    break;
                }
            }
            Task::Convergence => {
                fail = 0;
                break;
            }
            Task::Warning => {
                fail = 51;
                break;
            }
            Task::Error => {
                fail = 52;
                break;
            }
            _ => {
                fail = 52;
                break;
            }
        }
    }

    OptimResult {
        par: x[1..=n].to_vec(),
        value: f,
        convergence: fail,
    }
}
