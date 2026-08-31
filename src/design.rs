//! Design formula parsing and model-matrix construction, replicating R's
//! `model.matrix.default` for additive formulas (`~ cov1 + cov2 + condition`)
//! with treatment (dummy) coding:
//!
//! * factor columns contribute one indicator column per non-reference level,
//!   with levels ordered as R's `factor()` orders them (sorted unique values)
//!   and the first level as the reference;
//! * numeric columns contribute a single column with their values.
//!
//! Interactions are not supported. The contrast (case vs control of one factor)
//! is resolved the way DESeq2's `results(dds, contrast=c(var, case, control))`
//! resolves it: a single coefficient when the control is the reference level, a
//! negated coefficient when the case is the reference, and a general numeric
//! contrast vector otherwise.

use crate::io::ColData;

pub enum VarKind {
    /// Sorted unique levels; `levels[0]` is the reference.
    Factor { levels: Vec<String> },
    Numeric,
}

pub struct DesignVar {
    pub name: String,
    pub kind: VarKind,
}

pub struct Design {
    /// Model matrix, row-major n x p (first column = intercept).
    pub x: Vec<f64>,
    pub n: usize,
    pub p: usize,
    /// Column names: "Intercept", then "var_level_vs_ref" / "var".
    pub col_names: Vec<String>,
    pub vars: Vec<DesignVar>,
    /// Values of each design variable per sample (aligned to count columns).
    pub values: Vec<Vec<String>>,
}

pub enum Contrast {
    /// Case vs reference: use this coefficient directly.
    Coef(usize),
    /// Reference vs control (case is the reference level): -1 x coefficient.
    NegCoef(usize),
    /// General contrast vector c: c'beta / sqrt(c' Sigma c).
    Vector(Vec<f64>),
}

pub struct ResolvedContrast {
    pub kind: Contrast,
    /// Samples belonging to the case or control level (for the all-zero rule).
    pub in_contrast_levels: Vec<bool>,
}

/// Parse the design specification: either a formula "~ a + b + c" or a bare
/// column name (legacy single-factor form). Returns the variable names.
pub fn parse_design_formula(design: &str) -> Result<Vec<String>, String> {
    let s = design.trim();
    let body = s.strip_prefix('~').unwrap_or(s).trim();
    if body.contains(':') || body.contains('*') {
        return Err("interaction terms (':' or '*') in the design are not supported".into());
    }
    if body.is_empty() || body == "1" {
        return Err("the design must contain at least one variable".into());
    }
    let vars: Vec<String> = body
        .split('+')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && v != "1")
        .collect();
    if vars.is_empty() {
        return Err("the design must contain at least one variable".into());
    }
    let mut seen = std::collections::HashSet::new();
    for v in &vars {
        if !seen.insert(v.clone()) {
            return Err(format!("variable '{v}' appears twice in the design"));
        }
    }
    Ok(vars)
}

/// Values in `coldata` for column `name`, reordered to match `sample_order`.
fn aligned_column(
    coldata: &ColData,
    sample_ids: &[String],
    name: &str,
    sample_order: &[String],
) -> Result<Vec<String>, String> {
    let vals = coldata.column(name)?;
    let mut out = Vec::with_capacity(sample_order.len());
    for s in sample_order {
        let idx = sample_ids
            .iter()
            .position(|id| id == s)
            .ok_or_else(|| format!("sample '{s}' from counts not found in colData"))?;
        out.push(vals[idx].clone());
    }
    Ok(out)
}

/// Does every value parse as a finite number? (Mirrors R reading the column as
/// numeric with `read.delim`, which DESeq2 then treats as a continuous
/// covariate.)
fn all_numeric(vals: &[String]) -> bool {
    vals.iter()
        .all(|v| v.trim().parse::<f64>().map(|x| x.is_finite()).unwrap_or(false))
}

/// Build the model matrix for the additive design over `var_names`.
/// `force_factor` lists columns to treat as factors even if numeric-looking.
pub fn build_design(
    coldata: &ColData,
    sample_ids: &[String],
    sample_order: &[String],
    var_names: &[String],
    force_factor: &[String],
) -> Result<Design, String> {
    let n = sample_order.len();
    let mut vars = Vec::new();
    let mut values = Vec::new();

    for name in var_names {
        let vals = aligned_column(coldata, sample_ids, name, sample_order)?;
        let numeric = all_numeric(&vals) && !force_factor.iter().any(|f| f == name);
        let kind = if numeric {
            VarKind::Numeric
        } else {
            let mut levels: Vec<String> = vals.to_vec();
            levels.sort();
            levels.dedup();
            if levels.len() < 2 {
                return Err(format!(
                    "design variable '{name}' has a single level '{}'",
                    levels[0]
                ));
            }
            VarKind::Factor { levels }
        };
        vars.push(DesignVar {
            name: name.clone(),
            kind,
        });
        values.push(vals);
    }

    // Count columns.
    let mut p = 1; // intercept
    for v in &vars {
        p += match &v.kind {
            VarKind::Factor { levels } => levels.len() - 1,
            VarKind::Numeric => 1,
        };
    }
    if p >= n {
        return Err(format!(
            "the design has {p} coefficients for {n} samples; \
             there are no replicates to estimate the dispersion"
        ));
    }

    let mut x = vec![0.0; n * p];
    let mut col_names = vec!["Intercept".to_string()];
    for row in x.iter_mut().step_by(p) {
        *row = 1.0;
    }
    let mut col = 1;
    for (vi, v) in vars.iter().enumerate() {
        match &v.kind {
            VarKind::Factor { levels } => {
                for lv in &levels[1..] {
                    col_names.push(format!("{}_{}_vs_{}", v.name, lv, levels[0]));
                    for i in 0..n {
                        if &values[vi][i] == lv {
                            x[i * p + col] = 1.0;
                        }
                    }
                    col += 1;
                }
            }
            VarKind::Numeric => {
                col_names.push(v.name.clone());
                for i in 0..n {
                    x[i * p + col] = values[vi][i].trim().parse::<f64>().unwrap();
                }
                col += 1;
            }
        }
    }

    Ok(Design {
        x,
        n,
        p,
        col_names,
        vars,
        values,
    })
}

impl Design {
    /// Resolve the requested contrast `(var, case, control)`.
    pub fn resolve_contrast(
        &self,
        var: &str,
        case: &str,
        control: &str,
    ) -> Result<ResolvedContrast, String> {
        let vi = self
            .vars
            .iter()
            .position(|v| v.name == var)
            .ok_or_else(|| format!("contrast variable '{var}' is not in the design"))?;
        let levels = match &self.vars[vi].kind {
            VarKind::Factor { levels } => levels,
            VarKind::Numeric => {
                return Err(format!(
                    "contrast variable '{var}' is numeric; level contrasts need a factor"
                ))
            }
        };
        if case == control {
            return Err("case and control levels must differ".into());
        }
        for lv in [case, control] {
            if !levels.iter().any(|l| l == lv) {
                return Err(format!("level '{lv}' not found in design variable '{var}'"));
            }
        }
        let reference = &levels[0];
        let col_of = |lv: &str| -> usize {
            let name = format!("{}_{}_vs_{}", var, lv, reference);
            self.col_names.iter().position(|c| c == &name).unwrap()
        };
        let kind = if control == reference {
            Contrast::Coef(col_of(case))
        } else if case == reference {
            Contrast::NegCoef(col_of(control))
        } else {
            let mut c = vec![0.0; self.p];
            c[col_of(case)] = 1.0;
            c[col_of(control)] = -1.0;
            Contrast::Vector(c)
        };
        let in_contrast_levels: Vec<bool> = self.values[vi]
            .iter()
            .map(|v| v == case || v == control)
            .collect();
        Ok(ResolvedContrast {
            kind,
            in_contrast_levels,
        })
    }

    /// Is this a single-variable design whose variable is a two-level factor?
    /// (Gates DESeq2's Cook's-filtering heuristic in `results()`.)
    pub fn single_two_level_factor(&self) -> bool {
        self.vars.len() == 1
            && matches!(&self.vars[0].kind, VarKind::Factor { levels } if levels.len() == 2)
    }

    /// Hash of each sample's model-matrix row, defining the "cells" used by
    /// DESeq2's `nOrMoreInCell` / `modelMatrixGroups`.
    pub fn row_cells(&self) -> Vec<String> {
        (0..self.n)
            .map(|i| {
                self.x[i * self.p..(i + 1) * self.p]
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join("_")
            })
            .collect()
    }

    /// For each sample: does its cell (identical model-matrix row) contain at
    /// least `n_min` samples?
    pub fn n_or_more_in_cell(&self, n_min: usize) -> Vec<bool> {
        let cells = self.row_cells();
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        for c in &cells {
            *counts.entry(c.as_str()).or_insert(0) += 1;
        }
        cells
            .iter()
            .map(|c| counts[c.as_str()] >= n_min)
            .collect()
    }

    /// DESeq2's `linearMu` rule: use a linear model for the expected counts
    /// when the number of distinct model-matrix rows equals the number of
    /// coefficients.
    pub fn use_linear_mu(&self) -> bool {
        let cells = self.row_cells();
        let mut uniq = std::collections::HashSet::new();
        for c in cells {
            uniq.insert(c);
        }
        uniq.len() == self.p
    }
}
