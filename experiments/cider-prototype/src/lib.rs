//! Isolated 1D numerical semiconductor feasibility prototype; SI-facing parameters, nondimensional FV.
//! Uniform-material Poisson / electron and hole continuity, SG flux,
//! Boltzmann statistics, SRH, BE time integration. No CIDER parser or ABI.
use std::{
    array,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

const Q: f64 = 1.602_191_770e-19;
const K: f64 = 1.380_622_59e-23;
const EPS: f64 = 8.854e-12 * 11.7;
const NREF: f64 = 1e22;
const AREA: f64 = 1e-6;
const NC: f64 = 2.8e25;
const NV: f64 = 1.04e25;
const GAP: f64 = 1.12;
const MU_N: f64 = 0.1;
const MU_P: f64 = 0.04;
const TAU: f64 = 1e-6;
type V = [f64; 3];
type M = [[f64; 3]; 3];

#[derive(Clone, Copy)]
struct D {
    x: f64,
    d: [f64; 10],
}
impl D {
    fn c(x: f64) -> Self {
        Self { x, d: [0.; 10] }
    }
    fn var(x: f64, i: usize) -> Self {
        let mut a = Self::c(x);
        a.d[i] = 1.;
        a
    }
    fn map(self, x: f64, derivative: f64) -> Self {
        Self {
            x,
            d: self.d.map(|d| d * derivative),
        }
    }
    fn exp(self) -> Self {
        let x = self.x.exp();
        self.map(x, x)
    }
    fn expm1(self) -> Self {
        self.map(self.x.exp_m1(), self.x.exp())
    }
    fn b(self) -> Self {
        let x = self.x;
        if x.abs() < 1e-3 {
            let x2 = x * x;
            self.map(
                1. - x / 2. + x2 / 12. - x2 * x2 / 720.,
                -0.5 + x / 6. - x * x2 / 180.,
            )
        } else if x > 50. {
            let e = (-x).exp();
            self.map(x * e / (1. - e), e * (1. - x - e) / (1. - e).powi(2))
        } else {
            let m = x.exp_m1();
            self.map(x / m, (m - x * (m + 1.)) / (m * m))
        }
    }
}
impl Add for D {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            x: self.x + b.x,
            d: array::from_fn(|i| self.d[i] + b.d[i]),
        }
    }
}
impl Sub for D {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            x: self.x - b.x,
            d: array::from_fn(|i| self.d[i] - b.d[i]),
        }
    }
}
impl Mul for D {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            x: self.x * b.x,
            d: array::from_fn(|i| self.d[i].mul_add(b.x, self.x * b.d[i])),
        }
    }
}
fn quotient_derivative(
    numerator: f64,
    quotient: f64,
    denominator_derivative: f64,
    denominator: f64,
) -> f64 {
    (numerator - quotient * denominator_derivative) / denominator
}

impl Div for D {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let x = self.x / b.x;
        Self {
            x,
            d: array::from_fn(|i| quotient_derivative(self.d[i], x, b.d[i], b.x)),
        }
    }
}
impl Neg for D {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            d: self.d.map(|d| -d),
        }
    }
}

#[derive(Clone)]
/// Immutable accepted or trial state, in internal normalized coordinates.
#[derive(Debug)]
pub struct State {
    nodes: Vec<V>,
    geometry: Arc<Geometry>,
    temp: f64,
    bias: f64,
}
#[derive(Clone)]
/// Uniform-material one-dimensional device geometry with ohmic contacts.
#[derive(Debug)]
pub struct Model {
    geometry: Arc<Geometry>,
}
#[derive(Debug)]
struct Geometry {
    x: Vec<f64>,
    doping: Vec<f64>,
}

struct Scale {
    vt: f64,
    l: f64,
    t: f64,
    ni: f64,
    j: f64,
    mu_p: f64,
    tau: f64,
}
impl Scale {
    fn new(temp: f64) -> Self {
        let vt = K * temp / Q;
        let l = (EPS * vt / (Q * NREF)).sqrt();
        let mu = MU_N * (300. / temp).powf(2.3);
        let t = l * l / (mu * vt);
        let gap = GAP + 4.73e-4 * (300. * 300. / 936. - temp * temp / (temp + 636.));
        let ni = (NC * NV).sqrt() * (temp / 300.).powf(1.5) * (-gap / (2. * vt)).exp() / NREF;
        let tau = TAU * (300. / temp).sqrt() * (3.8667 * (300. / temp - 1.)).exp();
        Self {
            vt,
            l,
            t,
            ni,
            j: Q * NREF * mu * vt / l,
            mu_p: MU_P / MU_N,
            tau,
        }
    }

    fn valid(&self) -> bool {
        [
            self.vt, self.l, self.t, self.ni, self.j, self.mu_p, self.tau,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v > 0.)
    }
}
fn density(s: V, scale: &Scale, bias: f64) -> (f64, f64) {
    (
        scale.ni * (s[0] + s[1]).exp(),
        scale.ni * (-s[0] + s[2] + bias / scale.vt).exp(),
    )
}
fn boundary(doping: f64, contact: f64, bias: f64, scale: &Scale) -> V {
    let v = contact / scale.vt;
    [
        (doping / (2. * scale.ni)).asinh() + v,
        -v,
        (contact - bias) / scale.vt,
    ]
}
fn flux(a: [D; 3], b: [D; 3], h: f64, s: &Scale, bias: D) -> (D, D) {
    let dp = b[0] - a[0];
    let n = D::c(s.ni) * (a[0] + a[1]).exp();
    let p = D::c(s.ni) * (-a[0] + a[2] + (bias / D::c(s.vt))).exp();
    let jn = n * (-dp).b() * (b[1] - a[1]).expm1() / D::c(h);
    let jp = -D::c(s.mu_p) * p * dp.b() * (b[2] - a[2]).expm1() / D::c(h);
    (jn, jp)
}
#[derive(Clone, Copy)]
struct Row<const N: usize> {
    lo: [[f64; N]; N],
    diag: [[f64; N]; N],
    hi: [[f64; N]; N],
    rhs: [f64; N],
    bias_rhs: [f64; N],
    mass: [[f64; N]; N],
    mass_bias_rhs: [f64; N],
}
fn mm<const N: usize>(a: [[f64; N]; N], b: [[f64; N]; N]) -> [[f64; N]; N] {
    array::from_fn(|i| array::from_fn(|j| (0..N).map(|k| a[i][k] * b[k][j]).sum()))
}
fn mv<const N: usize>(a: [[f64; N]; N], b: [f64; N]) -> [f64; N] {
    array::from_fn(|i| (0..N).map(|k| a[i][k] * b[k]).sum())
}
fn msub<const N: usize>(a: [[f64; N]; N], b: [[f64; N]; N]) -> [[f64; N]; N] {
    array::from_fn(|i| array::from_fn(|j| a[i][j] - b[i][j]))
}
fn vsub<const N: usize>(a: [f64; N], b: [f64; N]) -> [f64; N] {
    array::from_fn(|i| a[i] - b[i])
}
fn local_solve<const N: usize>(
    mut a: [[f64; N]; N],
    mut b: [[f64; N]; N],
    mut r: [f64; N],
) -> Option<([[f64; N]; N], [f64; N])> {
    for k in 0..N {
        let pivot = (k..N).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        a.swap(k, pivot);
        b.swap(k, pivot);
        r.swap(k, pivot);
        let p = a[k][k];
        if p == 0. || !p.is_finite() {
            return None;
        }
        for value in &mut a[k][k..] {
            *value /= p;
        }
        for value in &mut b[k] {
            *value /= p;
        }
        r[k] /= p;
        for i in 0..N {
            if i != k {
                let f = a[i][k];
                for j in k..N {
                    a[i][j] -= f * a[k][j];
                }
                for j in 0..N {
                    b[i][j] -= f * b[k][j];
                }
                r[i] -= f * r[k];
            }
        }
    }
    if b.iter().flatten().chain(&r).any(|v| !v.is_finite()) {
        return None;
    }
    Some((b, r))
}
fn block_solve<const N: usize>(rows: &mut [Row<N>]) -> Option<Vec<[f64; N]>> {
    let mut up = [[0.; N]; N];
    let mut right = [0.; N];
    for row in rows.iter_mut() {
        let a = msub(row.diag, mm(row.lo, up));
        let r = vsub(row.rhs, mv(row.lo, right));
        (up, right) = local_solve(a, row.hi, r)?;
        row.hi = up;
        row.rhs = right;
    }
    let mut next = [0.; N];
    let mut sol = vec![[0.; N]; rows.len()];
    for (i, row) in rows.iter().enumerate().rev() {
        next = vsub(row.rhs, mv(row.hi, next));
        if next.iter().any(|v| !v.is_finite()) {
            return None;
        }
        sol[i] = next;
    }
    Some(sol)
}
#[derive(Clone, Copy)]
/// Complex Laplace frequency or terminal admittance.
#[derive(Debug)]
pub struct C {
    pub re: f64,
    pub im: f64,
}
impl C {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}
impl Add for C {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl Sub for C {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl Mul for C {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl Mul<f64> for C {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self::new(self.re * b, self.im * b)
    }
}
fn lift(a: M, m: M, p: C) -> [[f64; 6]; 6] {
    array::from_fn(|i| {
        array::from_fn(|j| {
            let (r, c) = (i % 3, j % 3);
            if i / 3 == j / 3 {
                a[r][c] + p.re * m[r][c]
            } else if i < 3 {
                -p.im * m[r][c]
            } else {
                p.im * m[r][c]
            }
        })
    })
}
/// Immutable DC linearization, reusable for each Laplace frequency.
pub struct DynamicTerminal {
    rows: Vec<Row<3>>,
    edges: Vec<([f64; 10], f64)>,
    vt: f64,
}
impl DynamicTerminal {
    pub fn evaluate(&self, p: C) -> Result<(C, f64), String> {
        if !p.re.is_finite() || !p.im.is_finite() {
            return Err("nonfinite Laplace frequency".into());
        }
        let mut complex_rows: Vec<Row<6>> = self
            .rows
            .iter()
            .map(|r| {
                let mut row = Row {
                    lo: lift(r.lo, [[0.; 3]; 3], p),
                    diag: lift(r.diag, r.mass, p),
                    hi: lift(r.hi, [[0.; 3]; 3], p),
                    rhs: array::from_fn(|i| {
                        if i < 3 {
                            r.bias_rhs[i] + p.re * r.mass_bias_rhs[i]
                        } else {
                            p.im * r.mass_bias_rhs[i - 3]
                        }
                    }),
                    bias_rhs: [0.; 6],
                    mass: [[0.; 6]; 6],
                    mass_bias_rhs: [0.; 6],
                };
                // Re-equilibrate the dynamic system at its actual frequency.
                for i in 0..6 {
                    let w = row.lo[i]
                        .iter()
                        .chain(&row.diag[i])
                        .chain(&row.hi[i])
                        .map(|v| v.abs())
                        .sum::<f64>()
                        .max(1e-300);
                    for j in 0..6 {
                        row.lo[i][j] /= w;
                        row.diag[i][j] /= w;
                        row.hi[i][j] /= w;
                    }
                    row.rhs[i] /= w;
                }
                row
            })
            .collect();
        let sensitivities = block_solve(&mut complex_rows).ok_or("singular dynamic system")?;
        let mut x = Vec::with_capacity(self.edges.len() + 1);
        x.push([
            C::new(1. / self.vt, 0.),
            C::new(-1. / self.vt, 0.),
            C::new(0., 0.),
        ]);
        x.extend(
            sensitivities
                .iter()
                .map(|v| array::from_fn(|j| C::new(v[j], v[j + 3]))),
        );
        x.push([C::new(0., 0.), C::new(0., 0.), C::new(-1. / self.vt, 0.)]);
        let mut y = Vec::with_capacity(x.len() - 1);
        for i in 0..x.len() - 1 {
            let (current, cap) = self.edges[i];
            let mut di = C::new(current[9], 0.);
            for j in 0..3 {
                di = di + x[i][j] * current[j] + x[i + 1][j] * current[j + 3];
            }
            let dq = (x[i][0] - x[i + 1][0]) * cap;
            y.push(di + p * dq);
        }
        if y.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err("nonfinite terminal admittance".into());
        }
        let center = y[y.len() / 2];
        let error = y.iter().map(|v| (*v - center).abs()).fold(0., f64::max);
        Ok((center, error))
    }
}
impl State {
    pub fn bias(&self) -> f64 {
        self.bias
    }
    pub fn temperature(&self) -> f64 {
        self.temp
    }
}

impl Model {
    /// Positions in metres and signed net ionized doping in m^-3.
    /// The preset uniform silicon transport laws and contact area are fixed.
    pub fn new(x: Vec<f64>, doping_per_m3: Vec<f64>) -> Result<Self, String> {
        if x.len() < 3
            || x.len() != doping_per_m3.len()
            || x.iter().any(|v| !v.is_finite())
            || x.windows(2)
                .any(|w| !(w[1] - w[0]).is_finite() || w[1] <= w[0])
            || doping_per_m3.iter().any(|v| !v.is_finite())
        {
            return Err("invalid one-dimensional mesh or doping".into());
        }
        Ok(Self {
            geometry: Arc::new(Geometry {
                x,
                doping: doping_per_m3.into_iter().map(|d| d / NREF).collect(),
            }),
        })
    }

    pub fn positions(&self) -> &[f64] {
        &self.geometry.x
    }

    fn check_state(&self, state: &State) -> Result<(), String> {
        if Arc::ptr_eq(&self.geometry, &state.geometry) {
            Ok(())
        } else {
            Err("state belongs to a different prepared mesh".into())
        }
    }

    fn check_history(&self, old: Option<(&State, f64)>) -> Result<(), String> {
        if let Some((state, dt)) = old {
            self.check_state(state)?;
            if !dt.is_finite() || dt <= 0. {
                return Err("invalid time step".into());
            }
        }
        Ok(())
    }

    pub fn midpoint_field(&self, state: &State) -> Result<f64, String> {
        self.check_state(state)?;
        let j = state.nodes.len() / 2;
        Ok(
            -(state.nodes[j + 1][0] - state.nodes[j - 1][0]) * Scale::new(state.temp).vt
                / (self.geometry.x[j + 1] - self.geometry.x[j - 1]),
        )
    }

    /// Exact homogeneous, neutral material small-signal admittance.
    pub fn uniform_admittance(&self, doping_per_m3: f64, p: C) -> C {
        let scale = Scale::new(300.);
        let (n, holes) = density(boundary(doping_per_m3 / NREF, 0., 0., &scale), &scale, 0.);
        let length = self.geometry.x[self.geometry.x.len() - 1] - self.geometry.x[0];
        C::new(AREA * Q * NREF * (MU_N * n + MU_P * holes) / length, 0.) + p * (AREA * EPS / length)
    }

    pub fn abrupt(nodes: usize, clustered: bool) -> Self {
        assert!(nodes >= 3 && nodes % 2 == 1);
        let x: Vec<_> = (0..nodes)
            .map(|i| {
                let u = 2. * i as f64 / (nodes - 1) as f64 - 1.;
                if clustered {
                    1e-6 * (1. + (2. * u).sinh() / 2f64.sinh())
                } else {
                    1e-6 * (1. + u)
                }
            })
            .collect();
        let doping = (0..nodes)
            .map(|i| {
                if 2 * i < nodes - 1 {
                    -1.
                } else if 2 * i > nodes - 1 {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        Self {
            geometry: Arc::new(Geometry { x, doping }),
        }
    }
    pub fn initial(&self, temp: f64) -> Result<State, String> {
        let s = Scale::new(temp);
        if !s.valid() {
            return Err("invalid temperature normalization".into());
        }
        Ok(State {
            nodes: self
                .geometry
                .doping
                .iter()
                .map(|&d| boundary(d, 0., 0., &s))
                .collect(),
            geometry: Arc::clone(&self.geometry),
            temp,
            bias: 0.,
        })
    }
    fn assemble(&self, state: &State, previous: Option<(&State, f64)>) -> (Vec<Row<3>>, f64) {
        let s = Scale::new(state.temp);
        let mut rows = Vec::with_capacity(state.nodes.len() - 2);
        let mut norm = 0f64;
        let old_scale = previous.map(|(old, _)| Scale::new(old.temp));
        for i in 1..state.nodes.len() - 1 {
            let mut a: [D; 3] = array::from_fn(|j| D::var(state.nodes[i - 1][j], j));
            if i == 1 {
                a[0].d[9] = 1. / s.vt;
                a[1].d[9] = -1. / s.vt;
            }
            let b: [D; 3] = array::from_fn(|j| D::var(state.nodes[i][j], j + 3));
            let mut c: [D; 3] = array::from_fn(|j| D::var(state.nodes[i + 1][j], j + 6));
            if i + 2 == state.nodes.len() {
                c[2].d[9] = -1. / s.vt;
            }
            let bias = D::var(state.bias, 9);
            let hl = (self.geometry.x[i] - self.geometry.x[i - 1]) / s.l;
            let hr = (self.geometry.x[i + 1] - self.geometry.x[i]) / s.l;
            let vol = D::c((hl + hr) * 0.5);
            let n = D::c(s.ni) * (b[0] + b[1]).exp();
            let p = D::c(s.ni) * (-b[0] + b[2] + (bias / D::c(s.vt))).exp();
            let recomb = D::c(s.ni * s.ni * s.t / s.tau)
                * (b[1] + b[2] + (bias / D::c(s.vt))).expm1()
                / (n + p + D::c(2. * s.ni));
            let (jl, kl) = flux(a, b, hl, &s, bias);
            let (jr, kr) = flux(b, c, hr, &s, bias);
            let (dn, dp) = if let Some((old, dt)) = previous {
                let (no, po) = density(old.nodes[i], old_scale.as_ref().unwrap(), old.bias);
                let rate = D::c(s.t / dt);
                ((n - D::c(no)) * rate, (p - D::c(po)) * rate)
            } else {
                (D::c(0.), D::c(0.))
            };
            let f = [
                (b[0] - a[0]) / D::c(hl)
                    - (c[0] - b[0]) / D::c(hr)
                    - vol * (p - n + D::c(self.geometry.doping[i])),
                jl - jr + vol * (dn + recomb),
                kr - kl + vol * (dp + recomb),
            ];
            // Row scaling by Jacobian norm: dimensionless Newton residual.
            let w: V =
                array::from_fn(|k| f[k].d[..9].iter().map(|d| d.abs()).sum::<f64>().max(1e-300));
            if f.iter()
                .any(|value| !value.x.is_finite() || value.d.iter().any(|d| !d.is_finite()))
                || w.iter().any(|value| !value.is_finite())
            {
                norm = f64::INFINITY;
            } else {
                norm = norm.max((0..3).map(|k| f[k].x.abs() / w[k]).fold(0., f64::max));
            }
            let block = |off| array::from_fn(|k| array::from_fn(|j| f[k].d[j + off] / w[k]));
            rows.push(Row {
                lo: if i == 1 { [[0.; 3]; 3] } else { block(0) },
                diag: block(3),
                hi: if i + 2 == state.nodes.len() {
                    [[0.; 3]; 3]
                } else {
                    block(6)
                },
                rhs: array::from_fn(|k| -f[k].x / w[k]),
                bias_rhs: array::from_fn(|k| -f[k].d[9] / w[k]),
                mass: array::from_fn(|k| {
                    array::from_fn(|j| {
                        [D::c(0.), vol * n * D::c(s.t), vol * p * D::c(s.t)][k].d[3 + j] / w[k]
                    })
                }),
                mass_bias_rhs: array::from_fn(|k| {
                    -[D::c(0.), vol * n * D::c(s.t), vol * p * D::c(s.t)][k].d[9] / w[k]
                }),
            });
        }
        (rows, norm)
    }
    pub fn solve(
        &self,
        mut state: State,
        bias: f64,
        temp: f64,
        previous: Option<(&State, f64)>,
    ) -> Result<(State, usize, f64), String> {
        if !bias.is_finite()
            || !temp.is_finite()
            || temp <= 0.
            || !Arc::ptr_eq(&state.geometry, &self.geometry)
            || previous.is_some_and(|(old, dt)| {
                !Arc::ptr_eq(&old.geometry, &self.geometry) || !dt.is_finite() || dt <= 0.
            })
        {
            return Err("invalid trial bias, temperature, state or time step".into());
        }
        if !Scale::new(temp).valid() {
            return Err("invalid temperature normalization".into());
        }
        // Translate previous physical densities/potential to the new normalization.
        if state.temp != temp {
            let old = Scale::new(state.temp);
            let new = Scale::new(temp);
            for v in &mut state.nodes {
                let (n, p) = density(*v, &old, state.bias);
                v[0] *= old.vt / new.vt;
                v[1] = (n / new.ni).ln() - v[0];
                v[2] = (p / new.ni).ln() + v[0] - bias / new.vt;
            }
        } else {
            let shift = (state.bias - bias) / Scale::new(temp).vt;
            for v in &mut state.nodes {
                v[2] += shift;
            }
        }
        state.temp = temp;
        state.bias = bias;
        let s = Scale::new(temp);
        let last = state.nodes.len() - 1;
        state.nodes[0] = boundary(self.geometry.doping[0], bias, bias, &s);
        state.nodes[last] = boundary(self.geometry.doping[last], 0., bias, &s);
        for it in 0..150 {
            let (mut rows, norm) = self.assemble(&state, previous);
            if !norm.is_finite() {
                return Err("nonfinite initial residual".into());
            }
            let delta = block_solve(&mut rows).ok_or("singular block")?;
            let step = delta.iter().flatten().map(|x| x.abs()).fold(0., f64::max);
            if step < 2e-12 && norm < 1e-12 {
                return Ok((state, it, norm));
            }
            let mut factor = (4. / step).min(1.);
            let mut accepted = None;
            while factor >= 1e-8 {
                let mut trial = state.clone();
                for (v, d) in trial.nodes[1..last].iter_mut().zip(&delta) {
                    for j in 0..3 {
                        v[j] += factor * d[j];
                    }
                }
                let (_, next) = self.assemble(&trial, previous);
                if next.is_finite() && (next < norm || next < 1e-13) {
                    accepted = Some(trial);
                    break;
                }
                factor *= 0.5;
            }
            state = accepted.ok_or_else(|| {
                format!("line search failed at bias {bias}, step {step}, norm {norm}")
            })?;
        }
        Err(format!("iteration limit at bias {bias}"))
    }
    pub fn currents(&self, state: &State, old: Option<(&State, f64)>) -> Result<Vec<f64>, String> {
        self.check_state(state)?;
        self.check_history(old)?;
        let s = Scale::new(state.temp);
        Ok(state
            .nodes
            .windows(2)
            .enumerate()
            .map(|(i, pair)| {
                let h = self.geometry.x[i + 1] - self.geometry.x[i];
                let (n, p) = flux(
                    pair[0].map(D::c),
                    pair[1].map(D::c),
                    h / s.l,
                    &s,
                    D::c(state.bias),
                );
                let disp = old.map_or(0., |(old, dt)| {
                    let o = Scale::new(old.temp);
                    -EPS * ((pair[1][0] - pair[0][0]) * s.vt
                        - (old.nodes[i + 1][0] - old.nodes[i][0]) * o.vt)
                        / (h * dt)
                });
                AREA * ((n.x + p.x) * s.j + disp)
            })
            .collect())
    }
    /// Implicit-function elimination of the converged internal PDE. All
    /// other terminal voltages, temperature and accepted history held fixed.
    pub fn terminal_stamp(
        &self,
        state: &State,
        old: Option<(&State, f64)>,
    ) -> Result<(f64, f64), String> {
        self.check_state(state)?;
        self.check_history(old)?;
        let s = Scale::new(state.temp);
        let (mut rows, _) = self.assemble(state, old);
        for row in &mut rows {
            row.rhs = row.bias_rhs;
        }
        let sensitivity = block_solve(&mut rows).ok_or("singular voltage sensitivity")?;
        let mut a = state.nodes[0].map(D::c);
        a[0].d[9] = 1. / s.vt;
        a[1].d[9] = -1. / s.vt;
        let mut b = state.nodes[1].map(D::c);
        for j in 0..3 {
            b[j].d[9] = sensitivity[0][j];
        }
        let h = self.geometry.x[1] - self.geometry.x[0];
        let (n, p) = flux(a, b, h / s.l, &s, D::var(state.bias, 9));
        let mut current = D::c(AREA * s.j) * (n + p);
        if let Some((accepted, dt)) = old {
            let scale_old = Scale::new(accepted.temp);
            let previous = D::c((accepted.nodes[1][0] - accepted.nodes[0][0]) * scale_old.vt);
            current =
                current - D::c(AREA * EPS / (h * dt)) * ((b[0] - a[0]) * D::c(s.vt) - previous);
        }
        Ok((current.x, current.d[9]))
    }
    pub fn prepare_admittance(&self, state: &State) -> Result<DynamicTerminal, String> {
        self.check_state(state)?;
        let scale = Scale::new(state.temp);
        let (rows, _) = self.assemble(state, None);
        let edges = state
            .nodes
            .windows(2)
            .enumerate()
            .map(|(i, pair)| {
                let a = array::from_fn(|j| D::var(pair[0][j], j));
                let b = array::from_fn(|j| D::var(pair[1][j], j + 3));
                let h = self.geometry.x[i + 1] - self.geometry.x[i];
                let (n, hole) = flux(a, b, h / scale.l, &scale, D::var(state.bias, 9));
                (
                    (D::c(AREA * scale.j) * (n + hole)).d,
                    AREA * EPS * scale.vt / h,
                )
            })
            .collect();
        Ok(DynamicTerminal {
            rows,
            edges,
            vt: scale.vt,
        })
    }
    /// Series resistor coupling demonstrates using condensed terminal
    /// current and Jacobian, with no PDE rows in the circuit system.
    pub fn series(
        &self,
        initial: &State,
        source: f64,
        resistance: f64,
        old: Option<(&State, f64)>,
    ) -> Result<(State, f64, f64, usize), String> {
        if !source.is_finite() || !resistance.is_finite() || resistance <= 0. {
            return Err("invalid series source or resistance".into());
        }
        let mut state = initial.clone();
        let mut bias = state.bias;
        for it in 0..80 {
            state = self.solve(state, bias, initial.temp, old)?.0;
            let (current, g) = self.terminal_stamp(&state, old)?;
            let f = current + (bias - source) / resistance;
            let delta = -f / (g + 1. / resistance);
            if delta.abs() < 1e-11 && f.abs() < 1e-10 {
                return Ok((state, current, g, it));
            }
            bias += delta.clamp(-0.05, 0.05);
        }
        Err(format!("series circuit iteration limit at {source}"))
    }
    pub fn dissipation(&self, state: &State) -> Result<f64, String> {
        self.check_state(state)?;
        let s = Scale::new(state.temp);
        let mut power = 0.;
        for (i, pair) in state.nodes.windows(2).enumerate() {
            let h = (self.geometry.x[i + 1] - self.geometry.x[i]) / s.l;
            let (n, p) = flux(
                pair[0].map(D::c),
                pair[1].map(D::c),
                h,
                &s,
                D::c(state.bias),
            );
            power += AREA
                * s.j
                * s.vt
                * (n.x * (pair[1][1] - pair[0][1]) - p.x * (pair[1][2] - pair[0][2]));
        }
        for i in 1..state.nodes.len() - 1 {
            let vol = (self.geometry.x[i + 1] - self.geometry.x[i - 1]) * 0.5 * AREA;
            let (n, p) = density(state.nodes[i], &s, state.bias);
            let affinity = state.nodes[i][1] + state.nodes[i][2] + state.bias / s.vt;
            let rate = NREF * s.ni * s.ni * affinity.exp_m1() / (s.tau * (n + p + 2. * s.ni));
            power += vol * Q * s.vt * rate * affinity;
        }
        Ok(power)
    }
    pub fn heat_step(
        &self,
        accepted: &State,
        bias: f64,
        dt: f64,
        ambient: f64,
        rth: f64,
        cth: f64,
    ) -> Result<(State, usize, f64, f64), String> {
        if [dt, ambient, rth, cth]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err("invalid thermal parameters".into());
        }
        // Uniform-temperature extension, not a replacement for a heat PDE.
        // Backward Euler temperature equation iterated with carrier solve.
        let mut temp = accepted.temp;
        let mut guess = accepted.clone();
        let mut iterations = 0;
        for _ in 0..30 {
            let (next, n, r) = self.solve(guess, bias, temp, Some((accepted, dt)))?;
            iterations += n;
            let power = self.dissipation(&next)?;
            let target =
                (accepted.temp + (dt / cth) * (power + ambient / rth)) / (1. + dt / (cth * rth));
            if (target - temp).abs() < 1e-9 {
                let residual = cth * (temp - accepted.temp) / dt + (temp - ambient) / rth - power;
                return Ok((next, iterations, r, residual));
            }
            guess = next;
            temp = target;
        }
        Err("thermal iteration limit".into())
    }
    pub fn stored(&self, state: &State) -> Result<(f64, f64, f64), String> {
        self.check_state(state)?;
        let s = Scale::new(state.temp);
        let mut nq = 0.;
        let mut pq = 0.;
        for i in 1..self.geometry.x.len() - 1 {
            let vol = (self.geometry.x[i + 1] - self.geometry.x[i - 1]) * 0.5 * AREA;
            let (n, p) = density(state.nodes[i], &s, state.bias);
            nq += n * Q * NREF * vol;
            pq += p * Q * NREF * vol;
        }
        let left = -EPS * (state.nodes[1][0] - state.nodes[0][0]) * s.vt
            / (self.geometry.x[1] - self.geometry.x[0])
            * AREA;
        Ok((nq, pq, left))
    }
}
