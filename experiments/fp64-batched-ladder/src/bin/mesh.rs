//! Complex FP64 fixed-symbolic sparse LDL^T prototype for grounded RC meshes.
//! Deliberately isolated: no SPICE device load or general pivoting yet.
#![feature(portable_simd)]
#![forbid(unsafe_code)]

use anyhow::{Result, ensure};
use fp64_batched_ladder::{Gpu, median_ms};
use rayon::prelude::*;
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    hint::black_box,
    ops::{Add, Div, Mul, Neg, Sub},
    simd::Simd,
};

const SIDE: usize = 8;
const N: usize = SIDE * SIDE;
const TILE: usize = 64;
const BLOCK: usize = TILE * 2;
const PIVOT: u32 = 0;
const MULTIPLIER: u32 = 1;
const UPDATE: u32 = 2;
const STORE: u32 = 3;
const BACK: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
struct C<T> {
    re: T,
    im: T,
}

impl<T> C<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
    // Well-scaled matrices only. General SPICE needs scaled complex division
    // and a lane-specific conditioning/pivot failure policy.
    fn inverse(self) -> Self {
        let norm = self.re * self.re + self.im * self.im;
        Self {
            re: self.re / norm,
            im: -self.im / norm,
        }
    }
}

impl C<f64> {
    const ZERO: Self = Self { re: 0., im: 0. };
    fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }
}

struct Plan {
    entries: Vec<(usize, usize)>,
    original: usize,
    tape: Vec<[u32; 4]>,
    updates: usize,
}

fn entry(
    entries: &mut Vec<(usize, usize)>,
    indices: &mut BTreeMap<(usize, usize), u32>,
    i: usize,
    j: usize,
) -> u32 {
    *indices.entry((i, j)).or_insert_with(|| {
        let index = u32::try_from(entries.len()).expect("small mesh plan");
        entries.push((i, j));
        index
    })
}

impl Plan {
    /// Group typed column/neighbor/update ranges instead of interpreting
    /// thousands of heterogeneous operations. The original matrix's row
    /// adjacency is separate from the filled factors used for correction.
    fn grouped(&self) -> Vec<[u32; 4]> {
        let mut groups = vec![[0; 4]; 1 + N];
        let mut cursor = 1;
        for k in 0..N {
            let [kind, diag, row, _] = self.tape[cursor];
            assert_eq!((kind, row), (PIVOT, k as u32));
            cursor += 1;
            let mut neighbors = Vec::new();
            let update_start = groups.len() as u32;
            while cursor < self.tape.len() && self.tape[cursor][0] == MULTIPLIER {
                let [_, handle, row, _] = self.tape[cursor];
                cursor += 1;
                let start = groups.len() as u32;
                while self.tape[cursor][0] == UPDATE {
                    let [_, dest, source, _] = self.tape[cursor];
                    groups.push([dest, source, neighbors.len() as u32, 0]);
                    cursor += 1;
                }
                assert_eq!(self.tape[cursor], [STORE, handle, 0, 0]);
                cursor += 1;
                neighbors.push([row, handle, start, groups.len() as u32]);
            }
            let start = groups.len() as u32;
            groups.extend(neighbors);
            groups[k + 1] = [diag, start, groups.len() as u32, update_start];
        }
        // Reserve one range per original row; every entry in a row lists
        // the other column and its original lower-triangle coefficient.
        let residual = groups.len();
        groups.resize(residual + N, [0; 4]);
        for row in 0..N {
            let start = groups.len() as u32;
            for (handle, &(i, j)) in self.entries[..self.original].iter().enumerate() {
                if i == row {
                    groups.push([j as u32, handle as u32, 0, 0]);
                } else if j == row {
                    groups.push([i as u32, handle as u32, 0, 0]);
                }
            }
            groups[residual + row] = [start, groups.len() as u32, 0, 0];
        }
        groups[0] = [
            self.entries.len() as u32,
            N as u32,
            self.original as u32,
            residual as u32,
        ];
        groups
    }

    fn new() -> Self {
        let mut entries = Vec::new();
        let mut indices = BTreeMap::new();
        // Initial lower triangle of a symmetric nodal conductance matrix.
        for i in 0..N {
            entry(&mut entries, &mut indices, i, i);
        }
        for i in 0..N {
            if i % SIDE != 0 {
                entry(&mut entries, &mut indices, i, i - 1);
            }
            if i >= SIDE {
                entry(&mut entries, &mut indices, i, i - SIDE);
            }
        }
        let original = entries.len();
        let mut columns = Vec::with_capacity(N);
        let mut tape = vec![[0; 4]]; // GPU header filled after symbolic elimination.
        let mut updates = 0;
        for k in 0..N {
            let neighbours: Vec<_> = (k + 1..N)
                .filter_map(|i| indices.get(&(i, k)).map(|&handle| (i, handle)))
                .collect();
            tape.push([PIVOT, indices[&(k, k)], k as u32, 0]);
            for (pos, &(i, ik)) in neighbours.iter().enumerate() {
                tape.push([MULTIPLIER, ik, i as u32, 0]);
                // j>=i: matrix stores the symmetric lower triangle only.
                // A_jk remains unscaled until its own MULTIPLIER step.
                for &(j, jk) in &neighbours[pos..] {
                    let target = entry(&mut entries, &mut indices, j, i);
                    tape.push([UPDATE, target, jk, 0]);
                    updates += 1;
                }
                tape.push([STORE, ik, 0, 0]);
            }
            columns.push(neighbours);
        }
        for (k, neighbours) in columns.into_iter().enumerate().rev() {
            for (i, ik) in neighbours {
                tape.push([BACK, ik, i as u32, k as u32]);
            }
        }
        tape[0] = [entries.len() as u32, N as u32, tape.len() as u32, 0];
        Self {
            entries,
            original,
            tape,
            updates,
        }
    }
}

// Shared pure arithmetic, different packing across independent systems.
trait Packed: Copy {
    fn zero() -> Self;
    fn load(data: &[f64], index: usize, lane: usize) -> Self;
    fn store(self, data: &mut [f64], index: usize, lane: usize);
    fn mul(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn inverse(self) -> Self;
}

impl Packed for C<f64> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn load(data: &[f64], index: usize, lane: usize) -> Self {
        let i = index * BLOCK + lane;
        Self {
            re: data[i],
            im: data[i + TILE],
        }
    }
    fn store(self, data: &mut [f64], index: usize, lane: usize) {
        let i = index * BLOCK + lane;
        data[i] = self.re;
        data[i + TILE] = self.im;
    }
    fn mul(self, b: Self) -> Self {
        C::mul(self, b)
    }
    fn sub(self, b: Self) -> Self {
        C::sub(self, b)
    }
    fn inverse(self) -> Self {
        C::inverse(self)
    }
}

type V = Simd<f64, 4>;
const VECTORS: usize = TILE / 4;
#[derive(Clone, Copy)]
struct SimdBlock([C<V>; VECTORS]);

impl Packed for SimdBlock {
    fn zero() -> Self {
        Self(
            [C {
                re: V::splat(0.),
                im: V::splat(0.),
            }; VECTORS],
        )
    }
    fn load(data: &[f64], index: usize, _lane: usize) -> Self {
        let block = &data[index * BLOCK..(index + 1) * BLOCK];
        let (re, im) = block.split_at(TILE);
        Self(std::array::from_fn(|i| C {
            re: V::from_slice(&re[i * 4..i * 4 + 4]),
            im: V::from_slice(&im[i * 4..i * 4 + 4]),
        }))
    }
    fn store(self, data: &mut [f64], index: usize, _lane: usize) {
        let block = &mut data[index * BLOCK..(index + 1) * BLOCK];
        let (re, im) = block.split_at_mut(TILE);
        for (i, v) in self.0.into_iter().enumerate() {
            v.re.copy_to_slice(&mut re[i * 4..i * 4 + 4]);
            v.im.copy_to_slice(&mut im[i * 4..i * 4 + 4]);
        }
    }
    fn mul(self, b: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].mul(b.0[i])))
    }
    fn sub(self, b: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].sub(b.0[i])))
    }
    fn inverse(self) -> Self {
        Self(self.0.map(C::inverse))
    }
}

fn execute<T: Packed>(tape: &[[u32; 4]], matrix: &mut [f64], rhs: &mut [f64], lane: usize) {
    let mut inverse = T::zero();
    let mut pivot_rhs = T::zero();
    let mut multiplier = T::zero();
    for &[kind, a, b, c] in &tape[1..] {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        match kind {
            PIVOT => {
                inverse = T::load(matrix, a, lane).inverse();
                pivot_rhs = T::load(rhs, b, lane);
                inverse.mul(pivot_rhs).store(rhs, b, lane);
            }
            MULTIPLIER => {
                multiplier = T::load(matrix, a, lane).mul(inverse);
                T::load(rhs, b, lane)
                    .sub(multiplier.mul(pivot_rhs))
                    .store(rhs, b, lane);
            }
            UPDATE => {
                T::load(matrix, a, lane)
                    .sub(multiplier.mul(T::load(matrix, b, lane)))
                    .store(matrix, a, lane);
            }
            STORE => multiplier.store(matrix, a, lane),
            BACK => T::load(rhs, c, lane)
                .sub(T::load(matrix, a, lane).mul(T::load(rhs, b, lane)))
                .store(rhs, c, lane),
            _ => unreachable!("validated symbolic plan"),
        }
    }
}

// Match the GPU's grouped factorization and reusable triangular solves.
// The original interpreter remains as a separately benchmarked baseline.
fn factor_grouped<T: Packed>(plan: &[[u32; 4]], matrix: &mut [f64], lane: usize) {
    for k in 0..N {
        let [diag, first, end, _] = plan[1 + k];
        let inverse = T::load(matrix, diag as usize, lane).inverse();
        inverse.store(matrix, diag as usize, lane);
        for &[_, handle, start, stop] in &plan[first as usize..end as usize] {
            let l = T::load(matrix, handle as usize, lane).mul(inverse);
            for &[dest, source, _, _] in &plan[start as usize..stop as usize] {
                T::load(matrix, dest as usize, lane)
                    .sub(l.mul(T::load(matrix, source as usize, lane)))
                    .store(matrix, dest as usize, lane);
            }
            l.store(matrix, handle as usize, lane);
        }
    }
}

fn solve_grouped<T: Packed>(plan: &[[u32; 4]], matrix: &[f64], rhs: &mut [f64], lane: usize) {
    for k in 0..N {
        let [diag, first, end, _] = plan[1 + k];
        let y = T::load(rhs, k, lane);
        for &[row, handle, _, _] in &plan[first as usize..end as usize] {
            T::load(rhs, row as usize, lane)
                .sub(T::load(matrix, handle as usize, lane).mul(y))
                .store(rhs, row as usize, lane);
        }
        T::load(matrix, diag as usize, lane)
            .mul(y)
            .store(rhs, k, lane);
    }
    for k in (0..N).rev() {
        let [_, first, end, _] = plan[1 + k];
        let mut value = T::load(rhs, k, lane);
        for &[row, handle, _, _] in &plan[first as usize..end as usize] {
            value = value.sub(T::load(matrix, handle as usize, lane).mul(T::load(
                rhs,
                row as usize,
                lane,
            )));
        }
        value.store(rhs, k, lane);
    }
}

struct Input {
    matrix: Vec<f64>,
    rhs: Vec<f64>,
    expected: Vec<f64>,
}

impl Input {
    fn new(plan: &Plan, tiles: usize) -> Self {
        let mut matrix = vec![0.; tiles * plan.entries.len() * BLOCK];
        let mut rhs = vec![0.; tiles * N * BLOCK];
        let mut expected = vec![0.; rhs.len()];
        for (t, ((matrix, rhs), expected)) in matrix
            .chunks_exact_mut(plan.entries.len() * BLOCK)
            .zip(rhs.chunks_exact_mut(N * BLOCK))
            .zip(expected.chunks_exact_mut(N * BLOCK))
            .enumerate()
        {
            for lane in 0..TILE {
                let frequency =
                    0.001 * 10.0_f64.powf(4.0 * ((t * TILE + lane) % 4096) as f64 / 4095.);
                for i in 0..N {
                    let degree = usize::from(i % SIDE > 0)
                        + usize::from(i % SIDE + 1 < SIDE)
                        + usize::from(i >= SIDE)
                        + usize::from(i + SIDE < N);
                    let d = C {
                        re: degree as f64 + 0.125 + ((i * 17 + lane * 3 + t) % 31) as f64 / 31.,
                        im: frequency * (0.5 + ((i * 11 + lane) % 13) as f64 / 13.),
                    };
                    let x = C {
                        re: ((t * 19 + i * 11 + lane * 17) % 97) as f64 / 23. - 2.,
                        im: ((t * 7 + i * 3 + lane * 13) % 83) as f64 / 19. - 2.,
                    };
                    d.store(matrix, i, lane);
                    x.store(expected, i, lane);
                    d.mul(x).store(rhs, i, lane);
                }
                for (index, &(i, j)) in plan.entries[..plan.original].iter().enumerate().skip(N) {
                    C { re: -1., im: 0. }.store(matrix, index, lane);
                    C::load(rhs, i, lane)
                        .sub(C::load(expected, j, lane))
                        .store(rhs, i, lane);
                    C::load(rhs, j, lane)
                        .sub(C::load(expected, i, lane))
                        .store(rhs, j, lane);
                }
            }
        }
        Self {
            matrix,
            rhs,
            expected,
        }
    }

    fn check(&self, plan: &Plan, x: &[f64]) -> Result<(f64, f64)> {
        ensure!(x.len() == self.expected.len());
        let mut error = 0.0_f64;
        let mut backward = 0.0_f64;
        for (((m, b), expected), x) in self
            .matrix
            .chunks_exact(plan.entries.len() * BLOCK)
            .zip(self.rhs.chunks_exact(N * BLOCK))
            .zip(self.expected.chunks_exact(N * BLOCK))
            .zip(x.chunks_exact(N * BLOCK))
        {
            for lane in 0..TILE {
                let mut residual = [C::ZERO; N];
                let mut norm = [0.; N];
                for i in 0..N {
                    let v = C::load(x, i, lane);
                    let wanted = C::load(expected, i, lane);
                    let rhs = C::load(b, i, lane);
                    ensure!(v.re.is_finite() && v.im.is_finite(), "nonfinite x");
                    error = error.max(v.sub(wanted).magnitude());
                    residual[i] = rhs;
                    norm[i] = rhs.magnitude();
                }
                for (index, &(i, j)) in plan.entries[..plan.original].iter().enumerate() {
                    let a = C::load(m, index, lane);
                    let xj = C::load(x, j, lane);
                    residual[i] = residual[i].sub(a.mul(xj));
                    norm[i] += a.magnitude() * xj.magnitude();
                    if i != j {
                        let xi = C::load(x, i, lane);
                        residual[j] = residual[j].sub(a.mul(xi));
                        norm[j] += a.magnitude() * xi.magnitude();
                    }
                }
                for i in 0..N {
                    ensure!(
                        residual[i].re.is_finite()
                            && residual[i].im.is_finite()
                            && norm[i].is_finite(),
                        "nonfinite residual"
                    );
                    backward =
                        backward.max(residual[i].magnitude() / norm[i].max(f64::MIN_POSITIVE));
                }
            }
        }
        ensure!(
            error <= 2e-11 && backward <= 2e-13,
            "solution error {error:e}, backward error {backward:e}"
        );
        Ok((error, backward))
    }
}

fn solve_tile(
    plan: &Plan,
    original: &[f64],
    b: &[f64],
    m: &mut [f64],
    x: &mut [f64],
    vector: bool,
) {
    m.copy_from_slice(original);
    x.copy_from_slice(b);
    if vector {
        execute::<SimdBlock>(&plan.tape, m, x, 0)
    } else {
        for lane in 0..TILE {
            execute::<C<f64>>(&plan.tape, m, x, lane)
        }
    }
}

fn grouped_shader(mixed: bool) -> String {
    let template = include_str!("../mesh_grouped.wgsl");
    let refinement = if mixed {
        include_str!("../mesh_refine.wgsl")
    } else {
        ""
    };
    // Use concrete scalar types and literal conversions in the generated
    // shader, but keep both factor/solve algorithms identical.
    template
        .replace("FACTOR_SCALAR", if mixed { "f32" } else { "f64" })
        .replace("REFINEMENT_BODY", refinement)
}

/// Compile the validated fixed symbolic schedule into a kernel. Declare
/// numeric entries on first use; eliminated values are dead except for the
/// saved lower factors used in the backsolve. This lets the shader compiler
/// retain the active numerical frontier in registers, without a global
/// factor read/write per Schur update or any runtime operation dispatch.
fn specialized_shader(plan: &Plan) -> String {
    fn matrix_decl(code: &mut String, declared: &mut [bool], index: usize, original: usize) {
        if !std::mem::replace(&mut declared[index], true) {
            if index < original {
                writeln!(code, "var a{index}=read_a(mb+{index}u*128u);").unwrap();
            } else {
                writeln!(code, "var a{index}=Z(0.0lf,0.0lf);").unwrap();
            }
        }
    }
    fn rhs_decl(code: &mut String, declared: &mut [bool], row: usize) {
        if !std::mem::replace(&mut declared[row], true) {
            writeln!(code, "var b{row}=read_b(xb+{row}u*128u);").unwrap();
        }
    }
    let mut body = String::new();
    let mut matrix = vec![false; plan.entries.len()];
    let mut rhs = [false; N];
    let mut pivot = 0;
    let mut handle = 0;
    let mut backward_row = None;
    for &[kind, a, b, c] in &plan.tape[1..] {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        match kind {
            PIVOT => {
                pivot = b;
                matrix_decl(&mut body, &mut matrix, a, plan.original);
                rhs_decl(&mut body, &mut rhs, b);
                writeln!(
                    body,
                    "let inv{b}=inverse(a{a}); put_x(xb+{b}u*128u,mul(inv{b},b{b}));"
                )
                .unwrap();
            }
            MULTIPLIER => {
                handle = a;
                matrix_decl(&mut body, &mut matrix, a, plan.original);
                rhs_decl(&mut body, &mut rhs, b);
                writeln!(
                    body,
                    "let l{a}=mul(a{a},inv{pivot}); b{b}-=mul(l{a},b{pivot});"
                )
                .unwrap();
            }
            UPDATE => {
                matrix_decl(&mut body, &mut matrix, a, plan.original);
                matrix_decl(&mut body, &mut matrix, b, plan.original);
                writeln!(body, "a{a}-=mul(l{handle},a{b});").unwrap();
            }
            STORE => {
                assert_eq!(handle, a);
                writeln!(body, "put_m(mb+{a}u*128u,l{a});").unwrap();
            }
            BACK => {
                if backward_row != Some(c) {
                    if let Some(row) = backward_row {
                        writeln!(body, "put_x(xb+{row}u*128u,x{row});").unwrap();
                    }
                    writeln!(body, "var x{c}=read_x(xb+{c}u*128u);").unwrap();
                    backward_row = Some(c);
                }
                writeln!(
                    body,
                    "x{c}-=mul(read_m(mb+{a}u*128u),read_x(xb+{b}u*128u));"
                )
                .unwrap();
            }
            _ => unreachable!("validated symbolic instruction"),
        }
    }
    if let Some(row) = backward_row {
        writeln!(body, "put_x(xb+{row}u*128u,x{row});").unwrap();
    }
    include_str!("../mesh_specialized.wgsl")
        .replace("ENTRY_COUNT", &plan.entries.len().to_string())
        .replace("ROW_COUNT", &N.to_string())
        .replace("NUMERIC_BODY", &body)
}

fn main() -> Result<()> {
    let plan = Plan::new();
    println!(
        "mesh: {SIDE}x{SIDE}, nodes={N}, original lower entries={}, filled lower entries={}, updates={}, tape operations={}",
        plan.original,
        plan.entries.len(),
        plan.updates,
        plan.tape.len() - 1
    );
    let gpu = pollster::block_on(Gpu::new(
        "Complex FP64 RC mesh",
        include_str!("../mesh.wgsl"),
        0,
    ))?;
    let grouped = plan.grouped();
    let grouped_gpu =
        pollster::block_on(Gpu::new("Grouped complex FP64", &grouped_shader(false), 0))?;
    let mixed_gpu = pollster::block_on(Gpu::new(
        "Grouped FP32 with FP64 refinement",
        &grouped_shader(true),
        0,
    ))?;
    const LOCAL_SYSTEMS: usize = 4;
    let max_neighbours = grouped[1..=N]
        .iter()
        .map(|col| col[2] - col[1])
        .max()
        .unwrap() as usize;
    let local_bytes =
        ((plan.entries.len() + N + max_neighbours) * LOCAL_SYSTEMS * size_of::<[f64; 2]>()) as u32;
    let local_source = include_str!("../mesh_local.wgsl")
        .replace(
            "MATRIX_CELLS",
            &(plan.entries.len() * LOCAL_SYSTEMS).to_string(),
        )
        .replace("RHS_CELLS", &(N * LOCAL_SYSTEMS).to_string())
        .replace(
            "MULTIPLIER_CELLS",
            &(max_neighbours * LOCAL_SYSTEMS).to_string(),
        )
        .replace("ROW_COUNT", &N.to_string())
        .replace("ENTRY_COUNT", &plan.entries.len().to_string());
    let local_gpu = pollster::block_on(Gpu::new(
        "Workgroup-resident complex FP64",
        &local_source,
        local_bytes,
    ))?;
    let generated = specialized_shader(&plan);
    println!("generated shader bytes: {}", generated.len());
    let specialized_gpu = pollster::block_on(Gpu::new("Specialized sparse FP64", &generated, 0))?;
    println!("local shared memory bytes: {local_bytes}");
    println!("persistent CPU workers: {}", rayon::current_num_threads());
    println!(
        "systems,scalar_ms,simd_ms,parallel_simd_ms,grouped_simd_ms,grouped_parallel_simd_ms,gpu_resident_ms,gpu_amortized_ms,gpu_roundtrip_ms,grouped_resident_ms,grouped_amortized_ms,grouped_roundtrip_ms,mixed_resident_ms,mixed_amortized_ms,mixed_roundtrip_ms,mixed_error,mixed_backward,local_resident_ms,local_amortized_ms,local_roundtrip_ms,specialized_resident_ms,specialized_amortized_ms,specialized_roundtrip_ms,tape_kernel_ms,grouped_kernel_ms,mixed_kernel_ms,local_kernel_ms,specialized_kernel_ms,max_solution_error,max_backward_error"
    );
    for groups in [1_u32, 4, 16, 64] {
        let input = Input::new(&plan, groups as usize);
        let stride = plan.entries.len() * BLOCK;
        let mut matrix = vec![0.; input.matrix.len()];
        let mut x = vec![0.; input.rhs.len()];
        let mut error = 0.0_f64;
        let mut backward = 0.0_f64;
        let mut times = Vec::new();
        for mode in 0..5 {
            times.push(median_ms(|| {
                if mode >= 3 {
                    let solve = |m: &[f64], b: &[f64], work: &mut [f64], x: &mut [f64]| {
                        work.copy_from_slice(m);
                        x.copy_from_slice(b);
                        factor_grouped::<SimdBlock>(&grouped, work, 0);
                        solve_grouped::<SimdBlock>(&grouped, work, x, 0);
                    };
                    if mode == 4 {
                        input
                            .matrix
                            .par_chunks_exact(stride)
                            .zip(input.rhs.par_chunks_exact(N * BLOCK))
                            .zip(matrix.par_chunks_exact_mut(stride))
                            .zip(x.par_chunks_exact_mut(N * BLOCK))
                            .for_each(|(((m, b), work), x)| solve(m, b, work, x));
                    } else {
                        for (((m, b), work), x) in input
                            .matrix
                            .chunks_exact(stride)
                            .zip(input.rhs.chunks_exact(N * BLOCK))
                            .zip(matrix.chunks_exact_mut(stride))
                            .zip(x.chunks_exact_mut(N * BLOCK))
                        {
                            solve(m, b, work, x);
                        }
                    }
                } else if mode == 2 {
                    input
                        .matrix
                        .par_chunks_exact(stride)
                        .zip(input.rhs.par_chunks_exact(N * BLOCK))
                        .zip(matrix.par_chunks_exact_mut(stride))
                        .zip(x.par_chunks_exact_mut(N * BLOCK))
                        .for_each(|(((m, b), work), x)| solve_tile(&plan, m, b, work, x, true));
                } else {
                    for (((m, b), work), x) in input
                        .matrix
                        .chunks_exact(stride)
                        .zip(input.rhs.chunks_exact(N * BLOCK))
                        .zip(matrix.chunks_exact_mut(stride))
                        .zip(x.chunks_exact_mut(N * BLOCK))
                    {
                        solve_tile(&plan, m, b, work, x, mode == 1);
                    }
                }
                black_box(&x);
                Ok(())
            })?);
            let (e, r) = input.check(&plan, &x)?;
            error = error.max(e);
            backward = backward.max(r);
        }
        let (resident, amortized, roundtrip, k) = gpu.trial(
            &[&input.matrix, &input.rhs],
            &[matrix.len(), x.len()],
            Some(&plan.tape),
            &mut x,
            groups,
        )?;
        let (e, r) = input.check(&plan, &x)?;
        let (gr, ga, gt, gk) = grouped_gpu.trial(
            &[&input.matrix, &input.rhs],
            &[matrix.len(), x.len(), x.len()],
            Some(&grouped),
            &mut x,
            groups,
        )?;
        let (ge, gb) = input.check(&plan, &x)?;
        // Factor and correction storage uses f32; the runner sizes workspace
        // in f64 units, so allocate exactly half as many units.
        let (mr, ma, mt, mk) = mixed_gpu.trial(
            &[&input.matrix, &input.rhs],
            &[matrix.len() / 2, x.len() / 2, x.len()],
            Some(&grouped),
            &mut x,
            groups,
        )?;
        let (me, mb) = input.check(&plan, &x)?;
        let (lr, la, lt, lk) = local_gpu.trial(
            &[&input.matrix, &input.rhs],
            &[x.len()],
            Some(&grouped),
            &mut x,
            groups * (TILE / LOCAL_SYSTEMS) as u32,
        )?;
        let (le, lb) = input.check(&plan, &x)?;
        let (sr, sa, st, sk) = specialized_gpu.trial(
            &[&input.matrix, &input.rhs],
            &[matrix.len(), x.len()],
            None,
            &mut x,
            groups,
        )?;
        let (se, sb) = input.check(&plan, &x)?;
        println!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{resident:.4},{amortized:.4},{roundtrip:.4},{gr:.4},{ga:.4},{gt:.4},{mr:.4},{ma:.4},{mt:.4},{me:.3e},{mb:.3e},{lr:.4},{la:.4},{lt:.4},{sr:.4},{sa:.4},{st:.4},{k:.4},{gk:.4},{mk:.4},{lk:.4},{sk:.4},{:.3e},{:.3e}",
            groups * TILE as u32,
            times[0],
            times[1],
            times[2],
            times[3],
            times[4],
            error.max(e).max(ge).max(me).max(le).max(se),
            backward.max(r).max(gb).max(mb).max(lb).max(sb)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooperative_column_updates_are_disjoint() {
        use std::collections::BTreeSet;
        let plan = Plan::new();
        let grouped = plan.grouped();
        let mut update_count = 0;
        for k in 0..N {
            let [diag, first, end, update_start] = grouped[k + 1];
            assert_eq!(plan.entries[diag as usize], (k, k));
            let edges = &grouped[first as usize..end as usize];
            let column_handles: BTreeSet<_> = edges.iter().map(|edge| edge[1]).collect();
            assert_eq!(column_handles.len(), edges.len());
            let mut destinations = BTreeSet::new();
            let mut cursor = update_start;
            for (i, &[row, handle, start, stop]) in edges.iter().enumerate() {
                assert_eq!(plan.entries[handle as usize], (row as usize, k));
                assert_eq!(cursor, start);
                for &[dest, source, multiplier, _] in &grouped[start as usize..stop as usize] {
                    assert_eq!(multiplier as usize, i);
                    assert!(column_handles.contains(&source));
                    assert!(!column_handles.contains(&dest));
                    assert_ne!(dest, diag);
                    assert!(
                        destinations.insert(dest),
                        "two updates write the same destination"
                    );
                    let (j, other_k) = plan.entries[source as usize];
                    assert_eq!(other_k, k);
                    assert_eq!(plan.entries[dest as usize], (j, row as usize));
                    update_count += 1;
                }
                cursor = stop;
            }
            assert_eq!(cursor, first);
        }
        assert_eq!(update_count, plan.updates);
    }

    #[test]
    fn grouped_plan_factors_can_solve_repeated_rhs() -> Result<()> {
        let plan = Plan::new();
        let grouped = plan.grouped();
        let input = Input::new(&plan, 1);
        let mut factors = input.matrix.clone();
        factor_grouped::<SimdBlock>(&grouped, &mut factors, 0);
        let mut output = input.rhs.clone();
        solve_grouped::<SimdBlock>(&grouped, &factors, &mut output, 0);
        input.check(&plan, &output)?;
        let mut second = input.rhs.clone();
        solve_grouped::<SimdBlock>(&grouped, &factors, &mut second, 0);
        assert_eq!(output, second);
        // Also test the same typed plan in scalar mode for every lane.
        let mut scalar_factors = input.matrix.clone();
        let mut scalar_result = input.rhs.clone();
        for lane in 0..TILE {
            factor_grouped::<C<f64>>(&grouped, &mut scalar_factors, lane);
            solve_grouped::<C<f64>>(&grouped, &scalar_factors, &mut scalar_result, lane);
        }
        input.check(&plan, &scalar_result)?;
        Ok(())
    }
}
