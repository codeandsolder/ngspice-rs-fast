//! Complex FP64 fixed-symbolic sparse LDL^T prototype for grounded RC meshes.
//! Deliberately isolated: no SPICE device load or general pivoting yet.
#![feature(portable_simd)]
#![forbid(unsafe_code)]

use anyhow::{Result, ensure};
use fp64_batched_ladder::{Gpu, median_ms};
use rayon::prelude::*;
use std::{
    collections::BTreeMap,
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
    ))?;
    println!("persistent CPU workers: {}", rayon::current_num_threads());
    println!(
        "systems,scalar_ms,simd_ms,parallel_simd_ms,gpu_resident_ms,gpu_amortized_ms,gpu_roundtrip_ms,max_solution_error,max_backward_error"
    );
    for groups in [1_u32, 4, 16, 64] {
        let input = Input::new(&plan, groups as usize);
        let stride = plan.entries.len() * BLOCK;
        let mut matrix = vec![0.; input.matrix.len()];
        let mut x = vec![0.; input.rhs.len()];
        let mut error = 0.0_f64;
        let mut backward = 0.0_f64;
        let mut times = Vec::new();
        for mode in 0..3 {
            times.push(median_ms(|| {
                if mode == 2 {
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
        let (resident, amortized, roundtrip) = gpu.trial(
            &[&input.matrix, &input.rhs],
            &[matrix.len(), x.len()],
            Some(&plan.tape),
            &mut x,
            groups,
        )?;
        let (e, r) = input.check(&plan, &x)?;
        println!(
            "{},{:.4},{:.4},{:.4},{resident:.4},{amortized:.4},{roundtrip:.4},{:.3e},{:.3e}",
            groups * TILE as u32,
            times[0],
            times[1],
            times[2],
            error.max(e),
            backward.max(r)
        );
    }
    Ok(())
}
