//! Separable Fourier projection of physical Jacobian samples, followed by
//! harmonic convolution. The transform includes all sums and differences
//! of retained indices, regardless of their physical frequency ordering.

use std::f64::consts::TAU;

use crate::primitives::{Complex, Real, SpiceResult, filled_vec};

use super::{HarmonicLattice, failure};

struct Axis {
    input: usize,
    output: usize,
    stride: usize,
    outer: usize,
    weights: Vec<Complex>,
}

impl Axis {
    fn apply(&self, input: &[Complex], output: &mut [Complex]) {
        let center = self.output / 2;
        for outer in 0..self.outer {
            for (mode, weights) in self.weights.chunks_exact(self.input).enumerate() {
                for inner in 0..self.stride {
                    let start = outer * self.input * self.stride + inner;
                    // The constant part cancels analytically at every
                    // nonzero Fourier index, avoiding artificial leakage.
                    let offset = if mode == center {
                        Complex::ZERO
                    } else {
                        input[start]
                    };
                    let mut sum = Complex::ZERO;
                    for (sample, &weight) in weights.iter().enumerate() {
                        let value = input[start + sample * self.stride] - offset;
                        sum = sum + value * weight;
                    }
                    output[(outer * self.output + mode) * self.stride + inner] = sum;
                }
            }
        }
    }
}

pub(super) struct Projection {
    axes: Vec<Axis>,
    modes: Vec<usize>,
    center: usize,
    size: usize,
    cost: usize,
    first: Vec<Complex>,
    second: Vec<Complex>,
}

impl Projection {
    pub(super) fn new(
        lattice: &HarmonicLattice,
        limit: usize,
        storage_limit: usize,
    ) -> SpiceResult<Self> {
        let mut axes = Vec::new();
        let mut stride = 1usize;
        let mut outer = lattice.samples();
        let mut cost = 0usize;
        let mut storage = 0usize;
        for (axis, &samples) in lattice.axes().iter().zip(lattice.grid()) {
            let input = usize::try_from(samples).map_err(|_| failure("phase grid overflow"))?;
            let output = 4 * usize::from(axis.order) + 1;
            outer /= input;
            let entries = input
                .checked_mul(output)
                .ok_or_else(|| failure("projection weight overflow"))?;
            storage = storage
                .checked_add(entries)
                .filter(|&n| n <= storage_limit)
                .ok_or_else(|| failure("projection weight storage limit exceeded"))?;
            let axis_cost = entries
                .checked_mul(stride)
                .and_then(|n| n.checked_mul(outer))
                .ok_or_else(|| failure("projection work overflow"))?;
            cost = cost
                .checked_add(axis_cost)
                .filter(|&n| n <= limit)
                .ok_or_else(|| failure("projection work limit exceeded"))?;
            let mut weights = Vec::new();
            weights
                .try_reserve_exact(entries)
                .map_err(|e| failure(e.to_string()))?;
            let order = 2 * i32::from(axis.order);
            for mode in -order..=order {
                for phase in 0..samples {
                    let angle = -TAU * Real::from(mode) * Real::from(phase) / Real::from(samples);
                    let (sin, cos) = angle.sin_cos();
                    weights.push(Complex::new(
                        cos / Real::from(samples),
                        sin / Real::from(samples),
                    ));
                }
            }
            axes.push(Axis {
                input,
                output,
                stride,
                outer,
                weights,
            });
            stride = stride
                .checked_mul(output)
                .ok_or_else(|| failure("projection size overflow"))?;
        }
        let center = stride / 2;
        let mut modes = vec![center];
        for harmonic in lattice.harmonics() {
            let mut index = 0;
            for (&mode, axis) in harmonic.indices().iter().zip(&axes) {
                let offset = i32::try_from(axis.output / 2)
                    .map_err(|_| failure("projection index overflow"))?;
                let digit = usize::try_from(offset + mode)
                    .map_err(|_| failure("invalid projection index"))?;
                index += digit * axis.stride;
            }
            modes.push(index);
        }
        Ok(Self {
            axes,
            modes,
            center,
            size: stride,
            cost,
            first: filled_vec(lattice.samples(), Complex::ZERO, "projection scratch")?,
            second: filled_vec(lattice.samples(), Complex::ZERO, "projection scratch")?,
        })
    }

    pub(super) const fn cost(&self) -> usize {
        self.cost
    }

    pub(super) fn transform(&mut self, samples: &[Real], result: &mut [Complex]) {
        if samples
            .iter()
            .all(|sample| sample.to_bits() == samples[0].to_bits())
        {
            result.fill(Complex::ZERO);
            result[self.center] = Complex::real(samples[0]);
            return;
        }
        for (value, &sample) in self.first.iter_mut().zip(samples) {
            *value = Complex::real(sample);
        }
        for axis in &self.axes {
            axis.apply(&self.first, &mut self.second);
            std::mem::swap(&mut self.first, &mut self.second);
        }
        result.copy_from_slice(&self.first[..self.size]);
    }

    pub(super) const fn size(&self) -> usize {
        self.size
    }

    /// Flattened (output, input) difference indices in signed-frequency order.
    pub(super) fn signed_differences(&self) -> SpiceResult<Vec<usize>> {
        let mut modes: Vec<_> = self.modes[1..]
            .iter()
            .rev()
            .map(|&mode| 2 * self.center - mode)
            .collect();
        modes.extend(&self.modes);
        let count = modes
            .len()
            .checked_mul(modes.len())
            .ok_or_else(|| failure("sideband difference count overflow"))?;
        let mut differences = filled_vec(count, 0, "sideband difference indices")?;
        for (row, &output) in differences.chunks_exact_mut(modes.len()).zip(&modes) {
            for (value, &input) in row.iter_mut().zip(&modes) {
                *value = output + self.center - input;
            }
        }
        Ok(differences)
    }

    pub(super) fn block(
        &self,
        lattice: &HarmonicLattice,
        conductance: &[Complex],
        capacitance: &[Complex],
        output: &mut [Real],
    ) {
        let m = lattice.coordinates();
        for (k, &out_mode) in self.modes.iter().enumerate() {
            let omega = if k == 0 {
                0.
            } else {
                TAU * lattice.harmonics()[k - 1].frequency()
            };
            let derivative =
                |index: usize| conductance[index] + Complex::imaginary(omega) * capacitance[index];
            let first = derivative(out_mode);
            if k == 0 {
                output[0] = first.re;
            } else {
                output[(2 * k - 1) * m] = 2. * first.re;
                output[2 * k * m] = -2. * first.im;
            }
            for (l, &in_mode) in self.modes.iter().enumerate().skip(1) {
                let minus = derivative(out_mode + self.center - in_mode);
                let plus = derivative(out_mode + in_mode - self.center);
                let cosine = (minus + plus) * Complex::real(0.5);
                let sine = (minus - plus) * Complex::imaginary(-0.5);
                if k == 0 {
                    output[2 * l - 1] = cosine.re;
                    output[2 * l] = sine.re;
                } else {
                    output[(2 * k - 1) * m + 2 * l - 1] = 2. * cosine.re;
                    output[(2 * k - 1) * m + 2 * l] = 2. * sine.re;
                    output[2 * k * m + 2 * l - 1] = -2. * cosine.im;
                    output[2 * k * m + 2 * l] = -2. * sine.im;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::harmonic_balance::{LatticeLimits, ToneAxis};

    #[test]
    fn spectral_convolution_matches_direct_sample_blocks() {
        let axes = [
            ToneAxis {
                frequency: 1.0e6,
                order: 3,
            },
            ToneAxis {
                frequency: 1.013e6,
                order: 2,
            },
            ToneAxis {
                frequency: 0.317e6,
                order: 1,
            },
        ];
        for dimension in 1..=axes.len() {
            let lattice =
                HarmonicLattice::new(&axes[..dimension], 2, LatticeLimits::default()).unwrap();
            let mut projection = Projection::new(&lattice, 250_000_000, 8_000_000).unwrap();
            let m = lattice.coordinates();
            let n = Real::from(u32::try_from(lattice.samples()).unwrap());
            let (g, c): (Vec<_>, Vec<_>) = lattice
                .rows()
                .map(|row| {
                    let x = row[1] + row[2] * 0.5 + row[m - 1] * 0.8;
                    (0.3 + (2. * x).exp(), 1e-12 * (1. + x * x))
                })
                .unzip();
            let mut gs = vec![Complex::ZERO; projection.size()];
            let mut cs = gs.clone();
            projection.transform(&g, &mut gs);
            projection.transform(&c, &mut cs);
            let mut actual = vec![0.; m * m];
            projection.block(&lattice, &gs, &cs, &mut actual);
            let mut expected = vec![0.; m * m];
            for ((basis, &g), &c) in lattice.rows().zip(&g).zip(&c) {
                for (i, row) in expected.chunks_exact_mut(m).enumerate() {
                    let weight = if i == 0 {
                        g / n
                    } else {
                        let k = (i - 1) / 2;
                        let omega = TAU * lattice.harmonics()[k].frequency();
                        let derivative = if i % 2 == 1 {
                            omega * basis[i + 1]
                        } else {
                            -omega * basis[i - 1]
                        };
                        2. * (g * basis[i] + c * derivative) / n
                    };
                    for (v, &b) in row.iter_mut().zip(basis) {
                        *v = weight.mul_add(b, *v);
                    }
                }
            }
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() < 5e-12 * b.abs().max(1.),
                    "axis count {dimension}, entry {i}: {a} vs {b}"
                );
            }
            assert!(projection.cost() * 2 + m * m < lattice.samples() * m * m);
        }
    }
}
