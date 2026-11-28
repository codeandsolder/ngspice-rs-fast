//! Bounded sparse SPICE2 polynomial lowering. No generated expression is
//! cloned or allocated until the complete output tree has been budgeted.
use super::forms::{Numeric, PolyControls, PolyForm};
use super::{check_coefficient, generated, instance, malformed, quantity};
use crate::netlist::ast::{DeviceInstance, ParameterAssignment};
use crate::netlist::bexpr::{BBinaryOp, BExpr, BExprKind};
use crate::netlist::expr::{MAX_TREE_NODES, TreeBudget};
use crate::primitives::{SourceLoc, SpiceResult};

/// The `nxtpwr` order from spice2poly, represented as sorted nonzero runs.
/// Each advance changes at most two runs, regardless of dimension or degree.
struct PolyPowers {
    dimension: usize,
    degree: u32,
    powers: Vec<(usize, u32)>,
}

impl PolyPowers {
    fn new(dimension: usize) -> Option<Self> {
        (dimension != 0).then(|| Self {
            dimension,
            degree: 0,
            powers: Vec::new(),
        })
    }

    fn advance(&mut self) -> Option<&[(usize, u32)]> {
        let tail = if self
            .powers
            .last()
            .is_some_and(|&(index, _)| index == self.dimension - 1)
        {
            self.powers.pop()?.1
        } else {
            0
        };
        if let Some((index, power)) = self.powers.last_mut() {
            let next = *index + 1;
            *power -= 1;
            if *power == 0 {
                self.powers.pop();
            }
            self.powers.push((next, tail.checked_add(1)?));
        } else {
            self.degree = self.degree.checked_add(1)?;
            self.powers.push((0, self.degree));
        }
        Some(&self.powers)
    }
}

/// Visit only nonzero literals and unresolved expressions, in source order.
/// Skipped coefficients still advance the sparse cursor exactly once.
fn terms(
    device: &DeviceInstance,
    form: &PolyForm<'_>,
    mut visit: impl FnMut(&Numeric<'_>, &PolyPowers) -> SpiceResult<()>,
) -> SpiceResult<()> {
    let mut powers = PolyPowers::new(form.dimension)
        .ok_or_else(|| malformed(device, "POLY needs a positive integer dimension"))?;
    for (index, coefficient) in form.coefficients.iter().enumerate() {
        if index != 0 {
            powers
                .advance()
                .ok_or_else(|| malformed(device, "POLY degree exceeds supported range"))?;
        }
        if !coefficient.literal_zero() {
            visit(coefficient, &powers)?;
        }
    }
    Ok(())
}

/// Height in edges of a count-balanced binary tree.
const fn height(count: usize) -> usize {
    if count <= 1 {
        0
    } else {
        (count - 1).bit_width() as usize
    }
}

/// Exact depth of leaf `index` for the split used in `balanced`.
const fn leaf_depth(mut index: usize, mut count: usize) -> usize {
    let mut depth = 1;
    while count > 1 {
        let left = count / 2;
        if index < left {
            count = left;
        } else {
            index -= left;
            count -= left;
        }
        depth += 1;
    }
    depth
}

struct Plan {
    used: Vec<bool>,
    count: usize,
}

impl Plan {
    fn prepare(device: &DeviceInstance, form: &PolyForm<'_>) -> SpiceResult<Self> {
        // Bound cursor traversal independently of the size of the retained AST.
        let mut work = TreeBudget::new(MAX_TREE_NODES);
        work.charge(1, form.coefficients.len(), &device.location)?;
        let mut used = vec![false; form.dimension];
        let mut count = 0;
        terms(device, form, |coefficient, powers| {
            work.charge(1, powers.powers.len(), &coefficient.parameter.location)?;
            for &(index, _) in &powers.powers {
                used[index] = true;
            }
            count += 1;
            Ok(())
        })?;
        // One zero-times-quantity anchor per otherwise unused control. These
        // retain the original binding checks, including invalid branch names,
        // without ever computing omitted high powers.
        count += used.iter().filter(|&&value| !value).count();
        let plan = Self { used, count };
        plan.check_tree(device, form)?;
        Ok(plan)
    }

    fn check_tree(&self, device: &DeviceInstance, form: &PolyForm<'_>) -> SpiceResult<()> {
        let mut budget = TreeBudget::new(MAX_TREE_NODES);
        budget.charge(height(self.count), self.count - 1, &device.location)?;
        let mut index = 0;
        terms(device, form, |coefficient, powers| {
            let depth = leaf_depth(index, self.count);
            index += 1;
            if powers.degree == 0 {
                check_coefficient(coefficient.parameter, depth, &mut budget)?;
            } else {
                let degree = powers.degree as usize;
                let location = &coefficient.parameter.location;
                budget.charge(depth, 1, location)?; // coefficient times product
                check_coefficient(coefficient.parameter, depth + 1, &mut budget)?;
                let nodes = degree
                    .checked_mul(2)
                    .and_then(|n| n.checked_sub(1))
                    .ok_or_else(|| malformed(device, "POLY product node limit exceeded"))?;
                budget.charge(depth + 1 + height(degree), nodes, location)?;
            }
            Ok(())
        })?;
        for _ in self.used.iter().filter(|&&value| !value) {
            budget.charge(leaf_depth(index, self.count) + 1, 3, &device.location)?;
            index += 1;
        }
        Ok(())
    }
}

fn binary(op: BBinaryOp, lhs: BExpr, rhs: BExpr, location: &SourceLoc) -> BExpr {
    quantity(
        BExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        location,
    )
}

/// Preserve leaf order without a linear-depth fold. Recursion is logarithmic
/// and leaves are consumed exactly once, including cloned quantity nodes.
fn balanced(
    values: &mut impl Iterator<Item = BExpr>,
    count: usize,
    op: BBinaryOp,
    location: &SourceLoc,
) -> Option<BExpr> {
    match count {
        0 => None,
        1 => values.next(),
        _ => {
            let left = count / 2;
            Some(binary(
                op,
                balanced(values, left, op, location)?,
                balanced(values, count - left, op, location)?,
                location,
            ))
        }
    }
}

fn controls(form: &PolyForm<'_>) -> Vec<BExprKind> {
    match form.controls {
        PolyControls::Voltage(pairs) => pairs
            .iter()
            .map(|[positive, negative]| BExprKind::Voltage {
                positive: positive.clone(),
                negative: Some(negative.clone()),
            })
            .collect(),
        PolyControls::Current(sources) => sources
            .iter()
            .map(|parameter| BExprKind::Current(parameter.value.clone()))
            .collect(),
    }
}

pub(super) fn lower(device: &DeviceInstance, form: &PolyForm<'_>) -> SpiceResult<DeviceInstance> {
    let plan = Plan::prepare(device, form)?;
    let location = &device.location;
    let controls = controls(form);
    let mut monomials = Vec::with_capacity(plan.count);
    terms(device, form, |coefficient, powers| {
        let value = coefficient.expression();
        if powers.degree == 0 {
            monomials.push(value);
        } else {
            let mut factors = powers.powers.iter().flat_map(|&(index, power)| {
                let control = &controls[index];
                std::iter::repeat_with(move || quantity(control.clone(), location))
                    .take(power as usize)
            });
            let product = balanced(
                &mut factors,
                powers.degree as usize,
                BBinaryOp::Mul,
                location,
            )
            .ok_or_else(|| malformed(device, "POLY nonconstant term has no controlling factors"))?;
            monomials.push(binary(BBinaryOp::Mul, value, product, location));
        }
        Ok(())
    })?;
    for (control, used) in controls.into_iter().zip(plan.used) {
        if !used {
            let zero = quantity(
                BExprKind::Number {
                    value: 0.,
                    spelling: "0".to_owned(),
                },
                location,
            );
            monomials.push(binary(
                BBinaryOp::Mul,
                zero,
                quantity(control, location),
                location,
            ));
        }
    }
    let sum = balanced(
        &mut monomials.into_iter(),
        plan.count,
        BBinaryOp::Add,
        location,
    )
    .ok_or_else(|| malformed(device, "POLY needs at least one coefficient"))?;
    let output = if matches!(device.designator, 'e' | 'h') {
        "v"
    } else {
        "i"
    };
    let text = format!(
        "spice2poly({}) with {} coefficient(s)",
        form.dimension,
        form.coefficients.len()
    );
    let mut parameters = vec![ParameterAssignment {
        name: output.to_owned(),
        value: text.clone(),
        kind: generated(sum, text, location),
        location: location.clone(),
    }];
    parameters.extend_from_slice(form.setters);
    Ok(instance(
        device.name.clone(),
        device.designator,
        device.nodes[..2].to_vec(),
        parameters,
        location,
    ))
}

#[cfg(test)]
mod tests {
    use super::PolyPowers;

    fn poly_exponents(dimension: usize, count: usize) -> Vec<Vec<u32>> {
        let mut powers = PolyPowers::new(dimension).unwrap();
        (0..count)
            .map(|_| {
                let mut dense = vec![0; dimension];
                for &(index, power) in powers.advance().unwrap() {
                    dense[index] = power;
                }
                dense
            })
            .collect()
    }

    /// Independent recursive composition enumerator, limited to the tiny
    /// reference cases; the production lowering never allocates these lists.
    fn reference_terms(dimension: usize, count: usize) -> Vec<Vec<u32>> {
        fn compose(
            prefix: &mut Vec<u32>,
            remaining: u32,
            dimension: usize,
            out: &mut Vec<Vec<u32>>,
            count: usize,
        ) {
            if out.len() == count {
                return;
            }
            if prefix.len() + 1 == dimension {
                prefix.push(remaining);
                out.push(prefix.clone());
                prefix.pop();
            } else {
                for power in (0..=remaining).rev() {
                    prefix.push(power);
                    compose(prefix, remaining - power, dimension, out, count);
                    prefix.pop();
                    if out.len() == count {
                        break;
                    }
                }
            }
        }
        let mut result = Vec::new();
        let mut degree = 1;
        while result.len() < count {
            compose(&mut Vec::new(), degree, dimension, &mut result, count);
            degree += 1;
        }
        result
    }

    #[test]
    fn poly_power_cursor_matches_compositions() {
        for dimension in [1, 2, 3, 4, 5, 6, 64] {
            assert_eq!(
                poly_exponents(dimension, 300),
                reference_terms(dimension, 300)
            );
        }
        assert!(PolyPowers::new(0).is_none());
        let mut overflow = PolyPowers {
            dimension: 1,
            degree: u32::MAX,
            powers: vec![(0, u32::MAX)],
        };
        assert!(overflow.advance().is_none());
        let mut limit = PolyPowers {
            dimension: 2,
            degree: u32::MAX,
            powers: vec![(0, 1), (1, u32::MAX - 1)],
        };
        assert_eq!(limit.advance(), Some([(1, u32::MAX)].as_slice()));
        assert!(limit.advance().is_none());
    }

    #[test]
    fn poly_terms_follow_the_spice2_order() {
        assert_eq!(
            poly_exponents(2, 5),
            vec![vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1], vec![0, 2]]
        );
        assert_eq!(
            poly_exponents(3, 9),
            vec![
                vec![1, 0, 0],
                vec![0, 1, 0],
                vec![0, 0, 1],
                vec![2, 0, 0],
                vec![1, 1, 0],
                vec![1, 0, 1],
                vec![0, 2, 0],
                vec![0, 1, 1],
                vec![0, 0, 2],
            ]
        );
        assert_eq!(poly_exponents(1, 3), vec![vec![1], vec![2], vec![3]]);
        assert_eq!(poly_exponents(2, 7)[5], vec![3, 0]);
    }
}
