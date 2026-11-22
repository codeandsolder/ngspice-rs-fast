//! Third-order derivative arithmetic for the distortion Taylor coefficients.
//! Differentiate the defining product, quotient and composition identities
//! rather than preserving the operation order of C's generated expressions. Each value carries a function of three variables `p`, `q`, `r` and
//! its partial derivatives up to the third order (derivatives, not Taylor
//! coefficients: `d2_p2` is the second derivative).

use crate::primitives::Real;

/// Hardware fusion is useful when it is part of the compilation target.
/// The baseline x86-64 target lowers scalar FMA to a costly libm call:
/// the composed derivative-chain benchmark measured 10.1x slowdown.
/// These analytic recurrences reduce cancellation separately; on targets
/// without native fusion, ordinary products and sums are the fast path.
#[inline]
#[cfg_attr(
    not(any(target_feature = "fma", target_arch = "aarch64")),
    expect(
        clippy::suboptimal_flops,
        reason = "portable scalar FMA measured 10.1x slower in the derivative-chain benchmark"
    )
)]
fn accumulate_product(a: Real, b: Real, c: Real) -> Real {
    #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
    {
        a * b + c
    }
}

/// C's `Dderivs`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(in crate::devices) struct Dderivs {
    /// Function value at the expansion point.
    pub value: Real,
    /// Order-1 partial derivative with respect to p.
    pub d1_p: Real,
    /// Order-1 partial derivative with respect to q.
    pub d1_q: Real,
    /// Order-1 partial derivative with respect to r.
    pub d1_r: Real,
    /// Order-2 partial derivative with respect to p2.
    pub d2_p2: Real,
    /// Order-2 partial derivative with respect to q2.
    pub d2_q2: Real,
    /// Order-2 partial derivative with respect to r2.
    pub d2_r2: Real,
    /// Order-2 partial derivative with respect to pq.
    pub d2_pq: Real,
    /// Order-2 partial derivative with respect to qr.
    pub d2_qr: Real,
    /// Order-2 partial derivative with respect to pr.
    pub d2_pr: Real,
    /// Order-3 partial derivative with respect to p3.
    pub d3_p3: Real,
    /// Order-3 partial derivative with respect to q3.
    pub d3_q3: Real,
    /// Order-3 partial derivative with respect to r3.
    pub d3_r3: Real,
    /// Order-3 partial derivative with respect to p2r.
    pub d3_p2r: Real,
    /// Order-3 partial derivative with respect to p2q.
    pub d3_p2q: Real,
    /// Order-3 partial derivative with respect to q2r.
    pub d3_q2r: Real,
    /// Order-3 partial derivative with respect to pq2.
    pub d3_pq2: Real,
    /// Order-3 partial derivative with respect to pr2.
    pub d3_pr2: Real,
    /// Order-3 partial derivative with respect to qr2.
    pub d3_qr2: Real,
    /// Order-3 partial derivative with respect to pqr.
    pub d3_pqr: Real,
}

impl Dderivs {
    /// The variable `p`, `q` or `r` (`index` 0, 1, 2) at zero: bjtdset.c's
    /// `d_p`, `d_q`, `d_r`.
    pub(crate) fn variable(index: usize) -> Self {
        let mut d = Self::default();
        match index {
            0 => d.d1_p = 1.0,
            1 => d.d1_q = 1.0,
            _ => d.d1_r = 1.0,
        }
        d
    }

    /// C `plusder.c`.
    pub(crate) fn plus(&self, other: &Self) -> Self {
        Self {
            value: self.value + other.value,
            d1_p: self.d1_p + other.d1_p,
            d1_q: self.d1_q + other.d1_q,
            d1_r: self.d1_r + other.d1_r,
            d2_p2: self.d2_p2 + other.d2_p2,
            d2_q2: self.d2_q2 + other.d2_q2,
            d2_r2: self.d2_r2 + other.d2_r2,
            d2_pq: self.d2_pq + other.d2_pq,
            d2_qr: self.d2_qr + other.d2_qr,
            d2_pr: self.d2_pr + other.d2_pr,
            d3_p3: self.d3_p3 + other.d3_p3,
            d3_q3: self.d3_q3 + other.d3_q3,
            d3_r3: self.d3_r3 + other.d3_r3,
            d3_p2r: self.d3_p2r + other.d3_p2r,
            d3_p2q: self.d3_p2q + other.d3_p2q,
            d3_q2r: self.d3_q2r + other.d3_q2r,
            d3_pq2: self.d3_pq2 + other.d3_pq2,
            d3_pr2: self.d3_pr2 + other.d3_pr2,
            d3_qr2: self.d3_qr2 + other.d3_qr2,
            d3_pqr: self.d3_pqr + other.d3_pqr,
        }
    }

    /// C `timesder.c` (`k * old`).
    pub(crate) fn times(&self, k: Real) -> Self {
        Self {
            value: k * self.value,
            d1_p: k * self.d1_p,
            d1_q: k * self.d1_q,
            d1_r: k * self.d1_r,
            d2_p2: k * self.d2_p2,
            d2_q2: k * self.d2_q2,
            d2_r2: k * self.d2_r2,
            d2_pq: k * self.d2_pq,
            d2_qr: k * self.d2_qr,
            d2_pr: k * self.d2_pr,
            d3_p3: k * self.d3_p3,
            d3_q3: k * self.d3_q3,
            d3_r3: k * self.d3_r3,
            d3_p2r: k * self.d3_p2r,
            d3_p2q: k * self.d3_p2q,
            d3_q2r: k * self.d3_q2r,
            d3_pq2: k * self.d3_pq2,
            d3_pr2: k * self.d3_pr2,
            d3_qr2: k * self.d3_qr2,
            d3_pqr: k * self.d3_pqr,
        }
    }

    /// Compose a scalar function using its first three derivatives.
    fn compose(&self, value: Real, first: Real, second: Real, third: Real) -> Self {
        let pure = |d1: Real, d2: Real, d3: Real| {
            accumulate_product(
                first,
                d3,
                accumulate_product(second, 3. * d2 * d1, third * d1 * d1 * d1),
            )
        };
        let repeated = |di: Real, dj: Real, dii: Real, dij: Real, diij: Real| {
            accumulate_product(
                first,
                diij,
                accumulate_product(
                    second,
                    accumulate_product(dii, dj, 2. * dij * di),
                    third * di * di * dj,
                ),
            )
        };
        Self {
            value,
            d1_p: first * self.d1_p,
            d1_q: first * self.d1_q,
            d1_r: first * self.d1_r,
            d2_p2: accumulate_product(first, self.d2_p2, second * self.d1_p * self.d1_p),
            d2_q2: accumulate_product(first, self.d2_q2, second * self.d1_q * self.d1_q),
            d2_r2: accumulate_product(first, self.d2_r2, second * self.d1_r * self.d1_r),
            d2_pq: accumulate_product(first, self.d2_pq, second * self.d1_p * self.d1_q),
            d2_qr: accumulate_product(first, self.d2_qr, second * self.d1_q * self.d1_r),
            d2_pr: accumulate_product(first, self.d2_pr, second * self.d1_p * self.d1_r),
            d3_p3: pure(self.d1_p, self.d2_p2, self.d3_p3),
            d3_q3: pure(self.d1_q, self.d2_q2, self.d3_q3),
            d3_r3: pure(self.d1_r, self.d2_r2, self.d3_r3),
            d3_p2r: repeated(self.d1_p, self.d1_r, self.d2_p2, self.d2_pr, self.d3_p2r),
            d3_p2q: repeated(self.d1_p, self.d1_q, self.d2_p2, self.d2_pq, self.d3_p2q),
            d3_q2r: repeated(self.d1_q, self.d1_r, self.d2_q2, self.d2_qr, self.d3_q2r),
            d3_pq2: repeated(self.d1_q, self.d1_p, self.d2_q2, self.d2_pq, self.d3_pq2),
            d3_pr2: repeated(self.d1_r, self.d1_p, self.d2_r2, self.d2_pr, self.d3_pr2),
            d3_qr2: repeated(self.d1_r, self.d1_q, self.d2_r2, self.d2_qr, self.d3_qr2),
            d3_pqr: accumulate_product(
                first,
                self.d3_pqr,
                accumulate_product(
                    second,
                    accumulate_product(
                        self.d2_pq,
                        self.d1_r,
                        accumulate_product(self.d2_pr, self.d1_q, self.d2_qr * self.d1_p),
                    ),
                    third * self.d1_p * self.d1_q * self.d1_r,
                ),
            ),
        }
    }

    pub(crate) fn exp(&self) -> Self {
        let value = self.value.exp();
        self.compose(value, value, value, value)
    }

    pub(crate) fn inv(&self) -> Self {
        Self {
            value: 1.,
            ..Self::default()
        }
        .div(self)
    }

    /// The multivariate Leibniz rule. Repeated terms use their multiplicity
    /// instead of computing and adding the same product twice.
    pub(crate) fn mul(&self, other: &Self) -> Self {
        let (a, b) = (self, other);
        let mut result = Self {
            value: a.value * b.value,
            d1_p: accumulate_product(a.d1_p, b.value, a.value * b.d1_p),
            d1_q: accumulate_product(a.d1_q, b.value, a.value * b.d1_q),
            d1_r: accumulate_product(a.d1_r, b.value, a.value * b.d1_r),
            d2_p2: accumulate_product(
                2. * a.d1_p,
                b.d1_p,
                accumulate_product(a.d2_p2, b.value, a.value * b.d2_p2),
            ),
            d2_q2: accumulate_product(
                2. * a.d1_q,
                b.d1_q,
                accumulate_product(a.d2_q2, b.value, a.value * b.d2_q2),
            ),
            d2_r2: accumulate_product(
                2. * a.d1_r,
                b.d1_r,
                accumulate_product(a.d2_r2, b.value, a.value * b.d2_r2),
            ),
            d2_pq: accumulate_product(
                a.d1_q,
                b.d1_p,
                accumulate_product(
                    a.d1_p,
                    b.d1_q,
                    accumulate_product(a.d2_pq, b.value, a.value * b.d2_pq),
                ),
            ),
            d2_qr: accumulate_product(
                a.d1_r,
                b.d1_q,
                accumulate_product(
                    a.d1_q,
                    b.d1_r,
                    accumulate_product(a.d2_qr, b.value, a.value * b.d2_qr),
                ),
            ),
            d2_pr: accumulate_product(
                a.d1_r,
                b.d1_p,
                accumulate_product(
                    a.d1_p,
                    b.d1_r,
                    accumulate_product(a.d2_pr, b.value, a.value * b.d2_pr),
                ),
            ),
            ..Self::default()
        };
        result.mul_third_order(a, b);
        result
    }

    fn mul_third_order(&mut self, a: &Self, b: &Self) {
        self.d3_p3 = accumulate_product(
            3. * a.d2_p2,
            b.d1_p,
            accumulate_product(
                3. * a.d1_p,
                b.d2_p2,
                accumulate_product(a.d3_p3, b.value, a.value * b.d3_p3),
            ),
        );
        self.d3_q3 = accumulate_product(
            3. * a.d2_q2,
            b.d1_q,
            accumulate_product(
                3. * a.d1_q,
                b.d2_q2,
                accumulate_product(a.d3_q3, b.value, a.value * b.d3_q3),
            ),
        );
        self.d3_r3 = accumulate_product(
            3. * a.d2_r2,
            b.d1_r,
            accumulate_product(
                3. * a.d1_r,
                b.d2_r2,
                accumulate_product(a.d3_r3, b.value, a.value * b.d3_r3),
            ),
        );
        self.mul_repeated_third_order(a, b);
        self.d3_pqr = accumulate_product(
            a.d2_qr,
            b.d1_p,
            accumulate_product(
                a.d2_pr,
                b.d1_q,
                accumulate_product(
                    a.d2_pq,
                    b.d1_r,
                    accumulate_product(
                        a.d1_r,
                        b.d2_pq,
                        accumulate_product(
                            a.d1_q,
                            b.d2_pr,
                            accumulate_product(
                                a.d1_p,
                                b.d2_qr,
                                accumulate_product(a.d3_pqr, b.value, a.value * b.d3_pqr),
                            ),
                        ),
                    ),
                ),
            ),
        );
    }

    fn mul_repeated_third_order(&mut self, a: &Self, b: &Self) {
        // Leibniz rule for a derivative with one index occurring twice.
        let repeated = |[ai, aj, aii, aij, aiij]: [Real; 5],
                        [bi, bj, bii, bij, biij]: [Real; 5]| {
            accumulate_product(
                2. * aij,
                bi,
                accumulate_product(
                    aii,
                    bj,
                    accumulate_product(
                        aj,
                        bii,
                        accumulate_product(
                            2. * ai,
                            bij,
                            accumulate_product(aiij, b.value, a.value * biij),
                        ),
                    ),
                ),
            )
        };
        self.d3_p2r = repeated(
            [a.d1_p, a.d1_r, a.d2_p2, a.d2_pr, a.d3_p2r],
            [b.d1_p, b.d1_r, b.d2_p2, b.d2_pr, b.d3_p2r],
        );
        self.d3_p2q = repeated(
            [a.d1_p, a.d1_q, a.d2_p2, a.d2_pq, a.d3_p2q],
            [b.d1_p, b.d1_q, b.d2_p2, b.d2_pq, b.d3_p2q],
        );
        self.d3_q2r = repeated(
            [a.d1_q, a.d1_r, a.d2_q2, a.d2_qr, a.d3_q2r],
            [b.d1_q, b.d1_r, b.d2_q2, b.d2_qr, b.d3_q2r],
        );
        self.d3_pq2 = repeated(
            [a.d1_q, a.d1_p, a.d2_q2, a.d2_pq, a.d3_pq2],
            [b.d1_q, b.d1_p, b.d2_q2, b.d2_pq, b.d3_pq2],
        );
        self.d3_pr2 = repeated(
            [a.d1_r, a.d1_p, a.d2_r2, a.d2_pr, a.d3_pr2],
            [b.d1_r, b.d1_p, b.d2_r2, b.d2_pr, b.d3_pr2],
        );
        self.d3_qr2 = repeated(
            [a.d1_r, a.d1_q, a.d2_r2, a.d2_qr, a.d3_qr2],
            [b.d1_r, b.d1_q, b.d2_r2, b.d2_qr, b.d3_qr2],
        );
    }

    /// Solve the derivative of b*n = a in increasing derivative order.
    /// This avoids dividing repeatedly by powers of b or re-evaluating
    /// the same nearly cancelling numerator.
    pub(crate) fn div(&self, other: &Self) -> Self {
        let (a, b) = (self, other);
        let mut n = Self {
            value: a.value / b.value,
            ..Self::default()
        };
        n.d1_p = (accumulate_product(-n.value, b.d1_p, a.d1_p)) / b.value;
        n.d1_q = (accumulate_product(-n.value, b.d1_q, a.d1_q)) / b.value;
        n.d1_r = (accumulate_product(-n.value, b.d1_r, a.d1_r)) / b.value;
        n.d2_p2 = (accumulate_product(
            -2. * n.d1_p,
            b.d1_p,
            accumulate_product(-n.value, b.d2_p2, a.d2_p2),
        )) / b.value;
        n.d2_q2 = (accumulate_product(
            -2. * n.d1_q,
            b.d1_q,
            accumulate_product(-n.value, b.d2_q2, a.d2_q2),
        )) / b.value;
        n.d2_r2 = (accumulate_product(
            -2. * n.d1_r,
            b.d1_r,
            accumulate_product(-n.value, b.d2_r2, a.d2_r2),
        )) / b.value;
        n.d2_pq = (accumulate_product(
            -n.d1_q,
            b.d1_p,
            accumulate_product(
                -n.d1_p,
                b.d1_q,
                accumulate_product(-n.value, b.d2_pq, a.d2_pq),
            ),
        )) / b.value;
        n.d2_qr = (accumulate_product(
            -n.d1_r,
            b.d1_q,
            accumulate_product(
                -n.d1_q,
                b.d1_r,
                accumulate_product(-n.value, b.d2_qr, a.d2_qr),
            ),
        )) / b.value;
        n.d2_pr = (accumulate_product(
            -n.d1_r,
            b.d1_p,
            accumulate_product(
                -n.d1_p,
                b.d1_r,
                accumulate_product(-n.value, b.d2_pr, a.d2_pr),
            ),
        )) / b.value;
        n.div_third_order(a, b);
        n
    }

    fn div_third_order(&mut self, a: &Self, b: &Self) {
        self.d3_p3 = (accumulate_product(
            -3. * self.d2_p2,
            b.d1_p,
            accumulate_product(
                -3. * self.d1_p,
                b.d2_p2,
                accumulate_product(-self.value, b.d3_p3, a.d3_p3),
            ),
        )) / b.value;
        self.d3_q3 = (accumulate_product(
            -3. * self.d2_q2,
            b.d1_q,
            accumulate_product(
                -3. * self.d1_q,
                b.d2_q2,
                accumulate_product(-self.value, b.d3_q3, a.d3_q3),
            ),
        )) / b.value;
        self.d3_r3 = (accumulate_product(
            -3. * self.d2_r2,
            b.d1_r,
            accumulate_product(
                -3. * self.d1_r,
                b.d2_r2,
                accumulate_product(-self.value, b.d3_r3, a.d3_r3),
            ),
        )) / b.value;
        self.div_repeated_third_order(a, b);
        self.d3_pqr = (accumulate_product(
            -self.d2_qr,
            b.d1_p,
            accumulate_product(
                -self.d2_pr,
                b.d1_q,
                accumulate_product(
                    -self.d2_pq,
                    b.d1_r,
                    accumulate_product(
                        -self.d1_r,
                        b.d2_pq,
                        accumulate_product(
                            -self.d1_q,
                            b.d2_pr,
                            accumulate_product(
                                -self.d1_p,
                                b.d2_qr,
                                accumulate_product(-self.value, b.d3_pqr, a.d3_pqr),
                            ),
                        ),
                    ),
                ),
            ),
        )) / b.value;
    }

    fn div_repeated_third_order(&mut self, a: &Self, b: &Self) {
        // Differentiate b*n = a for an index occurring twice, solving
        // the remaining linear equation for n's third derivative.
        let value = self.value;
        let repeated = |aiij: Real,
                        [ni, nj, nii, nij]: [Real; 4],
                        [bi, bj, bii, bij, biij]: [Real; 5]| {
            accumulate_product(
                -2. * nij,
                bi,
                accumulate_product(
                    -nii,
                    bj,
                    accumulate_product(
                        -nj,
                        bii,
                        accumulate_product(-2. * ni, bij, accumulate_product(-value, biij, aiij)),
                    ),
                ),
            ) / b.value
        };
        self.d3_p2r = repeated(
            a.d3_p2r,
            [self.d1_p, self.d1_r, self.d2_p2, self.d2_pr],
            [b.d1_p, b.d1_r, b.d2_p2, b.d2_pr, b.d3_p2r],
        );
        self.d3_p2q = repeated(
            a.d3_p2q,
            [self.d1_p, self.d1_q, self.d2_p2, self.d2_pq],
            [b.d1_p, b.d1_q, b.d2_p2, b.d2_pq, b.d3_p2q],
        );
        self.d3_q2r = repeated(
            a.d3_q2r,
            [self.d1_q, self.d1_r, self.d2_q2, self.d2_qr],
            [b.d1_q, b.d1_r, b.d2_q2, b.d2_qr, b.d3_q2r],
        );
        self.d3_pq2 = repeated(
            a.d3_pq2,
            [self.d1_q, self.d1_p, self.d2_q2, self.d2_pq],
            [b.d1_q, b.d1_p, b.d2_q2, b.d2_pq, b.d3_pq2],
        );
        self.d3_pr2 = repeated(
            a.d3_pr2,
            [self.d1_r, self.d1_p, self.d2_r2, self.d2_pr],
            [b.d1_r, b.d1_p, b.d2_r2, b.d2_pr, b.d3_pr2],
        );
        self.d3_qr2 = repeated(
            a.d3_qr2,
            [self.d1_r, self.d1_q, self.d2_r2, self.d2_qr],
            [b.d1_r, b.d1_q, b.d2_r2, b.d2_qr, b.d3_qr2],
        );
    }

    /// Differentiate n*n = self. The source model defines the derivatives
    /// as zero at a zero argument, including the singular square-root limit.
    pub(crate) fn sqrt(&self) -> Self {
        let mut n = Self {
            value: self.value.sqrt(),
            ..Self::default()
        };
        if self.value == 0. {
            return n;
        }
        let denominator = 2. * n.value;
        n.d1_p = (self.d1_p) / denominator;
        n.d1_q = (self.d1_q) / denominator;
        n.d1_r = (self.d1_r) / denominator;
        n.d2_p2 = (accumulate_product(-2. * n.d1_p, n.d1_p, self.d2_p2)) / denominator;
        n.d2_q2 = (accumulate_product(-2. * n.d1_q, n.d1_q, self.d2_q2)) / denominator;
        n.d2_r2 = (accumulate_product(-2. * n.d1_r, n.d1_r, self.d2_r2)) / denominator;
        n.d2_pq = (accumulate_product(-2. * n.d1_p, n.d1_q, self.d2_pq)) / denominator;
        n.d2_qr = (accumulate_product(-2. * n.d1_q, n.d1_r, self.d2_qr)) / denominator;
        n.d2_pr = (accumulate_product(-2. * n.d1_p, n.d1_r, self.d2_pr)) / denominator;
        n.d3_p3 = (accumulate_product(-6. * n.d1_p, n.d2_p2, self.d3_p3)) / denominator;
        n.d3_q3 = (accumulate_product(-6. * n.d1_q, n.d2_q2, self.d3_q3)) / denominator;
        n.d3_r3 = (accumulate_product(-6. * n.d1_r, n.d2_r2, self.d3_r3)) / denominator;
        n.d3_p2r = (accumulate_product(
            -2. * n.d2_p2,
            n.d1_r,
            accumulate_product(-4. * n.d1_p, n.d2_pr, self.d3_p2r),
        )) / denominator;
        n.d3_p2q = (accumulate_product(
            -2. * n.d2_p2,
            n.d1_q,
            accumulate_product(-4. * n.d1_p, n.d2_pq, self.d3_p2q),
        )) / denominator;
        n.d3_q2r = (accumulate_product(
            -2. * n.d2_q2,
            n.d1_r,
            accumulate_product(-4. * n.d1_q, n.d2_qr, self.d3_q2r),
        )) / denominator;
        n.d3_pq2 = (accumulate_product(
            -4. * n.d2_pq,
            n.d1_q,
            accumulate_product(-2. * n.d1_p, n.d2_q2, self.d3_pq2),
        )) / denominator;
        n.d3_pr2 = (accumulate_product(
            -4. * n.d2_pr,
            n.d1_r,
            accumulate_product(-2. * n.d1_p, n.d2_r2, self.d3_pr2),
        )) / denominator;
        n.d3_qr2 = (accumulate_product(
            -4. * n.d2_qr,
            n.d1_r,
            accumulate_product(-2. * n.d1_q, n.d2_r2, self.d3_qr2),
        )) / denominator;
        n.d3_pqr = (accumulate_product(
            -2. * n.d2_pr,
            n.d1_q,
            accumulate_product(
                -2. * n.d2_pq,
                n.d1_r,
                accumulate_product(-2. * n.d1_p, n.d2_qr, self.d3_pqr),
            ),
        )) / denominator;
        n
    }

    pub(crate) fn tan(&self) -> Self {
        let value = self.value.tan();
        let squared = value * value;
        let first = 1. + squared;
        self.compose(
            value,
            first,
            2. * value * first,
            2. * first * accumulate_product(3_f64, squared, 1.),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotient_and_square_root_recover_all_mixed_derivatives() {
        let a = Dderivs {
            value: 2.,
            d1_p: 0.2,
            d1_q: -0.3,
            d1_r: 0.4,
            d2_p2: 0.5,
            d2_q2: -0.6,
            d2_r2: 0.7,
            d2_pq: 0.8,
            d2_qr: -0.9,
            d2_pr: 1.1,
            d3_p3: -0.2,
            d3_q3: 0.3,
            d3_r3: -0.4,
            d3_p2r: 0.5,
            d3_p2q: -0.6,
            d3_q2r: 0.7,
            d3_pq2: -0.8,
            d3_pr2: 0.9,
            d3_qr2: -1.1,
            d3_pqr: 1.2,
        };
        let b = Dderivs {
            value: 3.,
            d1_p: -0.3,
            d1_q: 0.8,
            d1_r: 0.1,
            d2_pq: -0.4,
            d2_pr: 0.5,
            d2_q2: 0.2,
            d3_pqr: -1.4,
            ..Dderivs::default()
        };
        for result in [a.mul(&b).div(&b), a.mul(&a).sqrt()] {
            assert!((result.value - a.value).abs() <= 8e-15, "value: {result:?}");
            assert!((result.d1_p - a.d1_p).abs() <= 8e-15, "d1_p: {result:?}");
            assert!((result.d1_q - a.d1_q).abs() <= 8e-15, "d1_q: {result:?}");
            assert!((result.d1_r - a.d1_r).abs() <= 8e-15, "d1_r: {result:?}");
            assert!((result.d2_p2 - a.d2_p2).abs() <= 8e-15, "d2_p2: {result:?}");
            assert!((result.d2_q2 - a.d2_q2).abs() <= 8e-15, "d2_q2: {result:?}");
            assert!((result.d2_r2 - a.d2_r2).abs() <= 8e-15, "d2_r2: {result:?}");
            assert!((result.d2_pq - a.d2_pq).abs() <= 8e-15, "d2_pq: {result:?}");
            assert!((result.d2_qr - a.d2_qr).abs() <= 8e-15, "d2_qr: {result:?}");
            assert!((result.d2_pr - a.d2_pr).abs() <= 8e-15, "d2_pr: {result:?}");
            assert!((result.d3_p3 - a.d3_p3).abs() <= 8e-15, "d3_p3: {result:?}");
            assert!((result.d3_q3 - a.d3_q3).abs() <= 8e-15, "d3_q3: {result:?}");
            assert!((result.d3_r3 - a.d3_r3).abs() <= 8e-15, "d3_r3: {result:?}");
            assert!(
                (result.d3_p2r - a.d3_p2r).abs() <= 8e-15,
                "d3_p2r: {result:?}"
            );
            assert!(
                (result.d3_p2q - a.d3_p2q).abs() <= 8e-15,
                "d3_p2q: {result:?}"
            );
            assert!(
                (result.d3_q2r - a.d3_q2r).abs() <= 8e-15,
                "d3_q2r: {result:?}"
            );
            assert!(
                (result.d3_pq2 - a.d3_pq2).abs() <= 8e-15,
                "d3_pq2: {result:?}"
            );
            assert!(
                (result.d3_pr2 - a.d3_pr2).abs() <= 8e-15,
                "d3_pr2: {result:?}"
            );
            assert!(
                (result.d3_qr2 - a.d3_qr2).abs() <= 8e-15,
                "d3_qr2: {result:?}"
            );
            assert!(
                (result.d3_pqr - a.d3_pqr).abs() <= 8e-15,
                "d3_pqr: {result:?}"
            );
        }
    }

    #[test]
    fn quotient_does_not_overflow_squared_denominator_at_extreme_scale() {
        let a = Dderivs {
            value: 1e200,
            d1_p: 1e200,
            ..Dderivs::default()
        };
        let b = Dderivs {
            value: 1e200,
            d1_q: 1e200,
            ..Dderivs::default()
        };
        let q = a.div(&b);
        assert_eq!(q.value.to_bits(), 1_f64.to_bits());
        assert_eq!(q.d1_p.to_bits(), 1_f64.to_bits());
        assert_eq!(q.d1_q.to_bits(), (-1_f64).to_bits());
        assert_eq!(q.d2_q2.to_bits(), 2_f64.to_bits());
        assert_eq!(q.d2_pq.to_bits(), (-1_f64).to_bits());
        assert_eq!(q.d3_q3.to_bits(), (-6_f64).to_bits());
    }
}
