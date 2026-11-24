# BSIM4 junction, leakage and stamp numerics

The physical source and drain use the same junction law, with independent
temperature-prepared saturation currents, breakdown parameters, emission
coefficients and forward/reverse continuation fields. The polarity-normalized
junction biases and C selector/clamp boundaries are unchanged.

## Junctions

Within the unclipped exponential branch, use expm1(v/Vt) for forward current
and -XExpBV * expm1(-v/Vt) for the change in reverse breakdown current
from equilibrium. Do not apply the latter identity across a clamp. Mode 0
uses the unclipped reverse exponential, mode 1 the forward continuation,
and mode 2 both limiting continuations. The zero-saturation branch retains
gmin and the mode selector still determines which branch owns the state.

Reverse depletion uses the shared nonlinear-device integral and derivative.
The forward polynomial remains the reference continuation. The area,
sidewall and gate-sidewall terms retain their order, including QBS/QBD
writes. At a 1 aV bias, Is=1 pA and Vt=26 mV, the unclipped diode
current is approximately 3.846153846153846e-29 A rather than zero.
A 1 pF junction has charge -1e-30 C at -1 aV to leading order.

Trap-assisted tunneling uses the derivative of the rational voltage law:
for threshold T, voltage v and positive emission thermal voltage Vt,
the magnitude of the exponent derivative is (T/(T-v))²/Vt. This avoids
subtracting nearly equal terms at large reverse bias. The separate
near-threshold continuation and DEXP clamps remain in force.

## Gate leakage

Softplus, depletion and effective drain voltage use the conditioned
constitutive laws. For channel tunneling partition argument x and
D=x²+2e-4, define P(x)=sum(x^k/(k+3)!, k=0..12). In |x|<=0.5 the
C regularized partition can be written as

    s = 1/2 + x³ P(x)/D
    d = 1/2 + x³ [1/2+(x-1)P(x)]/D.

Differentiate these centered forms rather than subtracting two O(x)
terms. Evaluate x²/D before multiplying by x to avoid premature
underflow. The first omitted exponential term is at order 16.
At x=±1e-18 the source and drain slopes are about 2.5e-33 and
5e-33 respectively, despite both fractions rounding to 1/2.
For ordinary arguments retain the exponential formula; for finite
clipped arguments use reciprocal scaling to avoid overflow in x².
The regularized reference fractions are not renormalized.

## Stamping

The RHS and mixed current/charge matrix entries use ordered product
sums. Recover finite products with the existing ExactSum accumulator
when cancellation is severe (absolute result <= 1/8 of absolute
term sum), or a product or reduction loses exponent range. Preserve
the ordinary nonfinite propagation and signed exact zero. Mixed
unit-weight terms need only addition; with one unit weight a fused
multiply-add rounds both terms together. Stamp destinations, event
order and internal-node aliases are not changed.

The conservation fixture checks the physical row sum of the RHS and
each matrix column, plus common-mode voltage derivatives for every
row, including the NQS equation. It covers 36 combinations of
polarity, forward/reverse/near-zero drain bias, DC/transient and
three internal-node configurations. NQS charge is not itself an
electrical terminal. The existing ordered-load trace is retained.

## Hot-path measurements

Acer, optimized standalone extraction of the actual numerical kernels,
8.192 million evaluations, median user CPU time of three runs. These are
kernel measurements, not full-device or circuit speedups.

| Partition distribution | Target | C order | Initial conditioned loop | Horner (separate) | Horner (FMA) |
| --- | --- | ---: | ---: | ---: | ---: |
| Small arguments | native | 0.10 s | 0.41 s | 0.14 s | 0.11 s |
| Mixed arguments | native | 0.10 s | 0.22 s | 0.15 s | 0.13 s |
| Small arguments | generic x86-64 | 0.11 s | 0.49 s | 0.16 s | 0.32 s |
| Mixed arguments | generic x86-64 | 0.10 s | 0.31 s | 0.21 s | 0.26 s |

Use native FMA for Horner when available, separate operations for generic
targets. On the same test scale, ordinary ordered residual reduction
costs approximately 0.01 s versus 0.03 s for adaptive reduction; with
10% severe cancellation the latter costs 0.21–0.22 s and recovers
819500 units omitted by the ordered reduction. The fast path avoids
the exact accumulator for well-conditioned products. Unit-weight
matrix sums bypass that recovery entirely.

An independent 600-digit decimal sweep of 3589 finite arguments covered
both signs, the polynomial/clamp boundaries, log-spaced near-zero
arguments down to 1e-150 and extreme arguments up to 1e308. For the
actual final native and generic x86-64 kernels, the maximum relative
errors were 4.1e-16 (source fraction), 4.9e-16 (drain fraction),
5.3e-15 (source slope), and 2.2e-15 (drain slope), using the smallest
normal binary64 value as the error denominator floor. No finite input
in that sweep produced a nonfinite output.

## Shared oxide tunneling and derivative reductions

All five oxide-tunneling barrier paths (channel, source and drain
overlap, bulk accumulation and inversion) evaluate the common
polynomial as (a-bv)(1+cv), with fused evaluation of the linear
factors and a conditioned product-rule derivative. The exponential
clamps still have exactly zero derivative. Nonfinite inputs use the
reference expanded arithmetic, including NaN propagation. The root
regression uses (1+epsilon)(1-epsilon)=1-epsilon² to exercise each
factor's cancellation independently; amplification by the scale
factor makes the previously lost residual observable in the current.

The reduction module is shared by leakage, its product-rule
derivatives, and nonlinear stamps; there is no leakage-to-stamp
dependency. The broad FMA suppression on the leakage evaluation
stage is removed. Local exceptions describe the original
nonfinite fallback, the native/generic Horner distinction, the
exact unit-weight sum, and the prepared/clipped diode identities.

An additional 8.192-million-evaluation barrier microbenchmark used
ordinary positive coefficient tuples with negative oxide scale and
a second distribution containing 10% near-root cancellations.
Median user CPU time over three runs:

| Target | Distribution | Expanded | Factored/conditioned |
| --- | --- | ---: | ---: |
| native | ordinary | 0.05 s | 0.12 s |
| native | roots | 0.05 s | 0.11 s |
| generic x86-64 | ordinary | 0.05 s | 0.14 s |
| generic x86-64 | roots | 0.06 s | 0.14 s |

These costs are for the optional barrier kernel, not a measured
circuit slowdown. Moving input-finiteness checks and derivative
evaluation after the clamp made little difference on native and
only a modest improvement on generic in a second standalone run;
the directly auditable finite/nonfinite branch was retained.
