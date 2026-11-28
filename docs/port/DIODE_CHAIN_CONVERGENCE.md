# Diode-chain convergence qualification

Issue #239 concerns the former bitwise assertions in
`tests/convergence.rs`. These circuits consist of N identical
100-ohm resistors and forward diodes (IS = 10 fA, N = 1), a 5 V
source, and a 1 kohm load. The default junction gmin is 1 pS.
All currents are below the usual absolute current tolerance, so
individual node voltages can change with sparse-factor arithmetic
even while KCL is satisfied to essentially machine precision.

## Independent reference

The unique positive-bias solution follows from the monotone equation

    I = IS * expm1(Vd / VT) + gmin * Vd
    5 = N * Vd + (100 * N + 1000) * I

with VT = (1.38064852e-23 / 1.6021766208e-19) * 300.15.
A 60-decimal-digit bisection gives the following physical points.

| Stages | First stage output (V) | Load current (A) |
|---:|---:|---:|
| 100 | 4.9500000000010911 | 1.0911042185529007e-13 |
| 1000 | 4.9950000000000071 | 7.1326132646201586e-15 |

Independent Linux ngspice 47 (Aug 19, 2026 build, x86-64) gave:

| Stages | First stage output (V) | Load current (A) | Source current magnitude (A) | Iterations |
|---:|---:|---:|---:|---:|
| 100 | 4.950001782213789 | 1.091143634516647e-13 | 1.090932899572294e-13 | 7 |
| 1000 | 4.994932578213038 | 7.083563566626339e-15 | 7.230327447871332e-15 | 229 |

Direct Newton succeeds for 100 stages; dynamic nodal gmin succeeds
after direct failure for 1000. The 100-stage C values exactly
reproduce the historical test. Recomputing every KCL equation using
all node voltages in the ASCII raw file gives maximum residuals
1.78e-17 and 2.17e-17 A. Maximum deviations of any junction drop
from the exact scalar point are 5.22 and 113.09 microvolts.

## Numerical origin

Commit 1d62e0c introduced recovery of ill-conditioned Schur updates
using the raw pivot and cross-products. The current real sparse path
uses this when cancellation makes the rounded multiplier unreliable.
It also retains raw factor recovery and conditioned junction laws.
At architecture base db2e4d0 the observed Rust results were:

| Stages | First stage output (V) | Load current (A) | Source current magnitude (A) | Iterations |
|---:|---:|---:|---:|---:|
| 100 | 4.950000681252841 | 1.091153120142441e-13 | 1.091071677450373e-13 | 7 |
| 1000 | 4.994920408579700 | 7.127090465461164e-15 | 7.251144129583054e-15 | 491 |

The long chain uses true junction gmin after nodal gmin fails, and
returns the full-source original equations. Maximum physical KCL
residuals are 9.87e-18 and 1.24e-17 A, smaller than C in both cases.
The load current is closer to the exact root for 1000 stages.

In an isolated, uncommitted experiment, replace only the scalar real
Schur update with the plain rounded subtraction `value - mult * lower`,
holding every other implementation detail fixed. The 100-stage result
then exactly reproduces all historical C values. The 1000-stage chain
fails every continuation strategy after 1434 iterations, including
source stepping. The experiment is reverted. Restoring bit parity by
removing the cancellation safeguard would therefore regress convergence.

## Regression contract

The test independently solves the scalar monotone equation, checks all
200 or 2000 junction/resistor KCL rows plus source and load, and bounds
the drop deviation using rounded-up independently measured C errors.
The KCL bound (8 machine epsilons at 5 V / 100 ohm) accounts for
subtracting nearly equal node voltages, and is far tighter than default
ABSTOL. Load-current error bounds are 100 ppm and 1%, respectively.
The 100-stage direct path and seven iterations remain required; the
1000-stage case requires a successful gmin continuation and a final
unregularized stage within the configured budget. This tests the
physical solution without forcing the implementation to reproduce an
obsolete floating-point execution trace.
