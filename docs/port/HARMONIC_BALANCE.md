# Native multi-tone harmonic balance — numerical architecture

Tracking: https://github.com/codeandsolder/ngspice-rs-fast/issues/172

A forced nonlinear periodic/quasiperiodic steady-state solver is not
the same as DC-linearized AC/SP, perturbative disto, or a transient
startup followed by Fourier extraction. An autonomous oscillator requires
an unknown frequency and an explicit phase condition; a fixed-frequency
forced solution must not pretend to support free-running oscillators.

## Multitone frequency lattice

Tone declarations can be independent or locked by an explicit positive
rational ratio to an earlier tone. Nested locked ratios are supported.
The minimal integer group axis is computed using denominator LCM and
coefficient GCD; for instance 300 MHz plus 450 MHz at 3/2 share a
150-MHz fundamental, and all driven harmonics are represented exactly
once. A truly incommensurate source adds its own torus dimension.

A lattice point is the full tuple of signed harmonic indices. Its
physical frequency is the dot product with independent fundamentals.
The pair (1,-1) denotes difference frequency and (2,-1) is a genuine
two-tone third-order intermodulation product. Opposite tuples share
conjugate coefficients. Unintentional frequency-coincident independent
tuples are rejected instead of silently coalescing two independent
physical states or guessing a rational relationship from a tolerance.

For each torus axis, retained harmonics are -H to +H; at least 4H+1
real samples are taken along that axis. This 2x oversampling does not
eliminate aliasing for arbitrary nonlinear exp/pow devices: harmonic
order and sample-resolution convergence require independent checks.
Mode order, samples, tone rational groups and conjugate mapping are
deterministic and validated before allocating. Coefficient/sample and
direct DFT-work budgets prevent combinatorial growth and overflow.

## Physical MNA and analytical Jacobian

For real-valued MNA state x(t), linear static operator G,
linear storage C, nonlinear conduction i(x), nonlinear stored charge
q(x), and independent excitation b(t), define:

    F(t) = G x(t) + i(x(t))
           + d/dt [C x(t) + q(x(t))] - b(t).

The residual of every harmonic tuple k is

    F_k = G X_k + I_k(X)
          + j 2pi (k dot f) [C X_k + Q_k(X)] - B_k.

The same formula handles DC (k=0), which has no charge derivative.
Algebraic ideal-source branch rows use C=0. No gmin or artificial
positive conductance is inserted.

The constitutive callback must return complete MNA current and charge
contributions plus their true local Jacobians. Projected real sample
derivatives give the analytical Jacobian-vector action

    J dX = G dX + DFT[ (di/dx) IDFT(dX) ]
           + j omega [C dX + DFT[ (dq/dx) IDFT(dX) ]].

No finite-difference Jacobian or simplified nonlinearity is assumed.
The current numerical baseline assembles a bounded real dense Newton
matrix from those actions and reuses the SPARSE-backed dense factor
solver. Bounded line search accepts only finite residual-decreasing
steps. Singular, invalid and nonconverged systems produce structured
errors, not a silent fallback to AC, disto or transient approximation.

This baseline is not intended to assert scalability to massive RF
systems. Reusable topology-bound sparse workspaces and matrix-free
Newton/Krylov can consume the same correct Jv interface later. Benchmarks
must cover actual device evaluation, DFT, factorization and backtracking,
not synthetic FFT-only or isolated vector-operation wins.

## Integration and acceptance

1. Numerical core: correct analytic linear RC/RLC, nonlinear exp diode
   and stored-charge rectification, two incommensurate tones with mixer
   products, exact conjugacy, invalid near-duplicate frequencies,
   resource and nonconvergence policy.
2. Native adapter: real ngspice device I/Q and partial derivatives,
   prepared model/instance state, MNA ordering, no re-elaboration,
   dynamic charge and history consistent with DC/transient.
3. Expose in typed library, netlist and CLI planning and report frequency
   observation metadata. Common analysis/mod.rs, driver and parser files
   must be reserved after the concurrently active PSS/control workers
   have released them. No backward compatibility wrappers.
4. Independent HB reference: compare overlapping models to QucsatorRF/
   Xyce and to long transient/Fourier on stable driven circuits. Verify
   nonlinear KCL/charge residuals, harmonic order, oversampling, gain
   compression, stiff diode, resonator and mixer cases. No claim of
   oscillator support without a frequency/phase formulation.

The reserved source files implement a real numerical core and plan.
Until the other campaign worker releases analysis/mod.rs, the isolated
harmonic_balance integration test includes those exact source files
as a test module. This **does not mean** the simulator has a public
netlist or native device HB endpoint yet; that is the remaining
integration work of issue #172.
