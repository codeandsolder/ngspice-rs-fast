# LTspice diode-family selection and prepared binding

Source normalization labels every LTspice D declaration `ltspice_diode`.
This label identifies the dialect, not its constitutive law. Native SPICE D
and the explicit SIMetrix level-1 diode retain their separate policies.

The engine resolves every ordered setter in the owning model and occurrence
scope before selecting a backend. The last finite LEVEL, default 1, selects
Fowler–Nordheim when it lies in [1.5, 2.5), regardless of the presence of
ideal selectors. Outside that interval, the presence of any of RON, ROFF,
VFWD, VREV or RREV selects the idealized diode; otherwise the semiconductor
law applies. This rule covers signed LEVEL values and duplicate setters.
The first raw LEVEL cache is only a parser consistency check, not the
physical selection. Expressions in native SPICE's integer LEVEL position
remain subject to its own parser policy.

## Fowler–Nordheim

The physical kernel in the preprocessor workspace owns the current, charge,
Jacobian and shot-noise PSD. For terminal voltage V, signed series count n,
oxide thickness T and effective geometry A (L W when both are nonzero, or
1 otherwise), its intrinsic voltage is v = V/n. The current is

    I = m A sign(v) 1e10 (v/T)^2 exp(-T/abs(v))

and is exactly zero at v = 0. Terminal charge is C V with

    C = m A 0.3453133/(n T).

L and W on the instance override their model defaults independently;
an explicit zero is a value, not an omitted override. T and n must be
nonzero. Temperature and the ordinary area and D setters are not geometry
coefficients of this law. The producer uses range-safe products and
explicitly diagnoses nonrepresentable physical values. The physical
Jacobian is the derivative of the above law and does not add a leakage
conductance. The shot-noise law is 2 q abs(I)/n; a negative nonzero PSD
is rejected instead of clamped.

## Semiconductor

The distinct semiconductor kernel owns bulk and optional sidewall
transport, geometry, conductance ceilings, depletion and diffusion charge,
and noise. Independent physical series paths become separate hidden
electrical coordinates; a sidewall without its own series path shares
the bulk intrinsic coordinate. The engine stamps the producer's
conservative current and charge vectors and both Jacobians. It does not
duplicate intrinsic laws or supply an alternate native diode fallback.

PERIM is an LTspice occurrence setter; the native PJ spelling is not an
alias in this dialect. Native SPICE resolves PERIM as its existing PJ
alias when binding native model parameters. Instance TEMP is an absolute
temperature override. VP-dependent dynamic recovery is a separate
physical state law; active VP/TT is not silently discarded.

## Evidence

The independent LTspice DC, AC, transient and noise witness decks and
measurements are in tests/fixtures/ltspice_diode. The solver integration
tests consume the actual source preprocessor output and compare terminal
observations, including independently connected series-node topologies,
charge transfer, geometry, duplicate LEVEL setters and nested model
scope. Kernel-level tests separately qualify the constitutive equations,
thermal corrections, signed inputs and limiting domains. The engine
retains only the numerical state, not independent copies of these laws.
