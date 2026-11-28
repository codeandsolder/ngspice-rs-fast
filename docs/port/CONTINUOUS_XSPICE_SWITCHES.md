# Native continuous XSPICE switches

The `pswitch` and `aswitch` A-instance models are memoryless analog conductance laws. Both use the ordered differential output `%gd(p n)`. The first port selects the control:

- `pswitch`: `%gd(cp cn)` (default) or `%g(cp)`. The control draws current through `r_cntl_in` (default 1 TΩ).
- `aswitch`: unloaded `%v(cp)` (default) or `%vd(cp cn)`, an ideal current-sensing short `%i(cp)` or `%id(cp cn)`, or `%vnam(source)` to read an existing voltage-defined branch. Named branches are scoped as instance references, not electrical nodes.

All model setters are evaluated in the declared lexical scope; duplicates follow source order. The immutable material is shared across instances. No discrete state, equivalent threshold switch, hidden source rewriting or source-dialect recognition is used.

For control value (c), let (t=(c-c_{off})/(c_{on}-c_{off})), (G=1/R), and (V=v_p-v_n). The branch current is (I=G(t)V).

- P-switch logarithmic mode uses `log(R) = log(Roff) + (log(Ron)-log(Roff)) * t² * (3-2t)` inside the control interval and fixed endpoint resistances outside it.
- A-switch logarithmic mode uses exponential interpolation, `log(R) = log(Roff) + (log(Ron)-log(Roff)) * t`.
- Both linear modes interpolate R itself. A-switch extrapolates unless `limit=true`; P-switch always bounds its control interval.
- The minimum on resistance is 1 mΩ. P-switch caps its off resistance at 1 TΩ and expands a control-threshold difference below 1 p-unit by ±1 m-unit. A-switch rejects that near-coincident threshold. Unbounded A-switch and linear P-switch retain the native 1 nΩ resistance floor. Reversed control thresholds use the same normalized coordinate with its signed denominator.

DC and transient use the signed residual and Jacobian, with the control derivative `V * dG/dc` and equal/opposite output KCL stamps. AC uses that same bias Jacobian, including the control conductance, and the ideal current-input branch constraints. At the non-differentiable linear/exponential endpoint joins the interior slope is selected; outside a fixed resistance interval the slope is zero. No storage or noise sources are declared by these code models.

## Independent qualification

Reference: ngspice-47, `src/xspice/icm/xtradev/{pswitch,aswitch}/{ifspec.ifs,cfunc.mod}`.

The released C binary's DC sweep supports arbitrary threshold values and both directions, including cubic interpolation. The C AC loader, however, has a sign inconsistency: in a 1 V output with unit output AC excitation at fixed bias, the measured supply-current increment is **positive** G rather than the negative DC slope. Its cross derivative for a P-switch with threshold span 2 also lacks the factor 1/2, and its AC input current omits `r_cntl_in`. At limited A-switch endpoints the C loader continues stamping an extrapolation derivative outside the interval. These are not physical linearizations of its own DC characteristic. The native solver intentionally uses the signed DC derivative and preserves the actual control port's conductance; the regression tests measure the AC result against converged DC finite differences, not the erroneous C AC sign.

The source-level tests in `tests/native_continuous_switch.rs` exercise all four resistance laws, both threshold directions, differential and single-ended control, current-port polarity, branch-reference hierarchy and writer round trip, extrapolation, floors, invalid declarations, DC finite-difference AC checks and a loaded transient ramp. The opt-in live C DC reference test compares synthetic decks directly.
