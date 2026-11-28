# Independent BSIM1/2 circuit witnesses

Generated from the checked-in decks by ngspice 47 with KLU, using the
adjacent generator (uv run --no-project python generate.py --ngspice ngspice).
The script uses a temporary directory for intermediate decks.

The direct deck covers both levels, both polarities and both orientations,
including asymmetric diffusion areas, perimeters, sheet-resistance squares,
overlap coefficients and multiplicity. DC and AC current vectors are
recorded in SI units.

The forward-equivalent deck represents the same physical reverse-mode
devices with reversed drain/source connections and correspondingly reversed
terminal-specific geometry and overlap coefficients. This avoids a native
reverse-mode charge-state bug: b1ld.c/b2ld.c return the intrinsic drain
charge derivatives into csgb/cssb/csdb in inverse mode, while
b1acld.c/b2acld.c consume cdgb/cddb/cdsb without a matching permutation.
Both direct and transformed native DC currents agree. Physical AC must
agree with the transformed forward model, not the erroneous native
reverse-mode AC matrix. No reference data is modified by the Rust solver.
