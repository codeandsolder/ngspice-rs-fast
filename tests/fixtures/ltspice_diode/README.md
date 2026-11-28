# LTspice diode source-family dispatch

These independent source decks and measurements were captured with LTspice
24.0.12. The DC sweep has 181 voltage points and 14 diodes, with 12 identical
LEVEL=2 physical responses and two identical idealized responses. The compact
table retains two independently measured columns for each family. The AC decks
cover all rounded LEVEL boundaries, plain and ideal-selector declarations,
duplicate setter order and per-subcircuit occurrence LEVEL expressions.
AC terminal admittance is measured by a zero-volt probe in series with each
diode (not the vendor's internal current display). Noise references are
constant-current bias measurements; the transient reference is a linear
voltage ramp with independently measured terminal current. The raw sample
times are retained. The source decks are passed through the actual source
preprocessor, not manually rewritten to a native model.
