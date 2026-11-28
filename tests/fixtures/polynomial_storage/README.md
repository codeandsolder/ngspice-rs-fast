# Independent physical polynomial-storage witnesses

Run the synthetic decks with ngspice-47 from a scratch directory, e.g.
`ngspice -b signed-ramp.cir`. These use explicit integrated Q(V) laws,
not the PSpice syntax or native Rust kernel. The second branch is the
voltage/current dual of an inductor: its driven voltage represents
inductor current and its capacitor current represents inductor voltage.
The signed current through each driving voltage source is negative
the storage rate.

With a -1 to +1 ramp over 1 ms, the differential is B*(1+0.2*x+0.3*x*x).
The 960 samples between 20 and 980 microseconds in ngspice-47 gave
maximum rate errors 1.60000067e-9 A and 1.60000090e-6 V respectively.
Trapezoid integration of the emitted samples gives:

| Quantity | Observed | Analytic endpoint difference |
| --- | ---: | ---: |
| Capacitor charge (C) | 2.093469642862476e-6 | 2.093470792795904e-6 |
| Capacitor energy (J) | 1.1579625849364815e-7 | 1.157960026772801e-7 |
| Inductor flux (Vs) | 0.002093469642862479 | 0.0020934707927959036 |
| Inductor energy (J) | 0.0001157962584936486 | 0.0001157960026772802 |

AC at 1 kHz and bias -2 V / -0.5 A: source currents in the explicit
charge/dual circuits are -j0.004021238596594937 and
-j39.39557187601601. The positive voltage/current orientation on the
inductor in the native test reverses the second source-current sign.

The source-specific PSpice dialect law is qualified separately; these
fixtures establish independent numerical agreement of the physical
constitutive equations only. No raw outputs are needed in the repository.
