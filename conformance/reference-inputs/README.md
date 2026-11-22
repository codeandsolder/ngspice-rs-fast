# Equivalent decimal inputs for the C reference

The original decks in `conformance/netlists` and original C captures in
`conformance/golden` are retained unchanged. The three decks here differ
only in equivalent spellings of numeric quantities. Rust always simulates
the original decks. The C capture/check path selects the equivalent deck,
and verify selects its independently captured output in
`conformance/golden/exact-input`. There is no fallback to the historical
capture and no change to the numerical comparison policy.

## Reason

ngspice's INPevaluate digit accumulation and final pow(10, exponent)
multiplication do not guarantee correctly rounded decimal conversion.
A one-ULP difference in the input width can become a visible difference
in cancellation-sensitive finite-difference sensitivity coefficients.
The equivalent spellings below were verified mathematically with decimal
arithmetic and tested using the pinned C executable, not by adjusting
simulator output or selecting an input to fit the Rust output.

| Original | Equivalent decimal |
| --- | --- |
| 20u | 2000e-8 |
| 40u | 4000e-8 |
| -0.3 | -30e-2 |
| -0.7 | -700000e-6 |
| 0.3 | 30e-2 |
| 0.6 | 60e-2 |
| 10u | 1000e-8 |
| 15u | 1500e-8 |
| 30u | 3000e-8 |
| 50u | 5000e-8 |

HFET2 uses the first two substitutions; MOS3 and MOS9 use the remaining
substitutions except 40u, and also 20u. The values in the following table
are measurements of the actual binary64 C output before ASCII serialization.

| Quantity | C original spelling | C equivalent spelling | Rust correctly rounded |
| --- | ---: | ---: | ---: |
| HFET2 v(z3:d2) | -0.7158704142536727 | -0.71586967938218538 | -0.71587004669587639 |
| MOS3 v(m1:tox) | 3.232009098014089 | 3.2320144991908988 | 3.2320144991908988 |
| MOS9 v(m1:tox) | 3.232009098014089 | 3.2320144991908988 | 3.2320144991908988 |

MOS results agree bit-for-bit for these quantities with the equivalent
input. HFET2 agrees within the existing non-linear sensitivity tolerance
(absolute difference 3.67314e-7). Every output column is still compared
under the existing policy.

## Reproduction

Capture environment: Linux x86_64 GNU, ngspice-47+ pinned source
`059066b5`, build Fri Nov 20 08:20:29 UTC 2026.
Executable SHA256:

```
1bbe5cce36480daecc64f2797c638d89dc72274c625a407507090c6c33cb492c
```

For each of sens_hfet2, sens_mos3 and sens_mos9:

```sh
cargo xtask golden capture --netlist NAME --ngspice /path/to/ngspice
cargo xtask golden check --netlist NAME --ngspice /path/to/ngspice
cargo xtask golden verify --netlist NAME
```

The C deck is instrumented with the usual control block (set ASCII filetype,
run, write NAME.raw, then endc) and run with ngspice -b. The rawfiles are
unmodified C output. Date headers are ignored by the existing drift check.
The other platforms have not been independently recaptured; no assertion of
bit-exact output across C builds/libm implementations is made.

## Original and equivalent deck identities

```
72782a7bcc089f7c73a8e1daab59d1acf31f9e6d13f399e6da3d842af8de3b82  conformance/netlists/sens_hfet2.cir
b89c8d30441c8ac3400709ece226f190972736b1945d6537417b4eb046f3984a  conformance/netlists/sens_mos3.cir
95c3d0cb496d2282f77126cc5726d35b63b8d28372d86298cdcc2786a0cb29c3  conformance/netlists/sens_mos9.cir
4df877b17749ffbe1b0929c9d9b7bcdb3b423fc5e244a150e60c2decb178baa7  conformance/reference-inputs/sens_hfet2.cir
c1ef773b366eb19593125cf95954724240affae15c46fb348b325cfcda72bf74  conformance/reference-inputs/sens_mos3.cir
14ee381fbff5ca416a1802db4fa665854446c8dc0468cfd7f5e78e3e57054e27  conformance/reference-inputs/sens_mos9.cir
```

## Historical pole-zero trace input

The separate diagnostic fixture `conformance/pencil_search_traces.txt`
preserves the original C trace data. Its notch case originally used
100n, which the pinned C executable parses as binary64
0x1.ad7f29abcaf49p-24, one ULP above correctly rounded 100n.
The diagnostic now spells that same captured input as
1000000000000000100e-25. Both pinned C and the correctly rounded parser
produce the historical binary64 value from this spelling.

In a separate minimal capacitor deck, pinned C with
`set numdgt=17`, `op`, and `print @c2[capacitance]` reported
1.00000000000000009e-07 for both original 100n and the explicit
historical-input spelling. The shorter 1.0000000000000001e-7
instead produced 9.99999999999999955e-08 in C, so is not a reproducible
spelling of the captured input on this reference frontend. The original
production notch deck is not modified.

## Independent decimal conversion oracle

The preceding implementation checkpoint was also checked against
27,673 independently generated exact decimal cases by the original
issue-94 worker. The expected and actual binary64 output files had
the identical SHA256
`73f6f7a374b5c5f505e297a6397da603491b20e6af2220525425a4780288b8ea`.
The generator, probe source and hashes are retained in the campaign
log directory worker-arch-gpt6-1122-0042-k7m-issue94/decimal-oracle.
The in-tree decimal boundary tests include independently calculated
bit patterns and adversarial significant-digit tails.
