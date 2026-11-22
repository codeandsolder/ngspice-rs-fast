# M9 observation capture oracle

`m9_device_dc.raw` is the original upstream C capture from Rust commit
[`6fb9e491`](https://github.com/codeandsolder/ngspice-rs-fast/commit/6fb9e491f11f40d1856a61d42830f27e51a1bee9).
It is not a Rust-generated file. The fixture and its deck must not be rewritten
to make a different reference executable pass the reproduction test.

## Historical provenance

| Field | Evidence |
| --- | --- |
| Source | `059066b532f7073fccf4173ef269309eafdc7f0c` (ngspice-47+), the documented macOS reference checkout; see [55a751b](https://github.com/codeandsolder/ngspice-rs-fast/commit/55a751b85131aaca50a59b1395503a8a3ef45836). |
| Capture date | Rawfile header: `Sat Oct 10 01:19:41  2026` (no timezone recorded). |
| Platform | macOS arm64 according to the reference provenance above; the October 10 M9 performance baseline identifies Apple M5. |
| Optimization | M9 baseline: `-O2`, KLU-enabled. Subsequent M10 notes distinguish the FMA-enabled capture oracle from the `-ffp-contract=off` fidelity build. |
| Exact compiler / SDK / OS build / configure argv | Not recorded for this individual historical capture; the rawfile cannot identify these details. |
| Invocation | The original test in `6fb9e49` invokes `ngspice -b -r <rawfile> <deck>` directly. |
| Format | Binary, 6 rows, 12 real binary64 columns, little endian. |

The [M10 reference notes at 98f7f1d](https://github.com/codeandsolder/ngspice-rs-fast/blob/98f7f1d2ea688051545eb6f10eefaffc17af60ed/docs/port/M10.md)
document the October 10 baseline and the two distinct floating-point builds.
This supports the FMA-enabled macOS build as the historical capture environment,
but does not establish its exact compiler release, contraction flag spelling,
complete configuration, startup files, or runtime library versions. A matching
reproduction on another platform would not in itself recover that history.

Immutable SHA-256 fingerprints:

- Deck: `8d2a2986e19965b531e14dabb53ef0467225e198df49d3d3d79c5b5344a010b5`
- Complete original rawfile: `295d2ff898a5928eeab5069433546e405bbac90f85db28a0215628a872e6542d`
- Binary payload after `Binary:\n`: `66ac552248614313a75f00299ea9b6e076614ea7c1e6e4a7b759a9e96bf5e0c9`

## Reproduced discrepancy (2026-11-22)

Two independent C executables run directly on the unchanged deck, with
`-n -b -r`, both produced the same payload SHA-256
`d52e7ccdf85625162beba808db68921f0026ea9c0d2c8ece350acac654a7057d`.
The only differing cell was point 2, column 2 (`i(vdrive)`, 0.2 V),
zero-based flattened index 26:

| Capture | Binary64 bits | Hexadecimal floating-point value |
| --- | --- | --- |
| Original | `0xbf2a36e34fd642fa` | `-0x1.a36e34fd642fap-13` |
| Both current C builds | `0xbf2a36e34fd642f9` | `-0x1.a36e34fd642f9p-13` |

- Arch Linux package `ngspice 47-1`, x86_64, build 2026-08-19, installed on CachyOS.
  Executable SHA-256: `3e0f126c93dc66ac2b80ac3043eba5c57b0ee545aa323cdc06f59b3d86f6a70e`.
- Pinned C source `059066b532f7073fccf4173ef269309eafdc7f0c`; Linux x86_64,
  GCC 16.2.1 (20260810), `-O2 -ffp-contract=off -std=gnu11`, KLU enabled,
  `--without-x --disable-debug --disable-ndev --disable-xspice --disable-cider
  --disable-openmp --without-readline`. Executable SHA-256:
  `1bbe5cce36480daecc64f2797c638d89dc72274c625a407507090c6c33cb492c`.

The host's glibc was 2.44. In additional scratch-only diagnostic decks,
explicit `.options sparse` and `.options klu` produced the same differing cell,
so solver selection alone does not explain the observed discrepancy.
Neither current executable is a verified historical capture oracle; the
floating-point contraction hypothesis needs a matching C build to establish
reproduction. No simulator or tolerance change follows from this measurement.

## Separate verification contracts

Ordinary regression against the original fixture (no C executable):

```sh
cargo test --locked --test m9_observations
```

Rust versus a general, independent live C build at the existing numerical bounds:

```sh
NGSPICE_BIN=/absolute/path/to/ngspice \
  cargo test --locked --test m9_observations device_observations_match_live_c -- --ignored
```

Capture reproducibility (C versus original C, no Rust simulation):

```sh
NGSPICE_CAPTURE_BIN=/absolute/path/to/verified-capture-ngspice \
  cargo test --locked --test m9_observations device_golden_matches_capture_oracle_bit_exactly -- --ignored
```

The latter requires an explicitly selected capture executable; it never
falls back to `NGSPICE_BIN` or passes a drifting value. Its comparison includes
plot metadata, row/column shape, and each real and imaginary binary64 bit
pattern, including signed zero. Dates and executable build strings are not
numerical values and are excluded from the reproduction assertion. The live
invocations use an isolated working directory, `-n` to suppress initialization
files, the C locale, and unset `SPICE_ASCIIRAWFILE`; the historical invocation's
startup environment is unknown. Neither test writes the committed fixture.

For a broad ignored-test sweep with only a general live oracle, explicitly
exclude the separate capture test:

```sh
NGSPICE_BIN=/absolute/path/to/ngspice cargo test --workspace --locked -- --ignored \
  --skip device_golden_matches_capture_oracle_bit_exactly
```

Run the capture test independently when a matching oracle is available.
Future captures must record the source revision, complete build invocation,
compiler and runtime versions, platform, binary hash, deck hash, capture
command, and output hash at capture time rather than infer them afterward.
