# Native control language

The explicit `spice-rs control deck.cir` entry point executes the deck's
`.control` sections, in source order. Ordinary batch simulation does not
execute scripts. The library entry point is `cli::control::run_file`, which
accepts independent resource limits and an output writer and returns the
retained typed plots and scalar variables.

## Supported commands

- `let name = expression`: scalar expression or current-plot vector expression.
- `foreach name value ...` and `while expression [comparison expression]`:
  nested blocks terminated by `end`. Numeric scalar substitution accepts
  `$&name`, `$name` and bare names. Scalar arithmetic follows the parameter
  expression grammar; vector arithmetic operates on complex samples.
- `alterparam name = expression` stages a top-level parameter update;
  `reset` applies all staged updates transactionally. A failed update does not
  replace the current circuit. Reusable candidate parameter plans avoid
  rebuilding topology where possible; solver state is independent per run.
- `run` executes deck analyses in their ordinary batch order. Supported
  analysis names can also be invoked directly (e.g. `op`, `dc v1 0 3 1`,
  `tran 1u 1m`). Numeric analysis arguments can use scalar substitution.
- `meas analysis name ...` uses the batch measurement grammar and evaluator,
  including named parameter expressions, crossings, derivative, extrema,
  peak-to-peak and gain/phase margins. It selects the most recent matching
  analysis plot and binds the result as a scalar.
- `print name ...` prints scalars or vectors. A retained plot can be qualified
  by name, e.g. `dc1.v(out)`. `setplot name` changes the current plot;
  `destroy all` releases retained plots without resetting scalar variables.
- `linearize [np=N] [vector ...]` interpolates a transient plot onto its
  requested output grid. N optionally limits the number of grid samples.
  `fft [vector ...]` requires uniform real time samples. Omitting names
  selects all eligible vectors. The forward spectrum is single-sided and
  uses radix-2 zero padding where needed, normalization by actual input
  sample count, and undoubled DC and Nyquist bins.
- `set specwindow=none|hann|hamming|triangle|blackman|blackmanharris|flattop`
  selects the FFT window (default Hann).
- `quit` terminates script execution.

Derived vectors support named columns, voltage/current probes, arithmetic,
and selected real/complex functions (including magnitude, phase, real and
imaginary components). The immutable schema is bound once per expression;
the packed plot representation is shared with measurements. Linearization
uses the existing ordered-axis interpolation geometry and refuses
extrapolation or invalid/nonadvancing grids. FFT follows the normalization
conventions of ngspice frontend/com_fft.c and maths/fft/fftext.c.

## Example

```spice
Divider
.param r=1k
v1 in 0 3
r1 in out {r}
r2 out 0 1k
.dc v1 0 3 1
.control
foreach resistance 1k 2k
  alterparam r = $&resistance
  reset
  run
  meas dc output find v(out) at=2
  print output
  destroy all
end
quit
.endc
.end
```

## Limits and semantics

Scripts are compiled before numerical execution; malformed or unsupported
commands are explicit errors. Runtime errors stop execution; text already
written is not rolled back. Resource budgets cover instruction/loop steps,
scalar expression evaluations, retained plots and cells, deterministic
analysis grid points, transform work, scalar names and printed bytes.
Plot destruction releases the retention budget but does not replenish
cumulative work budgets. The standard solver limits remain in force.

This is a bounded native subset, not a general Nutmeg interpreter. Shell
execution, arbitrary file operations, interactive commands, direct instance
alterations, and mixed batch output/measurement cards in script mode are
not implemented. Unsupported syntax must fail rather than be ignored.
