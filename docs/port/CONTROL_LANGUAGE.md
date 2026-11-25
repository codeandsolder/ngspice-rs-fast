# Native control language

The explicit `spice-rs control deck.cir` entry point executes the deck's
`.control` sections, in source order. Ordinary batch simulation does not
execute scripts. The library entry point is `cli::control::run_file`, which
accepts independent resource limits and an output writer and returns the
retained typed plots and scalar variables.

## Supported commands

- `let name = expression`: scalar expression or current-plot vector expression.
- `foreach name value ...`, `while condition`, `dowhile condition`,
  `repeat [count]`, and `if condition` / `else`: nested blocks terminated
  by `end`. `break [levels]` and `continue [levels]` transfer within
  enclosing loops. Omitted repeat counts run until a transfer or the step
  budget is exhausted. Numeric scalar substitution accepts
  `$&name`, `$name` and bare names. Scalar arithmetic follows the parameter
  expression grammar; vector arithmetic operates on complex samples.
- `set name = expression` binds a numeric scalar; `echo text` expands
  scalar substitutions, respecting quoting, and writes a line.
- `alter instance [parameter] = expression`, `alter @instance[parameter]
  = expression`, and `alter instance expression` modify a typed instance
  setter transactionally. Edits persist across analyses, and can coexist
  with parameter-derived values; a DC sweep temporarily supersedes the
  corresponding edited setter. `reset` clears instance edits.
- `alterparam name = expression` stages a top-level parameter update;
  `reset` applies all staged updates transactionally. A failed update does not
  replace the current circuit. Reusable candidate parameter plans avoid
  rebuilding topology where possible; solver state is independent per run.
- `run` executes deck analyses in their ordinary batch order. Deck
  `.save`, `.print`, `.measure` and `.four` use the ordinary streaming
  output routes; measurements bind scalar results and consume full rows
  before save projection. Direct analyses honor save cards but do not
  implicitly execute deck print/measure/Fourier reports. Supported
  analysis names can also be invoked directly (e.g. `op`, `dc v1 0 3 1`,
  `tran 1u 1m`). Numeric analysis arguments can use scalar substitution.
- `meas analysis name ...` uses the batch measurement grammar and evaluator,
  including named parameter expressions, crossings, derivative, extrema,
  peak-to-peak and gain/phase margins. It selects the most recent matching
  analysis plot unless the current plot matches, in which case it uses
  that plot. It binds the result as a scalar.
- `print name ...` prints scalars or vectors. A retained plot can be qualified
  by name, e.g. `dc1.v(out)`. `setplot name` changes the current plot;
  `destroy` releases the current plot, `destroy name ...` releases named
  plots, and `destroy all` releases all retained plots without resetting
  scalar variables. Plot numbering follows the batch naming rules.
- `linearize [np=N] [vector ...]` interpolates a transient plot onto its
  requested output grid. N optionally limits the number of grid samples.
  `fft [length=exact|padded] [mode=amplitude|power|psd] [vector ...]`
  requires a uniform real time axis. Omitting names selects all eligible
  vectors. Exact length is the default, including odd and prime lengths.
  Explicit padding interpolates the frequency grid to the next power of two.
  Real signals produce a single-sided spectrum with undoubled DC and even
  Nyquist bins; complex signals use the full DFT frequency order.
- `set specwindow=none|hann|hamming|triangle|blackman|blackmanharris|flattop`
  selects the FFT window (default Hann). Amplitude normalization divides
  by the actual discrete window sum; power and PSD divide by window energy.
  Power bins sum to the window-weighted mean square, and integrating PSD
  across the frequency bins gives the same quantity. Neither correction
  depends on padding. The reusable spectrum plan exposes coherent gain and
  equivalent noise bandwidth in bins and Hz.
- `quit` terminates script execution.

Derived vectors support named columns, voltage/current probes, arithmetic,
and selected real/complex functions (including magnitude, phase, real and
imaginary components). The immutable schema is bound once per expression;
the packed plot representation is shared with measurements. Linearization
uses the existing ordered-axis interpolation geometry and refuses
extrapolation or invalid/nonadvancing grids. FFT uses the ngspice window conventions, with discrete normalization and
explicit odd-length and complex-spectrum semantics.

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
analysis grid points, transform work, scalar names, retained instance
setter edits and printed bytes.
Plot destruction releases the retention budget but does not replenish
cumulative work budgets. The standard solver limits remain in force.

This is a bounded native subset, not a general Nutmeg interpreter. Shell
execution, arbitrary file operations, interactive graphics, model
alterations, and topology-changing instance alterations are not implemented. Unsupported syntax must fail rather than be ignored.
