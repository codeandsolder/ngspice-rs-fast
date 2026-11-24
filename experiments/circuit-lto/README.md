# Circuit-specialized whole-program LLVM experiment

Research branch, not a production feature. Base: c7860ad (main). The normal solver is retained, including sparse factorization, Newton convergence, acceptance/rejection, and analysis drivers. The test matrix is embedded in the benchmark executable. Each device is elaborated through the existing parser and model setup; the emitter walks the actual C-compatible load order, and emits source with fixed device positions and concrete Rust types. BSIM4 prepared model and geometry records become immutable static values (floating-point fields emitted by their exact bits). Literal resistor node IDs and conductances are also compiled in. Mutable device and solver states remain dynamic. BSIM4's setup-only version string is not used in the load kernel; its numeric version is preserved.

The experiment uses LLVM through rustc, not a native in-process JIT. Source generation plus whole-program compilation is a deliberately slow stand-in for an on-demand compiler. It is not a general arbitrary-deck compiler, and it does not yet compile the sparse factorization schedule or fully bake the other compact models. The specialization API is only for the exact embedded decks. The production generic path remains the default, and replacements or different temperatures use generic dispatch.

## Reproduce on Acer

Use the managed sccache wrapper from the existing Cargo configuration; do not bypass it. Build output goes outside the worktree.

```sh
cargo +nightly-2026-11-20 build --release --example circuit_lto --locked \
  --target-dir /home/jan/build/ngspice-rs-fast/circuit-lto/target \
  --config 'profile.release.lto="fat"' --config profile.release.codegen-units=1
B=/home/jan/build/ngspice-rs-fast/circuit-lto/target/release/examples/circuit_lto
# Re-emit the specializations from the real elaborator if the input decks change:
experiments/circuit-lto/generate.sh "$B"
# Repeat the build above to compile the emitted specializations.
for case in 0 1 2 3 4; do "$B" verify "$case"; done
experiments/circuit-lto/benchmark.sh "$B" 7
```

Three modes in the same executable: `generic` (original dynamic load order), `dispatch` (constant order and typed calls), `baked` (also static BSIM4 model/size and R values). All share the same compilation options and numerical kernels. Time circuit setup, analysis execution and output hashing separately. The verify mode compares every point and column, including exact floating-point bits. Pin comparisons to one CPU and rotate mode order between rounds. Compile time is excluded. No I/O or parser work is included in the solve measurement.
