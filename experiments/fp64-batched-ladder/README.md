# FP64 batch acceleration feasibility experiment

This standalone crate is deliberately outside the production simulator workspace.
It measures one specific favorable-to-batching system family, not general SPICE
performance. Each system is a 64-node grounded conductance ladder with off-diagonal
entries -1 and positive diagonals in [2.125, 3.125). The known solution differs
across rows and systems; construct its RHS, solve the tridiagonal system, and
verify every returned coordinate and the componentwise normalized residual.

## Execution

Use the repository's pinned nightly and the managed compiler cache. From this
directory on a Vulkan adapter with FP64, run (choose a disposable external
directory on your build volume):

```sh
build=$(mktemp -d /home/jan/build/ngspice-rs-fast/fp64-ladder.XXXXXX)
trap 'rm -rf -- "$build"' EXIT
cargo rustc --locked --release --bin fp64-batched-ladder --target-dir "$build/target" \
  --config "build.build-dir='$build/target'" -- -C target-cpu=native
"$build/target/release/fp64-batched-ladder"
```

All backends use binary64, store numeric data as 64-system tiles with
contiguous lanes at every row, and perform the same forward elimination and
back substitution. The scalar baseline is one independent lane at a time;
portable SIMD uses four lanes (AVX2 width on the initial Ryzen host), and
multithreaded SIMD distributes whole tiles over persistent Rayon CPU workers.
GPU uses one invocation per system in a 64-invocation workgroup. Each
invocation follows the dependency chain in system order. There is no
cross-lane reduction or per-step host synchronization.

Reported times are medians of seven measurements after one warmup. CPU
workspaces and GPU buffers/pipeline are allocated outside the timing.
GPU resident means synchronized dispatch with inputs/results remaining in GPU
buffers, not shader timestamp-only measurement. The amortized result measures
eight separate compute passes per submission, divided by eight. Roundtrip includes input
writes, dispatch, output copy to staging, map and full copy into an existing
host vector. Numerical verification is outside the timed region.

## Limitations

- Strictly diagonally dominant, fixed tridiagonal matrix, no pivoting.
- No nonlinear device load, events, continuation, dynamic topology, or
  arbitrary sparse fill-in.
- No general performance claims from one machine under concurrent builds.
- The shader uses native Vulkan FP64, not portable browser WebGPU. A GPU
  without FP64 is not a valid backend for this experiment.
- Future generalization must use the simulator's validated packed symbolic
  traversal, separate numeric state, handle lane-local pivot failures,
  and preserve ordered results and physical tolerances.

## Complex sparse RC mesh

The second binary (`--bin mesh`) uses the same runner and measures 64-node
8x8 complex nodal systems, with shared symbolic fill-in and elimination
tape. Run it as above, substituting `mesh` for the binary name. The real
part is strictly diagonally dominant; frequency-dependent shunt imaginary
admittance is varied independently by lane. It validates normalized
backward error against original coefficients in addition to known-solution
error. Unlike a general pivoting solver, the symbolic traversal is fixed;
no general MNA or nonlinear device support is implied.

See [RESULTS.md](RESULTS.md) for measured results and version details.
See #101 (GPU experiment), #96 (representative baseline), #97/#98
(independent numerical batches and multiple RHS). This branch is not intended
to add WGPU or experimental build dependencies to production.
