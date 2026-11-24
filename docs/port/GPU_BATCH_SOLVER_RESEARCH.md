# FP64 GPU-resident batched circuit solve experiment (#101)

Scope: independent numerical systems with shared topology, strictly separate from
the production simulator and default dependencies. Grounded diagonally dominant
32-row conductance ladders establish a reproducible correctness and cost baseline.
Input layout [system][row][lower, diagonal, upper, rhs] uses only IEEE binary64.
A tridiagonal Thomas solver runs one system per Vulkan compute invocation; scalar
and portable SIMD four-lane CPU variants are independent references. All variants
are checked against a known analytic solution and the original Ax-b residual.
No synthetic result is evidence of a production SPICE solver speedup.

The GPU requires Vulkan SHADER_F64, GLSL -> SPIR-V (glslc) and wgpu's
native Vulkan shader-module passthrough; WGSL currently cannot represent FP64
arithmetic. Shader modules are compiled from in-repo sources, not from untrusted
input. The experiment has its own Cargo workspace and lockfile and is not a
production simulator dependency.

Run from experiments/fp64-gpu-solver using managed sccache:

    cargo +nightly test --locked
    cargo +nightly run --locked --release

The benchmark separates the GPU's warm resident dispatch (amortized over
repeated compute dispatches without buffer upload) from complete upload +
solve + download. SIMD and GPU measurements require per-sample residual/error
checks. Performance conclusions and general sparse fixed-pivot extensions
are recorded only after verifying on Acer RADV hardware.
