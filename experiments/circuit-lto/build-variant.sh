#!/usr/bin/env bash
# Native and PGO variants use the managed Cargo/sccache wrapper. All outputs
# stay under a caller-owned benchmark directory, not in the checkout.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
out=${1:?build directory}
cpu=${2:?LLVM CPU}
variant=${3:?variant}
toolchain=nightly-2026-11-20
target=x86_64-unknown-linux-gnu
dir="$out/$cpu/$variant"
mkdir -p "$dir"
flags=("-Ctarget-cpu=$cpu" "-Ctarget-feature=+crt-static" "-Clink-arg=-Wl,--emit-relocs" "-Cstrip=none")
case "$variant" in
  native) ;;
  bolt-base) flags=("-Ctarget-cpu=$cpu" "-Clink-arg=-Wl,--emit-relocs" "-Cstrip=none");;
  polly) ;;
  fma) flags+=("-Cllvm-args=--fp-contract=fast");;
  aggressive)
    flags+=("-Zmir-opt-level=4" "-Zinline-mir=yes" "-Cllvm-args=--inline-threshold=450" "-Cllvm-args=--unroll-threshold-aggressive=450" "-Cllvm-args=--enable-loopinterchange" "-Cllvm-args=--enable-loop-distribute" "-Cllvm-args=--enable-loop-flatten")
    ;;
  pgo-gen)
    mkdir -p "$dir/raw"
    flags+=("-Cprofile-generate=$dir/raw")
    ;;
  pgo)
    flags+=("-Cprofile-use=$out/$cpu/pgo-gen/merged.profdata" "-Cllvm-args=--pgo-warn-missing-function")
    ;;
  pgo-aggressive)
    flags+=("-Cprofile-use=$out/$cpu/pgo-gen/merged.profdata" "-Zmir-opt-level=4" "-Zinline-mir=yes" "-Cllvm-args=--inline-threshold=450" "-Cllvm-args=--unroll-threshold-aggressive=450" "-Cllvm-args=--enable-loopinterchange" "-Cllvm-args=--enable-loop-distribute" "-Cllvm-args=--enable-loop-flatten")
    ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
# Use target-scoped rustflags so every Rust dependency and the final LTO link
# see the same architecture, without modifying host build-script flags.
printf -v rustflags '"%s",' "${flags[@]}"
rustflags="[${rustflags%,}]"
printf '%s\n' "${flags[@]}" > "$dir/flags.txt"
common=(--release --example circuit_lto --locked --target "$target" --target-dir "$dir/target" --config 'profile.release.lto="fat"' --config profile.release.codegen-units=1 --config "target.$target.rustflags=$rustflags")
if [[ $variant == polly ]]; then
    # Distributed workers cannot package arbitrary LLVM plugins yet.
    # Only the final executable receives the plugin at the fat LTO link.
    cargo +"$toolchain" rustc "${common[@]}" -- "-Zllvm-plugins=$out/tools/polly/usr/lib/LLVMPolly.so" -Cpasses=polly > "$dir/build.log" 2>&1
else
    cargo +"$toolchain" build "${common[@]}" > "$dir/build.log" 2>&1
fi
cp "$dir/target/$target/release/examples/circuit_lto" "$dir/circuit_lto"
file "$dir/circuit_lto"
