#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?benchmark directory}
cpu=${2:?LLVM target CPU}
base=${3:-native}
kind=${4:-fast}
target=x86_64-unknown-linux-gnu
toolchain=nightly-2026-11-20
plugin="$out/tools/fast-math/fast-math-pass.so"
dir="$out/$cpu/$base-$kind"
mkdir -p "$dir"
flags=("-Ctarget-cpu=$cpu" "-Ctarget-feature=+crt-static" "-Clink-arg=-Wl,--emit-relocs" "-Cstrip=none")
if [[ $base == pgo ]]; then
  flags+=("-Cprofile-use=$out/$cpu/pgo-gen/merged.profdata")
fi
printf -v rustflags '"%s",' "${flags[@]}"
rustflags="[${rustflags%,}]"
extra=()
if [[ $kind == fast ]]; then
  extra+=("-Zllvm-plugins=$plugin")
elif [[ $kind != strict ]]; then
  exit 2
fi
cargo +"$toolchain" rustc --release --example circuit_lto --locked --target "$target" --target-dir "$out/$cpu/$base/target" --config 'profile.release.lto="fat"' --config profile.release.codegen-units=1 --config "target.$target.rustflags=$rustflags" -- "${extra[@]}" > "$dir/build.log" 2>&1
cp "$out/$cpu/$base/target/$target/release/examples/circuit_lto" "$dir/circuit_lto"
if [[ $kind == fast ]]; then
  # Prove the module pass reached device code at the actual full LTO stage.
  rg 'CIRCUIT_FAST_MATH stage=full-lto-early .*device_functions=[1-9][0-9]*' "$dir/build.log"
fi
