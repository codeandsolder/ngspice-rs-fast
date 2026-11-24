#!/usr/bin/env bash
# Uses the managed clang++ wrapper on Acer. Plugin uses the LLVM symbols
# exported by the compiler loading it; do not link a second LLVM library.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?benchmark directory}/tools/fast-math
mkdir -p "$out"
clang++ $(llvm-config --cxxflags) -std=c++20 -O2 -fPIC -c experiments/circuit-lto/fast-math-pass.cpp -o "$out/fast-math-pass.o"
clang++ -shared "$out/fast-math-pass.o" -o "$out/fast-math-pass.so"
printf 'define double @f(double %%x, double %%y) { %%z = fmul double %%x, %%y\n ret double %%z }\n' | opt --load-pass-plugin "$out/fast-math-pass.so" -passes=blanket-fast-math -S > "$out/probe.ll"
rg 'fmul fast' "$out/probe.ll"
