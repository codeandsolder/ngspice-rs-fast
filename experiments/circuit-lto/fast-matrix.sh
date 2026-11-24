#!/usr/bin/env bash
# Each variant is a directory with circuit_lto. Rotated order; all invocations
# run on the same pinned CPU. Keep failing cases out of the timing matrix.
set -euo pipefail
cpu=${1:?cpu}
root=${2:?root}
shift 2
variants=("$@")
counts=(400 100 50 400 100)
for c in 0 1 2 3 4; do
  for round in 0 1 2 3 4; do
    for ((i=0; i<${#variants[@]}; i++)); do
      v=${variants[$(((i+round)%${#variants[@]}))]}
      if (( round % 2 )); then modes=(baked generic); else modes=(generic baked); fi
      for m in "${modes[@]}"; do
        printf '%s ' "$v"
        timeout 20 taskset -c "$cpu" "$root/$v/circuit_lto" "$m" "$c" "${counts[c]}"
      done
    done
  done
done
