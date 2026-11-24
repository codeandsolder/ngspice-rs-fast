#!/usr/bin/env bash
# Interleave build variants and modes on the same pinned CPU.
# Usage: benchmark-matrix.sh CPU BASE_DIR variant ...
set -euo pipefail
cpu=${1:?CPU}
root=${2:?directory}
shift 2
variants=("$@")
(( ${#variants[@]} > 0 ))
counts=(400 100 50 400 100)
for c in 0 1 2 3 4; do
  for round in 0 1 2 3 4; do
    for ((i=0; i<${#variants[@]}; i++)); do
      v=${variants[$(((i+round)%${#variants[@]}))]}
      if (( round % 2 )); then modes=(baked generic); else modes=(generic baked); fi
      for mode in "${modes[@]}"; do
        printf '%s ' "$v"
        taskset -c "$cpu" "$root/$v/circuit_lto" "$mode" "$c" "${counts[c]}"
      done
    done
  done
done
