#!/usr/bin/env bash
# Same relative case weighting as the timing workload; optional scale.
set -euo pipefail
binary=${1:?binary}
cpu=${2:?CPU}
scale=${3:-1}
counts=(200 50 25 200 50)
for c in 0 1 2 3 4; do
  for mode in generic dispatch baked; do
    taskset -c "$cpu" "$binary" "$mode" "$c" "$((counts[c]*scale))"
  done
done
