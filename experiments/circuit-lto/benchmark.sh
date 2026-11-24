#!/usr/bin/env bash
set -euo pipefail
binary=${1:?path to benchmark binary}
cpu=${2:-7}
for case in 0 1 2 3 4; do
  for mode in generic dispatch baked; do
    taskset -c "$cpu" "$binary" "$mode" "$case" 1
  done
  for round in 0 1 2 3 4; do
    case "$((round % 3))" in
      0) modes=(generic dispatch baked);;
      1) modes=(dispatch baked generic);;
      2) modes=(baked generic dispatch);;
    esac
    for mode in "${modes[@]}"; do
      taskset -c "$cpu" "$binary" "$mode" "$case"
    done
  done
done
