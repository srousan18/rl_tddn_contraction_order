#!/usr/bin/env bash
# Plot RL results against the baselines.
#
#   scripts/plot.sh <name> [<name> ...]
#
# Each name is read from results/<name>/ if present, else from
# reference_results/<name>/ (released models: model1_curriculum,
# structured_mix, structured_mix_long, two_stage). Figures go to
# results/plots/{benchmarks,heldout,scaling}/.
source "$(dirname "$0")/common.sh"
[[ $# -ge 1 ]] || { echo "usage: scripts/plot.sh <name> [<name> ...]" >&2; exit 2; }
BENCH=(); SCALE=()
for name in "$@"; do
  dir="results/$name"; [[ -d "$dir" ]] || dir="reference_results/$name"
  [[ -f "$dir/benchmarks/results.csv" ]] && BENCH+=(--rl "$name=$dir/benchmarks")
  [[ -f "$dir/scaling/results.csv" ]] && SCALE+=(--results "$dir/scaling")
done
BL=reference_results/baselines
if [[ ${#BENCH[@]} -gt 0 ]]; then
  $PY -m tddn.eval.plots "${BENCH[@]}" --baselines "$BL/benchmarks" --out results/plots/benchmarks \
    --note "All 95 benchmark miters, 5-23 qubits."
  $PY -m tddn.eval.plots "${BENCH[@]}" --baselines "$BL/benchmarks" --out results/plots/heldout \
    --heldout lists/benchmark_train.txt --note "Benchmark miters not used in training."
fi
if [[ ${#SCALE[@]} -gt 0 ]]; then
  $PY -m tddn.eval.scaling_plots --results "$BL/scaling" "${SCALE[@]}" --out results/plots/scaling
fi
