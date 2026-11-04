#!/usr/bin/env bash
# Run the baselines (cotengra KaHyPar, Greedy, Betweenness, Random Greedy, three
# repeats each, and Counting).
#
#   scripts/baselines.sh benchmarks       95 benchmark miters, 5-23 qubits
#   scripts/baselines.sh scaling          benchmarks at 24-50 qubits (many hours)
#   scripts/baselines.sh equivalent114    the 114 equivalent miters of the non-equivalence set
#
# Output in results/baselines/<set>/; re-running resumes. Reference results of
# the first two sets are in reference_results/baselines/.
source "$(dirname "$0")/common.sh"
SET="${1:?usage: scripts/baselines.sh benchmarks|scaling|equivalent114}"
OUT="results/baselines/$SET"
case "$SET" in
  benchmarks)
    $PY -m tddn.baselines.runner --out "$OUT" --list lists/benchmark_5to23q.txt ;;
  equivalent114)
    $PY -m tddn.baselines.runner --out "$OUT" --list lists/nonequivalence_equivalent.txt ;;
  scaling)
    $PY -m tddn.eval.scaling --out "$OUT" \
      --methods counting,cotengra_kahypar,cotengra_greedy,cotengra_between,cotengra_rgreedy ;;
  *)
    echo "unknown set: $SET" >&2; exit 2 ;;
esac
