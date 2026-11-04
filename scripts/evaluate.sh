#!/usr/bin/env bash
# Evaluate a checkpoint.
#
#   scripts/evaluate.sh <checkpoint.zip> <name> [parts...]
#
# Parts (default: all), each written to results/<name>/<part>/results.csv:
#   benchmarks      95 benchmark miters, 5-23 qubits                 (minutes)
#   synthetic_test  held-out synthetic miters                         (minutes)
#   scaling         benchmarks at 24-50 qubits, 120 s per instance    (about 1 h)
#   nonequivalence  304 non-equivalent and 114 equivalent miters      (about 15 min)
#   random          31 unstructured random miters                     (about 15 min)
#
# Re-running resumes; delete results/<name>/<part> for a fresh run (for timing,
# run evaluations one at a time on an otherwise idle machine).
source "$(dirname "$0")/common.sh"
CKPT="${1:?usage: scripts/evaluate.sh <checkpoint.zip> <name> [parts...]}"
NAME="${2:?usage: scripts/evaluate.sh <checkpoint.zip> <name> [parts...]}"
shift 2
PARTS=("$@")
[[ ${#PARTS[@]} -eq 0 ]] && PARTS=(benchmarks synthetic_test scaling nonequivalence random)
OUT="results/$NAME"

for part in "${PARTS[@]}"; do
  echo "== $NAME: $part"
  case "$part" in
    benchmarks)
      $PY -m tddn.eval.runner --checkpoint "$CKPT" --out "$OUT/benchmarks" --list lists/benchmark_5to23q.txt
      $PY -m tddn.eval.compare table --rl "$NAME=$OUT/benchmarks" --baselines reference_results/baselines/benchmarks
      echo; echo "-- held out from training"
      $PY -m tddn.eval.compare table --rl "$NAME=$OUT/benchmarks" --baselines reference_results/baselines/benchmarks \
        --heldout lists/benchmark_train.txt ;;
    synthetic_test)
      $PY -m tddn.eval.runner --checkpoint "$CKPT" --out "$OUT/synthetic_test" --list lists/synthetic_test.txt ;;
    scaling)
      $PY -m tddn.eval.scaling --out "$OUT/scaling" --methods rl --model "$NAME=$CKPT"
      $PY -m tddn.eval.summary scaling "$OUT/scaling" reference_results/baselines/scaling ;;
    nonequivalence)
      $PY -m tddn.eval.fileset --out "$OUT/nonequivalence" --model "$NAME=$CKPT" --list lists/nonequivalence.txt
      $PY -m tddn.eval.summary fileset "$OUT/nonequivalence" ;;
    random)
      $PY -m tddn.eval.fileset --out "$OUT/random" --model "$NAME=$CKPT" --list lists/random_unstructured.txt
      $PY -m tddn.eval.summary fileset "$OUT/random" ;;
    *)
      echo "unknown part: $part" >&2; exit 2 ;;
  esac
done
