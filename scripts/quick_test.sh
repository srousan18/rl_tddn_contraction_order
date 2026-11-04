#!/usr/bin/env bash
# A few-minute end-to-end check: every released model on six benchmark miters
# (peaks compared with reference_results), two scaling rows, and two baselines.
source "$(dirname "$0")/common.sh"
OUT=results/quick_test
rm -rf "$OUT"
CIRCUITS=(data/circuits/benchmarks/combined/combined_{bv_12,dj_9,ghz_11,qnn_8,wstate_10,wstate_17}q.qasm)
for m in model1_curriculum structured_mix structured_mix_long two_stage; do
  echo "== $m"
  $PY -m tddn.eval.runner --checkpoint "models/$m.zip" --out "$OUT/$m" --recheck 1 "${CIRCUITS[@]}" | tail -1
  $PY -m tddn.eval.compare fingerprint "$OUT/$m" "reference_results/$m/benchmarks"
done
echo "== scaling: dj 30q (policy alone) and wstate 30q (Counting prefix + policy)"
$PY -m tddn.eval.scaling --out "$OUT/scaling" --methods rl --families dj,wstate --sizes 30 \
  --model two_stage=models/two_stage.zip | grep -E "^\[" || true
echo "== baselines"
$PY -m tddn.baselines.runner --out "$OUT/baselines" --methods counting,greedy --repeats 1 \
  "${CIRCUITS[@]:0:3}" | tail -1
echo "quick test passed"
