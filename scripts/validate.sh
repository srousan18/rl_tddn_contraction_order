#!/usr/bin/env bash
# Check circuit equivalence with MQT QCEC and the statevector probe.
#
#   scripts/validate.sh benchmarks       benchmark pairs (dj, ghz, wstate) and all miters 5-23q
#   scripts/validate.sh synthetic        the 864 synthetic pairs (slow)
#   scripts/validate.sh structured_mix   the 73 structured random miters
#   scripts/validate.sh random           the 31 unstructured random miters
#   scripts/validate.sh nonequivalence   the non-equivalence set: every miter except
#                                        cross_instance_control must be not_equivalent
#
# Output in results/validation/<set>/results.csv (column verdict).
source "$(dirname "$0")/common.sh"
SET="${1:?usage: scripts/validate.sh benchmarks|synthetic|structured_mix|random|nonequivalence}"
OUT="results/validation"
V="$PY -m tddn.verify.validate"
case "$SET" in
  benchmarks)
    $V pairs --root data/circuits/benchmarks --sv-max-qubits 23 --out "$OUT/benchmark_pairs"
    $V combined --root data/circuits/benchmarks --min-qubits 5 --max-qubits 23 --sv-max-qubits 23 \
      --out "$OUT/benchmark_miters" ;;
  synthetic)
    $V pairs --root data/circuits/synthetic --out "$OUT/synthetic_pairs" ;;
  structured_mix)
    $V combined --out "$OUT/structured_mix" $(cat lists/structured_mix_p{1,2,3}.txt) ;;
  random)
    $V combined --out "$OUT/random_unstructured" $(cat lists/random_unstructured.txt) ;;
  nonequivalence)
    # Exit status 1 is expected here: non-equivalent miters are proven not equivalent.
    $V combined --out "$OUT/nonequivalence" $(grep -v "/combined/" lists/nonequivalence.txt) || true ;;
  *)
    echo "unknown set: $SET" >&2; exit 2 ;;
esac
