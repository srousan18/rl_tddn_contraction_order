#!/usr/bin/env bash
# Train a policy with one of the released recipes.
#
#   scripts/train.sh <recipe> [seed] [extra arguments for tddn.training.train]
#
# Recipes (output in runs/<recipe>_s<seed>/, final checkpoint final.zip):
#   benchmark            benchmark curriculum: 52 training circuits, 100k/100k/500k steps
#   structured_mix       benchmark curriculum plus structured random miters,
#                        100% / 50% / 29.7% random episodes in phases 1 / 2 / 3
#   structured_mix_long  structured_mix with phase 3 extended to 700k steps, continued
#                        from runs/structured_mix_s<seed>/phase2.zip (or INIT=<checkpoint>)
#   pretrain             two-stage, stage 1: the synthetic corpus, 100k/100k/500k steps
#   two_stage            stage 1 (pretrain), then stage 2: the benchmark curriculum
#                        initialised from the stage-1 weights
#
# Each 700k-step run takes several hours on a CPU; two_stage runs both stages.
source "$(dirname "$0")/common.sh"
RECIPE="${1:?usage: scripts/train.sh <recipe> [seed] [extra args]}"
SEED="${2:-1000}"
shift $(( $# >= 2 ? 2 : 1 ))

BENCH=(--train-list lists/benchmark_train.txt --val-list lists/benchmark_val.txt)
SYNTH=(--train-list lists/synthetic_train.txt --val-list lists/synthetic_val.txt)
POOLS=(--random-pools lists/structured_mix_p1.txt,lists/structured_mix_p2.txt,lists/structured_mix_p3.txt
       --random-ratios 1.0,0.5,0.2973)

train() { $PY -m tddn.training.train --seed "$SEED" "$@"; }

case "$RECIPE" in
  benchmark)
    train --out "runs/benchmark_s$SEED" "${BENCH[@]}" "$@" ;;
  structured_mix)
    train --out "runs/structured_mix_s$SEED" "${BENCH[@]}" "${POOLS[@]}" "$@" ;;
  structured_mix_long)
    INIT="${INIT:-runs/structured_mix_s$SEED/phase2.zip}"
    train --out "runs/structured_mix_long_s$SEED" "${BENCH[@]}" "${POOLS[@]}" \
      --init-from "$INIT" --start-phase 3 --phase-steps 100000,100000,700000 "$@" ;;
  pretrain)
    train --out "runs/pretrain_s$SEED" "${SYNTH[@]}" "$@" ;;
  two_stage)
    if [[ ! -f "runs/pretrain_s$SEED/final.zip" ]]; then
      train --out "runs/pretrain_s$SEED" "${SYNTH[@]}" "$@"
    fi
    train --out "runs/two_stage_s$SEED" "${BENCH[@]}" \
      --init-from "runs/pretrain_s$SEED/final.zip" "$@" ;;
  *)
    echo "unknown recipe: $RECIPE" >&2; exit 2 ;;
esac
