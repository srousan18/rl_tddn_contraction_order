# RL contraction ordering for tensor decision diagram networks

Code, data and trained models for reinforcement-learned contraction ordering
of tensor decision diagram (TDD) networks, applied to quantum circuit
equivalence checking.

Two circuits C and C' are equivalent exactly when the miter C · (C')⁻¹ is the
identity (up to a global phase). The miter is turned into a tensor network,
and the network is contracted pairwise into a single TDD, which is then
compared with the identity. The order of the pairwise contractions decides
how large the intermediate TDDs grow. A graph neural network policy, trained
with PPO, chooses the next pair to contract so that the peak TDD size stays
small. It is compared with cotengra planners (KaHyPar, Greedy, Betweenness,
Random Greedy) and the Counting heuristic.

## Installation

Python 3.13 (3.11 or newer works; the pinned versions were tested with 3.13).

```bash
scripts/setup.sh          # creates .venv, installs requirements.txt and the package, runs the tests
```

or by hand:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

`requirements.txt` pins every package (PyTorch 2.6.0, Stable-Baselines3 2.5.0,
PyTorch Geometric 2.6.1, Qiskit 1.3.2, cotengra 0.7.0, MQT QCEC 3.8.0).
MQT QCEC is only needed for circuit verification (`pip install -e ".[verify]"`
when installing from `pyproject.toml` instead of `requirements.txt`).

All scripts run from the repository root and use `.venv/bin/python` (override
with `PY=...`).

## Quick start

```bash
scripts/quick_test.sh                                     # a few minutes: models, scaling, baselines
scripts/evaluate.sh models/two_stage.zip two_stage benchmarks
scripts/plot.sh model1_curriculum two_stage               # figures from the reference results
```

## Repository structure

```
src/tddn/
  backend/      TDD library (third party, see below): TDDs, tensor networks, circuit tensorization
  network/      contraction graph, TDD tables, Counting prefix for large networks
  circuits/     QASM input, miters, circuit -> network -> graph, circuit generators
  envs/         contraction environment (Gymnasium)
  agents/       graph policy (GINE, GAT, TransformerConv, edge-scoring head) and loading
  training/     PPO training with the three-phase curriculum
  eval/         evaluation, scaling, miter lists, comparison tables, summaries, plots
  baselines/    cotengra planners and Counting, executed with the TDD backend
  verify/       identity check of the final TDD, statevector probe, MQT QCEC, corpus validation
scripts/        one script per step (setup, train, evaluate, baselines, validate, plot)
data/circuits/  all circuits (OpenQASM 2)
lists/          instance lists used by training and evaluation
models/         released checkpoints
reference_results/  results of the released models and of the baselines
tests/          test suite
```

## Data

All circuits are miters C · (C')⁻¹ named `combined_<family>_<n>q.qasm`, unless
noted otherwise.

| Directory | Contents |
| --- | --- |
| `data/circuits/benchmarks/combined/` | 230 benchmark miters from MQT Bench: BV, DJ, GHZ, QNN and W-State, 5-50 qubits. C is the algorithm-level circuit, C' its compilation to {rz, sx, x, cx}. |
| `data/circuits/benchmarks/{alg,compiled}/` | the separate pairs (C, C') for DJ, GHZ and W-State |
| `data/circuits/synthetic/` | 864 synthetic pairs from six seeded families (`tddn.circuits.synthetic`), with `alg/`, `compiled/`, `combined/` and `generation_log.jsonl` |
| `data/circuits/structured_mix/p{1,2,3}/` | 73 structured random miters used by the structured-mix curriculum (`tddn.circuits.structured_mix`) |
| `data/circuits/nonequivalence/` | 304 non-equivalent miters (one gate dropped, one CNOT flipped, one stray gate inserted, or C and C' from different instances) and 19 equivalent controls |
| `data/circuits/random_unstructured/` | 31 miters C · C⁻¹ of unstructured random circuits |

| List | Use |
| --- | --- |
| `benchmark_train.txt`, `benchmark_val.txt`, `benchmark_test.txt` | benchmark split (52 / 12 / 12; DJ, GHZ, QNN and W-State, 5-23 qubits; BV is never used for training) |
| `benchmark_5to23q.txt` | all 95 benchmark miters with 5-23 qubits |
| `synthetic_train.txt`, `synthetic_val.txt`, `synthetic_test.txt` | synthetic split (587 / 124 / 116), restricted to graphs within the observation bounds |
| `structured_mix_p{1,2,3}.txt` | random pools of the structured-mix curriculum phases |
| `nonequivalence.txt` | 304 non-equivalent miters and 114 equivalent ones (the 95 benchmark miters and 19 controls) |
| `nonequivalence_equivalent.txt` | the 114 equivalent miters of that set |
| `random_unstructured.txt` | the 31 unstructured random miters |

The equivalent circuits were checked with MQT QCEC and the statevector probe
(see [Verifying circuits](#verifying-circuits)). One unstructured random
miter (18 qubits) is inconclusive for QCEC within its time limit; its
statevector overlap is 1.0, and it is C · C⁻¹ by construction.

## Models

All policies share the architecture, the reward and the observation bounds
(87 graph nodes, 151 graph edges: the largest training graph, W-State at 23
qubits). Training uses PPO with learning rate 1e-4, batch size 256, 10 epochs,
clip range 0.2, entropy coefficient 0.01, value coefficient 0.5, 8 parallel
environments and 512 steps per environment and update, for 700k steps in
three curriculum phases (100k / 100k / 500k).

| Checkpoint | Training |
| --- | --- |
| `models/model1_curriculum.zip` | the benchmark curriculum, unseeded: 52 benchmark training miters |
| `models/structured_mix.zip` | the benchmark curriculum with structured random miters added: 100% / 50% / 29.7% of the episodes in phases 1 / 2 / 3 are random |
| `models/structured_mix_long.zip` | as `structured_mix`, with phase 3 extended to 700k steps (continued from the phase-2 weights) |
| `models/two_stage.zip` | stage 1 on the synthetic corpus (`models/two_stage_pretrained.zip`), then the benchmark curriculum initialised from the stage-1 weights |

Results (from `reference_results/`; peaks are TDD node counts, lower is
better, geometric means unless noted):

| | model1_curriculum | structured_mix | structured_mix_long | two_stage |
| --- | --- | --- | --- | --- |
| 95 benchmarks, 5-23q: mean / geometric mean peak | 52.3 / 45.8 | 62.5 / 49.0 | 48.5 / 44.0 | 45.3 / 41.8 |
| W-State, 5-23q (19): peak | 73.3 | 103.4 | 56.8 | 46.9 |
| W-State 24-50q solved within 120 s (of 27) | 26 | 15 | 13 | 26 |
| BV / DJ / GHZ / QNN 24-50q solved (of 108) | 108 | 108 | 108 | 108 |
| Equivalent miters (114): peak | 44.8 | 47.5 | 43.4 | 41.6 |
| Non-equivalent miters (304): peak | 79.0 | 85.9 | 80.4 | 78.1 |
| Non-equivalent miters: rejected / wrongly accepted | 296 / 0 | 296 / 0 | 296 / 0 | 296 / 0 |
| Unstructured random (31): solved, peak | 27, 1781 | 27, 1501 | 24, 1246 | 26, 1333 |

The policies are deterministic: re-evaluating a checkpoint reproduces its
peaks exactly (`tddn.eval.compare fingerprint`). Times depend on the machine.

## Training

```bash
scripts/train.sh <recipe> [seed] [extra arguments]
```

| Recipe | What it trains | Output |
| --- | --- | --- |
| `benchmark` | the benchmark curriculum | `runs/benchmark_s<seed>/` |
| `structured_mix` | benchmark curriculum plus the structured random pools | `runs/structured_mix_s<seed>/` |
| `structured_mix_long` | phase 3 for 700k steps from `runs/structured_mix_s<seed>/phase2.zip` (or `INIT=<checkpoint>`) | `runs/structured_mix_long_s<seed>/` |
| `pretrain` | stage 1 of the two-stage recipe (synthetic corpus) | `runs/pretrain_s<seed>/` |
| `two_stage` | stage 1 (skipped if `runs/pretrain_s<seed>/final.zip` exists), then stage 2 | `runs/two_stage_s<seed>/` |

The released models used seed 1000. Each 700k-step run takes several hours
on a recent CPU (no GPU is used); `two_stage` runs two of them. A run
directory holds `manifest.json` (configuration and exact circuit lists),
`phase{1,2,3}.zip`, `final.zip`, `best/` (best validation reward),
`checkpoints/` and `tensorboard/` (`tensorboard --logdir runs`).

Training is not bit-reproducible (the training environment has wall-clock
limits per step), so a retrained policy matches a released one statistically,
not instance by instance. Run-to-run variation is largest on W-State.

The training CLI (`python -m tddn.training.train --help`) takes the training
and validation lists, the curriculum (`--steps`, `--phase-steps`,
`--start-phase`), random pools (`--random-pools`, `--random-ratios`), weight
initialisation (`--init-from`) and the PPO settings.

## Evaluation

```bash
scripts/evaluate.sh <checkpoint.zip> <name> [parts...]
```

| Part | Instances | Output (`results/<name>/...`) | Summary |
| --- | --- | --- | --- |
| `benchmarks` | 95 benchmark miters, 5-23 qubits | `benchmarks/` | table against the reference baselines, all and held-out instances |
| `synthetic_test` | 116 held-out synthetic miters | `synthetic_test/` | |
| `scaling` | 135 benchmark miters, 24-50 qubits; W-State with a Counting prefix | `scaling/` | solved counts and peaks against the reference baselines |
| `nonequivalence` | 304 non-equivalent and 114 equivalent miters | `nonequivalence/` | identity / rejection counts and peaks |
| `random` | 31 unstructured random miters | `random/` | solved counts and peaks |

Every row records the peak TDD size, the times and the result of checking the
final TDD against the identity (`identity_status`). An equivalent miter must
end `identity`, a non-equivalent one `not_identity`. Evaluations append to
`results.csv` and resume when re-run; delete the directory for a fresh run.
Run timing measurements one at a time on an otherwise idle machine.

Further tools:

```bash
python -m tddn.eval.compare table --rl a=results/a/benchmarks --rl b=results/b/benchmarks \
    --baselines reference_results/baselines/benchmarks [--heldout lists/benchmark_train.txt]
python -m tddn.eval.compare fingerprint results/two_stage/benchmarks reference_results/two_stage/benchmarks
python -m tddn.eval.summary scaling results/two_stage/scaling reference_results/baselines/scaling
python -m tddn.eval.summary fileset results/two_stage/nonequivalence
python -m tddn.eval.summary runs results/s1000/nonequivalence results/s1001/nonequivalence \
    results/s1002/nonequivalence --baselines results/baselines/equivalent114
```

`summary runs` gives the mean and standard deviation across several training
runs on the 114 equivalent miters, and with `--baselines` the ratios to the
Greedy and Counting peaks and to the KaHyPar time.

## Baselines

```bash
scripts/baselines.sh benchmarks       # 95 benchmark miters, 5-23 qubits
scripts/baselines.sh scaling          # 24-50 qubits (many hours)
scripts/baselines.sh equivalent114    # the 114 equivalent miters of the non-equivalence set
```

The cotengra search is randomized and not reproducible between runs (no seed is
passed, KaHyPar trials draw their own seeds, and the search has a 10 s
wall-clock budget), so each cotengra method runs three times (per-instance
medians); Counting is deterministic. Paths are executed with the same TDD
backend as the policy, and every completed row ends with the identity check. The reference results of the first two sets are in
`reference_results/baselines/`. Single runs: `python -m tddn.baselines.runner
--help`.

## Verifying circuits

```bash
scripts/validate.sh benchmarks        # benchmark pairs and miters (5-23 qubits)
scripts/validate.sh synthetic         # the synthetic pairs (slow)
scripts/validate.sh structured_mix
scripts/validate.sh random
scripts/validate.sh nonequivalence    # every miter except the controls must be not_equivalent
```

`tddn.verify.validate` checks separate pairs (`pairs` mode) or miters against
the identity (`combined` mode) with MQT QCEC, retrying without the ZX checker
if the first run is inconclusive, and with the statevector probe up to
`--sv-max-qubits`. The verdict is in the `verdict` column of
`results/validation/<set>/results.csv`.

## Plotting

```bash
scripts/plot.sh two_stage model1_curriculum
```

Each name is read from `results/<name>/` or, if absent, from
`reference_results/<name>/`. The figures (PNG and PDF) go to
`results/plots/benchmarks/` and `results/plots/heldout/` (peak and time per
qubit count, per-instance panels per family) and `results/plots/scaling/` (one
figure per family and metric, 24-50 qubits).

## Generating circuits

```bash
python -m tddn.circuits.synthetic --root data/circuits/synthetic_new --total 1000 --seed 5000 --max-qubits 26
python -m tddn.circuits.structured_mix --out data/circuits/structured_mix_new \
    --train-list lists/benchmark_train.txt --candidates 1500 --lists lists/structured_mix_new
```

The synthetic generator draws an entanglement level per instance (low / mid /
high with probabilities 0.35 / 0.40 / 0.25), uses rotation angles on a π/16
grid, compiles C' with Qiskit at optimization level 3 to {rz, sx, x, cx}, and
keeps a pair only if MQT QCEC proves it equivalent. The BV-like family is
excluded by default (`--exclude`), since BV is a benchmark family.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests cover the identity check and the statevector probe on equivalent
and non-equivalent miters, gate decomposition before tensorization, the
environment, the baselines, QCEC validation, short training runs, and exact
reference peaks of every released model.

## Third-party code

`src/tddn/backend/` contains the TDD library of Hong et al.
([github.com/Veriqc/TDD](https://github.com/Veriqc/TDD)) with the extensions
of FTDD by Qirui Zhang ([github.com/QiruiZhang/FTDD](https://github.com/QiruiZhang/FTDD)).
The file headers list the changes. The benchmark circuits were generated with
[MQT Bench](https://github.com/cda-tum/mqt-bench).
