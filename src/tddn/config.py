"""Project paths, released checkpoints and observation bounds."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("TDDN_DATA", PROJECT_ROOT / "data"))
BENCHMARK_DIR = DATA_DIR / "circuits" / "benchmarks" / "combined"
MODELS_DIR = Path(os.environ.get("TDDN_MODELS", PROJECT_ROOT / "models"))
RUNS_DIR = PROJECT_ROOT / "runs"

# Released checkpoints (see README.md, "Models").
MODELS = {
    "model1_curriculum": MODELS_DIR / "model1_curriculum.zip",
    "structured_mix": MODELS_DIR / "structured_mix.zip",
    "structured_mix_long": MODELS_DIR / "structured_mix_long.zip",
    "two_stage": MODELS_DIR / "two_stage.zip",
}
TWO_STAGE_PRETRAINED = MODELS_DIR / "two_stage_pretrained.zip"

# Observation bounds (graph nodes, graph edges) of every released policy: the
# largest training graph, W-State at 23 qubits.
N_MAX = 87
M_MAX = 151
