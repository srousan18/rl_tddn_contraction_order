"""A minimal training run (all three curriculum phases) produces a loadable checkpoint."""

from pathlib import Path

from tddn.config import BENCHMARK_DIR


def test_short_curriculum_run(tmp_path):
    from tddn.agents.loading import load_policy
    from tddn.training.train import main

    circuits = [BENCHMARK_DIR / f"combined_ghz_{n}q.qasm" for n in (5, 6, 7)]
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("\n".join(map(str, circuits)) + "\n")
    val.write_text(str(BENCHMARK_DIR / "combined_dj_5q.qasm") + "\n")
    pool = tmp_path / "pool.txt"
    pool.write_text(str(BENCHMARK_DIR / "combined_dj_6q.qasm") + "\n")
    out = tmp_path / "run"
    main(["--seed", "1", "--out", str(out), "--train-list", str(train), "--val-list", str(val),
          "--num-envs", "2", "--n-steps", "16", "--phase-steps", "32,32,32",
          "--eval-freq", "16", "--random-pools", f"{pool},{pool},{pool}",
          "--random-ratios", "1.0,0.5,0.0"])
    for name in ("manifest.json", "phase1.zip", "phase2.zip", "phase3.zip", "final.zip"):
        assert (out / name).is_file(), name
    model = load_policy(out / "final.zip")
    assert Path(out / "final.zip").stat().st_size > 0 and model.n_steps == 16
