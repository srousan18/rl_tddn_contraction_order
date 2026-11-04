"""Train the contraction policy with PPO and a three-phase curriculum.

Training and validation circuits are given as list files. Training circuits
are sorted by graph size; phase 1 uses the smallest third, phase 2 the
smallest two thirds and phase 3 all of them, for 1/7, 1/7 and 5/7 of
``--steps`` (default 700k: 100k / 100k / 500k) or ``--phase-steps a,b,c``.

Optional random pools (``--random-pools p1,p2,p3`` with ``--random-ratios
r1,r2,r3``) give each phase a pool of extra circuits; an episode draws from
the phase's pool with probability ``r_k`` and from the training circuits
otherwise. ``--init-from`` initialises the policy weights from a checkpoint
(optimizer and reward normalisation start fresh); ``--start-phase k`` skips
the phases before ``k``.

Outputs, in ``--out``:

    manifest.json        configuration and exact training / validation files
    phase{1,2,3}.zip     checkpoint after each phase
    final.zip            final checkpoint
    best/                best checkpoint by validation reward
    checkpoints/         periodic checkpoints
    tensorboard/         training curves

Example (the benchmark curriculum):

    python -m tddn.training.train --seed 1000 --out runs/benchmark_s1000 \\
        --train-list lists/benchmark_train.txt --val-list lists/benchmark_val.txt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as im
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as sb_logger_configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from tddn.agents.loading import load_policy
from tddn.agents.policy import HybridGraphActorCriticPolicy
from tddn.backend.TDD import Clear_TDD
from tddn.circuits.build import build_network
from tddn.circuits.qasm import read_list
from tddn.envs.contraction_env import TensorContractionEnv

# PPO hyperparameters (per environment: n_steps transitions per update).
PPO_KWARGS = dict(n_steps=512, batch_size=256, n_epochs=10, learning_rate=1e-4,
                  clip_range=0.2, ent_coef=0.01, vf_coef=0.5)


def init_weights(model, path) -> None:
    """Copy the policy weights of checkpoint ``path`` into ``model``.

    The weights do not depend on the observation bounds, so a checkpoint
    trained with other bounds can initialise this run; tensor names and
    shapes must match.
    """

    src = load_policy(path).policy.state_dict()
    dst = model.policy.state_dict()
    bad = sorted(set(src) ^ set(dst)) + [k for k in src.keys() & dst.keys()
                                         if src[k].shape != dst[k].shape]
    if bad:
        raise SystemExit(f"--init-from {path}: incompatible weights {bad[:5]}")
    model.policy.load_state_dict(src)
    print(f">>> initialised from {path}: {len(src)} tensors")


def graph_bounds(qasm_files):
    """Largest (nodes, edges) over the contraction graphs of ``qasm_files``."""
    max_nodes = max_edges = 0
    for path in qasm_files:
        try:
            net = build_network(path)
            max_nodes = max(max_nodes, net.graph.number_of_nodes())
            max_edges = max(max_edges, net.graph.number_of_edges())
        except Exception as exc:
            print(f"skipping {path}: {exc}")
        finally:
            Clear_TDD()
    return max_nodes, max_edges


class CircuitSamplingEnv(gym.Env):
    """Draws a circuit per episode; circuits beyond the bounds are redrawn."""

    def __init__(self, paths, n_max, m_max, env_kwargs=None, random_ratio=0.0,
                 random_paths=None):
        super().__init__()
        self.paths = paths
        self.n_max, self.m_max = n_max, m_max
        self.env_kwargs = env_kwargs or {}
        self.random_ratio = random_ratio
        self.random_paths = random_paths or []
        self._load()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _load(self):
        for _ in range(50):
            Clear_TDD()
            draw = random.random() if self.paths else 0.0
            use_random = bool(self.random_paths) and (not self.paths or draw < self.random_ratio)
            try:
                path = random.choice(self.random_paths if use_random else self.paths)
                net = build_network(path)
                if (net.graph.number_of_nodes() <= self.n_max
                        and net.graph.number_of_edges() <= self.m_max):
                    net.reinit_session()
                    self.env = TensorContractionEnv(net.graph, net.tn, max_nodes=self.n_max,
                                                    max_edges=self.m_max, **self.env_kwargs)
                    return
            except Exception:
                continue
        raise RuntimeError("no circuit within the bounds found in 50 draws")

    def reset(self, **kw):
        self._load()
        return self.env.reset(**kw)

    def step(self, action):
        return self.env.step(action)


class FixedSequenceEnv(gym.Env):
    """Cycles through the validation circuits in order."""

    def __init__(self, paths, n_max, m_max, env_kwargs=None):
        super().__init__()
        self.paths, self.n_max, self.m_max = paths, n_max, m_max
        self.env_kwargs = env_kwargs or {}
        self._idx = 0
        self._load()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _load(self):
        Clear_TDD()
        net = build_network(self.paths[self._idx % len(self.paths)])
        net.reinit_session()
        self.env = TensorContractionEnv(net.graph, net.tn, max_nodes=self.n_max,
                                        max_edges=self.m_max, **self.env_kwargs)

    def reset(self, **kw):
        self._idx += 1
        self._load()
        return self.env.reset(**kw)

    def step(self, action):
        return self.env.step(action)


def write_manifest(out: Path, args, bounds, train_files, val_files, phases) -> None:
    manifest = {
        "seed": args.seed,
        "bounds": list(bounds),
        "train_list": args.train_list,
        "val_list": args.val_list,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "phases": [{"circuits": len(p), "steps": s} for p, s in phases],
        "start_phase": args.start_phase,
        "random_pools": args.random_pools,
        "random_ratios": args.random_ratios,
        "init_from": args.init_from,
        "init_from_sha256": (hashlib.sha256(Path(args.init_from).read_bytes()).hexdigest()
                             if args.init_from else None),
        "reward_exponent": args.reward_exponent,
        "num_envs": args.num_envs,
        "ppo": {**PPO_KWARGS, "n_steps": args.n_steps},
        "versions": {pkg: im.version(pkg) for pkg in
                     ("torch", "stable_baselines3", "torch-geometric", "gymnasium",
                      "qiskit", "numpy")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--train-list", required=True, help="file listing the training circuits")
    ap.add_argument("--val-list", required=True, help="file listing the validation circuits")
    ap.add_argument("--steps", type=int, default=700_000, help="total timesteps")
    ap.add_argument("--phase-steps", default=None,
                    help="timesteps per curriculum phase 'a,b,c' (default 1/7, 1/7, 5/7)")
    ap.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3],
                    help="skip the phases before this one (requires --init-from)")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialise the weights")
    ap.add_argument("--random-pools", default=None,
                    help="three list files of extra circuits, one per phase")
    ap.add_argument("--random-ratios", default=None,
                    help="share of episodes drawn from the pool, per phase, e.g. 1.0,0.5,0.3")
    ap.add_argument("--bounds", default="87,151",
                    help="observation bounds 'nodes,edges'; training circuits must fit")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=PPO_KWARGS["n_steps"],
                    help="PPO rollout length per environment")
    ap.add_argument("--reward-exponent", type=float, default=1.5)
    ap.add_argument("--eval-freq", type=int, default=10 * 512,
                    help="validation interval in calls per environment")
    ap.add_argument("--checkpoint-freq", type=int, default=50_000,
                    help="periodic checkpoint interval in timesteps")
    args = ap.parse_args(argv)

    if args.init_from and not Path(args.init_from).is_file():
        raise SystemExit(f"--init-from: no such checkpoint {args.init_from}")
    if args.start_phase > 1 and not args.init_from:
        raise SystemExit("--start-phase > 1 requires --init-from")
    total = args.steps
    warmup = max(1, total // 7)
    steps_per = [warmup, warmup, total - 2 * warmup]
    if args.phase_steps:
        steps_per = [int(x) for x in args.phase_steps.split(",")]
        if len(steps_per) != 3 or min(steps_per) < 1:
            raise SystemExit("--phase-steps needs three positive integers")
    pools, ratios = [[], [], []], [0.0, 0.0, 0.0]
    if args.random_pools or args.random_ratios:
        if not (args.random_pools and args.random_ratios):
            raise SystemExit("--random-pools and --random-ratios go together")
        pools = [read_list(f) for f in args.random_pools.split(",")]
        ratios = [float(x) for x in args.random_ratios.split(",")]
        if len(pools) != 3 or len(ratios) != 3 or not all(pools):
            raise SystemExit("--random-pools/--random-ratios need three non-empty entries")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_files = [Path(p) for p in read_list(args.train_list)]
    val_files = [Path(p) for p in read_list(args.val_list)]
    missing = [str(f) for f in train_files + val_files if not f.exists()]
    if missing:
        raise SystemExit(f"missing listed files: {missing[:3]} (+{max(0, len(missing) - 3)} more)")

    n_max, m_max = (int(x) for x in args.bounds.split(","))
    need = graph_bounds(sorted({str(p) for p in train_files + val_files}))
    if need[0] > n_max or need[1] > m_max:
        raise SystemExit(f"training circuits need bounds {need}, larger than {(n_max, m_max)}")
    print(f">>> seed {args.seed}, bounds {n_max}/{m_max}, "
          f"train {len(train_files)}, val {len(val_files)}")

    # Curriculum: training circuits sorted by graph size.
    sized = []
    for p in train_files:
        try:
            sized.append((build_network(p).graph.number_of_nodes(), p))
        except Exception:
            sized.append((9999, p))
        finally:
            Clear_TDD()
    sized.sort()
    paths = [p for _, p in sized]
    third = len(paths) // 3
    phases = [(paths[:third], steps_per[0]), (paths[:2 * third], steps_per[1]),
              (paths, steps_per[2])]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tb_dir = out / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    write_manifest(out, args, (n_max, m_max), train_files, val_files, phases)

    env_kwargs = dict(reward_exponent=args.reward_exponent)
    val_env = VecNormalize(VecMonitor(DummyVecEnv(
        [lambda: FixedSequenceEnv(val_files, n_max, m_max, env_kwargs)])),
        norm_obs=False, norm_reward=False, training=False)
    callbacks = [
        EvalCallback(val_env, best_model_save_path=str(out / "best"), log_path=str(tb_dir),
                     eval_freq=args.eval_freq, n_eval_episodes=10, deterministic=True),
        CheckpointCallback(save_freq=max(args.checkpoint_freq // args.num_envs, 1),
                           save_path=str(out / "checkpoints"), name_prefix="step"),
    ]

    def build_vec(phase_paths, ratio, pool):
        def make():
            return CircuitSamplingEnv(phase_paths, n_max, m_max, env_kwargs, ratio, pool)
        vec = SubprocVecEnv([make for _ in range(args.num_envs)], start_method="fork")
        return VecNormalize(VecMonitor(vec), norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = None
    for i, (phase_paths, steps) in enumerate(phases, 1):
        if i < args.start_phase:
            print(f">>> phase {i}: skipped (weights from {args.init_from})")
            continue
        print(f">>> phase {i}: {len(phase_paths)} circuits, {steps} steps"
              + (f", random pool {len(pools[i - 1])} at ratio {ratios[i - 1]}"
                 if pools[i - 1] else ""))
        vec = build_vec(phase_paths, ratios[i - 1], pools[i - 1])
        first = model is None
        if first:
            model = PPO(HybridGraphActorCriticPolicy, vec, verbose=1, seed=args.seed,
                        policy_kwargs=dict(activation_fn=torch.nn.LeakyReLU), device="cpu",
                        tensorboard_log=str(tb_dir), **{**PPO_KWARGS, "n_steps": args.n_steps})
            model.set_logger(sb_logger_configure(str(tb_dir), ["stdout", "tensorboard"]))
            if args.init_from:
                init_weights(model, args.init_from)
        else:
            model.set_env(vec)
        model.learn(total_timesteps=steps, reset_num_timesteps=first, callback=callbacks,
                    progress_bar=True)
        model.save(out / f"phase{i}.zip")
        vec.close()
    model.save(out / "final.zip")
    print(f"saved {out / 'final.zip'}")


if __name__ == "__main__":
    main()
