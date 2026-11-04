"""Loading policy checkpoints."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from tddn.agents.policy import HybridGraphActorCriticPolicy


def load_policy(path: str | Path, env=None, device: str = "cpu") -> PPO:
    """Load a PPO checkpoint with this package's policy class."""

    return PPO.load(str(path), env=env, device=device,
                    custom_objects={"policy_class": HybridGraphActorCriticPolicy})
