"""A random-policy episode contracts a miter to the identity."""

from pathlib import Path

import numpy as np

from tddn.backend.TDD import Clear_TDD
from tddn.circuits.build import build_network
from tddn.config import M_MAX, N_MAX
from tddn.envs.contraction_env import TensorContractionEnv
from tddn.verify.tdd_identity import check_final_tdd

FIX = Path(__file__).parent / "fixtures"


def test_random_rollout_contracts_to_identity():
    net = build_network(FIX / "combined_ghz_5q.qasm")
    net.reinit_session()
    env = TensorContractionEnv(net.graph, net.tn, max_nodes=N_MAX, max_edges=M_MAX)
    env.set_test_mode(True)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    done = False
    for _ in range(200):
        feasible = np.flatnonzero(obs["feasible_actions"])
        action = int(rng.choice(feasible)) if len(feasible) else 0
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    assert done
    info = env.env_info()
    assert info["fully_contracted"]
    assert check_final_tdd(info, num_qubits=net.num_qubits)["status"] == "identity"
    Clear_TDD()


def test_training_rewards_are_negative_with_completion_bonus():
    net = build_network(FIX / "combined_ghz_5q.qasm")
    net.reinit_session()
    env = TensorContractionEnv(net.graph, net.tn, max_nodes=N_MAX, max_edges=M_MAX)
    obs, _ = env.reset()
    rewards, done = [], False
    while not done:
        obs, reward, done, truncated, _ = env.step(int(np.flatnonzero(obs["feasible_actions"])[0]))
        rewards.append(reward)
        done = done or truncated
    assert all(r <= 0 for r in rewards[:-1])
    Clear_TDD()
