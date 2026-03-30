# =============================================================================
# RAG-Enhanced LLM-Coordinated MARL for 6G Edge-Cloud Task Offloading
# Baseline evaluation — compares 4 policies over 1000 steps each
# =============================================================================

import os
import sys
import csv

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from config import (
    M, K,
    EMBEDDING_DIM, RAG_TOP_K,
    MODELS_DIR, RESULTS_DIR,
)
from env.edge_cloud_env import EdgeCloudEnv, N_ACTIONS_PER_DEVICE
from rag.state_serializer import StateSerializer
from rag.embedder import StateEmbedder
from rag.vector_store import VectorStore
from llm.coordinator import LLMCoordinator
from llm.hint_parser import HintParser
from marl.policies import MAPPOAgent

EVAL_STEPS = 1_000
EVAL_SEED  = 0
OBS_DIM    = K * M + 2 * M + 3 * K + K   # 170 + K = 190 (device-specific obs)


# =============================================================================
# Helpers
# =============================================================================

def _run_episode(env, action_fn, steps=EVAL_STEPS):
    """
    Roll out `steps` environment steps using `action_fn`.

    Parameters
    ----------
    env       : EdgeCloudEnv (already reset by caller)
    action_fn : callable(obs) → tuple of M np.ndarray each shape (K,)

    Returns
    -------
    dict with keys mean_latency, mean_energy, mean_sla
    """
    latencies = []
    energies  = []
    slas      = []

    obs, _ = env.reset(seed=EVAL_SEED)

    for _ in range(steps):
        joint_action = action_fn(obs)
        obs, _reward, terminated, truncated, info = env.step(joint_action)
        latencies.append(info["mean_latency"])
        energies.append(info["mean_energy"])
        slas.append(info["mean_sla_violation"])
        if terminated or truncated:
            obs, _ = env.reset()

    return {
        "mean_latency": float(np.mean(latencies)),
        "mean_energy" : float(np.mean(energies)),
        "mean_sla"    : float(np.mean(slas)),
    }


def _load_shared_agent(device):
    """
    Load the parameter-sharing MAPPOAgent from models/shared_agent.pt.
    Returns None if the checkpoint is missing.
    """
    path = os.path.join(MODELS_DIR, "shared_agent.pt")
    if not os.path.exists(path):
        return None
    agent = MAPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_PER_DEVICE, device=device)
    ckpt  = torch.load(path, map_location=device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()
    agent.critic.eval()
    return agent


_EYE_K_EVAL = np.eye(K, dtype=np.float32)   # (K, K) one-hot matrix


def _marl_action_fn(shared_agent):
    """
    Returns an action_fn that calls shared_agent K times — once per device —
    each with a device-specific obs (global obs + one-hot device index).
    All M edge nodes receive the same K-device action array (parameter sharing).
    """
    device = shared_agent.device

    def action_fn(obs):
        device_actions = []
        with torch.no_grad():
            for k in range(K):
                device_obs = np.concatenate([obs, _EYE_K_EVAL[k]])
                obs_t      = torch.tensor(device_obs, dtype=torch.float32, device=device)
                logits     = shared_agent.actor(obs_t)
                dist       = torch.distributions.Categorical(logits=logits)
                device_actions.append(int(dist.sample().item()))
        joint_action = tuple(
            np.array(device_actions, dtype=np.int32) for _ in range(M)
        )
        return joint_action

    return action_fn


# =============================================================================
# Baseline 1 — Random
# =============================================================================

def run_random(env):
    rng = np.random.default_rng(EVAL_SEED)

    def action_fn(obs):
        return tuple(
            rng.integers(0, N_ACTIONS_PER_DEVICE, size=K).astype(np.int32)
            for _ in range(M)
        )

    return _run_episode(env, action_fn)


# =============================================================================
# Baseline 2 — Greedy (shortest queue)
# =============================================================================

def run_greedy(env):
    # Queue lengths live at obs[K*M : K*M+M]
    q_start = K * M
    q_end   = K * M + M

    def action_fn(obs):
        queues   = obs[q_start:q_end]          # shape (M,)
        best_m   = int(np.argmin(queues))       # 0-indexed edge node
        action   = best_m + 1                  # env action: 1…M → edge node
        return tuple(np.full(K, action, dtype=np.int32) for _ in range(M))

    return _run_episode(env, action_fn)


# =============================================================================
# Baseline 3 — MARL only (parameter-sharing shared_agent, greedy eval)
# =============================================================================

def run_marl_only(env, device):
    shared_agent = _load_shared_agent(device)
    if shared_agent is None:
        print("  [MARL only] models/shared_agent.pt not found — skipping.")
        return None
    return _run_episode(env, _marl_action_fn(shared_agent))


# =============================================================================
# Baseline 4 — Full system (MARL + RAG + LLM)
# =============================================================================

def run_full_system(env, device):
    print("  [Full system] LLM training pending — skipping.")
    return None


# =============================================================================
# Table formatting
# =============================================================================

def _print_table(results):
    col_w = [20, 16, 16, 16]
    sep   = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr   = "| {:<18} | {:>14} | {:>14} | {:>14} |".format(
        "Baseline", "Latency (s)", "Energy (J)", "SLA viol (s)"
    )

    print()
    print(sep)
    print(hdr)
    print(sep)
    for name, r in results:
        if r is None:
            row = "| {:<18} | {:>14} | {:>14} | {:>14} |".format(
                name, "N/A", "N/A", "N/A"
            )
        else:
            row = "| {:<18} | {:>14.6f} | {:>14.4e} | {:>14.6f} |".format(
                name,
                r["mean_latency"],
                r["mean_energy"],
                r["mean_sla"],
            )
        print(row)
    print(sep)
    print()


def _save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "mean_latency_s", "mean_energy_J", "mean_sla_violation_s"])
        for name, r in results:
            if r is None:
                writer.writerow([name, "", "", ""])
            else:
                writer.writerow([
                    name,
                    f"{r['mean_latency']:.6f}",
                    f"{r['mean_energy']:.6e}",
                    f"{r['mean_sla']:.6f}",
                ])
    print(f"Results saved → {path}")


# =============================================================================
# Entry point
# =============================================================================

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env    = EdgeCloudEnv(seed=EVAL_SEED)

    print(f"Evaluating {EVAL_STEPS} steps per baseline | device={device}")
    print()

    baselines = [
        ("Random",      lambda: run_random(env)),
        ("Greedy",      lambda: run_greedy(env)),
        ("MARL only",   lambda: run_marl_only(env, device)),
        ("Full system", lambda: run_full_system(env, device)),
    ]

    results = []
    for name, fn in baselines:
        print(f"Running: {name} ...", end=" ", flush=True)
        r = fn()
        status = (
            f"latency={r['mean_latency']:.4f}s  energy={r['mean_energy']:.3e}J  sla={r['mean_sla']:.4f}s"
            if r is not None else "skipped"
        )
        print(status)
        results.append((name, r))

    _print_table(results)
    _save_csv(results, os.path.join(RESULTS_DIR, "baseline_comparison.csv"))


if __name__ == "__main__":
    evaluate()
