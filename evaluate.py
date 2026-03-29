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
OBS_DIM    = K * M + 2 * M + 3 * K   # 170


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


def _load_agents(device):
    """
    Load trained MAPPOAgent checkpoints from MODELS_DIR.
    Returns None if any checkpoint is missing.
    """
    agents = []
    for i in range(M):
        path = os.path.join(MODELS_DIR, f"agent_{i}.pt")
        if not os.path.exists(path):
            return None
        agent = MAPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_PER_DEVICE, device=device)
        ckpt  = torch.load(path, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.actor.eval()
        agent.critic.eval()
        agents.append(agent)
    return agents


def _marl_action_fn(agents, device):
    """Returns an action_fn that queries all M trained agents (no LLM)."""
    def action_fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        actions = []
        with torch.no_grad():
            for agent in agents:
                act, _ = agent.act(obs_t)
                actions.append(int(act.item()))
        return tuple(np.full(K, actions[i], dtype=np.int32) for i in range(M))
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
# Baseline 3 — MARL only (trained agents, no LLM)
# =============================================================================

def run_marl_only(env, device):
    agents = _load_agents(device)
    if agents is None:
        print("  [MARL only] No checkpoints found in models/ — skipping.")
        return None
    return _run_episode(env, _marl_action_fn(agents, device))


# =============================================================================
# Baseline 4 — Full system (MARL + RAG + LLM)
# =============================================================================

def run_full_system(env, device):
    agents = _load_agents(device)
    if agents is None:
        print("  [Full system] No checkpoints found in models/ — skipping.")
        return None

    serializer   = StateSerializer()
    embedder     = StateEmbedder()
    vector_store = VectorStore(embedding_dim=EMBEDDING_DIM)
    coordinator  = LLMCoordinator()
    hint_parser  = HintParser()

    current_hint = None
    step_idx     = [0]    # mutable counter inside closure

    # Build a base MARL action fn for execution
    marl_fn = _marl_action_fn(agents, device)

    def action_fn(obs):
        step_idx[0] += 1

        # Serialize + embed every step for RAG storage
        state_text = serializer.serialize(obs)
        embedding  = embedder.embed(state_text)

        # LLM coordination every N_COORDINATION steps (use 10 here)
        nonlocal current_hint
        if step_idx[0] % 10 == 0:
            if len(vector_store) > 0:
                results        = vector_store.retrieve(embedding, top_k=RAG_TOP_K)
                context_string = vector_store.build_context_string(results)
            else:
                context_string = "No past situations stored yet."
            current_hint = coordinator.get_hint(state_text, context_string)

        # Get base MARL action
        joint_action = marl_fn(obs)

        # Override action for preferred nodes (reward shaping at eval = routing hint)
        if current_hint and current_hint["preferred_nodes"]:
            best_node = current_hint["preferred_nodes"][0]   # 1-indexed
            action    = int(best_node)
            joint_action = tuple(
                np.full(K, action, dtype=np.int32) for _ in range(M)
            )

        # Store in RAG (use 0.0 as placeholder reward — we don't know it yet)
        vector_store.add(embedding, state_text, 0.0)

        return joint_action

    return _run_episode(env, action_fn)


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
