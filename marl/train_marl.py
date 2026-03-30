# =============================================================================
# RAG-Enhanced LLM-Coordinated MARL for 6G Edge-Cloud Task Offloading
# REINFORCE training loop — device-specific 18-dim observations
# =============================================================================

import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import M, K, LR, TOTAL_STEPS, RESULTS_DIR, MODELS_DIR, LOGS_DIR
from env.edge_cloud_env import EdgeCloudEnv, N_ACTIONS_PER_DEVICE
from marl.policies import MAPPOAgent

# -----------------------------------------------------------------------------
# Observation dimensions
# -----------------------------------------------------------------------------
# Per-device 18-dim obs layout:
#   channel rates to all M servers  obs[k*M : (k+1)*M]              (M=5)
#   queue lengths                   obs[K*M : K*M+M]                (M=5)
#   server CPU                      obs[K*M+M : K*M+2*M]            (M=5)
#   task_data[k]                    obs[K*M+2*M+k]                  (1)
#   task_cycles[k]                  obs[K*M+2*M+K+k]                (1)
#   task_deadline[k]                obs[K*M+2*M+2*K+k]              (1)
#   Total: 5+5+5+1+1+1 = 18
GLOBAL_OBS_DIM = K * M + 2 * M + 3 * K   # 170  (full env obs, used for slicing)
OBS_DIM        = M + M + M + 3            # 18   (per-device input to network)
N_ACTIONS      = N_ACTIONS_PER_DEVICE     # 7

# -----------------------------------------------------------------------------
# Training hyper-parameters
# -----------------------------------------------------------------------------
BATCH         = 256     # steps between REINFORCE updates
ENTROPY_COEF  = 0.01
GRAD_CLIP     = 0.5
EVAL_INTERVAL = 1_000
PROG_INTERVAL = 10_000


# =============================================================================
# Device obs builder
# =============================================================================

def _device_obs(obs: np.ndarray, k: int) -> np.ndarray:
    """
    Build the 18-dim device-specific observation for device k from the
    170-dim global observation vector.
    """
    return np.concatenate([
        obs[k * M : (k + 1) * M],                               # channel rates (M,)
        obs[K * M : K * M + M],                                  # queue lengths (M,)
        obs[K * M + M : K * M + 2 * M],                         # server CPU    (M,)
        obs[K * M + 2 * M + k       : K * M + 2 * M + k + 1],   # task_data[k]  (1,)
        obs[K * M + 2 * M + K + k   : K * M + 2 * M + K + k + 1],   # task_cycles[k] (1,)
        obs[K * M + 2 * M + 2*K + k : K * M + 2 * M + 2*K + k + 1], # task_deadline[k] (1,)
    ], dtype=np.float32)   # shape (18,)


# =============================================================================
# Checkpoint saving
# =============================================================================

def _save_checkpoint(agent: MAPPOAgent, step: int):
    data  = {
        "actor" : agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "step"  : step,
    }
    paths = [os.path.join(MODELS_DIR, "shared_agent.pt")]
    if os.path.exists("/kaggle/working"):
        paths.append("/kaggle/working/shared_agent.pt")
    for path in paths:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(data, path)
            print(f"Saved checkpoint step {step} -> {path}", flush=True)
        except Exception as e:
            print(f"Save failed {path}: {e}", flush=True)


# =============================================================================
# Training loop
# =============================================================================

def train(seed: int = 42):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_agent = MAPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, device=device)
    optimizer    = torch.optim.Adam(shared_agent.actor.parameters(), lr=LR)

    env = EdgeCloudEnv(seed=seed)
    obs, _ = env.reset(seed=seed)

    # Rollout buffers — cleared after every REINFORCE update
    log_probs_buf: list[torch.Tensor] = []   # mean log_prob over K devices per step
    entropies_buf: list[torch.Tensor] = []   # mean entropy  over K devices per step
    rewards_buf:   list[float]        = []

    # Metrics
    step_latencies: list[float] = []
    step_energies:  list[float] = []
    step_sla:       list[float] = []
    reward_history: list[float] = []

    best_reward = -float("inf")
    best_dir    = os.path.join(MODELS_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)

    print(
        f"Training start | total_steps={TOTAL_STEPS:,} | "
        f"obs_dim={OBS_DIM} | n_actions={N_ACTIONS} | "
        f"batch={BATCH} | device={device}"
    )
    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):

        # -----------------------------------------------------------------
        # Step 1: Each device k gets its own 18-dim obs and samples an action
        # -----------------------------------------------------------------
        device_actions = []
        step_log_probs = []
        step_entropies = []

        for k in range(K):
            dev_obs_t = torch.tensor(
                _device_obs(obs, k), dtype=torch.float32, device=device
            )
            logits  = shared_agent.actor(dev_obs_t)
            dist    = torch.distributions.Categorical(logits=logits)
            action  = dist.sample()
            device_actions.append(int(action.item()))
            step_log_probs.append(dist.log_prob(action))
            step_entropies.append(dist.entropy())

        # -----------------------------------------------------------------
        # Step 2: Execute all K device actions; clip reward
        # -----------------------------------------------------------------
        joint_action = (np.array(device_actions, dtype=np.int32),)
        obs, reward, terminated, truncated, info = env.step(joint_action)
        reward = max(reward, -5.0)

        log_probs_buf.append(torch.stack(step_log_probs).mean())
        entropies_buf.append(torch.stack(step_entropies).mean())
        rewards_buf.append(reward)

        step_latencies.append(info["mean_latency"])
        step_energies.append(info["mean_energy"])
        step_sla.append(info["mean_sla_violation"])

        if terminated or truncated:
            obs, _ = env.reset()

        # -----------------------------------------------------------------
        # Step 3: REINFORCE update every BATCH steps
        # -----------------------------------------------------------------
        if len(rewards_buf) >= BATCH:
            r = np.array(rewards_buf, dtype=np.float32)
            if r.std() > 1e-8:
                r = (r - r.mean()) / r.std()

            r_t      = torch.tensor(r, dtype=torch.float32, device=device)
            lp_t     = torch.stack(log_probs_buf)         # (BATCH,)
            ent_mean = torch.stack(entropies_buf).mean()  # scalar

            loss = -(lp_t * r_t).mean() - ENTROPY_COEF * ent_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_agent.actor.parameters(), GRAD_CLIP)
            optimizer.step()

            log_probs_buf = []
            entropies_buf = []
            rewards_buf   = []

        # -----------------------------------------------------------------
        # Step 4: Progress logging every EVAL_INTERVAL steps
        # -----------------------------------------------------------------
        if step % EVAL_INTERVAL == 0:
            win = slice(-EVAL_INTERVAL, None)
            recent_r         = rewards_buf if rewards_buf else [reward]
            mean_step_reward = float(np.mean(recent_r[-EVAL_INTERVAL:]))
            print(
                f"step {step:>8,} | "
                f"mean_step_reward {mean_step_reward:+.4f} | "
                f"latency {np.mean(step_latencies[win]):.4f}s | "
                f"energy {np.mean(step_energies[win]):.4e}J | "
                f"SLA viol {np.mean(step_sla[win]):.4f}s | "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )

        # -----------------------------------------------------------------
        # Step 5: Snapshot + best-model every 1000 steps
        # -----------------------------------------------------------------
        if step % 1_000 == 0:
            snapshot = float(np.mean(step_latencies[-1_000:]))
            reward_history.append(snapshot)

            if -snapshot > best_reward:
                best_reward = -snapshot
                torch.save(
                    {"actor": shared_agent.actor.state_dict(),
                     "critic": shared_agent.critic.state_dict()},
                    os.path.join(best_dir, "shared_agent.pt"),
                )

        # -----------------------------------------------------------------
        # Step 6: Checkpoint + progress bar every PROG_INTERVAL steps
        # -----------------------------------------------------------------
        if step % PROG_INTERVAL == 0:
            _save_checkpoint(shared_agent, step)

            pct     = 100.0 * step / TOTAL_STEPS
            filled  = int(30 * step / TOTAL_STEPS)
            bar     = "#" * filled + "." * (30 - filled)
            elapsed = time.time() - t0
            eta     = (elapsed / step) * (TOTAL_STEPS - step)

            last_1k = reward_history[-1] if reward_history else 0.0
            if len(reward_history) >= 11:
                delta = reward_history[-1] - reward_history[-11]
                trend = "IMPROVING" if delta < -0.005 else ("DEGRADING" if delta > 0.005 else "STABLE")
            else:
                trend = "STABLE"

            print(f"[{bar}] {pct:5.1f}%  step {step:,}/{TOTAL_STEPS:,}  ETA {eta/60:.1f}min")
            print(
                f"Progress: {pct:.0f}% | "
                f"Best latency: {-best_reward:.4f}s | "
                f"Last 1k mean latency: {last_1k:.4f}s | "
                f"Trend: {trend}"
            )

    # -----------------------------------------------------------------
    # Final checkpoint
    # -----------------------------------------------------------------
    _save_checkpoint(shared_agent, TOTAL_STEPS)
    print("Training complete.")
    return shared_agent


# =============================================================================
if __name__ == "__main__":
    train()
