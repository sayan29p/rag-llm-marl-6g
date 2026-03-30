# =============================================================================
# RAG-Enhanced LLM-Coordinated MARL for 6G Edge-Cloud Task Offloading
# Hierarchical training loop — parameter-sharing MARL
# =============================================================================

import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    M, K,
    GAMMA, LR, BATCH_SIZE, HIDDEN_DIM,
    N_COORDINATION, TOTAL_STEPS,
    W1, W2, W3,
    RAG_TOP_K, EMBEDDING_DIM,
    RESULTS_DIR, MODELS_DIR, LOGS_DIR,
)
from env.edge_cloud_env import EdgeCloudEnv, N_ACTIONS_PER_DEVICE
from env.channel_model import ChannelModel
from rag.state_serializer import StateSerializer
from rag.embedder import StateEmbedder
from rag.vector_store import VectorStore
from llm.coordinator import LLMCoordinator
from llm.hint_parser import HintParser
from marl.policies import MAPPOAgent


def _save_checkpoint(shared_agent, step):
    import os
    import torch
    save_data = {
        "actor": shared_agent.actor.state_dict(),
        "critic": shared_agent.critic.state_dict(),
        "step": step
    }
    paths = [
        os.path.join(MODELS_DIR, "shared_agent.pt"),
    ]
    kaggle_path = "/kaggle/working/shared_agent.pt"
    if os.path.exists("/kaggle/working"):
        paths.append(kaggle_path)
    for path in paths:
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(save_data, path)
            print(f"Saved checkpoint step {step} → {path}", flush=True)
        except Exception as e:
            print(f"Save failed {path}: {e}", flush=True)

# -----------------------------------------------------------------------------
# PPO hyper-parameters (not in config — training-loop internals)
# -----------------------------------------------------------------------------
PPO_EPSILON   = 0.2     # clipping range for importance-sampling ratio
ENTROPY_COEF  = 0.01    # entropy regularisation coefficient
CRITIC_COEF   = 0.5     # critic loss weight (CleanRL default)
PPO_EPOCHS    = 4       # gradient update passes per collected batch
GAE_LAMBDA    = 0.90    # GAE λ for return computation
GRAD_CLIP     = 0.5     # global gradient-norm clip
EVAL_INTERVAL = 1_000   # steps between progress log lines

# Observation dimension (K=20, M=5):  K*M + 2M + 3K + K = 100 + 10 + 60 + 20 = 190
# Global obs (170) + one-hot device index (K=20) = 190
OBS_DIM = K * M + 2 * M + 3 * K + K


# =============================================================================
# 1.  setup()
# =============================================================================

def setup(seed: int = 42):
    """
    Instantiate and return all system components.

    Parameter-sharing MARL: a single shared_agent network is used by all
    M edge nodes and all K devices.  All agents share weights; only the
    global observation differs between time steps.

    Returns
    -------
    env          : EdgeCloudEnv
    shared_agent : MAPPOAgent  — one network shared across all agents
    serializer   : StateSerializer
    embedder     : StateEmbedder
    vector_store : VectorStore
    coordinator  : LLMCoordinator
    hint_parser  : HintParser
    """
    env = EdgeCloudEnv(seed=seed)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_agent = MAPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_PER_DEVICE, device=device)

    serializer   = StateSerializer()
    embedder     = StateEmbedder()
    vector_store = VectorStore(embedding_dim=EMBEDDING_DIM)
    coordinator  = LLMCoordinator()
    hint_parser  = HintParser()

    return env, shared_agent, serializer, embedder, vector_store, coordinator, hint_parser


# =============================================================================
# 2.  collect_experience()
# =============================================================================

_EYE_K = np.eye(K, dtype=np.float32)   # (K, K) one-hot matrix, built once


def collect_experience(env, shared_agent, obs):
    """
    Execute one environment step using the shared policy.

    Each of the K devices receives a device-specific observation formed by
    concatenating the global obs (170-dim) with a one-hot encoding of its
    device index k (K-dim), yielding a 190-dim input unique to each device.
    This breaks the symmetry that would otherwise cause identical policies.

    Parameters
    ----------
    env          : EdgeCloudEnv
    shared_agent : MAPPOAgent
    obs          : np.ndarray, shape (170,)  — global observation

    Returns
    -------
    obs              : np.ndarray, shape (170,) — global obs (for serializer/reward shaping)
    device_obs_list  : list[np.ndarray(190,)]  — per-device augmented obs, length K
    device_actions   : list[int]               — K sampled actions, one per device
    device_lps       : list[float]             — K log π(a|s) values
    value            : float                   — V(s) from critic (device-0 aug obs)
    reward           : float                   — base environment reward
    next_obs         : np.ndarray, shape (170,)
    terminated       : bool
    truncated        : bool
    info             : dict  {mean_latency, mean_energy, mean_sla_violation, ...}
    """
    device_obs_list = []
    device_actions  = []
    device_lps      = []

    with torch.no_grad():
        # K independent action samples — one per IoT device, each with its own obs
        for k in range(K):
            device_obs   = np.concatenate([obs, _EYE_K[k]])          # (190,)
            obs_tensor_k = torch.tensor(device_obs, dtype=torch.float32)
            a, lp = shared_agent.act(obs_tensor_k)
            device_obs_list.append(device_obs)
            device_actions.append(int(a.item()))
            device_lps.append(float(lp.item()))

        # Critic evaluated on device-0 augmented obs (representative baseline value)
        value = float(
            shared_agent.critic(
                torch.tensor(device_obs_list[0], dtype=torch.float32).to(shared_agent.device)
            ).item()
        )

    # All M edge nodes use the same K-device routing decisions (parameter sharing)
    joint_action = tuple(
        np.array(device_actions, dtype=np.int32) for _ in range(M)
    )

    next_obs, reward, terminated, truncated, info = env.step(joint_action)

    return obs, device_obs_list, device_actions, device_lps, value, float(reward), next_obs, terminated, truncated, info


# =============================================================================
# 3.  compute_returns_and_advantages()
# =============================================================================

def compute_returns_and_advantages(rewards, values, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Compute discounted returns and Generalised Advantage Estimates (GAE).

    Parameters
    ----------
    rewards    : list[float], length T
    values     : list[float], length T  — V(s_t) from shared critic
    last_value : float                  — V(s_{T+1}) bootstrap
    gamma      : float                  — discount factor
    lam        : float                  — GAE lambda

    Returns
    -------
    returns    : list[float], length T
    advantages : list[float], length T
    """
    returns    = [0.0] * len(rewards)
    advantages = [0.0] * len(rewards)

    gae = 0.0
    R   = last_value

    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else last_value
        delta    = rewards[t] + gamma * next_val - values[t]
        gae      = delta + gamma * lam * gae
        R        = rewards[t] + gamma * R
        returns[t]    = R
        advantages[t] = gae

    return returns, advantages


# =============================================================================
# 4.  ppo_update()
# =============================================================================

def ppo_update(shared_agent, buffer):
    """
    CleanRL-style PPO update for the shared_agent.

    Buffer obs shape (T, K, OBS_DIM) is flattened to (T*K, OBS_DIM).
    Returns (length T) are repeated K times to align with per-device
    transitions.  Advantages = returns - critic baseline, normalised.
    Old log-probs come from the collection phase (importance ratio).

    Parameters
    ----------
    shared_agent : MAPPOAgent
    buffer : dict with keys
        obs       : list[list[np.ndarray(OBS_DIM,)]]  — shape (T, K)
        actions   : list[list[int]]                   — shape (T, K)
        log_probs : list[list[float]]                 — shape (T, K)
        returns   : list[float]                       — length T
    """
    obs_b     = torch.tensor(
        np.array(buffer["obs"], dtype=np.float32),
        device=shared_agent.device,
    ).reshape(-1, OBS_DIM)                                          # (T*K, OBS_DIM)

    act_b     = torch.tensor(
        np.array(buffer["actions"]).flatten(),
        dtype=torch.int64, device=shared_agent.device,
    )                                                               # (T*K,)

    old_lp_b  = torch.tensor(
        np.array(buffer["log_probs"]).flatten(),
        dtype=torch.float32, device=shared_agent.device,
    )                                                               # (T*K,)

    ret_b     = torch.tensor(
        np.repeat(np.array(buffer["returns"]), K),
        dtype=torch.float32, device=shared_agent.device,
    )                                                               # (T*K,)

    # Advantage = return − critic baseline (computed once, before update epochs)
    with torch.no_grad():
        val_b = shared_agent.critic(obs_b).squeeze(-1)             # (T*K,)
    adv_b = ret_b - val_b
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        new_val, new_lp_b, entropy = shared_agent.evaluate(obs_b, act_b)
        new_val = new_val.squeeze(-1)                               # (T*K,)

        # Clipped surrogate actor loss (CleanRL formulation)
        ratio    = torch.exp(new_lp_b - old_lp_b)
        pg_loss1 = -adv_b * ratio
        pg_loss2 = -adv_b * torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON)
        pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

        # Critic MSE loss
        v_loss = 0.5 * ((new_val - ret_b) ** 2).mean()

        loss = pg_loss + CRITIC_COEF * v_loss - ENTROPY_COEF * entropy

        shared_agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(shared_agent.actor.parameters()) + list(shared_agent.critic.parameters()),
            GRAD_CLIP,
        )
        shared_agent.optimizer.step()


# =============================================================================
# 5.  train()  — main loop
# =============================================================================

def train(seed: int = 42):
    """
    Full hierarchical training loop.

    Every step              : collect one transition; store in RAG vector store.
    Every N_COORD steps     : query LLM coordinator; shape reward.
    Every BATCH_SIZE steps  : PPO update on shared_agent (T*K transitions).
    Every EVAL_INTERVAL     : print progress metrics.
    """
    N_COORD = 999_999  # LLM disabled for now; set to N_COORDINATION to re-enable

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    (env, shared_agent, serializer, embedder,
     vector_store, coordinator, hint_parser) = setup(seed)

    obs, _ = env.reset(seed=seed)

    # -----------------------------------------------------------------
    # Rollout buffer — cleared after every PPO update
    # -----------------------------------------------------------------
    def _empty_buffer():
        return {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": []}

    buffer = _empty_buffer()

    # -----------------------------------------------------------------
    # Running metrics
    # -----------------------------------------------------------------
    episode_rewards  = []   # per-episode cumulative reward
    ep_reward_accum  = 0.0
    step_latencies   = []   # per-step mean latency
    step_energies    = []   # per-step mean energy
    step_sla         = []   # per-step mean SLA violation
    reward_history   = []   # mean reward snapshot every 1 000 steps

    current_hint: dict | None = None   # most recently computed LLM hint

    # -----------------------------------------------------------------
    # Early stopping / best-model tracking
    # -----------------------------------------------------------------
    PROGRESS_INTERVAL   = 10_000   # steps between progress-bar prints
    EARLY_STOP_WINDOW   = 50_000   # steps to look back for improvement check
    EARLY_STOP_MIN_IMPV = 0.001    # minimum 0.1% improvement required
    CONVERGE_WINDOW     = 5_000    # steps to average for convergence check
    CONVERGE_THRESHOLD  = -0.3     # mean reward above this → converged

    best_reward     = -float("inf")
    best_models_dir = os.path.join(MODELS_DIR, "best")
    os.makedirs(best_models_dir, exist_ok=True)

    def _save_best():
        torch.save(
            {"actor": shared_agent.actor.state_dict(), "critic": shared_agent.critic.state_dict()},
            os.path.join(best_models_dir, "shared_agent.pt"),
        )

    device = shared_agent.device
    print(
        f"Training start | total_steps={TOTAL_STEPS:,} | "
        f"parameter-sharing MARL | K={K} devices | obs_dim={OBS_DIM} | "
        f"n_actions={N_ACTIONS_PER_DEVICE} | device={device}"
    )
    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):

        # -----------------------------------------------------------------
        # Step 1: Collect one transition
        # -----------------------------------------------------------------
        (obs, device_obs_list, device_actions, device_lps, value,
         base_reward, next_obs,
         terminated, truncated, info) = collect_experience(env, shared_agent, obs)

        # -----------------------------------------------------------------
        # Step 2: Serialize + embed current state for RAG
        # -----------------------------------------------------------------
        state_text = serializer.serialize(obs)
        embedding  = embedder.embed(state_text)

        # -----------------------------------------------------------------
        # Step 3: LLM coordination every N_COORD steps
        # -----------------------------------------------------------------
        if step % N_COORD == 0:
            if len(vector_store) > 0:
                results        = vector_store.retrieve(embedding, top_k=RAG_TOP_K)
                context_string = vector_store.build_context_string(results)
            else:
                context_string = "No past situations stored yet."
            current_hint = coordinator.get_hint(state_text, context_string)

        # -----------------------------------------------------------------
        # Step 4: Reward shaping using the latest LLM hint
        # -----------------------------------------------------------------
        if current_hint is not None:
            queue_lengths      = obs[K * M : K * M + M]
            rates_flat         = obs[: K * M].reshape(K, M)
            mean_rate_per_node = rates_flat.mean(axis=0)
            reward = hint_parser.parse_to_shaped_reward(
                base_reward, current_hint, queue_lengths, mean_rate_per_node
            )
        else:
            reward = base_reward

        # -----------------------------------------------------------------
        # Step 5: Store experience in RAG vector store
        # -----------------------------------------------------------------
        vector_store.add(embedding, state_text, reward)

        # -----------------------------------------------------------------
        # Step 6: Append to rollout buffer
        # -----------------------------------------------------------------
        buffer["obs"].append(device_obs_list)      # list of K arrays, each (190,)
        buffer["actions"].append(device_actions)   # list of K ints
        buffer["log_probs"].append(device_lps)     # list of K floats
        buffer["values"].append(value)             # single float
        buffer["rewards"].append(reward)

        # Track metrics
        ep_reward_accum += base_reward
        step_latencies.append(info["mean_latency"])
        step_energies.append(info["mean_energy"])
        step_sla.append(info["mean_sla_violation"])

        # Advance observation; reset on episode end
        if terminated or truncated:
            episode_rewards.append(ep_reward_accum)
            ep_reward_accum = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # -----------------------------------------------------------------
        # Step 7: PPO update every BATCH_SIZE steps
        # -----------------------------------------------------------------
        if len(buffer["rewards"]) >= BATCH_SIZE:
            with torch.no_grad():
                # Bootstrap value: use device-0 one-hot augmented next_obs
                next_obs_aug = np.concatenate([next_obs, _EYE_K[0]])
                next_obs_t   = torch.tensor(next_obs_aug, dtype=torch.float32)
                last_val     = float(
                    shared_agent.critic(next_obs_t.to(shared_agent.device)).item()
                )

            rewards_arr = np.array(buffer["rewards"])
            if rewards_arr.std() > 1e-8:
                buffer["rewards"] = (
                    (rewards_arr - rewards_arr.mean()) / rewards_arr.std()
                ).tolist()

            returns, _ = compute_returns_and_advantages(
                buffer["rewards"], buffer["values"], last_val
            )
            buffer["returns"] = returns

            ppo_update(shared_agent, buffer)

            buffer = _empty_buffer()

        # -----------------------------------------------------------------
        # Step 8: Progress logging
        # -----------------------------------------------------------------
        if step % EVAL_INTERVAL == 0:
            mean_step_reward = (
                float(np.mean(buffer["rewards"][-EVAL_INTERVAL:]))
                if len(buffer["rewards"]) >= EVAL_INTERVAL
                else base_reward
            )
            win = slice(-EVAL_INTERVAL, None)

            print(
                f"step {step:>8,} | "
                f"mean_step_reward {mean_step_reward:+.4f} | "
                f"latency {np.mean(step_latencies[win]):.4f}s | "
                f"energy {np.mean(step_energies[win]):.4e}J | "
                f"SLA viol {np.mean(step_sla[win]):.4f}s | "
                f"RAG {len(vector_store):,} | "
                f"elapsed {time.time() - t0:.0f}s"
            )

        # -----------------------------------------------------------------
        # Step 9: Best-model saving
        # -----------------------------------------------------------------
        if len(step_latencies) >= EVAL_INTERVAL:
            current_mean_reward = float(np.mean(
                episode_rewards[-20:] if episode_rewards else [ep_reward_accum]
            ))
            if current_mean_reward > best_reward:
                best_reward = current_mean_reward
                _save_best()

        # -----------------------------------------------------------------
        # Step 10: reward_history snapshot + progress bar + trend report
        # -----------------------------------------------------------------
        if step % 1_000 == 0:
            snapshot = float(np.mean(
                episode_rewards[-20:] if episode_rewards else [ep_reward_accum]
            ))
            reward_history.append(snapshot)

        if step % PROGRESS_INTERVAL == 0:
            _save_checkpoint(shared_agent, step)

            pct     = 100.0 * step / TOTAL_STEPS
            bar_len = 30
            filled  = int(bar_len * step / TOTAL_STEPS)
            bar     = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.time() - t0
            eta     = (elapsed / step) * (TOTAL_STEPS - step)

            last_1k = reward_history[-1] if reward_history else ep_reward_accum

            # Trend: compare last 1 000-step snapshot to 10 snapshots earlier
            if len(reward_history) >= 11:
                delta = reward_history[-1] - reward_history[-11]
                if delta > 0.05:
                    trend = "IMPROVING"
                elif delta < -0.05:
                    trend = "DEGRADING"
                else:
                    trend = "STABLE"
            else:
                trend = "STABLE"

            print(
                f"[{bar}] {pct:5.1f}%  step {step:,}/{TOTAL_STEPS:,}  "
                f"ETA {eta/60:.1f}min"
            )
            print(
                f"Progress: {pct:.0f}% | "
                f"Best reward: {best_reward:+.4f} | "
                f"Last 1000 mean: {last_1k:+.4f} | "
                f"Trend: {trend}"
            )

        # -----------------------------------------------------------------
        # Step 11: Early stopping — disabled
        # -----------------------------------------------------------------
        # if step >= 100_000 and step >= EARLY_STOP_WINDOW * 2 and step % EARLY_STOP_WINDOW == 0:
        #     old_mean = float(np.mean(step_latencies[-2 * EARLY_STOP_WINDOW : -EARLY_STOP_WINDOW]))
        #     new_mean = float(np.mean(step_latencies[-EARLY_STOP_WINDOW:]))
        #     if old_mean > 0 and (old_mean - new_mean) / old_mean < EARLY_STOP_MIN_IMPV:
        #         print(
        #             f"Early stopping at step {step:,}: latency improvement "
        #             f"({(old_mean-new_mean)/old_mean*100:.2f}%) below "
        #             f"{EARLY_STOP_MIN_IMPV*100:.0f}% threshold."
        #         )
        #         break

        # -----------------------------------------------------------------
        # Step 12: Convergence detection
        # -----------------------------------------------------------------
        if len(step_latencies) >= CONVERGE_WINDOW:
            recent_rewards_conv = (
                episode_rewards[-50:] if len(episode_rewards) >= 50
                else episode_rewards or [ep_reward_accum]
            )
            if float(np.mean(recent_rewards_conv)) > CONVERGE_THRESHOLD:
                print(
                    f"CONVERGED at step {step:,}: mean reward "
                    f"{np.mean(recent_rewards_conv):+.4f} > {CONVERGE_THRESHOLD}"
                )
                break

    # -----------------------------------------------------------------
    # Save final model checkpoint
    # -----------------------------------------------------------------
    _save_checkpoint(shared_agent, TOTAL_STEPS)

    print("Training complete.")
    return shared_agent


# =============================================================================
if __name__ == "__main__":
    train()
