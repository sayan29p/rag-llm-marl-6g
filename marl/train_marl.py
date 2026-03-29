# =============================================================================
# RAG-Enhanced LLM-Coordinated MARL for 6G Edge-Cloud Task Offloading
# Hierarchical training loop — wires together all system components
# =============================================================================

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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

# -----------------------------------------------------------------------------
# PPO hyper-parameters (not in config — training-loop internals)
# -----------------------------------------------------------------------------
PPO_EPSILON   = 0.2     # clipping range for importance-sampling ratio
ENTROPY_COEF  = 0.01    # entropy regularisation coefficient
CRITIC_COEF   = 0.5     # critic loss coefficient
PPO_EPOCHS    = 4       # gradient update passes per collected batch
GAE_LAMBDA    = 0.95    # GAE λ for advantage estimation
GRAD_CLIP     = 0.5     # global gradient-norm clip
EVAL_INTERVAL = 1_000   # steps between progress log lines

# Observation dimension (K=20, M=5):  K*M + 2M + 3K = 100 + 10 + 60 = 170
OBS_DIM = K * M + 2 * M + 3 * K


# =============================================================================
# 1.  setup()
# =============================================================================

def setup(seed: int = 42):
    """
    Instantiate and return all system components.

    Returns
    -------
    env          : EdgeCloudEnv
    agents       : list[MAPPOAgent]  — length M
    serializer   : StateSerializer
    embedder     : StateEmbedder
    vector_store : VectorStore
    coordinator  : LLMCoordinator
    hint_parser  : HintParser
    """
    env = EdgeCloudEnv(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = [
        MAPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_PER_DEVICE, device=device)
        for _ in range(M)
    ]

    serializer   = StateSerializer()
    embedder     = StateEmbedder()
    vector_store = VectorStore(embedding_dim=EMBEDDING_DIM)
    coordinator  = LLMCoordinator()
    hint_parser  = HintParser()

    return env, agents, serializer, embedder, vector_store, coordinator, hint_parser


# =============================================================================
# 2.  collect_experience()
# =============================================================================

def collect_experience(env, agents, obs):
    """
    Execute one environment step using all M agents.

    Each agent sees the shared global observation and independently selects
    one discrete action (which destination to route tasks to).  That action
    is replicated across all K devices for the env step.

    Parameters
    ----------
    env    : EdgeCloudEnv
    agents : list[MAPPOAgent], length M
    obs    : np.ndarray, shape (OBS_DIM,)

    Returns
    -------
    obs        : np.ndarray  — observation used this step  (same as input)
    actions    : list[int]   — mean action index per agent (length M), for PPO storage
    log_probs  : list[float] — mean log π(a|s) across K samples per agent (length M)
    values     : list[float] — V(s) per agent              (length M)
    reward     : float       — base environment reward
    next_obs   : np.ndarray
    terminated : bool
    truncated  : bool
    info       : dict  {mean_latency, mean_energy, mean_sla_violation, ...}
    """
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    actions   = []
    log_probs = []
    values    = []

    with torch.no_grad():
        for agent in agents:
            # Sample K independent actions — one routing decision per device
            k_actions   = []
            k_log_probs = []
            for _ in range(K):
                a, lp = agent.act(obs_tensor)
                k_actions.append(int(a.item()))
                k_log_probs.append(float(lp.item()))
            value = agent.critic(obs_tensor.to(agent.device))

            # Store mean action/log_prob for PPO update (scalar per agent)
            actions.append(int(round(float(np.mean(k_actions)))))
            log_probs.append(float(np.mean(k_log_probs)))
            values.append(float(value.item()))

    # Each agent sends K independent per-device routing decisions to the env
    joint_action = tuple(
        np.array([agents[i].act(obs_tensor)[0].item() for _ in range(K)], dtype=np.int32)
        for i in range(M)
    )

    next_obs, reward, terminated, truncated, info = env.step(joint_action)

    return obs, actions, log_probs, values, float(reward), next_obs, terminated, truncated, info


# =============================================================================
# 3.  compute_returns_and_advantages()
# =============================================================================

def compute_returns_and_advantages(rewards, values, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Compute discounted returns and Generalised Advantage Estimates (GAE).

    Parameters
    ----------
    rewards    : list[float], length T
    values     : list[float], length T  — V(s_t) estimates (mean over agents)
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

def ppo_update(agents, buffer):
    """
    Run PPO parameter updates for all M agents on the collected buffer.

    Uses clipped surrogate objective + MSE critic loss + entropy bonus.

    Parameters
    ----------
    agents : list[MAPPOAgent]
    buffer : dict with keys
        obs        : list[np.ndarray(OBS_DIM,)]
        actions    : list[list[int]]              — shape (T, M)
        log_probs  : list[list[float]]            — shape (T, M)
        returns    : list[float]                  — shape (T,)
        advantages : list[float]                  — shape (T,)
    """
    obs_arr  = np.array(buffer["obs"],       dtype=np.float32)  # (T, OBS_DIM)
    ret_arr  = np.array(buffer["returns"],   dtype=np.float32)  # (T,)
    adv_arr  = np.array(buffer["advantages"],dtype=np.float32)  # (T,)

    # Normalise advantages over the whole batch
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

    for agent_idx, agent in enumerate(agents):
        acts_arr   = np.array(
            [a[agent_idx]  for a in buffer["actions"]],   dtype=np.int64
        )                                                          # (T,)
        old_lp_arr = np.array(
            [lp[agent_idx] for lp in buffer["log_probs"]], dtype=np.float32
        )                                                          # (T,)

        obs_t    = torch.tensor(obs_arr,    device=agent.device)
        acts_t   = torch.tensor(acts_arr,   device=agent.device)
        old_lp_t = torch.tensor(old_lp_arr, device=agent.device)
        ret_t    = torch.tensor(ret_arr,    device=agent.device)
        adv_t    = torch.tensor(adv_arr,    device=agent.device)

        for _ in range(PPO_EPOCHS):
            values_t, new_lp_t, entropy = agent.evaluate(obs_t, acts_t)
            values_t = values_t.squeeze(-1)                        # (T,)

            # Importance-sampling ratio  π_new / π_old
            ratio      = torch.exp(new_lp_t - old_lp_t)
            clip_ratio = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON)

            # Actor loss: clipped surrogate objective (maximised → negated)
            actor_loss  = -torch.min(ratio * adv_t, clip_ratio * adv_t).mean()

            # Critic loss: mean-squared TD error
            critic_loss = F.mse_loss(values_t, ret_t)

            # Combined loss with entropy bonus to encourage exploration
            loss = actor_loss + CRITIC_COEF * critic_loss - ENTROPY_COEF * entropy

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.actor.parameters()) + list(agent.critic.parameters()),
                max_norm=GRAD_CLIP,
            )
            agent.optimizer.step()


# =============================================================================
# 5.  train()  — main loop
# =============================================================================

def train(seed: int = 42):
    """
    Full hierarchical training loop.

    Every step       : collect one transition; store in RAG vector store.
    Every N_COORDINATION steps : query LLM coordinator; shape reward.
    Every BATCH_SIZE steps     : PPO update for all agents.
    Every EVAL_INTERVAL steps  : print progress metrics.
    """
    N_COORD = 999_999  # fires every 10 steps as configured in config.py

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    (env, agents, serializer, embedder,
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
        for i, agent in enumerate(agents):
            torch.save(
                {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
                os.path.join(best_models_dir, f"agent_{i}.pt"),
            )

    device = agents[0].device
    print(
        f"Training start | total_steps={TOTAL_STEPS:,} | "
        f"M={M} agents | obs_dim={OBS_DIM} | n_actions={N_ACTIONS_PER_DEVICE} | "
        f"device={device}"
    )
    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):

        # -----------------------------------------------------------------
        # Step 1: Collect one transition
        # -----------------------------------------------------------------
        (obs, actions, log_probs, values,
         base_reward, next_obs,
         terminated, truncated, info) = collect_experience(env, agents, obs)

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
        buffer["obs"].append(obs.copy())
        buffer["actions"].append(actions)
        buffer["log_probs"].append(log_probs)
        buffer["values"].append(float(np.mean(values)))
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
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
                last_val   = float(np.mean([
                    agent.critic(next_obs_t.to(agent.device)).item()
                    for agent in agents
                ]))

            returns, advantages = compute_returns_and_advantages(
                buffer["rewards"], buffer["values"], last_val
            )
            buffer["returns"]    = returns
            buffer["advantages"] = advantages

            ppo_update(agents, buffer)

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
            pct     = 100.0 * step / TOTAL_STEPS
            bar_len = 30
            filled  = int(bar_len * step / TOTAL_STEPS)
            bar     = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.time() - t0
            eta     = (elapsed / step) * (TOTAL_STEPS - step)

            last_1k = reward_history[-1] if reward_history else ep_reward_accum

            # Trend: compare the most-recent 1 000-step snapshot to the one
            # 10 snapshots earlier (= 10 000 steps ago) when available.
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
        #     # Latency decreasing means improvement; check relative change
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
    # Save final model checkpoints
    # -----------------------------------------------------------------
    for i, agent in enumerate(agents):
        path = os.path.join(MODELS_DIR, f"agent_{i}.pt")
        torch.save(
            {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
            path,
        )
        print(f"Saved agent {i} → {path}")

    print("Training complete.")
    return agents


# =============================================================================
if __name__ == "__main__":
    train()
