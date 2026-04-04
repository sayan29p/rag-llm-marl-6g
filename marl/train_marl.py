# =============================================================================
# RAG-Enhanced LLM-Coordinated MARL for 6G Edge-Cloud Task Offloading
# RLlib PPO training — 170-dim observation, Discrete(7) action
#
# Install dependency before running:
#   pip install "ray[rllib]==2.9.0"
# =============================================================================

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import M, K, RESULTS_DIR, MODELS_DIR, LOGS_DIR
from env.edge_cloud_env import EdgeCloudEnv, N_ACTIONS_PER_DEVICE

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
OBS_DIM   = K * M + 2 * M + 3 * K   # 170
N_ACTIONS = N_ACTIONS_PER_DEVICE     # 7  (local + M edge nodes + cloud)


# =============================================================================
# Single-agent wrapper
#
# The underlying EdgeCloudEnv expects actions as a tuple (agent0_actions, ...)
# where agent0_actions is a shape-(K,) array.  We expose a simpler Discrete(7)
# space and broadcast the single chosen action to all K devices.
# =============================================================================

class EdgeCloudRLlibEnv(gym.Env):
    """
    Single-agent RLlib-compatible wrapper around EdgeCloudEnv.

    observation_space : Box(170,)    — full environment state vector
    action_space      : Discrete(7)  — one destination applied to all K devices
                         0         → local execution
                         1 … M     → offload to edge node m
                         M+1       → offload to cloud
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict = None):
        super().__init__()
        env_config = env_config or {}
        self._env = EdgeCloudEnv(seed=env_config.get("seed", None))

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: int):
        # Broadcast single action to all K devices
        device_actions = np.full(K, int(action), dtype=np.int32)
        joint_action   = (device_actions,)
        obs, reward, terminated, truncated, info = self._env.step(joint_action)
        return obs, reward, terminated, truncated, info

    def render(self):
        pass


# =============================================================================
# Training
# =============================================================================

def train(num_iterations: int = 200, seed: int = 42):
    """
    Train a PPO agent using RLlib.

    Parameters
    ----------
    num_iterations : int
        Number of RLlib training iterations.  Each iteration collects
        train_batch_size=2000 environment steps before an SGD update.
    seed : int
        Random seed passed to the environment.
    """
    # Lazy import so the module is importable even without ray installed
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
    except ImportError as exc:
        raise ImportError(
            'ray[rllib] is required. Install with:\n'
            '  pip install "ray[rllib]==2.9.0"'
        ) from exc

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(
            EdgeCloudRLlibEnv,
            env_config={"seed": seed},
        )
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            lr=3e-4,
            num_sgd_iter=4,
            train_batch_size=2000,
            # Standard PPO clip and entropy settings
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
        )
        .rollouts(
            num_rollout_workers=0,   # run rollouts on the driver process
            rollout_fragment_length="auto",
        )
        .resources(num_gpus=0)
        .debugging(seed=seed)
    )

    algo = config.build()

    best_mean_reward = -float("inf")
    best_dir         = os.path.join(MODELS_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)

    print(
        f"RLlib PPO | obs_dim={OBS_DIM} | n_actions={N_ACTIONS} | "
        f"train_batch_size=2000 | num_sgd_iter=4 | lr=3e-4"
    )

    for i in range(1, num_iterations + 1):
        result = algo.train()

        mean_reward  = result["episode_reward_mean"]
        total_steps  = result["timesteps_total"]
        episodes     = result.get("episodes_total", "?")

        print(
            f"iter {i:>4}/{num_iterations} | "
            f"steps {total_steps:>9,} | "
            f"episodes {episodes} | "
            f"mean_reward {mean_reward:+.4f}",
            flush=True,
        )

        # Save best checkpoint
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            ckpt_path = algo.save(best_dir)
            print(f"  → new best ({mean_reward:+.4f})  checkpoint: {ckpt_path}", flush=True)

        # Periodic checkpoint every 20 iterations
        if i % 20 == 0:
            ckpt_path = algo.save(os.path.join(MODELS_DIR, f"ckpt_iter{i:04d}"))
            print(f"  → checkpoint: {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = algo.save(MODELS_DIR)
    print(f"Training complete. Final checkpoint: {final_path}")

    algo.stop()
    ray.shutdown()

    return algo


# =============================================================================
if __name__ == "__main__":
    train()
