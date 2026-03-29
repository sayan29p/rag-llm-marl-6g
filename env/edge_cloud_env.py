import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    M, K,
    SERVER_CPU_MIN_HZ, SERVER_CPU_MAX_HZ,
    QUEUE_CAPACITY,
    TASK_DATA_MIN_MB, TASK_DATA_MAX_MB,
    CPU_CYCLES_MIN, CPU_CYCLES_MAX,
    DEADLINE_MIN_S, DEADLINE_MAX_S,
    POISSON_LAMBDA,
    TX_POWER_MIN_W, TX_POWER_MAX_W,
    KAPPA,
    W1, W2, W3,
)
from env.channel_model import ChannelModel


# Cloud server is treated as a single logical node with fixed ample resources
CLOUD_CPU_HZ      = 20e9    # 20 GHz — effectively unlimited vs edge
CLOUD_QUEUE_CAP   = 1000    # large enough to never fill

# Action encoding per device:
#   0          → local execution on the device itself
#   1 … M      → offload to edge node m  (1-indexed)
#   M+1        → offload to cloud
N_ACTIONS_PER_DEVICE = M + 2   # local + M edge nodes + cloud


class EdgeCloudEnv(gym.Env):
    """
    Multi-agent Gymnasium environment for 6G edge-cloud task offloading.

    Agents  : M edge nodes, each controlling the offloading decisions for
              all K IoT devices that are associated with it.
    State   : channel rates, server queues, CPU frequencies, task attributes.
    Action  : each agent selects one destination (local/edge/cloud) per device.
    Reward  : negative weighted sum of latency, energy, and SLA violations.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 1000, seed: int = None):
        super().__init__()

        self.max_steps   = max_steps
        self._step_count = 0

        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Channel model (K devices × M edge nodes)
        # ------------------------------------------------------------------
        self.channel = ChannelModel(rng=self.rng)

        # ------------------------------------------------------------------
        # Edge server CPU frequencies — fixed per episode (Hz), shape (M,)
        # ------------------------------------------------------------------
        self.server_cpu = self.rng.uniform(
            SERVER_CPU_MIN_HZ, SERVER_CPU_MAX_HZ, size=M
        )

        # Cloud treated as a single node with fixed frequency
        self.cloud_cpu = CLOUD_CPU_HZ

        # ------------------------------------------------------------------
        # Queue state — number of tasks currently queued, shape (M,)
        # Queue for cloud is tracked separately as a scalar.
        # ------------------------------------------------------------------
        self.queue_lengths  = np.zeros(M,    dtype=np.float32)
        self.cloud_queue    = 0.0

        # Per-device transmit power sampled fresh each episode, shape (K,)
        self.tx_power = self.rng.uniform(TX_POWER_MIN_W, TX_POWER_MAX_W, size=K)

        # Task buffers — populated by reset() / _sample_tasks()
        self.task_data      = np.zeros(K, dtype=np.float32)   # bytes
        self.task_cycles    = np.zeros(K, dtype=np.float32)   # cycles
        self.task_deadlines = np.zeros(K, dtype=np.float32)   # seconds
        self.n_active       = 0                               # tasks this slot

        # ------------------------------------------------------------------
        # Action space
        # Each of the M agents produces a flat vector of K discrete actions.
        # Represented as a Tuple of M MultiDiscrete spaces, each of shape (K,).
        # ------------------------------------------------------------------
        self.action_space = spaces.Tuple(
            tuple(
                spaces.MultiDiscrete([N_ACTIONS_PER_DEVICE] * K)
                for _ in range(M)
            )
        )

        # ------------------------------------------------------------------
        # Observation space  — single flat Box shared across all agents
        # Components:
        #   channel rates  : (K, M)  — bps
        #   queue lengths  : (M,)    — task count
        #   server CPU     : (M,)    — Hz
        #   task data      : (K,)    — bytes
        #   task cycles    : (K,)    — cycles
        #   task deadlines : (K,)    — seconds
        # ------------------------------------------------------------------
        obs_dim = K * M + M + M + K + K + K
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = np.inf,
            shape = (obs_dim,),
            dtype = np.float32,
        )

    # ======================================================================
    # Gymnasium API
    # ======================================================================

    def reset(self, seed: int = None, options: dict = None):
        """
        Reset the environment to a fresh episode.

        Returns
        -------
        obs   : np.ndarray, shape (obs_dim,)
        info  : dict
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel = ChannelModel(rng=self.rng)

        self._step_count = 0

        # Regenerate slow-fading parameters
        self.server_cpu = self.rng.uniform(
            SERVER_CPU_MIN_HZ, SERVER_CPU_MAX_HZ, size=M
        )
        self.tx_power = self.rng.uniform(TX_POWER_MIN_W, TX_POWER_MAX_W, size=K)

        # Reset queues
        self.queue_lengths = np.zeros(M,    dtype=np.float32)
        self.cloud_queue   = 0.0

        # Regenerate channel for first slot
        self.channel.step()

        # Sample new task batch
        self._sample_tasks()

        return self._get_obs(), {}

    def step(self, actions):
        """
        Execute one time slot given actions from all M agents.

        Parameters
        ----------
        actions : tuple of M arrays, each shape (K,)
            actions[i][k] ∈ {0, 1, …, M+1}
              0   → local execution by device k
              1…M → offload to edge node (action - 1)
              M+1 → offload to cloud

        Returns
        -------
        obs        : np.ndarray, shape (obs_dim,)
        reward     : float
        terminated : bool  (always False — no natural terminal state)
        truncated  : bool  (True when max_steps reached)
        info       : dict  {mean_latency, mean_energy, mean_sla_violation,
                            n_active, queue_lengths}
        """
        self._step_count += 1

        # Use the action from agent 0 as the global scheduling decision.
        # (In a true MARL setting each agent controls its own devices;
        #  here we consolidate to a single (K,) decision vector.)
        decision = np.array(actions[0], dtype=np.int32)   # shape (K,)

        # Transmission delay matrix: t_tx[k,m] = data[k] / rate[k,m]  (s)
        data_bits = self.task_data * 8.0                   # bytes → bits
        t_tx_matrix = self.channel.get_transmission_delay(data_bits)  # (K, M)

        # Accumulators over active tasks
        latencies       = np.zeros(K, dtype=np.float64)
        energies        = np.zeros(K, dtype=np.float64)
        sla_violations  = np.zeros(K, dtype=np.float64)

        # Track queue updates (applied after all tasks processed)
        queue_delta = np.zeros(M, dtype=np.float64)

        for k in range(K):
            dest = int(decision[k])

            if dest == 0:
                # --------------------------------------------------------
                # Local execution on IoT device k
                # Assume device has 0.5 GHz CPU (fixed, modest)
                # --------------------------------------------------------
                local_cpu   = 0.5e9                        # Hz
                t_exec      = self.task_cycles[k] / local_cpu   # s
                t_tx        = 0.0                          # no transmission
                t_queue     = 0.0                          # no remote queue
                # Energy: Dinur model  e = kappa * f^2 * C  (Joules)
                e_comp      = KAPPA * local_cpu**2 * self.task_cycles[k]
                e_tx        = 0.0

            elif 1 <= dest <= M:
                # --------------------------------------------------------
                # Offload to edge node  m = dest - 1  (0-indexed)
                # --------------------------------------------------------
                m       = dest - 1
                f_m     = self.server_cpu[m]               # Hz
                t_tx    = t_tx_matrix[k, m]                # uplink delay (s)

                # Execution delay at edge
                t_exec  = self.task_cycles[k] / f_m        # s

                # Queuing delay: existing tasks ahead × mean service time
                # mean_service_time ≈ mean_task_cycles / f_m
                mean_cycles     = (CPU_CYCLES_MIN + CPU_CYCLES_MAX) / 2.0
                t_queue         = self.queue_lengths[m] * (mean_cycles / f_m)

                # Transmission energy
                e_tx    = self.tx_power[k] * t_tx          # W·s = J
                # Computation energy at edge: Dinur model  e = kappa * f^2 * C  (Joules)
                e_comp  = KAPPA * f_m**2 * self.task_cycles[k]

                # Queue update
                queue_delta[m] += 1.0

            else:
                # --------------------------------------------------------
                # Offload to cloud  (dest == M+1)
                # Fixed propagation + processing model
                # --------------------------------------------------------
                # Use best available edge link for backhaul approximation
                best_m  = int(np.argmax(self.channel.rate[k]))
                t_tx    = t_tx_matrix[k, best_m]           # uplink to nearest edge

                # Cloud propagation latency (WAN RTT approximation)
                t_prop  = 0.02                             # 20 ms WAN latency

                f_cloud = self.cloud_cpu
                t_exec  = self.task_cycles[k] / f_cloud

                # Cloud queue: modelled similarly to edge
                mean_cycles = (CPU_CYCLES_MIN + CPU_CYCLES_MAX) / 2.0
                t_queue     = self.cloud_queue * (mean_cycles / f_cloud)

                e_tx    = self.tx_power[k] * t_tx
                # Computation energy at cloud: Dinur model  e = kappa * f^2 * C  (Joules)
                e_comp  = KAPPA * f_cloud**2 * self.task_cycles[k]

                self.cloud_queue = min(self.cloud_queue + 1.0, CLOUD_QUEUE_CAP)

            # Total latency for task k
            total_latency   = t_tx + t_queue + t_exec
            latencies[k]    = total_latency

            # Total energy
            energies[k]     = e_tx + e_comp

            # SLA violation: excess beyond deadline (seconds over budget)
            sla_violations[k] = max(0.0, total_latency - self.task_deadlines[k])

        # ------------------------------------------------------------------
        # Update edge queues (capped at QUEUE_CAPACITY)
        # Queue drains by 1 task per slot (service discipline)
        # ------------------------------------------------------------------
        self.queue_lengths = np.clip(
            self.queue_lengths + queue_delta - 1.0,
            0.0, QUEUE_CAPACITY
        )
        # Cloud queue drains by a fixed number of tasks per slot
        self.cloud_queue = max(0.0, self.cloud_queue - K * 0.1)

        # ------------------------------------------------------------------
        # Reward  r = -(W1*mean_latency + W2*mean_energy*ENERGY_SCALE + W3*mean_sla)
        #
        # The Dinur model (e = κ·f²·C) is calibrated for IoT devices (~MHz).
        # At edge/cloud server frequencies (GHz) it produces 10–400 J per task,
        # which would overwhelm the latency signal (0.1–2 s).  ENERGY_SCALE
        # corrects for this so all three terms contribute comparably to the reward.
        # Target reward range: −1 to −5.
        # ------------------------------------------------------------------
        ENERGY_SCALE = 1e-2   # brings GHz-server Dinur energy (10–400 J) into ~0.1–4 J effective range

        mean_latency  = float(np.mean(latencies))
        mean_energy   = float(np.mean(energies))
        mean_sla      = float(np.mean(sla_violations))

        reward = -(
            W1 * mean_latency +
            W2 * mean_energy * ENERGY_SCALE +
            W3 * mean_sla
        )

        # ------------------------------------------------------------------
        # Advance channel and sample next task batch
        # ------------------------------------------------------------------
        self.channel.step()
        self._sample_tasks()

        obs        = self._get_obs()
        terminated = False
        truncated  = self._step_count >= self.max_steps

        info = {
            "mean_latency"      : mean_latency,
            "mean_energy"       : mean_energy,
            "mean_sla_violation": mean_sla,
            "n_active"          : int(self.n_active),
            "queue_lengths"     : self.queue_lengths.copy(),
        }

        return obs, reward, terminated, truncated, info

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _sample_tasks(self):
        """
        Sample a new batch of IoT tasks for the current time slot.

        Number of active tasks follows Poisson(POISSON_LAMBDA), capped at K.
        Inactive device slots are zeroed out.
        """
        n = min(int(self.rng.poisson(POISSON_LAMBDA)), K)
        self.n_active = n

        # Active devices (first n slots — randomly shuffled each slot)
        active_idx = self.rng.choice(K, size=n, replace=False)

        self.task_data      = np.zeros(K, dtype=np.float32)
        self.task_cycles    = np.zeros(K, dtype=np.float32)
        self.task_deadlines = np.zeros(K, dtype=np.float32)

        if n > 0:
            # Data size: MB → bytes
            self.task_data[active_idx] = self.rng.uniform(
                TASK_DATA_MIN_MB * 1e6,
                TASK_DATA_MAX_MB * 1e6,
                size=n,
            ).astype(np.float32)

            self.task_cycles[active_idx] = self.rng.uniform(
                CPU_CYCLES_MIN, CPU_CYCLES_MAX, size=n
            ).astype(np.float32)

            self.task_deadlines[active_idx] = self.rng.uniform(
                DEADLINE_MIN_S, DEADLINE_MAX_S, size=n
            ).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Flatten all state components into a single float32 observation vector.

        Layout (concatenated):
          [0       : K*M)       channel rates (K,M) flattened row-major  (bps)
          [K*M     : K*M+M)     queue lengths (M,)                       (tasks)
          [K*M+M   : K*M+2M)    server CPU    (M,)                       (Hz)
          [K*M+2M  : K*M+2M+K)  task data     (K,)                       (bytes)
          [K*M+2M+K: K*M+2M+2K) task cycles   (K,)                       (cycles)
          [K*M+2M+2K: end)      task deadlines (K,)                      (s)

        Returns
        -------
        obs : np.ndarray, shape (K*M + 2*M + 3*K,), dtype float32
        """
        obs = np.concatenate([
            self.channel.rate.flatten().astype(np.float32),   # (K*M,)
            self.queue_lengths.astype(np.float32),            # (M,)
            self.server_cpu.astype(np.float32),               # (M,)
            self.task_data,                                   # (K,)
            self.task_cycles,                                 # (K,)
            self.task_deadlines,                              # (K,)
        ])
        return obs
