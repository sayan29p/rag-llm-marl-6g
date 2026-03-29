import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import K, M, QUEUE_CAPACITY


class StateSerializer:
    """
    Converts the flat observation vector from EdgeCloudEnv into a descriptive
    English string suitable for consumption by an LLM coordinator.

    Observation layout (170 values for K=20, M=5):
      [0        : K*M)        channel rates  (K, M) row-major  (bps)
      [K*M      : K*M+M)      queue lengths  (M,)              (tasks)
      [K*M+M    : K*M+2M)     server CPU     (M,)              (Hz)
      [K*M+2M   : K*M+2M+K)   task data      (K,)              (bytes)
      [K*M+2M+K : K*M+2M+2K)  task cycles    (K,)              (cycles)
      [K*M+2M+2K: end)        task deadlines (K,)              (seconds)
    """

    # Congestion threshold: fraction of QUEUE_CAPACITY above which a node is
    # considered congested.
    CONGESTION_THRESHOLD = 0.6

    def serialize(self, obs: np.ndarray) -> str:
        """
        Convert a flat observation array into a human-readable English summary.

        Parameters
        ----------
        obs : np.ndarray, shape (K*M + 2*M + 3*K,)

        Returns
        -------
        str  — multi-sentence description of the current network state.
        """
        obs = np.asarray(obs, dtype=np.float64)

        # ------------------------------------------------------------------
        # Slice out components
        # ------------------------------------------------------------------
        rates_flat  = obs[0          : K * M]
        queues      = obs[K * M      : K * M + M]
        server_cpu  = obs[K * M + M  : K * M + 2 * M]
        task_data   = obs[K * M + 2 * M         : K * M + 2 * M + K]
        task_cycles = obs[K * M + 2 * M + K     : K * M + 2 * M + 2 * K]
        task_dl     = obs[K * M + 2 * M + 2 * K : K * M + 2 * M + 3 * K]

        rates = rates_flat.reshape(K, M)   # (K, M)

        # ------------------------------------------------------------------
        # 1. Queue / congestion summary
        # ------------------------------------------------------------------
        congestion_thresh = self.CONGESTION_THRESHOLD * QUEUE_CAPACITY
        congested   = [m for m in range(M) if queues[m] >= congestion_thresh]
        moderate    = [m for m in range(M) if
                       0 < queues[m] < congestion_thresh]
        idle        = [m for m in range(M) if queues[m] == 0]

        queue_parts = []
        for m in range(M):
            pct = 100.0 * queues[m] / QUEUE_CAPACITY
            queue_parts.append(
                f"Node {m + 1}: {int(queues[m])} tasks ({pct:.0f}% full, "
                f"CPU {server_cpu[m] / 1e9:.1f} GHz)"
            )
        queue_str = "; ".join(queue_parts)

        if congested:
            cong_str = (
                f"Nodes {[m + 1 for m in congested]} are CONGESTED "
                f"(≥{int(self.CONGESTION_THRESHOLD * 100)}% queue capacity)."
            )
        else:
            cong_str = "No edge nodes are currently congested."

        if idle:
            idle_str = f"Nodes {[m + 1 for m in idle]} are idle (empty queue)."
        else:
            idle_str = "All edge nodes have tasks queued."

        # ------------------------------------------------------------------
        # 2. Channel quality summary
        # ------------------------------------------------------------------
        # Per-device: best-node rate and its identity
        best_node_per_dev = np.argmax(rates, axis=1)          # (K,)
        best_rate_per_dev = rates[np.arange(K), best_node_per_dev]  # (K,)

        # Per-node: mean rate across all devices
        mean_rate_per_node = rates.mean(axis=0)               # (M,)

        # Network-wide statistics
        overall_mean = rates.mean()
        overall_min  = rates.min()
        overall_max  = rates.max()

        # Classify nodes by mean channel quality
        q75 = np.percentile(mean_rate_per_node, 75)
        q25 = np.percentile(mean_rate_per_node, 25)
        good_ch  = [m for m in range(M) if mean_rate_per_node[m] >= q75]
        poor_ch  = [m for m in range(M) if mean_rate_per_node[m] <= q25]

        channel_str = (
            f"Channel rates range from {overall_min / 1e6:.1f} Mbps to "
            f"{overall_max / 1e6:.1f} Mbps (network mean: "
            f"{overall_mean / 1e6:.1f} Mbps). "
            f"Best-channel nodes (highest mean rate): "
            f"{[m + 1 for m in good_ch]}. "
            f"Weakest-channel nodes: {[m + 1 for m in poor_ch]}."
        )

        # ------------------------------------------------------------------
        # 3. Pending task summary
        # ------------------------------------------------------------------
        active_mask = task_data > 0
        n_active    = int(active_mask.sum())

        if n_active == 0:
            task_str = "There are no active tasks this time slot."
        else:
            data_mb   = task_data[active_mask] / 1e6        # bytes → MB
            cycles_mc = task_cycles[active_mask] / 1e6      # cycles → Mcycles
            dls       = task_dl[active_mask]                 # seconds

            urgent  = int((dls < 1.0).sum())
            normal  = int(((dls >= 1.0) & (dls < 2.0)).sum())
            relaxed = int((dls >= 2.0).sum())

            # Identify devices with urgency (tight deadline)
            urgent_ids = [
                k + 1 for k in range(K)
                if task_data[k] > 0 and task_dl[k] < 1.0
            ]

            task_str = (
                f"There are {n_active} active tasks this slot. "
                f"Data sizes: {data_mb.min():.2f}–{data_mb.max():.2f} MB "
                f"(mean {data_mb.mean():.2f} MB). "
                f"Compute requirements: {cycles_mc.min():.0f}–"
                f"{cycles_mc.max():.0f} Mcycles "
                f"(mean {cycles_mc.mean():.0f} Mcycles). "
                f"Deadline breakdown: {urgent} urgent (<1 s), "
                f"{normal} normal (1–2 s), {relaxed} relaxed (≥2 s)."
            )
            if urgent_ids:
                task_str += (
                    f" Urgent task devices (deadline <1 s): "
                    f"{urgent_ids}."
                )

            # Best offload candidate per urgent task
            if urgent_ids and n_active > 0:
                # Suggest the least-loaded, best-channel node for urgent tasks
                # Score: lower queue is better, higher mean rate is better
                norm_q = queues / (QUEUE_CAPACITY + 1e-9)
                norm_r = mean_rate_per_node / (mean_rate_per_node.max() + 1e-9)
                score  = norm_r - norm_q          # higher is better
                best_m = int(np.argmax(score))
                task_str += (
                    f" Recommended node for urgent tasks: Node {best_m + 1} "
                    f"(queue {int(queues[best_m])}/{QUEUE_CAPACITY}, "
                    f"mean rate {mean_rate_per_node[best_m] / 1e6:.1f} Mbps)."
                )

        # ------------------------------------------------------------------
        # 4. Assemble final string
        # ------------------------------------------------------------------
        summary = (
            f"=== Edge-Cloud Network State Summary ===\n"
            f"[Queue Status] {queue_str}. "
            f"{cong_str} {idle_str}\n"
            f"[Channel Quality] {channel_str}\n"
            f"[Pending Tasks] {task_str}"
        )
        return summary
