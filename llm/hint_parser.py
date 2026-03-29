import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import M, QUEUE_CAPACITY


class HintParser:
    """
    Converts an LLM coordinator hint dict into a scalar reward bonus that can
    be used for reward shaping during MARL training.
    """

    # Queue thresholds (fraction of QUEUE_CAPACITY)
    _LOW_QUEUE_THRESH  = 0.50   # preferred_node bonus: queue below this
    _HIGH_QUEUE_THRESH = 0.70   # avoid_node bonus: queue above this

    # Per-node bonus increments
    _PREFERRED_NODE_BONUS = 0.1
    _AVOID_NODE_BONUS     = 0.1
    _MAX_PREFERRED_BONUS  = 0.3   # cap on cumulative preferred-node bonus

    def parse(
        self,
        hint: dict,
        queue_lengths: np.ndarray,
        mean_rate_per_node: np.ndarray,
    ) -> float:
        """
        Convert a hint dict into a scalar bonus in [-1.0, +1.0].

        Parameters
        ----------
        hint               : dict returned by LLMCoordinator.get_hint()
        queue_lengths      : np.ndarray shape (M,) — current queue occupancy
        mean_rate_per_node : np.ndarray shape (M,) — mean channel rate per node (bps)

        Returns
        -------
        float in [-1.0, +1.0]
        """
        queues     = np.asarray(queue_lengths,      dtype=np.float64)
        confidence = float(hint.get("confidence", 0.5))
        urgency    = hint.get("urgency", "medium")

        bonus = 0.0

        # ------------------------------------------------------------------
        # 1. Urgency adjustment
        # ------------------------------------------------------------------
        if urgency == "high":
            bonus += 0.2
        elif urgency == "low":
            bonus -= 0.1

        # ------------------------------------------------------------------
        # 2. Preferred-node bonus: reward for recommending low-queue nodes
        # ------------------------------------------------------------------
        preferred_bonus = 0.0
        for node_num in hint.get("preferred_nodes", []):
            m = int(node_num) - 1          # convert 1-indexed → 0-indexed
            if 0 <= m < M:
                if queues[m] < self._LOW_QUEUE_THRESH * QUEUE_CAPACITY:
                    preferred_bonus += self._PREFERRED_NODE_BONUS
        bonus += min(preferred_bonus, self._MAX_PREFERRED_BONUS)

        # ------------------------------------------------------------------
        # 3. Avoid-node bonus: reward for correctly flagging congested nodes
        # ------------------------------------------------------------------
        for node_num in hint.get("avoid_nodes", []):
            m = int(node_num) - 1
            if 0 <= m < M:
                if queues[m] > self._HIGH_QUEUE_THRESH * QUEUE_CAPACITY:
                    bonus += self._AVOID_NODE_BONUS

        # ------------------------------------------------------------------
        # 4. Scale by confidence and clip
        # ------------------------------------------------------------------
        bonus *= confidence
        return float(np.clip(bonus, -1.0, 1.0))

    def parse_to_shaped_reward(
        self,
        base_reward: float,
        hint: dict,
        queue_lengths: np.ndarray,
        mean_rate_per_node: np.ndarray,
        lambda_s: float = 0.3,
    ) -> float:
        """
        Add a scaled LLM hint bonus to the environment base reward.

        shaped_reward = base_reward + lambda_s * bonus

        Parameters
        ----------
        base_reward        : float — raw reward from EdgeCloudEnv.step()
        hint               : dict returned by LLMCoordinator.get_hint()
        queue_lengths      : np.ndarray shape (M,)
        mean_rate_per_node : np.ndarray shape (M,)
        lambda_s           : float — shaping weight (default 0.3)

        Returns
        -------
        float — shaped reward
        """
        bonus = self.parse(hint, queue_lengths, mean_rate_per_node)
        return float(base_reward + lambda_s * bonus)
