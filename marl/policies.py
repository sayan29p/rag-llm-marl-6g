import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import HIDDEN_DIM, LR


class ActorNetwork(nn.Module):
    """
    Stochastic policy network: maps observation → action logits.

    Architecture: obs_dim → HIDDEN_DIM → HIDDEN_DIM → n_actions
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,    HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor, shape (..., obs_dim)

        Returns
        -------
        logits : Tensor, shape (..., n_actions)
        """
        return self.net(obs)


class CriticNetwork(nn.Module):
    """
    Centralised value function: maps observation → scalar value estimate.

    Architecture: obs_dim → HIDDEN_DIM → HIDDEN_DIM → 1
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,    HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor, shape (..., obs_dim)

        Returns
        -------
        value : Tensor, shape (..., 1)
        """
        return self.net(obs)


class MAPPOAgent:
    """
    Single MAPPO agent holding one actor and one critic.

    Each of the M edge-node agents in the system instantiates one
    MAPPOAgent.  The critic receives the global observation so it can
    learn a centralised value function while the actor conditions only
    on the shared obs (CTDE — centralised training, decentralised
    execution).

    Parameters
    ----------
    obs_dim   : int — length of the flat observation vector
    n_actions : int — number of discrete actions per device
                      (N_ACTIONS_PER_DEVICE from env)
    device    : torch.device — cpu or cuda
    """

    def __init__(self, obs_dim: int, n_actions: int, device: torch.device = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actor  = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LR,
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        obs : Tensor, shape (obs_dim,) or (batch, obs_dim)

        Returns
        -------
        action   : Tensor, shape () or (batch,)  — sampled discrete action
        log_prob : Tensor, shape () or (batch,)  — log π(action | obs)
        """
        obs = obs.to(self.device)
        logits = self.actor(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    # ------------------------------------------------------------------
    # Evaluation (used during policy update)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored (obs, action) pairs under the current policy.

        Parameters
        ----------
        obs    : Tensor, shape (batch, obs_dim)
        action : Tensor, shape (batch,)

        Returns
        -------
        value    : Tensor, shape (batch, 1) — V(obs) from critic
        log_prob : Tensor, shape (batch,)   — log π(action | obs)
        entropy  : Tensor, scalar           — mean policy entropy (for regularisation)
        """
        obs    = obs.to(self.device)
        action = action.to(self.device)

        logits   = self.actor(obs)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy().mean()

        value = self.critic(obs)

        return value, log_prob, entropy
