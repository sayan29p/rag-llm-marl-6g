import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import HIDDEN_DIM


class ActorNetwork(nn.Module):
    """
    Stochastic policy network: maps observation → action logits.
    Architecture: obs_dim → HIDDEN_DIM → HIDDEN_DIM → n_actions
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,    HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, n_actions),
        )
        # Orthogonal initialization for stability
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        # Smaller gain for output layer — keeps initial probs near uniform
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNetwork(nn.Module):
    """
    Value function: maps observation → scalar value estimate.
    Architecture: obs_dim → HIDDEN_DIM → HIDDEN_DIM → 1
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,    HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        # Orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MAPPOAgent:
    """
    Shared MAPPO agent.

    One instance serves ALL K devices via parameter sharing.
    Actor uses device-specific 18-dim observations.
    Critic uses the same 18-dim observation for value estimation.

    Parameters
    ----------
    obs_dim   : int — 18 (device-specific) or 170 (global)
    n_actions : int — 7 (local + 5 edge + cloud)
    device    : torch.device
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 device: torch.device = None,
                 lr: float = 1e-4):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actor  = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5,
        )

    def act(self, obs: torch.Tensor) -> tuple:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        obs : Tensor shape (obs_dim,)

        Returns
        -------
        action   : Tensor scalar
        log_prob : Tensor scalar
        """
        obs    = obs.to(self.device)
        logits = self.actor(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor,
                 action: torch.Tensor) -> tuple:
        """
        Re-evaluate stored (obs, action) pairs under current policy.

        Parameters
        ----------
        obs    : Tensor shape (batch, obs_dim)
        action : Tensor shape (batch,)

        Returns
        -------
        value    : Tensor shape (batch, 1)
        log_prob : Tensor shape (batch,)
        entropy  : Tensor scalar
        """
        obs    = obs.to(self.device)
        action = action.to(self.device)

        logits   = self.actor(obs)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy().mean()
        value    = self.critic(obs)

        return value, log_prob, entropy