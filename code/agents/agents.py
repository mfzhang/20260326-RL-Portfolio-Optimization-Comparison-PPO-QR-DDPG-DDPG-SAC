"""
Deep Reinforcement Learning Agents Implementation.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:

    def __init__(self, capacity: int = 1_000_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Network Architectures
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    """Deterministic policy network (shared by DDPG and QR-DDPG)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.hidden(state)))


class Critic(nn.Module):
    """Q-value network for standard DDPG."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.output(self.hidden(x))


class QuantileCritic(nn.Module):
    """
    Distributional critic that outputs N quantile values.
    Used by QR-DDPG for tail-risk-aware policy optimisation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 50,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_quantiles = n_quantiles

        layers: list = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, n_quantiles)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return tensor of shape (batch, n_quantiles)."""
        x = torch.cat([state, action], dim=1)
        return self.output(self.hidden(x))


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient.

    Gradient clipping (max_norm=1.0) is applied to both actor and critic
    updates to prevent exploding gradients.
    """

    _GRAD_CLIP = 1.0  # max gradient norm

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        device: str = "cpu",
        hidden_dims: List[int] = None,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action; add Gaussian exploration noise during training."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        action += np.random.normal(0, noise, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size: int = 128) -> None:
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ---- Critic update ------------------------------------------------ #
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self._GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- Actor update ------------------------------------------------- #
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self._GRAD_CLIP)
        self.actor_optimizer.step()

        # ---- Soft target update ------------------------------------------- #
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _soft_update(self, local: nn.Module, target: nn.Module) -> None:
        for tp, lp in zip(target.parameters(), local.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)


# ---------------------------------------------------------------------------
# QR-DDPG Agent
# ---------------------------------------------------------------------------


class QRDDPGAgent(DDPGAgent):
    """
    Quantile-Regression DDPG.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_quantiles: int = 50,
        buffer_size: int = 1_000_000,
        device: str = "cpu",
        hidden_dims: List[int] = None,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Initialise base DDPG (builds actor, replay buffer, optimizers …)
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            device=device,
            hidden_dims=hidden_dims,
        )

        self.n_quantiles = n_quantiles

        # Replace the standard critic with the quantile variant
        self.critic = QuantileCritic(
            state_dim, action_dim, n_quantiles, hidden_dims
        ).to(device)
        self.critic_target = QuantileCritic(
            state_dim, action_dim, n_quantiles, hidden_dims
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Re-create critic optimiser to point at the new network
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Quantile mid-points τ_i = (i + 0.5) / N
        self.quantile_tau = torch.FloatTensor(
            [(i + 0.5) / n_quantiles for i in range(n_quantiles)]
        ).to(device)

    # ------------------------------------------------------------------ #
    # Override update() for distributional Bellman + CVaR actor loss       #
    # ------------------------------------------------------------------ #

    def update(self, batch_size: int = 128) -> None:
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # ---- Distributional critic update ---------------------------------- #
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_quantiles = self.critic_target(next_states_t, next_actions)
            # shape: (batch, n_quantiles)
            target_quantiles = (
                rewards_t.unsqueeze(1)
                + (1.0 - dones_t.unsqueeze(1)) * self.gamma * target_quantiles
            )

        current_quantiles = self.critic(states_t, actions_t)  # (batch, n_quantiles)
        critic_loss = self._quantile_huber_loss(
            current_quantiles, target_quantiles, self.quantile_tau
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self._GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- CVaR actor update --------------------------------------------- #
        # Maximise the mean of the lowest 5 % quantiles (most pessimistic)
        quantiles = self.critic(states_t, self.actor(states_t))  # (batch, n_quantiles)
        n_cvar = max(1, int(0.05 * self.n_quantiles))
        actor_loss = -quantiles[:, :n_cvar].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self._GRAD_CLIP)
        self.actor_optimizer.step()

        # ---- Soft target update ------------------------------------------- #
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    # ------------------------------------------------------------------ #
    # Quantile Huber loss                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _quantile_huber_loss(
        quantiles: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        quantiles : (batch, N)   – predicted quantile values
        targets   : (batch, N)   – Bellman target quantile values
        tau       : (N,)         – quantile levels

        Returns
        -------
        scalar loss
        """
        # pairwise_delta: (batch, N_pred, N_target)
        pairwise_delta = targets.unsqueeze(1) - quantiles.unsqueeze(2)
        abs_delta = pairwise_delta.abs()
        huber = torch.where(abs_delta > 1.0, abs_delta - 0.5, 0.5 * pairwise_delta**2)
        quantile_loss = (
            tau.view(1, -1, 1) - (pairwise_delta < 0).float()
        ).abs() * huber
        return quantile_loss.mean()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _state_dim, _action_dim = 100, 10

    ddpg = DDPGAgent(_state_dim, _action_dim)
    qr = QRDDPGAgent(_state_dim, _action_dim, n_quantiles=50)

    _s = np.zeros(_state_dim)
    print("DDPG action:", ddpg.select_action(_s).shape)
    print("QR-DDPG action:", qr.select_action(_s).shape)
    print("DRL Agents module loaded successfully")
