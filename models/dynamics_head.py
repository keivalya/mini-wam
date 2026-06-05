"""Dynamics prediction head for mini-WAM."""

import torch.nn as nn


class DynamicsHeadMLP(nn.Module):
    """
    Predicts the next environment state from the fused observation context.

    WAM idea:
        current observation/context -> predicted next state
    """

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, cond):
        return self.net(cond)