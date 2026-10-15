"""Action heads for picoWAM.

Every head maps the fused latent world state z_t to an action a_t and
exposes the same two methods:

    loss(actions, cond) -> scalar training loss
    sample(cond)        -> (B, action_dim) actions

so the WAM never needs to know which head it is talking to.

    mlp       : regress a_t directly (MSE)
    diffusion : DDPM denoising over actions
    flow      : flow matching from noise to actions
"""

import torch.nn as nn
import torch.nn.functional as F

from .diffusion_head import DiffusionConfig, DiffusionPolicyHead
from .flow_matching_head import FlowMatchingActionHead


class MLPActionHead(nn.Module):
    def __init__(self, action_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def loss(self, actions, cond):
        return F.mse_loss(self.net(cond), actions)

    def sample(self, cond):
        return self.net(cond)


def build_action_head(name, action_dim, cond_dim, diffusion_T=16, flow_steps=8):
    if name == "mlp":
        return MLPActionHead(action_dim, cond_dim, hidden_dim=cond_dim)

    if name == "diffusion":
        cfg = DiffusionConfig(T=diffusion_T, action_dim=action_dim, cond_dim=cond_dim)
        return DiffusionPolicyHead(cfg)

    if name == "flow":
        return FlowMatchingActionHead(
            action_dim=action_dim,
            cond_dim=cond_dim,
            hidden_dim=cond_dim,
            num_inference_steps=flow_steps,
        )

    raise ValueError(f"Unknown action_head: {name}")
