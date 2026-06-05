"""World Action Model Policy.

mini-VLA:
    image_t + text + state_t -> action_t

mini-WAM:
    image_t + text + state_t -> action_t + state_t+1

Supported action heads:
    mlp
    diffusion
    flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead
from .dynamics_head import DynamicsHeadMLP
from .flow_matching_head import FlowMatchingActionHead


class MiniWAMPolicy(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_dim,
        action_dim,
        d_model=128,
        diffusion_T=16,
        action_head="mlp",
        predict_state=True,
        state_loss_weight=0.5,
        flow_steps=8,
    ):
        super().__init__()

        self.action_head = action_head
        self.predict_state = predict_state
        self.state_loss_weight = state_loss_weight

        self.img_encoder = ImageEncoderTinyCNN(d_model=d_model)

        self.txt_encoder = TextEncoderTinyGRU(
            vocab_size=vocab_size,
            d_word=64,
            d_model=d_model,
        )

        self.state_encoder = StateEncoderMLP(
            state_dim=state_dim,
            d_model=d_model,
        )

        self.fusion = FusionMLP(d_model=d_model)

        if action_head == "mlp":
            self.policy_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, action_dim),
            )

        elif action_head == "diffusion":
            cfg = DiffusionConfig(
                T=diffusion_T,
                action_dim=action_dim,
                cond_dim=d_model,
            )
            self.policy_head = DiffusionPolicyHead(cfg)

        elif action_head == "flow":
            self.policy_head = FlowMatchingActionHead(
                action_dim=action_dim,
                cond_dim=d_model,
                hidden_dim=d_model,
                num_inference_steps=flow_steps,
            )

        else:
            raise ValueError(f"Unknown action_head: {action_head}")

        if predict_state:
            self.dynamics_head = DynamicsHeadMLP(
                d_model=d_model,
                state_dim=state_dim,
            )

    def encode_obs(self, image, text_tokens, state):
        img_token = self.img_encoder(image)
        txt_token = self.txt_encoder(text_tokens)
        state_token = self.state_encoder(state)

        cond = self.fusion(
            img_token,
            txt_token,
            state_token,
        )

        return cond

    def forward(
        self,
        image,
        text_tokens,
        state,
        actions=None,
    ):
        cond = self.encode_obs(
            image=image,
            text_tokens=text_tokens,
            state=state,
        )

        outputs = {}

        if self.action_head == "mlp":
            outputs["action"] = self.policy_head(cond)

        elif self.action_head == "diffusion":
            if actions is not None:
                outputs["action_loss"] = self.policy_head.loss(
                    actions,
                    cond,
                )

            outputs["action"] = self.policy_head.sample(cond)

        elif self.action_head == "flow":
            if actions is not None:
                outputs["action_loss"] = self.policy_head.loss(
                    actions,
                    cond,
                )

            outputs["action"] = self.policy_head.sample(cond)

        else:
            raise ValueError(f"Unknown action_head: {self.action_head}")

        if self.predict_state:
            outputs["next_state"] = self.dynamics_head(cond)

        return outputs

    def loss(
        self,
        image,
        text_tokens,
        state,
        actions,
        next_state=None,
    ):
        outputs = self.forward(
            image=image,
            text_tokens=text_tokens,
            state=state,
            actions=actions,
        )

        losses = {}

        if self.action_head == "mlp":
            losses["action"] = F.mse_loss(
                outputs["action"],
                actions,
            )

        elif self.action_head in ["diffusion", "flow"]:
            losses["action"] = outputs["action_loss"]

        else:
            raise ValueError(f"Unknown action_head: {self.action_head}")

        total_loss = losses["action"]

        if self.predict_state and next_state is not None:
            losses["state"] = F.mse_loss(
                outputs["next_state"],
                next_state,
            )

            total_loss = total_loss + self.state_loss_weight * losses["state"]

        return total_loss, losses

    @torch.no_grad()
    def act(self, image, text_tokens, state):
        cond = self.encode_obs(
            image=image,
            text_tokens=text_tokens,
            state=state,
        )

        if self.action_head == "mlp":
            return self.policy_head(cond)

        if self.action_head in ["diffusion", "flow"]:
            return self.policy_head.sample(cond)

        raise ValueError(f"Unknown action_head: {self.action_head}")

    @torch.no_grad()
    def imagine_next_state(self, image, text_tokens, state):
        if not self.predict_state:
            raise RuntimeError("predict_state=False, so no dynamics head exists.")

        cond = self.encode_obs(
            image=image,
            text_tokens=text_tokens,
            state=state,
        )

        return self.dynamics_head(cond)