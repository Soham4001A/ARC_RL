"""
PPO agent that paints one pixel at a time using the ARCModel + LMA backbone.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Utils import arc_env
from Utils.model import ARCModel          # original model

LATENT_DIM  = 3840
VALUE_HIDDEN_DIM = 512
NUM_COLORS  = 10
GRID_H = GRID_W = 30

# ─── feature extractor (unchanged LMA) ─────────────────────
class ARCFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, LATENT_DIM)
        self.arc = ARCModel((GRID_H, GRID_W), LATENT_DIM, 1024, NUM_COLORS)

    def forward(self, obs: Tensor) -> Tensor:
        return self.arc.feature_extractor(obs)

# ─── custom policy ─────────────────────────────────────────
class ARCPolicy(ActorCriticPolicy):
    """
    1. Use ARCModel to produce logits for every pixel/colour.
    2. Treat pixel index and colour as two-stage decision:
         • pick pixel (Categorical over 900 choices)
         • pick colour (Categorical over 10 colours at that pixel)
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kw):
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch=[], activation_fn=nn.Tanh,
                         features_extractor_class=ARCFeatures, **kw)

        self.pred_head = self.features_extractor.arc.prediction_head
        self.value_net = nn.Sequential(
            nn.Linear(LATENT_DIM, VALUE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_DIM, 1) # Output single value
        )

    # split logits to pixel-heatmap & per-pixel colour logits
    def _logits(self, latent):
        logits = self.pred_head(latent)                 # (B, 10, 30, 30)
        logits = logits.permute(0, 2, 3, 1).contiguous()  # (B, 30, 30, 10)
        logits = logits.view(logits.size(0), -1, 10)      # (B, 900, 10)
        pixel_scores = logits.logsumexp(-1)                # (B, 900)
        return logits, pixel_scores

    def forward(self, obs: Tensor, deterministic=False):
        latent = self.extract_features(obs)
        colour_logits, pixel_scores = self._logits(latent)

        # choose pixel
        pixel_prob  = torch.softmax(pixel_scores, dim=-1)
        pixel_dist  = torch.distributions.Categorical(pixel_prob)
        pixel_idx   = pixel_dist.sample() if not deterministic else pixel_prob.argmax(-1)

        # choose colour at that pixel
        colour_log  = colour_logits[torch.arange(obs.size(0)), pixel_idx]  # (B, 10)
        colour_dist = torch.distributions.Categorical(logits=colour_log)
        colour_act  = colour_dist.sample() if not deterministic else colour_log.argmax(-1)

        row = pixel_idx // GRID_W
        col = pixel_idx % GRID_W
        action = torch.stack([row, col, colour_act], dim=1)

        log_prob = pixel_dist.log_prob(pixel_idx) + colour_dist.log_prob(colour_act)
        value    = self.value_net(latent)
        return action, value, log_prob

    def _predict(self, obs: Tensor, deterministic=False):
        act, _, _ = self.forward(obs, deterministic)
        return act

    def evaluate_actions(self, obs: Tensor, actions: Tensor):
        actions = actions.long()                  
        latent = self.extract_features(obs)
        colour_logits, pixel_scores = self._logits(latent)

        pixel_idx = actions[:, 0] * GRID_W + actions[:, 1]
        colour    = actions[:, 2]

        pixel_prob  = torch.softmax(pixel_scores, dim=-1)
        pixel_dist  = torch.distributions.Categorical(pixel_prob)
        colour_log  = colour_logits[torch.arange(obs.size(0)), pixel_idx]
        colour_dist = torch.distributions.Categorical(logits=colour_log)

        log_prob = pixel_dist.log_prob(pixel_idx) + colour_dist.log_prob(colour)
        entropy  = pixel_dist.entropy() + colour_dist.entropy()
        value    = self.value_net(latent)
        return value, log_prob, entropy

# ─── training setup ────────────────────────────────────────
TOTAL_STEPS = 500_000
N_ENVS      = 4
LR          = 3e-4
N_STEPS     = GRID_H * GRID_W          # 900
BATCH_SIZE  = N_STEPS * N_ENVS

def make_env():
    return arc_env.ARCPuzzleEnv()

venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

model = PPO(
    policy=ARCPolicy,
    env=venv,
    learning_rate=LR,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    clip_range=0.1,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log="./tensorboard/",
)

model.learn(total_timesteps=TOTAL_STEPS)
model.save("ppo_arc_paint")
print("Finished training and saved as ppo_arc_paint")