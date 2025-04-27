# --- In train_ppo.py ---

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Dict, Any
from gymnasium import spaces # Import spaces for type hint

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the modified environment
from Utils import arc_env
# --- Import LMA ---
from Utils.classes import LMAFeaturesExtractor, LMAConfigRL # Make sure LMAConfigRL is also imported if needed separately

# --- Constants ---
H, W = arc_env.H, arc_env.W
N_COLORS = arc_env.N_COLORS
TOTAL_CHANNELS = arc_env.TOTAL_CHANNELS
GRID_H, GRID_W = H, W

# --- LMA Configuration ---
# IMPORTANT: These parameters need careful tuning!
LMA_SEQ_LEN = H * W  # Treat each pixel as a step in the sequence (900)
# Features per step = TOTAL_CHANNELS (implicitly calculated by LMAFeaturesExtractor)
LMA_EMBED_DIM = 128       # d0: Dimension after initial projection of channels
LMA_NUM_HEADS_STACKING = 128 # Must divide LMA_EMBED_DIM
LMA_TARGET_L_NEW = LMA_SEQ_LEN  # Target latent sequence length (adjust based on total features)
LMA_D_NEW = 128           # Latent embedding dimension
LMA_NUM_HEADS_LATENT = 32  # Must divide LMA_D_NEW
LMA_FF_LATENT_HIDDEN = LMA_D_NEW * 4 # Standard MLP hidden dim
LMA_NUM_LAYERS = 4        # Number of LMA blocks
LMA_DROPOUT = 0.1
LMA_BIAS = True

# --- Calculate LMA Output Dimension ---
# We need to instantiate a temporary config to find the actual L_new and calculate features_dim
try:
    _temp_lma_config = LMAConfigRL(
        seq_len=LMA_SEQ_LEN, embed_dim=LMA_EMBED_DIM, num_heads_stacking=LMA_NUM_HEADS_STACKING,
        target_l_new=LMA_TARGET_L_NEW, d_new=LMA_D_NEW, num_heads_latent=LMA_NUM_HEADS_LATENT
    )
    LMA_OUTPUT_FEATURES_DIM = _temp_lma_config.L_new * _temp_lma_config.d_new
    print(f"Calculated LMA Output Feature Dim (L_new * d_new): {LMA_OUTPUT_FEATURES_DIM}")
except ValueError as e:
    print(f"Error calculating LMA config: {e}")
    print("Please adjust LMA parameters.")
    exit()
# --- End LMA Config ---


# --- Head Dimensions (using LMA output) ---
VALUE_HIDDEN_DIM = 256 # Hidden dim for the value network head
POLICY_HIDDEN_DIM = 512 # Hidden dim for the policy network head


# --- ADDED STARTUP PRINT ---
print("--- PPO Training Script with LMA ---")
print(f"Grid Dimensions (H, W): ({H}, {W})")
print(f"Observation Channels: {TOTAL_CHANNELS}")
print(f"LMA Feature Dim: {LMA_OUTPUT_FEATURES_DIM}")
print(f"Value Head Hidden Dim: {VALUE_HIDDEN_DIM}")
print(f"Policy Head Hidden Dim: {POLICY_HIDDEN_DIM}")
print("-" * 30)


# --- Feature Extractor is now LMA (imported from Utils.classes) ---
# No need to redefine CNNARCFeatures


# ─── Custom Policy using LMA Features ───────────────────────
class ARCPolicy(ActorCriticPolicy):
    """
    Uses the LMAFeaturesExtractor.
    Defines its OWN prediction and value heads operating on LMA features.
    Still uses the two-stage pixel/color decision process.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        # Add other PPO args if needed
        **kwargs
    ):
        # --- Configure to use LMAFeaturesExtractor ---
        lma_kwargs = dict(
            embed_dim=LMA_EMBED_DIM,
            num_heads_stacking=LMA_NUM_HEADS_STACKING,
            target_l_new=LMA_TARGET_L_NEW, # Target L_new
            d_new=LMA_D_NEW,
            num_heads_latent=LMA_NUM_HEADS_LATENT,
            ff_latent_hidden=LMA_FF_LATENT_HIDDEN,
            num_lma_layers=LMA_NUM_LAYERS,
            seq_len=LMA_SEQ_LEN, # Critical: H*W
            dropout=LMA_DROPOUT,
            bias=LMA_BIAS
        )
        kwargs['features_extractor_class'] = LMAFeaturesExtractor
        kwargs['features_extractor_kwargs'] = lma_kwargs
        kwargs['net_arch'] = [] # No shared layers between policy/value after LMA extractor
        kwargs['activation_fn'] = nn.ReLU # Use ReLU for heads

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
        # --- Sanity check feature dimension ---
        # LMA calculates its feature dim internally based on L_new * d_new
        calculated_feature_dim = self.features_extractor.features_dim
        if calculated_feature_dim != LMA_OUTPUT_FEATURES_DIM:
             print(f"WARNING: LMA feature dim mismatch! Expected {LMA_OUTPUT_FEATURES_DIM}, Got {calculated_feature_dim} from extractor.")
             # Potentially update LMA_OUTPUT_FEATURES_DIM if the internal calculation is trusted
             # LMA_OUTPUT_FEATURES_DIM = calculated_feature_dim


        # --- Define Heads based on LMA_OUTPUT_FEATURES_DIM ---
        # Policy head: Takes LMA features
        self.policy_ffn = nn.Sequential(
            nn.Linear(LMA_OUTPUT_FEATURES_DIM, POLICY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_DIM, GRID_H * GRID_W * N_COLORS)
        )

        # Value head: Takes LMA features
        self.value_net = nn.Sequential(
            nn.Linear(LMA_OUTPUT_FEATURES_DIM, VALUE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_DIM, 1) # Output single value
        )
        print(f"\nARCPolicy initialized with LMAFeaturesExtractor.")
        print(f"  Policy head input dim: {LMA_OUTPUT_FEATURES_DIM}, hidden: {POLICY_HIDDEN_DIM}")
        print(f"  Value head input dim: {LMA_OUTPUT_FEATURES_DIM}, hidden: {VALUE_HIDDEN_DIM}")
        print("-" * 30)

    # --- Methods _get_policy_logits, _logits_from_features, forward, ---
    # --- _predict, evaluate_actions remain the same conceptually, ---
    # --- as they operate on the 'features' output by the extractor ---
    # --- and use self.policy_ffn / self.value_net which are now correctly sized ---

    def _get_policy_logits(self, features: Tensor) -> Tensor:
        logits_flat = self.policy_ffn(features)
        logits = logits_flat.view(-1, N_COLORS, GRID_H, GRID_W)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        return logits

    def _logits_from_features(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        logits_bhwc = self._get_policy_logits(features)
        logits = logits_bhwc.view(features.size(0), -1, N_COLORS)
        pixel_scores = logits.logsumexp(dim=-1)
        return logits, pixel_scores

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        # LMAFeaturesExtractor handles the reshape from flattened obs to (B, L, F) internally
        features = self.extract_features(obs) # Shape (B, LMA_OUTPUT_FEATURES_DIM)
        colour_logits_b900c10, pixel_scores_b900 = self._logits_from_features(features)
        values = self.value_net(features)
        pixel_prob = torch.softmax(pixel_scores_b900, dim=-1)
        pixel_dist = torch.distributions.Categorical(probs=pixel_prob)
        pixel_indices = pixel_dist.sample() if not deterministic else pixel_prob.argmax(dim=-1)
        batch_indices = torch.arange(features.size(0), device=pixel_indices.device)
        selected_colour_logits = colour_logits_b900c10[batch_indices, pixel_indices]
        colour_dist = torch.distributions.Categorical(logits=selected_colour_logits)
        colour_actions = colour_dist.sample() if not deterministic else selected_colour_logits.argmax(dim=-1)
        rows = pixel_indices // GRID_W
        cols = pixel_indices % GRID_W
        actions = torch.stack([rows, cols, colour_actions], dim=1)
        log_prob_pixels = pixel_dist.log_prob(pixel_indices)
        log_prob_colours = colour_dist.log_prob(colour_actions)
        log_probs = log_prob_pixels + log_prob_colours
        # Ensure value has shape (B, 1) for SB3 calculations
        return actions, values.flatten(), log_probs # Use flatten() as SB3 expects (B,) or (B,1) for value

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.extract_features(obs) # Shape (B, LMA_OUTPUT_FEATURES_DIM)
        colour_logits_b900c10, pixel_scores_b900 = self._logits_from_features(features)
        values = self.value_net(features) # Shape (B, 1)
        actions = actions.long()
        rows, cols, colour_actions = actions[:, 0], actions[:, 1], actions[:, 2]
        pixel_indices = rows * GRID_W + cols
        pixel_prob = torch.softmax(pixel_scores_b900, dim=-1)
        pixel_dist = torch.distributions.Categorical(probs=pixel_prob)
        batch_indices = torch.arange(features.size(0), device=pixel_indices.device)
        selected_colour_logits = colour_logits_b900c10[batch_indices, pixel_indices]
        colour_dist = torch.distributions.Categorical(logits=selected_colour_logits)
        log_prob_pixels = pixel_dist.log_prob(pixel_indices)
        log_prob_colours = colour_dist.log_prob(colour_actions)
        log_probs = log_prob_pixels + log_prob_colours
        entropy_pixels = pixel_dist.entropy()
        entropy_colours = colour_dist.entropy()
        entropy = entropy_pixels + entropy_colours
        # Return value shape (B, 1)
        return values, log_probs, entropy


# ─── Training Setup (Adjust hyperparameters as needed) ──────────────
TOTAL_STEPS = 1_000_000 # Might need significantly more steps
N_ENVS      = 8
LR          = 5e-5     # Potentially lower LR for complex models/tasks
N_STEPS     = H * W    # Rollout buffer size per env (900)
MINIBATCH_SIZE = 64    # PPO minibatch size (ensure divisible by N_STEPS*N_ENVS if possible, or accept warning)
N_EPOCHS    = 10
CLIP_RANGE  = 0.1
ENT_COEF    = 0.01
VF_COEF     = 0.5
GAE_LAMBDA  = 0.95
GAMMA       = 0.99
MAX_GRAD_NORM = 0.5

def make_env():
    return arc_env.ARCPuzzleEnv() # Uses the modified env

print("Setting up vectorized environment...")
venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
print(f"Vectorized environment created with {N_ENVS} parallel envs.")

print("\n--- PPO Configuration ---")
print(f"Using LMAFeaturesExtractor")
print(f"  LMA Output Dim: {LMA_OUTPUT_FEATURES_DIM}")
print(f"Learning Rate (LR): {LR}")
print(f"Steps per Env per Update (n_steps): {N_STEPS}")
print(f"Total Rollout Buffer Size: {N_STEPS * N_ENVS}")
print(f"Minibatch Size: {MINIBATCH_SIZE}")
print(f"PPO Epochs per Update: {N_EPOCHS}")
# (Keep other hyperparameter prints)
print(f"Discount Factor (gamma): {GAMMA}")
print(f"GAE Lambda: {GAE_LAMBDA}")
print(f"Clip Range: {CLIP_RANGE}")
print(f"Entropy Coefficient (ent_coef): {ENT_COEF}")
print(f"Value Function Coefficient (vf_coef): {VF_COEF}")
print(f"Max Grad Norm: {MAX_GRAD_NORM}")
print("-" * 30)

print("Initializing PPO model with LMA Policy...")
model = PPO(
    policy=ARCPolicy,
    env=venv,
    learning_rate=LR,
    n_steps=N_STEPS,
    batch_size=MINIBATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    verbose=1,
    tensorboard_log="./tensorboard_arc_lma_fewshot/",
)
print("PPO model initialized.")

print("\n--- Starting Training ---")
print(f"Total Timesteps: {TOTAL_STEPS}")
print("-" * 30)
model.learn(total_timesteps=TOTAL_STEPS, log_interval=1)

print("\n" + "-" * 30)
print("--- Training Finished ---")
save_path = "ppo_arc_paint_lma_fewshot"
model.save(save_path)
print(f"Model saved to {save_path}.zip")
print("-" * 30)