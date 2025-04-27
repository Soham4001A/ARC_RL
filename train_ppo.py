import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F # Import functional for pooling etc.
from torch import Tensor
from typing import Tuple, Dict, Any # For type hints

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN # Can use NatureCNN as a base or guide
from gymnasium import spaces

# Import the modified environment
from Utils import arc_env # Assuming arc_env.py is in Utils directory
# We no longer need ARCModel directly here, as we define new heads
# from Utils.model import ARCModel

# --- Constants ---
# Import from environment for consistency
H, W = arc_env.H, arc_env.W
N_COLORS = arc_env.N_COLORS
TOTAL_CHANNELS = arc_env.TOTAL_CHANNELS # Get number of channels from env
GRID_H, GRID_W = H, W # Alias for clarity

# Define output dimension for the NEW CNN Feature Extractor
CNN_FEATURES_DIM = 512 # Example dimension, adjust as needed
VALUE_HIDDEN_DIM = 256 # Hidden dim for the value network head
POLICY_HIDDEN_DIM = 512 # Hidden dim for the policy network head


class CNNARCFeatures(BaseFeaturesExtractor):
    """
    CNN Feature extractor for the multi-channel ARC environment observation.
    Processes the (B, TOTAL_CHANNELS, H, W) input.
    Inspired by NatureCNN structure used in SB3.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = CNN_FEATURES_DIM):
        # features_dim is the output dimension of this CNN extractor
        super().__init__(observation_space, features_dim)

        n_input_channels = arc_env.TOTAL_CHANNELS
        self.input_height = arc_env.H
        self.input_width = arc_env.W

        # Define CNN layers (Keep your CNN definition here)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            dummy_flat_obs = observation_space.sample()
            dummy_input_shape = (1, n_input_channels, self.input_height, self.input_width)
            dummy_input = torch.as_tensor(dummy_flat_obs).reshape(dummy_input_shape).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        # Linear layer to project flattened features to the desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        # (Keep print statements)
        print(f"CNNARCFeatures initialized.")
        print(f"  Input channels: {n_input_channels}, H: {self.input_height}, W: {self.input_width}")
        print(f"  CNN flattened output size: {n_flatten}")
        print(f"  Final features dim: {features_dim}")


    def forward(self, observations: Tensor) -> Tensor:
        # Observations from SB3 are likely flattened (B, TOTAL_CHANNELS * H * W)
        # Reshape to (B, TOTAL_CHANNELS, H, W)
        batch_size = observations.shape[0]
        # Ensure input is float
        obs_reshaped = observations.view(
            batch_size, TOTAL_CHANNELS, self.input_height, self.input_width
        ).float() # Make sure TOTAL_CHANNELS is correctly defined/imported

        # Pass through CNN and linear layers
        cnn_features = self.cnn(obs_reshaped)
        output_features = self.linear(cnn_features)
        return output_features

# ─── Custom Policy with NEW Heads ───────────────────────────
class ARCPolicy(ActorCriticPolicy):
    """
    Uses the CNNARCFeatures extractor.
    Defines its OWN prediction and value heads operating on CNN features.
    Still uses the two-stage pixel/color decision process.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        # Add other PPO args if needed, like activation_fn, ortho_init
        **kwargs
    ):
         # Explicitly set the feature extractor class and its output dimension
        kwargs['features_extractor_class'] = CNNARCFeatures
        kwargs['features_extractor_kwargs'] = dict(features_dim=CNN_FEATURES_DIM)
        # Can specify shared net architecture after extractor if desired, e.g., net_arch=[128]
        kwargs['net_arch'] = [] # No shared layers between policy/value after CNN extractor
        kwargs['activation_fn'] = nn.ReLU # Use ReLU

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        # --- Define NEW Heads ---
        # These heads take features from CNNARCFeatures (CNN_FEATURES_DIM)

        # Policy head: Mimics ARCPredictionHead structure
        self.policy_ffn = nn.Sequential(
            nn.Linear(CNN_FEATURES_DIM, POLICY_HIDDEN_DIM),
            nn.ReLU(),
            # Output needs to produce logits for all pixels and colors
            nn.Linear(POLICY_HIDDEN_DIM, GRID_H * GRID_W * N_COLORS)
        )

        # Value head: Takes CNN features
        self.value_net = nn.Sequential(
            nn.Linear(CNN_FEATURES_DIM, VALUE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_DIM, 1) # Output single value
        )
        print(f"ARCPolicy initialized with CNNARCFeatures.")
        print(f"  Policy head input dim: {CNN_FEATURES_DIM}, hidden: {POLICY_HIDDEN_DIM}")
        print(f"  Value head input dim: {CNN_FEATURES_DIM}, hidden: {VALUE_HIDDEN_DIM}")

    def _get_policy_logits(self, features: Tensor) -> Tensor:
        """ Applies the policy head FFN and reshapes to (B, H, W, C) """
        # features are the output of CNNARCFeatures
        logits_flat = self.policy_ffn(features) # (B, H*W*C)
        # Reshape to (B, C, H, W) -> (B, H, W, C)
        logits = logits_flat.view(-1, N_COLORS, GRID_H, GRID_W) # (B, C, H, W)
        logits = logits.permute(0, 2, 3, 1).contiguous()        # (B, H, W, C)
        return logits

    # split logits to pixel-heatmap & per-pixel colour logits
    def _logits_from_features(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """ Gets logits from features and calculates pixel scores """
        logits_bhwc = self._get_policy_logits(features)         # (B, H, W, C=10)
        logits = logits_bhwc.view(features.size(0), -1, N_COLORS) # (B, 900, 10)
        pixel_scores = logits.logsumexp(dim=-1)                   # (B, 900)
        return logits, pixel_scores # (B, 900, 10), (B, 900)

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the policy and value network.
        """
        # Get features from the CNN extractor
        features = self.extract_features(obs) # Shape (B, CNN_FEATURES_DIM)

        # Calculate policy action distributions and value estimate from features
        colour_logits_b900c10, pixel_scores_b900 = self._logits_from_features(features)
        values = self.value_net(features) # Shape (B, 1)

        # --- Action Sampling (Two-stage) ---
        # 1. Choose pixel location
        pixel_prob = torch.softmax(pixel_scores_b900, dim=-1)
        pixel_dist = torch.distributions.Categorical(probs=pixel_prob)
        pixel_indices = pixel_dist.sample() if not deterministic else pixel_prob.argmax(dim=-1) # Shape (B,)

        # 2. Choose colour at the selected pixel location
        batch_indices = torch.arange(features.size(0), device=pixel_indices.device)
        # Select the color logits for the chosen pixels: (B, 10)
        selected_colour_logits = colour_logits_b900c10[batch_indices, pixel_indices]
        colour_dist = torch.distributions.Categorical(logits=selected_colour_logits)
        colour_actions = colour_dist.sample() if not deterministic else selected_colour_logits.argmax(dim=-1) # Shape (B,)

        # --- Combine action components ---
        rows = pixel_indices // GRID_W
        cols = pixel_indices % GRID_W
        actions = torch.stack([rows, cols, colour_actions], dim=1) # Shape (B, 3)

        # --- Calculate log probability ---
        log_prob_pixels = pixel_dist.log_prob(pixel_indices)
        log_prob_colours = colour_dist.log_prob(colour_actions)
        log_probs = log_prob_pixels + log_prob_colours # Shape (B,)

        return actions, values, log_probs

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        """
        Predict actions based on observation. (Required by ActorCriticPolicy)
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations. (Required for PPO)
        """
        # Get features from the CNN extractor
        features = self.extract_features(obs) # Shape (B, CNN_FEATURES_DIM)

        # Get policy distributions and value estimates
        colour_logits_b900c10, pixel_scores_b900 = self._logits_from_features(features)
        values = self.value_net(features) # Shape (B, 1)

        # --- Reconstruct distributions based on evaluated actions ---
        # Actions shape (B, 3) -> rows, cols, colours (long type)
        actions = actions.long()
        rows, cols, colour_actions = actions[:, 0], actions[:, 1], actions[:, 2]
        pixel_indices = rows * GRID_W + cols # Shape (B,)

        # 1. Pixel distribution
        pixel_prob = torch.softmax(pixel_scores_b900, dim=-1)
        pixel_dist = torch.distributions.Categorical(probs=pixel_prob)

        # 2. Colour distribution (for the actions taken)
        batch_indices = torch.arange(features.size(0), device=pixel_indices.device)
        selected_colour_logits = colour_logits_b900c10[batch_indices, pixel_indices] # Shape (B, 10)
        colour_dist = torch.distributions.Categorical(logits=selected_colour_logits)

        # --- Calculate log probability and entropy ---
        log_prob_pixels = pixel_dist.log_prob(pixel_indices)
        log_prob_colours = colour_dist.log_prob(colour_actions)
        log_probs = log_prob_pixels + log_prob_colours # Shape (B,)

        entropy_pixels = pixel_dist.entropy()
        entropy_colours = colour_dist.entropy()
        entropy = entropy_pixels + entropy_colours # Total entropy, Shape (B,)

        return values, log_probs, entropy


# ─── Training Setup ────────────────────────────────────────
TOTAL_STEPS = 1_000_000 # Increased steps might be needed
N_ENVS      = 8        # Increase parallel environments if possible
LR          = 1e-4     # May need tuning
N_STEPS     = H * W    # Rollout buffer size per env (900)
MINIBATCH_SIZE = 64    # PPO minibatch size for updates (adjust based on GPU memory)
N_EPOCHS    = 10       # PPO epochs per rollout
CLIP_RANGE  = 0.1      # PPO clip range
ENT_COEF    = 0.01     # Entropy coefficient to encourage exploration
VF_COEF     = 0.5      # Value function loss coefficient
GAE_LAMBDA  = 0.95     # GAE lambda parameter
GAMMA       = 0.99     # Discount factor
MAX_GRAD_NORM = 0.5    # Gradient clipping

# Use the modified environment
def make_env():
    # Pass seed if needed, though make_vec_env handles seeding typically
    return arc_env.ARCPuzzleEnv()

# Set up vectorized environment
venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
# Consider VecNormalize for reward scaling/clipping, but maybe not observation normalization
# venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=GAMMA)


# Initialize PPO model
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
    tensorboard_log="./tensorboard_arc_fewshot/",
    # Consider policy_kwargs for activation fns etc if not set in ARCPolicy init
    # policy_kwargs=dict(activation_fn=nn.ReLU)
)

# Train the model
print("Starting training with few-shot observation environment...")
model.learn(total_timesteps=TOTAL_STEPS, log_interval=1) # Log every rollout
model.save("ppo_arc_paint_fewshot")
print("Finished training and saved as ppo_arc_paint_fewshot")

# Optional: Add evaluation on the real ARC dataset here
# You would need a function that loads ARC tasks, processes them using the policy's
# _predict method (likely deterministically), and compares to ground truth.
# Remember the evaluation function would need to handle the multi-channel input prep.