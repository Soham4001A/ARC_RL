# --- In train_ppo.py ---

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Dict, Any, Callable, List
from gymnasium import spaces # Import spaces for type hint
import os # <-- Add os
import json # <-- Add json
import time # <-- Add time
from pathlib import Path # <-- Add Path
import random
from optim.sgd import DAG

from stable_baselines3 import PPO as BasePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback, CallbackList 

# Import the modified environment
from Utils import arc_env
# --- Import LMA ---
from Utils.classes import LMAFeaturesExtractor, LMAConfigRL # Make sure LMAConfigRL is also imported if needed separately

# --- Constants ---
H, W = arc_env.H, arc_env.W
N_COLORS = arc_env.N_COLORS
TOTAL_CHANNELS = arc_env.TOTAL_CHANNELS
GRID_H, GRID_W = H, W
PAD_VALUE = arc_env.PAD_VALUE             # <-- Get PAD_VALUE from env
MAX_TRAIN_PAIRS = arc_env.MAX_TRAIN_PAIRS # <-- Get MAX_TRAIN_PAIRS from env
EVAL_DATA_DIR = Path("./data/evaluation") # <-- Define eval data path
EVAL_LOG_DIR = Path("./evaluation_logs")  # <-- Define log directory

USE_CNN = True

# --- LMA Configuration ---
# (Keep your LMA configuration as before)
LMA_SEQ_LEN = H * W
#LMA_SEQ_LEN = 11
LMA_EMBED_DIM = 256
LMA_NUM_HEADS_STACKING = 256
#LMA_TARGET_L_NEW = 11
LMA_TARGET_L_NEW = 256
LMA_D_NEW = 256
LMA_NUM_HEADS_LATENT = 32
LMA_FF_LATENT_HIDDEN = LMA_D_NEW * 4
LMA_NUM_LAYERS = 4
LMA_DROPOUT = 0.1
LMA_BIAS = True

# --- Calculate LMA Output Dimension ---
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

# --- Head Dimensions (using LMA output) ---
VALUE_HIDDEN_DIM = 256
POLICY_HIDDEN_DIM = 512
CNN_FEATURES_DIM = 512 # Example dimension, adjust as needed

# --- ADDED STARTUP PRINT ---
print("--- PPO Training Script with LMA ---")
print(f"Grid Dimensions (H, W): ({H}, {W})")
print(f"Observation Channels: {TOTAL_CHANNELS}")
print(f"LMA Feature Dim: {LMA_OUTPUT_FEATURES_DIM}")
print(f"Value Head Hidden Dim: {VALUE_HIDDEN_DIM}")
print(f"Policy Head Hidden Dim: {POLICY_HIDDEN_DIM}")
print(f"Padding Value: {PAD_VALUE}")
print(f"Max Train Pairs in Obs: {MAX_TRAIN_PAIRS}")
print(f"Evaluation Data Dir: {EVAL_DATA_DIR}")
print(f"Evaluation Log Dir: {EVAL_LOG_DIR}")
print("-" * 30)

# Ensure log directory exists
EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Extractor is now LMA (imported from Utils.classes) ---
# Sample CNN Feature Extractor
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

# ─── Custom Policy using LMA Features ───────────────────────
# (Keep ARCPolicy class exactly as before)
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
        if USE_CNN:
            kwargs['features_extractor_class'] = CNNARCFeatures
            kwargs['features_extractor_kwargs'] = dict(features_dim=CNN_FEATURES_DIM)
        else:
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
        if USE_CNN:
            feature_dim = self.features_extractor.features_dim
        else:
            calculated_feature_dim = self.features_extractor.features_dim
            if calculated_feature_dim != LMA_OUTPUT_FEATURES_DIM:
                 print(f"WARNING: LMA feature dim mismatch! Expected {LMA_OUTPUT_FEATURES_DIM}, Got {calculated_feature_dim} from extractor.")

        if USE_CNN:
            output_feature_dim = feature_dim
        else:
            output_feature_dim = LMA_OUTPUT_FEATURES_DIM
            
        self.policy_ffn = nn.Sequential(
            nn.Linear(output_feature_dim, POLICY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_DIM, GRID_H * GRID_W * N_COLORS)
        )
        self.value_net = nn.Sequential(
            nn.Linear(output_feature_dim, VALUE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_DIM, 1) # Output single value
        )
        # Commenting out initialization prints within policy as they are redundant now
        # print(f"\nARCPolicy initialized with LMAFeaturesExtractor.")
        # print(f"  Policy head input dim: {LMA_OUTPUT_FEATURES_DIM}, hidden: {POLICY_HIDDEN_DIM}")
        # print(f"  Value head input dim: {LMA_OUTPUT_FEATURES_DIM}, hidden: {VALUE_HIDDEN_DIM}")
        # print("-" * 30)

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
        features = self.extract_features(obs)
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
        return actions, values.flatten(), log_probs

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.extract_features(obs)
        colour_logits_b900c10, pixel_scores_b900 = self._logits_from_features(features)
        values = self.value_net(features)
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
        return values, log_probs, entropy


# ─── Grid Padding Utility (Needed for Evaluation) ───────────
def pad_grid(grid, target_shape=(H, W), fill_value=0):
    """Pads/crops a grid (list or np.array) to the target shape."""
    grid_np = np.array(grid, dtype=np.int8)
    h, w = grid_np.shape
    g = np.full(target_shape, fill_value, dtype=np.int8)
    h_slice = min(h, target_shape[0])
    w_slice = min(w, target_shape[1])
    g[:h_slice, :w_slice] = grid_np[:h_slice, :w_slice]
    return g

# ─── Evaluation Callback ────────────────────────────────────
class ARCEvalCallback(BaseCallback):
    """
    Callback for evaluating the PPO agent on the ARC evaluation dataset.
    Handles padding, prediction simulation, cropping, and logging.
    """
    def __init__(self, eval_freq: int, eval_data_dir: Path, log_dir: Path, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_data_dir = eval_data_dir
        self.log_dir = log_dir
        self.last_eval_step = 0
        self.eval_count = 0
        self._start_time = time.time()

        if not self.eval_data_dir.is_dir():
            print(f"Warning: Evaluation data directory not found: {self.eval_data_dir}")
            self.eval_tasks = []
        else:
            self.eval_tasks = sorted([f for f in self.eval_data_dir.glob("*.json")])
            print(f"Found {len(self.eval_tasks)} evaluation tasks.")

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if it's time for evaluation
        if self.num_timesteps >= self.last_eval_step + self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self.eval_count += 1
            start_eval_time = time.time()

            print(f"\n--- Running Evaluation {self.eval_count} at Timestep {self.num_timesteps} ---")
            accuracy, log_entries = self._run_evaluation()
            end_eval_time = time.time()
            eval_duration = end_eval_time - start_eval_time

            # Log to TensorBoard
            if self.logger:
                self.logger.record("eval/accuracy", accuracy)
                self.logger.record("eval/duration_seconds", eval_duration)
                self.logger.dump(step=self.num_timesteps) # Ensure logs are written

            # Print results
            print(f"Evaluation Accuracy: {accuracy:.4f}")
            print(f"Evaluation Duration: {eval_duration:.2f} seconds")

            # Write detailed log file
            log_filename = self.log_dir / f"eval_log_{self.eval_count}_step_{self.num_timesteps}.json"
            try:
                with open(log_filename, 'w') as f:
                    json.dump(log_entries, f, indent=4)
                print(f"Detailed evaluation log saved to: {log_filename}")
            except Exception as e:
                print(f"Error writing evaluation log: {e}")

            print("-" * (len(f"--- Running Evaluation {self.eval_count} at Timestep {self.num_timesteps} ---"))) # Match length

        return True # Continue training

    def _run_evaluation(self) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Performs the evaluation loop over a random sample of the evaluation dataset tasks.
        Handles padding, prediction simulation, cropping, timing, detailed metrics, and logging.
        """
        # --- Initial Checks ---
        if not hasattr(self, 'eval_tasks') or not self.eval_tasks:
            print("[Eval Info] No evaluation tasks found or loaded (self.eval_tasks is missing or empty). Skipping evaluation.")
            return 0.0, []
        if not hasattr(self, 'model') or not hasattr(self.model, 'policy'):
             print("[Eval Error] Model or policy not found in callback. Cannot evaluate.")
             return 0.0, []
        # --- End Initial Checks ---
    
        total_predictions = 0 # Counts processed test pairs
        correct_predictions = 0 # Counts exact matches
        log_entries = []
    
        # --- Use Global Constants (defined outside the callback, imported from arc_env) ---
        # These should be defined in the main script scope using imports from arc_env
        # Example: H, W, N_COLORS, PAD_VALUE, MAX_TRAIN_PAIRS, TOTAL_CHANNELS, episode_len
        # We assume they are accessible here without redefinition.
        # If not, you MUST ensure they are correctly passed or made available, e.g., via self.H, self.W etc.
        # For safety, let's add fallbacks here, but fixing global scope is better.
        try:
             H = arc_env.H
             W = arc_env.W
             N_COLORS = arc_env.N_COLORS
             PAD_VALUE = arc_env.PAD_VALUE
             MAX_TRAIN_PAIRS = arc_env.MAX_TRAIN_PAIRS
             TOTAL_CHANNELS = arc_env.TOTAL_CHANNELS
             episode_len = H * W
        except AttributeError:
             print("[Eval Warn] Could not access constants from arc_env module. Using hardcoded defaults.")
             H, W = 30, 30
             episode_len = 900
             N_COLORS = 10
             PAD_VALUE = -2
             MAX_TRAIN_PAIRS = 5
             TOTAL_CHANNELS = 11
    
        # --- Select Subset of Tasks ---
        num_tasks_to_eval = 20
        tasks_to_run = []
        if not self.eval_tasks:
             print("[Eval Info] No evaluation tasks available. Skipping.")
             return 0.0, []
    
        num_available_tasks = len(self.eval_tasks)
        num_to_sample = min(num_tasks_to_eval, num_available_tasks)
    
        if num_available_tasks > 0:
            try:
                 tasks_to_run = random.sample(self.eval_tasks, num_to_sample)
                 print(f"[Eval Info] Evaluating on a random subset of {len(tasks_to_run)} task(s) from {num_available_tasks} total.")
            except ValueError as e:
                 print(f"[Eval Warn] Error sampling tasks: {e}. Evaluating all {num_available_tasks} tasks.")
                 tasks_to_run = self.eval_tasks
        # --- End Subset Selection ---
    
        if not tasks_to_run:
             print("[Eval Info] No tasks selected for evaluation run. Skipping.")
             return 0.0, []
    
        # --- Start Evaluation ---
        eval_start_time = time.time()
        original_training_mode = self.model.policy.training
        self.model.policy.eval() # Set policy to evaluation mode
    
        # --- Get Policy Device Once ---
        try:
            policy_device = next(self.model.policy.parameters()).device
        except StopIteration:
            policy_device = self.model.device # Fallback
    
        # Loop over the selected tasks (tasks_to_run)
        for task_idx, task_file in enumerate(tasks_to_run):
            task_id = task_file.stem
            task_start_time = time.time()
            print(f"\n[Eval Progress] Starting Task {task_idx + 1}/{len(tasks_to_run)}: {task_id}")
    
            # --- Load Task Data ---
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
            except Exception as e:
                print(f"  [Eval Warn] Failed to load/parse {task_id}: {e}")
                log_entries.append({"task_id": task_id, "status": "error", "message": f"Failed to load/parse JSON: {e}"})
                continue # Skip task
    
            task_train_pairs = task_data.get('train', [])
            task_test_pairs = task_data.get('test', [])
    
            if not task_test_pairs:
                print(f"  [Eval Info] No test pairs found in {task_id}. Skipping task.")
                continue # Skip task
    
            # --- Pre-pad training pairs for observation ---
            padded_task_train_pairs = []
            for i, pair in enumerate(task_train_pairs):
                if i >= MAX_TRAIN_PAIRS: break
                try:
                    p_in = pad_grid(pair['input'], target_shape=(H, W), fill_value=PAD_VALUE)
                    p_out = pad_grid(pair['output'], target_shape=(H, W), fill_value=PAD_VALUE)
                    padded_task_train_pairs.append((p_in, p_out))
                except Exception as e:
                     print(f"  [Eval Warn] Failed to pad training pair {i} for task {task_id}: {e}.")
                     # Continue task even if one pair fails padding
    
            # --- Loop through test pairs within the task ---
            task_correct = 0
            task_total = 0
            for test_idx, test_pair in enumerate(task_test_pairs):
                total_predictions += 1 # Global counter for processed pairs
                task_total += 1        # Task-specific counter
                test_pair_start_time = time.time()
                simulation_failed = False # Reset flag for each pair
                detailed_metrics = None # Reset metrics for each pair
    
                # --- Prepare Input/Output ---
                try:
                    test_input_orig = np.array(test_pair['input'], dtype=np.int8)
                    target_output_orig = np.array(test_pair['output'], dtype=np.int8)
                    target_h, target_w = target_output_orig.shape
                    padded_test_input = pad_grid(test_input_orig, target_shape=(H, W), fill_value=0) # Pad input with 0
                except Exception as e:
                    print(f"    [Eval Warn] Failed to process input/output for test pair {test_idx} in {task_id}: {e}")
                    log_entries.append({"task_id": task_id, "test_index": test_idx, "status": "error", "message": f"Input/output processing error: {e}"})
                    continue # Skip test pair
    
                # --- Simulate Episode ---
                predicted_canvas = np.full((H, W), -1, dtype=np.int8)
                sim_start_time = time.time()
                total_obs_build_time, total_model_forward_time, total_other_step_time = 0, 0, 0
    
                try:
                    with torch.no_grad():
                        for step in range(episode_len): # Uses episode_len defined at function start
                            obs_build_start = time.time()
                            # _build_eval_observation uses H,W,PAD_VALUE,MAX_TRAIN_PAIRS,TOTAL_CHANNELS
                            current_obs = self._build_eval_observation(
                                predicted_canvas, padded_test_input, padded_task_train_pairs
                            )
                            obs_build_end = time.time()
                            total_obs_build_time += (obs_build_end - obs_build_start)
    
                            model_forward_start = time.time()
                            action_tensor, _, _ = self.model.policy.forward(current_obs.to(policy_device), deterministic=True)
                            model_forward_end = time.time()
                            total_model_forward_time += (model_forward_end - model_forward_start)
    
                            other_step_start = time.time()
                            action = action_tensor.cpu().numpy()[0]
                            r, c, color = map(int, action)
                            # Uses H, W, N_COLORS for clipping
                            r, c, color = np.clip(r, 0, H-1), np.clip(c, 0, W-1), np.clip(color, 0, N_COLORS-1)
                            if predicted_canvas[r, c] == -1:
                                predicted_canvas[r, c] = color
                            other_step_end = time.time()
                            total_other_step_time += (other_step_end - other_step_start)
    
                except Exception as e:
                     print(f"    [Eval Error] Exception during simulation for test pair {test_idx} in {task_id}: {e}")
                     simulation_failed = True
                     if not any(le['task_id'] == task_id and le['test_index'] == test_idx and le['status'] == 'error' for le in log_entries):
                          log_entries.append({"task_id": task_id, "test_index": test_idx, "status": "error", "message": f"Runtime error during simulation: {e}"})
    
                sim_duration = time.time() - sim_start_time
    
                # --- Post-Processing & Metrics Calculation ---
                is_match = False
                if not simulation_failed:
                    try:
                        # reconstruct and pad
                        final_board = np.where(predicted_canvas >= 0,
                                               predicted_canvas,
                                               padded_test_input)
                
                        # build core mask
                        core_mask = np.zeros((H, W), dtype=bool)
                        core_mask[:target_h, :target_w] = True
                
                        # pad true target
                        padded_target = pad_grid(target_output_orig, target_shape=(H, W), fill_value=0)
                
                        # compute pure core accuracy
                        correct_core = int(np.sum(final_board[core_mask] == padded_target[core_mask]))
                        total_core   = int(np.sum(core_mask))
                        core_acc     = (correct_core / total_core) if total_core > 0 else 0.0
                
                        # exact-match = “did we get 100% of core pixels?”
                        if core_acc == 1.0:
                            correct_predictions += 1
                            task_correct += 1
                            is_match = True
                
                        detailed_metrics = {
                            'core_acc':  round(core_acc, 4),
                            'core_count': total_core
                        }
                
                    except Exception as e:
                        print(f"    [Eval Warn] Post-processing error for {task_id}, pair {test_idx}: {e}")
                        is_match = False
                        detailed_metrics = None
                # --- End Post-Processing ---
    
    
                # --- Log Test Pair Results ---
                test_pair_duration = time.time() - test_pair_start_time
                print(f"    Test Pair {test_idx + 1}/{len(task_test_pairs)} finished. Match: {is_match}. Duration: {test_pair_duration:.2f}s")
                if detailed_metrics:
                    print(f"      Core accuracy: {detailed_metrics['core_acc']:.2f} ({detailed_metrics['core_count']} pixels)")
                existing_log = next((item for item in log_entries if item.get("task_id") == task_id and item.get("test_index") == test_idx), None)
    
                if existing_log and existing_log.get("status") == "error":
                     existing_log.update({"simulation_duration_sec": round(sim_duration, 4)})
                else:
                    log_entry_update = {
                        "task_id": task_id, "test_index": test_idx, "match": is_match,
                        "status": "completed" if not simulation_failed else "error",
                        "message": f"Runtime error during simulation" if simulation_failed else None,
                        "target_shape": (target_h, target_w) if 'target_h' in locals() else None,
                        "predicted_shape_before_crop": tuple(predicted_canvas.shape) if 'predicted_canvas' in locals() else None,
                        "cropped_prediction_shape": tuple(cropped_prediction.shape) if 'cropped_prediction' in locals() else None,
                        "simulation_duration_sec": round(sim_duration, 4),
                        "metrics": detailed_metrics,
                        "avg_step_time_sec": round(sim_duration / episode_len, 6) if episode_len > 0 else 0,
                        "avg_obs_build_time_sec": round(total_obs_build_time / episode_len, 6) if episode_len > 0 else 0,
                        "avg_model_forward_time_sec": round(total_model_forward_time / episode_len, 6) if episode_len > 0 else 0,
                    }
                    if log_entry_update["message"] is None: del log_entry_update["message"]
                    log_entries = [item for item in log_entries if not (item.get("task_id") == task_id and item.get("test_index") == test_idx and item.get("status") != "error")]
                    log_entries.append(log_entry_update)
                # --- End Logging Logic ---
            # --- End test pair loop ---
    
            task_accuracy = task_correct / task_total if task_total > 0 else 0.0
            task_duration = time.time() - task_start_time
            print(f"[Eval Progress] Finished Task {task_id}. Accuracy: {task_accuracy:.2f} ({task_correct}/{task_total}). Duration: {task_duration:.2f}s")
        # --- End task loop ---
    
        # --- Restore Training Mode & Final Reporting ---
        if original_training_mode:
            self.model.policy.train()
        eval_duration = time.time() - eval_start_time
    
        valid_predictions = sum(1 for le in log_entries if le['status'] != 'error')
        accuracy = (correct_predictions / valid_predictions) if valid_predictions > 0 else 0.0
    
        print(f"\n--- Evaluation Complete ---")
        print(f"Tasks Evaluated (Sampled/Total): {len(tasks_to_run)}/{num_available_tasks}")
        print(f"Total Test Pairs Processed (Attempted): {total_predictions}")
        print(f"Total Test Pairs Successfully Simulated & Compared: {valid_predictions}")
        print(f"Total Correct Predictions (Exact Match): {correct_predictions}")
        print(f"Overall Accuracy (on processed pairs): {accuracy:.4f}")
        print(f"Total Evaluation Duration: {eval_duration:.2f}s")
        print(f"---------------------------\n")
    
        return accuracy, log_entries
    
    
    # Make sure _build_eval_observation is defined within the class as well
    def _build_eval_observation(self, current_canvas: np.ndarray, padded_test_input: np.ndarray, padded_train_pairs: list) -> torch.Tensor:
        """
        Constructs the multi-channel observation tensor for evaluation,
        mirroring the structure used during training in ARCPuzzleEnv.
        Creates the tensor directly on the model's device.
        """
        # --- Constants Needed ---
        # These need to be accessible here, e.g., self.H, self.W, self.TOTAL_CHANNELS, etc.
        # Or fetched from self.training_env as done initially in _run_evaluation
        try:
            H, W = self.training_env.get_attr("H")[0], self.training_env.get_attr("W")[0]
            PAD_VALUE = self.training_env.get_attr("PAD_VALUE")[0]
            MAX_TRAIN_PAIRS = self.training_env.get_attr("MAX_TRAIN_PAIRS")[0]
            TOTAL_CHANNELS = self.training_env.get_attr("TOTAL_CHANNELS")[0]
        except Exception: # Fallback if get_attr fails here too
            H, W = 30, 30
            PAD_VALUE = -2
            MAX_TRAIN_PAIRS = 5
            TOTAL_CHANNELS = 11
    
        # 1. Get current state
        current_canvas = np.asarray(current_canvas, dtype=np.int8)
        padded_test_input = np.asarray(padded_test_input, dtype=np.int8)
        current_state = np.where(current_canvas >= 0, current_canvas, padded_test_input)
    
        # 2. Initialize observation tensor
        obs_tensor_np = np.full((TOTAL_CHANNELS, H, W), PAD_VALUE, dtype=np.int8)
    
        # 3. Fill channel 0
        obs_tensor_np[0, :, :] = current_state
    
        # 4. Fill channels for train pairs
        num_pairs_to_use = min(len(padded_train_pairs), MAX_TRAIN_PAIRS)
        for i in range(num_pairs_to_use):
            padded_train_input = np.asarray(padded_train_pairs[i][0], dtype=np.int8)
            padded_train_output = np.asarray(padded_train_pairs[i][1], dtype=np.int8)
            obs_tensor_np[1 + i, :, :] = padded_train_input
            obs_tensor_np[1 + MAX_TRAIN_PAIRS + i, :, :] = padded_train_output
    
        # 5. Flatten, convert, add batch dim, specify device
        obs_flat = obs_tensor_np.flatten()
        obs_tensor = torch.tensor(obs_flat.astype(np.float32), device=self.model.device).unsqueeze(0)
        return obs_tensor


# ─── Training Setup (Adjust hyperparameters as needed) ──────────────
TOTAL_STEPS = 10_000_000 # Might need significantly more steps
N_ENVS      = 32
LR          = 1e-4
N_STEPS     = H * W    # Rollout buffer size per env (900)
MINIBATCH_SIZE = 64
N_EPOCHS    = 10
CLIP_RANGE  = 0.1
ENT_COEF    = 0.01
VF_COEF     = 0.5 #Vf loss is way too high compared to policy loss
GAE_LAMBDA  = 0.95
GAMMA       = 0.99
MAX_GRAD_NORM = 0.5

# --- Evaluation Frequency ---
EVAL_FREQ_STEPS = 900_000 # Evaluate every 90k training steps

# --- Checkpoint Saving ---
CHECKPOINT_FREQ = 5_000_000 # Save a checkpoint every 500k total training steps
CHECKPOINT_DIR = "./checkpoints_arc_lma/" # Directory to save checkpoints

def make_env():
    return arc_env.ARCPuzzleEnv() # Uses the modified env

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

k_schedule = DAG.cosine_decay(k0=2, total_steps=TOTAL_STEPS)
optimizer_kwargs = {
    # required args
    "momentum": 0.9,
    "weight_decay": 1e-5,
    # DAG-specific α-controller knobs
    "hyper": {
        "tau":    1.25,
        "p_star": 0.10,
        "beta":   1/3,
        "eta":    0.3,
        "rho":    0.1,
        "eps":    1e-8,
        # alpha_min, alpha_max, and kappa will be auto-computed if omitted
    },
    # RMS-shrink settings
    "shrink": {
        "lambda_rms":   0.3,
        "s_min":        0.1,
        "gamma":        1.0,
        "ema_beta":     0.98,
        "warmup_steps": 500,
    },
    # schedule your k_val across training
    "k_val":   1.0,
    "k_sched": k_schedule,

    # you can also flip these switches if you want
    "use_exact_sigma": False,
    "sigma_every":     10,
    "sat_every":       50,
}



print("Setting up vectorized environment...")
venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
#venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=GAMMA)
print(f"Vectorized environment created with {N_ENVS} parallel envs.")

print("\n--- PPO Configuration ---")
print(f"Using LMAFeaturesExtractor")
print(f"  LMA Output Dim: {LMA_OUTPUT_FEATURES_DIM}")
print(f"Learning Rate (LR): {LR}")
print(f"Steps per Env per Update (n_steps): {N_STEPS}")
print(f"Total Rollout Buffer Size: {N_STEPS * N_ENVS}")
print(f"Minibatch Size: {MINIBATCH_SIZE}")
print(f"PPO Epochs per Update: {N_EPOCHS}")
print(f"Discount Factor (gamma): {GAMMA}")
print(f"GAE Lambda: {GAE_LAMBDA}")
print(f"Clip Range: {CLIP_RANGE}")
print(f"Entropy Coefficient (ent_coef): {ENT_COEF}")
print(f"Value Function Coefficient (vf_coef): {VF_COEF}")
print(f"Max Grad Norm: {MAX_GRAD_NORM}")
print(f"Evaluation Frequency: {EVAL_FREQ_STEPS} steps")
print("-" * 30)

class PPOWithCustomOpt(BasePPO):
    def __init__(self, *args, optimizer_class=None, optimizer_kwargs=None, **kwargs):
        # stash your optimizer info before BasePPO.__init__ builds the default one
        self._custom_optimizer_class  = optimizer_class
        self._custom_optimizer_kwargs = optimizer_kwargs or {}
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        # 1) run all the usual setup (schedules, policy, etc.)
        super()._setup_model()

        # 2) if user gave us a custom optimizer, replace it
        if self._custom_optimizer_class is not None:
            # SB3 keeps a lr_schedule; use that to get the initial LR
            initial_lr = self.lr_schedule(1.0)
            # build your optimizer over the policy parameters
            self.optimizer = self._custom_optimizer_class(
                self.policy.parameters(),
                lr=initial_lr,
                **self._custom_optimizer_kwargs
            )

print("Initializing PPO model with LMA Policy...")
model = PPOWithCustomOpt(
    policy=ARCPolicy,
    env=venv,
    learning_rate=linear_schedule(LR),
    optimizer_class=DAG,   
    optimizer_kwargs=optimizer_kwargs,
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
    # device='cuda' # Or 'cpu'. SB3 usually detects automatically.
)
print("PPO model initialized.")

# --- Create Checkpoint Directory ---
import os # Make sure os is imported
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoint directory created/exists: {CHECKPOINT_DIR}")

# --- Instantiate Callbacks ---
print("Initializing Callbacks...")
# Evaluation Callback
eval_callback = ARCEvalCallback(
    eval_freq=EVAL_FREQ_STEPS, # Check based on total steps (num_timesteps)
    eval_data_dir=EVAL_DATA_DIR,
    log_dir=EVAL_LOG_DIR,
    verbose=1
)

# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
  save_freq=CHECKPOINT_FREQ, # Check based on total steps (num_timesteps)
  save_path=CHECKPOINT_DIR,
  name_prefix='ppo_arc_lma_ckpt', # Prefix for saved model files
  save_replay_buffer=False, # Not needed for PPO
  save_vecnormalize=False, # Not used here
  verbose=1 # Print message on save
)

# Combine Callbacks into a list
# Using CallbackList is recommended but a simple list usually works too
callback_list = CallbackList([eval_callback, checkpoint_callback])
print("Callbacks initialized (Evaluation and Checkpoint).")


print("\n--- Starting Training ---")
print(f"Total Timesteps: {TOTAL_STEPS}")
print(f"Evaluation Frequency: {EVAL_FREQ_STEPS} steps")
print(f"Checkpoint Frequency: {CHECKPOINT_FREQ} steps")
print("-" * 30)

# --- Pass the list of callbacks to model.learn ---
model.learn(
    total_timesteps=TOTAL_STEPS,
    log_interval=1, # Log training stats less frequently (e.g., every 10 updates) to reduce clutter
    callback=callback_list # Pass the list containing both callbacks
)

print("\n" + "-" * 30)
print("--- Training Finished ---")
# Final save is still good practice
save_path = os.path.join(CHECKPOINT_DIR, "ppo_arc_paint_lma_final") # Save final model in checkpoints dir
model.save(save_path)
print(f"Final model saved to {save_path}.zip")
print("-" * 30)

# --- How to Load a Checkpoint ---
# print("\n--- Example: Loading Checkpoint ---")
# checkpoint_to_load = os.path.join(CHECKPOINT_DIR, "ppo_arc_lma_ckpt_500000_steps.zip") # Example filename
# if os.path.exists(checkpoint_to_load):
#     print(f"Loading model from: {checkpoint_to_load}")
#     loaded_model = PPO.load(checkpoint_to_load, env=venv) # Need to provide env
#     # You can then continue training:
#     # loaded_model.learn(total_timesteps=TOTAL_STEPS - 500000, callback=callback_list, reset_num_timesteps=False)
#     # Or use it for evaluation:
#     # Run evaluation using loaded_model.policy
# else:
#     print(f"Checkpoint {checkpoint_to_load} not found.")
# print("-" * 30)