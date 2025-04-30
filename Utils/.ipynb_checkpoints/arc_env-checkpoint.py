# --- In Utils/arc_env.py ---

import random
import importlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
from gymnasium import Env, spaces

# ─── constants ──────────────────────────────────────────────
H, W = 30, 30                       # fixed board size
N_COLORS = 10                       # palette 0-9
MAX_TRAIN_PAIRS = 5                 # Max original train pairs to include in obs
TOTAL_CHANNELS = 1 + 2 * MAX_TRAIN_PAIRS # 1 current_state + N train_inputs + N train_outputs
PAD_VALUE = -2                      # A value distinct from colors (0-9) and unpainted (-1)

# ─── Paths and Data Loading ─────────────────────────────────
ARC_DATA_DIR = Path("./data/training")
assert ARC_DATA_DIR.is_dir(), f"ARC training data not found at: {ARC_DATA_DIR}"
ORIGINAL_ARC_TASKS = {f.stem: f for f in ARC_DATA_DIR.glob("*.json")}

# ─── import RE-ARC generators ───────────────────────────────
RE_ARC_DIR = Path(__file__).parent / "re-arc"
TASK_IDS = sorted([
    fn.split("_", 1)[1] for fn in dir(importlib.import_module("re-arc.generators"))
    if fn.startswith("generate_") and fn.split("_", 1)[1] in ORIGINAL_ARC_TASKS
])
assert TASK_IDS, "No common Task IDs found between re-arc generators and ARC_DATA_DIR"
generators = importlib.import_module("re-arc.generators")

# ─── Grid Padding Utility ───────────────────────────────────
def pad_grid(grid, target_shape=(H, W), fill=0):
    """Pads/crops a grid to the target shape."""
    grid_np = np.array(grid, dtype=np.int8)
    h, w = grid_np.shape
    g = np.full(target_shape, fill, dtype=np.int8)
    h_slice = min(h, target_shape[0])
    w_slice = min(w, target_shape[1])
    g[:h_slice, :w_slice] = grid_np[:h_slice, :w_slice]
    return g

# ─── environment ────────────────────────────────────────────
class ARCPuzzleEnv(Env):
    """
    Autoregressive painting with few-shot examples in observation.
    Includes immediate pixel rewards, patch-based shaping reward,
    repaint penalty, time penalty, and terminal bonus.
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)
        self.action_space = spaces.MultiDiscrete([H, W, N_COLORS])
        obs_shape_flat = TOTAL_CHANNELS * H * W
        self.observation_space = spaces.Box(low=PAD_VALUE, high=N_COLORS - 1,
                                            shape=(obs_shape_flat,), dtype=np.int8)

        # State variables
        self.canvas = np.full((H, W), -1, dtype=np.int8)
        self.generated_input = None
        self.generated_output = None # Padded target grid
        self.original_train_pairs = []
        self.task_id = None
        self.cur_step = 0
        self.episode_len = H * W

        # --- Attributes to store original grid info ---
        self.original_input_grid = None
        self.original_height = 0
        self.original_width = 0
        # --- End new attributes ---

        # --- Patch Reward Parameters ---
        self.patch_size = 3
        self.max_patch_reward = 1.0 # Max bonus for a perfectly correct patch (scaled linearly)
        # --- End Patch Parameters ---


    # --- (_load_original_task, _generate_example_for_task, _obs methods - unchanged) ---
    def _load_original_task(self, task_id):
        task_file = ORIGINAL_ARC_TASKS.get(task_id)
        if not task_file: raise FileNotFoundError(f"Original task file not found: {task_id}")
        try:
            with open(task_file, 'r') as f: task_data = json.load(f)
            return task_data.get('train', [])[:MAX_TRAIN_PAIRS]
        except Exception as e: print(f"Error loading task {task_id}: {e}"); return []

    def _generate_example_for_task(self, task_id):
        gen_fn_name = f"generate_{task_id}"
        if not hasattr(generators, gen_fn_name): return None
        gen = getattr(generators, gen_fn_name)
        try:
            n = len(inspect.signature(gen).parameters)
            if n == 2: return gen(0.0, 1.0)
            if n == 1: return gen(self.rng.uniform(0.0, 1.0))
            return gen()
        except Exception as e: print(f"Generator {gen_fn_name} failed: {e}"); return None

    def _obs(self):
        current_state = np.where(self.canvas >= 0, self.canvas, self.generated_input)
        obs_tensor = np.full((TOTAL_CHANNELS, H, W), PAD_VALUE, dtype=np.int8)
        obs_tensor[0, :, :] = current_state
        num_pairs_to_use = min(len(self.original_train_pairs), MAX_TRAIN_PAIRS)
        for i in range(num_pairs_to_use):
            train_pair = self.original_train_pairs[i]
            try: # Add try-except for robustness during obs building
                padded_train_input = pad_grid(train_pair['input'], fill=PAD_VALUE)
                padded_train_output = pad_grid(train_pair['output'], fill=PAD_VALUE)
                obs_tensor[1 + i, :, :] = padded_train_input
                obs_tensor[1 + MAX_TRAIN_PAIRS + i, :, :] = padded_train_output
            except Exception as e:
                 print(f"Warning: Error padding train pair {i} for obs: {e}")
                 # Fill problematic channels with PAD_VALUE maybe?
                 obs_tensor[1 + i, :, :] = PAD_VALUE
                 obs_tensor[1 + MAX_TRAIN_PAIRS + i, :, :] = PAD_VALUE
        return obs_tensor.flatten()
    # --- End unchanged methods ---


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng.seed(seed)
        self.cur_step = 0
        generated_example = None
        while generated_example is None:
            self.task_id = self.rng.choice(TASK_IDS)
            self.original_train_pairs = self._load_original_task(self.task_id)
            if not self.original_train_pairs: continue
            generated_example = self._generate_example_for_task(self.task_id)
            if generated_example is None: continue # Try another task if generator failed

            # Validate generated example structure
            if not isinstance(generated_example, dict) or 'input' not in generated_example or 'output' not in generated_example:
                print(f"Warning: Invalid example structure from generator for {self.task_id}. Retrying.")
                generated_example = None # Force retry
                continue

        try:
            original_output_grid_tmp = np.array(generated_example["output"], dtype=np.int8)
            self.original_input_grid = np.array(generated_example["input"], dtype=np.int8)
            self.original_height, self.original_width = original_output_grid_tmp.shape
            self.generated_input = pad_grid(self.original_input_grid, fill=0)
            self.generated_output = pad_grid(original_output_grid_tmp, fill=0) # Padded target
        except Exception as e:
            print(f"Error processing generated example for {self.task_id}: {e}. Resetting might fail.")
            # Handle error state appropriately, maybe raise or retry reset logic
            raise RuntimeError(f"Failed to process generated example during reset: {e}") from e


        self.canvas.fill(-1)
        initial_obs = self._obs()
        info = {"task_id": self.task_id, "num_train_pairs": len(self.original_train_pairs)}
        return initial_obs, info

    def step(self, action):
        try:
            r, c, color = map(int, action)
        except (ValueError, TypeError):
             print(f"Warning: Received invalid action format: {action}. Using (0,0,0).")
             r, c, color = 0, 0, 0

        r = np.clip(r, 0, H - 1)
        c = np.clip(c, 0, W - 1)
        color = np.clip(color, 0, N_COLORS - 1)

        # Initialize rewards and flags
        reward = 0.0
        pixel_reward = 0.0
        patch_shaping_reward = 0.0
        repaint_penalty = 0.0
        time_penalty = -0.01 # Constant time penalty per step

        painted_this_step = False
        pixel_correct = 0 # 0: incorrect, 1: correct (only if painted_this_step)
        patch_completeness = 0.0 # Initialize patch completeness

        # --- Action Execution & Reward Calculation ---
        if self.canvas[r, c] == -1: # Only evaluate reward if cell is unpainted
            painted_this_step = True
            target_color = int(self.generated_output[r, c]) # Target color from padded final grid

            # 1) Calculate Immediate Per-Pixel Reward
            is_core = (r < self.original_height and c < self.original_width)
            if color == target_color:
                pixel_correct = 1
                if is_core:
                    pixel_reward = 1.0 # Core Correct
                else:
                    pixel_reward = 0.1 # Padding Correct
            else: # Incorrect color painted
                if is_core:
                    pixel_reward = -0.5 # Core Incorrect
                else:
                    pixel_reward = -1.5 # Padding Incorrect

            # Apply the paint action *after* determining correctness based on previous state
            self.canvas[r, c] = color

            # 2) Calculate Patch Shaping Reward (only if we painted)
            # Define patch boundaries centered at (r, c) - clamped to grid edges
            half_patch = self.patch_size // 2
            r_start = max(0, r - half_patch)
            r_end = min(H, r + half_patch + 1) # Slice goes up to, but not including, end
            c_start = max(0, c - half_patch)
            c_end = min(W, c + half_patch + 1)

            # Extract patches (predicted uses current canvas, target uses final target)
            predicted_patch = self.canvas[r_start:r_end, c_start:c_end]
            target_patch = self.generated_output[r_start:r_end, c_start:c_end]

            # Calculate match score (partial credit)
            # Ignore -1 in predicted patch when comparing? Or assume target isn't -1?
            # Let's count matches where prediction is not -1.
            valid_prediction_mask = (predicted_patch != -1)
            matching_pixels = np.sum(predicted_patch[valid_prediction_mask] == target_patch[valid_prediction_mask])
            total_pixels_in_patch = predicted_patch.size # e.g., 9 for 3x3

            if total_pixels_in_patch > 0:
                patch_completeness = matching_pixels / total_pixels_in_patch
                patch_shaping_reward = self.max_patch_reward * patch_completeness
            else:
                patch_completeness = 0.0 # Should not happen if patch_size >= 1
                patch_shaping_reward = 0.0

        else: # Cell was already painted
            painted_this_step = False
            repaint_penalty = -5 # Apply penalty for repainting

        # Combine reward components
        reward = pixel_reward + patch_shaping_reward + repaint_penalty + time_penalty
    
        # --- Episode Termination & Final Bonus ---
        self.cur_step += 1
        done = (self.cur_step >= self.episode_len)
        final_board = None
        solved = 0  # Initialize solved status
    
        if done:
            # Use the standard final board calculation (fills unpainted with input)
            final_board = np.where(self.canvas >= 0, self.canvas, self.generated_input)
            solved = int(np.array_equal(final_board, self.generated_output))
            reward += 100.0 * solved  # Significant final bonus
    
        # Always return a valid observation, even if done.
        # SB3 needs a valid array for "terminal_observation."
        next_obs = self._obs()
    
        info = {
            "painted_cell": painted_this_step,
            "pixel_correct": pixel_correct,  # 1 if painted and correct, 0 otherwise
            "task_id": self.task_id,         # Include task_id
            "step_reward_components": {
                "pixel": pixel_reward,
                "patch_shape": patch_shaping_reward,
                "repaint": repaint_penalty,
                "time": time_penalty,
                "patch_completeness": round(patch_completeness, 3)
            }
        }
    
        if done:
            info["final_exact_match"] = solved
            # Optionally log the final bonus, e.g.:
            # info["step_reward_components"]["final_bonus"] = 100.0 * solved
    
        # Ensure reward is float
        reward = float(reward)
    
        return next_obs, reward, done, False, info