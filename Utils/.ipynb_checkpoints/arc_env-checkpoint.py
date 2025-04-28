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
    Modified reward function.
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
        self.generated_output = None
        self.original_train_pairs = []
        self.task_id = None
        self.cur_step = 0
        self.episode_len = H * W

        # --- Attributes to store original grid info ---
        self.original_input_grid = None
        self.original_height = 0
        self.original_width = 0
        # --- End new attributes ---


    # --- (_load_original_task, _generate_example_for_task, _obs methods remain the same) ---
    def _load_original_task(self, task_id):
        """Loads the original train pairs for a given task ID."""
        task_file = ORIGINAL_ARC_TASKS.get(task_id)
        if not task_file:
            raise FileNotFoundError(f"Original ARC task file not found for ID: {task_id}")
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            return task_data.get('train', [])[:MAX_TRAIN_PAIRS]
        except Exception as e:
            print(f"Error loading/parsing original task {task_id}: {e}")
            return []

    def _generate_example_for_task(self, task_id):
        """Generates a new input/output pair using re-arc for the given task_id."""
        gen_fn_name = f"generate_{task_id}"
        if not hasattr(generators, gen_fn_name):
             print(f"Warning: re-arc generator {gen_fn_name} not found. Skipping.")
             return None
        gen = getattr(generators, gen_fn_name)
        try:
            n = len(inspect.signature(gen).parameters)
            if n == 2: return gen(0.0, 1.0)
            if n == 1: return gen(self.rng.uniform(0.0, 1.0))
            return gen()
        except Exception as e:
            print(f"Warning: Generator {gen_fn_name} failed: {e}. Trying another task.")
            return None

    def _obs(self):
        """Constructs the multi-channel observation tensor."""
        current_state = np.where(self.canvas >= 0, self.canvas, self.generated_input)
        obs_tensor = np.full((TOTAL_CHANNELS, H, W), PAD_VALUE, dtype=np.int8)
        obs_tensor[0, :, :] = current_state
        num_pairs_to_use = min(len(self.original_train_pairs), MAX_TRAIN_PAIRS)
        for i in range(num_pairs_to_use):
            train_pair = self.original_train_pairs[i]
            padded_train_input = pad_grid(train_pair['input'], fill=PAD_VALUE)
            padded_train_output = pad_grid(train_pair['output'], fill=PAD_VALUE)
            obs_tensor[1 + i, :, :] = padded_train_input
            obs_tensor[1 + MAX_TRAIN_PAIRS + i, :, :] = padded_train_output
        return obs_tensor.flatten()
    # --- End unchanged methods ---


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.rng.seed(seed)

        self.cur_step = 0
        generated_example = None
        while generated_example is None:
            self.task_id = self.rng.choice(TASK_IDS)
            self.original_train_pairs = self._load_original_task(self.task_id)
            if not self.original_train_pairs: continue
            generated_example = self._generate_example_for_task(self.task_id)

        # --- Store original grid and dimensions BEFORE padding ---
        original_output_grid_tmp = np.array(generated_example["output"], dtype=np.int8)
        self.original_input_grid = np.array(generated_example["input"], dtype=np.int8) # Store original input
        self.original_height, self.original_width = original_output_grid_tmp.shape     # Store original output dimensions
        # --- End Store ---

        # Pad for internal state and observation building
        self.generated_input = pad_grid(self.original_input_grid, fill=0)
        self.generated_output = pad_grid(original_output_grid_tmp, fill=0) # Target for comparison

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

        # Initialize per-step bookkeeping
        reward = 0.0
        pixel_correct = 0
        painted_this_step = False
        
        # --- FIRST-PAINT REWARD LOGIC ---
        if self.canvas[r, c] == -1:
            painted_this_step = True
            self.canvas[r, c] = color
        #     target_color = self.generated_output[r, c]
        #     is_core = (0 <= r < self.original_height and 0 <= c < self.original_width)
        #     needed  = (self.generated_input[r, c] != target_color)
        
        #     if is_core and color == target_color and needed:
        #         reward = +1.0;  pixel_correct = 1
        #     elif is_core and color != target_color:
        #         reward = -0.3
        #     elif not is_core and color == target_color:
        #         reward = 0.0
        #     else:
        #         reward = -0.7
        # else:
        #     reward = -0.5         # Penalise any repaint
        
        # reward += -0.01          # Tiny time penalty

        correct = int(color == self.generated_output[r, c])
        reward  = 1.0 * correct - 0.2 * (1 - correct)

        self.cur_step += 1
        terminated = (self.cur_step >= self.episode_len)
        if self.cur_step >= self.episode_len:
            final_board = np.where(self.canvas >= 0, self.canvas, self.generated_input)
            exact_match = int(np.array_equal(final_board, self.generated_output))
            reward += 100.0 * exact_match  # Big terminal bonus

        # --- Return Values ---
        # Use self._obs() to get the next observation if not terminated
        next_obs = self._obs()
        info = {
            "task_id": self.task_id,
            "pixel_correct": pixel_correct if painted_this_step else 0,
            "action_taken": (r, c, color),
            "painted_cell": painted_this_step,
        }
        if terminated:
            info["final_exact_match"] = exact_match # Add final match info

        return next_obs, reward, terminated, False, info # Assuming no truncation