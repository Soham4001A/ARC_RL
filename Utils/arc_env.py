import random, importlib, inspect
from pathlib import Path
import numpy as np
from gymnasium import Env, spaces

# ─── constants ──────────────────────────────────────────────
H, W = 30, 30                       # fixed board size
N_COLORS = 10                       # palette 0-9

# ─── import RE-ARC generators ───────────────────────────────
RE_ARC_DIR = Path(__file__).parent / "re-arc"
generators = importlib.import_module("re-arc.generators")
TASK_IDS = [fn.split("_", 1)[1] for fn in dir(generators) if fn.startswith("generate_")]

def pad_grid(grid, fill=0):
    g = np.full((H, W), fill, dtype=np.int8)
    g[:min(len(grid), H), :min(len(grid[0]), W)] = np.asarray(grid, dtype=np.int8)[:H, :W]
    return g

# ─── environment ────────────────────────────────────────────
class ARCPuzzleEnv(Env):
    """
    Autoregressive painting:
        • Action = (row, col, color)  – one pixel per step
        • Observation = current canvas ⊕ test input (flattened)
        • Episode ends after H×W steps
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)

        # action space: choose row, column, colour
        self.action_space = spaces.MultiDiscrete([H, W, N_COLORS])
        self.observation_space = spaces.Box(low=0, high=N_COLORS - 1,
                                            shape=(H * W,), dtype=np.int8)

        self.canvas = np.full((H, W), -1, dtype=np.int8)   # -1 = unpainted
        self.input_grid = None
        self.gt_grid = None
        self.task_id = None
        self.cur_step = 0
        self.episode_len = H * W

    # ─── helpers ────────────────────────────────────────────
    def _sample_example(self):
        while True:
            self.task_id = self.rng.choice(TASK_IDS)
            gen = getattr(generators, f"generate_{self.task_id}")
            try:
                n = len(inspect.signature(gen).parameters)
                if n == 2:
                    return gen(0.0, 1.0)
                if n == 1:
                    return gen(self.rng.uniform(0.0, 1.0))
                return gen()
            except Exception:
                continue

    def _obs(self):
        visible = np.where(self.canvas >= 0, self.canvas, self.input_grid)
        return visible.flatten()

    # ─── Gym API ────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        self.rng.seed(seed)
        self.cur_step = 0

        ex = self._sample_example()
        self.input_grid = pad_grid(ex["input"])
        self.gt_grid    = pad_grid(ex["output"])
        self.canvas.fill(-1)
        return self._obs(), {}

    def step(self, action):
        r, c, color = action
        if self.canvas[r, c] == -1:           # first paint only
            self.canvas[r, c] = color

        correct = int(color == self.gt_grid[r, c])
        reward  = 1.0 * correct - 0.2 * (1 - correct)

        self.cur_step += 1
        terminated = self.cur_step >= self.episode_len
        if terminated:
            final_board = np.where(self.canvas >= 0, self.canvas, self.input_grid)
            reward += 10.0 * int(np.array_equal(final_board, self.gt_grid))

        print(f"[Env] step={self.cur_step:03d} "
              f"pixel_correct={correct} reward={reward:+.1f}")

        obs  = None if terminated else self._obs()
        info = {"pixel_correct": correct}
        return obs, reward, terminated, False, info