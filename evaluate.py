#!/usr/bin/env python3
# evaluate_ppo.py  –  exact-match ARC score for a trained PPO agent
# ---------------------------------------------------------------
import json, pathlib, numpy as np, torch
from tqdm import tqdm

import Utils.arc_env as arc_env                       # your custom Gymnasium env
from train_ppo import ARCPolicy      # ← reuse policy & feature extractor

CHECKPOINT = "ppo_arc_paint"         # <name used in model.save(...)>
EVAL_DIR   = pathlib.Path(".data/evaluation")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- helper -------------------------------------------------
def pad_grid(grid, H=30, W=30, fill=0):
    g = np.full((H, W), fill, dtype=np.int8)
    g[:min(len(grid), H), :min(len(grid[0]), W)] = np.asarray(grid, dtype=np.int8)[:H, :W]
    return g

# -------------- load evaluation tasks ---------------------------------
tasks = []
for jf in sorted(EVAL_DIR.glob("*.json")):
    with open(jf) as f:
        ex = json.load(f)
    tasks.append((jf.name, pad_grid(ex["input"]), pad_grid(ex["output"])))
print(f"Loaded {len(tasks)} evaluation tasks")

# -------------- rebuild policy & load weights -------------------------
dummy_env = arc_env.ARCPuzzleEnv(seed=0)          # obs-/action-space only
policy     = ARCPolicy(
    observation_space=dummy_env.observation_space,
    action_space=dummy_env.action_space,
    lr_schedule=lambda _: 0.0,                    # not used in eval
)
policy = policy.to(DEVICE)
policy.load_state_dict(torch.load(f"{CHECKPOINT}.zip",
                                  map_location=DEVICE)["policy"])  # SB3 saves dicts

policy.eval()                                     # switch to inference mode

# -------------- run deterministic rollout --------------------------------
def paint(example_input):
    env = arc_env.ARCPuzzleEnv()                  # fresh env per task
    env.input_grid = example_input.copy()
    env.canvas.fill(-1)
    obs = env._obs()

    with torch.no_grad():
        for _ in range(env.episode_len):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
            action, _, _ = policy(obs_t, deterministic=True)
            r,c,color = action.squeeze(0).cpu().numpy()
            obs, *_ = env.step((int(r), int(c), int(color)))
            if obs is None:                       # episode finished
                break
    final_board = np.where(env.canvas >= 0, env.canvas, env.input_grid)
    return final_board

# -------------- iterate through tasks -----------------------------------
perfect, total = 0, len(tasks)
for name, inp, gt in tqdm(tasks, desc="evaluating"):
    pred = paint(inp)
    ok   = np.array_equal(pred, gt)
    perfect += ok
    print(f"{name:20s}  {'✓' if ok else '✗'}")

acc = perfect / total * 100.0
print(f"\nTask-level accuracy: {perfect}/{total}  =  {acc:.2f}%")