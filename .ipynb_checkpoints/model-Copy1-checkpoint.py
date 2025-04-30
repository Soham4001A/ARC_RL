import os
import json
import inspect
import random
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
from Utils.classes import LMAFeaturesExtractor, CombinedLoss

# RE-ARC generators
generators = importlib.import_module("re-arc.generators")

# Paths
TRAIN_TASKS_DIR = "./data/training"
EVAL_TASKS_DIR  = "./data/evaluation"

# Hyperparams
BATCH_SIZE      = 32
NUM_EPOCHS      = 100
LEARNING_RATE   = 1e-4
LATENT_DIM      = 3840
HIDDEN_DIM      = 1024
NUM_COLORS      = 10
GRID_H, GRID_W  = 30, 30
MAX_TRAIN_PAIRS = 5
CHANNELS        = 1 + 2 * MAX_TRAIN_PAIRS  # 11
PAD_VALUE       = 0
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_grid(grid, target_shape=(GRID_H, GRID_W), pad_value=PAD_VALUE):
    arr = np.array(grid, dtype=np.int64)
    cropped = arr[:target_shape[0], :target_shape[1]]
    out = np.full(target_shape, pad_value, dtype=np.int64)
    out[:cropped.shape[0], :cropped.shape[1]] = cropped
    return out

def pad_and_crop_tensor(x: torch.Tensor):
    h, w = x.shape
    h0, w0 = min(h, GRID_H), min(w, GRID_W)
    cropped = x[:h0, :w0]
    return F.pad(cropped, (0, GRID_W - w0, 0, GRID_H - h0), value=PAD_VALUE)

class OnlineREARCDataset(IterableDataset):
    def __init__(self, train_tasks_dir=TRAIN_TASKS_DIR, pairs_per_epoch=100_000):
        super().__init__()
        self.gen = generators
        self.pairs_per_epoch = pairs_per_epoch
        # preload few-shot
        self.true_shots = {}
        for fname in os.listdir(train_tasks_dir):
            if not fname.endswith(".json"): continue
            tid = fname[:-5]
            data = json.load(open(os.path.join(train_tasks_dir, fname)))
            pairs = data.get("train", [])[:MAX_TRAIN_PAIRS]
            shot_tensors = []
            for p in pairs:
                ti = torch.from_numpy(np.array(p["input"], np.int64))
                to = torch.from_numpy(np.array(p["output"], np.int64))
                shot_tensors.append((ti, to))
            if shot_tensors:
                self.true_shots[tid] = shot_tensors

        gen_names = {fn.split("_",1)[1] for fn in dir(self.gen) if fn.startswith("generate_")}
        self.task_ids = [tid for tid in self.true_shots if tid in gen_names]
        if not self.task_ids:
            raise RuntimeError("No ARC tasks match RE-ARC generators")

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        rng = random.Random(wid)
        per_worker = (self.pairs_per_epoch + num_workers - 1) // num_workers
        yielded = 0

        while yielded < per_worker:
            tid = rng.choice(self.task_ids)
            shots = self.true_shots[tid]
            gen_fn = getattr(self.gen, f"generate_{tid}")
            sig    = inspect.signature(gen_fn).parameters
            ex = None
            for _ in range(5):
                try:
                    if len(sig)==2:
                        ex = gen_fn(0.0, 1.0)
                    elif len(sig)==1:
                        ex = gen_fn(rng.uniform(0.0,1.0))
                    else:
                        ex = gen_fn()
                    break
                except:
                    continue
            if ex is None:
                continue
            # unpack
            if isinstance(ex, dict):
                in_g, out_g = ex["input"], ex["output"]
            elif isinstance(ex, (list,tuple)) and len(ex)==2:
                in_g, out_g = ex
            else:
                continue

            try:
                gen_in  = pad_and_crop_tensor(torch.from_numpy(np.array(in_g, np.int64)))
                gen_out = pad_and_crop_tensor(torch.from_numpy(np.array(out_g, np.int64)))
            except:
                continue

            obs = torch.full((CHANNELS, GRID_H, GRID_W), PAD_VALUE, dtype=torch.int64)
            obs[0] = gen_in
            for i,(ti,to) in enumerate(shots):
                obs[1+i]                   = pad_and_crop_tensor(ti)
                obs[1+MAX_TRAIN_PAIRS+i]   = pad_and_crop_tensor(to)

            yielded += 1
            yield obs.float(), gen_out.long()

class ARCPredictionHead(nn.Module):
    def __init__(self, latent_dim, hidden_dim, gh, gw, num_classes=NUM_COLORS):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gh*gw*num_classes)
        )
        self.gh, self.gw, self.num_classes = gh, gw, num_classes

    def forward(self, features):
        x = self.ffn(features)
        return x.view(-1, self.num_classes, self.gh, self.gw)

class DummyObservationSpace:
    def __init__(self, shape): self.shape = shape

class ARCModel(nn.Module):
    def __init__(self):
        super().__init__()
        gh, gw = GRID_H, GRID_W
        seq_len = gh*gw
        target_l = gh
        d_new    = 128
        assert target_l * d_new == LATENT_DIM

        dummy_dim = CHANNELS * seq_len
        dummy_obs = DummyObservationSpace(shape=(dummy_dim,))

        self.feature_extractor = LMAFeaturesExtractor(
            observation_space = dummy_obs,
            embed_dim          = 128,
            num_heads_stacking = 128,
            target_l_new       = target_l,
            d_new              = d_new,
            num_heads_latent   = 32,
            ff_latent_hidden   = d_new*4,
            num_lma_layers     = 4,
            seq_len            = seq_len,
            dropout            = 0.1,
            bias               = True
        )
        if self.feature_extractor.features_dim != LATENT_DIM:
            raise RuntimeError("LMA output dim mismatch")

        self.pred_head = ARCPredictionHead(LATENT_DIM, HIDDEN_DIM, gh, gw)

    def forward(self, obs):
        B    = obs.size(0)
        flat = obs.view(B, -1)
        feats= self.feature_extractor(flat)
        return self.pred_head(feats)

if __name__ == "__main__":
    ds     = OnlineREARCDataset(pairs_per_epoch=100_000)
    loader = DataLoader(ds,
                        batch_size=BATCH_SIZE,
                        num_workers=4,
                        pin_memory=True)

    eval_files = [f for f in os.listdir(EVAL_TASKS_DIR) if f.endswith(".json")]

    model     = ARCModel().to(DEVICE)
    opt       = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5).to(DEVICE)

    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        tot_loss = 0.0
        for obs, tgt in loader:
            obs, tgt = obs.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            logits = model(obs)
            loss   = criterion(logits, tgt)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * obs.size(0)

        # <-- here is the fix: use ds.pairs_per_epoch -->
        avg_loss = tot_loss / ds.pairs_per_epoch

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for fn in eval_files:
                data = json.load(open(os.path.join(EVAL_TASKS_DIR, fn)))
                shots = data.get("train", [])[:MAX_TRAIN_PAIRS]
                padded = [(pad_grid(p["input"]), pad_grid(p["output"])) for p in shots]
                for test_pair in data.get("test", []):
                    ti, to = pad_grid(test_pair["input"]), pad_grid(test_pair["output"])
                    obs_np = np.full((CHANNELS, GRID_H, GRID_W), PAD_VALUE, np.int64)
                    obs_np[0] = ti
                    for i,(pti,pto) in enumerate(padded):
                        obs_np[1+i]                = pti
                        obs_np[1+MAX_TRAIN_PAIRS+i] = pto
                    obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(DEVICE)
                    pred  = model(obs_t).argmax(dim=1).squeeze(0).cpu().numpy()
                    if np.array_equal(pred, to):
                        correct += 1
                    total   += 1

        acc = correct/total if total else 0.0
        print(f"Epoch {ep:03d} | Loss {avg_loss:.4f} | Eval Acc {acc:.4f}")

    torch.save(model.state_dict(), "arc_sft_model.pt")