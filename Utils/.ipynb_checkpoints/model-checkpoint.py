import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math # Make sure math is imported if not already

# Import your LMA class (assuming classes.py is in the same directory or PYTHONPATH)
from Utils.classes import *

# ----- Configuration -----
# (Keep configurations as before)
DATA_PARENT_DIR = "./data" 
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4 
LATENT_DIM = 7680 # This MUST match the output dim of LMAFeaturesExtractor
HIDDEN_DIM = 1024
NUM_COLORS = 10
INPUT_SHAPE = (30, 30) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_SPLIT = 0.1 

# ----- Dataset Classes (pad_grid, ARCDataset, ToTensor) -----
# (Keep these classes as defined in the previous response)
def pad_grid(grid, target_shape=(30, 30), pad_value=0):
    """Pads a 2D list (grid) to the target shape."""
    grid_np = np.array(grid, dtype=np.int64)
    height, width = grid_np.shape
    target_height, target_width = target_shape

    if height > target_height or width > target_width:
        raise ValueError(f"Grid shape {grid_np.shape} exceeds target shape {target_shape}")

    padded_grid = np.full(target_shape, pad_value, dtype=np.int64)
    padded_grid[:height, :width] = grid_np
    return padded_grid

class ARCDataset(Dataset):
    def __init__(self, data_dir, split='training', transform=None, max_grid_size=(30, 30)):
        self.task_dir = data_dir 
        self.transform = transform
        self.max_grid_size = max_grid_size
        self.data_pairs = []

        if not os.path.isdir(self.task_dir):
             raise FileNotFoundError(f"Directory not found: {self.task_dir}")

        print(f"Loading tasks from: {self.task_dir}")
        task_files = [f for f in os.listdir(self.task_dir) if f.endswith('.json')]
        
        if not task_files:
             raise FileNotFoundError(f"No JSON task files found in: {self.task_dir}")

        for task_file in task_files:
            task_path = os.path.join(self.task_dir, task_file)
            try:
                with open(task_path, 'r') as f:
                    task_data = json.load(f)

                for example in task_data.get('train', []):
                    input_grid = example['input']
                    output_grid = example['output']
                    try:
                        padded_input = pad_grid(input_grid, self.max_grid_size)
                        padded_output = pad_grid(output_grid, self.max_grid_size)
                        self.data_pairs.append({'input': padded_input, 'output': padded_output})
                    except ValueError as e:
                        print(f"Skipping example in {task_file} due to size issue: {e}")
                        continue 

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {task_file}")
            except Exception as e:
                print(f"Warning: Error processing {task_file}: {e}")
        
        print(f"Loaded {len(self.data_pairs)} examples from {len(task_files)} tasks in {split} split.")
        if not self.data_pairs:
             print(f"Warning: No data pairs loaded from {self.task_dir}. Check dataset structure and content.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sample = self.data_pairs[idx] 
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        input_grid, output_grid = sample['input'], sample['output']
        # Input needs to be float for LMA (and many models)
        # LMA expects flattened input later, but keep it H,W for now
        input_tensor = torch.from_numpy(input_grid).float() 
        # Target shape should be (H, W) with Long type for CrossEntropyLoss
        output_tensor = torch.from_numpy(output_grid).long()
        return {'input': input_tensor, 'output': output_tensor}


# ----- Model Components -----

class ARCPredictionHead(nn.Module):
    # (Keep this class exactly as before)
    def __init__(self, latent_dim, hidden_dim, grid_height, grid_width, num_classes=10):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_height * grid_width * num_classes)
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_classes = num_classes

    def forward(self, features):
        x = self.ffn(features)  # (batch_size, H*W*C)
        x = x.view(-1, self.num_classes, self.grid_height, self.grid_width) # (batch_size, C, H, W)
        return x

# Dummy Observation Space for LMAFeaturesExtractor Initialization
class DummyObservationSpace:
    def __init__(self, shape):
        self.shape = shape

class ARCModel(nn.Module):
    def __init__(self, input_shape, latent_dim=3840, hidden_dim=1024, num_classes=10):
        super().__init__()
        self.grid_height, self.grid_width = input_shape
        self.input_dim_total = self.grid_height * self.grid_width # 30*30 = 900

        # --- LMA Feature Extractor Initialization ---
        # Parameters determined above
        lma_seq_len = 30 
        lma_embed_dim = 128
        lma_num_heads_stacking = 128
        lma_target_l_new = 30
        lma_d_new = 128
        lma_num_heads_latent = 32
        lma_ff_latent_hidden = 128*4 # Example: 4 * d_new
        lma_num_layers = 4
        lma_dropout = 0.1
        lma_bias = True

        # Create the dummy observation space LMA expects
        dummy_obs_space = DummyObservationSpace(shape=(self.input_dim_total,))

        print("Initializing ARCModel with LMAFeaturesExtractor...")
        self.feature_extractor = LMAFeaturesExtractor(
            observation_space=dummy_obs_space,
            embed_dim=lma_embed_dim,
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=lma_target_l_new,
            d_new=lma_d_new,
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=lma_seq_len,
            dropout=lma_dropout,
            bias=lma_bias
        )

        # --- Verify Output Dimension ---
        # The LMAFeaturesExtractor calculates its output 'features_dim' internally.
        # We need to ensure it matches our desired LATENT_DIM.
        calculated_lma_output_dim = self.feature_extractor.features_dim
        if calculated_lma_output_dim != latent_dim:
            raise ValueError(
                f"LMAFeaturesExtractor output dimension ({calculated_lma_output_dim}) "
                f"does not match the required latent_dim ({latent_dim}). "
                f"Check LMA config (L_new * d_new)."
            )
        print(f"  LMA Feature Extractor configured for output dimension: {calculated_lma_output_dim}")
        
        # --- Prediction Head ---
        self.prediction_head = ARCPredictionHead(
            latent_dim=latent_dim, # Should match LMA output
            hidden_dim=hidden_dim, 
            grid_height=self.grid_height, 
            grid_width=self.grid_width, 
            num_classes=num_classes
        )
        print("ARCModel Initialization Complete.")


    def forward(self, x):
        # Input x shape from DataLoader: (batch_size, H, W)
        batch_size = x.shape[0]
        
        # Flatten the H, W dimensions for LMAFeaturesExtractor
        # Input should be float, which ToTensor ensures
        x_flat = x.view(batch_size, -1) # Shape: (batch_size, H * W) -> (B, 900)
        
        # Pass flattened input to LMA feature extractor
        features = self.feature_extractor(x_flat) # Expected output: (batch_size, latent_dim) -> (B, 512)
        
        # Pass features to the prediction head
        logits = self.prediction_head(features) # Expected output: (batch_size, C, H, W)
        
        return logits

if __name__ == "__main__":
    # ----- Load Dataset -----
    transform = ToTensor()
    training_dir = os.path.join(DATA_PARENT_DIR, 'training')
    full_dataset = ARCDataset(data_dir=training_dir, split='training', transform=transform, max_grid_size=INPUT_SHAPE)

    if len(full_dataset) == 0:
        print("ERROR: No data loaded. Exiting.")
        exit() 

    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size

    if val_size == 0 and len(full_dataset) > 0:
        print("Warning: Dataset too small for validation split. Using all data for training.")
        train_dataset = full_dataset
        val_loader = None 
    else:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # ----- Initialize Model -----
    print(f"Using device: {DEVICE}")
    model = ARCModel(
        input_shape=INPUT_SHAPE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_COLORS
    ).to(DEVICE)

    # ----- Optimizer & Loss -----
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #criterion = nn.CrossEntropyLoss()
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5).to(DEVICE) 

    # ----- Helper Functions -----
    # (compute_exact_match_accuracy remains the same)
    def compute_exact_match_accuracy(logits, targets):
        with torch.no_grad():
            preds = logits.argmax(dim=1) 
            matches = (preds == targets)
            correct_per_sample = matches.view(matches.size(0), -1).all(dim=1)
            accuracy = correct_per_sample.float().mean().item()
        return accuracy