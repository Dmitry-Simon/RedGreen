import json
import torch
from torch.utils.data import DataLoader, random_split

from watermelon_eval.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset

# Step 1: Load the JSON metadata
json_path = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\balanced_ripeness.json"
with open(json_path, 'r', encoding='utf-8') as f:
    entries = json.load(f)

# Step 2: Initialize the dataset
dataset = WatermelonSpectrogramDataset(entries)

# Step 3: Split the dataset (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Step 4: Create dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Optional: Check label classes
print("Classes:", dataset.get_label_encoder().classes_)
