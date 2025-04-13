import json
import torch
from torch.utils.data import DataLoader, random_split
from watermelon_eval.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset

# === Load separate training and validation sets ===
train_json = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\balanced_ripeness.json"
val_json = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\ripeness_with_specs.json"

with open(train_json, 'r', encoding='utf-8') as f:
    train_entries = json.load(f)

with open(val_json, 'r', encoding='utf-8') as f:
    val_entries = json.load(f)

# === Create datasets ===
DATA_JSON = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\balanced_ripeness.json"
# === Load JSON and split ===
with open(DATA_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)
dataset = WatermelonSpectrogramDataset(entries)
train_dataset = WatermelonSpectrogramDataset(train_entries)
val_dataset = WatermelonSpectrogramDataset(val_entries)

# === Create DataLoaders ===
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
# Optional: print class labels
print("Classes:", train_dataset.get_label_encoder().classes_)

# Expose dataset if needed
train_set = train_dataset
val_set = val_dataset
dataset = train_dataset
