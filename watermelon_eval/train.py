import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import time
import numpy as np
import json
from torch.utils.data import DataLoader, random_split
from watermelon_eval.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset
from watermelon_eval.ECAPA_TDNN_Full import ECAPA_TDNN_Full
from sklearn.model_selection import train_test_split
from settings import *


# --------------------> Data
# === Load separate training and validation sets ===

# === Create datasets ===
DATA_JSON = r".\..\watermelon_dataset\ripeness_with_specs.json"
# === Load JSON and split ===
with open(DATA_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)

train_entries, test_entries = train_test_split(entries, test_size=0.2, random_state=42, stratify=[e['ripeness_label'] for e in entries])
# Create datasets
train_dataset = WatermelonSpectrogramDataset(train_entries)
test_dataset = WatermelonSpectrogramDataset(test_entries)

# === Create DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# Optional: print class labels
print("Classes:", train_dataset.get_label_encoder().classes_)
# --------------------> Data

# Extract class weights based on full dataset label distribution
# label_encoder = dataset.get_label_encoder()
# numeric_labels = dataset.labels  # Already encoded


# --------------------> Train
# Model, loss, optimizer
model = ECAPA_TDNN_Full(input_dim=64, num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

# Training loop
best_val_res = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    val_acc, val_f1 = evaluate(model, test_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "  # todo: create val dataset instead of using the test set for validation. see train/test split above.
          f"Val F1: {val_f1:.4f} | "
          f"Time: {time.time() - start_time:.1f}s")

    if val_acc > best_val_res:
        best_val_res = val_acc
        torch.save(model.state_dict(), "ecapa_best_model.pth")
# --------------------> Train

