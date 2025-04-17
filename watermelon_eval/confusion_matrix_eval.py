#!/usr/bin/env python3
"""
Compute and plot a confusion matrix for your local validation set,
using the same ECAPA‑TDNN model you deployed in `back_end`.
"""
import os
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from watermelon_eval.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset
from watermelon_eval.ECAPA_TDNN_Full import ECAPA_TDNN_Full
from sklearn.model_selection import train_test_split
from settings import BATCH_SIZE, DEVICE

# ——— Configuration ———
DATA_JSON      = r"..\watermelon_dataset\ripeness_with_specs.json"
BEST_MODEL_REL = r"..\back_end\ecapa_best_model.pth"

# ——— Load & split entries ———
with open(DATA_JSON, "r", encoding="utf-8") as f:
    entries = json.load(f)

train_entries, test_entries = train_test_split(
    entries,
    test_size=0.2,
    random_state=775,
    stratify=[e["ripeness_label"] for e in entries]
)

print(f"Total entries: {len(entries)}, "
      f"Train: {len(train_entries)}, Test: {len(test_entries)}")

# ——— Inspect test-label distribution ———
from collections import Counter
test_counts = Counter(e["ripeness_label"] for e in test_entries)
print("Test set label counts:", test_counts)

# ——— Build test DataLoader ———
test_dataset = WatermelonSpectrogramDataset(test_entries)
test_loader  = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

class_names = test_dataset.get_label_encoder().classes_
print("Class names:", list(class_names))

# ——— Load model weights ———
model = ECAPA_TDNN_Full(input_dim=64, num_classes=len(class_names)).to(DEVICE)

# Use an absolute path to be sure
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), BEST_MODEL_REL))
print("Loading model from:", model_path)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

state = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ——— Run inference on validation set ———
all_preds  = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        preds   = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# ——— Compute confusion matrix ———
labels_idx = list(range(len(class_names)))
cm_counts  = confusion_matrix(all_labels, all_preds, labels=labels_idx)

# Row-wise normalize with epsilon to avoid div-by-zero
row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float) + 1e-12
cm_norm  = cm_counts.astype(float) / row_sums

# ——— Plot raw counts & normalized percentages ———
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    cm_counts,
    annot=True, fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix (counts)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(
    cm_norm,
    annot=True, fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix (row-normalized)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()
