#!/usr/bin/env python3
"""
Evaluate your FastAPI `/predict` endpoint on the full dataset and plot a confusion matrix.
"""
import os
import json
import requests
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ——— Configuration ———
API_URL     = "http://127.0.0.1:8000/predict"
DATA_JSON   = r".\..\watermelon_dataset\ripeness_with_specs.json"
CLASS_NAMES = ['low_sweet', 'sweet', 'un_sweet', 'very_sweet']

# ——— Load metadata ———
with open(DATA_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)

true_indices = []
pred_indices = []

# ——— Iterate and call API ———
for entry in entries:
    wav_path   = os.path.normpath(entry["audio_path"])
    true_label = entry["ripeness_label"]

    # Ensure file exists
    if not os.path.exists(wav_path):
        print(f"Missing WAV: {wav_path}")
        continue

    # Call API
    with open(wav_path, "rb") as wav_file:
        response = requests.post(API_URL, files={"file": wav_file})
    try:
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        print(f"API error for {wav_path}: {e}")
        continue

    pred_label = result.get("predicted_label")
    if pred_label not in CLASS_NAMES:
        print(f"Unknown prediction '{pred_label}' for {wav_path}")
        continue

    true_indices.append(CLASS_NAMES.index(true_label))
    pred_indices.append(CLASS_NAMES.index(pred_label))

# ——— Build confusion matrix ———
cm = confusion_matrix(true_indices, pred_indices,
                      labels=list(range(len(CLASS_NAMES))))

# ——— Plot ———
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - API Predictions')
plt.tight_layout()
plt.show()
