import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Config ===
DATA_JSON = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\balanced_ripeness.json"
FIXED_WIDTH = 300
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset Class ===
from sklearn.preprocessing import LabelEncoder

class WatermelonSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, entries):
        self.entries = entries
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform([e['ripeness_label'] for e in entries])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        spec = np.load(entry['spectrogram_path'])

        if spec.shape[1] < FIXED_WIDTH:
            pad = FIXED_WIDTH - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')
        elif spec.shape[1] > FIXED_WIDTH:
            spec = spec[:, :FIXED_WIDTH]

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label

    def get_label_encoder(self):
        return self.label_encoder

# === Model Definition ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ECAPA_TDNN_Lite(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.res2block = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
        )
        self.conv2 = nn.Conv1d(128, 192, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = self.relu(self.conv1(x))
        x = self.res2block(x) + x
        x = self.conv2(x)
        x = self.pooling(x).squeeze(-1)
        return self.classifier(x)

# === Load JSON and split ===
with open(DATA_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)

dataset = WatermelonSpectrogramDataset(entries)
label_encoder = dataset.get_label_encoder()
class_names = label_encoder.classes_

_, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# === Load model ===
model = ECAPA_TDNN_Lite(input_dim=64, num_classes=4).to(DEVICE)
# If you saved a model:
# model.load_state_dict(torch.load("ecapa_best_model.pth"))

model.eval()

# === Generate predictions ===
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# === Plot confusion matrix ===
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.show()

from collections import Counter
labels = [e['ripeness_label'] for e in entries]
print(Counter(labels))

