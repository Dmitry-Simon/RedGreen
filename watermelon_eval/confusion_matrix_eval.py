import json
from watermelon_eval.ECAPA_TDNN_Full import ECAPA_TDNN_Full
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from settings import *


# === Create datasets ===
from watermelon_eval.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset
from sklearn.model_selection import train_test_split

DATA_JSON = r".\..\watermelon_dataset\ripeness_with_specs.json"
# === Load JSON and split ===
with open(DATA_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)

train_entries, test_entries = train_test_split(entries, test_size=0.2, random_state=775, stratify=[e['ripeness_label'] for e in entries])
# Create datasets
train_dataset = WatermelonSpectrogramDataset(train_entries)
test_dataset = WatermelonSpectrogramDataset(test_entries)

# === Create DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# Optional: print class labels
print("Classes:", train_dataset.get_label_encoder().classes_)

# === Create datasets ===


""" Extracting the classes names"""
# === Load JSON and split ===
with open(DATA_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)

dataset = WatermelonSpectrogramDataset(entries)
label_encoder = dataset.get_label_encoder()
class_names = label_encoder.classes_


# === Load model ===
model = ECAPA_TDNN_Full(input_dim=64, num_classes=4).to(DEVICE)
# If saved a model:
model.load_state_dict(torch.load("ecapa_best_model.pth"))

model.eval()

# === Generate predictions ===
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
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

