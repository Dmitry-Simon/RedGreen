import os
import json
import torch
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

from watermelon_eval.model.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset
from watermelon_eval.model.ECAPA_TDNN_Full import ECAPA_TDNN_Full
from sklearn.model_selection import train_test_split
from watermelon_eval.misc.settings import BATCH_SIZE, DEVICE

def add_model_info_text(fig, gs):
    model_info = (
        "Model Performance & Training Details\n"
        "------------------------------------\n"
        "Architecture: ECAPA-TDNN\n"
        "Input: Mel spectrograms (64 freq bins)\n"
        "Training Settings:\n"
        "- Batch Size: 8\n"
        "- Epochs: 60\n"
        "- Learning Rate: 1e-3\n"
        "- Optimizer: Adam\n"
        "- LR Scheduler: CosineAnnealingLR\n"
        "- Train/Test Split: 80/20\n"
        "Validation Accuracy: 97.14%\n"
        "F1-Score: 97.44%"
    )

    # Add text box on the right side of the figure
    ax_text = fig.add_subplot(gs[2])
    ax_text.text(0.05, 0.5, model_info, fontsize=9, ha='left', va='center',
                 transform=ax_text.transAxes, bbox=dict(facecolor='lightblue', alpha=0.8, pad=10))
    ax_text.axis('off')  # Hide axes for text panel

# ——— Configuration ———
DATA_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "watermelon_dataset", "ripeness_with_specs.json")
BEST_MODEL_REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "back_end", "ecapa_best_model.pth")

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
model_path = os.path.abspath(BEST_MODEL_REL)
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

# Calculate accuracy and F1 score
accuracy = (cm_counts.diagonal().sum() / cm_counts.sum()) * 100
f1 = f1_score(all_labels, all_preds, average='weighted') * 100

# Row-wise normalize with epsilon to avoid div-by-zero
row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float) + 1e-12
cm_norm  = cm_counts.astype(float) / row_sums

# ——— Plot confusion matrices and statistics ———
fig = plt.figure(figsize=(18, 7))  # Reduced overall height

# Create a more balanced grid with a smaller bottom row
gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[5, 0.7],  # Changed ratio to make bottom row smaller
                  hspace=0.15, wspace=0.3)  # Reduced vertical spacing

# Plot raw counts
ax1 = plt.subplot(gs[0, 0])
sns.heatmap(
    cm_counts,
    annot=True, fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax1,
    square=True
)
ax1.set_title("Confusion Matrix (counts)", fontsize=12, pad=10)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")

# Plot normalized values
ax2 = plt.subplot(gs[0, 1])
sns.heatmap(
    cm_norm,
    annot=True, fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax2,
    square=True
)
ax2.set_title("Confusion Matrix (normalized)", fontsize=12, pad=10)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")

# Add model info text in a more balanced way
ax3 = plt.subplot(gs[0, 2])
model_info = (
    "Model Performance & Training Details\n\n"
    "Architecture: ECAPA-TDNN\n"
    "Input: Mel spectrograms (64 freq bins)\n"
    "Output: 4 ripeness classes\n\n"
    "Training Settings:\n"
    "• Batch Size: 8\n"
    "• Epochs: 60\n"
    "• Learning Rate: 1e-3\n"
    "• Optimizer: Adam\n"
    "• LR Scheduler: CosineAnnealingLR\n"
    "• Train/Test Split: 80/20\n"
    "• Random Seed: 775\n\n"
    f"Current Test Results:\n"
    f"• Test Accuracy: {accuracy:.2f}%\n"
    f"• Test F1-Score: {f1:.2f}%"
)

# Make text larger and more prominent with clean styling
ax3.text(0.05, 0.95, model_info, fontsize=13, ha='left', va='top',
         transform=ax3.transAxes,
         bbox=dict(facecolor='lightblue', alpha=0.4, pad=20, boxstyle="round,pad=0.8"))
ax3.set_title("Model Performance", fontsize=14, fontweight='bold', pad=10)
ax3.axis('off')

# Make the lower text row bigger and more compact

# Add dataset statistics in bottom row - more compact format
ax4 = plt.subplot(gs[1, :])
dataset_info = (
    f"Dataset: {len(entries)} samples ({len(train_entries)} train, {len(test_entries)} test) | "
    f"Distribution: {dict(test_counts)}"
)
ax4.text(0.5, 0.5, dataset_info, fontsize=13, ha='center', va='center',
         transform=ax4.transAxes,
         bbox=dict(facecolor='lightblue', alpha=0.4, pad=2, boxstyle="round,pad=0.3"))
ax4.axis('off')

# Update main title and position it closer to the plots
plt.suptitle(f'Watermelon Ripeness Classification Results\nTest Accuracy: {accuracy:.2f}% | Test F1-Score: {f1:.2f}%',
             fontsize=14, y=0.97)

# Adjust spacing to reduce the gap at the bottom
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.08)

plt.show()
