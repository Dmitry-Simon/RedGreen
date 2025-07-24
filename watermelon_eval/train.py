# train.py
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import time
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model.WatermelonSpectrogramDataset import WatermelonSpectrogramDataset
from model.ECAPA_TDNN_Full import ECAPA_TDNN_Full
from watermelon_eval.misc.settings import *
from watermelon_eval.misc.file_loader_best_model import *


def get_device_info():
    """Detect available GPU acceleration for all platforms."""
    if torch.backends.mps.is_available():
        return "MPS (Apple Silicon GPU)", "Apple M4 Max GPU"
    elif torch.cuda.is_available():
        return "CUDA", torch.cuda.get_device_name(0)
    else:
        return "CPU", "No GPU acceleration"


def load_best_scores():
    """Load best scores from previous runs if they exist."""
    if os.path.exists(BEST_SCORE_FILE):
        with open(BEST_SCORE_FILE, "r") as f:
            val, f1 = f.read().strip().split(",")
            return float(val), float(f1)
    else:
        return 0.0, 0.0


def evaluate(model, loader):
    """Evaluate model on given data loader."""
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


def main():
    # Display device information
    gpu_available, device_name = get_device_info()
    print(f"GPU acceleration: {gpu_available}")
    print(f"Device: {device_name}")

    # Load best scores from previous runs
    all_time_best_val_acc, all_time_best_f1 = load_best_scores()

    # Load and split data
    DATA_JSON = "../watermelon_dataset/ripeness_with_specs.json"
    with open(DATA_JSON, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    train_entries, test_entries = train_test_split(
        entries,
        test_size=0.2,
        random_state=775,
        stratify=[e['ripeness_label'] for e in entries]
    )

    # Create datasets and data loaders
    train_dataset = WatermelonSpectrogramDataset(train_entries)
    test_dataset = WatermelonSpectrogramDataset(test_entries)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print("Classes:", train_dataset.get_label_encoder().classes_)

    # Initialize model, loss, optimizer, and scheduler
    model = ECAPA_TDNN_Full(input_dim=64, num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

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
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Time: {time.time() - start_time:.1f}s")

        # Calculate weighted score
        current_score = 0.4 * val_acc + 0.6 * val_f1
        best_score = 0.4 * all_time_best_val_acc + 0.6 * all_time_best_f1

        # Check if best of this run
        if val_acc > best_val_res:
            best_val_res = val_acc

            # Check if also best of all time
            if current_score > best_score:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                all_time_best_val_acc = val_acc
                all_time_best_f1 = val_f1
                with open(BEST_SCORE_FILE, "w") as f:
                    f.write(f"{val_acc:.6f},{val_f1:.6f}")
                print(f"🥇 New all-time best model! Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            else:
                print(f"✅ Best of this run. But not better than all-time best "
                      f"(Val Acc: {all_time_best_val_acc:.4f}, F1: {all_time_best_f1:.4f})")

        scheduler.step()

    print(f"\n✅ Training complete. Best of this run: Val Acc: {best_val_res:.4f}")
    print(f"🏆 Best model ever: Val Acc: {all_time_best_val_acc:.4f}, F1: {all_time_best_f1:.4f}")


if __name__ == "__main__":
    main()
