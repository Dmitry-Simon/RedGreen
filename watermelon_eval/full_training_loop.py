import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import time
import numpy as np

from watermelon_eval.SEBlock import ECAPA_TDNN_Lite
from watermelon_eval.traintest_split_dataloads import train_loader, val_loader
from watermelon_eval.traintest_split_dataloads import dataset

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Extract class weights based on full dataset label distribution
label_encoder = dataset.get_label_encoder()
numeric_labels = dataset.labels  # Already encoded

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(numeric_labels),
                                     y=numeric_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Model, loss, optimizer
model = ECAPA_TDNN_Lite(input_dim=64, num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
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
    val_acc, val_f1 = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f} | "
          f"Time: {time.time() - start_time:.1f}s")

    torch.save(model.state_dict(), "ecapa_best_model.pth")

