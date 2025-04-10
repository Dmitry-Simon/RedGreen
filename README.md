# 🍉 Watermelon Ripeness Classification - README Guide

This project uses audio signals from watermelon taps to classify ripeness levels using an ECAPA-TDNN model based on Mel spectrograms.

## 🗂️ Project Structure Overview

```
watermelon_dataset/
├── datasets/                        # Raw folders with .wav files (named with sugar level)
├── processed_spectrograms/
│   ├── ripeness_labels.json         # Original labeled dataset
│   ├── ripeness_with_specs.json     # Spectrogram paths + labels
│   └── balanced_ripeness.json       # Oversampled dataset
watermelon_eval/
├── move_in_dataset.py              # Scans raw dataset and creates ripeness_labels.json
├── preprocess_and_extract.py       # Converts audio to Mel spectrograms
├── oversample_ripeness_json.py     # Creates balanced training dataset
├── traintest_split_dataloads.py    # Loads dataset and builds DataLoaders
├── SEBlock.py                      # ECAPA-TDNN Lite model definition
├── full_training_loop.py           # Training script
├── confusion_matrix_eval.py        # Evaluation and confusion matrix
```

---

## ✅ Step-by-Step Usage

### 1. **Scan Dataset & Generate Labels**

Run this to extract sugar levels from folders and assign ripeness classes:
```bash
python move_in_dataset.py
```
- Output: `ripeness_labels.json`

---

### 2. **Convert Audio to Mel Spectrograms**

Generates spectrograms and saves them as `.npy`:
```bash
python preprocess_and_extract.py
```
- Output: `ripeness_with_specs.json`

---

### 3. **Balance the Dataset (Oversampling)**

To give equal weight to all ripeness classes:
```bash
python oversample_ripeness_json.py
```
- Output: `balanced_ripeness.json`

---

### 4. **Train/Test Split + Dataloaders**

Make sure `traintest_split_dataloads.py` is using the **balanced** JSON:
```python
json_path = r".../balanced_ripeness.json"
```
This script is imported in your training script.

---

### 5. **Train the Model**

Run this script to train ECAPA-TDNN Lite with class-weighted loss:
```bash
python full_training_loop.py
```
- Uses `CrossEntropyLoss(weight=...)` to balance learning
- Saves model to `ecapa_best_model.pth`

---

### 6. **Evaluate with Confusion Matrix**

Check model predictions on validation set:
```bash
python confusion_matrix_eval.py
```
- Make sure it also uses `balanced_ripeness.json`
- Shows matrix of predicted vs true labels

---

## 💬 Notes
- Make sure all `.py` files point to the correct dataset paths
- Class definitions: `unripe`, `mild`, `sweet`, `very_sweet`
- Model input: (1, 64, 300) Mel spectrogram tensors

---

## 🚀 Future Improvements
- Data augmentation for rare classes
- Fusion with image model
- Deployment as API endpoint (FastAPI / Flask)

---

Built with 🍉 and ML by the RedGreen team.

