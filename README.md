# 🍉 Redy: Watermelon Sweetness Classification Backend

Deep learning backend for classifying watermelon ripeness from tap audio using ECAPA-TDNN and Mel spectrograms.

**📱 [Download Android APK](https://drive.google.com/file/d/18OC_MR-IxNMIf2QQ-rpTmzTqaw61SDS5/view?usp=sharing)**  
**📱 [Frontend Repository](https://github.com/Dmitry-Simon/RedyApp/tree/master)**

---

## 🏗️ Project Structure

```

back_end/
  ├── app.py                         # FastAPI inference server
  ├── ECAPA_TDNN_Full.py             # Model architecture
  ├── mel_utils.py                   # Spectrogram processing utilities
  ├── ecapa_best_model.pth           # Trained model weights
  ├── evaluate_api_confusion.py      # API evaluation script
  ├── generate_spectrogram.py        # Spectrogram generation utility
  ├── predict_from_wav.py            # Single WAV file prediction
  ├── predict.py                     # Core prediction logic
  └── test.py                        # Testing utilities
  
watermelon_dataset/
  ├── datasets/                      # Raw audio and image data
  │   ├── ripeness_labels.json       # Raw labels
  │   └── [1-19]_*/                  # Watermelon samples with audio/picture/chu
  ├── processed_spectrograms/        # Generated Mel spectrograms by class
  │   ├── low_sweet/
  │   ├── sweet/
  │   ├── un_sweet/
  │   └── very_sweet/
  ├── ripeness_labels.json           # Main labels file
  └── ripeness_with_specs.json       # Processed dataset with spectrograms
  
watermelon_eval/
  ├── model/                         # Training modules
  │   ├── ECAPA_TDNN_Full.py         # Model definition
  │   └── WatermelonSpectrogramDataset.py  # Dataset class
  ├── misc/                          # Configuration
  │   ├── settings.py                # Training settings
  │   └── file_loader_best_model.py  # Model loading utilities
  ├── train.py                       # Model training script
  ├── confusion_matrix_eval.py       # Model evaluation
  ├── pre_processing.py              # Data preprocessing
  └── processing_dataset.py          # Dataset processing utilities

visualizations/
  ├── count.py                       # Dataset statistics
  └── visualize_local_mel.py         # Spectrogram visualization
```

---

## 🚀 Quick Start

### 1. Setup Dataset
```bash
# Process raw audio to spectrograms
python watermelon_eval/preprocess_and_extract.py

# Balance dataset (optional)
python watermelon_eval/oversample_ripeness_json.py
```

### 2. Train Model
```bash
python watermelon_eval/train.py
```

### 3. Run API Server
```bash
uvicorn back_end.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test Prediction
```python
import requests

with open("watermelon_tap.wav", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})
    print(response.json())
```

---

## 🧠 Model Details

- **Architecture:** ECAPA-TDNN with attention pooling and residual connections
- **Input:** Mel spectrogram (64 bands, 512 frames, 16kHz audio)
- **Classes:** `low_sweet`, `sweet`, `un_sweet`, `very_sweet`
- **Features:** SE blocks, Res2Net, attentive statistics pooling

---

## 📊 Evaluation

```bash
# Generate confusion matrix
python watermelon_eval/confusion_matrix_eval.py

# Test API performance
python back_end/evaluate_api_confusion.py
```

---

## 🛠️ Utilities

Generate spectrogram from WAV file:
```bash
python back_end/generate_spectrogram.py --wav input.wav --out ./output
```

View dataset statistics:
```bash
python visualizations/count.py
```

---

## 📋 API Response

```json
{
  "predicted_label": "sweet",
  "confidence": 0.87
}
```

---

Built with 🍉 by the Redy team.
