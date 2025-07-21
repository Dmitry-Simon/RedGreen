# 🍉 Redy: Watermelon Ripeness Classification

Classify watermelon ripeness from tap audio using deep learning (ECAPA-TDNN) and Mel spectrograms.
This is the backend part of "Project Phase B" of the final project.

Link for the backend:
[🔗 Backend (FastAPI, ECAPA-TDNN) code](https://github.com/Dmitry-Simon/RedyApp/tree/master)

---

## 🏗️ Project Structure

```
watermelon_dataset/
  ├── datasets/                     # Raw audio files organized by ripeness
      ├── low_sweet/                # Low sweet ripeness audio
      ├── sweet/                    # Sweet ripeness audio
      ├── un_sweet/                 # Unsweet ripeness audio
      └── very_sweet/               # Very sweet ripeness audio
  ├── processed_spectrograms/       # Mel spectrograms of audio files 
      ├── ripeness_labels.json
      ├── ripeness_with_specs.json
      └── balanced_ripeness.json
back_end/
  ├── app.py                        # FastAPI API for inference
  ├── ECAPA_TDNN_Full.py            # Model architecture
  ├── mel_utils.py                  # Mel spectrogram extraction
  ├── generate_spectrogram.py       # Spectrogram generation utility
  ├── evaluate_api_confusion.py     # API evaluation script
  └── ecapa_best_model.pth          # Saved model weights
watermelon_eval/
  ├── model/
      └── ECAPA_TDNN_Full.py        # Model definition (training/eval)
  ├── confusion_matrix_eval.py      # Offline confusion matrix
  ├── move_in_dataset.py            # Dataset organization
  ├── preprocess_and_extract.py     # Audio preprocessing
  ├── oversample_ripeness_json.py   # Dataset balancing
  ├── full_training_loop.py         # Model training script
  └── misc/
      └── file_loader_best_model.py # Model score loader
      └── best_score.txt            # Best model scores
visualizations/
  └── count.py                      # Dataset statistics
```

---

## 🚀 Quick Start

### 1. Prepare Dataset & Labels

- Scan dataset and assign ripeness classes:
  ```bash
  python watermelon_eval/move_in_dataset.py
  ```
- Convert audio to Mel spectrograms:
  ```bash
  python watermelon_eval/preprocess_and_extract.py
  ```
- Balance dataset (oversampling):
  ```bash
  python watermelon_eval/oversample_ripeness_json.py
  ```

### 2. Train the Model

- Train ECAPA-TDNN Lite:
  ```bash
  python watermelon_eval/full_training_loop.py
  ```
- Model weights saved to `back_end/ecapa_best_model.pth`

### 3. Run Inference API

- Start FastAPI server:
  ```bash
  uvicorn back_end.app:app --reload --host 0.0.0.0 --port 8000
  ```
- Predict ripeness:
  ```python
  import requests
  with open("tap.wav", "rb") as f:
      r = requests.post("http://localhost:8000/predict", files={"file": f})
      print(r.json())
  ```

### 4. Evaluate Model

- Offline confusion matrix:
  ```bash
  python watermelon_eval/confusion_matrix_eval.py
  ```
- API confusion matrix:
  ```bash
  python back_end/evaluate_api_confusion.py
  ```

---

## 🧠 Model & Features

- **Architecture:** ECAPA-TDNN with SE blocks, Res2Net, Attentive Statistics Pooling
- **Input:** Mel spectrogram (shape: [1, 64, 512])
- **Classes:** `low_sweet`, `sweet`, `un_sweet`, `very_sweet`
- **Audio Preprocessing:** 16kHz, 64 mel bands, 50Hz–8kHz, pre-emphasis α=0.97

---

## 🛠️ Utilities

- Generate spectrogram for a WAV:
  ```bash
  python back_end/generate_spectrogram.py --wav input.wav --out ./specs
  ```

- Visualize dataset audio counts:
  ```bash
  python visualizations/count.py
  ```

---

## 📄 API Response Example

```json
{
  "predicted_label": "sweet",
  "confidence": 0.87
}
```

---

## 📈 Performance Tracking

- Best model weights: `back_end/ecapa_best_model.pth`
- Best scores: `best_score.txt`

---

Built with 🍉 by the Redy team.
