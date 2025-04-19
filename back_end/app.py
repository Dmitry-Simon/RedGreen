import os
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ECAPA_TDNN_Full import ECAPA_TDNN_Full
import librosa
from mel_utils import extract_mel_spectrogram, SR

# ==== CONFIG ====
SAMPLE_RATE = 16000
N_FFT = 512
FRAME_SIZE = 0.025  # seconds
FRAME_STEP = 0.010  # seconds
N_MELS = 64
CLASS_NAMES = ['low_sweet', 'sweet', 'un_sweet', 'very_sweet']
MODEL_PATH = "ecapa_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_WIDTH = 512

# ==== INIT FASTAPI ====
app = FastAPI()

# ==== MODEL SETUP ====
model = ECAPA_TDNN_Full(input_dim=N_MELS, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)


# ==== API ENDPOINT ====
@app.post("/predict")
async def predict_ripeness(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary .wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            wav_path = tmp.name

        # Load audio
        signal, sr = librosa.load(wav_path, sr=SR)

        # Remove temp file to keep things clean
        os.remove(wav_path)

        # Extract Mel spectrogram using our refactored function
        mel_spec_db = extract_mel_spectrogram(signal, sr)

        if mel_spec_db.shape[1] < FIXED_WIDTH:
            mel_spec_db = np.pad(mel_spec_db,
                                 ((0, 0), (0, FIXED_WIDTH - mel_spec_db.shape[1])),
                                 mode="constant")
        elif mel_spec_db.shape[1] > FIXED_WIDTH:
            mel_spec_db = mel_spec_db[:, :FIXED_WIDTH]

        # Prepare input for the model: [batch, channel, n_mels, time]
        # The model architecture may differ. Adjust if your model expects [batch, n_mels, time].
        # Prepare input: [batch, n_mels, time]
        input_tensor = torch.tensor(
            mel_spec_db[np.newaxis, :, :],
            dtype = torch.float32
        ).to(DEVICE)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            label = CLASS_NAMES[top_class.item()]
            confidence = top_prob.item()

        # print("API   input_tensor shape:", input_tensor.shape)

        return JSONResponse({
            "predicted_label": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
