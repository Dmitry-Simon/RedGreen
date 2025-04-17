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

# ==== CONFIG ====
SAMPLE_RATE = 16000
N_FFT = 512
FRAME_SIZE = 0.025  # seconds
FRAME_STEP = 0.010  # seconds
N_MELS = 64
CLASS_NAMES = ['low_sweet', 'sweet', 'un_sweet', 'very_sweet']
MODEL_PATH = "ecapa_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== INIT FASTAPI ====
app = FastAPI()

# ==== MODEL SETUP ====
model = ECAPA_TDNN_Full(input_dim=N_MELS, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ==== MEL SPECTROGRAM PROCESS (IDENTICAL TO TRAINING) ====
def extract_mel_spectrogram(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Replicates the training code's mel spectrogram steps exactly:
      1) Pre-emphasis
      2) Framing
      3) Hamming window
      4) Power spectrum
      5) Mel filter bank
      6) Convert power to dB
    """
    # Step 1: Pre-emphasis
    alpha = 0.97
    emphasized = np.append(signal[0], signal[1:] - alpha * signal[:-1])

    # Step 2: Framing
    frame_length = int(FRAME_SIZE * sr)  # samples per frame
    frame_step = int(FRAME_STEP * sr)    # hop size in samples
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Step 3: Hamming Window
    hamming = np.hamming(frame_length)
    windowed_frames = frames * hamming

    # Step 4: Power spectrum
    mag_frames = np.abs(np.fft.rfft(windowed_frames, n=N_FFT))
    pow_frames = (1.0 / N_FFT) * (mag_frames ** 2)

    # Step 5: Mel filter bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    mel_spectrogram = np.dot(pow_frames, mel_basis.T)

    # Step 6: Convert power to dB
    # Training code used ref=np.max for dB scaling
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram.T, ref=np.max)

    return mel_spectrogram_db  # shape: (n_mels, time_frames)

# ==== API ENDPOINT ====
@app.post("/predict")
async def predict_ripeness(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary .wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            wav_path = tmp.name

        # Load audio
        signal, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        # Remove temp file to keep things clean
        os.remove(wav_path)

        # Extract Mel spectrogram using our refactored function
        mel_spec_db = extract_mel_spectrogram(signal, sr)

        # Prepare input for the model: [batch, channel, n_mels, time]
        # The model architecture may differ. Adjust if your model expects [batch, n_mels, time].
        input_tensor = torch.tensor(mel_spec_db[np.newaxis, np.newaxis, :, :],
                                    dtype=torch.float32).to(DEVICE)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            label = CLASS_NAMES[top_class.item()]
            confidence = top_prob.item()

        return JSONResponse({
            "predicted_label": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
