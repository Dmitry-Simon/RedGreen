import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from watermelon_eval.ECAPA_TDNN_Full import ECAPA_TDNN_Full

# ==== CONFIG ====
MODEL_PATH = "ecapa_best_model.pth"
CLASS_NAMES = ['low_sweet', 'sweet', 'un_sweet', 'very_sweet']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
def load_model():
    model = ECAPA_TDNN_Full(input_dim=64, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# ==== LOAD AND PREP SPECTROGRAM ====
def load_spectrogram(npy_path):
    spec = np.load(npy_path)  # shape: [mel_bins, time]
    spec = spec[np.newaxis, np.newaxis, :, :]  # shape: [1, 1, mel_bins, time]
    spec_tensor = torch.tensor(spec, dtype=torch.float32).to(DEVICE)
    return spec_tensor

# ==== PREDICT ====
def predict(model, spec_tensor):
    with torch.no_grad():
        output = model(spec_tensor)
        probs = F.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        return CLASS_NAMES[top_class.item()], top_prob.item()

# ==== MAIN ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True, help="Path to spectrogram .npy file")
    args = parser.parse_args()

    assert os.path.exists(args.npy), "❌ File not found!"

    model = load_model()
    spec_tensor = load_spectrogram(args.npy)
    label, confidence = predict(model, spec_tensor)

    print(f"\n🎯 Predicted class: **{label}** ({confidence*100:.2f}%)\n")
