import requests

wav_path = "./watermelon_dataset/datasets/1_10.5/audios/2.wav"
resp = requests.post(
    "http://127.0.0.1:8000/predict",
    files={"file": open(wav_path, "rb")}
)
print(resp.json())
