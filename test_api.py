import requests
import os

local_url = "http://127.0.0.1:8000"
audio_file_path = "dcase/zenodo_upload/mic_data/fold1_room1_mix001.wav" # multi-channel WAV file on your machine

# It's crucial that the WAV file has at least 4 channels to generate all 7 features for the model.
# If you only have mono audio, you would need to duplicate channels before sending or modify preprocess_audio_for_inference.

with open(audio_file_path, "rb") as f:
    files = {'audio_file': (os.path.basename(audio_file_path), f, 'audio/wav')}
    response = requests.post(f"{local_url}/predict", files=files)

print(response.json())