import os
import base64
import io
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torchaudio

# FastAPI specific imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sonarretina_spectrogram import *
import uvicorn

# Ensure these are already defined in your notebook (from previous cells)
# SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH, N_FFT, MEL_HOP, N_MELS, NUM_CLASSES, DISTANCE_BINS, get_distance_class, SpectrogramCNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading (Global for efficiency) ---
# IMPORTANT: Update this path to where your model file is located on your local machine.
model_path = "../spectrogram_model_v2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Initialize the model with the correct number of channels
    # Make sure SpectrogramCNN is defined with n_channels=7 (4 for magnitude + 3 for IPD)
    inference_model = SpectrogramCNN(n_channels=7, num_classes=NUM_CLASSES).to(device)
    inference_model.load_state_dict(torch.load(model_path, map_location=device))
    inference_model.eval() # Set model to evaluation mode
    print("Model loaded successfully for inference.")
except Exception as e:
    print(f"Error loading model: {e}")
    inference_model = None # Indicate model loading failure

# --- Global Transformations (similar to SoundDistanceDataset) ---
spectrogram_transform = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=MEL_HOP,
    return_complex=True
)

mel_scale_transform = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
    n_stft=N_FFT // 2 + 1
)

db_transform = torchaudio.transforms.AmplitudeToDB()

# --- Preprocessing Function for API Input ---
def preprocess_audio_for_inference(audio_bytes, target_sr=SAMPLE_RATE):
    # Read audio from bytes
    audio, received_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    # Convert to mono if stereo/multi-channel (for simplicity, or handle multi-channel explicitly)
    if audio.ndim > 1:
        # Assuming the input audio is multi-channel, keep all channels for feature extraction
        # Our dataset processes each channel for magnitude and then IPD from first channel pairs
        pass # audio will be (samples, channels)

    # Ensure audio is 2D (samples, channels) for consistent processing
    if audio.ndim == 1:
        audio = audio[:, np.newaxis] # (samples, 1)

    # Resample if needed
    if received_sr != target_sr:
        audio_t = torch.tensor(audio.T, dtype=torch.float32)
        audio_t = torchaudio.functional.resample(audio_t, received_sr, target_sr)
        audio = audio_t.T.numpy()

    # Ensure the audio has at least FRAME_LENGTH samples
    if len(audio) < FRAME_LENGTH:
        # Pad with zeros if shorter
        padded_audio = np.zeros((FRAME_LENGTH, audio.shape[1]), dtype=np.float32)
        padded_audio[:len(audio), :] = audio
        audio = padded_audio
    elif len(audio) > FRAME_LENGTH:
        # Take the first frame if longer (simplified for single-frame inference)
        audio = audio[:FRAME_LENGTH, :]

    waveform = torch.tensor(audio.T, dtype=torch.float32) # Shape: (num_channels, samples)

    mel_magnitude_specs_augmented = []
    ipd_mel_specs = []

    # Check for sufficient channels for IPD calculation
    if waveform.shape[0] < 2 and inference_model.features[0].in_channels > 1: # Check if model expects more than 1 channel
        raise ValueError(
            f"Input audio has {waveform.shape[0]} channel(s), "
            f"but model requires multiple channels for IPD features (input_channels={inference_model.features[0].in_channels})."
        )

    complex_spectrograms = []

    # Process each channel for magnitude spectrograms
    for ch in waveform:
        spec_complex = spectrogram_transform(ch)
        complex_spectrograms.append(spec_complex)

        mel_mag = mel_scale_transform(spec_complex.abs())
        mel_mag = db_transform(mel_mag)
        # Normalize (using pre-defined means/stds for consistency, or batch-wise as done in dataset)
        mel_mag = (mel_mag - mel_mag.mean()) / (mel_mag.std() + 1e-6)
        mel_magnitude_specs_augmented.append(mel_mag)

    # Calculate IPD features (assuming mic0 as reference)
    phase_mic0 = torch.angle(complex_spectrograms[0])

    if waveform.shape[0] > 1:
        phase_mic1 = torch.angle(complex_spectrograms[1])
        ipd_01 = phase_mic0 - phase_mic1
        ipd_01 = torch.atan2(torch.sin(ipd_01), torch.cos(ipd_01))
        ipd_mel_specs.append(mel_scale_transform(ipd_01))

    if waveform.shape[0] > 2:
        phase_mic2 = torch.angle(complex_spectrograms[2])
        ipd_02 = phase_mic0 - phase_mic2
        ipd_02 = torch.atan2(torch.sin(ipd_02), torch.cos(ipd_02))
        ipd_mel_specs.append(mel_scale_transform(ipd_02))

    if waveform.shape[0] > 3:
        phase_mic3 = torch.angle(complex_spectrograms[3])
        ipd_03 = phase_mic0 - phase_mic3
        ipd_03 = torch.atan2(torch.sin(ipd_03), torch.cos(ipd_03))
        ipd_mel_specs.append(mel_scale_transform(ipd_03))

    # Normalize IPD Mel features
    for i in range(len(ipd_mel_specs)):
        ipd_mel_specs[i] = (ipd_mel_specs[i] - ipd_mel_specs[i].mean()) / (ipd_mel_specs[i].std() + 1e-6)

    # Concatenate magnitude and IPD features
    mel_specs = torch.stack(mel_magnitude_specs_augmented + ipd_mel_specs)

    # Ensure the final output matches the model's expected input channels (7)
    expected_channels = inference_model.features[0].in_channels
    if mel_specs.shape[0] != expected_channels:
        raise ValueError(
            f"Processed audio has {mel_specs.shape[0]} feature channels, "
            f"but the model expects {expected_channels}. "
            f"Please ensure input audio has enough channels to generate all required features."
        )

    return mel_specs

from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    filename: str
    predicted_distance: float
    predicted_class: int
    confidence_score: float
    distance_bins_description: Dict[int, str]


# --- FastAPI Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_distance(audio_file: UploadFile = File(...)):
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check Colab output for loading errors.")

    try:
        audio_bytes = await audio_file.read()
        processed_input = preprocess_audio_for_inference(audio_bytes).unsqueeze(0).to(device) # Add batch dimension

        with torch.no_grad():
            prediction = inference_model(processed_input)

        predicted_distance = prediction.item()
        predicted_class = get_distance_class(predicted_distance)

        # Simulate a confidence score (since pure regression models don't output probabilities natively).
        # In a real scenario, this could be derived from an ensemble variance or signal-to-noise ratio.

        def calculate_confidence(distance, predicted_class):
        
            low, high = DISTANCE_BINS[predicted_class]

            center = (low + high) / 2
            half_range = (high - low) / 2

            deviation = abs(distance - center)

            confidence = np.exp(
                -deviation / (half_range + 1e-6)
            )

            confidence = max(0.0, min(confidence, 1.0))

            return round(float(confidence), 2)

        confidence = calculate_confidence(
            predicted_distance,
            predicted_class
        )

        return {
            "filename": audio_file.filename,
            "predicted_distance": float(predicted_distance),
            "predicted_class": int(predicted_class),
            "confidence_score": confidence,
            "distance_bins_description": {k: f"{v[0]}m to {v[1]}m" for k, v in DISTANCE_BINS.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )