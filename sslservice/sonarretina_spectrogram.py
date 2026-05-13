import os
import glob
import csv
import random

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import confusion_matrix
from collections import Counter


print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


SAMPLE_RATE = 16000

FRAME_LENGTH = 2048
HOP_LENGTH = 512

N_FFT = 1024
MEL_HOP = 256
N_MELS = 128

BATCH_SIZE = 32
EPOCHS = 100

NUM_CLASSES = 3


DISTANCE_BINS = {
    0: (0, 2.5),
    1: (2.5, 5.0),
    2: (5.0, 10.0)
}


def get_distance_class(distance):

    for cls, (low, high) in DISTANCE_BINS.items():

        if low <= distance < high:
            return cls

    return 2


class SoundDistanceDataset(Dataset):

    def __init__(self, csv_files, mic_dir, augment=False):

        self.data = []
        self.augment = augment

        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT,
            hop_length=MEL_HOP,
            return_complex=True
        )

        self.mel_scale_transform = torchaudio.transforms.MelScale(
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            n_stft=N_FFT // 2 + 1
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=12
        )

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=12
        )

        print(f"Building dataset from {len(csv_files)} CSV files")

        for csv_file in csv_files:

            base = os.path.basename(csv_file).replace(".csv", ".wav")

            wav_file = os.path.join(mic_dir, base)

            if not os.path.exists(wav_file):
                continue

            audio, sr = sf.read(wav_file)

            if audio.ndim == 1:
                audio = audio[:, np.newaxis]

            if sr != SAMPLE_RATE:

                audio_t = torch.tensor(
                    audio.T,
                    dtype=torch.float32
                )

                audio_t = torchaudio.functional.resample(
                    audio_t,
                    sr,
                    SAMPLE_RATE
                )

                audio = audio_t.T.numpy()

            labels = {}

            with open(csv_file) as f:

                reader = csv.reader(f)

                for row in reader:

                    frame_idx = int(row[0])

                    distance = float(row[5])

                    labels[frame_idx] = distance

            matched = 0

            for start in range(
                0,
                len(audio) - FRAME_LENGTH,
                HOP_LENGTH
            ):

                frame_idx = start // HOP_LENGTH

                if frame_idx in labels:

                    frame_audio = audio[
                        start:start + FRAME_LENGTH
                    ]

                    frame_audio = frame_audio.T.astype(np.float32)

                    self.data.append(
                        (
                            frame_audio,
                            labels[frame_idx]
                        )
                    )

                    matched += 1

            print(
                f"{os.path.basename(csv_file)} "
                f"-> {matched} labeled frames"
            )

        print(f"Total examples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        waveform, distance = self.data[idx]

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        mel_magnitude_specs_augmented = []
        ipd_mel_specs = []

        if waveform.shape[0] < 2:
            raise ValueError("IPD requires at least 2 channels")

        complex_spectrograms = []

        for ch in waveform:

            spec_complex = self.spectrogram_transform(ch)

            complex_spectrograms.append(spec_complex)

            mel_mag = self.mel_scale_transform(
                spec_complex.abs()
            )

            mel_mag = self.db_transform(mel_mag)

            mel_mag = (
                mel_mag - mel_mag.mean()
            ) / (
                mel_mag.std() + 1e-6
            )

            if self.augment:

                if torch.rand(1).item() < 0.5:
                    mel_mag = self.freq_mask(mel_mag)

                if torch.rand(1).item() < 0.5:
                    mel_mag = self.time_mask(mel_mag)

            mel_magnitude_specs_augmented.append(mel_mag)

        phase_mic0 = torch.angle(complex_spectrograms[0])

        if waveform.shape[0] > 1:

            phase_mic1 = torch.angle(complex_spectrograms[1])

            ipd_01 = phase_mic0 - phase_mic1

            ipd_01 = torch.atan2(
                torch.sin(ipd_01),
                torch.cos(ipd_01)
            )

            ipd_mel_specs.append(
                self.mel_scale_transform(ipd_01)
            )

        if waveform.shape[0] > 2:

            phase_mic2 = torch.angle(complex_spectrograms[2])

            ipd_02 = phase_mic0 - phase_mic2

            ipd_02 = torch.atan2(
                torch.sin(ipd_02),
                torch.cos(ipd_02)
            )

            ipd_mel_specs.append(
                self.mel_scale_transform(ipd_02)
            )

        if waveform.shape[0] > 3:

            phase_mic3 = torch.angle(complex_spectrograms[3])

            ipd_03 = phase_mic0 - phase_mic3

            ipd_03 = torch.atan2(
                torch.sin(ipd_03),
                torch.cos(ipd_03)
            )

            ipd_mel_specs.append(
                self.mel_scale_transform(ipd_03)
            )

        for i in range(len(ipd_mel_specs)):

            ipd_mel_specs[i] = (
                ipd_mel_specs[i] -
                ipd_mel_specs[i].mean()
            ) / (
                ipd_mel_specs[i].std() + 1e-6
            )

        mel_specs = torch.stack(
            mel_magnitude_specs_augmented +
            ipd_mel_specs
        )

        return mel_specs, torch.tensor(
            distance,
            dtype=torch.float32
        )


class SpectrogramCNN(nn.Module):

    def __init__(self, n_channels=7, num_classes=3):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x



def calculate_class_weights(dataset):

    class_counts = Counter(
        [get_distance_class(item[1]) for item in dataset.data]
    )

    total_samples = sum(class_counts.values())

    num_classes = len(DISTANCE_BINS)

    weights = []

    for cls in sorted(DISTANCE_BINS.keys()):

        count = class_counts.get(cls, 0)

        if count > 0:
            weights.append(
                total_samples / (count * num_classes)
            )
        else:
            weights.append(1.0)

    print(f"Calculated class weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)



def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    device,
    patience=10
):

    class_weights = calculate_class_weights(
        train_dataset
    ).to(device)

    criterion = nn.MSELoss(reduction='none')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    best_val_loss = float('inf')

    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()

        current_train_loss = 0

        correct = 0

        total = 0

        for x, y in train_loader:

            x = x.to(device)

            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            out = model(x)

            true_classes_batch = torch.tensor(
                [
                    get_distance_class(t.item())
                    for t in y.squeeze()
                ],
                device=device
            )

            sample_weights_batch = class_weights[
                true_classes_batch
            ].unsqueeze(1)

            loss = criterion(out, y)

            weighted_loss = (
                loss * sample_weights_batch
            ).mean()

            weighted_loss.backward()

            optimizer.step()

            current_train_loss += weighted_loss.item()

            pred_classes = torch.tensor(
                [
                    get_distance_class(p.item())
                    for p in out.squeeze()
                ],
                device=device
            )

            true_classes = torch.tensor(
                [
                    get_distance_class(t.item())
                    for t in y.squeeze()
                ],
                device=device
            )

            total += y.size(0)

            correct += (
                pred_classes == true_classes
            ).sum().item()

        epoch_train_acc = 100 * correct / total

        model.eval()

        current_val_loss = 0

        correct = 0

        total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)

                y = y.to(device).unsqueeze(1)

                out = model(x)

                loss = nn.MSELoss()(out, y)

                current_val_loss += loss.item()

                pred_classes = torch.tensor(
                    [
                        get_distance_class(p.item())
                        for p in out.squeeze()
                    ],
                    device=device
                )

                true_classes = torch.tensor(
                    [
                        get_distance_class(t.item())
                        for t in y.squeeze()
                    ],
                    device=device
                )

                total += y.size(0)

                correct += (
                    pred_classes == true_classes
                ).sum().item()

        epoch_val_loss = (
            current_val_loss / len(val_loader)
        )

        epoch_val_acc = 100 * correct / total

        scheduler.step(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc {epoch_train_acc:.2f}% | "
            f"Val Acc {epoch_val_acc:.2f}% | "
            f"Val Loss {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                "best_spectrogram_model.pth"
            )

            print("Saved best model")

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print("Early stopping triggered")

                break


def evaluate(model, loader, device):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device).unsqueeze(1)

            out = model(x)

            pred_classes = torch.tensor(
                [
                    get_distance_class(p.item())
                    for p in out.squeeze()
                ],
                device=device
            )

            true_classes = torch.tensor(
                [
                    get_distance_class(t.item())
                    for t in y.squeeze()
                ],
                device=device
            )

            total += y.size(0)

            correct += (
                pred_classes == true_classes
            ).sum().item()

    return 100 * correct / total


def plot_confusion_matrix(
    model,
    loader,
    device,
    save_path="confusion_matrix.png"
):

    model.eval()

    all_true_classes = []

    all_pred_classes = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device).unsqueeze(1)

            out = model(x)

            true_classes = [
                get_distance_class(t.item())
                for t in y.squeeze()
            ]

            pred_classes = [
                get_distance_class(p.item())
                for p in out.squeeze()
            ]

            all_true_classes.extend(true_classes)

            all_pred_classes.extend(pred_classes)

    cm = confusion_matrix(
        all_true_classes,
        all_pred_classes,
        labels=[0, 1, 2]
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[
            f'Class {c}'
            for c in DISTANCE_BINS.keys()
        ],
        yticklabels=[
            f'Class {c}'
            for c in DISTANCE_BINS.keys()
        ]
    )

    plt.xlabel('Predicted Label')

    plt.ylabel('True Label')

    plt.title('Confusion Matrix')

    plt.savefig(save_path)

    plt.show()

    print(f"Saved to {save_path}")



def get_sample_weights(dataset):

    class_counts = Counter(
        [
            get_distance_class(item[1])
            for item in dataset.data
        ]
    )

    total_samples = len(dataset)

    num_classes = len(DISTANCE_BINS)

    weights_per_class = {

        cls: total_samples /
        (class_counts[cls] * num_classes)

        for cls in DISTANCE_BINS.keys()
    }

    sample_weights = []

    for _, distance in dataset.data:

        cls = get_distance_class(distance)

        sample_weights.append(
            weights_per_class[cls]
        )

    return torch.tensor(
        sample_weights,
        dtype=torch.float32
    )



def main():

    base_dir = "dataset"

    meta_dir = os.path.join(
        base_dir,
        "meta_data"
    )

    mic_dir = os.path.join(
        base_dir,
        "mic_data"
    )

    all_csv_files = glob.glob(
        os.path.join(
            meta_dir,
            "*.csv"
        )
    )

    print(f"CSV files: {len(all_csv_files)}")

    random.seed(42)

    random.shuffle(all_csv_files)

    train_split = 0.7
    val_split = 0.1

    total_files = len(all_csv_files)

    train_idx = int(total_files * train_split)

    val_idx = int(
        total_files * (train_split + val_split)
    )

    train_files = all_csv_files[:train_idx]

    val_files = all_csv_files[
        train_idx:val_idx
    ]

    test_files = all_csv_files[val_idx:]

    train_dataset = SoundDistanceDataset(
        train_files,
        mic_dir,
        augment=True
    )

    val_dataset = SoundDistanceDataset(
        val_files,
        mic_dir,
        augment=False
    )

    test_dataset = SoundDistanceDataset(
        test_files,
        mic_dir,
        augment=False
    )

    train_sample_weights = get_sample_weights(
        train_dataset
    )

    train_sampler = WeightedRandomSampler(
        train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(device)

    model = SpectrogramCNN(
        n_channels=7,
        num_classes=NUM_CLASSES
    ).to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        train_dataset,
        device,
        patience=15
    )

    model.load_state_dict(
        torch.load(
            "best_spectrogram_model.pth",
            map_location=device
        )
    )

    test_acc = evaluate(
        model,
        test_loader,
        device
    )

    print(f"Test Accuracy: {test_acc:.2f}%")

    plot_confusion_matrix(
        model,
        test_loader,
        device
    )



if __name__ == "__main__":
    main()