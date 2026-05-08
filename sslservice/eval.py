import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# Import from your training file
from spectogram_model import SoundDistanceDataset, SpectrogramCNN


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(model_path, device, num_classes=3):
    model = SpectrogramCNN(
        n_channels=4,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    print(f"Loaded model from {model_path}")

    return model


# =========================================================
# EVALUATE
# =========================================================
def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []

    print("Running evaluation...")

    with torch.no_grad():
        for features, labels in test_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


# =========================================================
# CONFUSION MATRIX
# =========================================================
def plot_confusion_matrix(
    cm,
    class_names,
    output_file="confusion_matrix.png"
):
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    plt.title("Confusion Matrix - Distance Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()

    plt.savefig(output_file, dpi=100)

    print(f"Confusion matrix saved to {output_file}")

    plt.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # -------------------------
    # PATHS
    # -------------------------
    base_dir = "dcase/zenodo_upload"

    meta_dir = os.path.join(base_dir, "meta_data")
    mic_dir = os.path.join(base_dir, "mic_data")

    model_path = "best_spectrogram_model.pth"

    # -------------------------
    # DEVICE
    # -------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # -------------------------
    # CLASS NAMES
    # -------------------------
    class_names = [
        "Near (0-2.5m)",
        "Medium (2.5-5m)",
        "Far (5-10m)"
    ]

    # -------------------------
    # TEST FILES
    # -------------------------
    test_files = glob.glob(
        os.path.join(meta_dir, "fold2_*.csv")
    )

    print(f"Found {len(test_files)} test files")

    # -------------------------
    # DATASET
    # -------------------------
    test_dataset = SoundDistanceDataset(
        test_files,
        mic_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(f"Test examples: {len(test_dataset)}")

    # -------------------------
    # LOAD MODEL
    # -------------------------
    model = load_model(
        model_path,
        device
    )

    # -------------------------
    # EVALUATE
    # -------------------------
    predictions, true_labels = evaluate_model(
        model,
        test_loader,
        device
    )

    # -------------------------
    # METRICS
    # -------------------------
    cm = confusion_matrix(
        true_labels,
        predictions
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    # -------------------------
    # RESULTS
    # -------------------------
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Overall Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            digits=4
        )
    )

    # -------------------------
    # PER CLASS ACCURACY
    # -------------------------
    print("\nPer-Class Accuracy:")

    for i, class_name in enumerate(class_names):

        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()

            print(
                f"{class_name}: "
                f"{class_acc:.4f} "
                f"({cm[i, i]}/{cm[i].sum()})"
            )
        else:
            print(f"{class_name}: No samples")

    # -------------------------
    # PLOT MATRIX
    # -------------------------
    plot_confusion_matrix(
        cm,
        class_names,
        "confusion_matrix.png"
    )