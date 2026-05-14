# Sonar Retina 📡

**Sonar Retina** is a real-time assistive application designed for individuals who are deaf or hard of hearing. It transforms environmental audio into a spatial "Radar UI," allowing users to perceive the distance and relative presence of surrounding sounds through visual cues.

---

## 🏗 System Architecture

The project consists of two main components:

1.  **Frontend (Flutter)**: A mobile interface that captures audio chunks in real-time (400ms) and visualizes AI predictions on a custom-painted radar.
2.  **Backend (FastAPI)**: An AI service running a PyTorch Hybrid Model (Spectrogram + IPD) to estimate sound source distance.

---

## 🚀 Getting Started

### 1. Prerequisites

- **Flutter SDK**: [Install Flutter](https://docs.flutter.dev/get-started/install)
- **Python 3.8+**: [Install Python](https://www.python.org/downloads/)
- **PyTorch**: Required for running the AI inference model.

### 2. Backend Setup (SSL Service)

Navigate to the backend directory and install dependencies:

```bash
cd sslservice
pip install -r requirements.txt  # Ensure you have torch, fastapi, uvicorn, and librosa
```

**Start the AI Service:**

```bash
python app.py
```

The server will start at `http://localhost:8000`.

### 3. Frontend Setup (Interface)

Navigate to the interface directory and fetch Flutter packages:

```bash
cd interface
flutter pub get
```

**Run the Application:**

- Ensure your backend is running.
- If using an Android Emulator, the app is pre-configured to connect to `10.0.2.2:8000`.

```bash
flutter run
```

---

## 📂 Project Structure

- `interface/`: Flutter source code, including custom radar painters and audio streaming logic.
- `sslservice/`: FastAPI implementation and model loading utilities.
- `dcase/`: Data exploration scripts and dataset utilities.
- `spectrogram_model_v2.pth`: The trained PyTorch model weights used for inference.

---

## 📝 Documentation

For detailed information on the **UI-Model Interface Contract** (JSON Schemas, Sequence Diagrams, and Testing Scenarios), please refer to:

- [Task C Documentation](C:/Users/patth/.gemini/antigravity/brain/f7d01558-1927-40f5-bbae-1674c4d72f1a/c_task_documentation.md)

---

## 📚 Credits & Datasets

- **SSL Dataset**: DCASE (Sound source distance estimation in diverse and dynamic acoustic conditions).
- **SED Dataset**: FSD50K (Human-Labeled Sound Events).
