# BioGuard-X v4 — Multimodal Biometric Authentication

BioGuard-X is now focused on **three biometric modalities**:

- **Face:** Ultralytics YOLO face detection (optional) / InsightFace SCRFD + ArcFace embeddings.
- **Fingerprint:** `afis` AFIS/minutiae matching with NFIQ2-style quality support, plus a portable `fingerprint-recognition` fallback.
- **Voice:** SpeechBrain pretrained **ECAPA-TDNN** speaker embeddings + cosine similarity.
- **Fusion:** auditable score-level fusion after modality-specific thresholds/normalization.
- **Liveness:** the existing challenge-response face liveness module can gate authentication.
- **Evaluation:** FAR/FMR, FRR/FNMR, EER, ROC-AUC, confusion matrix, latency and quality-failure reporting.

## Removed from the active application

The following experimental/unrelated subsystems were removed from the active product to keep the framework small and accuracy-focused:

- local LLM/Ollama explanations
- speech command recognition
- automation/workflow engine
- approval/notification center
- security analyst agent
- Isolation Forest risk scoring
- old HOG/LBP/SVM/Random-Forest face classifier path
- old MFCC-only speaker identity model as the primary path
- old HOG/LBP fingerprint classifier as the primary path
- early feature-level fusion

Legacy material is intentionally not part of the v4 runtime.

## Installation

**Recommended on Windows:** Python 3.13 for the widest PyTorch/SpeechBrain compatibility.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app/main.py
```

The first use of InsightFace/ArcFace and ECAPA-TDNN downloads their pretrained model files. Keep the machine online for the first model initialization.

## Enrollment

For meaningful accuracy testing, use multiple capture sessions:

```text
participant_001/
  face/session_1/       # enrollment/reference
  face/session_2/       # validation
  face/session_3/       # final test
  fingerprint/session_1/
  fingerprint/session_2/
  fingerprint/session_3/
  voice/session_1/
  voice/session_2/
  voice/session_3/
```

Do not tune thresholds on session 3.

## Accuracy rule

The UI's default thresholds are **engineering defaults only**. Final thresholds and fusion weights must be calibrated on the validation split and then frozen before the final test. Never claim an accuracy number from enrollment data alone.

## Fingerprint notes

`afis` is the primary fingerprint backend. It provides dedicated minutiae extraction/matching and quality support. The documented SourceAFIS family uses a similarity score and a threshold; the final BioGuard-X threshold should still be calibrated on the project's own sensor/dataset.

## Voice notes

The primary voice model is SpeechBrain's `spkrec-ecapa-voxceleb`, a pretrained ECAPA-TDNN speaker verification model. It is used only for speaker identity, not speech-to-text or command recognition.

## Evaluation

```powershell
python -m scripts.evaluate_multimodal --root datasets/custom --split 2
```

Use session 2 for validation during development and session 3 for the final held-out test after you have frozen the thresholds.
