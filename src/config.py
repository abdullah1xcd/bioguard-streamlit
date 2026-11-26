from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "custom"
PUBLIC_DATASET_DIR = ROOT / "datasets" / "public" / "faciavox"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

FACE_SIZE = (128, 128)
SAMPLE_RATE = 16000
SEED = 42
SPLIT = (0.6, 0.2, 0.2)  # train / val / test, per participant
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FINGERPRINT_EXTS = IMAGE_EXTS | {".tif", ".tiff"}
FINGERPRINT_UPLOAD_TYPES = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
AUDIO_EXTS = {".wav"}
N_REFS, N_IMP = 5, 3

# Runtime-editable defaults (stored in the app_settings table once changed in the UI)
DEFAULT_SETTINGS = {
    "fail_threshold": 5, "fail_window_s": 120,
    "challenge_fail_threshold": 3, "unauthorized_threshold": 3, "unauthorized_window_s": 600,
    "disagreement_gap": 0.5, "failure_rate_threshold": 0.5, "failure_rate_min_attempts": 10,
    "automation_fail_threshold": 3, "high_risk_threshold": 50.0,
    "approval_ttl_s": 3600, "scrutiny_minutes": 30, "lockout_minutes": 15,
    "backup_keep": 7, "log_retention_days": 90, "model_degradation_eer": 0.05,
    "session_max_age_s": 300, "challenge_ttl_s": 60,
    "require_liveness": False,
    "ollama_enabled": False, "ollama_model": "llama3.2", "ollama_url": "http://localhost:11434",
}

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "bioguardx.db"
BACKUP_DIR = ROOT / "backups"
GENERATED_REPORTS = REPORTS_DIR / "generated"
LOGS_DIR = LOG_DIR
RETENTION_DAYS = 365
BACKUPS_TO_KEEP = 7
RISK_HIGH_THRESHOLD = 50          # risk score that fires the high_risk_session event
AGENT_CFG = {"window_minutes": 10, "failed_threshold": 3, "challenge_fail_threshold": 2,
             "unauthorized_threshold": 2, "disagreement_gap": 0.5, "failure_rate_threshold": 0.5,
             "failure_rate_min_attempts": 10, "high_risk_score": RISK_HIGH_THRESHOLD}


# ======================================================================================
# v2 (accuracy-focused) configuration.  Every value here is stored in the run manifest.
# Values marked UNVALIDATED are engineering defaults: tune them on TRAIN-split quality
# statistics (python -m scripts.audit_dataset), never on test results.
# ======================================================================================
SESSION_PREFIX = "session_"               # datasets/custom/<pid>/face/session_1/*.jpg ...
MODELS_ONNX_DIR = MODELS_DIR / "onnx"
YUNET_PATH = MODELS_ONNX_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_ONNX_DIR / "face_recognition_sface_2021dec.onnx"
# SHA-256 taken from the opencv_zoo Git-LFS pointers (verified again by scripts.download_face_models)
FACE_MODEL_SHA256 = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}
FACE_MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
EVAL_DIR = REPORTS_DIR / "eval"

QUALITY = {                                # UNVALIDATED defaults
    "face": {"min_face_px": 80, "min_sharpness": 25.0, "brightness": (45.0, 215.0),
             "min_contrast": 25.0, "max_abs_yaw": 0.55, "min_image_px": 100},
    "voice": {"min_speech_s": 0.8, "min_snr_db": 8.0, "max_clip_ratio": 0.01, "min_rms": 1e-3},
}
VOICE_CFG = {"vad_top_db": 30, "min_segment_s": 0.05, "denoise_alpha": 1.0, "denoise_floor": 0.1,
             "n_mfcc": 20, "variants": ("v1", "v2", "v2d")}
EVAL_CFG = {"n_refs": 5, "n_imp": 10, "n_boot": 500, "fusion_cv_folds": 5, "far_targets": (0.01, 0.001),
            "dup_hamming": 3}
LIVENESS_CFG = {"challenges": ("turn_left", "turn_right", "move_closer"),   # UNVALIDATED thresholds
                "yaw_turn_min": 0.30, "yaw_frontal_max": 0.18, "closer_ratio_min": 1.25,
                "min_frames": 6, "min_face_frames_ratio": 0.7, "sustain_frames": 2, "ttl_s": 20}


# ======================================================================================
# v3 face engine (pretrained detector + ArcFace embeddings, NO training step)
# ======================================================================================
YOLO_FACE_PATH = MODELS_ONNX_DIR / "yolov8n-face.pt"
YOLO_FACE_URL = "https://github.com/akanametov/yolov8-face/releases/download/1.0.0/yolov8n-face.pt"
GALLERY_PATH = MODELS_DIR / "face_gallery.joblib"
FACE_CFG = {
    # detector: "auto" = YOLO if `ultralytics` is installed, else InsightFace SCRFD.  Force with "yolo" or "scrfd".
    "detector": "auto",
    "yolo_conf": 0.5,
    "scrfd_thresh": 0.5,
    "det_size": 640,
    "model_pack": "buffalo_l",         # SCRFD detector + ArcFace (w600k_r50) recogniser
    "threshold": 0.38,                 # cosine similarity; ArcFace same-person ~0.45-0.8, different ~ -0.1-0.25
    "min_threshold": 0.30, "max_threshold": 0.60,
    "top_k": 3,                        # score = mean of the k best similarities to the enrolled photos
    "outlier_cos": 0.20,               # enrolled photo dropped when it is this dissimilar to the user's median template
    "min_enroll_images": 3,
    "min_face_px": 60, "min_sharpness": 15.0, "brightness": (35.0, 225.0), "max_abs_yaw": 0.65,
}

# ======================================================================================
# v4 multimodal biometric stack
# ======================================================================================
FINGERPRINT_CFG = {
    "backend": "afis",
    # SourceAFIS-style / AFIS scores are not on the same scale as cosine scores.
    # 40 is a documented starting point for SourceAFIS; final threshold must be
    # calibrated on the project's own validation set.
    "threshold": 40.0,
    "min_samples": 2,
}
VOICE_MODEL_CFG = {
    "source": "speechbrain/spkrec-ecapa-voxceleb",
    "threshold": 0.60,  # engineering default; calibrate on validation pairs
    "min_samples": 2,
    "min_duration_s": 1.0,
}
FUSION_CFG = {
    "weights": (0.40, 0.35, 0.25),  # face / fingerprint / voice; validation-tunable
    "threshold": 0.70,
}
