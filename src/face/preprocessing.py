import cv2
import numpy as np
from src.config import FACE_SIZE

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def cascade_available() -> bool:
    return not _cascade.empty()


def to_gray(img: np.ndarray) -> np.ndarray:
    """Accept gray (H,W) or BGR (H,W,3) uint8 and return gray."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("empty or invalid image")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2:
        raise ValueError(f"unsupported image shape {img.shape}")
    return img


def load_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def load_color(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def detect_box(gray: np.ndarray):
    """Largest Haar face box (x, y, w, h) or None. Raises if the cascade file is missing
    (the original code crashed with an opaque OpenCV assertion in that case)."""
    if _cascade.empty():
        raise RuntimeError("Haar cascade XML not found in this OpenCV build. Install opencv-python>=4.8,<5.")
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces):
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    return None


def detect_and_crop(gray: np.ndarray):
    """Haar cascade (classical). Falls back to a centre square crop and reports found=False."""
    box = detect_box(gray)
    if box is not None:
        x, y, w, h = box
        return gray[y:y + h, x:x + w], True
    h, w = gray.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return gray[y0:y0 + s, x0:x0 + s], False


def preprocess_gray(gray: np.ndarray):
    gray = to_gray(gray)
    crop, found = detect_and_crop(gray)
    crop = cv2.resize(crop, FACE_SIZE)
    return _clahe.apply(crop), found


def preprocess(path):
    return preprocess_gray(load_gray(path))


def quality(img: np.ndarray) -> dict:
    """Legacy quality score on the already-normalised crop (kept for backward compatibility).
    New code should use src.face.quality.assess_face_quality on the RAW image."""
    blur = float(cv2.Laplacian(img, cv2.CV_64F).var())
    bright = float(img.mean())
    return {"sharpness": blur, "brightness": bright,
            "ok": blur > 20 and 40 < bright < 220}
