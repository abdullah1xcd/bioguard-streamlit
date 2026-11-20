"""Face detection + alignment + embedding with OpenCV's built-in DNN face modules (no new pip package):
    YuNet  (FaceDetectorYN)  -> box + 5 landmarks        ~0.2 MB ONNX
    SFace  (FaceRecognizerSF)-> alignCrop (5-pt similarity transform) + 128-D embedding   ~37 MB ONNX
Download the two ONNX files once:  python -m scripts.download_face_models  (SHA-256 verified).
Everything degrades gracefully: FaceEmbedder.available() is False when files/API are missing and the classical
HOG/LBP pipeline keeps working."""
import cv2
import numpy as np
from src.config import SFACE_PATH, YUNET_PATH

MAX_SIDE = 960


class FaceNotFound(ValueError):
    pass


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("empty or invalid image")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"unsupported image shape {img.shape}")


class FaceEmbedder:
    def __init__(self, yunet_path=YUNET_PATH, sface_path=SFACE_PATH, score_threshold=0.8,
                 detector=None, recognizer=None):
        self.yunet_path, self.sface_path, self.score_threshold = str(yunet_path), str(sface_path), score_threshold
        self._det, self._rec = detector, recognizer     # injectable for tests

    @staticmethod
    def available(yunet_path=YUNET_PATH, sface_path=SFACE_PATH) -> bool:
        from pathlib import Path
        return (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
                and Path(yunet_path).is_file() and Path(sface_path).is_file())

    def _load(self):
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(self.yunet_path, "", (320, 320), self.score_threshold, 0.3, 5000)
        if self._rec is None:
            self._rec = cv2.FaceRecognizerSF.create(self.sface_path, "")

    @staticmethod
    def _resize(bgr):
        h, w = bgr.shape[:2]
        s = MAX_SIDE / max(h, w)
        return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr

    def detect(self, img):
        """Returns (bgr_resized, row) where row = [x,y,w,h, 5*(lx,ly), score] of the best face, or (bgr, None)."""
        self._load()
        bgr = self._resize(to_bgr(img))
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(bgr)
        if faces is None or len(faces) == 0:
            return bgr, None
        faces = np.asarray(faces)
        return bgr, faces[int(np.argmax(faces[:, 2] * faces[:, 3] * faces[:, -1]))]

    @staticmethod
    def box_landmarks(row):
        return tuple(float(v) for v in row[:4]), np.asarray(row[4:14], dtype=np.float64).reshape(5, 2)

    def embed(self, img, detection=None):
        """-> (L2-normalised 128-D float32 vector, info). Raises FaceNotFound when no face is detected."""
        self._load()
        bgr, row = detection if detection is not None else self.detect(img)
        if row is None:
            raise FaceNotFound("no face detected")
        aligned = self._rec.alignCrop(bgr, row)
        f = np.asarray(self._rec.feature(aligned), dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(f))
        if not np.isfinite(n) or n < 1e-8:
            raise FaceNotFound("degenerate embedding")
        box, lm = self.box_landmarks(row)
        return f / n, {"box": box, "landmarks": lm, "det_score": float(row[-1])}
