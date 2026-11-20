"""Face engine: detect -> align -> ArcFace embedding.  Pretrained models only, nothing to train.

Detector (FACE_CFG["detector"]):
    yolo  : Ultralytics Ultralytics YOLO face detector (box + 5 landmarks).  Needs `pip install ultralytics`; weights are auto-downloaded once.
    scrfd : InsightFace SCRFD (part of the buffalo_l pack).  Used when YOLO is missing and as a fallback when YOLO finds no face.
Recogniser: InsightFace ArcFace (w600k_r50, 512-D).  If InsightFace is missing, OpenCV SFace (128-D) is used when its ONNX
files are present (python -m scripts.download_face_models).
"""
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from src.config import FACE_CFG, SFACE_PATH, YOLO_FACE_PATH, YOLO_FACE_URL, YUNET_PATH


class FaceNotFound(ValueError):
    pass


class EngineUnavailable(RuntimeError):
    pass


def to_bgr(img):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("empty or invalid image")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"unsupported image shape {img.shape}")


def l2n(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        raise FaceNotFound("degenerate embedding")
    return v / n


class FaceEngine:
    def __init__(self, cfg=None):
        self.cfg = {**FACE_CFG, **(cfg or {})}
        self._fa = None          # insightface FaceAnalysis
        self._yolo = None        # ultralytics YOLO
        self._sface = None       # OpenCV fallback (detector, recogniser)
        self._yolo_failed = False
        self.backend = None      # filled in by _load(): e.g. "yolo+arcface"

    # ------------------------------------------------------------------ loading
    @staticmethod
    def has_insightface():
        try:
            import insightface  # noqa: F401
            import onnxruntime  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def has_yolo():
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False

    def _load_insight(self):
        if self._fa is None:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or ["CPUExecutionProvider"]
            fa = FaceAnalysis(name=self.cfg["model_pack"], allowed_modules=["detection", "recognition"], providers=prov)
            s = int(self.cfg["det_size"])
            fa.prepare(ctx_id=0 if prov[0].startswith("CUDA") else -1, det_size=(s, s), det_thresh=float(self.cfg["scrfd_thresh"]))
            self._fa = fa
        return self._fa

    def _load_yolo(self):
        if self._yolo is None and not self._yolo_failed:
            try:
                from ultralytics import YOLO
                p = Path(YOLO_FACE_PATH)
                if not p.is_file():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(YOLO_FACE_URL, p)
                self._yolo = YOLO(str(p))
            except Exception:             # no ultralytics / no internet -> SCRFD takes over
                self._yolo_failed = True
        return self._yolo

    def _load_sface(self):
        if self._sface is None:
            if not (hasattr(cv2, "FaceDetectorYN") and Path(YUNET_PATH).is_file() and Path(SFACE_PATH).is_file()):
                raise EngineUnavailable("No face model available. Run:  pip install insightface onnxruntime   "
                                        "(or python -m scripts.download_face_models for the OpenCV SFace fallback)")
            det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.8, 0.3, 5000)
            rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
            self._sface = (det, rec)
        return self._sface

    # ------------------------------------------------------------------ detection
    def _detect_yolo(self, bgr):
        m = self._load_yolo()
        if m is None:
            return None
        r = m.predict(bgr, conf=float(self.cfg["yolo_conf"]), verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return None
        xyxy, conf, kps = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.xy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        i = int(np.argmax(areas * conf))
        if kps[i].shape != (5, 2):
            return None
        return {"box": tuple(float(v) for v in xyxy[i]), "landmarks": kps[i].astype(np.float32), "det_score": float(conf[i]), "detector": "yolo"}

    def _detect_scrfd(self, bgr):
        fa = self._load_insight()
        det = fa.models["detection"]
        bboxes, kpss = det.detect(bgr, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas * bboxes[:, 4]))
        return {"box": tuple(float(v) for v in bboxes[i, :4]), "landmarks": kpss[i].astype(np.float32),
                "det_score": float(bboxes[i, 4]), "detector": "scrfd"}

    def detect(self, img):
        """-> dict(box=(x1,y1,x2,y2), landmarks(5,2), det_score, detector) of the best face, or None."""
        bgr = to_bgr(img)
        mode = self.cfg["detector"]
        if self.has_insightface():
            if mode in ("auto", "yolo") and (mode == "yolo" or self.has_yolo()):
                d = self._detect_yolo(bgr)
                if d is not None:
                    return d
            return self._detect_scrfd(bgr)
        det, _ = self._load_sface()                    # OpenCV fallback
        det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = np.asarray(faces)[int(np.argmax(np.asarray(faces)[:, 2] * np.asarray(faces)[:, 3] * np.asarray(faces)[:, -1]))]
        return {"box": (float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])), "landmarks": np.asarray(f[4:14], np.float32).reshape(5, 2),
                "det_score": float(f[-1]), "detector": "yunet", "_row": f}

    # ------------------------------------------------------------------ embedding
    def embed(self, img, detection=None):
        """-> (L2-normalised embedding, detection dict).  Raises FaceNotFound."""
        bgr = to_bgr(img)
        d = detection if detection is not None else self.detect(bgr)
        if d is None:
            raise FaceNotFound("no face detected")
        if self.has_insightface():
            from insightface.utils import face_align
            rec = self._load_insight().models["recognition"]
            aligned = face_align.norm_crop(bgr, d["landmarks"], image_size=112)
            feat = rec.get_feat(aligned)
            self.backend = f"{d['detector']}+arcface"
        else:
            _, rec = self._load_sface()
            feat = rec.feature(rec.alignCrop(bgr, d["_row"]))
            self.backend = "yunet+sface"
        return l2n(feat), d

    def info(self):
        return {"backend": self.backend, "insightface": self.has_insightface(), "yolo": self.has_yolo(), "detector_setting": self.cfg["detector"]}
