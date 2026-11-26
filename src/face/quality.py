"""Face sample quality gate. A bad sample is REJECTED WITH A REASON instead of being silently
treated as a normal biometric sample. Thresholds: src.config.QUALITY['face'] (UNVALIDATED defaults -
tune them on TRAIN-split statistics with scripts.audit_dataset, never on test results).

Covers: no face, tiny face / low resolution, blur, too dark / too bright, low contrast, extreme pose
(only when YuNet landmarks are supplied). NOT covered: occlusion (glasses/mask/hand) - needs a
dedicated model; reported as a limitation."""
import cv2
import numpy as np
from src.config import QUALITY
from src.face.preprocessing import detect_box, to_gray

_STD = 128  # crop size used for size-independent sharpness


def yaw_ratio(landmarks) -> float:
    """Head-yaw proxy from 5 YuNet landmarks [eye_R, eye_L, nose, mouth_R, mouth_L] (image coordinates):
    (nose_x - eye_midpoint_x) / inter-ocular distance. ~0 frontal, sign = direction in the image."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    eye_dist = abs(lm[1, 0] - lm[0, 0])
    if eye_dist < 1e-6:
        return 0.0
    return float((lm[2, 0] - (lm[0, 0] + lm[1, 0]) / 2.0) / eye_dist)


def assess_face_quality(img, box=None, landmarks=None, cfg=None) -> dict:
    """Return quality decision and metrics for the ORIGINAL image.

    Face-box dimensions are kept as floats for threshold comparison; integer truncation
    previously turned a valid 60.0 px detector box into 59 px in some cases.
    """
    cfg = cfg or QUALITY["face"]
    reasons, m = [], {}
    try:
        gray = to_gray(img)
    except ValueError as e:
        return {"ok": False, "reasons": [f"invalid_image: {e}"], "metrics": {}, "box": None}
    h, w = gray.shape
    m["image_h"], m["image_w"] = int(h), int(w)
    if min(h, w) < cfg["min_image_px"]:
        reasons.append("low_resolution_image")
    if box is None:
        box = detect_box(gray)
    if box is None:
        reasons.append("no_face_detected")
        return {"ok": False, "reasons": reasons, "metrics": m, "box": None}

    x, y, bw, bh = (float(v) for v in box)
    # Measure the visible detector box in original-image coordinates, without
    # truncating fractional widths/heights before applying min_face_px.
    visible_w = max(0.0, min(x + bw, float(w)) - max(x, 0.0))
    visible_h = max(0.0, min(y + bh, float(h)) - max(y, 0.0))
    x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
    x1 = min(int(np.ceil(x + bw)), w)
    y1 = min(int(np.ceil(y + bh)), h)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return {"ok": False, "reasons": reasons + ["no_face_detected"], "metrics": m, "box": None}

    m["face_px"] = float(min(visible_w, visible_h))
    if min(visible_w, visible_h) < cfg["min_face_px"]:
        reasons.append("face_too_small")
    std = cv2.resize(crop, (_STD, _STD))
    m["sharpness"] = float(cv2.Laplacian(std, cv2.CV_64F).var())
    m["brightness"] = float(std.mean())
    m["contrast"] = float(std.std())
    if m["sharpness"] < cfg["min_sharpness"]:
        reasons.append("too_blurry")
    lo, hi = cfg["brightness"]
    if m["brightness"] < lo:
        reasons.append("too_dark")
    elif m["brightness"] > hi:
        reasons.append("too_bright")
    if m["contrast"] < cfg["min_contrast"]:
        reasons.append("low_contrast")
    if landmarks is not None:
        m["yaw"] = yaw_ratio(landmarks)
        if abs(m["yaw"]) > cfg["max_abs_yaw"]:
            reasons.append("extreme_pose")
    return {"ok": not reasons, "reasons": reasons, "metrics": m,
            "box": (x0, y0, x1 - x0, y1 - y0)}


def dhash(img, size: int = 8) -> int:
    """64-bit difference hash used to find near-duplicate images / cross-person label collisions."""
    g = cv2.resize(to_gray(img), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
