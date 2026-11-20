"""EXPERIMENTAL classical-CV challenges (Haar cascades, frame differencing). Not production-grade liveness detection."""
import cv2
import numpy as np

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _face_box(gray):
    f = _face.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return max(f, key=lambda b: b[2] * b[3]) if len(f) else None


def head_movement(frames, direction="left", min_shift=0.12):
    """frames: list of grayscale images. direction as seen in the (un-mirrored) image: left/right/up/down."""
    centers = []
    for g in frames:
        b = _face_box(g)
        if b is not None:
            centers.append(((b[0] + b[2] / 2) / g.shape[1], (b[1] + b[3] / 2) / g.shape[0]))
    if len(centers) < max(3, len(frames) // 3):
        return {"passed": False, "reason": "face not tracked in enough frames"}
    c = np.array(centers); dx, dy = c[-1, 0] - c[0, 0], c[-1, 1] - c[0, 1]
    shift = {"left": -dx, "right": dx, "up": -dy, "down": dy}[direction]
    return {"passed": bool(shift >= min_shift), "shift": float(shift), "frames_tracked": len(centers)}


def blink_detected(frames):
    """Basic: eyes visible, then not visible (closed), then visible again inside the face box."""
    seq = []
    for g in frames:
        b = _face_box(g)
        if b is None:
            continue
        x, y, w, h = b
        seq.append(len(_eye.detectMultiScale(g[y:y + h // 2, x:x + w], 1.1, 4, minSize=(15, 15))) >= 1)
    s = "".join("1" if v else "0" for v in seq)
    return {"passed": "10" in s and "01" in s[s.find("10"):], "pattern": s}


def motion_score(frames):
    """Mean absolute frame difference (a static printed photo gives ~0 non-rigid motion)."""
    d = [np.mean(cv2.absdiff(frames[i], frames[i + 1])) for i in range(len(frames) - 1)]
    return float(np.mean(d)) if d else 0.0
