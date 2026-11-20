"""Basic face liveness = random, time-limited challenge-response.

    camera frames -> face detected (YuNet box + 5 landmarks) -> RANDOM challenge -> expected movement measured
    -> passed? -> continue with biometric verification            (failed -> reject, no biometric matching)

Challenges (src.config.LIVENESS_CFG):
  turn_left / turn_right : yaw proxy = (nose_x - eye_midpoint_x) / inter-ocular distance (landmark geometry).
                           'left'/'right' are from the SUBJECT's point of view in the RAW (un-mirrored) camera image.
  move_closer            : face-box width grows by >= closer_ratio_min.
Blink is intentionally NOT implemented: it needs eye landmarks that YuNet does not provide, and the old Haar-eye
heuristic in challenges/visual.py is unreliable.

HONEST LIMITS: thresholds are UNVALIDATED engineering defaults (no attack data was available); a hand-moved printed
photo or a replayed video of the right movement can still pass.  Measure APCER/BPCER with
evaluate_liveness_clips() on recorded genuine + attack clips before relying on this as a security control."""
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

from src.config import LIVENESS_CFG
from src.evaluation.metrics import liveness_rates
from src.face.quality import yaw_ratio

SIGN = {"turn_left": +1.0, "turn_right": -1.0}


@dataclass
class Challenge:
    name: str
    issued_at: float
    ttl_s: float
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))

    @property
    def instruction(self):
        return {"turn_left": "Slowly turn your head to YOUR left", "turn_right": "Slowly turn your head to YOUR right",
                "move_closer": "Move your face closer to the camera"}[self.name]


def new_challenge(now=None, cfg=None) -> Challenge:
    cfg = cfg or LIVENESS_CFG
    return Challenge(secrets.choice(list(cfg["challenges"])), time.time() if now is None else now, cfg["ttl_s"])


def observation(box=None, landmarks=None) -> dict:
    """One frame: box=(x,y,w,h) or None (no face); landmarks 5x2 or None."""
    return {"box": None if box is None else tuple(float(v) for v in box),
            "yaw": None if landmarks is None else yaw_ratio(landmarks)}


def _sustained(mask, k):
    run = 0
    for m in mask:
        run = run + 1 if m else 0
        if run >= k:
            return True
    return False


def _smooth(x, k=3):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def evaluate_challenge(ch: Challenge, obs: list, now=None, cfg=None) -> dict:
    cfg = cfg or LIVENESS_CFG
    now = time.time() if now is None else now
    res = lambda ok, why, **m: {"passed": bool(ok), "challenge": ch.name, "reason": why, "metrics": m}
    if now - ch.issued_at > ch.ttl_s:
        return res(False, "challenge expired")
    if not obs or len(obs) < cfg["min_frames"]:
        return res(False, f"too few frames ({len(obs) if obs else 0} < {cfg['min_frames']})")
    with_face = [o for o in obs if o["box"] is not None]
    ratio = len(with_face) / len(obs)
    if ratio < cfg["min_face_frames_ratio"]:
        return res(False, "face lost during challenge", face_frame_ratio=ratio)
    k = cfg["sustain_frames"]
    if ch.name in SIGN:
        yaws = [o["yaw"] for o in with_face if o["yaw"] is not None]
        if len(yaws) < cfg["min_frames"] // 2:
            return res(False, "no landmarks available for pose estimation")
        y = _smooth(yaws); base = float(np.median(y[:3]))
        if abs(base) > cfg["yaw_frontal_max"]:
            return res(False, "did not start from a frontal pose", baseline_yaw=base)
        d = (y - base) * SIGN[ch.name]
        ok = _sustained(d >= cfg["yaw_turn_min"], k)
        wrong = _sustained(-d >= cfg["yaw_turn_min"], k)
        return res(ok, "movement detected" if ok else ("turned the wrong way" if wrong else "no sufficient head turn"),
                   baseline_yaw=base, max_delta=float(d.max()))
    if ch.name == "move_closer":
        w = np.array([o["box"][2] for o in with_face]); base = float(np.median(w[:3]))
        rel = _smooth(w) / max(base, 1e-6)
        ok = _sustained(rel >= cfg["closer_ratio_min"], k)
        return res(ok, "movement detected" if ok else "face did not move closer", max_ratio=float(rel.max()))
    return res(False, f"unknown challenge {ch.name}")


def observations_from_frames(frames, embedder) -> list:
    """Run YuNet on each frame (BGR). Requires the ONNX models (scripts.download_face_models)."""
    out = []
    for fr in frames:
        _, row = embedder.detect(fr)
        out.append(observation(*embedder.box_landmarks(row)) if row is not None else observation())
    return out


def evaluate_liveness_clips(clips, cfg=None) -> dict:
    """clips: iterable of (challenge_name, observations, is_attack).  Returns APCER/BPCER/ACER (ISO/IEC 30107-3 style)."""
    flags, passed = [], []
    for name, obs, is_attack in clips:
        ch = Challenge(name, 0.0, 1e9)
        passed.append(evaluate_challenge(ch, obs, now=1.0, cfg=cfg)["passed"]); flags.append(is_attack)
    return liveness_rates(flags, passed)
