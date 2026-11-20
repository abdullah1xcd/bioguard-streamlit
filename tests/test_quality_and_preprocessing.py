import cv2
import numpy as np
import pytest
from src.face import preprocessing as pp
from src.face.quality import assess_face_quality, dhash, hamming, yaw_ratio
from src.voice.preprocessing import assess_audio_quality

rng = np.random.default_rng(0)
SHARP = (rng.random((240, 240)) * 255).astype(np.uint8)
BOX = (20, 20, 200, 200)       # supplied box bypasses the Haar detector


def test_good_sample_passes():
    q = assess_face_quality(SHARP, box=BOX)
    assert q["ok"] and q["reasons"] == []


def test_each_failure_is_reported_with_a_reason():
    assert "too_blurry" in assess_face_quality(cv2.GaussianBlur(SHARP, (0, 0), 12), box=BOX)["reasons"]
    assert "too_dark" in assess_face_quality((SHARP * 0.1).astype(np.uint8), box=BOX)["reasons"]
    assert "too_bright" in assess_face_quality(np.clip(SHARP.astype(int) // 4 + 200, 0, 255).astype(np.uint8), box=BOX)["reasons"]
    assert "low_contrast" in assess_face_quality(np.full((240, 240), 120, np.uint8), box=BOX)["reasons"]
    assert "face_too_small" in assess_face_quality(SHARP, box=(10, 10, 40, 40))["reasons"]
    assert "low_resolution_image" in assess_face_quality(SHARP[:60, :60], box=(0, 0, 60, 60))["reasons"]
    lm = [[60, 50], [100, 50], [110, 70], [65, 90], [95, 90]]         # nose far right of the eye midpoint (yaw 0.75)
    assert "extreme_pose" in assess_face_quality(SHARP, box=BOX, landmarks=lm)["reasons"]


def test_invalid_input_is_rejected_not_crashing():
    for bad in (None, np.zeros((0, 0), np.uint8), np.zeros((4, 4, 4, 4))):
        q = assess_face_quality(bad)
        assert not q["ok"] and q["reasons"][0].startswith("invalid_image")


@pytest.mark.skipif(not pp.cascade_available(), reason="OpenCV build without Haar cascades")
def test_no_face_is_rejected():
    assert assess_face_quality(np.full((240, 240), 120, np.uint8))["reasons"][0] == "no_face_detected"


def test_missing_cascade_raises_clear_error(monkeypatch):
    class Empty:
        def empty(self): return True
    monkeypatch.setattr(pp, "_cascade", Empty())
    with pytest.raises(RuntimeError, match="Haar cascade"):
        pp.detect_box(SHARP)


def test_preprocess_accepts_gray_and_bgr_and_resizes():
    if not pp.cascade_available():
        pytest.skip("no cascade")
    from src.config import FACE_SIZE
    g, _ = pp.preprocess_gray(SHARP); c, _ = pp.preprocess_gray(cv2.cvtColor(SHARP, cv2.COLOR_GRAY2BGR))
    assert g.shape == FACE_SIZE[::-1] or g.shape == FACE_SIZE
    assert c.shape == g.shape
    with pytest.raises(ValueError):
        pp.to_gray(None)


def test_duplicate_hash():
    assert hamming(dhash(SHARP), dhash(SHARP)) == 0
    assert hamming(dhash(SHARP), dhash(255 - SHARP)) > 20
    assert hamming(dhash(SHARP), dhash(cv2.resize(SHARP, (120, 120)))) <= 6     # resolution-robust


def test_yaw_ratio_sign_and_degenerate():
    assert yaw_ratio([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]]) == 0.0
    assert yaw_ratio([[40, 50], [80, 50], [70, 70], [45, 90], [75, 90]]) > 0
    assert yaw_ratio([[50, 50], [50, 50], [60, 70], [45, 90], [75, 90]]) == 0.0


def _tone(sec=2.0, amp=0.5, noise=0.005):
    t = np.arange(int(16000 * sec)) / 16000
    return (amp * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + noise * rng.normal(size=t.size)).astype(np.float32)


def test_audio_quality_gate():
    assert assess_audio_quality(_tone())["ok"]
    assert "silent_or_too_quiet" in assess_audio_quality(np.zeros(16000, np.float32))["reasons"]
    assert "speech_too_short" in assess_audio_quality(_tone(0.3))["reasons"]
    assert "clipped" in assess_audio_quality(np.clip(_tone(amp=3.0), -1, 1))["reasons"]
    assert "low_snr" in assess_audio_quality(0.3 * rng.normal(size=32000).astype(np.float32))["reasons"]
    assert not assess_audio_quality(np.array([], np.float32))["ok"]
