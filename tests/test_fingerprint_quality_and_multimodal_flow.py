import numpy as np

from src.biometrics.fingerprint import FingerprintMatcher
from src.platform import Platform


def _fingerprint_image():
    image = np.zeros((128, 128), dtype=np.uint8)
    image[:, ::2] = 255
    return image


def test_low_nfiq2_is_reported_as_warning_not_hard_rejection():
    matcher = FingerprintMatcher(threshold=40.0)
    matcher.backend = "afis-safis"
    matcher._nfiq2 = lambda _image: 22

    result = matcher.quality(_fingerprint_image())

    assert result["ok"] is True
    assert result["nfiq2"] == 22.0
    assert result["nfiq2_warning"] is not None


def test_verification_attempts_matching_when_nfiq2_is_low():
    matcher = FingerprintMatcher(threshold=40.0)
    matcher.backend = "afis-safis"
    matcher._nfiq2 = lambda _image: 22
    calls = []
    matcher.score = lambda probe, candidate: calls.append((probe, candidate)) or 55.0

    image = _fingerprint_image()
    result = matcher.verify(image, [image.copy()])

    assert len(calls) == 1
    assert result["accepted"] is True
    assert result["score"] == 55.0
    assert result["status"] == "matched"
    assert result["quality"]["nfiq2"] == 22.0
    assert result["compared_candidates"] == 1


def test_multimodal_runs_voice_even_when_fingerprint_fails():
    platform = Platform.__new__(Platform)
    calls = []
    platform.verify_face = lambda claimed, image: (
        calls.append("face") or {"accepted": True, "score": 0.9}
    )
    platform.verify_fingerprint = lambda claimed, image: (
        calls.append("fingerprint")
        or {
            "accepted": False,
            "score": None,
            "status": "low_quality",
            "reason": "low basic image quality",
        }
    )
    platform.verify_voice = lambda claimed, wave: (
        calls.append("voice") or {"accepted": True, "score": 0.8}
    )

    result = platform.verify_multimodal("test_user", object(), object(), object())

    assert calls == ["face", "fingerprint", "voice"]
    assert result["accepted"] is False
    assert result["fingerprint"]["accepted"] is False
    assert result["voice"]["accepted"] is True
    assert result["failed_modalities"] == ["fingerprint"]
