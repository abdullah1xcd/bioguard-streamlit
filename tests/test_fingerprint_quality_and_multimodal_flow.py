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
    matcher._score_with_diagnostics = lambda probe, candidate: (calls.append((probe, candidate)) or 55.0, {"test": True})

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



def test_afis_diagnostics_report_identical_pixels_and_minutiae(monkeypatch):
    matcher = FingerprintMatcher(threshold=40.0)
    matcher.backend = "afis-safis"
    matcher._nfiq2 = lambda _image: 22

    class Template:
        def __init__(self, count):
            self.minutiae = [object() for _ in range(count)]

    class MatchResult:
        score = 52.0
        decision = True

    class SafisProcessor:
        def __init__(self):
            self.received_templates = None

        def match(self, probe_template, candidate_template):
            self.received_templates = (probe_template, candidate_template)
            return MatchResult()

    matcher._processor = SafisProcessor()
    matcher._afis_template = lambda _image: Template(37)

    image = _fingerprint_image()
    score, diagnostics = matcher._score_with_diagnostics(image, image.copy())

    assert score == 52.0
    assert matcher._processor.received_templates is not None
    assert all(isinstance(item, Template) for item in matcher._processor.received_templates)
    assert diagnostics["identical_decoded_pixels"] is True
    assert diagnostics["probe_template"]["minutiae_count"] == 37
    assert diagnostics["candidate_template"]["minutiae_count"] == 37
    assert diagnostics["matcher"] == "safis"
    assert diagnostics["matcher_raw_score"] == "52.0"
    assert diagnostics["matcher_decision"] == "True"


def test_safis_score_below_threshold_is_not_overridden_by_positive_decision():
    matcher = FingerprintMatcher(threshold=40.0)
    matcher.backend = "afis-safis"
    matcher._nfiq2 = lambda _image: 22

    class Template:
        minutiae = [object() for _ in range(18)]

    class MatchResult:
        score = 1.0
        decision = True

    class SafisProcessor:
        def match(self, _probe, _candidate):
            return MatchResult()

    matcher._processor = SafisProcessor()
    matcher._afis_template = lambda _image: Template()

    image = _fingerprint_image()
    result = matcher.verify(image, [image.copy()])

    assert result["accepted"] is False
    assert result["score"] == 1.0
    assert result["threshold"] == 40.0
    assert result["status"] == "no_match"



def test_safis_native_positive_decision_accepts_normalized_score():
    matcher = FingerprintMatcher(threshold=40.0)
    matcher.backend = "afis-safis"
    matcher._nfiq2 = lambda _image: 22
    matcher._score_with_diagnostics = lambda _probe, _candidate: (
        1.0,
        {"matcher_decision_bool": True},
    )

    image = _fingerprint_image()
    result = matcher.verify(image, [image.copy()])

    assert result["accepted"] is True
    assert result["score"] == 1.0
    assert result["decision_method"] == "safis_native"
    assert result["threshold_applied"] is False


def test_safis_native_negative_decision_rejects_even_if_score_is_high():
    matcher = FingerprintMatcher(threshold=40.0)
    matcher.backend = "afis-safis"
    matcher._nfiq2 = lambda _image: 22
    matcher._score_with_diagnostics = lambda _probe, _candidate: (
        100.0,
        {"matcher_decision_bool": False},
    )

    image = _fingerprint_image()
    result = matcher.verify(image, [image.copy()])

    assert result["accepted"] is False
    assert result["score"] == 100.0
    assert result["decision_method"] == "safis_native"
    assert result["threshold_applied"] is False
