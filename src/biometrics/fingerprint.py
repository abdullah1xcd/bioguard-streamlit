"""Fingerprint verification using a dedicated AFIS matcher.

Primary backend: `afis` (SAFIS minutiae extraction and matching).
Fallback: `fingerprint_recognition` when AFIS is unavailable.

NFIQ2 is reported as a quality indicator, but is not by itself a hard
authentication gate: a low NFIQ2 score should not prevent AFIS from attempting
a match. Basic image validity checks remain enforced.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np


class FingerprintUnavailable(RuntimeError):
    pass


class FingerprintMatcher:
    def __init__(self, threshold: float = 40.0):
        self.threshold = float(threshold)
        self.backend = None
        self._processor = None
        self._fallback = None

    def _load(self):
        if self.backend:
            return
        try:
            from afis import SafisExtractor, match_templates, compute_nfiq2

            self._processor = SafisExtractor()
            self._match_templates = match_templates
            self._nfiq2 = compute_nfiq2
            self.backend = "afis-safis"
            return
        except Exception:
            pass
        try:
            from fingerprint_recognition import compare

            self._fallback = compare
            self.backend = "fingerprint-recognition"
            self.threshold = 12.0
            return
        except Exception as exc:
            raise FingerprintUnavailable(
                "Install fingerprint support with: pip install afis fingerprint-recognition"
            ) from exc

    def quality(self, image: np.ndarray) -> dict:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "ok": False,
                "score": 0.0,
                "nfiq2": None,
                "reason": "empty fingerprint",
            }

        gray = image if image.ndim == 2 else image[..., 0]
        gray = np.asarray(gray, dtype=np.uint8)
        contrast = float(gray.std())
        foreground = float(np.mean(gray < 245))
        basic_ok = contrast >= 12.0 and foreground >= 0.08
        heuristic_score = max(0.0, min(100.0, contrast * 2.2 + foreground * 20.0))

        nfiq2 = None
        nfiq2_warning = None
        try:
            quality_result = self._nfiq2(gray) if hasattr(self, "_nfiq2") else None
            raw_score = getattr(quality_result, "score", quality_result)
            nfiq2 = float(raw_score) if raw_score is not None else None
            if nfiq2 is not None and not np.isfinite(nfiq2):
                nfiq2 = None
            if nfiq2 is not None and nfiq2 < 40.0:
                nfiq2_warning = "NFIQ2 below the recommended quality target"
        except Exception:
            # Quality estimation failure must not be confused with a failed match.
            nfiq2 = None
            nfiq2_warning = "NFIQ2 quality estimation unavailable"

        # NFIQ2 is advisory. The independent validity checks above remain a gate.
        return {
            "ok": bool(basic_ok),
            "score": nfiq2 if nfiq2 is not None else heuristic_score,
            "nfiq2": nfiq2,
            "nfiq2_warning": nfiq2_warning,
            "contrast": contrast,
            "foreground_ratio": foreground,
            "reason": "ok" if basic_ok else "low basic image quality",
        }

    def _afis_template(self, image):
        import cv2

        fd, name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            if not cv2.imwrite(name, image):
                raise ValueError("could not encode fingerprint image")
            return self._processor.extract_minutiae(name)
        finally:
            Path(name).unlink(missing_ok=True)

    @staticmethod
    def _image_fingerprint(image: np.ndarray) -> dict:
        """Describe decoded pixels without exposing or logging biometric image data."""
        arr = np.ascontiguousarray(image)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "pixel_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }

    @staticmethod
    def _template_diagnostics(template) -> dict:
        minutiae = getattr(template, "minutiae", template)
        try:
            count = len(minutiae)
        except (TypeError, AttributeError):
            count = None
        reported = getattr(template, "n_minutiae", None)
        return {
            "template_type": type(template).__name__,
            "minutiae_type": type(minutiae).__name__,
            "minutiae_count": int(reported if reported is not None else count)
            if reported is not None or count is not None else None,
        }

    def _score_with_diagnostics(self, probe: np.ndarray, candidate: np.ndarray):
        self._load()
        probe_quality = self.quality(probe)
        if not probe_quality["ok"]:
            raise ValueError(probe_quality["reason"])
        if candidate is None or not isinstance(candidate, np.ndarray) or candidate.size == 0:
            raise ValueError("enrolled fingerprint image is unreadable")

        diagnostics = {
            "probe_image": self._image_fingerprint(probe),
            "candidate_image": self._image_fingerprint(candidate),
        }
        diagnostics["identical_decoded_pixels"] = (
            diagnostics["probe_image"]["shape"] == diagnostics["candidate_image"]["shape"]
            and diagnostics["probe_image"]["pixel_sha256"] == diagnostics["candidate_image"]["pixel_sha256"]
        )

        if self.backend == "afis-safis":
            probe_template = self._afis_template(probe)
            candidate_template = self._afis_template(candidate)
            probe_minutiae = getattr(probe_template, "minutiae", probe_template)
            candidate_minutiae = getattr(candidate_template, "minutiae", candidate_template)
            result = self._match_templates(
                probe_minutiae, candidate_minutiae, method="safis"
            )
            raw_score = getattr(result, "score", result)
            value = float(raw_score)
            diagnostics.update({
                "probe_template": self._template_diagnostics(probe_template),
                "candidate_template": self._template_diagnostics(candidate_template),
                "matcher": "safis",
                "matcher_result_type": type(result).__name__,
                "matcher_raw_score": repr(raw_score)[:160],
                "matcher_decision": str(getattr(result, "decision", None)),
                "score_scale_note": "Raw SAFIS score; compare against the configured SAFIS threshold.",
            })
        else:
            import cv2

            with tempfile.TemporaryDirectory() as directory:
                probe_path = Path(directory) / "probe.png"
                candidate_path = Path(directory) / "candidate.png"
                if not cv2.imwrite(str(probe_path), probe):
                    raise ValueError("could not encode verification fingerprint")
                if not cv2.imwrite(str(candidate_path), candidate):
                    raise ValueError("could not encode enrolled fingerprint")
                result = self._fallback(str(probe_path), str(candidate_path))
                raw_score = getattr(result, "score", result)
                value = float(raw_score)
                diagnostics.update({
                    "matcher": "fingerprint-recognition",
                    "matcher_result_type": type(result).__name__,
                    "matcher_raw_score": repr(raw_score)[:160],
                })

        if not np.isfinite(value):
            raise ValueError("fingerprint matcher returned a non-finite score")
        return value, diagnostics

    def score(self, probe: np.ndarray, candidate: np.ndarray) -> float:
        value, _diagnostics = self._score_with_diagnostics(probe, candidate)
        return value

    def verify(self, probe: np.ndarray, candidates: list[np.ndarray]) -> dict:
        self._load()
        probe_quality = self.quality(probe)
        base = {
            "accepted": False,
            "score": None,
            "threshold": self.threshold,
            "backend": self.backend,
            "quality": probe_quality,
        }

        if not candidates:
            return {
                **base,
                "status": "not_enrolled",
                "reason": "no readable enrolled fingerprint samples",
                "candidate_count": 0,
                "compared_candidates": 0,
            }

        if not probe_quality["ok"]:
            return {
                **base,
                "status": "low_quality",
                "reason": probe_quality["reason"],
                "candidate_count": len(candidates),
                "compared_candidates": 0,
            }

        candidate_diagnostics = []
        scores = []
        for index, candidate in enumerate(candidates):
            candidate_quality = self.quality(candidate)
            diagnostic = {
                "candidate": index + 1,
                "quality": candidate_quality,
                "score": None,
                "error": None,
            }
            try:
                value, match_diagnostics = self._score_with_diagnostics(probe, candidate)
                diagnostic["score"] = value
                diagnostic["match_diagnostics"] = match_diagnostics
                scores.append(value)
            except Exception as exc:
                diagnostic["error"] = str(exc)
            candidate_diagnostics.append(diagnostic)

        if not scores:
            return {
                **base,
                "status": "inconclusive",
                "reason": "AFIS could not extract or compare templates for any enrolled sample",
                "candidate_count": len(candidates),
                "compared_candidates": 0,
                "candidate_diagnostics": candidate_diagnostics,
            }

        best = max(scores)
        accepted = best >= self.threshold
        return {
            **base,
            "accepted": accepted,
            "score": best,
            "status": "matched" if accepted else "no_match",
            "candidate_count": len(candidates),
            "compared_candidates": len(scores),
            "candidate_scores": scores,
            "candidate_diagnostics": candidate_diagnostics,
            "reason": "fingerprint match" if accepted else "fingerprint did not match enrolled samples",
        }
