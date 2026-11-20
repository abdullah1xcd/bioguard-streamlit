"""Fingerprint verification using a dedicated AFIS matcher.

Primary backend: `afis` (pure Python MINDTCT/SAFIS paths, optional NBIS).
Fallback: `fingerprint_recognition` for environments where AFIS is unavailable.
The old HOG/LBP fingerprint classifier is intentionally not used by the main app.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import os
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
        except Exception as e:
            raise FingerprintUnavailable(
                "Install fingerprint support with: pip install afis fingerprint-recognition"
            ) from e

    def quality(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            return {"ok": False, "score": 0.0, "reason": "empty fingerprint"}
        g = image if image.ndim == 2 else image[..., 0]
        g = np.asarray(g, dtype=np.uint8)
        contrast = float(g.std())
        foreground = float(np.mean(g < 245))
        # Quality is deliberately a gate, not a fake accuracy score.
        score = max(0.0, min(100.0, contrast * 2.2 + foreground * 20.0))
        ok = contrast >= 12.0 and foreground >= 0.08
        # Prefer the AFIS package's NFIQ2-style quality score when available.
        try:
            q = self._nfiq2(g) if hasattr(self, "_nfiq2") else None
            nfiq = float(getattr(q, "score", q)) if q is not None else None
            if nfiq is not None and np.isfinite(nfiq):
                score = nfiq
                ok = ok and nfiq >= 40.0
        except Exception:
            nfiq = None
        return {"ok": ok, "score": score, "nfiq2": nfiq, "contrast": contrast, "foreground_ratio": foreground,
                "reason": "ok" if ok else "low fingerprint image quality"}

    def _afis_template(self, image):
        # afis supports paths in its high-level processor; use a temporary PNG for
        # compatibility with both its pure-Python and optional native backends.
        import cv2
        fd, name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            if not cv2.imwrite(name, image):
                raise ValueError("could not encode fingerprint image")
            return self._processor.extract_minutiae(name)
        finally:
            Path(name).unlink(missing_ok=True)

    def score(self, probe: np.ndarray, candidate: np.ndarray) -> float:
        self._load()
        q = self.quality(probe)
        if not q["ok"]:
            raise ValueError(q["reason"])
        if self.backend == "afis-safis":
            a = self._afis_template(probe)
            b = self._afis_template(candidate)
            # Newer afis exposes match_templates; accept both template/minutiae forms.
            aa = getattr(a, "minutiae", a)
            bb = getattr(b, "minutiae", b)
            result = self._match_templates(aa, bb, method="safis")
            return float(getattr(result, "score", result))
        with tempfile.TemporaryDirectory() as td:
            import cv2
            p = Path(td) / "probe.png"; c = Path(td) / "candidate.png"
            cv2.imwrite(str(p), probe); cv2.imwrite(str(c), candidate)
            result = self._fallback(str(p), str(c))
            return float(getattr(result, "score", result))

    def verify(self, probe: np.ndarray, candidates: list[np.ndarray]) -> dict:
        q = self.quality(probe)
        if not candidates:
            return {"accepted": False, "score": None, "threshold": self.threshold,
                    "backend": self.backend, "quality": q, "reason": "no enrolled fingerprint"}
        if not q["ok"]:
            return {"accepted": False, "score": None, "threshold": self.threshold,
                    "backend": self.backend, "quality": q, "reason": q["reason"]}
        scores = [self.score(probe, c) for c in candidates]
        best = max(scores)
        return {"accepted": best >= self.threshold, "score": best, "threshold": self.threshold,
                "backend": self.backend, "quality": q, "candidate_scores": scores,
                "reason": "fingerprint match" if best >= self.threshold else "fingerprint rejected"}
