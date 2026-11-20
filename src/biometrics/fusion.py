"""Simple, auditable score-level fusion for three biometric modalities."""
from __future__ import annotations
import numpy as np


class MultimodalFusion:
    def __init__(self, weights=(0.4, 0.35, 0.25), thresholds=(0.45, 40.0, 0.60), decision_threshold=0.70):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.weights = self.weights / self.weights.sum()
        self.thresholds = thresholds
        self.decision_threshold = float(decision_threshold)

    @staticmethod
    def _face_voice_norm(score, threshold):
        # Convert cosine-like scores into a bounded confidence around the calibrated threshold.
        return float(np.clip((float(score) - threshold + 0.15) / 0.30, 0.0, 1.0))

    @staticmethod
    def _fp_norm(score, threshold):
        return float(np.clip(float(score) / max(2.0 * threshold, 1e-6), 0.0, 1.0))

    def combine(self, face, fingerprint, voice):
        parts = np.array([
            self._face_voice_norm(face, self.thresholds[0]),
            self._fp_norm(fingerprint, self.thresholds[1]),
            self._face_voice_norm(voice, self.thresholds[2]),
        ], dtype=np.float32)
        fused = float(np.dot(self.weights, parts))
        return {"normalized_scores": parts.tolist(), "weights": self.weights.tolist(),
                "fused_score": fused, "threshold": self.decision_threshold,
                "accepted": fused >= self.decision_threshold}
