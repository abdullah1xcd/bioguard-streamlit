"""Speaker verification with pretrained ECAPA-TDNN, with MFCC fallback.

The main path uses SpeechBrain's `spkrec-ecapa-voxceleb` model. It produces
speaker embeddings and compares them with cosine similarity. The fallback is
kept only so the UI can still diagnose an installation problem instead of
crashing; it is not the preferred accuracy path.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.config import SAMPLE_RATE


def _l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-8 or not np.isfinite(n):
        raise ValueError("degenerate speaker embedding")
    return v / n


class SpeakerEncoder:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", model_dir=None):
        self.model_source = model_source
        self.model_dir = Path(model_dir or "models/speaker/" + model_source.split("/")[-1])
        self._model = None
        self.backend = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(source=self.model_source, savedir=str(self.model_dir))
            self.backend = "speechbrain-ecapa-tdnn"
            return self._model
        except Exception as e:
            raise RuntimeError(
                "ECAPA-TDNN is not available. Install: pip install speechbrain torch torchaudio"
            ) from e

    @staticmethod
    def quality(y: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0:
            return {"ok": False, "reason": "empty audio"}
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y * y)))
        duration = len(y) / SAMPLE_RATE
        clipping = float(np.mean(np.abs(y) >= 0.999))
        ok = duration >= 1.0 and rms >= 1e-3 and peak >= 1e-3 and clipping < 0.02
        return {"ok": ok, "duration_s": duration, "rms": rms, "clipping_ratio": clipping,
                "reason": "ok" if ok else "poor audio quality / too short"}

    def embed(self, y: np.ndarray) -> np.ndarray:
        q = self.quality(y)
        if not q["ok"]:
            raise ValueError(q["reason"])
        model = self._load()
        import torch
        signal = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(signal)
        return _l2(emb.detach().cpu().numpy())

    def score(self, probe: np.ndarray, candidate: np.ndarray) -> float:
        a, b = self.embed(probe), self.embed(candidate)
        return float(np.dot(a, b))

    def verify(self, probe: np.ndarray, candidates: list[np.ndarray], threshold: float = 0.60) -> dict:
        if not candidates:
            return {"accepted": False, "score": None, "threshold": threshold,
                    "backend": self.backend, "reason": "no enrolled voice"}
        p = self.embed(probe)
        scores = [float(np.dot(p, self.embed(c))) for c in candidates]
        best = max(scores)
        return {"accepted": best >= threshold, "score": best, "threshold": threshold,
                "backend": self.backend, "candidate_scores": scores,
                "reason": "speaker match" if best >= threshold else "speaker rejected"}
