"""Spoken challenge. IMPORTANT LIMITS: no speech recognition (deep models are prohibited), so PHRASE MATCHING IS NOT DONE.
We only check (a) speaker verification (separate) and (b) that the number of spoken segments equals the number of
requested digits (a weak heuristic). Replay/TTS attacks can defeat it. EXPERIMENTAL - not production liveness."""
import secrets
import numpy as np
from src.config import SAMPLE_RATE


def new_phrase(n=4):
    return " ".join(str(secrets.randbelow(10)) for _ in range(n))


def count_segments(y, top_db=30, min_gap_s=0.15):
    import librosa
    iv = librosa.effects.split(y, top_db=top_db)
    merged = []
    for s, e in iv:
        if merged and (s - merged[-1][1]) / SAMPLE_RATE < min_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return len([m for m in merged if (m[1] - m[0]) / SAMPLE_RATE >= 0.08])


def evaluate_spoken_challenge(wave, expected_phrase):
    words = expected_phrase.split()
    seg = count_segments(np.asarray(wave, dtype=np.float32))
    return {"segments": seg, "expected_words": len(words), "completed": seg == len(words),
            "note": "segment-count heuristic only; phrase content NOT verified"}
