import numpy as np
import librosa
from src.config import SAMPLE_RATE


def load_audio(src) -> np.ndarray:
    """src: path or file-like WAV. Resampled to 16 kHz, mono."""
    y, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
    return y


def preprocess_audio(y: np.ndarray, min_dur: float = 0.5) -> np.ndarray:
    if y.size == 0:
        raise ValueError("empty audio")
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        raise ValueError("silent audio")
    y = y / peak                                   # peak normalisation
    yt, _ = librosa.effects.trim(y, top_db=25)     # silence trimming
    if len(yt) / SAMPLE_RATE < min_dur:
        raise ValueError(f"speech too short ({len(yt)/SAMPLE_RATE:.2f}s < {min_dur}s)")
    return yt


def audio_quality(y: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(y ** 2)))
    return {"duration_s": len(y) / SAMPLE_RATE, "rms": rms,
            "clipping_ratio": float(np.mean(np.abs(y) >= 0.999))}


# ----------------------------------------------------------------------------------------------
# v2 signal processing (baseline functions above are unchanged and remain the "v1" pipeline)
# ----------------------------------------------------------------------------------------------
from src.config import QUALITY, VOICE_CFG


def vad_concat(y: np.ndarray, top_db: float = None, min_seg_s: float = None) -> np.ndarray:
    """Energy VAD: keep every non-silent segment (also INTERNAL pauses are removed, unlike librosa.effects.trim)."""
    top_db = VOICE_CFG["vad_top_db"] if top_db is None else top_db
    min_seg = int((VOICE_CFG["min_segment_s"] if min_seg_s is None else min_seg_s) * SAMPLE_RATE)
    iv = librosa.effects.split(y, top_db=top_db)
    segs = [y[s:e] for s, e in iv if e - s >= min_seg]
    return np.concatenate(segs) if segs else y[:0]


def denoise_spectral(y: np.ndarray, alpha: float = None, floor: float = None) -> np.ndarray:
    """Spectral subtraction with a noise profile taken from the quietest 10 % of STFT frames."""
    alpha = VOICE_CFG["denoise_alpha"] if alpha is None else alpha
    floor = VOICE_CFG["denoise_floor"] if floor is None else floor
    if len(y) < 1024:
        return y
    S = librosa.stft(y, n_fft=512, hop_length=128)
    mag, ph = np.abs(S), np.angle(S)
    energy = mag.sum(axis=0)
    k = max(1, int(0.1 * mag.shape[1]))
    noise = mag[:, np.argsort(energy)[:k]].mean(axis=1, keepdims=True)
    clean = np.maximum(mag - alpha * noise, floor * mag)
    out = librosa.istft(clean * np.exp(1j * ph), hop_length=128, length=len(y))
    return out.astype(np.float32)


def preprocess_audio_v2(y: np.ndarray, min_dur: float = 0.5, denoise: bool = False) -> np.ndarray:
    """DC removal -> (optional) spectral-gate denoise -> peak normalisation -> VAD (drops internal silence)."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("empty audio")
    y = y - float(np.mean(y))
    if float(np.max(np.abs(y))) < 1e-4:
        raise ValueError("silent audio")
    if denoise:
        y = denoise_spectral(y)
    y = y / max(float(np.max(np.abs(y))), 1e-9)
    yv = vad_concat(y)
    if len(yv) / SAMPLE_RATE < min_dur:
        raise ValueError(f"speech too short ({len(yv)/SAMPLE_RATE:.2f}s < {min_dur}s)")
    return yv


def assess_audio_quality(y_raw: np.ndarray, cfg=None) -> dict:
    """Quality gate on the RAW recording. SNR is a crude dynamic-range estimate (90th vs 10th percentile frame RMS)."""
    cfg = cfg or QUALITY["voice"]
    y = np.asarray(y_raw, dtype=np.float32)
    reasons, m = [], {}
    if y.size == 0:
        return {"ok": False, "reasons": ["empty_audio"], "metrics": {}}
    m["rms"] = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    m["clipping_ratio"] = float(np.mean(np.abs(y) >= 0.999 * max(peak, 1e-9))) if peak >= 0.999 else 0.0
    if peak < 1e-4 or m["rms"] < cfg["min_rms"]:
        return {"ok": False, "reasons": ["silent_or_too_quiet"], "metrics": m}
    fr = librosa.feature.rms(y=y / peak, frame_length=400, hop_length=160)[0]
    p10, p90 = np.percentile(fr, 10), np.percentile(fr, 90)
    m["snr_db"] = float(20 * np.log10((p90 + 1e-9) / (p10 + 1e-9)))
    sp = vad_concat(y / peak)
    m["speech_s"] = float(len(sp) / SAMPLE_RATE)
    if m["speech_s"] < cfg["min_speech_s"]:
        reasons.append("speech_too_short")
    if m["clipping_ratio"] > cfg["max_clip_ratio"]:
        reasons.append("clipped")
    if m["snr_db"] < cfg["min_snr_db"]:
        reasons.append("low_snr")
    return {"ok": not reasons, "reasons": reasons, "metrics": m}
