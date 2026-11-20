"""EXPERIMENTAL spoof detectors (traditional ML). Need labelled data: datasets/spoof/{genuine_face,photo_attack,genuine_voice,replay_voice}.
Results on small data are NOT evidence of production-grade anti-spoofing."""
import numpy as np
import cv2


def photo_features(gray):
    g = cv2.resize(gray, (128, 128)).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))); c = 64
    yy, xx = np.ogrid[:128, :128]; r = np.hypot(yy - c, xx - c)
    hf = F[r > 32].sum() / (F.sum() + 1e-9)
    return np.array([np.log1p(lap), hf, g.std(), g.mean() / 255])


def replay_features(y, sr=16000):
    import librosa
    S = np.abs(librosa.stft(y)) ** 2
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr); flat = librosa.feature.spectral_flatness(S=S)
    freqs = librosa.fft_frequencies(sr=sr); hb = S[freqs > 4000].sum() / (S.sum() + 1e-9)
    return np.array([roll.mean(), roll.std(), flat.mean(), flat.std(), hb])
