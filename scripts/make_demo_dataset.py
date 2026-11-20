"""SYNTHETIC smoke-test data (NOT real biometrics) -> datasets/demo. Only proves the pipeline runs end to end.
Never report results obtained from this data.
   python -m scripts.make_demo_dataset [--sessions 3] && python -m scripts.run_evaluation datasets/demo --no-quality-gate"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.io import wavfile
from src.config import ROOT


def main(out=ROOT / "datasets" / "demo", people=5, n=16, sessions=0):
    rng = np.random.default_rng(0); out = Path(out)
    for p in range(people):
        d = out / f"participant_{p + 1:03d}"
        base = cv2.GaussianBlur(rng.integers(0, 255, (160, 160)).astype(np.float32), (0, 0), 6); base = cv2.normalize(base, None, 40, 220, cv2.NORM_MINMAX)
        f0 = 110 + 25 * p; harm = rng.uniform(0.3, 1, 5)
        for i in range(n):
            sub = f"session_{i % sessions + 1}" if sessions else ""
            (d / "face" / sub).mkdir(parents=True, exist_ok=True); (d / "voice" / sub).mkdir(parents=True, exist_ok=True)
            img = np.clip(base * rng.uniform(.8, 1.2) + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / "face" / sub / f"{i:02d}.jpg"), img)
            t = np.arange(int(16000 * 1.5)) / 16000; env = np.sin(np.pi * t / 1.5) ** 0.5
            y = sum(h * np.sin(2 * np.pi * f0 * (k + 1) * (1 + rng.normal(0, .01)) * t) for k, h in enumerate(harm)) * env
            wavfile.write(str(d / "voice" / sub / f"{i:02d}.wav"), 16000, (0.3 * y / np.abs(y).max() * 32767 + rng.normal(0, 60, y.size)).astype(np.int16))
    print("DEMO (synthetic) dataset written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", type=int, default=0); ap.add_argument("--out", default=str(ROOT / "datasets" / "demo"))
    ap.add_argument("--people", type=int, default=5); ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args(); main(a.out, a.people, a.n, a.sessions)
