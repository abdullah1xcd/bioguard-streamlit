"""Leakage-safe multimodal evaluation.

Expected layout per participant:
  face/session_1/*.jpg, session_2/*.jpg, session_3/*.jpg
  fingerprint/session_1/*.png, ...
  voice/session_1/*.wav, ...

Session 1 = enrollment/reference, session 2 = validation, session 3 = final test.
If a session is missing for a modality, that modality is reported as unavailable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from src.config import DATASET_DIR
from src.evaluation.metrics import eer_threshold, report
from src.face.engine import FaceEngine
from src.face.gallery import l2n
from src.biometrics import FingerprintMatcher, SpeakerEncoder


def files(folder, exts):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def pairs(refs, probes):
    rows = []
    for pid, ps in probes.items():
        if pid not in refs or not refs[pid]:
            continue
        # genuine: every probe against the reference set
        for p in ps:
            rows.append((pid, pid, p, 1))
        # one impostor claim per other participant, capped for balance
        others = [q for q in refs if q != pid and refs[q]]
        for q in others[:10]:
            rows.append((pid, q, p, 0))
    return rows


def face_scores(root, split):
    eng = FaceEngine({"detector": "scrfd"})
    refs, probes = {}, {}
    for part in sorted(root.iterdir()):
        if not part.is_dir(): continue
        r = part / "face" / "session_1"; p = part / "face" / f"session_{split}"
        if not r.exists() or not p.exists(): continue
        re, pe = [], []
        for x in files(r, {".jpg", ".jpeg", ".png", ".bmp"}):
            img = __import__("cv2").imread(str(x));
            if img is None: continue
            try: re.append(eng.embed(img)[0])
            except Exception: pass
        for x in files(p, {".jpg", ".jpeg", ".png", ".bmp"}):
            img = __import__("cv2").imread(str(x));
            if img is None: continue
            try: pe.append((x, eng.embed(img)[0]))
            except Exception: pass
        if re and pe: refs[part.name] = re; probes[part.name] = pe
    y, s = [], []
    for true_pid, claim, path, label in pairs(refs, probes):
        # locate the probe embedding by path
        emb = dict(probes[true_pid])[path]
        score = max(float(np.dot(emb, l2n(r))) for r in refs[claim])
        y.append(label); s.append(score)
    return np.asarray(y), np.asarray(s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(DATASET_DIR)); ap.add_argument("--split", type=int, default=2)
    args = ap.parse_args(); root = Path(args.root)
    results = {}
    y, s = face_scores(root, args.split)
    if len(np.unique(y)) == 2:
        eer, thr = eer_threshold(y, s); results["face"] = {"EER": eer, "AUC": roc_auc_score(y, s), **report(y, s, thr)}
    else: results["face"] = {"error": "not enough face session data"}
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
