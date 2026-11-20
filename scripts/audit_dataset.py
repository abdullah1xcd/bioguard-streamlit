"""Dataset quality audit (read-only).   python -m scripts.audit_dataset [dataset_dir]
Reports per participant/modality: counts, sessions, image sizes, quality-gate rejections (with reasons), metric
percentiles (use TRAIN-split percentiles to tune config.QUALITY - never test results), near-duplicates inside a
person, and identical/near-identical images shared between DIFFERENT people (possible label errors)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AUDIO_EXTS, DATASET_DIR, EVAL_CFG, IMAGE_EXTS, REPORTS_DIR, SESSION_PREFIX
from src.face.preprocessing import load_color
from src.face.quality import assess_face_quality, dhash, hamming
from src.voice.preprocessing import assess_audio_quality, load_audio


def main(root):
    root = Path(root); rows, hashes = [], {}
    for pdir in sorted(p for p in root.glob("*") if p.is_dir()):
        for sub, exts in (("face", IMAGE_EXTS), ("voice", AUDIO_EXTS)):
            d = pdir / sub
            files = sorted(f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in exts) if d.is_dir() else []
            for f in files:
                sess = next((x for x in f.relative_to(d).parts if x.startswith(SESSION_PREFIX)), None)
                r = {"pid": pdir.name, "modality": sub, "file": str(f), "session": sess}
                try:
                    if sub == "face":
                        img = load_color(f); q = assess_face_quality(img); hashes[str(f)] = (pdir.name, dhash(img))
                        r.update(h=img.shape[0], w=img.shape[1])
                    else:
                        q = assess_audio_quality(load_audio(f))
                except Exception as e:
                    q = {"ok": False, "reasons": [f"unreadable: {e}"], "metrics": {}}
                r.update(ok=q["ok"], reasons=";".join(q["reasons"]), **{f"m_{k}": v for k, v in q["metrics"].items()})
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no data found in", root); return
    summ = df.groupby(["pid", "modality"]).agg(n=("file", "size"), sessions=("session", lambda s: s.nunique()), rejected=("ok", lambda s: int((~s).sum())))
    print(summ.to_string())
    print("\nrejection reasons:", df.loc[~df["ok"], "reasons"].str.split(";").explode().value_counts().to_dict())
    num = [c for c in df.columns if c.startswith("m_")]
    print("\nmetric percentiles (5/50/95):\n", df[num].quantile([.05, .5, .95]).round(2).to_string())
    items = list(hashes.items()); within, cross = [], []
    thr = EVAL_CFG["dup_hamming"]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (fa, (pa, ha)), (fb, (pb, hb)) = items[i], items[j]
            if hamming(ha, hb) <= thr:
                (within if pa == pb else cross).append((fa, fb))
    print(f"\nnear-duplicate image pairs inside the same person: {len(within)} | across DIFFERENT people (label-error suspects): {len(cross)}")
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "dataset_audit.csv", index=False)
    (REPORTS_DIR / "dataset_audit_duplicates.json").write_text(json.dumps({"within_person": within, "cross_person": cross}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR)
