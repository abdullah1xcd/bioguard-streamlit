"""python -m scripts.validate_dataset  -> dataset manifest + statistics (reports/dataset_manifest.json)"""
import json
import sys
from pathlib import Path
from src.config import AUDIO_EXTS, DATASET_DIR, IMAGE_EXTS, REPORTS_DIR


def main(root):
    root = Path(root); man, bad = {}, []
    import cv2
    for p in sorted(root.glob("participant_*")):
        meta = json.loads((p / "metadata.json").read_text()) if (p / "metadata.json").exists() else {}
        n = {"face": 0, "voice": 0, "fingerprint": 0, "commands": 0}
        for sub in n:
            for f in (p / sub).rglob("*") if (p / sub).exists() else []:
                if f.suffix.lower() in IMAGE_EXTS | AUDIO_EXTS:
                    n[sub] += 1
                    if f.suffix.lower() in IMAGE_EXTS and cv2.imread(str(f)) is None: bad.append(str(f))
        man[p.name] = {"counts": n, "consent": meta.get("consent"), "source": "custom"}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dataset_manifest.json").write_text(json.dumps({"participants": man, "corrupt_files": bad}, indent=2))
    print(json.dumps({"participants": len(man), "corrupt_files": len(bad), "per_participant": man}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR)
