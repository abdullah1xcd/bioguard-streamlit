"""Run manifest: everything needed to trace a result back to
   dataset + preprocessing + features + model + parameters + threshold + evaluation split."""
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dataset_fingerprint(files) -> dict:
    """Content hash of every dataset file used (sorted by relative path) + one combined digest."""
    entries = {}
    for f in sorted(files, key=str):
        entries[str(f)] = sha256_file(f)
    combined = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"n_files": len(entries), "combined_sha256": combined, "files": entries}


def lib_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "sklearn", "skimage", "cv2", "librosa", "pandas", "matplotlib", "joblib"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=2))
