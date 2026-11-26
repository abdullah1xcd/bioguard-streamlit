"""Face gallery = the enrolled photos of every user, embedded once and cached.  This replaces the old train-a-model step:
enrol photos -> they are embedded (ArcFace) -> a probe is accepted when its cosine similarity to the user's photos >= threshold.
Nothing here needs `scripts.run_all`.  The cache only stores embeddings (not photos); deleting a user's folder removes them on the next refresh."""
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
from src.config import FACE_CFG, GALLERY_PATH, IMAGE_EXTS
from src.face.engine import FaceEngine, FaceNotFound, l2n
from src.face.quality import assess_face_quality


# Bump when face detection/quality logic changes so cached sample decisions are recomputed.
FACE_DETECTION_CACHE_VERSION = 2


def quality_cfg(cfg=FACE_CFG):
    return {"min_face_px": cfg["min_face_px"], "min_sharpness": cfg["min_sharpness"], "brightness": cfg["brightness"],
            "min_contrast": 15.0, "max_abs_yaw": cfg["max_abs_yaw"], "min_image_px": 80}


def xyxy_to_xywh(b):
    return (b[0], b[1], b[2] - b[0], b[3] - b[1])


class FaceGallery:
    def __init__(self, dataset_dir, engine=None, cache_path=GALLERY_PATH, cfg=None):
        self.base, self.engine, self.cache_path = Path(dataset_dir), engine or FaceEngine(), Path(cache_path)
        self.cfg = {**FACE_CFG, **(cfg or {})}
        self._cache = None                      # {"files": {path: entry}, "threshold": float|None}
        self._emb = {}                          # username -> (n, d) clean embeddings

    # ------------------------------------------------------------------ cache
    def _load(self):
        if self._cache is None:
            try:
                self._cache = joblib.load(self.cache_path)
            except Exception:
                self._cache = {"files": {}, "threshold": None}
        return self._cache

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._cache, self.cache_path)

    # ------------------------------------------------------------------ enrolled photos
    def photos(self, participant_id):
        folder = self.base / participant_id / "face"
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if folder.exists() else []

    def assess(self, img):
        """-> (embedding | None, info).  info = {face_detected, ok, reasons, box, det_score, detector}.  Never raises for 'no face'."""
        try:
            d = self.engine.detect(img)
        except ValueError as e:
            return None, {"face_detected": False, "ok": False, "reasons": [f"invalid_image: {e}"], "box": None}
        if d is None:
            return None, {"face_detected": False, "ok": False, "reasons": ["no_face_detected"], "box": None}
        q = assess_face_quality(img, box=xyxy_to_xywh(d["box"]), landmarks=d["landmarks"], cfg=quality_cfg(self.cfg))
        emb, _ = self.engine.embed(img, detection=d)
        return emb, {"face_detected": True, "ok": q["ok"], "reasons": q["reasons"], "metrics": q["metrics"], "box": d["box"],
                     "det_score": d["det_score"], "detector": d["detector"]}

    def refresh(self, users):
        """users: {username: participant_id}.  Embeds only new/changed photos, then rebuilds each user's clean template set."""
        cache = self._load(); files = cache["files"]; changed = False; seen = set()
        for uname, pid in users.items():
            for p in self.photos(pid):
                key = str(p); seen.add(key); st = p.stat(); sig = (st.st_mtime_ns, st.st_size, FACE_DETECTION_CACHE_VERSION)
                if key in files and files[key]["sig"] == sig:
                    continue
                img = cv2.imread(key, cv2.IMREAD_COLOR)
                if img is None:
                    files[key] = {"sig": sig, "emb": None, "ok": False, "reasons": ["unreadable_image"]}
                else:
                    emb, info = self.assess(img)
                    files[key] = {"sig": sig, "emb": emb, "ok": bool(info["ok"] and emb is not None), "reasons": info["reasons"]}
                changed = True
        for key in [k for k in files if k not in seen]:      # photo or user deleted
            del files[key]; changed = True
        if changed:
            self._save()
        self._emb = {}
        for uname, pid in users.items():
            E = [files[str(p)]["emb"] for p in self.photos(pid) if files[str(p)]["ok"]]
            if E:
                self._emb[uname] = self._drop_outliers(np.vstack(E))
        return self.summary(users)

    def _drop_outliers(self, E):
        """A mislabeled photo (someone else's face) would poison the template: remove photos far from the user's median embedding."""
        if len(E) < 4:
            return E
        med = l2n(np.median(E, axis=0)); keep = (E @ med) >= self.cfg["outlier_cos"]
        return E[keep] if keep.sum() >= 3 else E

    def summary(self, users):
        files = self._load()["files"]; out = {}
        for uname, pid in users.items():
            ps = [files[str(p)] for p in self.photos(pid) if str(p) in files]
            bad = {}
            for e in ps:
                if not e["ok"]:
                    for r in e["reasons"] or ["rejected"]:
                        bad[r] = bad.get(r, 0) + 1
            out[uname] = {"photos": len(ps), "usable": len(self._emb.get(uname, [])), "rejected_reasons": bad,
                          "ready": len(self._emb.get(uname, [])) >= self.cfg["min_enroll_images"]}
        return out

    def ready_users(self):
        return sorted(u for u, E in self._emb.items() if len(E) >= self.cfg["min_enroll_images"])

    # ------------------------------------------------------------------ scoring
    @property
    def threshold(self):
        t = self._load().get("threshold")
        return float(t if t is not None else self.cfg["threshold"])

    def set_threshold(self, t):
        self._load()["threshold"] = None if t is None else float(t); self._save()

    def score(self, username, probe):
        E = self._emb.get(username)
        if E is None or len(E) == 0:
            raise KeyError(username)
        sims = np.sort(E @ l2n(probe))[::-1]
        return float(sims[:max(1, min(self.cfg["top_k"], len(sims)))].mean())

    def verify(self, username, img, enforce_quality=True):
        t0 = time.perf_counter()
        base = {"mode": "face-only", "claimed": username, "threshold": self.threshold, "backend": self.engine.backend}
        n = len(self._emb.get(username, []))
        if n < self.cfg["min_enroll_images"]:
            return {**base, "face_detected": None, "accepted": False, "reason": f"user has only {n} usable enrolled face photos "
                    f"(need {self.cfg['min_enroll_images']}+). Enrol more photos on the User Enrollment page.", "quality": {"ok": True, "reasons": []}}
        probe, info = self.assess(img)
        q = {"ok": info["ok"], "reasons": info["reasons"], **info.get("metrics", {})}
        if probe is None or (enforce_quality and not info["ok"]):
            return {**base, "face_detected": info["face_detected"], "quality": q, "box": info["box"], "accepted": False,
                    "reason": "poor sample: " + ", ".join(info["reasons"]), "ms": (time.perf_counter() - t0) * 1000}
        s = self.score(username, probe)
        return {**base, "face_detected": True, "quality": q, "box": info["box"], "detector": info["detector"], "score": s,
                "accepted": s >= base["threshold"], "enrolled_photos": n, "ms": (time.perf_counter() - t0) * 1000}

    def identify(self, img):
        """1:N search (optional helper): best matching enrolled user and score."""
        probe, info = self.assess(img)
        if probe is None:
            return None, None
        best = max(((u, self.score(u, probe)) for u in self._emb), key=lambda t: t[1], default=(None, None))
        return best

    # ------------------------------------------------------------------ calibration (optional, needs >= 2 enrolled users)
    def calibrate(self, save=True):
        users = {u: E for u, E in self._emb.items() if len(E) >= self.cfg["min_enroll_images"]}
        if len(users) < 2:
            return {"calibrated": False, "reason": "need at least 2 enrolled users with 3+ usable photos", "threshold": self.threshold}
        k = self.cfg["top_k"]; gen, imp = [], []
        for u, E in users.items():
            for i in range(len(E)):
                rest = np.delete(E, i, axis=0); gen.append(float(np.sort(rest @ E[i])[::-1][:k].mean()))
                for v, F in users.items():
                    if v != u:
                        imp.append(float(np.sort(F @ E[i])[::-1][:k].mean()))
        gen, imp = np.array(gen), np.array(imp)
        thr = float(np.clip(max(np.percentile(imp, 99), np.max(imp) if len(imp) < 100 else 0) + 0.02, self.cfg["min_threshold"], self.cfg["max_threshold"]))
        res = {"calibrated": True, "threshold": thr, "genuine_pairs": len(gen), "impostor_pairs": len(imp),
               "FAR": float((imp >= thr).mean()), "FRR": float((gen < thr).mean()),
               "genuine_mean": float(gen.mean()), "impostor_mean": float(imp.mean())}
        res["note"] = ("classes overlap - add better photos" if res["FRR"] > 0.2 else "ok") + "; estimated on enrolment photos only, not an independent test"
        if save:
            self.set_threshold(thr)
        return res
