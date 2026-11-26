"""BioGuard-X core platform.

The application intentionally contains only the three biometric modalities:
face, fingerprint, and speaker recognition. Operational extras such as LLMs,
automation workflows, speech commands, and the experimental security agent are
not part of the authentication path.
"""
from pathlib import Path
import time
import uuid
import numpy as np

from src.config import BACKUP_DIR, DATASET_DIR, DB_PATH, GALLERY_PATH, GENERATED_REPORTS, FINGERPRINT_EXTS
from src.database.db import Database
from src.enrollment.service import EnrollmentService
from src.face.gallery import FaceGallery
from src.utils.audit import AuditLogger
from src.biometrics import FingerprintMatcher, SpeakerEncoder, MultimodalFusion


class Platform:
    def __init__(self, db_path=DB_PATH, dataset_dir=DATASET_DIR, **_ignored):
        self.db = Database(db_path)
        self.audit = AuditLogger(self.db)
        self.enrollment = EnrollmentService(self.db, self.audit, dataset_dir)
        self.dataset_dir = Path(dataset_dir)
        self.face = FaceGallery(self.dataset_dir, cache_path=GALLERY_PATH if self.dataset_dir == DATASET_DIR else self.dataset_dir.with_name(self.dataset_dir.name + "_face_gallery.joblib"))
        self.fingerprint = FingerprintMatcher(threshold=float(self.db.get_setting("fingerprint_threshold", 40.0)))
        self.speaker = SpeakerEncoder(model_dir=Path("models") / "speaker" / "ecapa-tdnn")
        self.fusion = MultimodalFusion(
            weights=tuple(self.db.get_setting("fusion_weights", [0.40, 0.35, 0.25])),
            thresholds=(self.face.threshold, self.fingerprint.threshold, float(self.db.get_setting("voice_threshold", 0.60))),
            decision_threshold=float(self.db.get_setting("fusion_threshold", 0.70)),
        )

    # ---------- enrolled samples ----------
    def _samples(self, username, modality):
        u = self.enrollment.get_user(username)
        if not u:
            return []
        folder = self.dataset_dir / u["participant_id"] / modality
        exts = {"face": {".jpg", ".jpeg", ".png", ".bmp"}, "fingerprint": FINGERPRINT_EXTS, "voice": {".wav"}}[modality]
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts) if folder.exists() else []

    def face_users(self, ready_only=True):
        status = self.sync_faces()
        return sorted(u for u, s in status.items() if s["ready"] or not ready_only)

    def sync_faces(self):
        return self.face.refresh({u["username"]: u["participant_id"] for u in self.enrollment.list_users() if u["status"] == "active"})

    def voice_ready(self, claimed=None):
        users = [claimed] if claimed else [u["username"] for u in self.enrollment.list_users() if u["status"] == "active"]
        return all(len(self._samples(u, "voice")) >= 2 for u in users)

    def fingerprint_ready(self, claimed=None):
        users = [claimed] if claimed else [u["username"] for u in self.enrollment.list_users() if u["status"] == "active"]
        return all(len(self._samples(u, "fingerprint")) >= 2 for u in users)

    # ---------- single modalities ----------
    def verify_face(self, claimed, img):
        self.sync_faces()
        return self.face.verify(claimed, img)

    def verify_fingerprint(self, claimed, image):
        paths = self._samples(claimed, "fingerprint")
        if not paths:
            return {"accepted": False, "mode": "fingerprint-only", "claimed": claimed,
                    "reason": "no enrolled fingerprint samples"}
        import cv2
        candidates = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
        candidates = [x for x in candidates if x is not None]
        t0 = time.perf_counter()
        r = self.fingerprint.verify(image, candidates)
        r.update({"mode": "fingerprint-only", "claimed": claimed, "ms": (time.perf_counter() - t0) * 1000})
        return r

    def verify_voice(self, claimed, wave):
        paths = self._samples(claimed, "voice")
        if not paths:
            return {"accepted": False, "mode": "voice-only", "claimed": claimed,
                    "reason": "no enrolled voice samples"}
        from src.voice.preprocessing import load_audio
        candidates = [load_audio(str(p)) for p in paths]
        t0 = time.perf_counter()
        r = self.speaker.verify(wave, candidates, threshold=float(self.db.get_setting("voice_threshold", 0.60)))
        r.update({"mode": "voice-only", "claimed": claimed, "ms": (time.perf_counter() - t0) * 1000})
        return r

    # ---------- multimodal ----------
    def verify_multimodal(self, claimed, face_img, fingerprint_img, wave, liveness=None):
        if liveness is not None and not liveness.get("passed", False):
            return {"accepted": False, "mode": "face+fingerprint+voice", "claimed": claimed,
                    "reason": "liveness check failed", "liveness": liveness}
        fr = self.verify_face(claimed, face_img)
        if not fr.get("accepted"):
            return {"accepted": False, "mode": "face+fingerprint+voice", "claimed": claimed,
                    "reason": "face rejected", "face": fr}
        fp = self.verify_fingerprint(claimed, fingerprint_img)
        if not fp.get("accepted"):
            return {"accepted": False, "mode": "face+fingerprint+voice", "claimed": claimed,
                    "reason": "fingerprint rejected", "face": fr, "fingerprint": fp}
        vr = self.verify_voice(claimed, wave)
        if not vr.get("accepted"):
            return {"accepted": False, "mode": "face+fingerprint+voice", "claimed": claimed,
                    "reason": "voice rejected", "face": fr, "fingerprint": fp, "voice": vr}
        fusion = self.fusion.combine(fr["score"], fp["score"], vr["score"])
        return {"accepted": fusion["accepted"], "mode": "face+fingerprint+voice", "claimed": claimed,
                "face": fr, "fingerprint": fp, "voice": vr, **fusion,
                "reason": "three-modality fusion: " + ("accept" if fusion["accepted"] else "reject")}

    # ---------- compatibility authentication entry point ----------
    def authenticate(self, claimed, face_img, wave=None, action="open_dashboard", challenge_passed=False, liveness=None):
        """Single entry point used by older callers; the active app uses explicit modality methods."""
        if self.db.get_setting("require_liveness", False) and not (isinstance(liveness, dict) and liveness.get("passed") is True):
            return {"final_status": "rejected", "reason": "liveness check failed", "verification": {"claimed": claimed}}
        # High/critical actions require a second biometric factor.
        try:
            from src.security.permissions import COMMANDS
            needs_voice = COMMANDS.get(action, {}).get("sensitivity") in ("high", "critical")
        except Exception:
            needs_voice = False
        fr = self.verify_face(claimed, face_img)
        if not fr.get("accepted"):
            self.record_attempt(claimed, "face-only", False, face_score=fr.get("score"), reason=fr.get("reason", "face rejected"))
            return {"final_status": "rejected", "reason": fr.get("reason", "face rejected"), "verification": fr}
        if needs_voice:
            if wave is None:
                return {"final_status": "rejected", "reason": "this action requires face + voice: a voice sample is needed", "verification": fr}
            if not self.voice_ready(claimed):
                return {"final_status": "rejected", "reason": "voice model not trained / no enrolled voice samples", "verification": fr}
            vr = self.verify_voice(claimed, wave)
            if not vr.get("accepted"):
                self.record_attempt(claimed, "face+voice", False, face_score=fr.get("score"), voice_score=vr.get("score"), reason="voice rejected")
                return {"final_status": "rejected", "reason": "voice rejected", "verification": {"face": fr, "voice": vr}}
            self.record_attempt(claimed, "face+voice", True, face_score=fr.get("score"), voice_score=vr.get("score"), reason="face and voice accepted")
            return {"final_status": "authenticated", "reason": "verified", "verification": {"face": fr, "voice": vr}}
        self.record_attempt(claimed, "face-only", True, face_score=fr.get("score"), reason="face accepted")
        return {"final_status": "authenticated", "reason": "verified", "verification": fr}

    # ---------- audit ----------
    def record_attempt(self, claimed, mode, accepted, stage=1, face_score=None, voice_score=None,
                       fused_score=None, reason="", face_detected=None, fingerprint_score=None, vresults=None):
        aid = self.db.insert("auth_attempts", {
            "claimed": claimed, "mode": mode, "accepted": int(bool(accepted)), "stage": stage,
            "face_score": face_score, "voice_score": voice_score, "fused_score": fused_score,
            "risk_score": None, "risk_level": None, "action": "authentication", "reason": reason,
            "face_detected": None if face_detected is None else int(face_detected), "created_at": __import__("src.utils.timeutil", fromlist=["iso"]).iso(),
        })
        for v in vresults or []:
            self.db.insert("verification_results", {"attempt_id": aid, **v, "created_at": __import__("src.utils.timeutil", fromlist=["iso"]).iso()})
        self.audit.log(claimed, "auth_attempt", f"attempt:{aid}", "accepted" if accepted else "rejected",
                       {"mode": mode, "reason": reason})
        return aid

    def dashboard_stats(self):
        d = self.db
        return {"users": d.count("users"), "attempts": d.count("auth_attempts"),
                "successes": d.count("auth_attempts", "accepted=1"),
                "failures": d.count("auth_attempts", "accepted=0")}

    def clear_verification_data(self, actor="local_operator"):
        """Delete verification history while preserving enrollment and the audit trail."""
        attempts = self.db.count("auth_attempts")
        results = self.db.count("verification_results")
        # Child result rows are removed first; enrollment and audit tables are untouched.
        self.db.execute("DELETE FROM verification_results")
        self.db.execute("DELETE FROM auth_attempts")
        self.audit.log(
            actor,
            "verification_data_cleared",
            "auth_attempts+verification_results",
            "success",
            {"deleted_attempts": attempts, "deleted_results": results},
        )
        return {"attempts": attempts, "results": results}

    def authentication_capabilities(self):
        return {
            "face": True,
            "fingerprint": self.fingerprint.backend or "afis (install required)",
            "voice": self.speaker.backend or "speechbrain ECAPA-TDNN (download on first use)",
            "fusion": True,
        }
