import json
import re
import shutil
import uuid
from pathlib import Path
from src.config import AUDIO_EXTS, DATASET_DIR, IMAGE_EXTS
from src.security.permissions import ROLES
from src.utils.timeutil import iso

USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
EXTS = {"face": IMAGE_EXTS, "voice": AUDIO_EXTS, "fingerprint": IMAGE_EXTS, "commands": AUDIO_EXTS}
PSEUDONYMISED = ("auth_attempts", "challenges", "speech_commands", "security_events", "agent_analyses", "scrutiny")


class ConsentRequired(Exception):
    pass


class EnrollmentService:
    """Enrolment with informed consent; biometric files stay on local disk (datasets/custom/<participant>/...)."""
    def __init__(self, db, audit, dataset_dir=DATASET_DIR):
        self.db, self.audit, self.base = db, audit, Path(dataset_dir)

    def enroll(self, username, role="standard_user", consent=False, actor="operator"):
        if not consent:
            self.audit.log(actor, "enroll_denied", f"user:{username}", "denied", {"reason": "no consent"})
            raise ConsentRequired("informed consent is required before any biometric data is stored")
        if not USERNAME_RE.match(username or ""):
            raise ValueError("username must be 2-32 chars: letters, digits, _ or -")
        if role not in ROLES:
            raise ValueError(f"unknown role {role}")
        if self.get_user(username):
            raise ValueError(f"user '{username}' already exists")
        n = (self.db.fetchone("SELECT COALESCE(MAX(id),0)+1 AS n FROM users")["n"])
        while (self.base / f"participant_{n:03d}").exists():
            n += 1
        pid = f"participant_{n:03d}"; folder = self.base / pid
        for sub in EXTS:
            (folder / sub).mkdir(parents=True, exist_ok=True)
        uid = self.db.insert("users", {"username": username, "participant_id": pid, "role": role, "consent": 1, "consent_at": iso(),
                                       "status": "active", "created_at": iso(), "updated_at": iso()})
        (folder / "metadata.json").write_text(json.dumps({"username": username, "role": role, "consent": True, "consent_at": iso()}, indent=2))
        self.audit.log(actor, "user_enrolled", f"user:{username}", details={"participant": pid, "role": role})
        return self.get_user(username)

    def get_user(self, key):
        if isinstance(key, int):
            return self.db.fetchone("SELECT * FROM users WHERE id=?", (key,))
        return self.db.fetchone("SELECT * FROM users WHERE username=? OR participant_id=?", (key, key))

    def list_users(self):
        return self.db.fetch("SELECT * FROM users ORDER BY id")

    def add_samples(self, user_id, modality, files, command=None, actor="operator", session=None):
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(user_id)
        if modality not in EXTS:
            raise ValueError(f"unknown modality {modality}")
        folder = self.base / user["participant_id"] / modality
        if modality == "commands":
            if not command or not re.match(r"^[a-z_]{2,40}$", command):
                raise ValueError("a valid command key is required for command samples")
            folder = folder / command
        if session is not None:                       # v2: recording session -> leakage-safe train/val/test split by session
            if not re.match(r"^[0-9]{1,3}$", str(session)):
                raise ValueError("session must be a number 1-999")
            folder = folder / f"session_{int(session)}"
        folder.mkdir(parents=True, exist_ok=True); saved = 0
        for name, data in files:
            ext = Path(name).suffix.lower()
            if ext not in EXTS[modality]:
                raise ValueError(f"{name}: extension {ext} not allowed for {modality}")
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name)
            path = folder / f"{uuid.uuid4().hex[:8]}_{safe}"
            path.write_bytes(data)
            self.db.insert("enrollment_samples", {"user_id": user["id"], "modality": modality, "path": str(path), "created_at": iso()})
            saved += 1
        self.audit.log(actor, "samples_added", f"user:{user['username']}", details={"modality": modality, "count": saved})
        return saved

    def update_user(self, user_id, role=None, status=None, actor="operator"):
        d = {"updated_at": iso()}
        if role:
            if role not in ROLES: raise ValueError(role)
            d["role"] = role
        if status:
            if status not in ("active", "suspended"): raise ValueError(status)
            d["status"] = status
        self.db.update("users", user_id, d)
        self.audit.log(actor, "user_updated", f"user:{user_id}", details=d)
        return self.get_user(user_id)

    def delete_user(self, user_id, actor="operator"):
        """Deletes biometric files + enrolment records; pseudonymises history. Models trained earlier may still encode the user -> retrain."""
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(user_id)
        folder = (self.base / user["participant_id"]).resolve()
        if self.base.resolve() not in folder.parents:
            raise RuntimeError("refusing to delete outside the dataset directory")
        n_files = sum(1 for f in folder.rglob("*") if f.is_file()) if folder.exists() else 0
        if folder.exists():
            shutil.rmtree(folder)
        pseudo = f"deleted_user_{user['id']}"
        self.db.execute("DELETE FROM enrollment_samples WHERE user_id=?", (user["id"],))
        for t in PSEUDONYMISED:
            self.db.execute(f"UPDATE {t} SET claimed=? WHERE claimed=?", (pseudo, user["username"]))
        self.db.execute("UPDATE audit_logs SET target=REPLACE(target,?,?) WHERE target LIKE ?", (user["username"], pseudo, f"%{user['username']}%"))
        self.db.execute("DELETE FROM users WHERE id=?", (user["id"],))
        self.audit.log(actor, "user_deleted", pseudo, details={"files_deleted": n_files, "models_need_retraining": True})
        return {"files_deleted": n_files, "pseudonym": pseudo, "models_need_retraining": True}
