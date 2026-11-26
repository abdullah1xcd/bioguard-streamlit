from src.database.db import Database
from src.platform import Platform
from src.utils.audit import AuditLogger


def test_clear_verification_data_preserves_enrollment_and_audit(tmp_path):
    db = Database(tmp_path / "bioguard.sqlite")
    db.insert("users", {
        "username": "alice",
        "participant_id": "participant-alice",
        "role": "user",
        "consent": 1,
        "status": "active",
    })
    db.insert("auth_attempts", {
        "claimed": "alice",
        "mode": "face-only",
        "accepted": 0,
        "stage": 1,
        "reason": "test",
    })
    db.insert("verification_results", {
        "attempt_id": 1,
        "modality": "face",
        "model": "test-model",
        "score": 0.1,
        "threshold": 0.7,
        "accepted": 0,
    })

    platform = object.__new__(Platform)
    platform.db = db
    platform.audit = AuditLogger(db)

    deleted = platform.clear_verification_data(actor="test_operator")

    assert deleted == {"attempts": 1, "results": 1}
    assert db.count("auth_attempts") == 0
    assert db.count("verification_results") == 0
    assert db.count("users") == 1
    audit = db.fetchone(
        "SELECT action, details FROM audit_logs ORDER BY id DESC LIMIT 1"
    )
    assert audit["action"] == "verification_data_cleared"
