from src.utils.timeutil import iso


class AuditLogger:
    """Every security-relevant action is written here (append-only from application code)."""
    def __init__(self, db):
        self.db = db

    def log(self, actor, action, target="", outcome="success", details=None):
        return self.db.insert("audit_logs", {"actor": str(actor), "action": action, "target": str(target),
                                             "outcome": outcome, "details": details or {}, "created_at": iso()})

    def recent(self, limit=200, action_like=None):
        if action_like:
            return self.db.fetch("SELECT * FROM audit_logs WHERE action LIKE ? ORDER BY id DESC LIMIT ?", (action_like, limit))
        return self.db.fetch("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
