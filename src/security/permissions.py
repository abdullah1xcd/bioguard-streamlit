"""Central definition of roles, permissions and speech commands."""
ROLES = ("admin", "manager", "standard_user", "guest")

COMMANDS = {  # key -> required permission, sensitivity, labels
    "open_dashboard":    {"permission": "view_dashboard",    "sensitivity": "low",    "en": "OPEN DASHBOARD",    "ar": "افتح لوحة التحكم"},
    "view_profile":      {"permission": "view_profile",      "sensitivity": "low",    "en": "VIEW PROFILE",      "ar": "اعرض الملف الشخصي"},
    "show_report":       {"permission": "view_reports",      "sensitivity": "medium", "en": "SHOW REPORT",       "ar": "اعرض التقرير"},
    "generate_report":   {"permission": "generate_report",   "sensitivity": "medium", "en": "GENERATE REPORT",   "ar": "أنشئ تقرير"},
    "show_security_log": {"permission": "view_security_log", "sensitivity": "high",   "en": "SHOW SECURITY LOG", "ar": "اعرض سجل الأمان"},
    "logout":            {"permission": "logout",            "sensitivity": "low",    "en": "LOGOUT",            "ar": "تسجيل الخروج"},
}
ACTION_SENSITIVITY = {k: v["sensitivity"] for k, v in COMMANDS.items()}
ALL_PERMISSIONS = sorted({v["permission"] for v in COMMANDS.values()} | {"manage_users", "approve_actions", "manage_workflows"})

ROLE_PERMISSIONS = {
    "admin": set(ALL_PERMISSIONS),
    "manager": {"view_dashboard", "view_profile", "view_reports", "generate_report", "logout", "approve_actions"},
    "standard_user": {"view_dashboard", "view_profile", "view_reports", "logout"},
    "guest": {"view_dashboard", "logout"},
}


class PermissionStore:
    """DB-backed permission table (seeded from ROLE_PERMISSIONS); supports grant/revoke."""
    def __init__(self, db, audit=None):
        self.db, self.audit = db, audit

    def allowed(self, role, permission) -> bool:
        return self.db.count("role_permissions", "role=? AND permission=?", (role, permission)) > 0

    def role_permissions(self, role):
        return sorted(r["permission"] for r in self.db.fetch("SELECT permission FROM role_permissions WHERE role=?", (role,)))

    def grant(self, role, permission, actor="system"):
        self.db.execute("INSERT OR IGNORE INTO role_permissions(role,permission) VALUES(?,?)", (role, permission))
        if self.audit: self.audit.log(actor, "permission_grant", f"{role}:{permission}")

    def revoke(self, role, permission, actor="system"):
        self.db.execute("DELETE FROM role_permissions WHERE role=? AND permission=?", (role, permission))
        if self.audit: self.audit.log(actor, "permission_revoke", f"{role}:{permission}")

    def check_command(self, role, command):
        meta = COMMANDS.get(command)
        if meta is None:
            return False, None
        return self.allowed(role, meta["permission"]), meta["permission"]
