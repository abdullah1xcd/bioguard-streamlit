import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS roles(name TEXT PRIMARY KEY, description TEXT);
CREATE TABLE IF NOT EXISTS permissions(name TEXT PRIMARY KEY, description TEXT);
CREATE TABLE IF NOT EXISTS role_permissions(role TEXT, permission TEXT, PRIMARY KEY(role, permission));
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, participant_id TEXT UNIQUE NOT NULL,
  role TEXT NOT NULL, consent INTEGER NOT NULL, consent_at TEXT, status TEXT DEFAULT 'active', created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS enrollment_samples(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, modality TEXT, path TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS model_versions(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, kind TEXT, path TEXT, metrics TEXT, status TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS auth_attempts(id INTEGER PRIMARY KEY AUTOINCREMENT, claimed TEXT, mode TEXT, accepted INTEGER, stage INTEGER,
  face_score REAL, voice_score REAL, fused_score REAL, risk_score REAL, risk_level TEXT, action TEXT, reason TEXT, face_detected INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS verification_results(id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id INTEGER, modality TEXT, model TEXT,
  score REAL, threshold REAL, accepted INTEGER, latency_ms REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS challenges(id INTEGER PRIMARY KEY AUTOINCREMENT, claimed TEXT, kind TEXT, expected TEXT, passed INTEGER, detail TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS speech_commands(id INTEGER PRIMARY KEY AUTOINCREMENT, claimed TEXT, command TEXT, confidence REAL,
  identity_verified INTEGER, permitted INTEGER, executed INTEGER, detail TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS security_events(id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT, claimed TEXT, severity TEXT, payload TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS agent_analyses(id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER, claimed TEXT, risk_level TEXT, pattern TEXT,
  evidence TEXT, explanation TEXT, recommendation TEXT, severity TEXT, requires_approval INTEGER, execution_status TEXT, llm_explanation TEXT,
  anomaly TEXT, false_positive INTEGER, analysis_ms REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS scrutiny(id INTEGER PRIMARY KEY AUTOINCREMENT, claimed TEXT, kind TEXT, until TEXT, reason TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS automation_definitions(id TEXT PRIMARY KEY, name TEXT, description TEXT, enabled INTEGER, definition TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS scheduled_jobs(id INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT UNIQUE, interval_seconds INTEGER, enabled INTEGER, last_run TEXT, next_run TEXT);
CREATE TABLE IF NOT EXISTS workflow_executions(id INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT, trigger TEXT, idempotency_key TEXT, status TEXT,
  payload TEXT, state TEXT, error TEXT, attempt INTEGER DEFAULT 0, started_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS action_executions(id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id INTEGER, step_index INTEGER, action TEXT, params TEXT,
  status TEXT, result TEXT, error TEXT, started_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS approval_requests(id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id INTEGER, action TEXT, params TEXT, description TEXT,
  risk_level TEXT, recommendation TEXT, evidence TEXT, requested_by TEXT, status TEXT, approver TEXT, reason TEXT, execution_result TEXT,
  created_at TEXT, expires_at TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY AUTOINCREMENT, level TEXT, title TEXT, message TEXT, is_read INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS retry_history(id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id INTEGER, action_execution_id INTEGER, attempt INTEGER,
  error TEXT, scheduled_for TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS audit_logs(id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT, target TEXT, outcome TEXT, details TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""
TABLES = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init()

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(str(self.path), timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def init(self):
        from src.security.permissions import ROLES, ALL_PERMISSIONS, ROLE_PERMISSIONS
        with self._lock, self._conn() as c:
            c.executescript(SCHEMA)
            for r in ROLES:
                c.execute("INSERT OR IGNORE INTO roles(name,description) VALUES(?,?)", (r, r))
            for p in ALL_PERMISSIONS:
                c.execute("INSERT OR IGNORE INTO permissions(name,description) VALUES(?,?)", (p, p))
            if c.execute("SELECT COUNT(*) FROM role_permissions").fetchone()[0] == 0:
                for role, perms in ROLE_PERMISSIONS.items():
                    for p in perms:
                        c.execute("INSERT OR IGNORE INTO role_permissions(role,permission) VALUES(?,?)", (role, p))

    @staticmethod
    def _enc(v):
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, default=str)
        if isinstance(v, bool):
            return int(v)
        return v

    @staticmethod
    def loads(s, default=None):
        try:
            return json.loads(s) if s else default
        except (TypeError, ValueError):
            return default

    def execute(self, sql, params=()):
        with self._lock, self._conn() as c:
            return c.execute(sql, params).lastrowid

    def fetch(self, sql, params=()):
        with self._lock, self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def fetchone(self, sql, params=()):
        rows = self.fetch(sql, params)
        return rows[0] if rows else None

    def insert(self, table, data):
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        cols = list(data)
        sql = f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        return self.execute(sql, [self._enc(data[c]) for c in cols])

    def update(self, table, key, data, key_col="id"):
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        sets = ",".join(f"{c}=?" for c in data)
        self.execute(f"UPDATE {table} SET {sets} WHERE {key_col}=?", [self._enc(v) for v in data.values()] + [key])

    def count(self, table, where="1=1", params=()):
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        return self.fetchone(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params)["n"]

    def get_setting(self, key, default=None):
        r = self.fetchone("SELECT value FROM settings WHERE key=?", (key,))
        return self.loads(r["value"], default) if r else default

    def set_setting(self, key, value):
        self.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, json.dumps(value)))

    def backup_to(self, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            src = sqlite3.connect(str(self.path)); dst = sqlite3.connect(str(dest))
            try:
                src.backup(dst)
            finally:
                dst.close(); src.close()
        return dest
