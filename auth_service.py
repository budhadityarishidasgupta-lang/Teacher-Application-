# auth_service.py
# Modular authentication/service layer for the Synonym app.
# Uses existing SQLAlchemy engine; safe, idempotent schema patches.

from datetime import datetime, timedelta
import secrets
from typing import Optional, Tuple, Dict

from passlib.hash import bcrypt
from sqlalchemy import text

class AuthService:
    def __init__(self, engine):
        self.engine = engine
        self._ensure_auth_columns()

    # ─────────────────────────────────────────────────────────────────
    # Schema patches (safe to run on every start)
    # ─────────────────────────────────────────────────────────────────
    def _ensure_auth_columns(self):
        with self.engine.begin() as conn:
            # Add new columns if they do not exist
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_password_change TIMESTAMPTZ"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token_hash TEXT"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token_expires_at TIMESTAMPTZ"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_attempts INTEGER DEFAULT 0"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ"))

            # Backfill sensible defaults (non-destructive)
            conn.execute(text("""
                UPDATE users
                SET last_password_change = COALESCE(last_password_change, CURRENT_TIMESTAMP)
            """))
            # Give existing students a fresh 365-day window if missing
            conn.execute(text("""
                UPDATE users
                SET expires_at = CURRENT_TIMESTAMP + INTERVAL '365 days'
                WHERE role = 'student' AND (expires_at IS NULL)
            """))
            # Optionally give admins a long horizon to avoid accidental lockout
            conn.execute(text("""
                UPDATE users
                SET expires_at = CURRENT_TIMESTAMP + INTERVAL '36500 days'
                WHERE role = 'admin' AND (expires_at IS NULL)
            """))

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────
    def _user_by_email(self, email: str) -> Optional[Dict]:
        with self.engine.begin() as conn:
            row = conn.execute(text("""
                SELECT user_id, name, email, password_hash, role, is_active,
                       expires_at, last_password_change, failed_attempts, locked_until
                FROM users WHERE lower(email)=:e
            """), {"e": email.strip().lower()}).mappings().fetchone()
        return dict(row) if row else None

    def is_student_expired(self, user: Dict) -> bool:
        if not user or user.get("role") != "student":
            return False
        exp = user.get("expires_at")
        if exp is None:
            return False  # treated as not expired; we already backfill
        # Rely on DB clock (UTC). Compare as naive via ISO strings if needed.
        try:
            # When coming from SQLAlchemy, exp may be a datetime already
            return exp < datetime.utcnow()
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Registration / Password lifecycle
    # ─────────────────────────────────────────────────────────────────
    def register_student(self, name: str, email: str, password: str) -> Tuple[bool, str]:
        """
        Create a student with a 365-day expiry window.
        Returns (ok, message). On success, message includes user_id.
        """
        email_lc = email.strip().lower()
        if not name or not email_lc or not password:
            return False, "Name, email, and password are required."

        with self.engine.begin() as conn:
            exists = conn.execute(text("SELECT 1 FROM users WHERE lower(email)=:e"), {"e": email_lc}).scalar()
            if exists:
                return False, "An account with this email already exists."

            uid = conn.execute(text("""
                INSERT INTO users(name,email,password_hash,role,is_active,expires_at,last_password_change)
                VALUES (:n,:e,:p,'student',TRUE, CURRENT_TIMESTAMP + INTERVAL '365 days', CURRENT_TIMESTAMP)
                RETURNING user_id
            """), {"n": name.strip(), "e": email_lc, "p": bcrypt.hash(password)}).scalar()

        return True, f"Student registered (user_id={uid})."

    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT password_hash FROM users WHERE user_id=:u"), {"u": int(user_id)}).mappings().fetchone()
            if not row:
                return False, "User not found."
            if not bcrypt.verify(old_password, row["password_hash"]):
                return False, "Old password is incorrect."

            conn.execute(text("""
                UPDATE users
                SET password_hash=:ph, last_password_change=CURRENT_TIMESTAMP,
                    reset_token_hash=NULL, reset_token_expires_at=NULL
                WHERE user_id=:u
            """), {"ph": bcrypt.hash(new_password), "u": int(user_id)})
        return True, "Password updated."

    def request_password_reset(self, email: str, ttl_minutes: int = 60) -> Tuple[bool, str, Optional[str]]:
        """
        Generates a one-time reset token. Stores only the hash; returns the plain token
        so the caller can email it or display it once. Returns (ok, message, token|None).
        """
        user = self._user_by_email(email)
        if not user:
            # For privacy, respond generically but do nothing.
            return True, "If the email exists, a reset code has been generated.", None

        token = secrets.token_urlsafe(32)
        token_hash = bcrypt.hash(token)
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE users
                SET reset_token_hash=:h,
                    reset_token_expires_at = CURRENT_TIMESTAMP + (:mins || ' minutes')::interval
                WHERE user_id=:u
            """), {"h": token_hash, "mins": int(ttl_minutes), "u": int(user["user_id"])})
        return True, "Password reset code generated.", token

    def reset_password_with_token(self, email: str, token: str, new_password: str) -> Tuple[bool, str]:
        user = self._user_by_email(email)
        if not user:
            return False, "Invalid reset request."

        with self.engine.begin() as conn:
            row = conn.execute(text("""
                SELECT reset_token_hash, reset_token_expires_at
                FROM users WHERE user_id=:u
            """), {"u": int(user["user_id"])}).mappings().fetchone()

            if not row or not row["reset_token_hash"] or not row["reset_token_expires_at"]:
                return False, "No valid reset code. Please request a new one."

            # Check expiry
            if row["reset_token_expires_at"] < datetime.utcnow():
                return False, "Reset code has expired. Please request a new one."

            # Verify token
            if not bcrypt.verify(token, row["reset_token_hash"]):
                return False, "Invalid reset code."

            # Update password & clear token
            conn.execute(text("""
                UPDATE users
                SET password_hash=:ph,
                    last_password_change=CURRENT_TIMESTAMP,
                    reset_token_hash=NULL,
                    reset_token_expires_at=NULL
                WHERE user_id=:u
            """), {"ph": bcrypt.hash(new_password), "u": int(user["user_id"])})
        return True, "Password has been reset."

    # ─────────────────────────────────────────────────────────────────
    # Optional: lockout helpers (not strictly required, but handy)
    # ─────────────────────────────────────────────────────────────────
    def mark_login_failed(self, user_id: int, max_attempts: int = 5, lock_minutes: int = 15):
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE users
                SET failed_attempts = COALESCE(failed_attempts,0) + 1
                WHERE user_id=:u
            """), {"u": int(user_id)})

            conn.execute(text("""
                UPDATE users
                SET locked_until = CASE
                    WHEN failed_attempts >= :max THEN CURRENT_TIMESTAMP + (:mins || ' minutes')::interval
                    ELSE locked_until
                END
                WHERE user_id=:u
            """), {"u": int(user_id), "max": int(max_attempts), "mins": int(lock_minutes)})

    def mark_login_success(self, user_id: int):
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE users
                SET failed_attempts = 0, locked_until=NULL
                WHERE user_id=:u
            """), {"u": int(user_id)})

    def is_locked(self, user: Dict) -> bool:
        lu = user.get("locked_until")
        if not lu:
            return False
        return lu > datetime.utcnow()

    # ─────────────────────────────────────────────────────────────────
    # Admin-only convenience actions
    # ─────────────────────────────────────────────────────────────────
    def reopen_student(self, user_id: int, days: int = 365) -> Tuple[bool, str]:
        """Admin action: re-enable access window and ensure active=True."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE users
                SET is_active=TRUE,
                    expires_at = CURRENT_TIMESTAMP + (:d || ' days')::interval,
                    failed_attempts = 0,
                    locked_until = NULL
                WHERE user_id=:u AND role='student'
            """), {"u": int(user_id), "d": int(days)})
        return True, "Student access reopened."

    # (Optional) Authenticate helper if you prefer one place to verify secrets.
    def authenticate(self, email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        user = self._user_by_email(email)
        if not user:
            return False, "User not found.", None
        if not user.get("is_active", True):
            return False, "Account disabled.", None
        if self.is_locked(user):
            return False, "Too many failed attempts. Try again later.", None
        if not bcrypt.verify(password, user.get("password_hash") or ""):
            # Side effect: raise failed counter (best-effort, non-fatal)
            try:
                self.mark_login_failed(user["user_id"])
            except Exception:
                pass
            return False, "Wrong password.", None
        # Success: clear any lock and counters
        try:
            self.mark_login_success(user["user_id"])
        except Exception:
            pass
        # Expiry check (block only students)
        if self.is_student_expired(user) and user.get("role") == "student":
            return False, "Account expired. Ask your teacher to reopen access.", user
        return True, "OK", user
