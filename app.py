import os, time, random, sqlite3, html, base64, json
from contextlib import closing
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from passlib.hash import bcrypt
from sqlalchemy import create_engine, text

import streamlit as st
import builtins
import hashlib

# Disable all help renderers (prevents the login_page methods panel)
try:
    st.help = lambda *args, **kwargs: None
except Exception:
    pass

try:
    builtins.help = lambda *args, **kwargs: None
except Exception:
    pass
    
# ─────────────────────────────────────────────────────────────────────
# Basic config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Learning English Made Easy", page_icon="📚", layout="wide")

APP_DIR = Path(__file__).parent
load_dotenv(APP_DIR / ".env", override=True)

# Global theme (student quiz surface)
THEME_CSS_PATH = APP_DIR / "static" / "theme.css"
if THEME_CSS_PATH.exists():
    try:
        st.markdown(f"<style>{THEME_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

# Student-only toggle (env or URL param)
FORCE_STUDENT = os.getenv("FORCE_STUDENT_MODE", "0") == "1"
try:
    qp = st.query_params  # Streamlit ≥1.30
except Exception:
    qp = st.experimental_get_query_params()  # older versions

def _first(qv):
    if qv is None: return None
    if isinstance(qv, list): return qv[0]
    return qv

_mode = (_first(qp.get("mode")) or "").strip().lower()
if _mode == "student":
    FORCE_STUDENT = True
elif _mode == "admin":
    FORCE_STUDENT = False

# GPT config
ENABLE_GPT     = os.getenv("ENABLE_GPT", "0") == "1"
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

gpt_client = None
if ENABLE_GPT and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        gpt_client = None
        ENABLE_GPT = False

# Admin bootstrap
ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", "admin@example.com").strip().lower()
ADMIN_NAME     = os.getenv("ADMIN_NAME", "Admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "ChangeMe!123")

# Feature flags (define early!)
TEACHER_UI_V2 = os.getenv("TEACHER_UI_V2", "0") == "1"

# Gamification constants (declared early so helper functions can use them)
LEVEL_BANDS: list[dict[str, object]] = [
    {"level": 1, "min": 0,   "max": 99,  "title": "Learner",   "color": "#22c55e"},  # Green
    {"level": 2, "min": 100, "max": 249, "title": "Achiever",  "color": "#f97316"},  # Orange
    {"level": 3, "min": 250, "max": 499, "title": "Explorer",  "color": "#3b82f6"},  # Blue
    {"level": 4, "min": 500, "max": 999, "title": "Champion",  "color": "#8b5cf6"},  # Purple
    {"level": 5, "min": 1000, "max": None, "title": "Legend", "color": "#fbbf24"},  # Gold
]


BADGE_DEFINITIONS = {
    "First Word Hero": {
        "emoji": "🥇",
        "xp_bonus": 20,
        "badge_type": "milestone",
        "milestone": "1 correct answer",
    },
    "Ten Words Mastered": {
        "emoji": "🧠",
        "xp_bonus": 50,
        "badge_type": "mastery",
        "milestone": "Master 10 unique words",
    },
    "Fifty Words Fluent": {
        "emoji": "🏆",
        "xp_bonus": 150,
        "badge_type": "mastery",
        "milestone": "Master 50 unique words",
    },
    "Lesson Champion": {
        "emoji": "📘",
        "xp_bonus": 100,
        "badge_type": "lesson",
        "milestone": "Lesson ≥90% accuracy",
    },
    "Course Finisher": {
        "emoji": "🎓",
        "xp_bonus": 250,
        "badge_type": "course",
        "milestone": "All lessons in a course ≥80% accuracy",
    },
    "Weekly Streaker": {
        "emoji": "🔥",
        "xp_bonus": 70,
        "badge_type": "streak",
        "milestone": "7-day login streak",
    },
    "Perfectionist": {
        "emoji": "💎",
        "xp_bonus": 100,
        "badge_type": "achievement",
        "milestone": "100% accuracy in a lesson",
    },
}

# ─────────────────────────────────────────────────────────────────────
# Database (Postgres via SQLAlchemy)
# ─────────────────────────────────────────────────────────────────────
_raw = os.environ.get("DATABASE_URL", "").strip()
if not _raw:
    st.error("DATABASE_URL is not set. In Render → Settings → Environment, add DATABASE_URL using your Postgres Internal Connection String.")
    st.stop()

def _normalize(url: str) -> str:
    # normalize Render's postgres:// to SQLAlchemy's postgresql+psycopg2://
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url

DATABASE_URL = _normalize(_raw)
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)

# ─────────────────────────────────────────────────────────────────────
# Schema creation + tiny self-healing patches
# ─────────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist (idempotent)."""
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS users (
          user_id       SERIAL PRIMARY KEY,
          name          TEXT NOT NULL,
          email         TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          role          TEXT NOT NULL CHECK (role IN ('admin','student')),
          is_active     BOOLEAN NOT NULL DEFAULT TRUE,
          expires_at    TIMESTAMPTZ,
          created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS courses (
          course_id   SERIAL PRIMARY KEY,
          title       TEXT NOT NULL,
          description TEXT,
          created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS lessons (
          lesson_id  SERIAL PRIMARY KEY,
          course_id  INTEGER NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
          title      TEXT NOT NULL,
          sort_order INTEGER DEFAULT 0,
          created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS words (
          word_id    SERIAL PRIMARY KEY,
          headword   TEXT NOT NULL,
          synonyms   TEXT NOT NULL,
          difficulty INTEGER DEFAULT 2
        );
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS words_uniq ON words(headword, synonyms);
        """,
        """
        CREATE TABLE IF NOT EXISTS lesson_words (
          lesson_id  INTEGER NOT NULL REFERENCES lessons(lesson_id) ON DELETE CASCADE,
          word_id    INTEGER NOT NULL REFERENCES words(word_id)   ON DELETE CASCADE,
          sort_order INTEGER DEFAULT 0,
          PRIMARY KEY (lesson_id, word_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS enrollments (
          user_id   INTEGER NOT NULL REFERENCES users(user_id)     ON DELETE CASCADE,
          course_id INTEGER NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
          PRIMARY KEY (user_id, course_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS attempts (
          id             BIGSERIAL PRIMARY KEY,
          user_id        INTEGER,
          course_id      INTEGER,
          lesson_id      INTEGER,
          headword       TEXT,
          is_correct     BOOLEAN,
          response_ms    INTEGER,
          chosen         TEXT,
          correct_choice TEXT,
          ts             TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS word_stats (
          user_id          INTEGER NOT NULL,
          headword         TEXT    NOT NULL,
          correct_streak   INTEGER DEFAULT 0,
          total_attempts   INTEGER DEFAULT 0,
          correct_attempts INTEGER DEFAULT 0,
          xp_points        INTEGER DEFAULT 0,
          streak_count     INTEGER DEFAULT 0,
          last_seen        TIMESTAMPTZ,
          mastered         BOOLEAN DEFAULT FALSE,
          difficulty       INTEGER DEFAULT 2,
          due_date         TIMESTAMPTZ,
          PRIMARY KEY (user_id, headword)
        );
        """
        """
        CREATE TABLE IF NOT EXISTS classes (
          class_id    SERIAL PRIMARY KEY,
          name        TEXT NOT NULL,
          start_date  DATE,
          is_archived BOOLEAN NOT NULL DEFAULT FALSE,
          archived_at TIMESTAMPTZ,
          created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        """
        CREATE TABLE IF NOT EXISTS class_students (
          class_id    INTEGER NOT NULL REFERENCES classes(class_id) ON DELETE CASCADE,
          user_id     INTEGER NOT NULL REFERENCES users(user_id)   ON DELETE CASCADE,
          assigned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (class_id, user_id)
        );
        """
        """
        CREATE TABLE IF NOT EXISTS achievements (
          achievement_id SERIAL PRIMARY KEY,
          user_id        INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
          badge_name     TEXT NOT NULL,
          badge_type     TEXT NOT NULL,
          emoji          TEXT NOT NULL,
          xp_bonus       INTEGER DEFAULT 0,
          awarded_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
          UNIQUE (user_id, badge_name)
        );
        """
    ]
    with engine.begin() as conn:
        for q in ddl:
            conn.execute(text(q))

def patch_users_table():
    """Ensure legacy users table has required cols/data; backfill if needed."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"))

    admin_email_lc = ADMIN_EMAIL.lower()
    with engine.begin() as conn:
        conn.execute(text("UPDATE users SET role='admin'   WHERE role IS NULL AND lower(email)=:e"),
                     {"e": admin_email_lc})
        conn.execute(text("UPDATE users SET role='student' WHERE role IS NULL AND lower(email)<>:e"),
                     {"e": admin_email_lc})
        conn.execute(text("UPDATE users SET is_active=TRUE WHERE is_active IS NULL"))

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT user_id, email, COALESCE(role,'student') AS role
                FROM users
                WHERE password_hash IS NULL OR password_hash=''
            """)
        ).mappings().all()
    if rows:
        with engine.begin() as conn:
            for r in rows:
                raw_pwd = ADMIN_PASSWORD if r["role"] == "admin" else "Learn123!"
                conn.execute(
                    text("UPDATE users SET password_hash=:p WHERE user_id=:u"),
                    {"p": bcrypt.hash(raw_pwd), "u": r["user_id"]}
                )

def patch_courses_table():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE courses ADD COLUMN IF NOT EXISTS description TEXT"))
        conn.execute(text("ALTER TABLE lessons ADD COLUMN IF NOT EXISTS sort_order INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE words   ADD COLUMN IF NOT EXISTS difficulty INTEGER DEFAULT 2"))

def patch_gamification_tables():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE word_stats ADD COLUMN IF NOT EXISTS xp_points INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE word_stats ADD COLUMN IF NOT EXISTS streak_count INTEGER DEFAULT 0"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS achievements (
              achievement_id SERIAL PRIMARY KEY,
              user_id        INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
              badge_name     TEXT NOT NULL,
              badge_type     TEXT NOT NULL,
              emoji          TEXT NOT NULL,
              xp_bonus       INTEGER DEFAULT 0,
              awarded_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
              UNIQUE (user_id, badge_name)
            )
        """))

# Bootstrap order
init_db()
patch_users_table()
patch_courses_table()
patch_gamification_tables()

def get_missed_words(user_id: int, lesson_id: int):
    """
    Returns a list of headwords whose latest attempt in this lesson was incorrect.
    Falls back to words with correct_streak=0 (but attempted) if no recent wrongs.
    """
    latest = pd.read_sql(
        text("""
            WITH last AS (
              SELECT headword, MAX(id) AS last_id
              FROM attempts
              WHERE user_id=:u AND lesson_id=:l
              GROUP BY headword
            )
            SELECT a.headword
            FROM attempts a
            JOIN last ON a.id = last.last_id
            WHERE a.is_correct = FALSE
        """),
        con=engine, params={"u": int(user_id), "l": int(lesson_id)}
    )
    missed = set(latest["headword"].tolist())

    if not missed:
        fallback = pd.read_sql(
            text("""
                SELECT DISTINCT w.headword
                FROM lesson_words lw
                JOIN words w ON w.word_id = lw.word_id
                LEFT JOIN word_stats s ON s.user_id=:u AND s.headword = w.headword
                WHERE lw.lesson_id = :l
                  AND s.total_attempts > 0
                  AND COALESCE(s.correct_streak, 0) = 0
            """),
            con=engine, params={"u": int(user_id), "l": int(lesson_id)}
        )
        missed = set(fallback["headword"].tolist())

    return sorted(missed)

# ─────────────────────────────────────────────────────────────────────
# DB helpers (CRUD) — Postgres
# ─────────────────────────────────────────────────────────────────────
def create_user(name, email, password, role):
    h = bcrypt.hash(password)
    with engine.begin() as conn:
        user_id = conn.execute(
            text("""INSERT INTO users(name,email,password_hash,role)
                    VALUES (:n,:e,:p,:r)
                    ON CONFLICT (email) DO NOTHING
                    RETURNING user_id"""),
            {"n": name, "e": email, "p": h, "r": role}
        ).scalar()
        if user_id is None:
            user_id = conn.execute(text("SELECT user_id FROM users WHERE email=:e"), {"e": email}).scalar()
        return user_id

def user_by_email(email):
    with engine.begin() as conn:
        row = conn.execute(
            text("""SELECT user_id,name,email,password_hash,role,is_active,expires_at
                    FROM users WHERE email=:e"""),
            {"e": email}
        ).mappings().fetchone()
    return dict(row) if row else None

def ensure_admin():
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM users WHERE role='admin' LIMIT 1")).scalar()
    if not exists:
        try:
            create_user(ADMIN_NAME, ADMIN_EMAIL, ADMIN_PASSWORD, "admin")
        except Exception:
            pass

def set_user_active(user_id, active: bool):
    with engine.begin() as conn:
        conn.execute(text("UPDATE users SET is_active=:a WHERE user_id=:u"),
                     {"a": bool(active), "u": user_id})

def all_students_df():
    df_users = pd.read_sql(
        text("SELECT user_id,name,email,is_active FROM users WHERE role='student'"),
        con=engine
    )
    df_stats = pd.read_sql(
        text("""
            SELECT user_id,
                   SUM(correct_attempts) AS correct_total,
                   SUM(total_attempts)   AS attempts_total,
                   SUM(CASE WHEN mastered THEN 1 ELSE 0 END) AS mastered_count,
                   MAX(last_seen)        AS last_active
            FROM word_stats GROUP BY user_id
        """),
        con=engine
    )
    df = df_users.merge(df_stats, on="user_id", how="left")
    for c in ["correct_total","attempts_total","mastered_count"]:
        df[c] = df[c].fillna(0).astype(int)
    return df.sort_values("name")

def create_classroom(name: str, start_date=None):
    with engine.begin() as conn:
        cid = conn.execute(
            text(
                """INSERT INTO classes(name,start_date)
                        VALUES (:n,:d)
                        RETURNING class_id"""
            ),
            {"n": name, "d": start_date},
        ).scalar()
    return cid

def get_classrooms(include_archived: bool = False) -> pd.DataFrame:
    sql = "SELECT class_id,name,start_date,is_archived,archived_at,created_at FROM classes"
    if not include_archived:
        sql += " WHERE is_archived=FALSE"
    sql += " ORDER BY is_archived, COALESCE(start_date, '1970-01-01'::date), name"
    return pd.read_sql(text(sql), con=engine)

def get_class_students(class_id: int) -> pd.DataFrame:
    sql = text(
        """
        SELECT u.user_id, u.name, u.email, u.is_active, cs.assigned_at
        FROM class_students cs
        JOIN users u ON u.user_id = cs.user_id
        WHERE cs.class_id = :cid
        ORDER BY u.name
        """
    )
    return pd.read_sql(sql, con=engine, params={"cid": int(class_id)})

def assign_students_to_class(class_id: int, student_ids: list[int]):
    if not student_ids:
        return
    with engine.begin() as conn:
        for sid in student_ids:
            conn.execute(
                text(
                    """INSERT INTO class_students(class_id,user_id)
                            VALUES (:c,:s)
                            ON CONFLICT (class_id,user_id) DO NOTHING"""
                ),
                {"c": int(class_id), "s": int(sid)},
            )

def unassign_students_from_class(class_id: int, student_ids: list[int]):
    if not student_ids:
        return
    with engine.begin() as conn:
        for sid in student_ids:
            conn.execute(
                text("DELETE FROM class_students WHERE class_id=:c AND user_id=:s"),
                {"c": int(class_id), "s": int(sid)},
            )

def set_class_archived(class_id: int, archive: bool):
    with engine.begin() as conn:
        if archive:
            conn.execute(
                text(
                    """UPDATE classes
                        SET is_archived=TRUE,
                            archived_at=COALESCE(archived_at, CURRENT_TIMESTAMP)
                      WHERE class_id=:c"""
                ),
                {"c": int(class_id)},
            )
        else:
            conn.execute(
                text(
                    """UPDATE classes
                        SET is_archived=FALSE,
                            archived_at=NULL
                      WHERE class_id=:c"""
                ),
                {"c": int(class_id)},
            )

def get_classes_for_student(user_id: int, include_archived: bool = True) -> pd.DataFrame:
    sql = """
        SELECT c.class_id,
               c.name,
               c.start_date,
               c.is_archived,
               c.archived_at,
               cs.assigned_at
        FROM class_students cs
        JOIN classes c ON c.class_id = cs.class_id
        WHERE cs.user_id = :uid
    """
    if not include_archived:
        sql += " AND c.is_archived=FALSE"
    sql += " ORDER BY c.is_archived, COALESCE(c.start_date, '1970-01-01'::date), c.name"
    return pd.read_sql(text(sql), con=engine, params={"uid": int(user_id)})

def lesson_words(course_id, lesson_id):
    sql = """
        SELECT w.headword, w.synonyms, w.difficulty
        FROM lesson_words lw
        JOIN words   w ON w.word_id = lw.word_id
        JOIN lessons l ON l.lesson_id = lw.lesson_id
        WHERE lw.lesson_id = :lid AND l.course_id = :cid
        ORDER BY lw.sort_order
    """
    return pd.read_sql(text(sql), con=engine, params={"lid": int(lesson_id), "cid": int(course_id)})

def mastered_count(user_id, lesson_id):
    words = pd.read_sql(
        text("""
            SELECT w.headword
            FROM lesson_words lw
            JOIN words w ON w.word_id=lw.word_id
            WHERE lw.lesson_id=:lid
        """),
        con=engine, params={"lid": int(lesson_id)}
    )["headword"].tolist()
    if not words:
        return 0, 0
    m = pd.read_sql(
        text("""
            SELECT COUNT(*) AS c
            FROM word_stats
            WHERE user_id=:u AND mastered=TRUE AND headword = ANY(:arr)
        """),
        con=engine,
        params={"u": int(user_id), "arr": words}
    )["c"].iloc[0]
    return int(m), len(words)


def level_for_xp(xp_total: int):
    xp_total = int(xp_total or 0)
    for band in LEVEL_BANDS:
        upper = band["max"]
        if upper is None or xp_total <= upper:
            return band
    return LEVEL_BANDS[-1]


def next_level_band(current_band: dict | None):
    if not current_band:
        return None
    for idx, band in enumerate(LEVEL_BANDS):
        if band["level"] == current_band["level"]:
            return LEVEL_BANDS[idx + 1] if idx + 1 < len(LEVEL_BANDS) else None
    return None


def compute_answer_streak(conn, user_id: int, limit: int = 200) -> int:
    rows = conn.execute(
        text(
            """
            SELECT is_correct
            FROM attempts
            WHERE user_id=:u
            ORDER BY id DESC
            LIMIT :lim
            """
        ),
        {"u": int(user_id), "lim": int(limit)},
    ).fetchall()

    streak = 0
    for row in rows:
        if row[0]:
            streak += 1
        else:
            break
    return streak


def compute_login_streak(conn, user_id: int) -> int:
    rows = conn.execute(
        text(
            """
            SELECT DISTINCT DATE(ts) AS day
            FROM attempts
            WHERE user_id=:u
            ORDER BY day DESC
            """
        ),
        {"u": int(user_id)},
    ).fetchall()

    dates = [r[0] for r in rows if r[0] is not None]
    if not dates:
        return 0

    streak = 1
    last_day = dates[0]
    for day in dates[1:]:
        if last_day == day:
            continue
        if (last_day - day) == timedelta(days=1):
            streak += 1
            last_day = day
        else:
            break
    return streak


def grant_badge(conn, user_id: int, badge_name: str):
    definition = BADGE_DEFINITIONS.get(badge_name)
    if not definition:
        return None

    exists = conn.execute(
        text("SELECT 1 FROM achievements WHERE user_id=:u AND badge_name=:b"),
        {"u": int(user_id), "b": badge_name},
    ).scalar()
    if exists:
        return None

    row = conn.execute(
        text(
            """
            INSERT INTO achievements (user_id, badge_name, badge_type, emoji, xp_bonus)
            VALUES (:u, :b, :t, :e, :xp)
            RETURNING achievement_id, badge_name, badge_type, emoji, xp_bonus, awarded_at
            """
        ),
        {
            "u": int(user_id),
            "b": badge_name,
            "t": definition["badge_type"],
            "e": definition["emoji"],
            "xp": int(definition.get("xp_bonus", 0)),
        },
    ).mappings().fetchone()

    return dict(row) if row else None


def evaluate_badges(conn, user_id: int):
    newly_awarded = []

    def maybe_award(name: str):
        badge = grant_badge(conn, user_id, name)
        if badge:
            newly_awarded.append(badge)

    correct_words = conn.execute(
        text("SELECT COUNT(*) FROM word_stats WHERE user_id=:u AND correct_attempts > 0"),
        {"u": int(user_id)},
    ).scalar() or 0
    if correct_words >= 1:
        maybe_award("First Word Hero")

    mastered_total = conn.execute(
        text("SELECT COUNT(*) FROM word_stats WHERE user_id=:u AND mastered IS TRUE"),
        {"u": int(user_id)},
    ).scalar() or 0
    if mastered_total >= 10:
        maybe_award("Ten Words Mastered")
    if mastered_total >= 50:
        maybe_award("Fifty Words Fluent")

    lesson_sql = text(
        """
        WITH lesson_totals AS (
          SELECT lesson_id, COUNT(DISTINCT word_id) AS total_words
          FROM lesson_words
          GROUP BY lesson_id
        ),
        lesson_attempts AS (
          SELECT lesson_id,
                 COUNT(*) AS total_attempts,
                 SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct_attempts,
                 COUNT(DISTINCT headword) AS distinct_words
          FROM attempts
          WHERE user_id = :u
          GROUP BY lesson_id
        )
        SELECT l.course_id,
               lt.lesson_id,
               lt.total_words,
               COALESCE(la.total_attempts, 0) AS total_attempts,
               COALESCE(la.correct_attempts, 0) AS correct_attempts,
               COALESCE(la.distinct_words, 0) AS distinct_words
        FROM lessons l
        JOIN lesson_totals lt ON lt.lesson_id = l.lesson_id
        LEFT JOIN lesson_attempts la ON la.lesson_id = lt.lesson_id
        JOIN enrollments e ON e.course_id = l.course_id AND e.user_id = :u
        """
    )

    lesson_rows = conn.execute(lesson_sql, {"u": int(user_id)}).mappings().all()

    has_lesson_champion = False
    has_perfectionist = False
    course_tracker: dict[int, dict[str, int]] = {}

    for row in lesson_rows:
        total_words = int(row.get("total_words") or 0)
        if total_words <= 0:
            continue

        total_attempts = int(row.get("total_attempts") or 0)
        correct_attempts = int(row.get("correct_attempts") or 0)
        distinct_words = int(row.get("distinct_words") or 0)
        attempted_all = distinct_words >= total_words
        accuracy = (correct_attempts / total_attempts) if total_attempts else 0.0

        if attempted_all and accuracy >= 0.90:
            has_lesson_champion = True
        if attempted_all and total_attempts > 0 and accuracy >= 0.9999:
            has_perfectionist = True

        course_id = int(row.get("course_id"))
        tracker = course_tracker.setdefault(course_id, {"total": 0, "attempted_all": 0, "meets": 0})
        tracker["total"] += 1
        if attempted_all:
            tracker["attempted_all"] += 1
            if accuracy >= 0.80:
                tracker["meets"] += 1

    if has_lesson_champion:
        maybe_award("Lesson Champion")
    if has_perfectionist:
        maybe_award("Perfectionist")

    for stats in course_tracker.values():
        if stats["total"] > 0 and stats["attempted_all"] == stats["total"] and stats["meets"] == stats["total"]:
            maybe_award("Course Finisher")
            break

    login_streak = compute_login_streak(conn, user_id)
    if login_streak >= 7:
        maybe_award("Weekly Streaker")

    return newly_awarded


def gamification_snapshot(user_id: int):
    with engine.begin() as conn:
        xp_words = conn.execute(
            text("SELECT COALESCE(SUM(xp_points), 0) FROM word_stats WHERE user_id=:u"),
            {"u": int(user_id)},
        ).scalar() or 0
        xp_badges = conn.execute(
            text("SELECT COALESCE(SUM(xp_bonus), 0) FROM achievements WHERE user_id=:u"),
            {"u": int(user_id)},
        ).scalar() or 0
        mastered_total = conn.execute(
            text("SELECT COUNT(*) FROM word_stats WHERE user_id=:u AND mastered IS TRUE"),
            {"u": int(user_id)},
        ).scalar() or 0
        correct_words = conn.execute(
            text("SELECT COUNT(*) FROM word_stats WHERE user_id=:u AND correct_attempts > 0"),
            {"u": int(user_id)},
        ).scalar() or 0
        badges = conn.execute(
            text(
                """
                SELECT badge_name, badge_type, emoji, xp_bonus, awarded_at
                FROM achievements
                WHERE user_id=:u
                ORDER BY awarded_at DESC
                """
            ),
            {"u": int(user_id)},
        ).mappings().all()
        answer_streak = compute_answer_streak(conn, user_id)
        login_streak = compute_login_streak(conn, user_id)

    xp_total = int(xp_words) + int(xp_badges)
    current_band = level_for_xp(xp_total)
    next_band = next_level_band(current_band)

    if current_band.get("max") is None:
        progress_pct = 100
        xp_to_next = 0
    else:
        span = max(current_band["max"] - current_band["min"], 1)
        progress_pct = int(
            max(
                0,
                min(100, round(100 * (xp_total - current_band["min"]) / span)),
            )
        )
        xp_to_next = max(current_band["max"] - xp_total + 1, 0)

    return {
        "xp_total": xp_total,
        "xp_from_words": int(xp_words),
        "xp_from_badges": int(xp_badges),
        "level": current_band["level"],
        "level_name": current_band["title"],
        "level_color": current_band["color"],
        "next_level": next_band["level"] if next_band else None,
        "next_level_name": next_band["title"] if next_band else None,
        "xp_to_next": xp_to_next,
        "progress_pct": progress_pct,
        "badges": [dict(b) for b in badges],
        "mastered_words": int(mastered_total),
        "correct_words": int(correct_words),
        "current_streak": int(answer_streak),
        "login_streak": int(login_streak),
    }


def celebrate_badges(badges):
    if not badges:
        return
    st.markdown(CONFETTI_SNIPPET, unsafe_allow_html=True)
    try:
        st.audio(BADGE_CHIME_AUDIO, format="audio/wav", start_time=0)
    except Exception:
        pass


def inject_gamification_css():
    if st.session_state.get("_gamification_css_injected"):
        return
    css = """
    <style>
      .gami-card-shell {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(16,185,129,0.08));
        border-radius: 18px;
        padding: 16px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        display: flex;
        flex-direction: column;
        gap: 14px;
      }
      .gami-top-row {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
      }
      .gami-stat {
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 110px;
      }
      .gami-stat .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(15, 23, 42, 0.6);
      }
      .gami-stat .value {
        font-size: 1.3rem;
        font-weight: 700;
      }
      .gami-stat.level .value {
        color: var(--gami-level-color, #2563eb);
      }
      .gami-progress {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .gami-progress-bar {
        background: rgba(148, 163, 184, 0.25);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
      }
      .gami-progress-bar .fill {
        height: 100%;
        border-radius: inherit;
        transition: width 0.6s ease;
      }
      .gami-progress small {
        font-size: 0.75rem;
        color: rgba(15, 23, 42, 0.7);
      }
      .gami-streaks {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 0.85rem;
        font-weight: 600;
      }
      .gami-streaks .secondary {
        opacity: 0.7;
      }
      .gami-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .gami-badge {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 62px;
        padding: 8px 6px;
        border-radius: 14px;
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }
      .gami-badge .emoji {
        font-size: 1.3rem;
      }
      .gami-badge small {
        font-size: 0.65rem;
        font-weight: 600;
        opacity: 0.85;
      }
      .gami-badge.earned {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.18);
      }
      .gami-badge.locked {
        opacity: 0.35;
      }
      .gami-badge.pulse {
        animation: gami-pulse 1.4s ease-in-out 3;
      }
      @keyframes gami-pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.08); }
        100% { transform: scale(1); }
      }
      .gami-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 0.8rem;
        color: rgba(15, 23, 42, 0.65);
      }
      .gami-mobile summary {
        cursor: pointer;
        list-style: none;
        font-weight: 700;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 14px;
        padding: 12px 16px;
        background: rgba(255,255,255,0.75);
        margin-bottom: 8px;
      }
      .gami-mobile summary::-webkit-details-marker {
        display: none;
      }
      .gami-mobile summary:after {
        content: '▾';
        float: right;
        opacity: 0.6;
      }
      .gami-mobile[open] summary:after {
        transform: rotate(180deg);
      }
      .gami-mobile .gami-card-shell {
        margin-top: 8px;
      }
      .gami-desktop {
        display: none;
      }
      @media (min-width: 900px) {
        .gami-desktop {
          display: block;
        }
        .gami-mobile {
          display: none;
        }
      }
      @media (max-width: 899px) {
        .gami-mobile {
          display: block;
          margin-bottom: 16px;
        }
      }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.session_state["_gamification_css_injected"] = True


def build_badge_row(snapshot: dict, highlight: set[str]):
    earned = {b["badge_name"] for b in snapshot.get("badges", [])}
    pieces = []
    for name, meta in BADGE_DEFINITIONS.items():
        classes = ["gami-badge", "earned" if name in earned else "locked"]
        if name in highlight:
            classes.append("pulse")
        tooltip = f"{name} — {meta['milestone']} (+{meta['xp_bonus']} XP)"
        pieces.append(
            f"<span class='{' '.join(classes)}' title='{html.escape(tooltip)}'>"
            f"<span class='emoji'>{meta['emoji']}</span>"
            f"<small>+{meta['xp_bonus']}</small>"
            "</span>"
        )
    return "".join(pieces)


def build_gamification_card(snapshot: dict, highlight: set[str]):
    inject_gamification_css()
    xp_total = int(snapshot.get("xp_total", 0))
    level = int(snapshot.get("level", 1))
    level_name = html.escape(snapshot.get("level_name", "Learner"))
    level_color = snapshot.get("level_color", "#2563eb")
    progress_pct = int(max(0, min(100, snapshot.get("progress_pct", 0))))
    next_level = snapshot.get("next_level")
    xp_to_next = int(snapshot.get("xp_to_next") or 0)
    mastered = int(snapshot.get("mastered_words", 0))
    correct_words = int(snapshot.get("correct_words", 0))
    login_streak = int(snapshot.get("login_streak", 0))
    answer_streak = int(snapshot.get("current_streak", 0))

    if next_level:
        progress_caption = f"{xp_to_next} XP to Level {next_level}"
    else:
        progress_caption = "Legend status unlocked!"

    badge_html = build_badge_row(snapshot, highlight)

    card = f"""
    <div class="gami-card-shell" style="--gami-level-color:{level_color};">
      <div class="gami-top-row">
        <div class="gami-stat">
          <span class="label">Total XP</span>
          <span class="value">{xp_total}</span>
        </div>
        <div class="gami-stat level">
          <span class="label">Level</span>
          <span class="value">Lv {level} · {level_name}</span>
        </div>
      </div>
      <div class="gami-progress">
        <div class="gami-progress-bar">
          <div class="fill" style="width:{progress_pct}%; background:{level_color};"></div>
        </div>
        <small>{progress_caption}</small>
      </div>
      <div class="gami-streaks">
        <span>🔥 {login_streak}-day streak</span>
        <span class="secondary">✅ {answer_streak} correct streak</span>
      </div>
      <div class="gami-meta">
        <span>🧠 Mastered words: {mastered}</span>
        <span>📚 Words practiced: {correct_words}</span>
      </div>
      <div class="gami-badges">{badge_html}</div>
    </div>
    """
    return card


def render_gamification_panels(snapshot: dict, highlight: set[str] | None = None):
    highlight = set(highlight or [])
    card = build_gamification_card(snapshot, highlight)
    sidebar_html = f"<div class='gami-desktop'>{card}</div>"
    level = int(snapshot.get("level", 1))
    level_name = html.escape(snapshot.get("level_name", "Learner"))
    xp_total = int(snapshot.get("xp_total", 0))
    mobile_summary = f"⭐ Level {level} · {level_name} — {xp_total} XP"
    mobile_html = f"""
    <details class="gami-mobile">
      <summary>{mobile_summary}</summary>
      {card}
    </details>
    """
    return sidebar_html, mobile_html

def update_after_attempt(user_id, course_id, lesson_id, headword, is_correct, response_ms, difficulty, chosen, correct_choice):
    xp_awarded = 0
    xp_for_word = 0
    new_badges: list[dict] = []

    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT correct_streak, streak_count, mastered, xp_points
                FROM word_stats
                WHERE user_id=:u AND headword=:h
                """
            ),
            {"u": user_id, "h": headword},
        ).mappings().fetchone()

        prior_streak = int((row or {}).get("streak_count") or (row or {}).get("correct_streak") or 0)
        prior_mastered = bool((row or {}).get("mastered"))

        new_streak = prior_streak + 1 if is_correct else 0
        became_mastered = is_correct and new_streak >= 3 and not prior_mastered
        mastered_flag = prior_mastered or (is_correct and new_streak >= 3)

        attempt_xp = 10 if is_correct else 0
        mastery_bonus = 50 if became_mastered else 0
        xp_for_word = attempt_xp + mastery_bonus
        xp_awarded += xp_for_word

        add_days = 3 if (is_correct and mastered_flag) else (1 if is_correct else 0)
        due = datetime.utcnow() + timedelta(days=add_days)

        conn.execute(
            text(
                """
                INSERT INTO word_stats (user_id, headword, correct_streak, total_attempts, correct_attempts, xp_points, streak_count, last_seen, mastered, difficulty, due_date)
                VALUES (:u, :h, :cs, 1, :ca, :xp, :sc, CURRENT_TIMESTAMP, :m, :d, :due)
                ON CONFLICT (user_id, headword) DO UPDATE SET
                    correct_streak   = EXCLUDED.correct_streak,
                    total_attempts   = word_stats.total_attempts + 1,
                    correct_attempts = word_stats.correct_attempts + (:ca),
                    xp_points        = word_stats.xp_points + :xp,
                    streak_count     = EXCLUDED.streak_count,
                    last_seen        = CURRENT_TIMESTAMP,
                    mastered         = CASE WHEN :m THEN TRUE ELSE word_stats.mastered END,
                    difficulty       = :d,
                    due_date         = :due
                """
            ),
            {
                "u": user_id,
                "h": headword,
                "cs": new_streak,
                "sc": new_streak,
                "ca": 1 if is_correct else 0,
                "xp": xp_for_word,
                "m": mastered_flag,
                "d": int(difficulty),
                "due": due,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO attempts(user_id,course_id,lesson_id,headword,is_correct,response_ms,chosen,correct_choice)
                VALUES (:u,:c,:l,:h,:ok,:ms,:ch,:cc)
                """
            ),
            {
                "u": user_id,
                "c": course_id,
                "l": lesson_id,
                "h": headword,
                "ok": bool(is_correct),
                "ms": int(response_ms),
                "ch": chosen,
                "cc": correct_choice,
            },
        )

        new_badges = evaluate_badges(conn, user_id)

    xp_awarded += sum(int(b.get("xp_bonus", 0) or 0) for b in new_badges)

    return {
        "xp_awarded": xp_awarded,
        "xp_for_word": xp_for_word,
        "new_badges": new_badges,
        "became_mastered": became_mastered,
    }

def recent_stats(user_id, course_id, lesson_id, n=10):
    df = pd.read_sql(
        text("""
            SELECT is_correct::int AS is_correct, response_ms
            FROM attempts
            WHERE user_id=:u AND course_id=:c AND lesson_id=:l
            ORDER BY id DESC LIMIT :n
        """),
        con=engine, params={"u": user_id, "c": course_id, "l": lesson_id, "n": int(n)}
    )
    if df.empty:
        return {"accuracy": 0.0, "avg_ms": 15000.0}
    return {"accuracy": float(df["is_correct"].mean()), "avg_ms": float(df["response_ms"].mean())}

def choose_next_word(user_id, course_id, lesson_id, df_words):
    """Adaptive next word (simple rule: recent accuracy & speed)."""
    stats = recent_stats(user_id, course_id, lesson_id, n=10)
    acc, avg = stats["accuracy"], stats["avg_ms"]
    if acc >= 0.75 and avg <= 8000:
        tgt = 3
    elif acc <= 0.5 or avg >= 12000:
        tgt = 1
    else:
        tgt = 2
    candidates = df_words[df_words["difficulty"] == tgt]["headword"].tolist() or df_words["headword"].tolist()
    hist = st.session_state.get("asked_history", [])
    pool = [w for w in candidates if w not in hist[-3:]] or candidates
    return random.choice(pool)

def build_question_payload(
    headword: str,
    synonyms_str: str,
    lesson_df: pd.DataFrame | None = None,
):
    """Construct a multiple-choice payload for the active headword.

    Correct answers come from the word's synonyms. Distractors are built from
    other words in the same lesson when available so each run feels fresh and
    contextually relevant. Remaining slots fall back to a small generic pool
    to ensure six options are always presented.
    """

    syn_list = [s.strip() for s in str(synonyms_str).split(",") if s.strip()]
    correct = syn_list[:2] if len(syn_list) >= 2 else syn_list[:1]
    if len(correct) == 1:
        correct = [correct[0], f"{correct[0]} (close)"]

    seen_lower = {c.lower() for c in correct}

    distractors: list[str] = []
    if lesson_df is not None and not lesson_df.empty:
        candidates: list[str] = []
        for _, row in lesson_df.iterrows():
            other_headword = str(row.get("headword", "")).strip()
            if other_headword.lower() == headword.lower():
                continue

            # Add the other headword and its synonyms as potential distractors.
            if other_headword:
                candidates.append(other_headword)

            other_synonyms = [
                s.strip() for s in str(row.get("synonyms", "")).split(",") if s.strip()
            ]
            candidates.extend(other_synonyms)

        random.shuffle(candidates)
        for cand in candidates:
            cand_l = cand.lower()
            if cand_l in seen_lower:
                continue
            distractors.append(cand)
            seen_lower.add(cand_l)
            if len(distractors) >= 4:
                break

    if len(distractors) < 4:
        fallback_pool = [
            "banana",
            "pencil",
            "soccer",
            "window",
            "pizza",
            "rainbow",
            "kitten",
            "tractor",
            "marble",
            "backpack",
            "ladder",
            "ocean",
            "camera",
            "blanket",
            "sandwich",
            "rocket",
            "helmet",
            "garden",
            "notebook",
            "button",
        ]
        random.shuffle(fallback_pool)
        for cand in fallback_pool:
            cand_l = cand.lower()
            if cand_l in seen_lower:
                continue
            distractors.append(cand)
            seen_lower.add(cand_l)
            if len(distractors) >= 4:
                break

    choices = correct + distractors[:4]
    random.shuffle(choices)
    return {"headword": headword, "choices": choices, "correct": set(correct)}

def gpt_feedback_examples(headword: str, correct_word: str):
    """
    Returns (why, [ex1, ex2]) with kid-friendly, spoken-English sentences.
    Uses GPT when enabled; otherwise a simple fallback.
    """
    def _fallback():
        why = f"'{correct_word}' is a good synonym for '{headword}' because they mean almost the same thing."
        return why, [
            f"I felt {correct_word} when I won the game.",
            f"Our teacher was {correct_word} about our project."
        ]

    if not (ENABLE_GPT and gpt_client):
        return _fallback()

    try:
        prompt = f"""
You are a tutor for ages 7–10. Write natural, spoken-English output.

HEADWORD: "{headword}"
CORRECT SYNONYM (use this in examples): "{correct_word}"

Output JSON only: {{"why": "...", "examples": ["...", "..."]}}

Rules:
- "why": 1 short sentence (≤ 16 words) in kid-friendly language explaining why "{correct_word}" matches "{headword}".
- "examples": Act as a english teacher teaching children age group of 7 to 11. Create TWO different sentences that is often used by english speaking people and that makes perfect gramatical sense.
- Use "{correct_word}" **exactly once** in each example. Prefer NOT to use "{headword}" unless it sounds natural.
- 8–12 words each, simple present/past, no semicolons/dashes/quotes. Avoid rare words and odd pairings.
- No proper names, brands, profanity, bias or metaphors. Keep it positive and clear.
- Return valid JSON only. No extra text.
"""
        resp = gpt_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Be concise, clear, and age-appropriate. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )

        import json
        payload = json.loads(resp.choices[0].message.content)

        why = (payload.get("why") or "").strip()
        examples = [str(x).strip() for x in (payload.get("examples") or []) if str(x).strip()]

        def _clean(s: str) -> str:
            s = s.replace("—", "-").replace(";", ",").replace('"', "").replace("'", "")
            s = " ".join(s.split())
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            if s and s[-1] not in ".!?":
                s += "."
            return s

        ok_examples = []
        for s in examples[:2]:
            w = s.lower().split()
            if (correct_word.lower() in w) and (7 <= len(w) <= 13) and (headword.lower() not in w):
                ok_examples.append(_clean(s))
        while len(ok_examples) < 2:
            ok_examples.append(_clean(
                random.choice([
                    f"I feel {correct_word} when my team wins.",
                    f"My friend was {correct_word} after the good news.",
                    f"The class grew {correct_word} during the fun activity.",
                    f"Dad looked {correct_word} when he saw my drawing.",
                ])
            ))

        if not why:
            why = f"'{correct_word}' means nearly the same as '{headword}', so it fits here."

        return why, ok_examples[:2]

    except Exception:
        return _fallback()

# Ensure a default admin exists
ensure_admin()

# ─────────────────────────────────────────────────────────────────────
# Tweaks requested — safe helpers
# ─────────────────────────────────────────────────────────────────────
def _hide_default_h1_and_set(title_text: str):
    # Hide the first-level Streamlit title (h1) and set our own
    st.markdown("""
        <style>
        h1 {display:none;}
        </style>
    """, unsafe_allow_html=True)
    st.title(title_text)

# ─────────────────────────────────────────────────────────────────────
# Teacher UI V2 helpers (caching + CRUD)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def td2_get_courses():
    return pd.read_sql(text("SELECT course_id, title, description FROM courses ORDER BY title"), con=engine)

@st.cache_data(ttl=10)
def td2_get_lessons(course_id: int):
    return pd.read_sql(text("""
        SELECT lesson_id, title, sort_order
        FROM lessons WHERE course_id=:c ORDER BY sort_order, lesson_id
    """), con=engine, params={"c": int(course_id)})

@st.cache_data(ttl=10)
def td2_get_active_students():
    return pd.read_sql(text("""
        SELECT user_id, name, email FROM users
        WHERE role='student' AND is_active=TRUE
        ORDER BY name
    """), con=engine)

@st.cache_data(ttl=10)
def td2_get_enrollments_for_course(course_id: int):
    return pd.read_sql(text("""
        SELECT E.user_id, U.name, U.email
        FROM enrollments E JOIN users U ON U.user_id=E.user_id
        WHERE E.course_id=:c ORDER BY U.name
    """), con=engine, params={"c": int(course_id)})

def td2_invalidate():
    st.cache_data.clear()

def td2_save_course_edits(df):
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(text("""
                UPDATE courses SET title=:t, description=:d WHERE course_id=:c
            """), {"t": str(r["title"]).strip(), "d": str(r.get("description") or "").strip(),
                   "c": int(r["course_id"])})

def td2_save_lesson_edits(course_id: int, df):
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(text("""
                UPDATE lessons SET title=:t, sort_order=:o
                WHERE lesson_id=:l AND course_id=:c
            """), {"t": str(r["title"]).strip(), "o": int(r.get("sort_order") or 0),
                   "l": int(r["lesson_id"]), "c": int(course_id)})

def td2_delete_course(course_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM courses WHERE course_id=:c"), {"c": int(course_id)})

def td2_delete_lesson(lesson_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM lessons WHERE lesson_id=:l"), {"l": int(lesson_id)})

def td2_import_words_csv(lesson_id: int, df_csv: pd.DataFrame, replace: bool):
    with engine.begin() as conn:
        if replace:
            conn.execute(text("DELETE FROM lesson_words WHERE lesson_id=:l"), {"l": int(lesson_id)})

        n = 0
        for _, r in df_csv.iterrows():
            hw = str(r.get("headword") or "").strip()
            syns = str(r.get("synonyms") or "").strip()
            if not hw or not syns:
                continue
            syn_list = [s.strip() for s in syns.split(",") if s.strip()]
            diff = 1 if (len(hw) <= 6 and len(syn_list) <= 3) else (2 if len(hw) <= 8 and len(syn_list) <= 5 else 3)

            wid = conn.execute(text("""
                INSERT INTO words(headword, synonyms, difficulty)
                VALUES(:h,:s,:d)
                ON CONFLICT DO NOTHING
                RETURNING word_id
            """), {"h": hw, "s": ", ".join(syn_list), "d": int(diff)}).scalar()
            if wid is None:
                wid = conn.execute(text("""
                    SELECT word_id FROM words WHERE headword=:h AND synonyms=:s
                """), {"h": hw, "s": ", ".join(syn_list)}).scalar()
                if wid is None:
                    continue

            conn.execute(text("""
                INSERT INTO lesson_words(lesson_id, word_id, sort_order)
                VALUES(:l,:w,:o)
                ON CONFLICT (lesson_id, word_id) DO NOTHING
            """), {"l": int(lesson_id), "w": int(wid), "o": int(n)})
            n += 1
    return n

def td2_import_course_csv(course_id: int, df_csv: pd.DataFrame,
                          refresh: bool, create_missing_lessons: bool = True):
    """
    Bulk course import: CSV columns lesson_title, headword, synonyms[, sort_order]
    refresh=True → clears words for lessons present in the file before importing (per-lesson refresh)
    """
    if df_csv is None or df_csv.empty:
        return 0, 0
    df = df_csv.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    required = {"lesson_title", "headword", "synonyms"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must have columns: lesson_title, headword, synonyms (optional: sort_order)")

    df["lesson_title"] = df["lesson_title"].astype(str).str.strip()
    df["headword"]     = df["headword"].astype(str).str.strip()
    df["synonyms"]     = df["synonyms"].astype(str).str.strip()
    if "sort_order" not in df.columns:
        df["sort_order"] = 0

    df_less = pd.read_sql(
        text("SELECT lesson_id, title FROM lessons WHERE course_id=:c"),
        con=engine, params={"c": int(course_id)}
    )
    title_to_id = {t.strip().lower(): int(lid) for lid, t in zip(df_less["lesson_id"], df_less["title"])}

    words_imported = 0
    lessons_created = 0
    pos_by_lid = {}

    with engine.begin() as conn:
        if refresh:
            titles_in_file = sorted(set(df["lesson_title"].str.lower()))
            lids_to_clear = [title_to_id.get(t) for t in titles_in_file if title_to_id.get(t) is not None]
            for lid in lids_to_clear:
                conn.execute(text("DELETE FROM lesson_words WHERE lesson_id=:l"), {"l": int(lid)})

        for _, r in df.iterrows():
            lt = r["lesson_title"]
            if not lt:
                continue
            key = lt.lower()
            lid = title_to_id.get(key)

            if lid is None and create_missing_lessons:
                so = int(r.get("sort_order") or 0)
                lid = conn.execute(
                    text("INSERT INTO lessons(course_id,title,sort_order) VALUES(:c,:t,:o) RETURNING lesson_id"),
                    {"c": int(course_id), "t": lt, "o": so}
                ).scalar()
                title_to_id[key] = lid
                lessons_created += 1
                pos_by_lid[lid] = 0

            if lid is None:
                continue

            hw, syns = r["headword"], r["synonyms"]
            if not hw or not syns:
                continue

            syn_list = [s.strip() for s in syns.split(",") if s.strip()]
            diff = 1 if (len(hw) <= 6 and len(syn_list) <= 3) else (2 if len(hw) <= 8 and len(syn_list) <= 5 else 3)

            wid = conn.execute(text("""
                INSERT INTO words(headword, synonyms, difficulty)
                VALUES(:h,:s,:d)
                ON CONFLICT DO NOTHING
                RETURNING word_id
            """), {"h": hw, "s": ", ".join(syn_list), "d": int(diff)}).scalar()

            if wid is None:
                wid = conn.execute(
                    text("SELECT word_id FROM words WHERE headword=:h AND synonyms=:s"),
                    {"h": hw, "s": ", ".join(syn_list)}
                ).scalar()
                if wid is None:
                    continue

            pos_by_lid.setdefault(lid, 0)
            conn.execute(text("""
                INSERT INTO lesson_words(lesson_id, word_id, sort_order)
                VALUES(:l,:w,:o)
                ON CONFLICT (lesson_id, word_id) DO NOTHING
            """), {"l": int(lid), "w": int(wid), "o": int(pos_by_lid[lid])})
            pos_by_lid[lid] += 1
            words_imported += 1

    return words_imported, lessons_created

# ─────────────────────────────────────────────────────────────────────
# Teacher UI V2 — Create / Manage
# ─────────────────────────────────────────────────────────────────────
def teacher_create_ui():
    st.subheader("Create")
    c1, c2 = st.columns(2)

    # New Course
    with c1, st.form("td2_create_course"):
        st.markdown("**New course**")
        title = st.text_input("Title", key="td2_new_course_title")
        desc  = st.text_area("Description", key="td2_new_course_desc")
        if st.form_submit_button("Create course", type="primary"):
            if title.strip():
                with engine.begin() as conn:
                    conn.execute(text("INSERT INTO courses(title, description) VALUES(:t,:d)"),
                                 {"t": title.strip(), "d": desc.strip()})
                td2_invalidate()
                st.success("Course created.")
                st.rerun()
            else:
                st.error("Title is required.")

    # New Lesson
    with c2, st.form("td2_create_lesson"):
        st.markdown("**New lesson**")
        dfc = td2_get_courses()
        if dfc.empty:
            st.info("Create a course first.")
        else:
            cid = st.selectbox("Course", dfc["course_id"].tolist(),
                               format_func=lambda x: dfc.loc[dfc["course_id"]==x, "title"].values[0],
                               key="td2_lesson_course")
            lt  = st.text_input("Lesson title", key="td2_lesson_title")
            so  = st.number_input("Sort order", 0, 999, 0, key="td2_lesson_sort")
            if st.form_submit_button("Create lesson", type="primary"):
                if lt.strip():
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO lessons(course_id, title, sort_order)
                            VALUES(:c,:t,:o)
                        """), {"c": int(cid), "t": lt.strip(), "o": int(so)})
                    td2_invalidate()
                    st.success("Lesson created.")
                    st.rerun()
                else:
                    st.error("Lesson title is required.")

def teacher_manage_ui():
    st.subheader("Manage")
    dfc = td2_get_courses()
    c1, c2, c3 = st.columns([1.2, 1.4, 1.2])

    # COL 1 — Courses list + inline edit + delete
    with c1:
        st.markdown("**Courses**")
        if dfc.empty:
            st.info("No courses yet.")
        else:
            q = st.text_input("Search", key="td2_course_q")
            dfc_view = dfc.copy()
            if q.strip():
                m = dfc_view["title"].str.contains(q, case=False, na=False) | dfc_view["description"].fillna("").str.contains(q, case=False, na=False)
                dfc_view = dfc_view[m]

            edited = st.data_editor(
                dfc_view[["course_id", "title", "description"]].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                key="td2_courses_editor",
                column_config={
                    "course_id": st.column_config.NumberColumn("ID", disabled=True),
                    "title": st.column_config.TextColumn("Title"),
                    "description": st.column_config.TextColumn("Description"),
                }
            )
            if st.button("Save course edits", key="td2_save_courses"):
                td2_save_course_edits(edited)
                td2_invalidate()
                st.success("Courses updated.")
                st.rerun()

            with st.expander("Delete a course"):
                cid_del = st.selectbox("Course", dfc["course_id"].tolist(),
                                       format_func=lambda x: dfc.loc[dfc["course_id"]==x, "title"].values[0],
                                       key="td2_course_delete_sel")
                confirm = st.text_input("Type DELETE to confirm", key="td2_course_delete_confirm")
                if st.button("Delete course", type="secondary", key="td2_course_delete_btn"):
                    if confirm.strip().upper() == "DELETE":
                        td2_delete_course(cid_del)
                        td2_invalidate()
                        st.success("Course deleted.")
                        st.rerun()
                    else:
                        st.error("Please type DELETE to confirm.")

    # COL 2 — Lessons for selected course + upload/replace + BULK IMPORT
    with c2:
        st.markdown("**Lessons**")
        if dfc.empty:
            st.info("Create a course first.")
            cid_sel = None
        else:
            cid_sel = st.selectbox("Course", dfc["course_id"].tolist(),
                                   format_func=lambda x: dfc.loc[dfc["course_id"]==x, "title"].values[0],
                                   key="td2_lessons_course_sel")

        if cid_sel is not None:
            dfl = td2_get_lessons(cid_sel)
            if dfl.empty:
                st.info("No lessons yet for this course.")
            else:
                edited_l = st.data_editor(
                    dfl[["lesson_id", "title", "sort_order"]].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    key="td2_lessons_editor",
                    column_config={
                        "lesson_id": st.column_config.NumberColumn("ID", disabled=True),
                        "title": st.column_config.TextColumn("Title"),
                        "sort_order": st.column_config.NumberColumn("Order", min_value=0, step=1),
                    }
                )
                if st.button("Save lesson edits", key="td2_save_lessons"):
                    td2_save_lesson_edits(cid_sel, edited_l)
                    td2_invalidate()
                    st.success("Lessons updated.")
                    st.rerun()

            with st.expander("Delete a lesson"):
                dfl_del = td2_get_lessons(cid_sel)
                if dfl_del.empty:
                    st.caption("No lessons.")
                else:
                    lid_del = st.selectbox("Lesson", dfl_del["lesson_id"].tolist(),
                                           format_func=lambda x: dfl_del.loc[dfl_del["lesson_id"]==x, "title"].values[0],
                                           key="td2_lesson_delete_sel")
                    confirm_l = st.text_input("Type DELETE to confirm", key="td2_lesson_delete_confirm")
                    if st.button("Delete lesson", type="secondary", key="td2_lesson_delete_btn"):
                        if confirm_l.strip().upper() == "DELETE":
                            td2_delete_lesson(lid_del)
                            td2_invalidate()
                            st.success("Lesson deleted.")
                            st.rerun()
                        else:
                            st.error("Please type DELETE to confirm.")

            # Upload CSV (append/replace) — per lesson
            with st.form("td2_upload_csv_form_single"):
                st.markdown("**Upload words CSV (headword,synonyms)**")
                f = st.file_uploader("CSV file", type=["csv"], key="td2_upload_csv")
                replace = st.checkbox("Replace existing words in this lesson", value=False, key="td2_replace_mode")
                lid_target = None
                dfl2 = td2_get_lessons(cid_sel) if cid_sel is not None else pd.DataFrame()
                if not dfl2.empty:
                    lid_target = st.selectbox("Target lesson", dfl2["lesson_id"].tolist(),
                                              format_func=lambda x: dfl2.loc[dfl2["lesson_id"]==x, "title"].values[0],
                                              key="td2_upload_lesson_sel")
                submit = st.form_submit_button("Import words")
                if submit:
                    if f is None or lid_target is None:
                        st.error("Please choose a CSV file and a lesson.")
                    else:
                        try:
                            df_csv = pd.read_csv(f)
                            ok_cols = set([c.lower().strip() for c in df_csv.columns])
                            if not {"headword","synonyms"}.issubset(ok_cols):
                                st.error("CSV must have columns: headword, synonyms")
                            else:
                                df_csv.columns = [c.lower().strip() for c in df_csv.columns]
                                n = td2_import_words_csv(int(lid_target), df_csv, replace)
                                td2_invalidate()
                                st.success(f"Imported {n} words.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Import failed: {e}")

            # --- Bulk import for entire course (multi-lesson) ---
            with st.form("td2_course_bulk_import"):
                st.markdown("**Bulk import entire course (multi-lesson CSV)**")
                st.caption("CSV columns: lesson_title, headword, synonyms  (optional: sort_order)")
                f2 = st.file_uploader("Course CSV", type=["csv"], key="td2_course_csv")
                refresh_course = st.checkbox("Refresh matching lessons (clear words first)", value=False, key="td2_course_refresh")
                create_missing = st.checkbox("Create missing lessons automatically", value=True, key="td2_course_create")
                go2 = st.form_submit_button("Import course CSV")
                if go2:
                    if f2 is None:
                        st.error("Choose a CSV file.")
                    else:
                        try:
                            df_bulk = pd.read_csv(f2)
                            n_words, n_lessons = td2_import_course_csv(int(cid_sel), df_bulk, refresh_course, create_missing)
                            td2_invalidate()
                            st.success(f"Imported {n_words} words; created {n_lessons} new lessons.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Bulk import failed: {e}")

# COL 3 — Assign / remove students for selected course
    with c3:
        st.markdown("**Assign students**")
        if dfc.empty:
            st.info("Create a course first.")
        else:
            cid_assign = st.selectbox(
                "Course",
                dfc["course_id"].tolist(),
                format_func=lambda x: dfc.loc[dfc["course_id"] == x, "title"].values[0],
                key="td2_assign_course_sel"
        )

        # Now we are INSIDE the same 'else:' block ↓
        df_students = td2_get_active_students()
        df_enrolled = pd.DataFrame()  # default to avoid NameError

        if df_students.empty:
            st.caption("No active students.")
        else:
            sid_assign = st.selectbox(
                "Student",
                df_students["user_id"].tolist(),
                format_func=lambda x: f"{df_students.loc[df_students['user_id'] == x, 'name'].values[0]} "
                                      f"({df_students.loc[df_students['user_id'] == x, 'email'].values[0]})",
                key="td2_assign_student_sel"
            )

            if st.button("Enroll", key="td2_assign_enroll_btn"):
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO enrollments(user_id, course_id)
                        VALUES(:u, :c)
                        ON CONFLICT (user_id, course_id) DO NOTHING
                    """), {"u": int(sid_assign), "c": int(cid_assign)})
                td2_invalidate()
                st.success("Enrolled.")

            st.markdown("**Currently enrolled**")
            df_enrolled = td2_get_enrollments_for_course(cid_assign)

        # Safe checks for enrolled list
        if df_enrolled.empty:
            st.caption("None yet.")
        else:
            to_remove = st.multiselect(
                "Remove students",
                df_enrolled["user_id"].tolist(),
                format_func=lambda x: f"{df_enrolled.loc[df_enrolled['user_id'] == x, 'name'].values[0]} "
                                      f"({df_enrolled.loc[df_enrolled['user_id'] == x, 'email'].values[0]})",
                key="td2_assign_remove"
            )
            if st.button("Remove selected", key="td2_assign_remove_btn"):
                with engine.begin() as conn:
                    for sid in to_remove:
                        conn.execute(
                            text("DELETE FROM enrollments WHERE user_id=:u AND course_id=:c"),
                            {"u": int(sid), "c": int(cid_assign)}
                        )
                td2_invalidate()
                st.success("Removed.")
                st.rerun()


def render_teacher_dashboard_v2():
    """Render the teacher dashboard experience using the v2 helper routines."""

    st.markdown("### Teacher workspace")
    st.caption("Create courses, organise lessons, and manage student enrolments from one place.")

    try:
        summary = pd.read_sql(
            text(
                """
                SELECT
                  (SELECT COUNT(*) FROM courses) AS courses,
                  (SELECT COUNT(*) FROM lessons) AS lessons,
                  (SELECT COUNT(*) FROM words)   AS words,
                  (SELECT COUNT(*) FROM enrollments) AS enrollments
                """
            ),
            con=engine,
        )
    except Exception:
        summary = pd.DataFrame()

    if not summary.empty:
        c_courses, c_lessons, c_words, c_enroll = st.columns(4)
        c_courses.metric("Courses", int(summary.iloc[0]["courses"]))
        c_lessons.metric("Lessons", int(summary.iloc[0]["lessons"]))
        c_words.metric("Words", int(summary.iloc[0]["words"]))
        c_enroll.metric("Enrollments", int(summary.iloc[0]["enrollments"]))

    tab_create, tab_manage, tab_help = st.tabs(["Create", "Manage", "Help"])

    with tab_create:
        teacher_create_ui()

    with tab_manage:
        teacher_manage_ui()

    with tab_help:
        st.markdown(
            """
            **CSV Tips**

            * Words CSV files must include `headword` and `synonyms` columns.
            * Bulk course imports support `lesson_title`, `headword`, `synonyms`, and optional `sort_order`.
            * Use the refresh checkbox to replace existing lesson vocabulary when re-importing.

            **Need a reset?** Use the delete expanders inside the *Manage* tab to remove courses or lessons.
            """
        )
# ─────────────────────────────────────────────────────────────────────
# AUTH INTEGRATION (optional / append-only)
# ─────────────────────────────────────────────────────────────────────
try:
    from auth_service import AuthService
    auth = AuthService(engine)
except Exception as _e:
    auth = None
    st.sidebar.warning("Auth service not initialized. Check auth_service.py")

# ─────────────────────────────────────────────────────────────────────
# Login / Session
# ─────────────────────────────────────────────────────────────────────
def login_form():
    # Don't reference auth in a way that displays it
    # Just use it internally
    
    # Get or create auth without triggering display
    if 'auth_service' not in st.session_state:
        try:
            from auth_service import AuthService
            st.session_state.auth_service = AuthService(engine)
        except Exception:
            st.session_state.auth_service = None
    
    auth_svc = st.session_state.auth_service
    
    # Optional: Add a warning if auth service isn't available
    if not auth_svc:
        st.sidebar.warning("Authentication service unavailable")
    
    st.sidebar.subheader("Sign in")

    try:
        qp = st.query_params
    except Exception:
        qp = st.experimental_get_query_params()

    def _first(qv):
        if qv is None: return None
        if isinstance(qv, list): return qv[0]
        return qv

    # (Reset by URL disabled for now)
    # reset_email = (_first(qp.get("reset_email")) or "").strip().lower()
    # reset_token = (_first(qp.get("reset_token")) or "").strip()

    mode = "Student" if FORCE_STUDENT else st.sidebar.radio(
        "Login as", ["Admin", "Student"], horizontal=True, key="login_mode"
    )
    email = st.sidebar.text_input("Email", key="login_email")
    pwd   = st.sidebar.text_input("Password", type="password", key="login_pwd")

    if st.sidebar.button("Login", type="primary", key="btn_login"):
        u = user_by_email(email.strip().lower())
        if not u:
            st.sidebar.error("User not found."); return
        if not u["is_active"]:
            st.sidebar.error("Account disabled."); return
        if not bcrypt.verify(pwd, u["password_hash"]):
            st.sidebar.error("Wrong password."); return

        # Role enforcement
        if mode == "Admin" and u["role"] != "admin":
            st.sidebar.error("Not an admin account."); return
        if mode == "Student" and u["role"] != "student":
            if FORCE_STUDENT:
                st.sidebar.error("This is a student-only link. Please use the admin URL."); return
            st.sidebar.error("Not a student account."); return

        # Expiry enforcement for students
        if auth and u["role"] == "student":
            try:
                if auth.is_student_expired(u):
                    st.sidebar.error("Your account has expired. Ask your teacher to reopen access.")
                    return
            except Exception:
                pass

        st.session_state.auth = {
            "user_id": u["user_id"],
            "name": u["name"],
            "email": u["email"],
            "role": u["role"],
        }
        st.sidebar.success(f"Welcome {u['name']}!")

    # Forgot password — email flow disabled for this release
    # with st.sidebar.expander("Forgot password?"):
    #     ...

    if st.sidebar.button("Log out", key="btn_logout"):
        st.session_state.pop("auth", None)

# Gate: not logged in yet
if "auth" not in st.session_state:
    login_form()
    st.title("Learning English Made Easy")
    st.write("Sign in as **Admin** to manage students, courses and tests; or as **Student** to learn and take tests.")
    st.sidebar.header("Health")
    if st.sidebar.button("DB ping"):
        try:
            with engine.connect() as conn:
                one = conn.execute(text("SELECT 1")).scalar()
            st.sidebar.success(f"DB OK (result={one})")
        except Exception as e:
            st.sidebar.error(f"DB error: {e}")
    st.stop()

# Session basics
ROLE   = st.session_state.auth["role"]
USER_ID= st.session_state.auth["user_id"]
NAME   = st.session_state.auth["name"]
st.sidebar.caption(f"Signed in as **{NAME}** ({ROLE})")

_defaults = {
    "answered": False, "eval": None, "active_word": None, "active_lid": None,
    "q_started_at": 0.0, "selection": set(), "asked_history": [],
    "gamification": {}, "badges_recent": [], "badge_details_recent": [], "last_xp_gain": 0,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
# NEW: per-lesson counters and a small review queue
if "q_index_per_lesson" not in st.session_state:
    st.session_state.q_index_per_lesson = {}   # {lesson_id: current_index_int}

if "review_queue" not in st.session_state:
    from collections import deque
    st.session_state.review_queue = deque()    # list of headwords to retry soon

# Enforce expiry AFTER any login (auto sign-out)
if auth and st.session_state["auth"]["role"] == "student":
    try:
        _u = user_by_email(st.session_state["auth"]["email"])
        if auth.is_student_expired(_u):
            st.sidebar.error("Your account has expired. Ask your teacher to reopen access.")
            st.session_state.pop("auth", None)
            st.rerun()
    except Exception:
        pass

# Sidebar account tools (change password, optional email reset)
# Sidebar account tools (change password only after login)
if auth and "auth" in st.session_state:
    st.sidebar.markdown("---")
    with st.sidebar.expander("Account"):
        _old = st.text_input("Old password", type="password", key="acct_old_pw")
        _new1 = st.text_input("New password", type="password", key="acct_new_pw1")
        _new2 = st.text_input("Confirm new password", type="password", key="acct_new_pw2")
        if st.button("Change password", key="acct_change_pw_btn"):
            if _new1 != _new2:
                st.warning("New passwords do not match.")
            elif not _old or not _new1:
                st.warning("Please fill all fields.")
            else:
                ok, msg = auth.change_password(st.session_state["auth"]["user_id"], _old, _new1)
                st.success(msg) if ok else st.error(msg)
#-----------------------------------------------------------------------------------------------------
# Admin-only: reopen student (+365 days)
if auth and st.session_state["auth"]["role"] == "admin":
    st.markdown("---")
    st.subheader("Admin: Account Tools")
    _adm_df = pd.read_sql(
        text("SELECT user_id, name, email, expires_at, is_active FROM users WHERE role='student' ORDER BY name"),
        con=engine
    )
    if _adm_df.empty:
        st.info("No students yet.")
    else:
        _sel = st.selectbox(
            "Select student",
            _adm_df["user_id"].tolist(),
            format_func=lambda x: f"{_adm_df.loc[_adm_df['user_id']==x,'name'].values[0]}  "
                                  f"({_adm_df.loc[_adm_df['user_id']==x,'email'].values[0]})",
            key="admin_tools_student"
        )
        _row = _adm_df[_adm_df["user_id"]==_sel].iloc[0]
        st.caption(f"Status: {'Active' if _row['is_active'] else 'Disabled'} • Expires at: {str(_row['expires_at'])}")
        if st.button("Reopen +365 days", key="btn_reopen_365"):
            ok, msg = auth.reopen_student(int(_sel), days=365) if auth else (False, "Auth disabled")
            st.success(msg) if ok else st.error(msg)

# Optional: SMTP diagnostics (keep as-is)
if st.session_state["auth"]["role"] == "admin":
    with st.expander("Email / SMTP Diagnostics"):
        import ssl, smtplib
        from email.message import EmailMessage
        host = os.getenv("SMTP_HOST"); port = os.getenv("SMTP_PORT")
        user = os.getenv("SMTP_USER"); pwd = os.getenv("SMTP_PASS")
        sender = os.getenv("SMTP_FROM"); base = os.getenv("APP_BASE_URL")
        st.write(f"APP_BASE_URL: {base or '(empty)'}")
        st.write(f"SMTP_HOST: {host or '(empty)'}")
        st.write(f"SMTP_PORT: {port or '(empty)'}")
        st.write(f"SMTP_USER: {user or '(empty)'}")
        st.write(f"SMTP_FROM: {sender or '(empty)'}")
        to_addr = st.text_input("Send a test email to:", value=(sender or ""))
        if st.button("Send SMTP test"):
            try:
                msg = EmailMessage()
                msg["Subject"] = "SMTP test — English Learning Made Easy"
                msg["From"] = sender; msg["To"] = to_addr
                msg.set_content("If you see this, SMTP is working from Render.")
                with smtplib.SMTP(host, int(port)) as s:
                    s.starttls(context=ssl.create_default_context())
                    s.login(user, pwd)
                    s.send_message(msg)
                st.success("✅ Test email sent. Check inbox and SendGrid → Email Activity.")
            except Exception as e:
                st.error(f"❌ SMTP error: {e}")

# ─────────────────────────────────────────────────────────────────────
# Course Progress
#---------------------------------------------------------------------
def course_progress(user_id: int, course_id: int):
    """
    Attempted-aware progress for the sidebar.
    - If the user has mastered ≥1 word, show mastered% (mastered/total).
    - Otherwise show attempted% (attempted/total).
    Returns: (mastered_count, total_words, percent_int)
    """
    all_words = pd.read_sql(
        text("""
            SELECT w.headword
            FROM lessons L
            JOIN lesson_words lw ON lw.lesson_id = L.lesson_id
            JOIN words w ON w.word_id = lw.word_id
            WHERE L.course_id = :c
        """),
        con=engine, params={"c": int(course_id)}
    )["headword"].tolist()

    total = len(set(all_words))
    if total == 0:
        return (0, 0, 0)

    df_row = pd.read_sql(
        text("""
            SELECT
              SUM(CASE WHEN mastered THEN 1 ELSE 0 END) AS mastered_count,
              SUM(CASE WHEN total_attempts > 0 THEN 1 ELSE 0 END) AS attempted_count
            FROM word_stats
            WHERE user_id = :u AND headword = ANY(:arr)
        """),
        con=engine, params={"u": int(user_id), "arr": list(set(all_words))}
    )

    if df_row.empty:
        mastered = 0
        attempted = 0
    else:
        mastered  = int(df_row.iloc[0]["mastered_count"]  or 0)
        attempted = int(df_row.iloc[0]["attempted_count"] or 0)

    basis = mastered if mastered > 0 else attempted
    percent = int(round(100 * min(basis, total) / total))
    return (mastered, total, percent)
#--------------------------------------------------------------------------
# App routing by role
# ─────────────────────────────────────────────────────────────────────

if st.session_state["auth"]["role"] == "admin":
    _hide_default_h1_and_set("welcome to English Learning made easy - Teacher Console")
    tab_admin, tab_teacher, tab_student = st.tabs(["Admin Section","Teacher Dashboard","Student Dashboard"])

    # ==== LEGACY ADMIN SECTION (commented for v3 redesign) ====
    # with tab_admin:
    #     st.subheader("Manage Students")
    #     df = all_students_df()
    #     st.dataframe(df, use_container_width=True)
    #
    #     st.markdown("**Create Student**")
    #     with st.form("create_student"):
    #         c1,c2,c3=st.columns(3)
    #         with c1: s_name  = st.text_input("Name", key="adm_create_name")
    #         with c2: s_email = st.text_input("Email", key="adm_create_email")
    #         with c3: s_pwd   = st.text_input("Temp Password", value="Learn123!", type="password", key="adm_create_pwd")
    #         go = st.form_submit_button("Create")
    #         if go and s_name and s_email and s_pwd:
    #             try:
    #                 create_user(s_name, s_email.strip().lower(), s_pwd, "student")
    #                 st.success("Student created.")
    #             except Exception as ex:
    #                 st.error(f"Could not create user: {ex}")
    #
    #     if not df.empty:
    #         st.markdown("**Enable / Disable**")
    #         sid = st.selectbox(
    #             "Student",
    #             df["user_id"].tolist(),
    #             format_func=lambda x: df.loc[df["user_id"]==x,"name"].values[0],
    #             key="admin_toggle_student"
    #         )
    #         active = st.radio("Status", ["Enable","Disable"], horizontal=True, key="admin_status_radio")
    #         if st.button("Apply status", key="admin_apply_status"):
    #             set_user_active(sid, active=="Enable"); st.success("Updated.")

    # ==== START: ADMIN CONSOLE v3 (Sprint 1) ====
    with tab_admin:
        tab_students, tab_teachers, tab_courses = st.tabs(["Students", "Teachers", "Courses & Lessons"])

        # ───────────────────────────────────────────────
        # TAB 1 — STUDENTS MANAGEMENT
        # ───────────────────────────────────────────────
        with tab_students:
            st.subheader("👩‍🎓 Students Management")

            search_q = st.text_input("Search student by name or email", key="adm_stu_search")
            df_students = all_students_df()
            if search_q.strip():
                m = df_students["name"].str.contains(search_q, case=False, na=False) | \
                    df_students["email"].str.contains(search_q, case=False, na=False)
                df_students = df_students[m]
            st.dataframe(df_students, use_container_width=True)

            st.markdown("### ➕ Add / Enroll Student")
            with st.form("adm_add_student"):
                c1, c2, c3 = st.columns(3)
                with c1: s_name = st.text_input("Name")
                with c2: s_email = st.text_input("Email")
                with c3: s_pwd = st.text_input("Temp Password", value="Learn123!", type="password")
                if st.form_submit_button("Create Student", type="primary"):
                    if s_name and s_email:
                        try:
                            create_user(s_name, s_email.strip().lower(), s_pwd, "student")
                            st.success("✅ Student created successfully.")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Creation failed: {ex}")
                    else:
                        st.warning("Please fill all fields.")

            st.markdown("### ⚙️ Manage Status")
            if not df_students.empty:
                selected_ids = st.multiselect(
                    "Select students",
                    df_students["user_id"].tolist(),
                    format_func=lambda x: f"{df_students.loc[df_students['user_id']==x,'name'].values[0]}"
                )
                action = st.selectbox("Action", ["Deactivate", "Reactivate", "Delete"])
                if st.button("Apply Action", type="primary"):
                    with engine.begin() as conn:
                        for sid in selected_ids:
                            if action == "Deactivate":
                                conn.execute(text("UPDATE users SET is_active=FALSE WHERE user_id=:u"), {"u": sid})
                            elif action == "Reactivate":
                                conn.execute(text("UPDATE users SET is_active=TRUE WHERE user_id=:u"), {"u": sid})
                            elif action == "Delete":
                                conn.execute(text("DELETE FROM users WHERE user_id=:u AND role='student'"), {"u": sid})
                    st.success(f"{action} applied to {len(selected_ids)} student(s).")
                    st.rerun()
            else:
                st.info("No students available yet.")

            st.markdown("### 🏫 Classrooms")
            show_archived = st.checkbox("Show archived classes", value=False, key="adm_show_archived_classes")
            df_classes = get_classrooms(include_archived=show_archived)
            if df_classes.empty:
                st.info("No classrooms yet. Create one below.")
            else:
                df_display = df_classes.copy()
                for col in ["start_date", "created_at", "archived_at"]:
                    if col in df_display:
                        df_display[col] = df_display[col].astype(str)
                st.dataframe(df_display, use_container_width=True)

            with st.form("adm_create_classroom"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    class_name = st.text_input("Class name")
                with c2:
                    default_date = date.today()
                    class_start = st.date_input("Commencement date", value=default_date)
                if st.form_submit_button("Create Classroom", type="primary"):
                    if class_name and class_name.strip():
                        create_classroom(class_name.strip(), class_start)
                        st.success("Classroom created.")
                        st.rerun()
                    else:
                        st.warning("Please provide a class name.")

            if not df_classes.empty:
                st.markdown("#### Manage classroom roster")
                class_options = df_classes["class_id"].tolist()
                selected_class = st.selectbox(
                    "Select classroom",
                    class_options,
                    format_func=lambda x: f"{df_classes.loc[df_classes['class_id']==x,'name'].values[0]}",
                    key="adm_class_select",
                )

                class_row = df_classes[df_classes["class_id"] == selected_class].iloc[0]
                start_label = class_row.get("start_date")
                status_label = "Archived" if class_row.get("is_archived") else "Active"
                st.caption(
                    f"Status: **{status_label}** • Commences: {start_label if start_label else 'TBD'}"
                )

                class_students_df = get_class_students(int(selected_class))
                if class_students_df.empty:
                    st.info("No students assigned yet.")
                else:
                    df_roster = class_students_df.copy()
                    df_roster["assigned_at"] = df_roster["assigned_at"].astype(str)
                    st.dataframe(df_roster[["name", "email", "is_active", "assigned_at"]], use_container_width=True)

                current_student_ids = (
                    class_students_df["user_id"].tolist() if not class_students_df.empty else []
                )
                available_students = df_students[~df_students["user_id"].isin(current_student_ids)]
                with st.form("adm_update_class_roster"):
                    add_choices = available_students["user_id"].tolist()
                    add_selection = st.multiselect(
                        "Add students",
                        add_choices,
                        format_func=lambda x: f"{available_students.loc[available_students['user_id']==x,'name'].values[0]}"
                        if not available_students.empty else str(x),
                    )
                    remove_selection = st.multiselect(
                        "Remove students",
                        class_students_df["user_id"].tolist() if not class_students_df.empty else [],
                        format_func=lambda x: f"{class_students_df.loc[class_students_df['user_id']==x,'name'].values[0]}"
                        if not class_students_df.empty else str(x),
                    )
                    if st.form_submit_button("Update Classroom", type="primary"):
                        assign_students_to_class(int(selected_class), add_selection)
                        unassign_students_from_class(int(selected_class), remove_selection)
                        st.success("Classroom roster updated.")
                        st.rerun()

                archive_label = "Restore Classroom" if class_row.get("is_archived") else "Archive Classroom"
                if st.button(archive_label, key="adm_toggle_archive_class", type="secondary"):
                    current_archived = bool(class_row.get("is_archived"))
                    set_class_archived(int(selected_class), not current_archived)
                    st.success("Classroom archived." if not current_archived else "Classroom restored.")
                    st.rerun()

        # ───────────────────────────────────────────────
        # TAB 2 — TEACHERS MANAGEMENT
        # ───────────────────────────────────────────────
        with tab_teachers:
            st.subheader("👨‍🏫 Teachers Management")
            df_teachers = pd.read_sql(
                text("SELECT user_id,name,email,is_active FROM users WHERE role='admin' ORDER BY name"),
                con=engine
            )
            st.dataframe(df_teachers, use_container_width=True)

            st.markdown("### ➕ Add Teacher Account")
            with st.form("adm_add_teacher"):
                c1, c2 = st.columns(2)
                with c1: t_name = st.text_input("Name")
                with c2: t_email = st.text_input("Email")
                pwd = st.text_input("Temp Password", value="Teach123!", type="password")
                if st.form_submit_button("Create Teacher", type="primary"):
                    if t_name and t_email:
                        try:
                            create_user(t_name, t_email.strip().lower(), pwd, "admin")
                            st.success("✅ Teacher account created.")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Failed: {ex}")
                    else:
                        st.warning("Please fill all fields.")

        # ───────────────────────────────────────────────
        # TAB 3 — COURSES & LESSONS OVERVIEW
        # ───────────────────────────────────────────────
        with tab_courses:
            st.subheader("📘 Courses & Lessons Overview")
            df_courses = td2_get_courses()
            if df_courses.empty:
                st.info("No courses found. Create new ones in the Teacher Dashboard.")
            else:
                search_course = st.text_input("Search course")
                if search_course.strip():
                    m = df_courses["title"].str.contains(search_course, case=False, na=False)
                    df_courses = df_courses[m]
                st.dataframe(df_courses, use_container_width=True)

                st.markdown("### 🧾 Quick Actions")
                selected_course = st.selectbox(
                    "Select course for details",
                    df_courses["course_id"].tolist(),
                    format_func=lambda x: df_courses.loc[df_courses["course_id"]==x,"title"].values[0],
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("View Lessons"):
                        dfl = td2_get_lessons(selected_course)
                        st.dataframe(dfl, use_container_width=True)
                with c2:
                    if st.button("View Enrolled Students"):
                        dfe = td2_get_enrollments_for_course(selected_course)
                        st.dataframe(dfe, use_container_width=True)

    # ==== END: ADMIN CONSOLE v3 (Sprint 1) ====

    # Teacher Dashboard
    with tab_teacher:
        if TEACHER_UI_V2:
            render_teacher_dashboard_v2()
        else:
            st.info("Legacy Teacher UI is disabled in this version. Set TEACHER_UI_V2=1 to enable V2.")

    # Student Dashboard — admin visibility
    with tab_student:
        st.subheader("Student Overview")
        attempts = pd.read_sql(
            text("""
                SELECT U.name, A.course_id, A.lesson_id, A.headword, A.is_correct, A.response_ms, A.ts
                FROM attempts A JOIN users U ON U.user_id=A.user_id
                ORDER BY A.id DESC LIMIT 500
            """),
            con=engine
        )
        st.dataframe(attempts, use_container_width=True)
        st.caption("Latest attempts across students. Filter/export via table menu.")

        st.markdown("### 🏫 Classrooms Snapshot")
        classes_overview = get_classrooms(include_archived=True)
        if classes_overview.empty:
            st.info("No classrooms have been created yet.")
        else:
            df_classes = classes_overview.copy()
            for col in ["start_date", "created_at", "archived_at"]:
                df_classes[col] = df_classes[col].astype(str)
            st.dataframe(df_classes, use_container_width=True)

            selected_class = st.selectbox(
                "View roster for",
                df_classes["class_id"].tolist(),
                format_func=lambda x: f"{classes_overview.loc[classes_overview['class_id']==x,'name'].values[0]}",
                key="student_tab_class_select",
            )
            roster_df = get_class_students(int(selected_class))
            if roster_df.empty:
                st.info("This classroom has no students assigned yet.")
            else:
                roster_df["assigned_at"] = roster_df["assigned_at"].astype(str)
                st.dataframe(roster_df[["name", "email", "is_active", "assigned_at"]], use_container_width=True)

# Student experience
if st.session_state["auth"]["role"] == "student":
    _hide_default_h1_and_set("welcome to English Learning made easy - Student login")

    st.session_state.gamification = gamification_snapshot(USER_ID)
    recent_badge_names = set(st.session_state.get("badges_recent", []))
    sidebar_card, mobile_card = render_gamification_panels(st.session_state.gamification, recent_badge_names)

    courses = pd.read_sql(
        text("""
            SELECT C.course_id, C.title
            FROM enrollments E JOIN courses C ON C.course_id=E.course_id
            WHERE E.user_id=:u
        """),
        con=engine, params={"u": USER_ID}
    )

    student_classes = get_classes_for_student(USER_ID, include_archived=True)

    selected_course_id = None
    selected_lesson_id = None
    lessons = pd.DataFrame()
    course_lessons: dict[int, pd.DataFrame] = {}

    # Sidebar is truly inside the student block ↓
    with st.sidebar:
        st.subheader("My courses & lessons")
        if courses.empty:
            st.info("No courses assigned yet.")
        else:
            st.markdown(
                """
                <style>
                [data-testid="stSidebar"] .stRadio label {
                    align-items: flex-start;
                }
                [data-testid="stSidebar"] .stRadio label p {
                    white-space: pre-line;
                    margin: 0;
                    font-size: 0.95rem;
                    text-indent: -0.75rem;
                    padding-left: 0.75rem;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            option_pairs: list[tuple[int, int]] = []
            display_map: dict[tuple[int, int], tuple[str, str]] = {}
            course_lessons: dict[int, pd.DataFrame] = {}
            empty_courses: list[str] = []

            prev_course = int(st.session_state.get("active_cid") or 0)
            prev_lesson = st.session_state.get("student_lesson_select")
            default_index = 0

            for _, rowc in courses.iterrows():
                cid = int(rowc["course_id"])
                lesson_df = pd.read_sql(
                    text(
                        """
                        SELECT lesson_id, title
                        FROM lessons
                        WHERE course_id = :c
                        ORDER BY sort_order
                        """
                    ),
                    con=engine,
                    params={"c": cid},
                )
                course_lessons[cid] = lesson_df

                if lesson_df.empty:
                    empty_courses.append(str(rowc["title"]))
                    continue

                for _, lesson_row in lesson_df.iterrows():
                    pair = (cid, int(lesson_row["lesson_id"]))
                    display_map[pair] = (str(rowc["title"]), str(lesson_row["title"]))
                    option_pairs.append(pair)
                    if prev_course == cid and prev_lesson == pair[1]:
                        default_index = len(option_pairs) - 1

            if not option_pairs:
                st.info("No lessons yet for your assigned courses.")
            else:
                labels: dict[tuple[int, int], str] = {}
                last_course_title = ""
                for pair in option_pairs:
                    course_title, lesson_title = display_map[pair]
                    prefix = ""
                    if course_title != last_course_title:
                        prefix = f"* {course_title}\n"
                        last_course_title = course_title
                    labels[pair] = f"{prefix}* {lesson_title}"

                def _format_pair(pair: tuple[int, int]) -> str:
                    return labels.get(pair, "")

                selected_pair = st.radio(
                    "Course and lesson",
                    option_pairs,
                    index=default_index,
                    format_func=_format_pair,
                    key="student_course_lesson",
                    label_visibility="collapsed",
                )

                selected_course_id, selected_lesson_id = selected_pair
                lessons = course_lessons.get(selected_course_id, pd.DataFrame())

                selected_course_title, _ = display_map[selected_pair]
                c_completed, c_total, c_pct = course_progress(USER_ID, int(selected_course_id))
                st.caption(f"Selected: {selected_course_title} — {c_pct}% complete")

                st.session_state["active_cid"] = selected_course_id
                st.session_state["student_lesson_select"] = selected_lesson_id

            if empty_courses:
                st.caption(
                    "\n".join(
                        [
                            "Lessons coming soon:",
                            *[f"• {title}" for title in empty_courses],
                        ]
                    )
                )

    lessons = course_lessons.get(int(selected_course_id), pd.DataFrame()) if selected_course_id else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────
# Helper for lesson progress (canonical — keep only ONE copy in file)
# ─────────────────────────────────────────────────────────────────────
from sqlalchemy import text as sa_text

@st.cache_data(ttl=5)
def lesson_progress(user_id: int, lesson_id: int):
    """
    One-shot, portable computation of lesson progress.
    Returns: (total_words, mastered_count, attempted_count)
    """
    sql = sa_text("""
        SELECT
          COUNT(DISTINCT w.headword) AS total,
          SUM(CASE WHEN s.mastered IS TRUE THEN 1 ELSE 0 END) AS mastered_count,
          SUM(CASE WHEN COALESCE(s.total_attempts,0) > 0 THEN 1 ELSE 0 END) AS attempted_count
        FROM lesson_words lw
        JOIN words w ON w.word_id = lw.word_id
        LEFT JOIN word_stats s
               ON s.user_id = :u
              AND s.headword = w.headword
        WHERE lw.lesson_id = :l
    """)
    df = pd.read_sql(sql, con=engine, params={"u": int(user_id), "l": int(lesson_id)})

    if df.empty:
        return 0, 0, 0

    total     = int(df.iloc[0]["total"] or 0)
    mastered  = int(df.iloc[0]["mastered_count"] or 0)
    attempted = int(df.iloc[0]["attempted_count"] or 0)
    if total <= 0:
        return 0, 0, 0
    return total, mastered, attempted

# ─────────────────────────────────────────────────────────────────────
# UI Helper: compact question header with inline progress bar (theme-agnostic)
# ─────────────────────────────────────────────────────────────────────
DIFFICULTY_THEME = {
    1: {"emoji": "🟢", "label": "Easy", "class": "difficulty-easy"},
    2: {"emoji": "🟠", "label": "Medium", "class": "difficulty-medium"},
    3: {"emoji": "🔴", "label": "Hard", "class": "difficulty-hard"},
}


BADGE_CHIME_BASE64 = (
    "UklGRmQGAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YUAGAAAAAPMwb0tMQ0UcRegpv8+zbssu+yMtYkp9RbIg5+zfwVuzBchg9iYpCUlnR/8knfHUxDSz"
    "1MSd8f8kZ0cJSSYpYPYFyFuz38Hn7LIgfUViSiMtLvtuy8+zKb9F6EUcTENvS/MwAAANz5G0tLy747sX10AxTJI00gTd0p61g7pO3xkTIT6lTPs3oAna1ve2mbgB22MO"
    "LDvMTCw7Yw4B25m497ba1qAJ+zelTCE+GRNO34O6nrXd0tIEkjQxTNdAuxe747S8kbQNzwAA8zBvS0xDRRxF6Cm/z7Nuyy77Iy1iSn1FsiDn7N/BW7MFyGD2JikJSWdH"
    "/ySd8dTENLPUxJ3x/yRnRwlJJilg9gXIW7PfwefssiB9RWJKIy0u+27Lz7Mpv0XoRRxMQ29L8zAAAA3PkbS0vLvjuxfXQDFMkjTSBN3SnrWDuk7fGRMhPqVM+zegCdrW"
    "97aZuAHbYw4sO8xMLDtjDgHbmbj3ttrWoAn7N6VMIT4ZE07fg7qetd3S0gSSNDFM10C7F7vjtLyRtA3PAADzMG9LTENFHEXoKb/Ps27LLvsjLWJKfUWyIOfs38FbswXI"
    "YPYmKQlJZ0f/JJ3x1MQ0s9TEnfH/JGdHCUkmKWD2Bchbs9/B5+yyIH1FYkojLS77bsvPsym/RehFHExDb0vzMAAADc+RtLS8u+O7F9dAMUySNNIE3dKetYO6Tt8ZEyE+"
    "pUz7N6AJ2tb3tpm4AdtjDiw7zEwsO2MOAduZuPe22tagCfs3pUwhPhkTTt+Dup613dLSBJI0MUzXQLsXu+O0vJG0Dc8AAPMwb0tMQ0UcRegpv8+zbssu+yMtYkp9RbIg"
    "5+zfwVuzBchg9iYpCUlnR/8knfHUxDSz1MSd8f8kZ0cJSSYpYPYFyFuz38Hn7LIgfUViSiMtLvtuy8+zKb9F6EUcTENvS/MwAAANz5G0tLy747sX10AxTJI00gTd0p61"
    "g7pO3xkTIT6lTPs3oAna1ve2mbgB22MOLDvMTCw7Yw4B25m497ba1qAJ+zelTCE+GRNO34O6nrXd0tIEkjQxTNdAuxe747S8kbQNzwAA8zBvS0xDRRxF6Cm/z7Nuyy77"
    "Iy1iSn1FsiDn7N/BW7MFyGD2JikJSWdH/ySd8dTENLPUxJ3x/yRnRwlJJilg9gXIW7PfwefssiB9RWJKIy0u+27Lz7Mpv0XoRRxMQ29L8zAAAA3PkbS0vLvjuxfXQDFM"
    "kjTSBN3SnrWDuk7fGRMhPqVM+zegCdrW97aZuAHbYw4sO8xMLDtjDgHbmbj3ttrWoAn7N6VMIT4ZE07fg7qetd3S0gSSNDFM10C7F7vjtLyRtA3PAADzMG9LTENFHEXo"
    "Kb/Ps27LLvsjLWJKfUWyIOfs38FbswXIYPYmKQlJZ0f/JJ3x1MQ0s9TEnfH/JGdHCUkmKWD2Bchbs9/B5+yyIH1FYkojLS77bsvPsym/RehFHExDb0vzMAAADc+RtLS8"
    "u+O7F9dAMUySNNIE3dKetYO6Tt8ZEyE+pUz7N6AJ2tb3tpm4AdtjDiw7zEwsO2MOAduZuPe22tagCfs3pUwhPhkTTt+Dup613dLSBJI0MUzXQLsXu+O0vJG0Dc8AAPMw"
    "b0tMQ0UcRegpv8+zbssu+yMtYkp9RbIg5+zfwVuzBchg9iYpCUlnR/8knfHUxDSz1MSd8f8kZ0cJSSYpYPYFyFuz38Hn7LIgfUViSiMtLvtuy8+zKb9F6EUcTENvS/Mw"
    "AAANz5G0tLy747sX10AxTJI00gTd0p61g7pO3xkTIT6lTPs3oAna1ve2mbgB22MOLDvMTCw7Yw4B25m497ba1qAJ+zelTCE+GRNO34O6nrXd0tIEkjQxTNdAuxe747S8"
    "kbQNzwAA8zBvS0xDRRxF6Cm/z7Nuyy77Iy1iSn1FsiDn7N/BW7MFyGD2JikJSWdH/ySd8dTENLPUxJ3x/yRnRwlJJilg9gXIW7PfwefssiB9RWJKIy0u+27Lz7Mpv0Xo"
    "RRxMQ29L8zAAAA3PkbS0vLvjuxfXQDFMkjTSBN3SnrWDuk7fGRMhPqVM+zegCdrW97aZuAHbYw4sO8xMLDtjDgHbmbj3ttrWoAn7N6VMIT4ZE07fg7qetd3S0gSSNDFM"
    "10C7F7vjtLyRtA3PAADzMG9LTENFHEXoKb/Ps27LLvsjLWJKfUWyIOfs38FbswXIYPYmKQlJZ0f/JJ3x1MQ0s9TEnfH/JGdHCUkmKWD2Bchbs9/B5+yyIH1FYko="
)

BADGE_CHIME_AUDIO = base64.b64decode(BADGE_CHIME_BASE64.encode())

CONFETTI_SNIPPET = """
<script>
(function(){
  const existing = window.__streamlit_confetti__;
  function fire(){
    if (window.confetti) {
      window.confetti({
        particleCount: 120,
        spread: 70,
        origin: { y: 0.6 }
      });
    }
  }
  if (!existing) {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js';
    script.onload = fire;
    document.body.appendChild(script);
    window.__streamlit_confetti__ = true;
  } else {
    fire();
  }
})();
</script>
"""


def render_q_header(
    q_now: int,
    total_q: int,
    pct: int,
    *,
    fill="#3b82f6",
    track_light="#e5e7eb",
    track_dark="#374151",
    login_streak: int = 0,
    badge_strip: list[dict] | None = None,
):
    import math
    import streamlit as st

    total_q = max(1, int(total_q or 1))
    q_now = max(1, min(int(q_now or 1), total_q))
    pct = max(0, min(100, int(math.floor(pct or 0))))

    badge_strip = badge_strip or []
    badge_html = []
    for item in badge_strip[:5]:
        classes = ["qhdr-badge"]
        if item.get("is_new"):
            classes.append("new")
        title = html.escape(item.get("name", ""))
        emoji = html.escape(item.get("emoji", ""))
        badge_html.append(
            f"<span class='{' '.join(classes)}' title='{title}'>{emoji}</span>"
        )
    if not badge_html:
        badge_html.append("<span class='qhdr-badge placeholder'>Earn badges ✨</span>")

    css = f"""
    <style>
      .qhdr {{
        display:flex;
        flex-direction:column;
        gap:10px;
        padding:12px 18px;
        background:rgba(59,130,246,0.08);
        border-radius:16px;
        border:1px solid rgba(59,130,246,0.18);
      }}
      .qhdr-top {{
        display:flex;
        align-items:center;
        gap:12px;
        flex-wrap:wrap;
        line-height:1;
        font-size: clamp(0.95rem, 0.4vw + 0.8rem, 1rem);
      }}
      .qhdr-top .label {{
        font-weight:700;
        white-space:nowrap;
        letter-spacing:0.01em;
      }}
      .qhdr-top .sub {{
        font-weight:600;
        opacity:.7;
        text-transform:uppercase;
        letter-spacing:0.12em;
        font-size:0.75rem;
      }}
      .qhdr-top .track {{
        position:relative;
        flex:1;
        min-width:200px;
        height:9px;
        border-radius:999px;
        background:linear-gradient(90deg,{track_light},{track_light});
        overflow:hidden;
      }}
      @media (prefers-color-scheme:dark) {{
        .qhdr-top .track {{
          background:linear-gradient(90deg,{track_dark},{track_dark});
        }}
      }}
      .qhdr-top .track .fill {{
        display:block;
        height:100%;
        width:var(--progress-target, {pct}%);
        border-radius:inherit;
        background:linear-gradient(90deg, var(--quiz-progress-fill, {fill}) 0%, var(--quiz-progress-fill, {fill}) 100%);
        box-shadow:0 0 6px rgba(59, 130, 246, 0.55);
        transition:width 0.55s cubic-bezier(0.4, 0, 0.2, 1);
      }}
      .qhdr-top .pct {{
        opacity:.75;
        font-weight:600;
      }}
      .qhdr-meta {{
        display:flex;
        flex-wrap:wrap;
        align-items:center;
        gap:12px;
        font-size:0.82rem;
      }}
      .qhdr-meta .streak {{
        font-weight:700;
        display:flex;
        align-items:center;
        gap:6px;
      }}
      .qhdr-badges {{
        display:flex;
        align-items:center;
        gap:8px;
        flex-wrap:wrap;
      }}
      .qhdr-badge {{
        font-size:1.2rem;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        transition:transform 0.35s ease;
      }}
      .qhdr-badge.placeholder {{
        font-size:0.75rem;
        font-weight:600;
        opacity:0.65;
        padding:2px 8px;
        border-radius:999px;
        border:1px dashed rgba(59,130,246,0.45);
      }}
      .qhdr-badge.new {{
        animation:qhdr-pop 1.4s ease-in-out 3;
      }}
      @keyframes qhdr-pop {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.25); }}
        100% {{ transform: scale(1); }}
      }}
    </style>
    """

    html_block = f"""
    <div class=\"qhdr\" aria-label=\"Question progress: {q_now} of {total_q} ({pct} percent)\">
      <div class=\"qhdr-top\">
        <div class=\"label\">Q {q_now} / {total_q}</div>
        <div class=\"sub\">Lesson Mastery</div>
        <div class=\"track\"><div class=\"fill\" style=\"--progress-target:{pct}%\"></div></div>
        <div class=\"pct\">{pct}%</div>
      </div>
      <div class=\"qhdr-meta\">
        <span class=\"streak\">🔥 {int(login_streak)}-day streak</span>
        <div class=\"qhdr-badges\">{''.join(badge_html)}</div>
      </div>
    </div>
    """
    st.markdown(css + html_block, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Navigation helper: go back to the previous served word
# ─────────────────────────────────────────────────────────────────────
def _go_back_to_prev_word(lid: int, words_df: pd.DataFrame):
    """
    Loads the most recent word from asked_history (if any) as the active question,
    resets the form state, and decrements the visible question counter.
    """
    hist = st.session_state.get("asked_history", [])
    if not hist:
        st.info("You're at the first question.")
        return

    prev = hist.pop()  # take the last served word
    st.session_state.active_word = prev
    st.session_state.q_started_at = time.time()

    row_prev = words_df[words_df["headword"] == prev]
    if row_prev.empty:
        # If the word vanished (lesson edited), just pick the next available one
        st.warning("Previous word is no longer in this lesson. Showing the next available word.")
        st.session_state.active_word = choose_next_word(USER_ID, cid, lid, words_df)
        row_prev = words_df[words_df["headword"] == st.session_state.active_word]

    row_prev = row_prev.iloc[0]
    st.session_state.qdata = build_question_payload(
        st.session_state.active_word,
        row_prev["synonyms"],
        lesson_df=words_df,
    )
    st.session_state.grid_for_word = st.session_state.active_word
    st.session_state.grid_keys = [
        f"opt_{st.session_state.active_word}_{i}"
        for i in range(len(st.session_state.qdata["choices"]))
    ]
    st.session_state.selection = set()
    st.session_state.answered = False
    st.session_state.eval = None

    # Decrement visible question index for this lesson (never below 1)
    st.session_state.q_index_per_lesson[int(lid)] = max(
        1, st.session_state.q_index_per_lesson.get(int(lid), 1) - 1
    )
    st.rerun()


# -----------------------------
# STUDENT FLOW (main content)
# -----------------------------
if st.session_state["auth"]["role"] == "student":
    if selected_course_id is None:
        st.info("Select a course from the sidebar to begin.")
        st.stop()

    cid = int(selected_course_id)

    if lessons.empty:
        st.info("This course has no lessons yet.")
        st.stop()

    l_map = dict(zip(lessons["lesson_id"], lessons["title"]))

    if selected_lesson_id is None:
        selected_lesson_id = lessons["lesson_id"].iloc[0]

    lid = int(selected_lesson_id)

    # NEW: lesson-level progress and question count
    total_q, mastered_q, attempted_q = lesson_progress(USER_ID, int(lid))
    basis = mastered_q if mastered_q > 0 else attempted_q
    pct = int(round(100 * (basis if total_q else 0) / (total_q or 1)))

    if st.session_state.q_index_per_lesson.get(int(lid)) is None:
        baseline = max(1, min(int(total_q or 1), int(attempted_q or 0) + 1))
        st.session_state.q_index_per_lesson[int(lid)] = baseline

    # Ensure a counter exists
    q_now = st.session_state.q_index_per_lesson.get(int(lid), 1)


    words_df = lesson_words(int(cid), int(lid))
    if words_df.empty:
        st.info("This lesson has no words yet.")
        st.stop()

    # ensure history state (must NOT be inside the 'words_df.empty' block)
    if "asked_history" not in st.session_state:
        st.session_state.asked_history = []

    # Active question state
    new_word_needed = ("active_word" not in st.session_state) or (st.session_state.get("active_lid") != lid)
    if new_word_needed:
        st.session_state.active_lid = lid
        st.session_state.active_word = choose_next_word(USER_ID, cid, lid, words_df)
        st.session_state.q_started_at = time.time()
        row_init = words_df[words_df["headword"] == st.session_state.active_word].iloc[0]
        st.session_state.qdata = build_question_payload(
            st.session_state.active_word,
            row_init["synonyms"],
            lesson_df=words_df,
        )
        st.session_state.grid_for_word = st.session_state.active_word
        st.session_state.grid_keys = [
            f"opt_{st.session_state.active_word}_{i}" for i in range(len(st.session_state.qdata['choices']))
        ]
        for _k in st.session_state.grid_keys:
            if _k in st.session_state:
                del st.session_state[_k]
        st.session_state.selection = set()
        st.session_state.answered = False
        st.session_state.eval = None

    if "answered" not in st.session_state:
        st.session_state.answered = False
    if "eval" not in st.session_state:
        st.session_state.eval = None

    active = st.session_state.active_word

# Harden lookup in case lesson changed mid-session
    filtered = words_df[words_df["headword"] == active]
    if filtered.empty:
        st.session_state.active_word = choose_next_word(USER_ID, cid, lid, words_df)
        st.session_state.q_started_at = time.time()
        row_init = words_df[words_df["headword"] == st.session_state.active_word].iloc[0]
        st.session_state.qdata = build_question_payload(
            st.session_state.active_word,
            row_init["synonyms"],
            lesson_df=words_df,
        )
        st.session_state.grid_for_word = st.session_state.active_word
        st.session_state.grid_keys = [
            f"opt_{st.session_state.active_word}_{i}"
            for i in range(len(st.session_state.qdata["choices"]))
        ]
        for _k in st.session_state.grid_keys:
            if _k in st.session_state:
                del st.session_state[_k]
        st.session_state.selection = set()
        st.session_state.answered = False
        st.session_state.eval = None
        st.rerun()
    else:
        row = filtered.iloc[0]

    qdata = st.session_state.qdata
    choices = qdata["choices"]
    correct_set = qdata["correct"]

# State hardening so we never hide both form and feedback
    if st.session_state.answered and st.session_state.eval is None:
        st.session_state.answered = False

# Render compact header with inline progress bar (before tabs)
    header_badges = [
        {
            "emoji": b.get("emoji", ""),
            "name": b.get("badge_name", ""),
            "is_new": b.get("badge_name") in recent_badge_names,
        }
        for b in st.session_state.gamification.get("badges", [])[:4]
    ]

    render_q_header(
        q_now,
        total_q,
        pct,
        login_streak=st.session_state.gamification.get("login_streak", 0),
        badge_strip=header_badges,
    )
    st.session_state.badges_recent = []

# Tabs for Practice vs Review (header stays ABOVE)
    tab_practice, tab_review = st.tabs(["Practice", "Review Mistakes"])

# ─────────────────────────────────────────────────────────────────────
# PRACTICE TAB — quiz form + after-submit feedback + Next
# ─────────────────────────────────────────────────────────────────────
    with tab_practice:
        # Always start with current selection state
        temp_selection = set(st.session_state.get("selection", set()))

        submitted = False

        if not st.session_state.answered:
            difficulty_level = int(row.get("difficulty", 2) or 2)
            diff = DIFFICULTY_THEME.get(difficulty_level, DIFFICULTY_THEME[2])
            safe_word = html.escape(active)

            st.markdown(
                f"<div class='quiz-heading'><h3>Word: <strong>{safe_word}</strong></h3>"
                f"<span class='difficulty-badge'>{diff['emoji']} {diff['label']}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='quiz-instructions'>Pick every option that matches the meaning of the word.</p>",
                unsafe_allow_html=True,
            )

            keys = st.session_state.grid_keys
            # Ensure checkbox widgets reflect any persisted selection
            for idx, opt in enumerate(choices):
                state_key = keys[idx]
                if state_key not in st.session_state:
                    st.session_state[state_key] = opt in temp_selection

            form_id = f"quiz_form_{st.session_state.active_word}"
            with st.form(form_id):
                st.markdown("<div class='quiz-options-grid'>", unsafe_allow_html=True)
                # Render options in a responsive 3-column grid
                for start in range(0, len(choices), 3):
                    row_choices = choices[start : start + 3]
                    row_keys = keys[start : start + 3]
                    cols = st.columns(len(row_choices))
                    for col, opt, state_key in zip(cols, row_choices, row_keys):
                        with col:
                            st.checkbox(opt, key=state_key)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='quiz-actions'>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Submit")
                st.markdown("</div>", unsafe_allow_html=True)

    #        st.markdown("</div>", unsafe_allow_html=True)

        # Allow going back even before submitting
        #if st.button("◀ Back", key="btn_back_form"):
        #    _go_back_to_prev_word(lid, words_df)

        # Always persist selection each render
        st.session_state.selection = {
            opt
            for opt, state_key in zip(choices, st.session_state.grid_keys)
            if st.session_state.get(state_key)
        }
        temp_selection = set(st.session_state.selection)

        # Handle Submit
        if submitted:
            elapsed_ms = (time.time() - st.session_state.q_started_at) * 1000
            picked_set = set(list(st.session_state.selection))
            is_correct = (picked_set == correct_set)

            correct_choice_for_log = list(correct_set)[0]
            result = update_after_attempt(
                USER_ID,
                cid,
                lid,
                active,
                is_correct,
                int(elapsed_ms),
                int(row["difficulty"]),
                ", ".join(sorted(picked_set)),
                correct_choice_for_log,
            )

            st.session_state.last_xp_gain = int(result.get("xp_awarded", 0) or 0)
            st.session_state.badges_recent = [
                b.get("badge_name") for b in result.get("new_badges", [])
            ]
            st.session_state.badge_details_recent = result.get("new_badges", [])
            if result.get("new_badges"):
                celebrate_badges(result["new_badges"])
            st.session_state.gamification = gamification_snapshot(USER_ID)

            st.session_state.answered = True
            st.session_state.eval = {
                "is_correct": bool(is_correct),
                "picked_set": set(picked_set),
                "correct_set": set(correct_set),
                "choices": list(choices)
            }

            # If wrong, push this headword to the front of the review queue
            if not is_correct:
                from collections import deque
                if "review_queue" not in st.session_state or st.session_state.review_queue is None:
                    st.session_state.review_queue = deque()
                if st.session_state.active_word not in st.session_state.review_queue:
                    st.session_state.review_queue.appendleft(st.session_state.active_word)

            st.rerun()

        # No direct "Next" action here; students continue from the feedback view.

# ========== PATCH START: Dynamic feedback by lesson type (Option A) ==========
# Detect lesson kind from course/lesson titles (synonym | antonym)
def detect_lesson_kind(course_title: str, lesson_title: str) -> str:
    t = f"{str(course_title or '')} {str(lesson_title or '')}".lower()
    antonym_keys = ["antonym", "antonyms", "opposite", "opposites", "contrary", "reverse"]
    return "antonym" if any(k in t for k in antonym_keys) else "synonym"

# Deterministic, kid-friendly text (no API needed)
def feedback_text(headword: str, correct_word: str, lesson_kind: str):
    h, c = (headword or "").strip(), (correct_word or "").strip()
    if lesson_kind == "antonym":
        why = f"'{c}' is an opposite of '{h}'. They mean very different things."
        examples = [
            f"I felt {h} in the sunshine, but {c} when plans were canceled.",
            f"A warm day feels {h}; a stormy day can feel {c}."
        ]
    else:
        # default = synonym
        why = f"'{c}' means almost the same as '{h}', so it fits here."
        examples = [
            f"I felt {c} when I finished my project.",
            f"Our class was {c} after we won the match."
        ]
    return why, examples[:2]

# Override: route old call sites to the new dynamic generator
def gpt_feedback_examples(headword: str, correct_word: str):
    """
    Backward-compatible wrapper.
    Uses title-based detection to choose synonym/antonym wording.
    Ignores external APIs (OpenAI/Gemini) for speed and zero cost.
    """
    try:
        # These globals are set in your Student flow
        course_title = selected_label          # sidebar radio (course label)
        lesson_title = l_map[lid]              # selected lesson title
    except Exception:
        course_title, lesson_title = "", ""

    kind = detect_lesson_kind(course_title, lesson_title)
    return feedback_text(headword, correct_word, kind)
# ========== PATCH END: Dynamic feedback by lesson type (Option A) ==========


# AFTER-SUBMIT feedback + Back & Next buttons
if st.session_state.get("answered") and st.session_state.get("eval"):
    ev = st.session_state.eval
    difficulty_level = int(row.get("difficulty", 2) or 2)
    diff = DIFFICULTY_THEME.get(difficulty_level, DIFFICULTY_THEME[2])
    safe_word_feedback = html.escape(st.session_state.active_word)

#    st.markdown(f"<div class='quiz-surface {diff['class']}'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='quiz-heading'><h3>Word: <strong>{safe_word_feedback}</strong></h3>"
        f"<span class='difficulty-badge'>{diff['emoji']} {diff['label']}</span></div>",
        unsafe_allow_html=True,
    )

    banner_class = "correct" if ev["is_correct"] else "try-again"
    banner_text = "🎉 Correct!" if ev["is_correct"] else "🤔 Try again!"
    st.markdown(
        f"<div class='feedback-banner {banner_class}'>{banner_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='quiz-instructions'>Review the breakdown below, then choose your next step.</p>",
        unsafe_allow_html=True,
    )

    xp_gain = int(st.session_state.get("last_xp_gain", 0) or 0)
    if xp_gain:
        st.success(f"⭐ You earned {xp_gain} XP!")

    new_badges = st.session_state.get("badge_details_recent", [])
    if new_badges:
        badge_list = ", ".join(
            f"{b.get('emoji', '')} {b.get('badge_name', '')}".strip()
            for b in new_badges
        )
        st.info(f"New badge unlocked: {badge_list}")
        st.session_state.badge_details_recent = []

    # Show explanation of options
    with st.expander("Why are these the best choices?", expanded=True):
        lines = []
        for opt in ev["choices"]:
            if opt in ev["correct_set"] and opt in ev["picked_set"]:
                tag = "✅ correct (you picked)"
            elif opt in ev["correct_set"]:
                tag = "✅ correct"
            elif opt in ev["picked_set"]:
                tag = "❌ your pick"
            else:
                tag = ""
            lines.append(f"- **{opt}** {tag}")
        st.markdown("\n".join(lines))

    # NEW: dynamic tip text based on lesson kind
        try:
            lesson_kind = detect_lesson_kind(selected_label, l_map[lid])
        except Exception:
            lesson_kind = "synonym"
        tip = (
            "Tip: pick all the options that **mean almost the same** as the main word."
            if lesson_kind == "synonym"
            else "Tip: pick the options that are **opposites** of the main word."
        )
        st.caption(tip)

    # GPT feedback (optional)
    try:
        correct_choice_for_text = sorted(list(ev["correct_set"]))[0]
        why, examples = gpt_feedback_examples(st.session_state.active_word, correct_choice_for_text)
        st.info(f"**Why:** {why}")
        st.markdown(f"**Examples:**\n\n- {examples[0]}\n- {examples[1]}")
    except Exception:
        pass

    # ───────────────────────────────────────────────────────────────
    # Buttons: Back and Next
    # ───────────────────────────────────────────────────────────────
    st.markdown("<div class='quiz-actions'>", unsafe_allow_html=True)
    if st.button("◀ Back", key="btn_back_feedback"):
        _go_back_to_prev_word(lid, words_df)
    if st.button("Next ▶", key="btn_next_feedback", type="primary"):
        st.session_state.asked_history.append(st.session_state.active_word)

        # Serve from review queue first
        if st.session_state.review_queue:
            next_word = st.session_state.review_queue.popleft()
        else:
            next_word = choose_next_word(USER_ID, cid, lid, words_df)

        # Load next word
        st.session_state.active_word = next_word
        st.session_state.q_started_at = time.time()
        next_row = words_df[words_df["headword"] == next_word].iloc[0]
        st.session_state.qdata = build_question_payload(
            next_word,
            next_row["synonyms"],
            lesson_df=words_df,
        )
        st.session_state.grid_for_word = next_word
        st.session_state.grid_keys = [
            f"opt_{next_word}_{i}"
            for i in range(len(st.session_state.qdata["choices"]))
        ]
        for _k in st.session_state.grid_keys:
            if _k in st.session_state:
                del st.session_state[_k]
        st.session_state.selection = set()
        st.session_state.answered = False
        st.session_state.eval = None

        # Bump lesson question index
        st.session_state.q_index_per_lesson[int(lid)] = \
            st.session_state.q_index_per_lesson.get(int(lid), 1) + 1

        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

#    st.markdown("</div>", unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────
    # REVIEW TAB — retry past mistakes (manual)
    # ─────────────────────────────────────────────────────────────────────
    with tab_review:
        st.write("Click a word you missed to retry it now:")

        missed = get_missed_words(USER_ID, int(lid))

        if not missed:
            n_queue = len(st.session_state.review_queue) if "review_queue" in st.session_state else 0
            if n_queue > 0:
                st.info(f"No recent wrong answers, but {n_queue} item(s) are queued for quick retry.")
            else:
                st.success("Nice! No mistakes to review for this lesson.")
        else:
            cols = st.columns(3)
            for i, hw in enumerate(missed):
                with cols[i % 3]:
                    if st.button(f"Retry: {hw}", key=f"retry_{int(lid)}_{hw}"):
                        # load this headword immediately into the quiz
                        st.session_state.active_lid = lid
                        st.session_state.active_word = hw
                        st.session_state.q_started_at = time.time()

                        row_retry = words_df[words_df["headword"] == hw].iloc[0]
                        st.session_state.qdata = build_question_payload(
                            hw,
                            row_retry["synonyms"],
                            lesson_df=words_df,
                        )
                        st.session_state.grid_for_word = hw
                        st.session_state.grid_keys = [
                            f"opt_{hw}_{j}" for j in range(len(st.session_state.qdata["choices"]))
                        ]
                        for _k in st.session_state.grid_keys:
                            if _k in st.session_state:
                                del st.session_state[_k]
                        st.session_state.selection = set()
                        st.session_state.answered = False
                        st.session_state.eval = None

                        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# Version footer (nice to show deployed tag)
# ─────────────────────────────────────────────────────────────────────
APP_VERSION = "v3-admin-sprint1"
st.markdown(f"<div style='text-align:center;opacity:0.6;'>Version: {APP_VERSION}</div>", unsafe_allow_html=True)









