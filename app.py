import os, time, random, sqlite3
from contextlib import closing
from datetime import datetime, timedelta
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
          last_seen        TIMESTAMPTZ,
          mastered         BOOLEAN DEFAULT FALSE,
          difficulty       INTEGER DEFAULT 2,
          due_date         TIMESTAMPTZ,
          PRIMARY KEY (user_id, headword)
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

# Bootstrap order
init_db()
patch_users_table()
patch_courses_table()

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

def update_after_attempt(user_id, course_id, lesson_id, headword, is_correct, response_ms, difficulty, chosen, correct_choice):
    with engine.begin() as conn:
        prior = conn.execute(
            text("SELECT correct_streak FROM word_stats WHERE user_id=:u AND headword=:h"),
            {"u": user_id, "h": headword}
        ).scalar()
        prior = int(prior or 0)
        new_streak = prior + 1 if is_correct else 0
        mastered = new_streak >= 3
        add_days = 3 if (is_correct and mastered) else (1 if is_correct else 0)
        due = datetime.utcnow() + timedelta(days=add_days)

        conn.execute(text("""
            INSERT INTO word_stats (user_id, headword, correct_streak, total_attempts, correct_attempts, last_seen, mastered, difficulty, due_date)
            VALUES (:u, :h, :cs, 1, :ca, CURRENT_TIMESTAMP, :m, :d, :due)
            ON CONFLICT (user_id, headword) DO UPDATE SET
                correct_streak   = EXCLUDED.correct_streak,
                total_attempts   = word_stats.total_attempts + 1,
                correct_attempts = word_stats.correct_attempts + (:ca),
                last_seen        = CURRENT_TIMESTAMP,
                mastered         = CASE WHEN :m THEN TRUE ELSE word_stats.mastered END,
                difficulty       = :d,
                due_date         = :due
        """), {
            "u": user_id, "h": headword, "cs": new_streak,
            "ca": 1 if is_correct else 0,
            "m": mastered, "d": int(difficulty), "due": due
        })

        conn.execute(text("""
            INSERT INTO attempts(user_id,course_id,lesson_id,headword,is_correct,response_ms,chosen,correct_choice)
            VALUES (:u,:c,:l,:h,:ok,:ms,:ch,:cc)
        """), {
            "u": user_id, "c": course_id, "l": lesson_id, "h": headword,
            "ok": bool(is_correct), "ms": int(response_ms),
            "ch": chosen, "cc": correct_choice
        })

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

def build_question_payload(headword: str, synonyms_str: str):
    """
    Build a 6-option question:
      - 2 correct (first two synonyms or 1 + '(close)')
      - 4 distractors from a small safe pool (stable per word)
    """
    syn_list = [s.strip() for s in str(synonyms_str).split(",") if s.strip()]
    correct = syn_list[:2] if len(syn_list) >= 2 else syn_list[:1]
    if len(correct) == 1:
        correct = [correct[0], f"{correct[0]} (close)"]

    distractor_pool = [
        "banana","pencil","soccer","window","pizza","rainbow","kitten","tractor","marble","backpack",
        "ladder","ocean","camera","blanket","sandwich","rocket","helmet","garden","notebook","button"
    ]
    pool = [d for d in distractor_pool if d.lower() not in {c.lower() for c in correct}]
    seed = int(hashlib.md5(headword.encode("utf-8")).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed) # stable per word across restarts
    distractors = []
    while len(distractors) < 4 and pool:
        cand = rnd.choice(pool)
        pool.remove(cand)
        if cand not in distractors:
            distractors.append(cand)

    choices = correct + distractors
    rnd.shuffle(choices)
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

    # Admin Section — manage student accounts
    with tab_admin:
        st.subheader("Manage Students")
        df = all_students_df()
        st.dataframe(df, use_container_width=True)

        st.markdown("**Create Student**")
        with st.form("create_student"):
            c1,c2,c3=st.columns(3)
            with c1: s_name  = st.text_input("Name", key="adm_create_name")
            with c2: s_email = st.text_input("Email", key="adm_create_email")
            with c3: s_pwd   = st.text_input("Temp Password", value="Learn123!", type="password", key="adm_create_pwd")
            go = st.form_submit_button("Create")
            if go and s_name and s_email and s_pwd:
                try:
                    create_user(s_name, s_email.strip().lower(), s_pwd, "student")
                    st.success("Student created.")
                except Exception as ex:
                    st.error(f"Could not create user: {ex}")

        if not df.empty:
            st.markdown("**Enable / Disable**")
            sid = st.selectbox(
                "Student",
                df["user_id"].tolist(),
                format_func=lambda x: df.loc[df["user_id"]==x,"name"].values[0],
                key="admin_toggle_student"
            )
            active = st.radio("Status", ["Enable","Disable"], horizontal=True, key="admin_status_radio")
            if st.button("Apply status", key="admin_apply_status"):
                set_user_active(sid, active=="Enable"); st.success("Updated.")

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

# Student experience
if st.session_state["auth"]["role"] == "student":
    _hide_default_h1_and_set("welcome to English Learning made easy - Student login")

    courses = pd.read_sql(
        text("""
            SELECT C.course_id, C.title
            FROM enrollments E JOIN courses C ON C.course_id=E.course_id
            WHERE E.user_id=:u
        """),
        con=engine, params={"u": USER_ID}
    )

    # Sidebar is truly inside the student block ↓
    with st.sidebar:
        st.subheader("My courses")
        if courses.empty:
            st.info("No courses assigned yet.")
            st.stop()
        else:
            labels = []
            id_by_label = {}
            for _, rowc in courses.iterrows():
                c_completed, c_total, c_pct = course_progress(USER_ID, int(rowc["course_id"]))
                label = f"{rowc['title']}"
                labels.append(label)
                id_by_label[label] = int(rowc["course_id"])

            prev = st.session_state.get("active_cid")
            if prev in id_by_label.values() and "student_course_select" not in st.session_state:
                default_label = [k for k, v in id_by_label.items() if v == prev][0]
                default_index = labels.index(default_label)
            else:
                default_index = 0

            selected_label = st.radio("Courses", labels, index=default_index, key="student_course_select")
            cid = id_by_label[selected_label]
            st.session_state["active_cid"] = cid

            c_completed, c_total, c_pct = course_progress(USER_ID, int(cid))
            st.caption(f"Selected: {selected_label} — {c_pct}% complete")



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


# -----------------------------
# STUDENT FLOW (main content)
# -----------------------------
if st.session_state["auth"]["role"] == "student":
    lessons = pd.read_sql(
        text("SELECT lesson_id,title FROM lessons WHERE course_id=:c ORDER BY sort_order"),
        con=engine, params={"c": int(cid)}
    )
    if lessons.empty:
        st.info("This course has no lessons yet.")
        st.stop()

    l_map = dict(zip(lessons["lesson_id"], lessons["title"]))
    lid = st.selectbox(
        "Lesson",
        list(l_map.keys()),
        format_func=lambda x: l_map[x],
        key="student_lesson_select"
    )

    # Initialize per-lesson question counter (once per lesson)
    if st.session_state.q_index_per_lesson.get(int(lid)) is None:
        st.session_state.q_index_per_lesson[int(lid)] = 1

    # NEW: lesson-level progress and question count
    total_q, mastered_q, attempted_q = lesson_progress(USER_ID, int(lid))
    basis = mastered_q if mastered_q > 0 else attempted_q
    pct = int(round(100 * (basis if total_q else 0) / (total_q or 1)))

    # Ensure a counter exists
    q_now = st.session_state.q_index_per_lesson.get(int(lid), 1)

    # Compact header (no duplicates, no progress bar)
    st.markdown(f"**Q {q_now} / {total_q}**  ·  Progress: **{pct}%** :")

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
        st.session_state.qdata = build_question_payload(st.session_state.active_word, row_init["synonyms"])
        st.session_state.grid_for_word = st.session_state.active_word
        st.session_state.grid_keys = [
            f"opt_{st.session_state.active_word}_{i}" for i in range(len(st.session_state.qdata['choices']))
        ]
        st.session_state.selection = set()
        st.session_state.answered = False
        st.session_state.eval = None

    if "answered" not in st.session_state:
        st.session_state.answered = False
    if "eval" not in st.session_state:
        st.session_state.eval = None

    active = st.session_state.active_word
    row = words_df[words_df["headword"] == active].iloc[0]
    qdata = st.session_state.qdata
    choices = qdata["choices"]
    correct_set = qdata["correct"]

    # NEW: tabs for Practice vs Review
    tab_practice, tab_review = st.tabs(["Practice", "Review Mistakes"])

    # ─────────────────────────────────────────────────────────────────────
    # PRACTICE TAB — quiz form + after-submit feedback + Next
    # ─────────────────────────────────────────────────────────────────────
    with tab_practice:
        # The quiz form (no auto-advance)
        if not st.session_state.answered:
            with st.form("quiz_form", clear_on_submit=False):
                st.subheader(f"Word: **{active}**")
                st.write("Pick the **synonyms** (select all that apply), then press **Submit**.")

                keys = st.session_state.grid_keys
                row1 = st.columns(3)
                row2 = st.columns(3)
                grid_rows = [row1, row2]

                temp_selection = set(st.session_state.selection)
                for i, opt in enumerate(choices):
                    col = grid_rows[0][i] if i < 3 else grid_rows[1][i - 3]
                    with col:
                        checked = opt in temp_selection
                        new_val = st.checkbox(opt, value=checked, key=keys[i])
                    if new_val:
                        temp_selection.add(opt)
                    else:
                        temp_selection.discard(opt)

                c1, c2 = st.columns([1, 1])
                with c1:
                    submitted = st.form_submit_button("Submit", type="primary")
                with c2:
                    nextq = st.form_submit_button("Next ▶")

            st.session_state.selection = temp_selection

            if submitted:
                elapsed_ms = (time.time() - st.session_state.q_started_at) * 1000
                picked_set = set(list(st.session_state.selection))
                is_correct = (picked_set == correct_set)

                correct_choice_for_log = list(correct_set)[0]
                update_after_attempt(
                    USER_ID, cid, lid, active,
                    is_correct, elapsed_ms, int(row["difficulty"]),
                    ", ".join(sorted(picked_set)), correct_choice_for_log
                )

                st.session_state.answered = True
                st.session_state.eval = {
                    "is_correct": bool(is_correct),
                    "picked_set": set(picked_set),
                    "correct_set": set(correct_set),
                    "choices": list(choices)
                }

                # If wrong, push this headword to the front of the review queue
                if not is_correct:
                    try:
                        from collections import deque
                        if "review_queue" not in st.session_state or st.session_state.review_queue is None:
                            st.session_state.review_queue = deque()
                        if st.session_state.active_word not in st.session_state.review_queue:
                            st.session_state.review_queue.appendleft(st.session_state.active_word)
                    except Exception:
                        pass

                st.rerun()

            elif nextq:
                st.warning("Please **Submit** your answer first, then click **Next**.")

        # AFTER-SUBMIT feedback + Next button
        if st.session_state.get("answered") and st.session_state.get("eval"):
            ev = st.session_state.eval
            st.subheader(f"Word: **{st.session_state.active_word}**")
            if ev["is_correct"]:
                st.success("✅ Correct!")
            else:
                st.error("❌ Not quite. Check the correct answers below.")

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
                st.caption("Tip: pick all the options that mean almost the same as the main word.")

            # GPT feedback (kept)
            try:
                correct_choice_for_text = sorted(list(ev["correct_set"]))[0]
                why, examples = gpt_feedback_examples(st.session_state.active_word, correct_choice_for_text)
                st.info(f"**Why:** {why}")
                st.markdown(f"**Examples:**\n\n- {examples[0]}\n- {examples[1]}")
            except Exception:
                pass

            if st.button("Next ▶", use_container_width=True):
                st.session_state.asked_history.append(st.session_state.active_word)

                # serve from review queue first
                if st.session_state.review_queue:
                    next_word = st.session_state.review_queue.popleft()
                else:
                    next_word = choose_next_word(USER_ID, cid, lid, words_df)

                # load next word
                st.session_state.active_word = next_word
                st.session_state.q_started_at = time.time()
                next_row = words_df[words_df["headword"] == next_word].iloc[0]
                st.session_state.qdata = build_question_payload(next_word, next_row["synonyms"])
                st.session_state.grid_for_word = next_word
                st.session_state.grid_keys = [
                    f"opt_{next_word}_{i}" for i in range(len(st.session_state.qdata["choices"]))
                ]
                st.session_state.selection = set()
                st.session_state.answered = False
                st.session_state.eval = None
                # increase question counter
                st.session_state.q_index_per_lesson[int(lid)] = \
                    st.session_state.q_index_per_lesson.get(int(lid), 1) + 1
                st.rerun()

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
                        st.session_state.qdata = build_question_payload(hw, row_retry["synonyms"])
                        st.session_state.grid_for_word = hw
                        st.session_state.grid_keys = [
                            f"opt_{hw}_{j}" for j in range(len(st.session_state.qdata["choices"]))
                        ]
                        st.session_state.selection = set()
                        st.session_state.answered = False
                        st.session_state.eval = None

                        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# Version footer (nice to show deployed tag)
# ─────────────────────────────────────────────────────────────────────
APP_VERSION = os.getenv("APP_VERSION", "dev")
st.markdown(f"<div style='text-align:center;opacity:0.6;'>Version: {APP_VERSION}</div>", unsafe_allow_html=True)


