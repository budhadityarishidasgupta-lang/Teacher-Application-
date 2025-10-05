import os, time, random, sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from passlib.hash import bcrypt
from sqlalchemy import create_engine, text

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
    # Add columns if missing
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE"))

    # Backfill role/is_active
    admin_email_lc = ADMIN_EMAIL.lower()
    with engine.begin() as conn:
        conn.execute(text("UPDATE users SET role='admin'   WHERE role IS NULL AND lower(email)=:e"),
                     {"e": admin_email_lc})
        conn.execute(text("UPDATE users SET role='student' WHERE role IS NULL AND lower(email)<>:e"),
                     {"e": admin_email_lc})
        conn.execute(text("UPDATE users SET is_active=TRUE WHERE is_active IS NULL"))

    # Backfill password hashes where missing
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
    """Ensure legacy courses/lessons/words have required columns."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE courses ADD COLUMN IF NOT EXISTS description TEXT"))
        conn.execute(text("ALTER TABLE lessons ADD COLUMN IF NOT EXISTS sort_order INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE words   ADD COLUMN IF NOT EXISTS difficulty INTEGER DEFAULT 2"))

# Bootstrap order
init_db()
patch_users_table()
patch_courses_table()

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
            text("""SELECT user_id,name,email,password_hash,role,is_active
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

def course_progress(user_id: int, course_id: int):
    """Return (completed, total, percent) for a course for this user."""
    all_words = pd.read_sql(
        text("""
            SELECT w.headword
            FROM lessons L
            JOIN lesson_words lw ON lw.lesson_id = L.lesson_id
            JOIN words w ON w.word_id = lw.word_id
            WHERE L.course_id=:c
        """),
        con=engine, params={"c": int(course_id)}
    )["headword"].tolist()
    total = len(set(all_words))
    if total == 0:
        return (0, 0, 0)
    completed = pd.read_sql(
        text("""
            SELECT COUNT(*) AS c
            FROM word_stats
            WHERE user_id=:u AND mastered=TRUE AND headword = ANY(:arr)
        """),
        con=engine, params={"u": int(user_id), "arr": list(set(all_words))}
    )["c"].iloc[0]
    percent = int(round(100 * completed / total)) if total else 0
    return (int(completed), total, percent)

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
    rnd = random.Random(hash(headword) % (2**32))  # stable per word
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

        # --- light validation/sanitization ---
        def _clean(s: str) -> str:
            s = s.replace("—", "-").replace(";", ",").replace('"', "").replace("'", "")
            s = " ".join(s.split())  # collapse whitespace
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            if s and s[-1] not in ".!?":
                s += "."
            return s

        ok_examples = []
        for s in examples[:2]:
            w = s.lower().split()
            # must contain the correct word, avoid super short/long, and avoid using both headword+correct_word together
            if (correct_word.lower() in w) and (7 <= len(w) <= 13) and (headword.lower() not in w):
                ok_examples.append(_clean(s))
        # backfill if GPT returned weak examples
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
# Auth UI (replaces your existing login_form)
# ─────────────────────────────────────────────────────────────────────
def login_form():
    # Lazy-init auth so this works even if AuthService is only defined later.
    global auth
    try:
        auth  # noqa: F401
    except NameError:
        try:
            from auth_service import AuthService
            auth = AuthService(engine)
        except Exception:
            auth = None

    st.sidebar.subheader("Sign in")

    # Handle password reset via URL params: ?reset_email=...&reset_token=...
    try:
        qp = st.query_params  # Streamlit ≥1.30
    except Exception:
        qp = st.experimental_get_query_params()

    def _first(qv):
        if qv is None: return None
        if isinstance(qv, list): return qv[0]
        return qv

    reset_email = (_first(qp.get("reset_email")) or "").strip().lower()
    reset_token = (_first(qp.get("reset_token")) or "").strip()

    # If a reset link was opened, show reset form (pre-login) in the sidebar
    if auth and reset_email and reset_token:
        st.sidebar.markdown("**Reset your password**")
        npw1 = st.sidebar.text_input("New password", type="password", key="rpw1")
        npw2 = st.sidebar.text_input("Confirm new password", type="password", key="rpw2")
        if st.sidebar.button("Set new password", key="btn_set_new_pw"):
            if not npw1 or npw1 != npw2:
                st.sidebar.error("Passwords must match and not be empty.")
            else:
                ok, msg = auth.reset_password_with_token(reset_email, reset_token, npw1)
                if ok:
                    st.sidebar.success("Password reset. Please log in.")
                else:
                    st.sidebar.error(msg)
        st.sidebar.markdown("---")

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

    # Pre-login "Forgot password?" (emails reset link; no codes shown)
    if auth:
        with st.sidebar.expander("Forgot password?"):
            fp_email = st.text_input("Your email", key="fp_email", value=email)
            if st.button("Email me a reset link", key="btn_forgot_send"):
                base_url = os.getenv("APP_BASE_URL", "")
                ok, msg = auth.email_password_reset(fp_email, base_url)
                st.info(msg) if ok else st.error(msg)

    if st.sidebar.button("Log out", key="btn_logout"):
        st.session_state.pop("auth", None)

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

ROLE   = st.session_state.auth["role"]
USER_ID= st.session_state.auth["user_id"]
NAME   = st.session_state.auth["name"]
st.sidebar.caption(f"Signed in as **{NAME}** ({ROLE})")

# Safe defaults for session (prevents admin crashes)
_defaults = {
    "answered": False, "eval": None, "active_word": None, "active_lid": None,
    "q_started_at": 0.0, "selection": set(), "asked_history": [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────
# Admin experience
# ─────────────────────────────────────────────────────────────────────
if ROLE == "admin":
    st.title("🛠️ Admin Console")
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

    # Teacher Dashboard — courses/lessons/words/enrollments
    with tab_teacher:
        st.subheader("Courses")
        with st.form("create_course"):
            title = st.text_input("Course title", key="td_course_title")
            desc  = st.text_area("Description", "", key="td_course_desc")
            ok = st.form_submit_button("Create course")
            if ok and title.strip():
                with engine.begin() as conn:
                    conn.execute(text("INSERT INTO courses(title,description) VALUES(:t,:d)"),
                                 {"t": title, "d": desc})
                st.success("Course created.")

        df_courses = pd.read_sql(text("SELECT course_id,title,description FROM courses ORDER BY course_id DESC"), con=engine)
        st.dataframe(df_courses, use_container_width=True)

        st.subheader("Lessons")
        if not df_courses.empty:
            cid_lessons = st.selectbox(
                "Course",
                df_courses["course_id"].tolist(),
                format_func=lambda x: df_courses.loc[df_courses["course_id"]==x,"title"].values[0],
                key="td_course_for_lessons"
            )
            with st.form("create_lesson"):
                lt = st.text_input("Lesson title", key="td_lesson_title")
                order = st.number_input("Sort order", 0, 999, 0, key="td_lesson_order")
                ok = st.form_submit_button("Create lesson")
                if ok and lt.strip():
                    with engine.begin() as conn:
                        conn.execute(
                            text("INSERT INTO lessons(course_id,title,sort_order) VALUES(:c,:t,:o)"),
                            {"c": int(cid_lessons), "t": lt, "o": int(order)}
                        )
                    st.success("Lesson created.")

            st.markdown("**Upload CSV of words (headword,synonyms)**")
            f = st.file_uploader("Upload CSV", type=["csv"], key="td_upload_csv")
            if f:
                df_up = pd.read_csv(f)
                st.dataframe(df_up.head(), use_container_width=True)

                df_less = pd.read_sql(
                    text("SELECT lesson_id,title FROM lessons WHERE course_id=:c ORDER BY sort_order"),
                    con=engine, params={"c": int(cid_lessons)}
                )
                if df_less.empty:
                    st.warning("Create a lesson first.")
                else:
                    lid_upload = st.selectbox(
                        "Target lesson",
                        df_less["lesson_id"].tolist(),
                        format_func=lambda x: df_less.loc[df_less["lesson_id"]==x,"title"].values[0],
                        key="td_lesson_upload"
                    )
                    if st.button("Import words", key="td_import_words_btn"):
                        n=0
                        with engine.begin() as conn:
                            for _,r in df_up.iterrows():
                                hw = str(r["headword"]).strip()
                                syns = str(r["synonyms"]).strip()
                                if not hw or not syns: continue
                                syn_list=[s.strip() for s in syns.split(",") if s.strip()]
                                diff = 1 if (len(hw)<=6 and len(syn_list)<=3) else (2 if len(hw)<=8 and len(syn_list)<=5 else 3)

                                wid = conn.execute(
                                    text("""INSERT INTO words(headword,synonyms,difficulty)
                                            VALUES(:h,:s,:d)
                                            ON CONFLICT DO NOTHING
                                            RETURNING word_id"""),
                                    {"h": hw, "s": ", ".join(syn_list), "d": int(diff)}
                                ).scalar()
                                if wid is None:
                                    wid = conn.execute(
                                        text("SELECT word_id FROM words WHERE headword=:h AND synonyms=:s"),
                                        {"h": hw, "s": ", ".join(syn_list)}
                                    ).scalar()

                                conn.execute(
                                    text("""INSERT INTO lesson_words(lesson_id,word_id,sort_order)
                                            VALUES(:l,:w,:o)
                                            ON CONFLICT (lesson_id,word_id) DO NOTHING"""),
                                    {"l": int(lid_upload), "w": int(wid), "o": int(n)}
                                )
                                n+=1
                        st.success(f"Imported {n} words.")

        st.subheader("Assign courses to students")
        students = pd.read_sql(text("SELECT user_id,name FROM users WHERE role='student' AND is_active=TRUE ORDER BY name"), con=engine)
        df_courses_assign = pd.read_sql(text("SELECT course_id,title FROM courses ORDER BY title"), con=engine)
        if students.empty or df_courses_assign.empty:
            st.info("Create students and courses first.")
        else:
            sid_assign = st.selectbox(
                "Student",
                students["user_id"].tolist(),
                format_func=lambda x: students.loc[students["user_id"]==x,"name"].values[0],
                key="assign_student"
            )
            cid_assign = st.selectbox(
                "Course",
                df_courses_assign["course_id"].tolist(),
                format_func=lambda x: df_courses_assign.loc[df_courses_assign["course_id"]==x,"title"].values[0],
                key="assign_course"
            )
            if st.button("Enroll", key="assign_enroll_btn"):
                with engine.begin() as conn:
                    conn.execute(
                        text("""INSERT INTO enrollments(user_id,course_id)
                                VALUES(:u,:c)
                                ON CONFLICT (user_id,course_id) DO NOTHING"""),
                        {"u": int(sid_assign), "c": int(cid_assign)}
                    )
                st.success("Enrolled.")

    # Student Dashboard — class overview
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

# ─────────────────────────────────────────────────────────────────────
# Student experience
# ─────────────────────────────────────────────────────────────────────
if ROLE == "student":
    st.title("🎓 Student")
    courses = pd.read_sql(
        text("""
            SELECT C.course_id, C.title
            FROM enrollments E JOIN courses C ON C.course_id=E.course_id
            WHERE E.user_id=:u
        """),
        con=engine, params={"u": USER_ID}
    )

    with st.sidebar:
        st.subheader("My courses")
        if courses.empty:
            st.info("No courses assigned yet.")
        else:
            labels = []
            id_by_label = {}
            for _, rowc in courses.iterrows():
                c_completed, c_total, c_pct = course_progress(USER_ID, int(rowc["course_id"]))
                label = f"{rowc['title']} — {c_pct}%"
                labels.append(label)
                id_by_label[label] = int(rowc["course_id"])
            selected_label = st.radio("Courses", labels, index=0, key="student_course_radio")
            cid = id_by_label[selected_label]

    if courses.empty:
        st.stop()

    lessons = pd.read_sql(
        text("SELECT lesson_id,title FROM lessons WHERE course_id=:c ORDER BY sort_order"),
        con=engine, params={"c": cid}
    )
    if lessons.empty:
        st.info("This course has no lessons yet."); st.stop()

    l_map = dict(zip(lessons["lesson_id"], lessons["title"]))
    lid = st.selectbox("Lesson", list(l_map.keys()), format_func=lambda x: l_map[x], key="student_lesson_select")

    words_df = lesson_words(cid, lid)
    if words_df.empty:
        st.info("This lesson has no words yet."); st.stop()

    if "asked_history" not in st.session_state:
        st.session_state.asked_history = []
    #m, total = mastered_count(USER_ID, lid)
    #st.progress(min(m / max(total, 1), 1.0), text=f"Mastered {m}/{total} words")

    # Active question state
    new_word_needed = ("active_word" not in st.session_state) or (st.session_state.get("active_lid") != lid)
    if new_word_needed:
        st.session_state.active_lid = lid
        st.session_state.active_word = choose_next_word(USER_ID, cid, lid, words_df)
        st.session_state.q_started_at = time.time()
        row_init = words_df[words_df["headword"] == st.session_state.active_word].iloc[0]
        st.session_state.qdata = build_question_payload(st.session_state.active_word, row_init["synonyms"])
        st.session_state.grid_for_word = st.session_state.active_word
        st.session_state.grid_keys = [f"opt_{st.session_state.active_word}_{i}" for i in range(len(st.session_state.qdata['choices']))]
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
                col = grid_rows[0][i] if i < 3 else grid_rows[1][i-3]
                with col:
                    checked = opt in temp_selection
                    new_val = st.checkbox(opt, value=checked, key=keys[i])
                if new_val: temp_selection.add(opt)
                else:       temp_selection.discard(opt)

            c1, c2 = st.columns([1, 1])
            with c1:
                submitted = st.form_submit_button("Submit", type="primary")
            with c2:
                nextq = st.form_submit_button("Next ▶")

        st.session_state.selection = temp_selection

        if submitted:
            elapsed_ms = (time.time()-st.session_state.q_started_at)*1000
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
            st.rerun()

        elif nextq:
            st.warning("Please **Submit** your answer first, then click **Next**.")

# After Submit: show feedback + Next (outside the form), and ONLY Next advances
if ROLE == "student" and st.session_state.get("answered") and st.session_state.get("eval"):
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

    # GPT: brief why + 2 examples
    try:
        correct_choice_for_text = sorted(list(ev["correct_set"]))[0]
        why, examples = gpt_feedback_examples(st.session_state.active_word, correct_choice_for_text)
        st.info(f"**Why:** {why}")
        st.markdown(f"**Examples:**\n\n- {examples[0]}\n- {examples[1]}")
    except Exception:
        pass

    if st.button("Next ▶", use_container_width=True):
        st.session_state.asked_history.append(st.session_state.active_word)
        next_word = choose_next_word(USER_ID, cid, lid, words_df)
        st.session_state.active_word = next_word
        st.session_state.q_started_at = time.time()
        next_row = words_df[words_df["headword"] == next_word].iloc[0]
        st.session_state.qdata = build_question_payload(next_word, next_row["synonyms"])
        st.session_state.grid_for_word = next_word
        st.session_state.grid_keys = [f"opt_{next_word}_{i}" for i in range(len(st.session_state.qdata['choices']))]
        st.session_state.selection = set()
        st.session_state.answered = False
        st.session_state.eval = None
        st.rerun()

# Sidebar health
#st.sidebar.header("Health")
#if st.sidebar.button("DB ping"):
 #   try:
  #      with engine.connect() as conn:
   #         one = conn.execute(text("SELECT 1")).scalar()
    #    st.sidebar.success(f"DB OK (result={one})")
    #except Exception as e:
     #   st.sidebar.error(f"DB error: {e}")
# ─────────────────────────────────────────────────────────────────────
# Tweaks requested on 2025-09-30 (append-only patch)
# - Custom titles for Admin/Student pages
# - Back button for previous question
# - Word counter: X of N completed in current lesson
# ─────────────────────────────────────────────────────────────────────

def _hide_default_h1_and_set(title_text: str):
    # Hide the first-level Streamlit title (h1) and set our own
    st.markdown("""
        <style>
        h1 {display:none;}
        </style>
    """, unsafe_allow_html=True)
    st.title(title_text)

# 1) Custom headline per role (post-login pages)
if "auth" in st.session_state:
    _role = st.session_state.get("auth", {}).get("role")
    if _role == "admin":
        _hide_default_h1_and_set("welcome to English Learning made easy - Teacher Console")
    elif _role == "student":
        _hide_default_h1_and_set("welcome to English Learning made easy - Student login")

# 2) Student-only enhancements: Back button + word counter
if "auth" in st.session_state and st.session_state.get("auth", {}).get("role") == "student":
    # Safety: ensure required vars exist (these are defined above in the student block)
    _cid = locals().get("cid", None)
    _lid = locals().get("lid", None)
    _words_df = locals().get("words_df", pd.DataFrame())
    _active = st.session_state.get("active_word", None)

    # --- Word counter state (unique words completed in this lesson) ---
    # We count a word as "completed" when the student has submitted an answer for it.
    if _lid is not None:
        _attempt_key = f"__attempted_words_{_lid}"
        if _attempt_key not in st.session_state:
            st.session_state[_attempt_key] = set()

        # When a submission just happened (answered True) record it once per word
        if st.session_state.get("answered") and st.session_state.get("eval"):
            _record_flag_key = "__last_recorded_word"
            if st.session_state.get(_record_flag_key) != st.session_state.get("active_word"):
                st.session_state[_attempt_key].add(st.session_state.get("active_word"))
                st.session_state[_record_flag_key] = st.session_state.get("active_word")

        # Compute X of N
        _total_words = int(_words_df.shape[0]) if isinstance(_words_df, pd.DataFrame) else 0
        _completed_words = len(st.session_state[_attempt_key])
        # Show a compact counter just under the page title area
        st.caption(f"Lesson progress: **{_completed_words} / {_total_words} words completed**")

    # --- Back button support ---
    # We keep a lightweight trail of previously shown questions (by headword) to step back.
    _trail_key = "__q_trail"
    if _trail_key not in st.session_state:
        st.session_state[_trail_key] = []

    # Detect forward navigation (active_word changed): push previous word to trail
    _last_seen_key = "__last_active_word"
    _last_active = st.session_state.get(_last_seen_key)
    if _active and _last_active and _active != _last_active:
        # push last question onto trail
        st.session_state[_trail_key].append(_last_active)
    # Update last active tracker
    if _active:
        st.session_state[_last_seen_key] = _active

    # Render a Back button (works whether the current question is answered or not)
    def _restore_question(prev_word: str):
        """Restore the previous question payload and reset state without logging a new attempt."""
        if prev_word is None or _words_df.empty:
            return
        row_prev = _words_df[_words_df["headword"] == prev_word]
        if row_prev.empty:
            return
        st.session_state.active_word = prev_word
        st.session_state.q_started_at = time.time()
        st.session_state.qdata = build_question_payload(prev_word, row_prev.iloc[0]["synonyms"])
        st.session_state.grid_for_word = prev_word
        st.session_state.grid_keys = [f"opt_{prev_word}_{i}" for i in range(len(st.session_state.qdata['choices']))]
        st.session_state.selection = set()    # clear selections on restore to avoid stale checks
        st.session_state.answered = False     # ensure they must submit again
        st.session_state.eval = None

    # Place the Back button below feedback area (or near the bottom otherwise)
    with st.container():
        c_back, _ = st.columns([1, 5])
        with c_back:
            if st.button("◀ Back", key="btn_back_to_prev"):
                if st.session_state[_trail_key]:
                    _prev = st.session_state[_trail_key].pop()  # last word
                    _restore_question(_prev)
                    st.rerun()
# ─────────────────────────────────────────────────────────────────────
# Option A: Attempted-aware sidebar progress (append-only override)
# Shows % Attempted until any words are Mastered; then shows % Mastered.
# Returns (mastered_count, total_words, percent_for_sidebar)
# ─────────────────────────────────────────────────────────────────────
def course_progress(user_id: int, course_id: int):
    # All unique headwords in this course
    all_words = pd.read_sql(
        text("""
            SELECT w.headword
            FROM lessons L
            JOIN lesson_words lw ON lw.lesson_id = L.lesson_id
            JOIN words w ON w.word_id = lw.word_id
            WHERE L.course_id=:c
        """),
        con=engine, params={"c": int(course_id)}
    )["headword"].tolist()
    unique_words = list(set(all_words))
    total = len(unique_words)
    if total == 0:
        return (0, 0, 0)

    # Mastered words in this course for this user
    mastered = pd.read_sql(
        text("""
            SELECT COUNT(*) AS c
            FROM word_stats
            WHERE user_id=:u AND mastered=TRUE AND headword = ANY(:arr)
        """),
        con=engine, params={"u": int(user_id), "arr": unique_words}
    )["c"].iloc[0]

    # Attempted at least once (distinct headwords) for this course
    attempted = pd.read_sql(
        text("""
            SELECT COUNT(DISTINCT headword) AS c
            FROM attempts
            WHERE user_id=:u AND course_id=:c AND headword = ANY(:arr)
        """),
        con=engine, params={"u": int(user_id), "c": int(course_id), "arr": unique_words}
    )["c"].iloc[0]

    # Sidebar percent logic: if no mastered yet, show attempted %
    numerator = mastered if mastered > 0 else attempted
    percent = int(round(100 * numerator / total))
    return (int(mastered), total, percent)
# ─────────────────────────────────────────────────────────────────────
# AUTH INTEGRATION (append-only)
# - init AuthService
# - enforce student expiry (auto sign-out + rerun)
# - sidebar "Account" → change password
# - admin tool: reopen student (+365 days)
# ─────────────────────────────────────────────────────────────────────
try:
    from auth_service import AuthService
    auth = AuthService(engine)
except Exception as _e:
    auth = None
    st.sidebar.warning("Auth service not initialized. Check auth_service.py")

# 1) Enforce expiry AFTER any login; if expired, immediately log out and rerun
if auth and "auth" in st.session_state and st.session_state["auth"].get("role") == "student":
    try:
        _u = user_by_email(st.session_state["auth"]["email"])
        if auth.is_student_expired(_u):
            st.sidebar.error("Your account has expired. Ask your teacher to reopen access.")
            st.session_state.pop("auth", None)
            st.rerun()
    except Exception:
        pass

# 2) Post-login: Sidebar "Account" to change password (for Admin and Student)
if auth and "auth" in st.session_state:
    st.sidebar.markdown("---")
    with st.sidebar.expander("Account"):
        # Show student expiry info
        if st.session_state["auth"]["role"] == "student":
            try:
                with engine.begin() as _conn:
                    _exp = _conn.execute(text("SELECT expires_at FROM users WHERE user_id=:u"),
                                         {"u": int(st.session_state["auth"]["user_id"])}).scalar()
                if _exp:
                    st.caption(f"Access until: {str(_exp)}")
            except Exception:
                pass

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

        # Optional: also email a reset link while logged in
        if st.button("Email me a reset link", key="acct_send_reset"):
            base_url = os.getenv("APP_BASE_URL", "")
            ok, msg = auth.email_password_reset(st.session_state["auth"]["email"], base_url)
            st.info(msg) if ok else st.error(msg)

# 3) Admin-only: quick tool to Reopen student (+365d)
if auth and "auth" in st.session_state and st.session_state["auth"]["role"] == "admin":
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
            ok, msg = auth.reopen_student(int(_sel), days=365)
            st.success(msg) if ok else st.error(msg)









