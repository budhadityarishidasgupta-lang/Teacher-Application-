import os, time, random, sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from passlib.hash import bcrypt

import os
from sqlalchemy import create_engine, text
import streamlit as st

# --- Student-only mode (env or URL param) ---
FORCE_STUDENT = os.getenv("FORCE_STUDENT_MODE", "0") == "1"
try:
    qp = st.query_params  # Streamlit ≥1.30
except Exception:
    qp = st.experimental_get_query_params()  # fallback for older versions

def _first(qv):
    if qv is None: return None
    if isinstance(qv, list): return qv[0]
    return qv

_mode = (_first(qp.get("mode")) or "").strip().lower()
if _mode == "student":
    FORCE_STUDENT = True
elif _mode == "admin":
    FORCE_STUDENT = False

# --- Normalize & validate DATABASE_URL (no other DB changes) ---
_raw = os.environ.get("DATABASE_URL", "").strip()
if not _raw:
    st.error("DATABASE_URL is not set. In Render → Settings → Environment, add DATABASE_URL using your Postgres Internal Connection String.")
    st.stop()

def _normalize(url: str) -> str:
    # Keep it minimal: fix deprecated scheme only
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url

DATABASE_URL = _normalize(_raw)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)

# Create tables on startup (safe no-op if they already exist)
try:
    from init_db import init as init_db
    init_db()
except Exception as e:
    st.sidebar.warning(f"DB init warning: {e}")

# ── Config ───────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "synquest.db"
load_dotenv(APP_DIR / ".env", override=True)

ENABLE_GPT    = os.getenv("ENABLE_GPT", "0") == "1"
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY", "")

ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", "admin@example.com").strip().lower()
ADMIN_NAME     = os.getenv("ADMIN_NAME", "Admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "ChangeMe!123")

# Optional GPT client
gpt_client = None
if ENABLE_GPT and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        gpt_client = None
        ENABLE_GPT = False

st.set_page_config(page_title="Synonym Quest — Admin & Student", page_icon="📚", layout="wide")

# ── DB schema (SQLite local) ─────────────────────────────────────────
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
        # users: role = admin | student
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
          user_id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          email TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('admin','student')),
          is_active INTEGER NOT NULL DEFAULT 1,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS courses(
          course_id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          description TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS lessons(
          lesson_id INTEGER PRIMARY KEY AUTOINCREMENT,
          course_id INTEGER NOT NULL,
          title TEXT NOT NULL,
          sort_order INTEGER DEFAULT 0,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(course_id) REFERENCES courses(course_id)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS words(
          word_id INTEGER PRIMARY KEY AUTOINCREMENT,
          headword TEXT NOT NULL,
          synonyms TEXT NOT NULL,
          difficulty INTEGER DEFAULT 2
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS lesson_words(
          lesson_id INTEGER,
          word_id INTEGER,
          sort_order INTEGER DEFAULT 0,
          PRIMARY KEY(lesson_id, word_id),
          FOREIGN KEY(lesson_id) REFERENCES lessons(lesson_id),
          FOREIGN KEY(word_id) REFERENCES words(word_id)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS enrollments(
          user_id INTEGER,
          course_id INTEGER,
          PRIMARY KEY(user_id, course_id),
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(course_id) REFERENCES courses(course_id)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS attempts(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER, course_id INTEGER, lesson_id INTEGER,
          headword TEXT, is_correct INTEGER, response_ms INTEGER,
          chosen TEXT, correct_choice TEXT,
          ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS word_stats(
          user_id INTEGER, headword TEXT,
          correct_streak INTEGER DEFAULT 0,
          total_attempts INTEGER DEFAULT 0,
          correct_attempts INTEGER DEFAULT 0,
          last_seen TIMESTAMP,
          mastered INTEGER DEFAULT 0,
          difficulty INTEGER DEFAULT 2,
          due_date TIMESTAMP,
          PRIMARY KEY (user_id, headword)
        );""")

def create_user(name, email, password, role):
    h = bcrypt.hash(password)
    with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
        cur.execute("INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)",
                    (name, email, h, role))
        return cur.lastrowid

def user_by_email(email):
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id,name,email,password_hash,role,is_active FROM users WHERE email=?", (email,))
        r = cur.fetchone()
    return None if not r else dict(user_id=r[0], name=r[1], email=r[2], password_hash=r[3], role=r[4], is_active=r[5])

def ensure_admin():
    """Create the admin account if it doesn't exist."""
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id FROM users WHERE role='admin' LIMIT 1")
        row = cur.fetchone()
    if row: return
    try:
        create_user(ADMIN_NAME, ADMIN_EMAIL, ADMIN_PASSWORD, "admin")
    except sqlite3.IntegrityError:
        pass

def set_user_active(user_id, active: bool):
    with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
        cur.execute("UPDATE users SET is_active=? WHERE user_id=?", (1 if active else 0, user_id))

def all_students_df():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df_users = pd.read_sql_query("SELECT user_id,name,email,is_active FROM users WHERE role='student'", conn)
        df_stats = pd.read_sql_query("""
            SELECT user_id,
                   SUM(correct_attempts) AS correct_total,
                   SUM(total_attempts)   AS attempts_total,
                   SUM(mastered)         AS mastered_count,
                   MAX(last_seen)        AS last_active
            FROM word_stats GROUP BY user_id
        """, conn)
    df = df_users.merge(df_stats, on="user_id", how="left")
    for c in ["correct_total","attempts_total","mastered_count"]: df[c]=df[c].fillna(0).astype(int)
    return df.sort_values("name")

def lesson_words(course_id, lesson_id):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        q = """
        SELECT w.headword, w.synonyms, w.difficulty
        FROM lesson_words lw JOIN words w ON w.word_id=lw.word_id
        WHERE lw.lesson_id=? AND EXISTS (SELECT 1 FROM lessons L WHERE L.lesson_id=lw.lesson_id AND L.course_id=?)
        ORDER BY lw.sort_order
        """
        return pd.read_sql_query(q, conn, params=(lesson_id, course_id))

def mastered_count(user_id, lesson_id):
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            "SELECT w.headword FROM lesson_words lw JOIN words w USING(word_id) WHERE lw.lesson_id=?",
            (lesson_id,)
        )
        words = [r[0] for r in cur.fetchall()]

    if not words:
        return 0, 0

    qmarks = ",".join("?" for _ in words)
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM word_stats WHERE user_id=? AND mastered=1 AND headword IN ({qmarks})",
            (user_id, *words)
        )
        (m,) = cur.fetchone()

    return m, len(words)

def update_after_attempt(user_id, course_id, lesson_id, headword, is_correct, response_ms, difficulty, chosen, correct_choice):
    with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
        cur.execute("SELECT correct_streak FROM word_stats WHERE user_id=? AND headword=?", (user_id, headword))
        row = cur.fetchone()
        prior = row[0] if row else 0
        new_streak = prior + 1 if is_correct else 0
        mastered = 1 if new_streak >= 3 else 0
        add_days = 3 if (is_correct and new_streak >= 3) else (1 if is_correct else 0)
        due = (datetime.utcnow() + timedelta(days=add_days)).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
            INSERT INTO word_stats (user_id, headword, correct_streak, total_attempts, correct_attempts, last_seen, mastered, difficulty, due_date)
            VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            ON CONFLICT(user_id, headword) DO UPDATE SET
                correct_streak=?,
                total_attempts=word_stats.total_attempts+1,
                correct_attempts=word_stats.correct_attempts + (?),
                last_seen=CURRENT_TIMESTAMP,
                mastered=CASE WHEN ?=1 THEN 1 ELSE word_stats.mastered END,
                difficulty=?,
                due_date=?""",
            (user_id, headword, new_streak, 1 if is_correct else 0, mastered, difficulty, due,
             new_streak, 1 if is_correct else 0, mastered, difficulty, due))
        cur.execute("""
            INSERT INTO attempts(user_id,course_id,lesson_id,headword,is_correct,response_ms,chosen,correct_choice)
            VALUES(?,?,?,?,?,?,?,?)""",
            (user_id, course_id, lesson_id, headword, 1 if is_correct else 0, int(response_ms), chosen, correct_choice))

def recent_stats(user_id, course_id, lesson_id, n=10):
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute("""SELECT is_correct, response_ms FROM attempts
                       WHERE user_id=? AND course_id=? AND lesson_id=?
                       ORDER BY id DESC LIMIT ?""", (user_id, course_id, lesson_id, n))
        rows = cur.fetchall()
    if not rows: return {"accuracy":0.0,"avg_ms":15000.0}
    return {"accuracy": float(np.mean([r[0] for r in rows])),
            "avg_ms": float(np.mean([r[1] for r in rows]))}

def choose_next_word(user_id, course_id, lesson_id, df_words):
    """Adaptive next word selector (targets difficulty 1/2/3 based on recent accuracy & speed)."""
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

# ── Enhancements helpers ─────────────────────────────────────────────
def course_progress(user_id: int, course_id: int):
    """Return (completed, total, percent) for a course for this user."""
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute("""
            SELECT w.headword
            FROM lessons L
            JOIN lesson_words lw ON lw.lesson_id = L.lesson_id
            JOIN words w ON w.word_id = lw.word_id
            WHERE L.course_id=?
        """, (course_id,))
        all_words = [r[0] for r in cur.fetchall()]
    total = len(set(all_words))
    if total == 0:
        return (0, 0, 0)

    qmarks = ",".join("?" for _ in set(all_words))
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM word_stats WHERE user_id=? AND mastered=1 AND headword IN ({qmarks})",
            (user_id, *list(set(all_words)))
        )
        (completed,) = cur.fetchone()
    percent = int(round(100 * completed / total))
    return (completed, total, percent)

def build_question_payload(headword: str, synonyms_str: str):
    """
    Build a stable 6-option question:
      - 2 correct (first two synonyms if available; fall back to 1+variant)
      - 4 distractors from a small pool (MVP)
    Returns: dict with keys: headword, choices (list), correct (set)
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
    rnd.shuffle(choices)  # shuffled once; stored in session so it doesn't change
    return {"headword": headword, "choices": choices, "correct": set(correct)}

def gpt_explain(headword, choices, correct_choice, user_choice):
    if not gpt_client:
        return ("Great! " if user_choice==correct_choice else "Nice try. ") + \
               (f"'{headword}' and '{correct_choice}' mean nearly the same thing.")
    try:
        prompt = f"""
        You're a kind tutor for kids 7–10.
        Word: '{headword}'. Choices: {choices}. Correct: '{correct_choice}'. Learner chose: '{user_choice}'.
        1) One short reason why the correct answer fits.
        2) One tiny example sentence using the word + correct answer.
        Keep it friendly & brief."""
        resp = gpt_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You teach vocabulary to kids with simple language."},
                      {"role":"user","content":prompt}],
            temperature=0.4, max_tokens=120)
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Tip: pick the choice that means almost the same as the word."

# ── Bootstrap ────────────────────────────────────────────────────────
init_db()
ensure_admin()

# ── Auth ─────────────────────────────────────────────────────────────
def login_form():
    st.sidebar.subheader("Sign in")

    # Force student UI when requested
    if FORCE_STUDENT:
        mode = "Student"
    else:
        mode = st.sidebar.radio("Login as", ["Admin","Student"], horizontal=True, key="login_mode")

    email = st.sidebar.text_input("Email", key="login_email")
    pwd   = st.sidebar.text_input("Password", type="password", key="login_pwd")

    if st.sidebar.button("Login", type="primary", key="btn_login"):
        u = user_by_email(email.strip().lower())
        if not u:
            st.sidebar.error("User not found.")
            return
        if not u["is_active"]:
            st.sidebar.error("Account disabled.")
            return
        if not bcrypt.verify(pwd, u["password_hash"]):
            st.sidebar.error("Wrong password.")
            return

        # Enforce role rules (student-only link cannot admit admins)
        if mode == "Admin" and u["role"] != "admin":
            st.sidebar.error("Not an admin account.")
            return
        if mode == "Student" and u["role"] != "student":
            if FORCE_STUDENT:
                st.sidebar.error("This is a student-only link. Please use the admin URL.")
                return
            st.sidebar.error("Not a student account.")
            return

        st.session_state.auth = {
            "user_id": u["user_id"],
            "name": u["name"],
            "email": u["email"],
            "role": u["role"]
        }
        st.sidebar.success(f"Welcome {u['name']}!")

    if st.sidebar.button("Log out", key="btn_logout"):
        st.session_state.pop("auth", None)

if "auth" not in st.session_state:
    login_form()
    st.title("Synonym Quest — Admin & Student")
    st.write("Sign in as **Admin** to manage students, courses and tests; or as **Student** to learn and take tests.")
    st.stop()

ROLE = st.session_state.auth["role"]
USER_ID = st.session_state.auth["user_id"]
NAME = st.session_state.auth["name"]
st.sidebar.caption(f"Signed in as **{NAME}** ({ROLE})")

# ── ADMIN EXPERIENCE ─────────────────────────────────────────────────
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
                except sqlite3.IntegrityError:
                    st.error("Email already exists.")

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

    # Teacher Dashboard — upload files, create tests, assign tests
    with tab_teacher:
        st.subheader("Courses")
        with st.form("create_course"):
            title = st.text_input("Course title", key="td_course_title")
            desc  = st.text_area("Description", "", key="td_course_desc")
            ok = st.form_submit_button("Create course")  # fixed: no key
            if ok and title.strip():
                with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
                    cur.execute("INSERT INTO courses(title,description) VALUES(?,?)", (title,desc))
                st.success("Course created.")

        with closing(sqlite3.connect(DB_PATH)) as conn:
            df_courses = pd.read_sql_query("SELECT course_id,title,description FROM courses ORDER BY course_id DESC", conn)
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
                ok = st.form_submit_button("Create lesson")  # fixed: no key
                if ok and lt.strip():
                    with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
                        cur.execute("INSERT INTO lessons(course_id,title,sort_order) VALUES(?,?,?)", (cid_lessons, lt, int(order)))
                    st.success("Lesson created.")

            st.markdown("**Upload CSV of words (headword,synonyms)**")
            f = st.file_uploader("Upload CSV", type=["csv"], key="td_upload_csv")
            if f:
                df_up = pd.read_csv(f)
                st.dataframe(df_up.head(), use_container_width=True)
                with closing(sqlite3.connect(DB_PATH)) as conn:
                    df_less = pd.read_sql_query("SELECT lesson_id,title FROM lessons WHERE course_id=? ORDER BY sort_order", conn, params=(cid_lessons,))
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
                        with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
                            for _,r in df_up.iterrows():
                                hw = str(r["headword"]).strip()
                                syns = str(r["synonyms"]).strip()
                                if not hw or not syns: continue
                                syn_list=[s.strip() for s in syns.split(",") if s.strip()]
                                diff = 1 if (len(hw)<=6 and len(syn_list)<=3) else (2 if len(hw)<=8 and len(syn_list)<=5 else 3)
                                cur.execute("INSERT INTO words(headword,synonyms,difficulty) VALUES(?,?,?)",(hw,", ".join(syn_list),diff))
                                wid = cur.lastrowid
                                cur.execute("INSERT OR IGNORE INTO lesson_words(lesson_id,word_id,sort_order) VALUES(?,?,?)",(lid_upload,wid,n)); n+=1
                        st.success(f"Imported {n} words.")

        st.subheader("Assign courses to students")
        with closing(sqlite3.connect(DB_PATH)) as conn:
            students = pd.read_sql_query("SELECT user_id,name FROM users WHERE role='student' AND is_active=1 ORDER BY name", conn)
            df_courses_assign = pd.read_sql_query("SELECT course_id,title FROM courses ORDER BY title", conn)
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
                with closing(sqlite3.connect(DB_PATH)) as conn, conn, closing(conn.cursor()) as cur:
                    cur.execute("INSERT OR IGNORE INTO enrollments(user_id,course_id) VALUES(?,?)", (sid_assign,cid_assign))
                st.success("Enrolled.")

    # Student Dashboard — class overview
    with tab_student:
        st.subheader("Student Overview")
        with closing(sqlite3.connect(DB_PATH)) as conn:
            attempts = pd.read_sql_query("""
                SELECT U.name, A.course_id, A.lesson_id, A.headword, A.is_correct, A.response_ms, A.ts
                FROM attempts A JOIN users U ON U.user_id=A.user_id
                ORDER BY A.id DESC LIMIT 500
            """, conn)
        st.dataframe(attempts, use_container_width=True)
        st.caption("Latest attempts across students. Filter/export via table menu.")

# ── STUDENT EXPERIENCE ───────────────────────────────────────────────
if ROLE == "student":
    st.title("🎓 Student")
    with closing(sqlite3.connect(DB_PATH)) as conn:
        courses = pd.read_sql_query("""
            SELECT C.course_id, C.title
            FROM enrollments E JOIN courses C ON C.course_id=E.course_id
            WHERE E.user_id=?""", conn, params=(USER_ID,))

    # Sidebar: My courses + completion %
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

    # Lessons for selected course
    with closing(sqlite3.connect(DB_PATH)) as conn:
        lessons = pd.read_sql_query("SELECT lesson_id,title FROM lessons WHERE course_id=? ORDER BY sort_order", conn, params=(cid,))
    if lessons.empty:
        st.info("This course has no lessons yet.")
        st.stop()

    l_map = dict(zip(lessons["lesson_id"], lessons["title"]))
    lid = st.selectbox("Lesson", list(l_map.keys()), format_func=lambda x: l_map[x], key="student_lesson_select")

    # Load words for the selected lesson
    words_df = lesson_words(cid, lid)
    if words_df.empty:
        st.info("This lesson has no words yet.")
        st.stop()

    # Progress header
    if "asked_history" not in st.session_state:
        st.session_state.asked_history = []
    m, total = mastered_count(USER_ID, lid)
    st.progress(min(m/max(total,1),1.0), text=f"Mastered {m}/{total} words")

    # --- Active question state (stable across reruns) ---
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
        # reset answered state when a new word loads
        st.session_state.answered = False
        st.session_state.eval = None

    # ensure flags exist
    if "answered" not in st.session_state:
        st.session_state.answered = False
    if "eval" not in st.session_state:
        st.session_state.eval = None

    active = st.session_state.active_word
    row = words_df[words_df["headword"] == active].iloc[0]
    qdata = st.session_state.qdata
    choices = qdata["choices"]
    correct_set = qdata["correct"]

    # ----- QUIZ FORM (no auto-advance on Submit) -----
    if not st.session_state.answered:
        with st.form("quiz_form", clear_on_submit=False):
            st.subheader(f"Word: **{active}**")
            st.write("Pick the **synonyms** (select all that apply), then press **Submit**.")

            # stable keys per word; reset selection when word changes already handled above
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
                if new_val:
                    temp_selection.add(opt)
                else:
                    temp_selection.discard(opt)

            c1, c2 = st.columns([1, 1])
            with c1:
                submitted = st.form_submit_button("Submit", type="primary")
            with c2:
                nextq = st.form_submit_button("Next ▶")

        # Commit temp selection after form interaction
        st.session_state.selection = temp_selection

        # Handle buttons
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

            # Persist evaluation & freeze form until Next
            st.session_state.answered = True
            st.session_state.eval = {
                "is_correct": bool(is_correct),
                "picked_set": set(picked_set),
                "correct_set": set(correct_set),
                "choices": list(choices)
            }
            st.rerun()

        elif nextq:
            # Prevent advancing without submit
            st.warning("Please **Submit** your answer first, then click **Next**.")

    # After Submit: show feedback + Next (outside the form), and ONLY Next advances
    if st.session_state.answered and st.session_state.eval:
        ev = st.session_state.eval
        st.subheader(f"Word: **{active}**")
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

        # Add GPT explanation + two examples
try:
    correct_choice_for_text = sorted(list(ev["correct_set"]))[0]
    why, examples = gpt_feedback_examples(active, correct_choice_for_text)
    st.info(f"**Why:** {why}")
    st.markdown(f"**Examples:**\n\n- {examples[0]}\n- {examples[1]}")
except Exception:
    pass

        # NEXT button (advances only after submit)
        if st.button("Next ▶", use_container_width=True):
            st.session_state.asked_history.append(active)
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

# --- Health check (put here or at the very end) ---
st.sidebar.header("Health")
if st.sidebar.button("DB ping"):
    try:
        with engine.connect() as conn:
            one = conn.execute(text("SELECT 1")).scalar()
        st.sidebar.success(f"DB OK (result={one})")
    except Exception as e:
        st.sidebar.error(f"DB error: {e}")

