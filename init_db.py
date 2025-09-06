# init_db.py
import os
from sqlalchemy import create_engine, text

# Use Render env var
DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

DDL = """
CREATE TABLE IF NOT EXISTS users (
  user_id SERIAL PRIMARY KEY,
  role TEXT CHECK (role IN ('student','teacher','admin')) NOT NULL,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS courses (
  course_id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  grade_min INT,
  grade_max INT,
  created_by INT REFERENCES users(user_id),
  created_at TIMESTAMPTZ DEFAULT now(),
  is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS enrollments (
  user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
  course_id INT REFERENCES courses(course_id) ON DELETE CASCADE,
  enrolled_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (user_id, course_id)
);

CREATE TABLE IF NOT EXISTS lessons (
  lesson_id SERIAL PRIMARY KEY,
  course_id INT REFERENCES courses(course_id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  position INT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS questions (
  question_id SERIAL PRIMARY KEY,
  course_id INT REFERENCES courses(course_id) ON DELETE CASCADE,
  prompt TEXT NOT NULL,
  correct_answers TEXT[] NOT NULL,  -- e.g. ARRAY['rapid','swift']
  difficulty INT
);

CREATE TABLE IF NOT EXISTS attempts (
  attempt_id SERIAL PRIMARY KEY,
  user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
  question_id INT REFERENCES questions(question_id) ON DELETE CASCADE,
  selected_options TEXT[] NOT NULL,
  is_correct BOOLEAN NOT NULL,
  taken_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
  event_id BIGSERIAL PRIMARY KEY,
  user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
  course_id INT REFERENCES courses(course_id) ON DELETE CASCADE,
  lesson_id INT REFERENCES lessons(lesson_id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,           -- 'login','start_quiz','submit','next','reveal','feedback'
  properties JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id SERIAL PRIMARY KEY,
  user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
  course_id INT REFERENCES courses(course_id) ON DELETE CASCADE,
  rating INT CHECK (rating BETWEEN 1 AND 5),
  comment TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for performance at ~100k events/attempts
CREATE INDEX IF NOT EXISTS idx_attempts_user_q ON attempts(user_id, question_id);
CREATE INDEX IF NOT EXISTS idx_attempts_time ON attempts(taken_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, created_at DESC);
"""

def init():
    with engine.begin() as conn:
        # Run each statement safely
        for stmt in DDL.split(";\n\n"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

if __name__ == "__main__":
    init()
