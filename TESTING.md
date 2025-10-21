# Testing Guide

This project is a Streamlit application without an automated test suite at the moment. You can still verify changes manually by running the app locally. The steps below outline the recommended approach.

## 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Provide Required Environment Variables

The app expects a PostgreSQL connection string in `DATABASE_URL`. For local testing you can point it to any Postgres instance, including one started with Docker:

```bash
export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/teacher_app
```

If you only need to explore the UI without a database, you can set a temporary in-memory placeholder:

```bash
export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/postgres
```

(Ensure the URL points to a reachable database, otherwise Streamlit will stop the app.)

## 3. Launch the Streamlit App

```bash
streamlit run app.py
```

Use the browser interface to exercise the flows you modified. For example, update quiz progress, verify the mastery bar, and confirm there are no console errors.

## 4. (Optional) Linting Checks

If you would like a quick sanity check on imports and formatting, run:

```bash
python -m compileall app.py
```

This ensures there are no syntax errors before deploying changes.

---

Automated tests can be added later; if you introduce them, update this guide with the appropriate commands (e.g., `pytest`).
