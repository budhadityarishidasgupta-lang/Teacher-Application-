import streamlit as st

def render_spelling_student(user):
    """
    Simple placeholder for the Spelling student app.
    This will be replaced with real logic later.
    """
    st.title("Spelling Trainer (Coming Soon)")

    if user:
        name = user.get("name") or user.get("email") or "learner"
    else:
        name = "learner"

    st.write(f"Hi {name} 👋")
    st.info("The spelling practice app will appear here once we connect it to the database.")
