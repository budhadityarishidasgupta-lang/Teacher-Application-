import streamlit as st

def render_spelling_admin(user):
    """
    Simple placeholder for the Spelling admin console.
    This will be replaced with real upload/management tools later.
    """
    st.title("Spelling Admin Console (Coming Soon)")

    if user:
        name = user.get("name") or user.get("email") or "admin"
    else:
        name = "admin"

    st.write(f"Welcome, {name} 👋")
    st.info("Here you will be able to upload spelling courses and lessons in the next phase.")

