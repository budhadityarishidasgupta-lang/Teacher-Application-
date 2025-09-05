import streamlit as st

st.set_page_config(page_title="Teacher Application", page_icon="✅", layout="centered")

st.title("Teacher Application — Synonym Quest (MVP)")
st.write("Hello 👋 This is a minimal Streamlit app deployed on Render (Free plan).")

st.subheader("Quick test")
name = st.text_input("Your name")
if st.button("Greet"):
    st.success(f"Hi {name or 'there'}! The app is working 🎉")
