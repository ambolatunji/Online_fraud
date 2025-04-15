import streamlit as st

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("🔐 Online Fraud Prediction Dashboard")

st.markdown("""
Welcome to the fraud prediction system.

Use the **sidebar** to navigate:
- 🔐 Single Transaction Prediction
- 📦 Batch Transaction Prediction
- 📘 View Training Notebooks
- 🏠 Instructions & Guide
""")