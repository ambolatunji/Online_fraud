import streamlit as st

st.set_page_config(page_title="Welcome to Fraud Predictor", layout="wide")

st.title("👋 Welcome to the Online Fraud Detection App")

st.markdown("""
### 🚦 How to Use This App

This application allows you to analyze and detect online fraud using machine learning models. Here's how you can get started:

---

#### 🔐 **Single Prediction**
Use this mode to:
- Manually input transaction details
- Predict whether the transaction is fraudulent
- Train a new model or upload a pre-trained one
- Visualize predictions and view confidence scores

---

#### 📦 **Batch Prediction**
Use this mode to:
- Upload a dataset (CSV, Excel, PDF, Word, Text)
- Automatically extract and visualize the data
- Engineer features, select model or upload
- Predict in bulk and download results in Excel/CSV
- View accuracy, AUC, ROC curves

---

#### 📘 **View Notebooks**
Use this mode to:
- Upload or select training notebooks (`.ipynb`)
- Convert them to PDF automatically
- Download for offline access or auditing

---

### 🧭 Navigation
Use the sidebar on the left to switch between:
- **Single Prediction**
- **Batch Prediction**
- **View Notebooks**

Happy analyzing!
""")