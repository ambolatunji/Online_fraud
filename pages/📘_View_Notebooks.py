import streamlit as st
import nbformat
from nbconvert import PDFExporter

st.set_page_config(page_title="View Notebooks", layout="wide")
st.title("📘 Notebook Viewer")

notebooks = ["EDA.ipynb", "Feature_Engineering_Full.ipynb", "ml_models_fraud.ipynb", 
             "dl_models_fraud.ipynb", "ml_hyperparameter_tuning.ipynb", 
             "dl_fast_hyperparameter_tuning.ipynb", "rl_model_fraud.ipynb", 
             "rl_advanced_fraud.ipynb"]

selected = st.selectbox("Select notebook to view", notebooks)

if selected:
    path = f"{selected}"
    with open(path, "r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)
        pdf_exporter = PDFExporter()
        pdf_exporter.exclude_input_prompt = True
        pdf_exporter.exclude_output_prompt = True
        try:
            body, _ = pdf_exporter.from_notebook_node(notebook_node)
            with open("notebook_view.pdf", "wb") as f:
                f.write(body)
            with open("notebook_view.pdf", "rb") as f:
                st.download_button("Download Notebook as PDF", f, file_name="notebook_view.pdf", mime="application/pdf")
        except Exception as e:
            st.error("PDF export failed. Ensure Pandoc is installed.")