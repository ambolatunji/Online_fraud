import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import compute_features, save_model, list_saved_models, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Single Prediction", layout="wide")
st.title("🔐 Single Transaction Prediction")

# --- Step 1: Input transaction ---
st.subheader("Step 1: Enter Transaction")

auto_feature = st.checkbox("Apply Auto Feature Engineering", value=True)

with st.form("txn_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", min_value=0.0)
        oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)
        newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)
    with col2:
        oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
        newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)
        isFlaggedFraud = st.selectbox("Flagged Fraud?", [0, 1])

    submit = st.form_submit_button("Generate Features")

user_input_df = None
if submit:
    user_input_df = pd.DataFrame([{
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrg': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': isFlaggedFraud
    }])
    if auto_feature:
        user_input_df = compute_features(user_input_df)
    st.success("✅ Features computed.")
    st.dataframe(user_input_df)

# --- Step 2: Model Options ---
st.subheader("Step 2: Select Mode")
mode = st.radio("Choose mode", ["Use Existing Model", "Train New Model"])

# --- Predict using existing model ---
if mode == "Use Existing Model":
    models = list_saved_models()
    selected = st.selectbox("Pick a model", models)
    if st.button("Run Prediction") and user_input_df is not None:
        model = load_model(selected)
        y_pred = model.predict(user_input_df)[0]
        prob = model.predict_proba(user_input_df)[0][1]
        st.metric("Prediction", "FRAUD" if y_pred == 1 else "LEGIT")
        st.metric("Confidence", f"{prob:.2%}")

# --- Train a new model ---
if mode == "Train New Model":
    st.subheader("Step 3: Train a Model")
    training_data = st.file_uploader("Upload Labeled CSV with 'isFraud'", type=["csv"])
    model_type = st.selectbox("Choose ML Model", ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"])
    model_name = st.text_input("Save model as", value="my_model")

    if training_data:
        df = pd.read_csv(training_data)
        st.dataframe(df.head())
        if auto_feature:
            df = compute_features(df)
            st.success("Auto Feature Engineering Applied")

        X = df.drop(columns=["isFraud", "nameOrig", "nameDest", "step"], errors="ignore")
        y = df["isFraud"]

        # Feature selection
        selected_features = st.multiselect("Select Features to Train On", X.columns.tolist(), default=X.columns.tolist())
        X = X[selected_features]
        st.write("📌 Training on features:", selected_features)

        # Encode only known categorical
        if "type" in X.columns:
            X = pd.get_dummies(X, columns=["type"], drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if st.button("Train and Predict"):
            if model_type == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
            elif model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            elif model_type == "XGBoost":
                from xgboost import XGBClassifier
                model = XGBClassifier()
            elif model_type == "LightGBM":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            st.success(f"✅ Trained Model. Accuracy: {acc:.2%}, AUC: {auc:.2f}")
            save_model(model, model_name)
            st.info(f"Model saved as: `{model_name}`")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            # Predict using user input
            if user_input_df is not None:
                user_input_df = user_input_df[selected_features]
                if "type" in user_input_df.columns:
                    user_input_df = pd.get_dummies(user_input_df, columns=["type"], drop_first=True)
                user_input_df = scaler.transform(user_input_df)
                y_user = model.predict(user_input_df)[0]
                prob_user = model.predict_proba(user_input_df)[0][1]
                st.subheader("📍 Prediction for Input Above")
                st.metric("Prediction", "FRAUD" if y_user == 1 else "LEGIT")
                st.metric("Confidence", f"{prob_user:.2%}")