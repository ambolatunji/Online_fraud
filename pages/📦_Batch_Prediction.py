import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from utils import compute_features, load_model, list_saved_models
from tensorflow.keras.models import load_model as load_dl_model
from stable_baselines3 import DQN
import io
import os

st.set_page_config(page_title="📦 Batch Prediction", layout="wide")
st.title("📦 Batch Fraud Prediction")

# Upload File
uploaded = st.file_uploader("Upload your transaction file", type=["csv", "xlsx"])
if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    st.success("✅ File Loaded")
    st.dataframe(df.head())
    X = df.drop(columns=["isFraud", "nameOrig", "nameDest", "step"], errors="ignore")
    if "isFraud" in df.columns:
        y = df["isFraud"]

    # Handle categorical columns in X only
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        # Limit categories to prevent memory explosion
        #max_categories = 100  # Adjust based on your system capabilities
        #for col in cat_cols:
         #   if X[col].nunique() > max_categories:
          #      st.warning(f"⚠️ Column '{col}' has too many unique values ({X[col].nunique()}). Limiting to top {max_categories}.")
           #     top_cats = X[col].value_counts().nlargest(max_categories).index
            #    X[col] = X[col].apply(lambda x: x if x in top_cats else 'Other')
        
        # Create dummies only on the features, not the entire dataframe
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        st.success("✅ Dummy Variables Created")

    if st.checkbox("Apply Feature Engineering"):
        df = compute_features(df)
        st.success("✅ Feature Engineering Done")

    if st.checkbox("Show Correlation Matrix"):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    # Feature Selection
    features = st.multiselect("Select features to train model on", df.columns.tolist(), default=df.columns.tolist())
    df = df[features + ['isFraud']] if 'isFraud' in df.columns else df[features]

    st.subheader("🔎 Choose Prediction Model")
    model_list = list_saved_models()
    selected_model = st.selectbox("Select model", model_list)

    if selected_model:
        X = df.drop(columns=["isFraud", "nameOrig", "nameDest", "step"], errors="ignore")
        if "type" in X.columns and X["type"].dtype == "object":
            X = pd.get_dummies(X, columns=["type"], drop_first=True)

        if selected_model.endswith("_dl"):
            model = load_dl_model(f"models/{selected_model}.h5")
            X = X.astype(np.float32)
            preds = (model.predict(X).ravel() > 0.5).astype(int)
            confs = model.predict(X).ravel()

        elif selected_model.endswith("_rl"):
            from gymnasium import spaces
            class DummyEnv:
                def __init__(self, X):
                    self.X = X.astype(np.float32)
                    self.index = 0
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
                    self.action_space = spaces.Discrete(2)
                def reset(self, seed=None, options=None):
                    self.index = 0
                    return self.X[self.index], {}
                def step(self, action):
                    self.index += 1
                    done = self.index >= len(self.X)
                    obs = self.X[self.index % len(self.X)] if not done else self.X[0]
                    return obs, 1, done, False, {}

            rl_model = DQN.load(f"models/{selected_model}.zip")
            env = DummyEnv(X)
            preds = []
            obs, _ = env.reset()
            for _ in range(len(env.X)):
                action, _ = rl_model.predict(obs, deterministic=True)
                preds.append(action)
                obs, _, done, _, _ = env.step(action)
            preds = np.array(preds)
            confs = preds
            

        else:
            if "isFraud" in df.columns:
                df = df.rename(columns={"isFraud": "isFraud_actual"})
                y = df["isFraud_actual"]  # Update y reference if you're using it

            model = load_model(selected_model)
            preds = model.predict(X)
            confs = model.predict_proba(X)[:, 1]
        

        # Show color-coded result
        st.subheader("🔎 Prediction Results")
        df["isFraud_predicted"] = preds  # Instead of "Predicted"
        df["Confidence"] = confs
        def label(row):
            if row["isFraud_predicted"] == 1:
                return f"🚨 FRAUD ({row['Confidence']:.2%})"
            else:
                return f"✅ LEGIT ({row['Confidence']:.2%})"

        df["Prediction_Label"] = df.apply(label, axis=1)
        #st.dataframe(df[["Predicted", "Confidence", "Prediction_Label"]].join(df.drop(columns=["Predicted", "Confidence", "Prediction_Label"])).head(50))
        st.dataframe(df.head(50))

        

        if "isFraud_actual" in df.columns:
            actual = df["isFraud_actual"].astype(int)
            predicted = df["isFraud_predicted"].astype(int)
            acc = accuracy_score(actual, predicted)
            auc = roc_auc_score(actual, df["Confidence"])
            cm = confusion_matrix(actual, predicted)

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(actual, df["Confidence"])

            st.metric("Accuracy", f"{acc:.2%}")
            st.metric("AUC Score", f"{auc:.2f}")

            # Display ROC curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1)
            ax1.set_title("Confusion Matrix")
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")
            
            # ROC Curve
            ax2.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.2f})')
            ax2.plot([0, 1], [0, 1], 'r--', label='Random')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig)

            # Display FPR and TPR values
            with st.expander("View FPR and TPR values"):
                fpr_tpr_df = pd.DataFrame({
                    'False Positive Rate': fpr,
                    'True Positive Rate': tpr,
                    'Thresholds': thresholds
                })
                st.dataframe(fpr_tpr_df)

        # Export
        # CSV Export
        st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv", mime="text/csv")

        # Excel Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Predictions")
            writer.save()
        st.download_button(
            label="Download Excel",
            data=excel_buffer.getvalue(),
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )