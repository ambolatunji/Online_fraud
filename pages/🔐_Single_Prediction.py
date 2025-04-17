import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from stable_baselines3 import DQN
from tensorflow.keras.models import load_model as load_dl_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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
        'newbalanceOrig': newbalanceOrig,
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
    model_category = st.radio("Choose Model Category", ["ML", "DL", "RL", "Hyperparameter"])

    if model_category == "Hyperparameter":
        model_type = st.selectbox("Choose Model Type", ["ML", "DL", "RL"])
    elif model_category == "ML":
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

        if model_category == "Hyperparameter":
            if model_type == "ML":
                ml_algo = st.selectbox("Choose ML Algorithm", ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"])
                n_trials = st.number_input("Number of trials", min_value=1, value=10)
                
                if st.button("Optimize and Train"):
                    import optuna
                    
                    def objective(trial):
                        if ml_algo == "Logistic Regression":
                            params = {
                                'C': trial.suggest_loguniform('C', 1e-5, 1e5),
                                'max_iter': trial.suggest_int('max_iter', 100, 1000)
                            }
                            model = LogisticRegression(**params)
                        elif ml_algo == "Random Forest":
                            params = {
                                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                                'max_depth': trial.suggest_int('max_depth', 2, 32),
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                            }
                            model = RandomForestClassifier(**params)
                        elif ml_algo == "XGBoost":
                            params = {
                                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                                'max_depth': trial.suggest_int('max_depth', 3, 9),
                                'n_estimators': trial.suggest_int('n_estimators', 50, 200)
                            }
                            model = XGBClassifier(**params)
                        elif ml_algo == "LightGBM":
                            params = {
                                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                                'n_estimators': trial.suggest_int('n_estimators', 50, 200)
                            }
                            model = LGBMClassifier(**params)
                        
                        model.fit(X_train, y_train)
                        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials)
                    
                    st.success(f"Best trial AUC: {study.best_value:.4f}")
                    st.write("Best parameters:", study.best_params)
                    
                    # Train final model with best params
                    if ml_algo == "Logistic Regression":
                        final_model = LogisticRegression(**study.best_params)
                    elif ml_algo == "Random Forest":
                        final_model = RandomForestClassifier(**study.best_params)
                    elif ml_algo == "XGBoost":
                        from xgboost import XGBClassifier
                        final_model = XGBClassifier(**study.best_params)
                    else:
                        final_model = LGBMClassifier(**study.best_params)
                    
                    final_model.fit(X_train, y_train)
                    save_model(final_model, model_name)

            elif model_type == "DL":
                if st.button("Optimize and Train"):
                    def objective(trial):
                        n_layers = trial.suggest_int('n_layers', 1, 5)
                        model = Sequential()
                        model.add(Dense(trial.suggest_int(f'n_units_1', 16, 256), 
                                      activation=trial.suggest_categorical('activation_1', ['relu', 'tanh']),
                                      input_shape=(X_train.shape[1],)))
                        
                        for i in range(2, n_layers + 1):
                            model.add(Dense(trial.suggest_int(f'n_units_{i}', 16, 256),
                                          activation=trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh'])))
                        
                        model.add(Dense(1, activation='sigmoid'))
                        model.compile(optimizer=Adam(trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)),
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'])
                        
                        history = model.fit(X_train, y_train, 
                                          validation_split=0.2,
                                          epochs=20,
                                          batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                                          verbose=0)
                        return history.history['val_accuracy'][-1]

                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials)
                    st.success(f"Best validation accuracy: {study.best_value:.4f}")
                    st.write("Best parameters:", study.best_params)
                    
                    # Train final model with best params
                    final_model = build_model(**study.best_params)
                    final_model.fit(X_train, y_train, epochs=20, batch_size=study.best_params['batch_size'])
                    final_model.save(f"models/{model_name}.h5")

            elif model_type == "RL":
                if st.button("Optimize and Train"):
                    def objective(trial):
                        params = {
                            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                            'gamma': trial.suggest_uniform('gamma', 0.9, 0.99),
                            'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000])
                        }
                        
                        env = DummyEnv(X, y)
                        model = DQN("MlpPolicy", env, verbose=0, **params)
                        model.learn(total_timesteps=5000)
                        
                        # Evaluate
                        y_pred = []
                        for x in X_test:
                            action, _ = model.predict(x)
                            y_pred.append(action)
                        return roc_auc_score(y_test, y_pred)

                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials)
                    st.success(f"Best AUC: {study.best_value:.4f}")
                    st.write("Best parameters:", study.best_params)
                    
                    # Train final model with best params
                    env = DummyEnv(X, y)
                    final_model = DQN("MlpPolicy", env, verbose=1, **study.best_params)
                    final_model.learn(total_timesteps=10000)
                    final_model.save(f"models/{model_name}.zip")

        if model_category == "ML":
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

        elif model_category == "DL":
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

            if st.button("Train DL Model"):
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

                y_prob = model.predict(X_test).ravel()
                y_pred = (y_prob > 0.5).astype(int)

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)

                st.success(f"✅ Trained DL Model. Accuracy: {acc:.2%}, AUC: {auc:.2f}")
                model.save(f"models/{model_name}.h5")
                st.info(f"Model saved as: `{model_name}.h5`")

        elif model_category == "RL":
            from stable_baselines3 import DQN
            from gymnasium import spaces

            class DummyEnv:
                def __init__(self, X, y):
                    self.X = X.astype(np.float32)
                    self.y = y.astype(int)
                    self.index = 0
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
                    self.action_space = spaces.Discrete(2)

                def reset(self, seed=None, options=None):
                    self.index = 0
                    return self.X[self.index], {}

                def step(self, action):
                    reward = 1 if action == self.y[self.index] else -1
                    self.index += 1
                    done = self.index >= len(self.X)
                    obs = self.X[self.index % len(self.X)] if not done else self.X[0]
                    return obs, reward, done, False, {}

            if st.button("Train RL Model"):
                env = DummyEnv(X, y)
                rl_model = DQN("MlpPolicy", env, verbose=1)
                rl_model.learn(total_timesteps=10000)
                rl_model.save(f"models/{model_name}.zip")
                st.success(f"✅ Trained RL Model. Model saved as: `{model_name}.zip`")

        # Fix single user prediction
        if user_input_df is not None and 'model' in locals():
            try:
                # Prepare input data
                user_features = user_input_df[selected_features].copy()
                if "type" in user_features.columns:
                    user_features = pd.get_dummies(user_features, columns=["type"], drop_first=True)
                
                # Scale features if scaler exists
                if 'scaler' in locals():
                    user_features = scaler.transform(user_features)
                
                st.subheader("📍 Prediction for Input Above")
                
                if isinstance(model, (LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier)):
                    y_pred = model.predict(user_features)[0]
                    y_prob = model.predict_proba(user_features)[0][1]
                    st.metric("Prediction", "FRAUD" if y_pred == 1 else "LEGIT")
                    st.metric("Confidence", f"{y_prob:.2%}")
                
                elif 'keras' in str(type(model)):
                    y_prob = model.predict(user_features)[0][0]
                    y_pred = 1 if y_prob > 0.5 else 0
                    st.metric("Prediction", "FRAUD" if y_pred == 1 else "LEGIT")
                    st.metric("Confidence", f"{y_prob:.2%}")
                
                elif isinstance(model, DQN):
                    action, _ = model.predict(user_features)
                    st.metric("Prediction", "FRAUD" if action == 1 else "LEGIT")
                    st.info("Note: RL models don't provide probability scores")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")