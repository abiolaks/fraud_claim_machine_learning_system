# fraud_detection.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path


def preprocess_and_train():
    print("Current working directory:", os.getcwd())
    print("Data path being tried:", os.path.abspath("../data/insurance_claims.csv"))
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / "data" / "insurance_claims.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Convert target variable
    df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # Feature engineering
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
    df["policy_age"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
    df["incident_month"] = df["incident_date"].dt.month

    # Drop unnecessary columns
    df.drop(
        [
            "policy_number",
            "policy_bind_date",
            "incident_date",
            "incident_location",
            "auto_model",
        ],
        axis=1,
        inplace=True,
    )

    # Handle missing values
    df.replace("?", np.nan, inplace=True)

    # Define features and target
    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Define feature categories
    numeric_features = [
        "months_as_customer",
        "age",
        "policy_deductable",
        "umbrella_limit",
        "capital-gains",
        "capital-loss",
        "incident_hour_of_the_day",
        "number_of_vehicles_involved",
        "bodily_injuries",
        "witnesses",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "policy_age",
        "incident_month",
    ]

    categorical_features = [
        "policy_state",
        "policy_csl",
        "insured_sex",
        "insured_education_level",
        "insured_occupation",
        "insured_hobbies",
        "insured_relationship",
        "incident_type",
        "collision_type",
        "incident_severity",
        "authorities_contacted",
        "incident_state",
        "incident_city",
        "property_damage",
        "police_report_available",
        "auto_make",
        "auto_year",
    ]

    # Preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # First, apply preprocessing to transform the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Now, apply SMOTE on the preprocessed training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_preprocessed, y_train
    )

    # Define and train the model
    model = XGBClassifier(scale_pos_weight=2, eval_metric="logloss", random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Save model and preprocessor for later use
    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(preprocessor, "fraud_preprocessor.pkl")

    return model, preprocessor


# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance Faraud Detection", layout="wide")
    st.title("Auto Insurance Fraud Detection System")

    # Load or train model


# Load or train model and preprocessor
try:
    model = joblib.load("fraud_model.pkl")
    preprocessor = joblib.load("fraud_preprocessor.pkl")
except FileNotFoundError:
    model, preprocessor = preprocess_and_train()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Feature engineering similar to training:
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
    df["policy_age"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
    df["incident_month"] = df["incident_date"].dt.month

    # Drop unnecessary columns
    df.drop(
        [
            "policy_number",
            "policy_bind_date",
            "incident_date",
            "incident_location",
            "auto_model",
        ],
        axis=1,
        inplace=True,
    )
    df.replace("?", np.nan, inplace=True)

    # Transform the new data using the saved preprocessor
    df_transformed = preprocessor.transform(df)

    # Make predictions on the transformed data
    probabilities = model.predict_proba(df_transformed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Create and display results
    results = pd.DataFrame(
        {
            "Prediction": ["Fraud" if p == 1 else "Legitimate" for p in predictions],
            "Confidence": probabilities.round(2),
            "Risk Level": pd.cut(
                probabilities,
                bins=[0, 0.3, 0.7, 1],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            ),
        }
    )

    st.subheader("Prediction Results")
    st.dataframe(
        results.style.background_gradient(cmap="RdBu_r", subset=["Confidence"])
    )

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )
