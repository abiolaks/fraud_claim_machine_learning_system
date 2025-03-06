import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# Set page configuration as the first Streamlit command
st.set_page_config(page_title="InsureGuard AI", page_icon="🔍", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    h1 {
        color: #2E86C1;
        text-align: center;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stFileUploader>div>div>div>div>div {
        border: 2px dashed #2E86C1;
        border-radius: 5px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .fraud {
        background-color: #F2D7D5;
    }
    .legit {
        background-color: #D5F5E3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Header Section
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(
                "<h1 style='text-align: center;'>🔍 InsureGuard AI</h1>",
                unsafe_allow_html=True,
            )
            st.markdown("### AI-Powered Insurance Fraud Detection System")
            st.write(
                "Upload your insurance claims data below to identify potential fraudulent cases"
            )

    # Main Content
    with st.container():
        st.divider()

        # File Upload Section
        with st.expander("📤 Upload Claims Data", expanded=True):
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload CSV file",
                type=["csv"],
                label_visibility="collapsed",
            )

        if uploaded_file:
            try:
                # Load CSV data
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return

            try:
                # Correct file paths for model and preprocessor
                st.write("Current Working Directory:", os.getcwd())
                model = joblib.load("fraud_model.pkl")
                preprocessor = joblib.load("fraud_preprocessor.pkl")
            except FileNotFoundError:
                st.error(
                    "Model and preprocessor files not found. Please check your file paths."
                )
                return

            try:
                # Feature engineering similar to training
                df["incident_date"] = pd.to_datetime(df["incident_date"])
                df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
                df["policy_age"] = (
                    df["incident_date"] - df["policy_bind_date"]
                ).dt.days
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

                # Results Summary
                st.success("✅ Analysis Complete!")
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Claims Analyzed", len(df))
                    with col2:
                        st.metric(
                            "Potential Fraud Cases",
                            f"{sum(predictions)} ({sum(predictions)/len(df):.1%})",
                        )
                    with col3:
                        st.metric("Average Confidence", f"{probabilities.mean():.1%}")

                # Detailed Results
                st.subheader("📄 Detailed Predictions")
                results = pd.DataFrame(
                    {
                        "Claim ID": df.index + 1,
                        "Status": [
                            "Fraud Alert! 🚨" if p == 1 else "Legitimate ✅"
                            for p in predictions
                        ],
                        "Confidence Level": probabilities.round(2),
                        "Risk Category": pd.cut(
                            probabilities,
                            bins=[0, 0.3, 0.7, 1],
                            labels=["Low", "Medium", "High"],
                            include_lowest=True,
                        ),
                    }
                )

                # Styled Results Table
                for index, row in results.iterrows():
                    card_class = (
                        "fraud" if row["Status"] == "Fraud Alert! 🚨" else "legit"
                    )
                    st.markdown(
                        f"""
                        <div class="prediction-card {card_class}">
                            <div style="display: flex; justify-content: space-between;">
                                <div><strong>Claim #{row['Claim ID']}</strong></div>
                                <div>{row['Status']}</div>
                            </div>
                            <div style="margin-top: 15px;">
                                Confidence: {row['Confidence Level']*100:.0f}%<br>
                                Risk Category: <span style="color: {'#E74C3C' if row['Risk Category'] == 'High' else '#F1C40F' if row['Risk Category'] == 'Medium' else '#2ECC71'}">● {row['Risk Category']}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Download Button
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Report",
                    data=csv,
                    file_name="fraud_analysis_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
        st.markdown("## About InsureGuard AI")
        st.write("""
        **InsureGuard AI** leverages advanced machine learning to:
        - Detect suspicious claim patterns
        - Analyze historical data trends
        - Provide risk assessments
        - Support fraud investigations
        """)
        st.markdown("---")
        st.markdown("### Model Performance")
        st.metric("Accuracy", "94.7%")
        st.metric("Precision", "92.3%")
        st.metric("Recall", "89.5%")


if __name__ == "__main__":
    main()
