# =============================
# 📁 File: app.py (Streamlit)
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# === Streamlit Config ===
st.set_page_config(page_title="Job Fraud Detector Dashboard", layout="wide")
st.title("🚨 Job Posting Fraud Detection Dashboard")

# === Load models and encoders ===
model = joblib.load("ai_models/xgb_model.pkl")
structured_scaler = joblib.load("ai_models/scaler.pkl")

# Encoders
le_title = joblib.load("ai_models/encoder_models/le_title.pkl")
le_department = joblib.load("ai_models/encoder_models/le_department.pkl")
le_location = joblib.load("ai_models/encoder_models/le_location.pkl")
le_employment_type = joblib.load("ai_models/encoder_models/le_employment_type.pkl")

te_industry = joblib.load("ai_models/encoder_models/te_industry.pkl")
te_function = joblib.load("ai_models/encoder_models/te_function.pkl")
te_required_education = joblib.load("ai_models/encoder_models/te_required_education.pkl")
te_required_experience = joblib.load("ai_models/encoder_models/te_required_experience.pkl")

# === Load SBERT model ===
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper: Parse salary ===
def parse_salary(s):
    try:
        low, high = s.split('-')
        return float(low.strip()), float(high.strip())
    except:
        return np.nan, np.nan

# === Preprocessing Function ===
def preprocess_uploaded_data(df):
    df = df.copy()
    df.fillna('', inplace=True)

    # === Parse salary range ===
    salary_split = df['salary_range'].fillna('').apply(parse_salary)
    df['salary_min'] = salary_split.apply(lambda x: x[0])
    df['salary_max'] = salary_split.apply(lambda x: x[1])

    # === Add missing flags ===
    missing_cols = [
        'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits',
        'location', 'employment_type', 'required_experience', 'required_education', 'industry', 'function'
    ]
    for col in missing_cols:
        df[f'{col}_missing'] = df[col] == ''

    # === Clean text columns (optional here, but can be extended) ===
    def clean_text(text):
        return str(text).replace('\n', ' ').strip()

    for col in ['description', 'requirements', 'company_profile', 'benefits']:
        df[f'clean_{col}'] = df[col].apply(clean_text)

    # === SBERT Embeddings ===
    long_text_cols = ['clean_description', 'clean_requirements', 'clean_company_profile', 'clean_benefits']
    sbert_embeddings = np.hstack([
        sbert_model.encode(df[col].tolist(), show_progress_bar=False) for col in long_text_cols
    ])
    sbert_embeddings_scaled = sbert_scaler.transform(sbert_embeddings)

    # === Label Encoding ===
    df['title_le'] = le_title.transform(df['title'].astype(str).fillna('missing'))
    df['department_le'] = le_department.transform(df['department'].astype(str).fillna('missing'))
    df['location_le'] = le_location.transform(df['location'].astype(str).fillna('missing'))
    df['employment_type_le'] = le_employment_type.transform(df['employment_type'].astype(str).fillna('missing'))

    # === Target Encoding ===
    df['industry_te'] = te_industry.transform(df['industry'])
    df['function_te'] = te_function.transform(df['function'])
    df['required_education_te'] = te_required_education.transform(df['required_education'])
    df['required_experience_te'] = te_required_experience.transform(df['required_experience'])

    # === Structured Features ===
    structured_cols = [
        'telecommuting', 'has_company_logo', 'has_questions',
        'salary_min', 'salary_max',
        'industry_te', 'function_te', 'required_education_te', 'required_experience_te',
        'title_le', 'department_le', 'location_le', 'employment_type_le'
    ] + [col for col in df.columns if col.endswith('_missing')]

    structured_data = df[structured_cols].fillna(0).values
    structured_data_scaled = structured_scaler.transform(structured_data)

    # === Final Feature Set ===
    X = np.hstack([sbert_embeddings_scaled, structured_data_scaled])
    return df, X

# === Streamlit File Upload ===
uploaded_file = st.file_uploader("📁 Upload CSV with job listings", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df, X_features = preprocess_uploaded_data(df)

    # === Make Predictions ===
    fraud_probs = model.predict_proba(X_features)[:, 1]
    predicted_labels = model.predict(X_features)

    df['Fraud_Probability'] = fraud_probs
    df['Predicted_Label'] = predicted_labels

    st.success("✅ Predictions complete!")

    # 🔟 Top Suspicious Listings
    st.markdown("### 🔎 Top 10 Most Suspicious Job Listings")
    st.dataframe(df.sort_values('Fraud_Probability', ascending=False).head(10)[['title', 'location', 'Fraud_Probability']])

    # 📊 Histogram
    st.markdown("### 📊 Fraud Probability Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Fraud_Probability'], bins=20, kde=True, ax=ax1)
    ax1.set_xlabel("Fraud Probability")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # 🥧 Pie Chart
    st.markdown("### 🥧 Fraud vs Real Predictions")
    pie_data = df['Predicted_Label'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(pie_data, labels=["Real", "Fraud"], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    ax2.axis('equal')
    st.pyplot(fig2)

    # 📋 Full Table
    st.markdown("### 📋 Full Prediction Table")
    st.dataframe(df[['title', 'Predicted_Label', 'Fraud_Probability']])

    # 💾 Download Button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Results as CSV", data=csv_data, file_name="fraud_predictions.csv", mime="text/csv")

else:
    st.info("📁 Please upload a job listings CSV to begin.")
