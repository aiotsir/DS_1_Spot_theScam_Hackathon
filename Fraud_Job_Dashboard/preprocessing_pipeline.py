import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load models and encoders
model = joblib.load("ai_models/xgb_model.pkl")
# sbert_scaler = joblib.load("sbert_scaler.pkl")
structured_scaler = joblib.load("ai_models/scaler.pkl")

le_title = joblib.load("le_title.pkl")
le_department = joblib.load("ai_models/encoder_models/le_department.pkl")
le_location = joblib.load("ai_models/encoder_models/le_location.pkl")
le_employment_type = joblib.load("ai_models/encoder_models/le_employment_type.pkl")

te_industry = joblib.load("ai_models/encoder_models/te_industry.pkl")
te_function = joblib.load("ai_models/encoder_models/te_function.pkl")
te_required_education = joblib.load("ai_models/encoder_models/te_required_education.pkl")
te_required_experience = joblib.load("ai_models/encoder_models/te_required_experience.pkl")

sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

long_text_cols = ['clean_description', 'clean_requirements', 'clean_company_profile', 'clean_benefits']

def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).strip()

def preprocess(df):
    # Create missing flags and fill NAs
    cols_with_na = ['department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits',
                    'location', 'employment_type', 'required_experience', 'required_education', 
                    'industry', 'function']
    for col in cols_with_na:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna("missing")

    # Clean text
    df['clean_description'] = df['description'].apply(clean_text)
    df['clean_requirements'] = df['requirements'].apply(clean_text)
    df['clean_company_profile'] = df['company_profile'].apply(clean_text)
    df['clean_benefits'] = df['benefits'].apply(clean_text)

    # SBERT embeddings

    long_text_cols = ['clean_description', 'clean_requirements', 'clean_company_profile', 'clean_benefits']
    sbert_embeddings = np.hstack([
        sbert_model.encode(df[col].tolist(), show_progress_bar=False) for col in long_text_cols
    ])
    sbert_embeddings_scaled = sbert_model.transform(sbert_embeddings)

    def parse_salary(s):
        try:
            low, high = s.split('-')
            return float(low.strip()), float(high.strip())
        except:
            return np.nan, np.nan

    # Parse salary range into min and max using robust custom function
    salary_split = df['salary_range'].fillna('').apply(parse_salary)
    df['salary_min'] = salary_split.apply(lambda x: x[0])
    df['salary_max'] = salary_split.apply(lambda x: x[1])

    # Salary features
    df[['salary_min', 'salary_max']] = df['salary_range'].str.extract(r'(\d+)-?(\d+)?').fillna(0).astype(float)

    # Encode structured categorical features
    df['title_le'] = le_title.transform(df['title'].fillna("missing"))
    df['department_le'] = le_department.transform(df['department'].fillna("missing"))
    df['location_le'] = le_location.transform(df['location'].fillna("missing"))
    df['employment_type_le'] = le_employment_type.transform(df['employment_type'].fillna("missing"))

    df['industry_te'] = te_industry.transform(df['industry'])
    df['function_te'] = te_function.transform(df['function'])
    df['required_education_te'] = te_required_education.transform(df['required_education'])
    df['required_experience_te'] = te_required_experience.transform(df['required_experience'])

    # Final structured features
    structured_cols = [
        'telecommuting', 'has_company_logo', 'has_questions',
        'salary_min', 'salary_max',
        'industry_te', 'function_te', 'required_education_te', 'required_experience_te',
        'title_le', 'department_le', 'location_le', 'employment_type_le',
        'department_missing', 'salary_range_missing', 'company_profile_missing', 'description_missing',
        'requirements_missing', 'benefits_missing', 'location_missing', 'employment_type_missing',
        'required_experience_missing', 'required_education_missing', 'industry_missing', 'function_missing'
    ]

    structured_data = df[structured_cols].values
    structured_scaled = structured_scaler.transform(structured_data)

    # Combine and return
    X = np.hstack([sbert_embeddings_scaled, structured_scaled])
    return X, df
