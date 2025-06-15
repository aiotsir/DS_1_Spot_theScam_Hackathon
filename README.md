# DS_1_Spot_theScam_Hackathon
Anvesh Hackathon June Edition 2025- DS-1 Spot the Scam - Flag the Fraudulent Job Postings

🕵️‍♀️ Spot the Scam: Job Posting Fraud Detection
Online job platforms are increasingly targeted by scammers. Fake job listings waste time and pose serious risks to users' money and personal data.
This project builds a machine learning system that detects fraudulent job posts and offers interactive insights through a Streamlit dashboard.

# Results
------------------
**F1-Score  | 0.99 **

**Classification Report (per class):**

| Metric    | Class 0 (Not Fraudulent) | Class 1 (Fraudulent) |
| --------- | ------------------------ | -------------------- |
| Precision | 1.00                     | 0.99                 |
| Recall    | 0.99                     | 1.00                 |
| F1-Score  | 0.99                     | 0.99                 |
| Support   | 2723                     | 2722                 |

**Confusion Matrix**
              Predicted
              0      1
Actual  0  | 2699   24
        1  |   6   2716
        
This means :

| Term                     | Meaning                                                        |
| ------------------------ | -------------------------------------------------------------- |
| **True Negatives (TN)**  | 2699 → Correctly predicted as **not fraudulent** (class 0)     |
| **False Positives (FP)** | 24 → Incorrectly predicted as **fraudulent** when actually not |
| **False Negatives (FN)** | 6 → Predicted as **not fraudulent**, but actually fraudulent   |
| **True Positives (TP)**  | 2716 → Correctly predicted as **fraudulent** (class 1)         |



🚀 Project Highlights
✅ Binary classifier trained on genuine vs. fraudulent job postings

📦 Accepts CSV uploads and returns predictions with fraud probabilities

📊 Visual dashboard with:

Fraud probability histogram

Real vs. fraud pie chart

Top-10 most suspicious jobs

📈 Handles class imbalance using SMOTE oversampling

✨ Built with SBERT embeddings + structured features + XGBoost

📂 Directory Structure
bash
Copy
Edit
.
├── app.py                        # Streamlit dashboard
├── test_predictions.csv          # Sample test output
├── xgb_model.pkl                 # Trained model
├── scaler.pkl                    # Final feature scaler
├── sbert_model.pkl               # SBERT model or encoder
├── test_data.csv                 # Test input (no labels)
├── requirements.txt
                   
⚙️ Technologies Used
Type	Tech
Language	Python
Modeling	XGBoost, SMOTE
Embeddings	Sentence-BERT (sentence-transformers)
Dashboard	Streamlit
Visuals	Seaborn, Matplotlib
Others	scikit-learn, pandas, numpy

🧠 Model Overview
Text Description → SBERT Embeddings (384 dims)

Structured Features → ['has_company_logo', 'has_questions', 'telecommuting']

Combined and scaled → Final feature vector

Trained with XGBoostClassifier on SMOTE-resampled data

Evaluation metric: F1-score

📥 How to Run Locally
1. Clone This Repository
 
git clone https://github.com/aiotsir/DS_1_Spot_theScam_Hackathon
 
2. Set Up Environment
 
pip install -r requirements.txt
3. Run the App
 
 streamlit run app.py
 

📊 Dashboard Preview
Upload CSV: with fields like title, description, location, etc.

Predictions: Class + fraud probability

Charts:

Histogram of fraud probabilities

Pie chart of real vs. fraud predictions

Table of top-10 suspicious listings

Download: Full prediction CSV
