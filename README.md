# Student Skill Gap Analysis & Job Role Prediction

> **Academic Project** | Second-Year CSE | Data Science Laboratory  
> **Disclaimer:** Uses a synthetic educational dataset for demonstration. Results do not represent real employment outcomes.

---

## 📋 Project Overview

An intelligent career guidance system that analyses a student's technical skill profile, identifies gaps relative to target IT job roles, predicts suitable careers using machine learning, and recommends a personalised learning roadmap.

## 🏗️ Project Structure

```
Student Skill Gap Analysis/
├── app.py                    ← Streamlit dashboard (main entry point)
├── requirements.txt          ← Python dependencies
├── README.md
├── data/
│   ├── generate_datasets.py  ← Generates all three CSV datasets
│   ├── students.csv          ← (auto-generated) Student skill profiles
│   ├── job_roles.csv         ← (auto-generated) Job role requirements
│   └── skill_mapping.csv     ← (auto-generated) Skill-to-career mapping
├── src/
│   ├── data_preprocessing.py ← Data loading, cleaning, feature engineering
│   ├── eda.py                ← Visualisation functions (Matplotlib/Plotly)
│   ├── skill_gap.py          ← Skill gap engine (transparent formula)
│   ├── train_model.py        ← ML model training & evaluation
│   ├── predict_job.py        ← Model loading & prediction
│   └── recommendations.py   ← Learning roadmap & resource engine
├── models/                   ← (auto-generated after training)
│   ├── job_role_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── feature_columns.joblib
├── reports/                  ← (auto-generated) EDA charts & reports
├── notebooks/
│   └── exploratory_analysis.ipynb
└── tests/
    └── test_skill_gap.py     ← pytest unit tests
```

## 🚀 Quick Start (Windows — VS Code Terminal)

### Step 1 — Verify Python

```powershell
python --version
# Must be 3.9 or higher
```

### Step 2 — Create and Activate Virtual Environment

```powershell
# Navigate to the project folder
cd "C:\Users\Ponnarasan P\Downloads\Student Skill Gap Analysis"

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

> You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> This installs Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, and Joblib.

### Step 4 — Generate Datasets

```powershell
python data/generate_datasets.py
```

**Expected output:**
```
Generating Student Skills Dataset (300 synthetic records)...
  Saved → data/students.csv  (300 rows)
Generating Job Role Requirements Dataset...
  Saved → data/job_roles.csv  (8 rows)
Generating Skill-to-Career Mapping Dataset...
  Saved → data/skill_mapping.csv  (120 rows)
✅ All datasets generated successfully.
```

### Step 5 — Train the Machine Learning Model

```powershell
python src/train_model.py
```

**Expected output:**
```
STEP 1 — Loading & Preprocessing Data
  students.csv → 300 rows, ...

STEP 5 — Model Comparison (5-Fold Cross-Validation)
  [Logistic Regression] CV F1 = 0.XXXX ± 0.XXXX ...
  [Random Forest] CV F1 = 0.XXXX ± 0.XXXX ...
  ...
✅ Training complete!
```

> Models are saved to the `models/` directory.

### Step 6 — Run the Streamlit Dashboard

```powershell
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

### Step 7 — Run Unit Tests

```powershell
python -m pytest tests/ -v
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | KPI cards, skill distribution, job role overview |
| 👤 Student Profile | Enter skill ratings and academic details |
| 📊 Skill Gap Analysis | Radar chart, bar chart, match score, priority gaps |
| 🎯 Job Role Prediction | ML predictions with probability gauges |
| 🗺️ Learning Roadmap | Beginner → Advanced resources, projects, certifications |
| 📄 Reports | Download CSV reports, view model classification report |

---

## 🤖 Machine Learning Models Evaluated

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline, interpretable |
| Decision Tree | Visualisable rules |
| Random Forest | Usually best; provides feature importance |
| SVM (RBF Kernel) | Effective for non-linear boundaries |

Best model selected by weighted F1-score on test set.

---

## 📐 Skill Gap Formula

```
Gap_i = max(Required_i − Student_i, 0)

WeightedMatchScore = (Σ importance_i × min(student_i, required_i)) 
                     ────────────────────────────────────────────── × 100
                     (Σ importance_i × required_i)
```

- **importance_i** ∈ {0, 1, 2, 3} from the skill mapping dataset
- Score always stays in the range [0%, 100%]

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Testing | pytest |

---

## ⚠️ Important Academic Notes

1. **Synthetic Dataset**: All 300 student records are computer-generated for educational demonstration. They do NOT represent real students.
2. **No Employment Guarantee**: Predictions are workflow demonstrations, not career counselling or job eligibility assessments.
3. **Correlation ≠ Causation**: Skill correlations shown in EDA do not imply causal relationships.
4. **Model Accuracy**: Accuracy metrics are computed on the synthetic test set only. Real-world performance would require real labelled employment data.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: students.csv` | Run `python data/generate_datasets.py` |
| `FileNotFoundError: job_role_model.joblib` | Run `python src/train_model.py` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| `(venv)` not showing | Re-run `venv\Scripts\activate` |
| Slow dashboard | Close other browser tabs; first load caches data |

---

## 📚 References

- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
- Pandas documentation: https://pandas.pydata.org/docs/
- Plotly Python: https://plotly.com/python/
- Google Data Analytics Certificate: https://grow.google/certificates/data-analytics/
- Kaggle Learn: https://www.kaggle.com/learn
