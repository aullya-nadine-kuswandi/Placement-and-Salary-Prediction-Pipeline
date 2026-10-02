# Placement & Salary Prediction

A machine learning project that predicts whether a student will be **placed** and, if so, estimates the expected **salary (LPA)**. It uses a two-stage approach: a classifier predicts placement status, and a regressor estimates salary for students predicted as placed. The project includes a reproducible training pipeline with MLflow tracking, a deployment approval gate, a FastAPI service, and two Streamlit apps.

🚀 **Live Demo:** [Open the Streamlit app]([https://your-app-url](https://placementandsalarymodeldeployment-aullya-nadine-kuswandi.streamlit.app/))

## Features

- **Two-stage prediction**: placement classification + salary regression
- **Reproducible pipeline**: ingestion → training → evaluation, with a fixed seed (`random_state=42`)
- **Experiment tracking** with MLflow (parameters, models, and test metrics)
- **Deployment approval gate** with separate thresholds for the classifier and the regressor
- **REST API** built with FastAPI
- **Two Streamlit apps**: a standalone app and one that calls the API
- **Interpretability**: the standalone app shows the top model coefficients that influence placement

## Dataset

5,000 students with 22 input features, split into two files:

| File | Content |
|------|---------|
| `A.csv` | Student features (academic, skills, activity, lifestyle, background) |
| `A_targets.csv` | Targets: `placement_status` (Placed / Not Placed) and `salary_lpa` |

The two files are joined on `Student_ID`. The classes are imbalanced (about 86% Placed, 14% Not Placed), so the classifier uses balanced class weights.

**Feature groups**

| Group | Columns |
|-------|---------|
| Academic | `cgpa`, `tenth_percentage`, `twelfth_percentage`, `backlogs`, `attendance_percentage` |
| Skills | `coding_skill_rating`, `communication_skill_rating`, `aptitude_skill_rating` |
| Activity | `projects_completed`, `internships_completed`, `hackathons_participated`, `certifications_count`, `extracurricular_involvement` |
| Lifestyle | `study_hours_per_day`, `sleep_hours`, `stress_level`, `part_time_job` |
| Background | `gender`, `branch`, `family_income_level`, `city_tier`, `internet_access` |

## Project Structure

```
.
├── A.csv                    # Raw features
├── A_targets.csv            # Raw targets
├── data_ingestion.py        # Step 1: merge features + targets and validate
├── train.py                 # Step 2: preprocessing + model training + MLflow logging
├── evaluation.py            # Step 3: test-set evaluation + MLflow metrics
├── pipeline.py              # Orchestrator: runs all steps + approval gate
├── fastApi.py               # FastAPI prediction service
├── app_usingAPI.py          # Streamlit app that calls the FastAPI service
├── app_streamlit.py         # Standalone Streamlit app (loads models directly)
├── exploration.ipynb        # Exploratory data analysis & model experiments
├── artifacts/               # Saved models (generated)
│   ├── placement_classifier.pkl
│   └── salary_regressor.pkl
└── ingested/                # Merged dataset A_join.csv (generated)
```

## Pipeline Overview

| Step | Module | What it does |
|------|--------|--------------|
| 1. Ingestion | `data_ingestion.py` | Merges `A.csv` and `A_targets.csv` on `Student_ID`, validates the data, and saves `ingested/A_join.csv` |
| 2. Classifier training | `train.py` | Trains a Logistic Regression model on the stratified 80% training split |
| 3. Classifier evaluation | `evaluation.py` | Evaluates on the 20% test split and logs metrics to MLflow |
| 4. Regressor training | `train.py` | Trains a Linear Regression model on **placed students only** |
| 5. Regressor evaluation | `evaluation.py` | Evaluates on placed students in the test split |
| 6. Approval | `pipeline.py` | Checks both models against the thresholds below |

### Preprocessing (shared by both models)

| Feature type | Columns | Transformation |
|--------------|---------|----------------|
| Numeric | 15 columns (e.g. `cgpa`, `backlogs`, `stress_level`) | Median imputation + `RobustScaler` |
| Nominal | `branch` | Most-frequent imputation + one-hot encoding |
| Ordinal | `gender`, `part_time_job`, `family_income_level`, `city_tier`, `internet_access`, `extracurricular_involvement` | Most-frequent imputation + ordinal encoding with explicit category order |

### Models

| Task | Model | Settings |
|------|-------|----------|
| Placement classification | Logistic Regression | `penalty='l1'`, `C=0.01`, `solver='liblinear'`, `class_weight='balanced'` |
| Salary regression | Linear Regression | Trained only on students with `placement_status = Placed` |

At inference time, the salary is estimated only when the classifier predicts **Placed**; otherwise it is returned as `0.0`.

## Results

Evaluated on the held-out 20% test set (label encoding: Not Placed = 0, Placed = 1).

### Classifier

| Metric | Value |
|--------|:-----:|
| Accuracy | 0.7830 |
| F1 (macro) | 0.6941 |
| Recall (Not Placed) | 0.8777 |
| Precision (Placed) | 0.9749 |

### Regressor (placed students only)

| Metric | Value |
|--------|:-----:|
| R² | 0.7929 |
| RMSE | 1.3525 |
| MAE | 1.0676 |

### Deployment Approval Gate

Each model must meet **all** of its thresholds to be approved:

| Model | Metric | Threshold | Result | Status |
|-------|--------|:---------:|:------:|:------:|
| Classifier | F1 (macro) | ≥ 0.65 | 0.6941 | ✅ Pass |
| Classifier | Recall (Not Placed) | ≥ 0.80 | 0.8777 | ✅ Pass |
| Classifier | Precision (Placed) | ≥ 0.90 | 0.9749 | ✅ Pass |
| Regressor | R² | ≥ 0.75 | 0.7929 | ✅ Pass |
| Regressor | RMSE | ≤ 1.5 | 1.3525 | ✅ Pass |
| Regressor | MAE | ≤ 1.5 | 1.0676 | ✅ Pass |

> Recall on **Not Placed** is prioritized so that at-risk students are not missed, while precision on **Placed** keeps positive predictions trustworthy.

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 1. Run the Pipeline

```bash
python pipeline.py
```

This ingests the data, trains and evaluates both models, saves them to `artifacts/`, logs everything to MLflow, and prints the approval decision.

### 2. Explore Experiments with MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 to view runs, parameters, and metrics.

### 3. Run the Standalone Streamlit App

```bash
streamlit run app_streamlit.py
```

Fill in the profile, activity, and lifestyle tabs, then click **Make Prediction** to see the placement result, placement probability, estimated salary, a profile radar chart, and the top factors influencing placement.

### 4. Run the API + Streamlit Client (optional)

Start the API:

```bash
uvicorn fastApi:app --reload
```

In a second terminal, start the client app (it calls `http://127.0.0.1:8000/predict`):

```bash
streamlit run app_usingAPI.py
```

Interactive API docs are available at http://127.0.0.1:8000/docs.

## API Reference

### `POST /predict`

**Request body**

```json
{
  "gender": "Male",
  "branch": "CSE",
  "cgpa": 8.2,
  "tenth_percentage": 75.0,
  "twelfth_percentage": 72.0,
  "backlogs": 0,
  "study_hours_per_day": 4.0,
  "attendance_percentage": 85.0,
  "projects_completed": 4,
  "internships_completed": 1,
  "coding_skill_rating": 4,
  "communication_skill_rating": 3,
  "aptitude_skill_rating": 4,
  "hackathons_participated": 2,
  "certifications_count": 3,
  "sleep_hours": 7.0,
  "stress_level": 5,
  "part_time_job": "No",
  "family_income_level": "Medium",
  "city_tier": "Tier 2",
  "internet_access": "Yes",
  "extracurricular_involvement": "Medium"
}
```

**Response**

```json
{
  "placement_status": "Placed",
  "proba_placed": 0.87,
  "estimated_salary_lpa": 12.4
}
```

## Tech Stack

- **Data & ML**: pandas, NumPy, scikit-learn
- **Experiment tracking**: MLflow
- **API**: FastAPI, Uvicorn, Pydantic
- **App & visualization**: Streamlit, Plotly

## Notes

- `pipeline.py` expects `A.csv` and `A_targets.csv` in the project root.
- Both Streamlit apps and the API require the files in `artifacts/`, so run `python pipeline.py` first (or commit the `.pkl` files).
- `app_usingAPI.py` requires the FastAPI server to be running on port 8000.
