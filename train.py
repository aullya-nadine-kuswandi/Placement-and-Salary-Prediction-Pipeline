import joblib
import mlflow
import mlflow.sklearn
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression

os.makedirs("artifacts", exist_ok=True)

def build_preprocessor():
    num_cols = ['cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
                'study_hours_per_day', 'attendance_percentage', 'projects_completed',
                'internships_completed', 'coding_skill_rating',
                'communication_skill_rating', 'aptitude_skill_rating',
                'hackathons_participated', 'certifications_count',
                'sleep_hours', 'stress_level']
    
    ohe_col = ['branch']
    ordinal_col = ['gender', 'part_time_job', 'family_income_level',
                   'city_tier', 'internet_access', 'extracurricular_involvement']
    ordinal_categories = [
        ['Male', 'Female'],
        ['No', 'Yes'],
        ['Low', 'Medium', 'High'],
        ['Tier 3', 'Tier 2', 'Tier 1'],
        ['No', 'Yes'],
        ['Low', 'Medium', 'High']
    ]
    
    numeric_preprocess = Pipeline([
        ('num_imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    ohe_preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    ordinal_preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_ordinal_encoder', OrdinalEncoder(categories=ordinal_categories))
    ])
    
    preprocess = ColumnTransformer([
        ('num', numeric_preprocess, num_cols),
        ('ohe', ohe_preprocess, ohe_col),
        ('ordinal', ordinal_preprocess, ordinal_col)
    ], remainder='drop')
    
    return preprocess


def train_classifier(x_train, y_train):
    preprocess = build_preprocessor()
    
    clf_pipeline = Pipeline([
        ('preprocessing', preprocess),
        ('classifier', LogisticRegression(
            random_state=42,
            class_weight='balanced',
            C=0.01,
            penalty='l1',
            solver='liblinear'
        ))
    ])
    
    mlflow.set_experiment("Placement Prediction - Classification")
    
    with mlflow.start_run(run_name="logistic_regression_tuned") as run:
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("C", 0.01)
        mlflow.log_param("penalty", "l1")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("random_state", 42)
        
        clf_pipeline.fit(x_train, y_train)
        
        joblib.dump(clf_pipeline, "artifacts/placement_classifier.pkl")
        mlflow.sklearn.log_model(clf_pipeline, artifact_path="model")
    
    return run.info.run_id


def train_regressor(x_train, y_train):
    preprocess = build_preprocessor()
    
    reg_pipeline = Pipeline([
        ('preprocessing', preprocess),
        ('regressor', LinearRegression())
    ])
    
    mlflow.set_experiment("Placement Prediction - Regression")
    
    with mlflow.start_run(run_name="linear_regression") as run:
        mlflow.log_param("model_type", "LinearRegression")
        
        reg_pipeline.fit(x_train, y_train)
        
        joblib.dump(reg_pipeline, "artifacts/salary_regressor.pkl")
        mlflow.sklearn.log_model(reg_pipeline, artifact_path="model")
    
    return run.info.run_id

if __name__ == "__main__":
    build_preprocessor()
    train_classifier()
    train_regressor()