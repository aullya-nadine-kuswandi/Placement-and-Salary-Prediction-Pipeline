from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

classifier = joblib.load('artifacts/placement_classifier.pkl')
regressor = joblib.load('artifacts/salary_regressor.pkl')

class Student(BaseModel):
    gender: str
    branch: str
    cgpa: float
    tenth_percentage: float
    twelfth_percentage: float
    backlogs: int
    study_hours_per_day: float
    attendance_percentage: float
    projects_completed: int
    internships_completed: int
    coding_skill_rating: int
    communication_skill_rating: int
    aptitude_skill_rating: int
    hackathons_participated: int
    certifications_count: int
    sleep_hours: float
    stress_level: int
    part_time_job: str
    family_income_level: str
    city_tier: str
    internet_access: str
    extracurricular_involvement: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post('/predict')
def predict(student: Student):
    data = student.dict()
    df = pd.DataFrame([data])
    
    placement_pred = classifier.predict(df)[0]
    placement_proba = classifier.predict_proba(df)[0]
    
    if placement_pred == 1:
        salary_pred = regressor.predict(df)[0]
        salary_pred = max(0.0, float(salary_pred))
    else:
        salary_pred = 0.0
    
    return {
        'placement_status': 'Placed' if placement_pred == 1 else 'Not Placed',
        'proba_placed': float(placement_proba[1]),
        'estimated_salary_lpa': round(salary_pred, 2)
    }