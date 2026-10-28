
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Locate the model in the project root
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Create the API
app = FastAPI(
    title="Student Exam Score Prediction API",
    description="Predict exam scores using student performance data.",
    version="1.0.0",
)

# These fields must match the feature columns used during Colab training.
# Confirm their exact names and order from X_reg.columns in Colab.
class StudentData(BaseModel):
    study_hours: float = Field(ge=0)
    attendance_percentage: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0)
    internet_usage: float = Field(ge=0)
    assignments_completed: int = Field(ge=0)
    previous_scores: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Student Exam Score Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: StudentData):
    # The column names and order must match the training data.
    features = pd.DataFrame(
        [[
            data.study_hours,
            data.attendance_percentage,
            data.sleep_hours,
            data.internet_usage,
            data.assignments_completed,
            data.previous_scores,
        ]],
    columns=[
        "study_hours",
        "attendance",
        "sleep_hours",
        "internet_usage",
        "assignments_completed",
        "previous_score",
        ],
    )

    prediction = float(model.predict(features)[0])

    return {
        "predicted_exam_score": round(prediction, 2)
    }
