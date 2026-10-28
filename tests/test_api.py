
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# Test 1: Check whether the home endpoint works
def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Exam Score Prediction API is running" in response.json()["message"]


# Test 2: Check the health endpoint
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


# Test 3: Check whether the model predicts an exam score
def test_predict_endpoint():
    payload = {
        "study_hours": 5,
        "attendance_percentage": 85,
        "sleep_hours": 7,
        "internet_usage": 3,
        "assignments_completed": 8,
        "previous_scores": 72
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert "predicted_exam_score" in result
    assert isinstance(result["predicted_exam_score"], (int, float))


# Test 4: Check that an invalid attendance value is rejected
def test_invalid_attendance():
    payload = {
        "study_hours": 5,
        "attendance_percentage": 150,
        "sleep_hours": 7,
        "internet_usage": 3,
        "assignments_completed": 8,
        "previous_scores": 72
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Test 5: Check that missing input fields are rejected
def test_missing_input():
    payload = {
        "study_hours": 5,
        "attendance_percentage": 85
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
