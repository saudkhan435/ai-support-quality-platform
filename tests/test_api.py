from fastapi.testclient import TestClient

from main import app

# Create a client that can communicate with our API
client = TestClient(app)


def test_valid_customer_question():

    response = client.post(
        "/predict",
        json={
            "question": "How can I delete my account?"
        }
    )

    data = response.json()

    assert response.status_code == 200
    assert data["predicted_intent"] == "delete_account"
    assert data["decision"] == "auto_classified"
    assert data["needs_human_review"] is False


def test_unfamiliar_question():

    response = client.post(
        "/predict",
        json={
            "question": "My Docker container keeps restarting."
        }
    )

    data = response.json()

    assert response.status_code == 200
    assert data["decision"] == "needs_human_review"
    assert data["needs_human_review"] is True

def test_missing_question():

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422

def test_empty_question():

    response = client.post(
        "/predict",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422

def test_whitespace_only_question():

    response = client.post(
        "/predict",
        json={
            "question": "   "
        }
    )

    assert response.status_code == 422