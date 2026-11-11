import httpx

# Address of our running Docker API
BASE_URL = "http://127.0.0.1:8001"


def test_docker_valid_question():

    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "question": "How can I delete my account?"
        },
        timeout=10.0
    )

    data = response.json()

    assert response.status_code == 200
    assert data["predicted_intent"] == "delete_account"
    assert data["decision"] == "auto_classified"
    assert data["needs_human_review"] is False


def test_docker_unfamiliar_question():

    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "question": "My Docker container keeps restarting."
        },
        timeout=10.0
    )

    data = response.json()

    assert response.status_code == 200
    assert data["decision"] == "needs_human_review"
    assert data["needs_human_review"] is True