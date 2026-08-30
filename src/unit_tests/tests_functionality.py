import os
import pytest
import requests


@pytest.fixture
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")

def test_health(base_url):
    response = requests.get(f"{base_url}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    
    
def test_model_info(base_url):
    response = requests.get(f"{base_url}/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["is_ready"] is True
    assert data["device"] in ["cpu", "cuda"]
    assert data["checkpoint_path"]
    assert data["classes"]

def test_prediction(base_url):
    with open("tests/data/happy_dog.jpg", "rb") as image:
        response = requests.post(
            f"{base_url}/predict",
            files={
                "image": ("happy_dog.jpg", image, "image/jpeg")
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"]
    assert data["probabilities"]

    assert 0 <= data["process_time_ms"]

def test_invalid_image(base_url):
    response = requests.post(
        f"{base_url}/predict",
        files={
            "image": ("not_an_image.txt", b"not an image", "text/plain")
        }
    )

    assert response.status_code == 400