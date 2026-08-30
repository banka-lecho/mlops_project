import os
import json
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
    assert data["model_name"]
    assert data["processor_name"]
    
def test_prediction(base_url):
    label_map = {
        "happy": "a photo of a happy dog",
        "sad": "a photo of a sad dog",
        "angry": "a photo of an angry dog",
        "relaxed": "a photo of an relaxed dog"
    }

    with open("tests/data/happy_dog.jpg", "rb") as image:
        response = requests.post(
            f"{base_url}/predict",
            files={
                "image": ("happy_dog.jpg", image, "image/jpeg")
            },
            data={
                "label_map_json": json.dumps(label_map)
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] in label_map
    assert set(data["probabilities"]) == set(label_map)

    assert 0 <= data["process_time_ms"]
    
def test_invalid_label_map(base_url):
    with open("tests/data/happy_dog.jpg", "rb") as image:
        response = requests.post(
            f"{base_url}/predict",
            files={
                "image": ("happy_dog.jpg", image, "image/jpeg")
            },
            data={
                "label_map_json": "not-json"
            }
        )

    assert response.status_code == 422