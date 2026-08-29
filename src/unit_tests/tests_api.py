import io
import json
import pytest
import configparser
from PIL import Image
from fastapi.testclient import TestClient

from src.api.main import app
from src.model import vlm_service

@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch):
    """
    Эта фикстура автоматически применяется ко всем тестам.
    Она подменяет чтение реального config.ini и тяжеловесную ML-модель.
    """
    
    dummy_cfg = configparser.ConfigParser()
    dummy_cfg.read_dict({
        "MODEL": {
            "model_path": "test-model-path",
            "processor_path": "test-processor-path",
            "device": "cpu"
        }
    })
    
    monkeypatch.setattr(
        "src.api.main.load_config",
        lambda *args, **kwargs: dummy_cfg
    )

    monkeypatch.setattr(
        "src.api.main.model_path",
        lambda cfg=None: "test-model-path"
    )

    def mock_load(model_name, processor_name, device="cpu"):
        vlm_service.is_ready = True
        vlm_service.model_name = model_name
        vlm_service.processor_name = processor_name
        vlm_service.device = device

    monkeypatch.setattr(vlm_service, "load", mock_load)

    def mock_predict(image, label_map):
        probs = {k: 0.1 for k in label_map}
        first_key = list(label_map.keys())[0]
        probs[first_key] = 0.9  
        return first_key, probs

    monkeypatch.setattr(vlm_service, "predict", mock_predict)


@pytest.fixture
def client():
    """
    Создаем клиент поверх приложения.
    Обязательно используем контекстный менеджер, 
    чтобы принудительно запустить события lifespan.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_image():
    """Генерирует тестовую картинку в оперативной памяти."""
    file = io.BytesIO()
    image = Image.new("RGB", (224, 224), color="blue")
    image.save(file, "jpeg")
    file.seek(0)
    return file


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_endpoint(client):
    """Проверяем, что API отдал пути из нашего замоканного конфига."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test-model-path"
    assert data["processor_name"] == "test-processor-path"
    assert data["is_ready"] is True


def test_predict_success(client, test_image):
    label_map = {"cat": "a photo of a cat", "dog": "a photo of a dog"}
    
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
        data={"label_map_json": json.dumps(label_map)}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["predicted_class"] == "cat"
    assert "cat" in data["probabilities"]
    assert data["probabilities"]["cat"] == 0.9
    assert "process_time_ms" in data
    assert "X-Process-Time-Ms" in response.headers


def test_predict_invalid_json(client, test_image):
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
        data={"label_map_json": '["это", "не", "словарь"]'}
    )
    assert response.status_code == 422


def test_predict_invalid_image(client):
    """Отправляем текстовую фигню под видом картинки"""
    label_map = {"dog": "a photo of a happy dog"}
    
    response = client.post(
        "/predict",
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data={"label_map_json": json.dumps(label_map)}
    )
    assert response.status_code == 400
    assert "Невозможно прочитать файл" in response.json()["detail"]