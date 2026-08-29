import time
import json
import io
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .schemas import HealthResponse, ModelInfoResponse, PredictResponse
from src.model import vlm_service, ModelNotLoadedError

from src.config import load_config, model_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Артефакт грузится один раз при старте."""
    cfg = load_config()
    
    m_path = str(model_path(cfg))
    
    p_path = cfg["MODEL"].get("processor_path", m_path)
    device = cfg["MODEL"].get("device", "cuda")
    
    try:
        vlm_service.load(model_name=m_path, processor_name=p_path, device=device)
        print(f"Модель успешно загружена из: {m_path}")
    except Exception as exc:
        print(f"Ошибка загрузки модели: {exc}")
    yield
    print("Остановка сервиса, очистка ресурсов.")

app = FastAPI(
    title="VLM Inference API", 
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Модель не загружена"},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok" if vlm_service.is_ready else "degraded",
        model_loaded=vlm_service.is_ready,
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info():
    if not vlm_service.is_ready:
        raise ModelNotLoadedError
    
    return ModelInfoResponse(
        model_name=vlm_service.model_name,
        processor_name=vlm_service.processor_name,
        device=vlm_service.device,
        is_ready=vlm_service.is_ready
    )

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="Изображение для анализа"),
    label_map_json: str = Form(..., description='JSON строка, например: {"happy": "a picture of a happy dog", "sad": "a picture of a sad dog"')
):
    """Основной метод инференса."""
    if not vlm_service.is_ready:
        raise ModelNotLoadedError

    try:
        label_map = json.loads(label_map_json)
        if not isinstance(label_map, dict) or not label_map:
            raise ValueError
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="label_map_json должен быть валидным JSON-словарем"
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Невозможно прочитать файл как изображение"
        )

    started = time.perf_counter()
    try:
        predicted_class, probabilities = vlm_service.predict(image=pil_image, label_map=label_map)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {str(exc)}"
        )
    process_time = (time.perf_counter() - started) * 1000

    return PredictResponse(
        predicted_class=predicted_class,
        probabilities=probabilities,
        process_time_ms=process_time
    )