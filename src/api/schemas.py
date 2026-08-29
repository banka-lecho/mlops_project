from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., description="ok или degraded")
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_name: str
    processor_name: str
    device: str
    is_ready: bool

class PredictResponse(BaseModel):
    predicted_class: str = Field(..., description="Класс с максимальной вероятностью")
    probabilities: dict[str, float] = Field(..., description="Распределение вероятностей по всем классам")
    process_time_ms: float = Field(..., description="Время инференса")