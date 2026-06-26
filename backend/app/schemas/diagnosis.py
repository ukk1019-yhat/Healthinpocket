from pydantic import BaseModel
from typing import List


class DiagnosisRequest(BaseModel):
    pass


class DiagnosisResult(BaseModel):
    class_id: int
    label: str
    confidence: float


class DiagnosisResponse(BaseModel):
    filename: str
    predictions: List[DiagnosisResult]
    primary_diagnosis: DiagnosisResult
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
