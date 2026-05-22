from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    content: str = Field(..., min_length=1, max_length=5000)


class AspectPrediction(BaseModel):
    aspect: str
    aspect_key: str
    label: Literal["None", "Positive", "Negative", "Neutral"]
    label_id: int
    confidence: float
    probs: Dict[str, float]


class PredictResponse(BaseModel):
    input: PredictRequest
    predictions: List[AspectPrediction]
    inference_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BatchSummary(BaseModel):
    total: int
    skipped: int
    truncated_from: int | None = None
    overall_sentiment: Dict[str, int]
    by_aspect: Dict[str, Dict[str, int]]
    by_rating_aspect_neg: Dict[str, Dict[str, float]]
    insights: List[str] = []


class BatchResponse(BaseModel):
    summary: BatchSummary
    rows: List[Dict[str, Any]]
    total_ms: int
