from typing import Any, List

from pydantic import BaseModel, Field


class ShapFactor(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Observed feature value")
    display_value: Any = Field(None, description="Human-readable feature value")
    shap_value: float = Field(..., description="SHAP contribution")


class ExplanationRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate identifier")
    prediction: str = Field(..., description="Predicted class or label")
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction probability or confidence score",
    )
    top_positive_factors: List[ShapFactor] = Field(default_factory=list)
    top_negative_factors: List[ShapFactor] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    """Response model for employee prediction with SHAP values."""
    employee_index: int = Field(..., description="Employee index in the dataset")
    prediction: str = Field(..., description="Prediction label (High Risk / Low Risk)")
    probability: float = Field(..., description="Probability of leaving (0-1)")
    top_positive_factors: List[ShapFactor] = Field(default_factory=list)
    top_negative_factors: List[ShapFactor] = Field(default_factory=list)


class LLMExplanationPayload(BaseModel):
    summary: str
    possible_explanation: str 
    detailed_explanation: str
    main_factors: List[str]
    caution: str


class ExplanationResponse(BaseModel):
    candidate_id: str
    prediction: str
    probability: float
    summary: str
    possible_explanation: str
    detailed_explanation: str
    main_factors: List[str]
    caution: str


class HealthResponse(BaseModel):
    status: str