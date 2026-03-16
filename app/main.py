from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.ollama_client import generate_explanation
from app.prompts import build_prompt
from app.schemas import ExplanationRequest, ExplanationResponse
from app.schemas import HealthResponse, PredictionResponse, ShapFactor
from app.settings import get_settings
from app.predictor import get_predictor, get_employee_prediction


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "API that transforms an ML prediction and SHAP factors into "
        "a structured natural-language explanation using Gemini."
    ),
)


@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend."""
    from pathlib import Path
    template_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(template_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/employees/count")
def get_employee_count() -> dict:
    """Get the total number of employees in the dataset."""
    try:
        predictor = get_predictor()
        return {"count": predictor.get_employee_count()}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get employee count: {str(exc)}",
        ) from exc


@app.get("/predict/{employee_index}", response_model=PredictionResponse)
def predict_employee(employee_index: int) -> PredictionResponse:
    """
    Get prediction and SHAP factors for a specific employee.
    
    Args:
        employee_index: The index of the employee in the dataset
        
    Returns:
        Prediction with probability and top SHAP factors
    """
    try:
        result = get_employee_prediction(employee_index)
        
        # Convert to ShapFactor objects
        positive_factors = [
            ShapFactor(
                feature=f["feature"],
                value=f["value"],
                display_value=f.get("display_value"),
                shap_value=f["shap_value"]
            )
            for f in result["top_positive_factors"]
        ]
        negative_factors = [
            ShapFactor(
                feature=f["feature"],
                value=f["value"],
                display_value=f.get("display_value"),
                shap_value=f["shap_value"]
            )
            for f in result["top_negative_factors"]
        ]
        
        return PredictionResponse(
            employee_index=result["employee_index"],
            prediction=result["prediction"],
            probability=result["probability"],
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors
        )
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Employee not found at index {employee_index}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.get("/predict/{employee_index}/explain", response_model=ExplanationResponse)
def predict_and_explain(employee_index: int) -> ExplanationResponse:
    """
    Get prediction with SHAP factors and generate LLM explanation.
    
    This endpoint combines prediction and explanation in one call.
    
    Args:
        employee_index: The index of the employee in the dataset
        
    Returns:
        Full explanation with prediction, probability, and LLM-generated text
    """
    try:
        # Get prediction with SHAP values
        result = get_employee_prediction(employee_index)
        
        # Build ExplanationRequest for the prompt
        positive_factors = [
            ShapFactor(
                feature=f["feature"],
                value=f["value"],
                display_value=f.get("display_value"),
                shap_value=f["shap_value"]
            )
            for f in result["top_positive_factors"]
        ]
        negative_factors = [
            ShapFactor(
                feature=f["feature"],
                value=f["value"],
                display_value=f.get("display_value"),
                shap_value=f["shap_value"]
            )
            for f in result["top_negative_factors"]
        ]
        
        explanation_request = ExplanationRequest(
            candidate_id=str(employee_index),
            prediction=result["prediction"],
            probability=result["probability"],
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors
        )
        
        # Generate explanation using Gemini
        prompt = build_prompt(explanation_request)
        return generate_explanation(explanation_request, prompt)
        
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Employee not found at index {employee_index}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction or explanation failed: {str(exc)}",
        ) from exc


@app.post("/explain", response_model=ExplanationResponse)
def explain(data: ExplanationRequest) -> ExplanationResponse:
    try:
        prompt = build_prompt(data)
        return generate_explanation(data, prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(exc)}",
        ) from exc