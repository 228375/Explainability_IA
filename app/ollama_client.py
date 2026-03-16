import json

import httpx

from app.schemas import ExplanationRequest, ExplanationResponse
from app.schemas import LLMExplanationPayload
from app.settings import get_settings


settings = get_settings()


def generate_explanation(
    data: ExplanationRequest,
    prompt: str,
) -> ExplanationResponse:
    """
    Generate explanation using Ollama API.
    """
    url = f"{settings.ollama_base_url}/api/generate"

    body = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "temperature": settings.ollama_temperature,
        "stream": False,
        "format": "json",
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=body)

    response.raise_for_status()
    raw_payload = response.json()
    
    text = raw_payload.get("response", "")
    if not text:
        raise RuntimeError(f"Ollama returned empty response: {raw_payload}")

    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned invalid JSON: {text}"
        ) from exc

    llm_payload = LLMExplanationPayload.model_validate(parsed_json)

    return ExplanationResponse(
        candidate_id=data.candidate_id,
        prediction=data.prediction,
        probability=data.probability,
        summary=llm_payload.summary,
        possible_explanation=getattr(llm_payload, 'possible_explanation', ''),
        detailed_explanation=llm_payload.detailed_explanation,
        main_factors=llm_payload.main_factors,
        caution=llm_payload.caution,
    )
