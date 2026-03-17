from typing import List

from app.schemas import ExplanationRequest, ShapFactor
from app.security import sanitize_input, sanitize_factor_value


def format_factor_list(factors: List[ShapFactor]) -> str:
    if not factors:
        return "- No factors provided"

    return "\n".join(
        [
            (
                f"- feature: {sanitize_factor_value(factor.feature)} | "
                f"value: {sanitize_factor_value(factor.display_value if factor.display_value is not None else factor.value)} | "
                f"shap_value: {factor.shap_value}"
            )
            for factor in factors
        ]
    )


def build_prompt(data: ExplanationRequest) -> str:
    positive = format_factor_list(data.top_positive_factors)
    negative = format_factor_list(data.top_negative_factors)

    # Sanitize input values
    sanitized_id = sanitize_input(data.candidate_id)
    sanitized_pred = sanitize_input(data.prediction)
    sanitized_prob = sanitize_input(data.probability)

    return f"""
You are an HR decision-support assistant.

Your task:
Explain the model output using only the information provided. It provides you the prediction and probability of a worker leaving.

Strict rules:
- Do not invent any justification.
- Do not add assumptions that are not present in the input.
- Use only the prediction, the probability, and the SHAP factors.
- Clearly separate favorable and unfavorable factors.
- Use a neutral, clear, and professional tone.
- Respond in English.
- Return only valid JSON.
- The possible_explanation field should provide a possible reason why the employee might leave based on the factors.
- The JSON must contain exactly these keys:
  summary, possible_explanation, detailed_explanation, main_factors, caution

Input data:
- candidate_id: {sanitized_id}
- prediction: {sanitized_pred}
- probability: {sanitized_prob}

Positive factors:
{positive}

Negative factors:
{negative}

Output requirements:
- summary: 2 to 3 sentences maximum
- possible_explanation: 1-2 sentences explaining why the employee might leave based on the factors
- detailed_explanation: one factual paragraph
- main_factors: list of 3 to 6 bullet points maximum
- caution: one sentence reminding that human validation is still required

Expected JSON format:
{{
  "summary": "string",
  "possible_explanation": "string",
  "detailed_explanation": "string",
  "main_factors": ["string", "string"],
  "caution": "string"
}}
""".strip()