"""
Security utilities for prompt injection protection.

This module provides input sanitization and output validation
to protect against prompt injection attacks.
"""

import json
import re
from typing import Any, List, Optional


# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions?|orders?|rules?)",
    r"forget\s+(everything|all|your)\s+(instructions?|training|rules?)",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"<\|system\|>",
    r"<\|assistant\|>",
    r"you\s+are\s+(now|a|an)\s+(different|new|)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"role\s*play",
    r"new\s+instructions",
    r"override",
    r"bypass",
    r"hack",
    r"jailbreak",
    r"dan\s+mode",
    r"dev\s*mode",
    r"sudo\s+mode",
]


def sanitize_input(value: Any, max_length: int = 200) -> str:
    """
    Sanitize a value to prevent prompt injection.
    
    Args:
        value: The value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if value is None:
        return ""
    
    # Convert to string
    str_value = str(value)
    
    # Truncate to max length
    if len(str_value) > max_length:
        str_value = str_value[:max_length] + "..."
    
    # Remove or escape potentially dangerous characters
    # Escape curly braces to prevent format string attacks
    str_value = str_value.replace("{", "\\{").replace("}", "\\}")
    
    # Escape quotes to prevent breaking out of strings
    str_value = str_value.replace('"', '\\"').replace("'", "\\'")
    
    # Remove null bytes
    str_value = str_value.replace("\x00", "")
    
    return str_value


def sanitize_factor_value(value: Any) -> str:
    """
    Sanitize feature values from SHAP factors for prompt inclusion.
    
    Args:
        value: The feature value to sanitize
        
    Returns:
        Sanitized string representation
    """
    return sanitize_input(value, max_length=100)


def detect_injection(text: str) -> bool:
    """
    Detect potential prompt injection in text.
    
    Args:
        text: The text to check
        
    Returns:
        True if potential injection detected
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def validate_output(text: str, max_length: int = 2000) -> Optional[str]:
    """
    Validate LLM output for potential injections.
    
    Args:
        text: The LLM output to validate
        max_length: Maximum allowed length
        
    Returns:
        Validated text or None if injection detected
    """
    if not text:
        return None
    
    # Check length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Check for injection patterns
    if detect_injection(text):
        # Log the detection (in production, you'd want proper logging)
        print(f"WARNING: Potential prompt injection detected: {text[:100]}...")
        return None
    
    # Try to parse as JSON (we expect JSON output)
    try:
        parsed = json.loads(text)
        # Verify it's a dict with expected keys
        if not isinstance(parsed, dict):
            return None
        return text
    except json.JSONDecodeError:
        # If not valid JSON, still return but log warning
        print(f"WARNING: LLM output is not valid JSON")
        return text


def sanitize_prediction_data(data: dict) -> dict:
    """
    Sanitize all prediction data before building prompt.
    
    Args:
        data: The prediction data dictionary
        
    Returns:
        Sanitized copy of the data
    """
    sanitized = data.copy()
    
    # Sanitize top positive factors
    if "top_positive_factors" in sanitized:
        sanitized["top_positive_factors"] = [
            {
                **factor,
                "feature": sanitize_input(factor.get("feature", "")),
                "display_value": sanitize_factor_value(factor.get("display_value")),
            }
            for factor in sanitized["top_positive_factors"]
        ]
    
    # Sanitize top negative factors
    if "top_negative_factors" in sanitized:
        sanitized["top_negative_factors"] = [
            {
                **factor,
                "feature": sanitize_input(factor.get("feature", "")),
                "display_value": sanitize_factor_value(factor.get("display_value")),
            }
            for factor in sanitized["top_negative_factors"]
        ]
    
    return sanitized
