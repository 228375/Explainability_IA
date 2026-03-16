import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


class Settings(BaseModel):
    app_name: str = "RH AI Explainer"
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"
    ollama_temperature: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
    )
