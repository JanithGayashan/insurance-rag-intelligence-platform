from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.configuration.dataset import PROJECT_ROOT


DEFAULT_MODEL = "gpt-5-nano"


class GenerationConfiguration(BaseModel):
    """Public generation settings. Credentials never enter serialized configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    reasoning_effort: str = "minimal"
    context_token_budget: int = Field(default=6000, ge=256, le=24000)
    max_output_tokens: int = Field(default=1800, ge=256, le=8192)
    max_question_tokens: int = Field(default=512, ge=1, le=2048)
    tokenizer_encoding: str = "o200k_base"
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    # One request, no automatic model changes, repairs or paid evaluator calls.
    max_retries: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_reasoning(self):
        if self.model.startswith("gpt-5-nano"):
            supported = {"minimal", "low", "medium", "high"}
        elif self.model.startswith("gpt-6-luna"):
            supported = {"none", "low", "medium", "high", "xhigh", "max"}
        else:
            raise ValueError("supported generation models are gpt-5-nano and gpt-6-luna")
        if self.reasoning_effort not in supported:
            raise ValueError("reasoning effort is not supported by the selected model")
        return self

    @classmethod
    def from_environment(cls, env_path: Path | None = None, **overrides):
        load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)
        model = (overrides.get("model") or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)).strip()
        values = {
            "model": model,
            "reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", "none" if model.startswith("gpt-6-luna") else "minimal"),
            "context_token_budget": os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "6000"),
            "max_output_tokens": os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1800"),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.lower().startswith(("your", "<", "sk-your", "sk-placeholder")):
        raise ValueError("Add OPENAI_API_KEY to the project .env file, or use --dry-run without an API key.")
    return key
