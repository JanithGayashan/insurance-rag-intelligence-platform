from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from src.generation.configuration import PROJECT_ROOT
from src.configuration.dataset import get_dataset_configuration


class APIConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    backend_api_key: SecretStr = Field(repr=False)
    allowed_origins: tuple[str, ...] = ()
    requests_per_minute: int = Field(default=20, ge=1, le=1000)
    requests_per_day: int = Field(default=200, ge=1, le=100000)
    save_artifacts: bool = True
    dataset_id: str = Field(default_factory=lambda: get_dataset_configuration().dataset_id)
    output_directory: Path = Field(default_factory=lambda data: get_dataset_configuration(data["dataset_id"]).api_output_directory)
    qdrant_path: Path = Field(default_factory=lambda data: get_dataset_configuration(data["dataset_id"]).qdrant_path)
    collection: str | None = Field(default_factory=lambda data: get_dataset_configuration(data["dataset_id"]).collection)

    @field_validator("backend_api_key")
    @classmethod
    def strong_backend_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value()
        if len(key) < 32 or not key.isascii() or key.lower().startswith(("your", "replace", "<")):
            raise ValueError("RAG_BACKEND_API_KEY must be a private random ASCII token of at least 32 characters; run python -m src.api.setup")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def explicit_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
                raise ValueError("CORS origins must be explicit http(s) origins without paths or credentials")
        return values

    @classmethod
    def from_environment(cls):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        dataset = get_dataset_configuration()
        return cls(
            backend_api_key=os.getenv("RAG_BACKEND_API_KEY", ""),
            allowed_origins=tuple(value.strip() for value in os.getenv("RAG_ALLOWED_ORIGINS", "").split(",") if value.strip()),
            requests_per_minute=os.getenv("RAG_API_REQUESTS_PER_MINUTE", "20"),
            requests_per_day=os.getenv("RAG_API_REQUESTS_PER_DAY", "200"),
            save_artifacts=os.getenv("RAG_API_SAVE_ARTIFACTS", "true").lower() == "true",
            dataset_id=dataset.dataset_id,
            output_directory=dataset.api_output_directory,
            qdrant_path=dataset.qdrant_path,
            collection=dataset.collection,
        )
