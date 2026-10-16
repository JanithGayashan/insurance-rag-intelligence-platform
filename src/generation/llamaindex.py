from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.generation.configuration import GenerationConfiguration, require_api_key
from src.generation.models import AnswerDraft, PreparedContext, TokenUsage
from src.generation.prompts import build_messages


class GenerationError(RuntimeError):
    """A credential-safe provider failure with no automatic paid fallback."""


@dataclass
class ProviderOutput:
    text: str
    status: str
    response_id: str | None
    refused: bool
    usage: TokenUsage
    model: str | None = None


def response_format() -> dict[str, Any]:
    return {"format": {"type": "json_schema", "name": "insurance_answer",
                       "strict": True, "schema": AnswerDraft.model_json_schema()}}


def usage_from_response(raw: dict[str, Any], model: str) -> TokenUsage:
    value = raw.get("usage") or {}
    usage = TokenUsage(
        input_tokens=value.get("input_tokens"), output_tokens=value.get("output_tokens"),
        cached_input_tokens=(value.get("input_tokens_details") or {}).get("cached_tokens"),
        reasoning_tokens=(value.get("output_tokens_details") or {}).get("reasoning_tokens"),
    )
    # Standard text token rates checked 2026-10-15. This is an estimate, not an
    # invoice. Output tokens already INCLUDE reasoning; never count them twice.
    rates = ((0.05, 0.005, 0.40) if model.startswith("gpt-5-nano") else
             (0.10, 0.01, 0.50) if model.startswith("gpt-6-luna") else None)
    if rates and usage.input_tokens is not None and usage.output_tokens is not None:
        cached = usage.cached_input_tokens or 0
        usage.estimated_cost_usd = ((usage.input_tokens - cached) * rates[0]
                                    + cached * rates[1] + usage.output_tokens * rates[2]) / 1_000_000
        usage.pricing_as_of = "2026-10-15"
    return usage


class LlamaIndexAnswerGenerator:
    """One native LlamaIndex OpenAIResponses call with strict structured output.

    Explicit prepared messages avoid hidden context splitting, refinement calls
    and model-selected retrieval. The existing local retrieval pipeline is reused.
    """

    def __init__(self, configuration: GenerationConfiguration, *, llm=None):
        self.configuration = configuration
        self._owned_client = None
        if llm is None:
            from llama_index.llms.openai import OpenAIResponses
            from openai import OpenAI
            key = require_api_key()
            self._owned_client = OpenAI(
                api_key=key, base_url="https://api.openai.com/v1",
                max_retries=configuration.max_retries, timeout=configuration.timeout_seconds,
            )
            llm = OpenAIResponses(
                model=configuration.model, api_key=key, openai_client=self._owned_client,
                api_base="https://api.openai.com/v1",
                max_output_tokens=configuration.max_output_tokens,
                reasoning_options={"effort": configuration.reasoning_effort},
                # Explicitly pass reasoning too: older integration model-name
                # registries may not yet recognize a newly released model.
                additional_kwargs={"reasoning": {"effort": configuration.reasoning_effort},
                                   "text": response_format()},
                context_window=400000 if configuration.model.startswith("gpt-5-nano") else 1050000,
                max_retries=configuration.max_retries, timeout=configuration.timeout_seconds,
                store=False, track_previous_responses=False,
            )
        self.llm = llm

    def close(self):
        """Release the owned provider client when a long-lived API shuts down."""
        if self._owned_client is not None:
            self._owned_client.close()

    def generate(self, context: PreparedContext) -> ProviderOutput:
        try:
            response = self.llm.chat(build_messages(context))
        except Exception as error:
            # Provider errors can contain request details. Do not log/serialize
            # their body, credentials, client objects or exception chains.
            kind = type(error).__name__
            messages = {
                "AuthenticationError": "OpenAI rejected the API key. Check OPENAI_API_KEY in .env.",
                "RateLimitError": "OpenAI rate limit or billing quota reached. Check the API account before retrying.",
                "NotFoundError": "The configured OpenAI model is unavailable to this API account. Check OPENAI_MODEL.",
                "APITimeoutError": "The OpenAI request timed out. No automatic retry was made.",
            }
            raise GenerationError(messages.get(kind, f"OpenAI request failed ({kind}). No automatic model fallback was made.")) from None
        raw = response.raw
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        if not isinstance(raw, dict):
            raise GenerationError("OpenAI response metadata was missing; answer was not published.")
        refused = any(part.get("type") == "refusal"
                      for item in raw.get("output", []) if item.get("type") == "message"
                      for part in item.get("content", []))
        return ProviderOutput(
            text=response.message.content or "", status=raw.get("status", "unknown"),
            response_id=raw.get("id"), refused=refused,
            usage=usage_from_response(raw, self.configuration.model),
            model=raw.get("model"),
        )
