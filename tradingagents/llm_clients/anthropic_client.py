from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
)


def supports_anthropic_effort(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized in {
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "claude-opus-4-6",
    }


def _emit_provider_event(callbacks: Any, event: dict[str, Any]) -> None:
    if not isinstance(callbacks, list):
        return
    for callback in callbacks:
        handler = getattr(callback, "on_provider_event", None)
        if callable(handler):
            try:
                handler(event)
            except Exception:
                pass


def _extract_text_delta(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        callbacks = getattr(self, "callbacks", None)
        _emit_provider_event(
            callbacks,
            {"type": "status", "text": f"Starting Claude response with {self.model}"},
        )
        accumulated: AIMessageChunk | None = None
        try:
            for chunk in self.stream(input, config=config, **kwargs):
                if isinstance(chunk, AIMessageChunk):
                    delta = _extract_text_delta(chunk.content)
                    if delta:
                        _emit_provider_event(
                            callbacks,
                            {
                                "type": "assistant_delta",
                                "item_id": getattr(chunk, "id", None) or "assistant",
                                "delta": delta,
                            },
                        )
                    accumulated = chunk if accumulated is None else accumulated + chunk
            if accumulated is None:
                response = normalize_content(super().invoke(input, config, **kwargs))
            else:
                response = normalize_content(message_chunk_to_message(accumulated))
            if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
                for tool_call in response.tool_calls:
                    if isinstance(tool_call, dict):
                        _emit_provider_event(
                            callbacks,
                            {
                                "type": "tool_call",
                                "tool": tool_call.get("name", ""),
                                "arguments": tool_call.get("args", {}),
                            },
                        )
            _emit_provider_event(
                callbacks,
                {"type": "status", "text": f"Claude response completed for {self.model}"},
            )
            return response
        except Exception as error:
            _emit_provider_event(callbacks, {"type": "error", "text": str(error)})
            raise


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                if key == "effort" and not supports_anthropic_effort(self.model):
                    continue
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
