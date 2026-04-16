import threading
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }


class CodexProgressCallbackHandler(BaseCallbackHandler):
    def __init__(self, on_live_update, on_message=None, on_tool_call=None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._on_live_update = on_live_update
        self._on_message = on_message
        self._on_tool_call = on_tool_call
        self._status = ""
        self._reasoning: Dict[str, str] = {}
        self._plan: Dict[str, str] = {}
        self._assistant: Dict[str, str] = {}
        self._stderr: List[str] = []
        self._tool_status = ""

    def on_codex_event(self, event: Dict[str, Any], **kwargs: Any) -> None:
        with self._lock:
            event_type = str(event.get("type", "")).strip()
            if event_type == "status":
                self._status = str(event.get("text", "")).strip()
            elif event_type == "reasoning_delta":
                self._append_delta(self._reasoning, event)
            elif event_type == "plan_delta":
                self._append_delta(self._plan, event)
            elif event_type == "assistant_delta":
                self._append_delta(self._assistant, event)
            elif event_type == "stderr":
                text = str(event.get("text", "")).strip()
                if text:
                    self._stderr.append(text)
                    self._stderr = self._stderr[-8:]
            elif event_type == "tool_call":
                tool_name = str(event.get("tool", "")).strip()
                args = event.get("arguments", {})
                self._tool_status = f"{tool_name}({args})" if tool_name else self._tool_status
                if self._on_tool_call and tool_name:
                    self._on_tool_call(tool_name, args)
            elif event_type == "tool_result":
                tool_name = str(event.get("tool", "")).strip()
                outcome = str(event.get("status", "")).strip() or "completed"
                if tool_name:
                    self._tool_status = f"{tool_name}: {outcome}"
            elif event_type == "error":
                text = str(event.get("text", "")).strip()
                if text:
                    self._status = f"error: {text}"
                    if self._on_message:
                        self._on_message("Codex", text)
            self._publish()

    def _append_delta(self, bucket: Dict[str, str], event: Dict[str, Any]) -> None:
        item_id = str(event.get("item_id", "default")).strip() or "default"
        delta = str(event.get("delta", ""))
        if not delta:
            return
        bucket[item_id] = (bucket.get(item_id, "") + delta)[-4000:]

    def _publish(self) -> None:
        payload = {
            "status": self._status,
            "reasoning": self._last_value(self._reasoning),
            "plan": self._last_value(self._plan),
            "assistant": self._last_value(self._assistant),
            "tool": self._tool_status,
            "stderr": "\n".join(self._stderr[-4:]),
        }
        self._on_live_update(payload)

    def _last_value(self, bucket: Dict[str, str]) -> str:
        if not bucket:
            return ""
        last_key = next(reversed(bucket))
        return bucket[last_key][-1200:]
