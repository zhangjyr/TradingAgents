import sys
import types
import unittest

langchain_core_callbacks = types.ModuleType("langchain_core.callbacks")
langchain_core_outputs = types.ModuleType("langchain_core.outputs")
langchain_core_messages = types.ModuleType("langchain_core.messages")


class BaseCallbackHandler:
    pass


class LLMResult:
    pass


class AIMessage:
    pass


langchain_core_callbacks.BaseCallbackHandler = BaseCallbackHandler
langchain_core_outputs.LLMResult = LLMResult
langchain_core_messages.AIMessage = AIMessage
sys.modules.setdefault("langchain_core.callbacks", langchain_core_callbacks)
sys.modules.setdefault("langchain_core.outputs", langchain_core_outputs)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages)

from cli.stats_handler import CodexProgressCallbackHandler


class TestCodexProgressCallbackHandler(unittest.TestCase):
    def test_progress_handler_aggregates_reasoning_and_tool_events(self):
        updates = []
        tool_calls = []
        handler = CodexProgressCallbackHandler(
            on_live_update=lambda payload: updates.append(payload.copy()),
            on_tool_call=lambda tool_name, args: tool_calls.append((tool_name, args)),
        )

        handler.on_codex_event({"type": "status", "text": "Starting Codex thread"})
        handler.on_codex_event({"type": "reasoning_delta", "item_id": "r1", "delta": "thinking"})
        handler.on_codex_event({"type": "tool_call", "tool": "lookup_quote", "arguments": {"symbol": "NVDA"}})
        handler.on_codex_event({"type": "stderr", "text": "background log"})

        self.assertTrue(updates)
        latest = updates[-1]
        self.assertEqual(latest["status"], "Starting Codex thread")
        self.assertEqual(latest["reasoning"], "thinking")
        self.assertIn("lookup_quote", latest["tool"])
        self.assertIn("background log", latest["stderr"])
        self.assertEqual(tool_calls, [("lookup_quote", {"symbol": "NVDA"})])


if __name__ == "__main__":
    unittest.main()
