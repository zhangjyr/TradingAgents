import sys
import types
import unittest
from unittest.mock import patch

try:
    import langchain_openai  # type: ignore
except Exception:
    langchain_openai = types.ModuleType("langchain_openai")

try:
    import langchain_anthropic  # type: ignore
except Exception:
    langchain_anthropic = types.ModuleType("langchain_anthropic")

try:
    import langchain_google_genai  # type: ignore
except Exception:
    langchain_google_genai = types.ModuleType("langchain_google_genai")

try:
    import langchain_core.messages as langchain_core_messages  # type: ignore
except Exception:
    langchain_core_messages = types.ModuleType("langchain_core.messages")


class ChatOpenAI:
    pass


class ChatAnthropic:
    pass


class ChatGoogleGenerativeAI:
    pass


class BaseMessage:
    def __init__(self, content=""):
        self.content = content


class HumanMessage(BaseMessage):
    type = "human"


class SystemMessage(BaseMessage):
    type = "system"


class AIMessage(BaseMessage):
    type = "ai"


class AIMessageChunk(AIMessage):
    pass


class ToolMessage(BaseMessage):
    type = "tool"

    def __init__(self, content="", tool_call_id="tool"):
        super().__init__(content=content)
        self.tool_call_id = tool_call_id


if not hasattr(langchain_openai, "ChatOpenAI"):
    langchain_openai.ChatOpenAI = ChatOpenAI
if not hasattr(langchain_anthropic, "ChatAnthropic"):
    langchain_anthropic.ChatAnthropic = ChatAnthropic
if not hasattr(langchain_google_genai, "ChatGoogleGenerativeAI"):
    langchain_google_genai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
if not hasattr(langchain_core_messages, "BaseMessage"):
    langchain_core_messages.BaseMessage = BaseMessage
    langchain_core_messages.HumanMessage = HumanMessage
    langchain_core_messages.SystemMessage = SystemMessage
    langchain_core_messages.AIMessage = AIMessage
    langchain_core_messages.AIMessageChunk = AIMessageChunk
    langchain_core_messages.ToolMessage = ToolMessage
    langchain_core_messages.message_chunk_to_message = lambda chunk: chunk
sys.modules.setdefault("langchain_openai", langchain_openai)
sys.modules.setdefault("langchain_anthropic", langchain_anthropic)
sys.modules.setdefault("langchain_google_genai", langchain_google_genai)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages)

from tradingagents.llm_clients.openai_oauth_client import (
    OpenAIOAuthClient,
    list_codex_models,
)


class TestOpenAIOAuthClient(unittest.TestCase):
    @patch("tradingagents.llm_clients.openai_oauth_client.list_codex_app_server_models")
    def test_list_codex_models_delegates_to_codex_app_server(self, mock_list):
        mock_list.return_value = ["gpt-5.3-codex", "gpt-5.4"]

        models = list_codex_models()

        self.assertEqual(models, ["gpt-5.3-codex", "gpt-5.4"])
        mock_list.assert_called_once_with()

    def test_codex_provider_validates_codex_models(self):
        client = OpenAIOAuthClient("gpt-5.3-codex")
        self.assertTrue(client.validate_model())


if __name__ == "__main__":
    unittest.main()
