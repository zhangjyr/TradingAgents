import json
import os
import sys
import tempfile
import types
import unittest
from langchain_core.callbacks import BaseCallbackHandler
from unittest.mock import patch

try:
    import langchain_anthropic  # type: ignore
except Exception:
    langchain_anthropic = types.ModuleType("langchain_anthropic")

    class ChatAnthropic:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.model = kwargs.get("model")
            self.default_headers = kwargs.get("default_headers")
            self.max_retries = kwargs.get("max_retries", 2)
            self.default_request_timeout = kwargs.get("timeout")
            self.anthropic_api_url = kwargs.get("base_url")
            self.anthropic_api_key = kwargs.get("anthropic_api_key", "")

    langchain_anthropic.ChatAnthropic = ChatAnthropic
    sys.modules.setdefault("langchain_anthropic", langchain_anthropic)

try:
    import langchain_openai  # type: ignore
except Exception:
    langchain_openai = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        pass

    langchain_openai.ChatOpenAI = ChatOpenAI
    sys.modules.setdefault("langchain_openai", langchain_openai)

try:
    import langchain_google_genai  # type: ignore
except Exception:
    langchain_google_genai = types.ModuleType("langchain_google_genai")

    class ChatGoogleGenerativeAI:
        pass

    langchain_google_genai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
    sys.modules.setdefault("langchain_google_genai", langchain_google_genai)

try:
    import pydantic  # type: ignore
except Exception:
    pydantic = types.ModuleType("pydantic")

    class SecretStr(str):
        def get_secret_value(self):
            return str(self)

    def Field(default=None, **kwargs):
        return default

    pydantic.SecretStr = SecretStr
    pydantic.Field = Field
    sys.modules.setdefault("pydantic", pydantic)

try:
    import langchain_core.messages  # type: ignore
    from langchain_core.messages import AIMessageChunk  # type: ignore
except Exception:
    langchain_core_messages = types.ModuleType("langchain_core.messages")

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
        def __init__(self, content="", id=None, tool_calls=None):
            super().__init__(content=content)
            self.id = id
            self.tool_calls = tool_calls or []

        def __add__(self, other):
            return AIMessageChunk(
                content=f"{self.content}{other.content}",
                id=self.id or getattr(other, "id", None),
                tool_calls=self.tool_calls or getattr(other, "tool_calls", []),
            )

    class ToolMessage(BaseMessage):
        type = "tool"

        def __init__(self, content="", tool_call_id="tool"):
            super().__init__(content=content)
            self.tool_call_id = tool_call_id

    langchain_core_messages.BaseMessage = BaseMessage
    langchain_core_messages.HumanMessage = HumanMessage
    langchain_core_messages.SystemMessage = SystemMessage
    langchain_core_messages.AIMessage = AIMessage
    langchain_core_messages.AIMessageChunk = AIMessageChunk
    langchain_core_messages.ToolMessage = ToolMessage
    langchain_core_messages.message_chunk_to_message = lambda chunk: chunk
    sys.modules.setdefault("langchain_core.messages", langchain_core_messages)
    AIMessageChunk = langchain_core_messages.AIMessageChunk

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeClient,
    NormalizedClaudeCodeChatAnthropic,
    get_claude_code_oauth_token,
    _persist_refreshed_tokens,
)
from tradingagents.llm_clients.anthropic_client import supports_anthropic_effort


class TestClaudeCodeClient(unittest.TestCase):
    def test_reads_oauth_token_from_claude_credentials_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = os.path.join(tmpdir, ".credentials.json")
            with open(credentials_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "claudeAiOauth": {
                            "accessToken": "file-token",
                            "refreshToken": "refresh-token",
                            "expiresAt": None,
                            "scopes": ["user:inference"],
                        }
                    },
                    handle,
                )

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": tmpdir}, clear=False):
                self.assertEqual(get_claude_code_oauth_token(), "file-token")

    def test_env_token_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "env-token",
                "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "env-refresh",
            },
            clear=False,
        ):
            self.assertEqual(get_claude_code_oauth_token(), "env-token")

    def test_auth_token_is_forwarded_to_anthropic_client(self):
        model = NormalizedClaudeCodeChatAnthropic(
            model="claude-sonnet-4-6",
            auth_token="oauth-token",
            anthropic_api_key="unused",
        )

        params = model._client_params

        self.assertEqual(params["auth_token"], "oauth-token")
        self.assertNotIn("api_key", params)
        self.assertEqual(params["default_headers"]["anthropic-beta"], "oauth-2025-04-20")

    def test_claude_code_client_validates_catalog(self):
        client = ClaudeCodeClient("claude-sonnet-4-6")
        self.assertTrue(client.validate_model())

    def test_supports_anthropic_effort_matches_verified_models(self):
        self.assertFalse(supports_anthropic_effort("claude-haiku-4-5"))
        self.assertFalse(supports_anthropic_effort("claude-sonnet-4-5"))
        self.assertTrue(supports_anthropic_effort("claude-sonnet-4-6"))
        self.assertTrue(supports_anthropic_effort("claude-opus-4-5"))
        self.assertTrue(supports_anthropic_effort("claude-opus-4-6"))

    def test_claude_code_client_omits_effort_for_unsupported_model(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "dummy-token"}, clear=False):
            client = ClaudeCodeClient("claude-sonnet-4-5", effort="high")
            llm = client.get_llm()
        self.assertIsNone(getattr(llm, "effort", None))

    def test_claude_code_client_keeps_effort_for_supported_model(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "dummy-token"}, clear=False):
            client = ClaudeCodeClient("claude-opus-4-6", effort="high")
            llm = client.get_llm()
        self.assertEqual(getattr(llm, "effort"), "high")

    def test_claude_code_invoke_emits_live_status_and_deltas(self):
        events = []

        class Callback(BaseCallbackHandler):
            def on_provider_event(self, event, **kwargs):
                events.append(event)

        model = NormalizedClaudeCodeChatAnthropic(
            model="claude-sonnet-4-6",
            auth_token="oauth-token",
            anthropic_api_key="unused",
            callbacks=[Callback()],
        )
        object.__setattr__(
            model,
            "stream",
            lambda *args, **kwargs: iter(
            [
                AIMessageChunk(content="Hel", id="msg-1"),
                AIMessageChunk(content="lo", id="msg-1"),
            ]
            ),
        )

        result = model.invoke("Say hello")

        self.assertEqual(result.content, "Hello")
        self.assertEqual(events[0]["type"], "status")
        self.assertTrue(any(event.get("type") == "assistant_delta" for event in events))
        self.assertEqual(events[-1]["type"], "status")

    def test_refresh_writes_updated_tokens_back_to_credentials_file(self):
        expires_at = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = os.path.join(tmpdir, ".credentials.json")
            with open(credentials_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "claudeAiOauth": {
                            "accessToken": "old-token",
                            "refreshToken": "refresh-token",
                            "expiresAt": expires_at,
                            "scopes": ["user:inference"],
                            "subscriptionType": "pro",
                        },
                        "otherField": {"preserved": True},
                    },
                    handle,
                )

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "access_token": "new-token",
                        "refresh_token": "new-refresh-token",
                        "expires_in": 3600,
                        "scope": "user:profile user:inference",
                    }

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": tmpdir}, clear=False):
                with patch("tradingagents.llm_clients.claude_code_client.requests.post", return_value=FakeResponse()):
                    token = get_claude_code_oauth_token()

            self.assertEqual(token, "new-token")
            with open(credentials_path, "r", encoding="utf-8") as handle:
                updated = json.load(handle)
            self.assertEqual(updated["claudeAiOauth"]["accessToken"], "new-token")
            self.assertEqual(updated["claudeAiOauth"]["refreshToken"], "new-refresh-token")
            self.assertEqual(updated["claudeAiOauth"]["subscriptionType"], "pro")
            self.assertEqual(updated["otherField"], {"preserved": True})

    def test_persist_refreshed_tokens_uses_keychain_on_macos(self):
        payload = {"claudeAiOauth": {"accessToken": "old-token"}}
        refreshed = {"accessToken": "new-token"}

        with patch("tradingagents.llm_clients.claude_code_client.sys_platform", return_value="darwin"):
            with patch("tradingagents.llm_clients.claude_code_client.subprocess.run") as mock_run:
                ok = _persist_refreshed_tokens("keychain", payload, refreshed)

        self.assertTrue(ok)
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertTrue(kwargs["check"])
        self.assertIn("add-generic-password", kwargs["input"])


if __name__ == "__main__":
    unittest.main()
