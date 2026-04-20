import sys
import threading
import types
import unittest
from pathlib import Path
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

from tradingagents.llm_clients.codex_client import (
    CodexAppServerRpcClient,
    CodexChatModel,
    CodexClient,
    _debug_emit,
    _model_loading_debug_emit,
    _resolve_debug_target,
    build_codex_subprocess_env,
    can_use_codex,
)


class FakeRpcClient:
    def __init__(self):
        self.request_handler = None
        self.tool_call_result = None
        self.thread_start_calls = 0
        self.notifications = [
            {
                "method": "item/reasoning/textDelta",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "itemId": "reason-1",
                    "delta": "thinking...",
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [
                            {"type": "agentMessage", "id": "msg-1", "text": "final codex answer"}
                        ],
                    },
                },
            },
        ]

    def set_request_handler(self, handler):
        self.request_handler = handler

    def clear_request_handler(self):
        self.request_handler = None

    def request(self, method, params=None, timeout=60.0):
        if method == "thread/start":
            self.thread_start_calls += 1
            return {"thread": {"id": "thread-1"}}
        if method == "turn/start":
            if self.request_handler is not None:
                self.tool_call_result = self.request_handler(
                    {
                        "id": "srv-1",
                        "method": "item/tool/call",
                        "params": {
                            "tool": "lookup_quote",
                            "arguments": {"symbol": "NVDA"},
                        },
                    }
                )
            return {"turn": {"id": "turn-1"}}
        if method == "model/list":
            return {
                "data": [
                    {"id": "gpt-5.4"},
                    {"model": "gpt-5.3-codex"},
                ]
            }
        raise AssertionError(f"Unexpected method: {method}")

    def next_notification(self, timeout=300.0):
        return self.notifications.pop(0)

    def close(self):
        return None


class FakeTool:
    name = "lookup_quote"
    description = "Lookup stock quote"
    args_schema = None

    def __init__(self):
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return "NVDA quote data"


class TestableCodexChatModel(CodexChatModel):
    def __init__(self, model, rpc_client, callbacks=None, reuse_thread=False):
        super().__init__(model=model, cwd="/tmp", callbacks=callbacks, reuse_thread=reuse_thread)
        self._rpc = rpc_client

    def _get_rpc(self):
        return self._rpc


class TestCodexClient(unittest.TestCase):
    def test_build_codex_subprocess_env_adds_codex_and_git_exec_path(self):
        with unittest.mock.patch.dict(
            "os.environ",
            {"PATH": "/usr/bin:/bin"},
            clear=True,
        ), unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.shutil.which",
            return_value="/opt/codex/bin/codex",
        ), unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.subprocess.check_output",
            return_value="/opt/git/libexec/git-core\n",
        ):
            env = build_codex_subprocess_env("codex")

        self.assertEqual(env["GIT_EXEC_PATH"], "/opt/git/libexec/git-core")
        self.assertEqual(
            env["PATH"].split(":")[:4],
            ["/opt/git/libexec/git-core", "/opt/codex/bin", "/usr/bin", "/bin"],
        )

    def test_can_use_codex_returns_true_when_auth_file_exists(self):
        with unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.has_codex_auth",
            return_value=True,
        ), unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.list_codex_models"
        ) as mock_list:
            self.assertTrue(can_use_codex())
        mock_list.assert_not_called()

    def test_can_use_codex_probes_models_when_auth_file_missing(self):
        with unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.has_codex_auth",
            return_value=False,
        ), unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.list_codex_models",
            return_value=["gpt-5.4"],
        ):
            self.assertTrue(can_use_codex())

    def test_can_use_codex_returns_false_when_probe_fails(self):
        with unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.has_codex_auth",
            return_value=False,
        ), unittest.mock.patch(
            "tradingagents.llm_clients.codex_client.list_codex_models",
            side_effect=RuntimeError("codex unavailable"),
        ):
            self.assertFalse(can_use_codex())

    def test_debug_emit_skips_network_when_debug_env_missing(self):
        _resolve_debug_target.cache_clear()
        with patch("tradingagents.llm_clients.codex_client.Path.exists", return_value=False), patch(
            "tradingagents.llm_clients.codex_client.urllib.request.urlopen"
        ) as mock_urlopen:
            _debug_emit("A", "codex_client.py:test", "skip network")
        mock_urlopen.assert_not_called()

    def test_model_loading_debug_emit_skips_network_when_debug_env_missing(self):
        _resolve_debug_target.cache_clear()
        with patch("tradingagents.llm_clients.codex_client.Path.exists", return_value=False), patch(
            "tradingagents.llm_clients.codex_client.urllib.request.urlopen"
        ) as mock_urlopen:
            _model_loading_debug_emit("skip network")
        mock_urlopen.assert_not_called()

    def test_debug_target_is_cached_when_env_exists(self):
        debug_dir = Path(".dbg")
        debug_dir.mkdir(exist_ok=True)
        env_path = debug_dir / "codex-notification-timeout.env"
        env_path.write_text(
            "DEBUG_SERVER_URL=http://127.0.0.1:9999/event\nDEBUG_SESSION_ID=test-session\n",
            encoding="utf-8",
        )
        self.addCleanup(lambda: env_path.unlink(missing_ok=True))
        _resolve_debug_target.cache_clear()
        first = _resolve_debug_target("codex-notification-timeout.env", "codex-notification-timeout")
        second = _resolve_debug_target("codex-notification-timeout.env", "codex-notification-timeout")
        self.assertEqual(first, ("http://127.0.0.1:9999/event", "test-session"))
        self.assertIs(first, second)

    def test_invoke_reuses_thread_when_enabled_and_signature_matches(self):
        rpc = FakeRpcClient()
        model = TestableCodexChatModel("gpt-5.4", rpc, reuse_thread=True)
        model.invoke("first prompt")
        rpc.notifications = [
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "id": "msg-2", "text": "second"}],
                    },
                },
            }
        ]
        model.invoke("second prompt")
        self.assertEqual(rpc.thread_start_calls, 1)

    def test_invoke_starts_new_thread_when_signature_changes(self):
        rpc = FakeRpcClient()
        model = TestableCodexChatModel("gpt-5.4", rpc, reuse_thread=True)
        model.invoke(
            [{"role": "system", "content": "system-a"}, {"role": "user", "content": "first prompt"}]
        )
        rpc.notifications = [
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "id": "msg-2", "text": "second"}],
                    },
                },
            }
        ]
        model.invoke(
            [{"role": "system", "content": "system-b"}, {"role": "user", "content": "second prompt"}]
        )
        self.assertEqual(rpc.thread_start_calls, 2)

    def test_codex_provider_validates_codex_models(self):
        client = CodexClient("gpt-5.3-codex")
        self.assertTrue(client.validate_model())

    def test_request_fails_fast_when_codex_process_exits_with_missing_dependency(self):
        class ExitedProcess:
            def poll(self):
                return 1

        client = object.__new__(CodexAppServerRpcClient)
        client._lock = threading.Lock()
        client._pending = {}
        client._next_id = 1
        client._process = ExitedProcess()
        client._env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "GIT_EXEC_PATH": "/tmp/git-core",
        }
        client._stderr_tail = [
            "Error: Missing optional dependency @openai/codex-darwin-x64. Reinstall Codex: npm install -g @openai/codex@latest",
        ]
        client._write = lambda message: None

        with self.assertRaises(RuntimeError) as context:
            client.request("initialize", timeout=0.01)

        self.assertIn("exited before responding to initialize", str(context.exception))
        self.assertIn("Reinstall Codex", str(context.exception))

    def test_list_models_reads_app_server_catalog(self):
        model = TestableCodexChatModel("gpt-5.4", FakeRpcClient())
        self.assertEqual(model.list_models(), ["gpt-5.3-codex", "gpt-5.4"])

    def test_invoke_handles_dynamic_tool_calls_inside_codex_turn(self):
        rpc_client = FakeRpcClient()
        model = TestableCodexChatModel("gpt-5.4", rpc_client)
        tool = FakeTool()

        result = model.bind_tools([tool]).invoke([("system", "You are helpful"), ("user", "Analyze NVDA")])

        self.assertEqual(result.content, "final codex answer")
        self.assertEqual(tool.calls, [{"symbol": "NVDA"}])
        self.assertEqual(
            rpc_client.tool_call_result,
            {
                "success": True,
                "contentItems": [{"type": "inputText", "text": "NVDA quote data"}],
            },
        )

    def test_invoke_emits_codex_progress_callbacks(self):
        events = []

        class Callback:
            def on_codex_event(self, event):
                events.append(event)

            def on_chat_model_start(self, serialized, messages, **kwargs):
                events.append({"type": "chat_start", "name": serialized.get("name")})

        rpc_client = FakeRpcClient()
        model = TestableCodexChatModel("gpt-5.4", rpc_client, callbacks=[Callback()])

        result = model.invoke([("user", "Analyze NVDA")])

        self.assertEqual(result.content, "final codex answer")
        self.assertTrue(any(event.get("type") == "chat_start" for event in events))
        self.assertTrue(any(event.get("type") == "reasoning_delta" for event in events))
        self.assertTrue(any(event.get("type") == "status" and "completed" in event.get("text", "") for event in events))

    def test_invoke_accepts_turn_completed_with_nested_turn_id(self):
        rpc_client = FakeRpcClient()
        rpc_client.notifications[1] = {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {
                    "id": "turn-1",
                    "status": "completed",
                    "items": [
                        {"type": "agentMessage", "id": "msg-1", "text": "final codex answer"}
                    ],
                },
            },
        }
        model = TestableCodexChatModel("gpt-5.4", rpc_client)

        result = model.invoke([("user", "Analyze NVDA")])

        self.assertEqual(result.content, "final codex answer")

    def test_invoke_uses_agent_message_stream_when_turn_items_are_empty(self):
        rpc_client = FakeRpcClient()
        rpc_client.notifications = [
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "itemId": "msg-1",
                    "delta": "Hello",
                },
            },
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "itemId": "msg-1",
                    "delta": ".",
                },
            },
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "type": "agentMessage",
                        "id": "msg-1",
                        "text": "Hello.",
                    },
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                    },
                },
            },
        ]
        model = TestableCodexChatModel("gpt-5.4", rpc_client)

        result = model.invoke([("user", "Say hello")])

        self.assertEqual(result.content, "Hello.")


if __name__ == "__main__":
    unittest.main()
