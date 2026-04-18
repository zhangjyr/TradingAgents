import sys
import types
import unittest

langchain_openai = types.ModuleType("langchain_openai")
langchain_anthropic = types.ModuleType("langchain_anthropic")
langchain_google_genai = types.ModuleType("langchain_google_genai")
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


class ToolMessage(BaseMessage):
    type = "tool"

    def __init__(self, content="", tool_call_id="tool"):
        super().__init__(content=content)
        self.tool_call_id = tool_call_id


langchain_openai.ChatOpenAI = ChatOpenAI
langchain_anthropic.ChatAnthropic = ChatAnthropic
langchain_google_genai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
langchain_core_messages.BaseMessage = BaseMessage
langchain_core_messages.HumanMessage = HumanMessage
langchain_core_messages.SystemMessage = SystemMessage
langchain_core_messages.AIMessage = AIMessage
langchain_core_messages.ToolMessage = ToolMessage
sys.modules.setdefault("langchain_openai", langchain_openai)
sys.modules.setdefault("langchain_anthropic", langchain_anthropic)
sys.modules.setdefault("langchain_google_genai", langchain_google_genai)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages)

from tradingagents.llm_clients.codex_client import (
    CodexChatModel,
    CodexClient,
    build_codex_subprocess_env,
    can_use_codex,
)


class FakeRpcClient:
    def __init__(self):
        self.request_handler = None
        self.tool_call_result = None
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
    def __init__(self, model, rpc_client, callbacks=None):
        super().__init__(model=model, cwd="/tmp", callbacks=callbacks)
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

    def test_codex_provider_validates_codex_models(self):
        client = CodexClient("gpt-5.3-codex")
        self.assertTrue(client.validate_model())

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
