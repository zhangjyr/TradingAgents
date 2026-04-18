import atexit
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .base_client import BaseLLMClient
from .validators import validate_model

DEFAULT_CODEX_COMMAND = "codex"
DEFAULT_CODEX_ARGS = ("app-server", "--listen", "stdio://")
DEFAULT_CODEX_MODELS = [
    ("GPT-5.4 - Flagship frontier, coding + reasoning", "gpt-5.4"),
    ("GPT-5.4 Mini - Fast, responsive coding", "gpt-5.4-mini"),
    ("GPT-5.3 Codex - Advanced coding", "gpt-5.3-codex"),
    ("GPT-5.3 Codex Spark - Near-instant (research preview)", "gpt-5.3-codex-spark"),
    ("GPT-5.2 Codex - Legacy, succeeded by 5.3 Codex", "gpt-5.2-codex"),
    ("GPT-5.1 Codex Max - Long-horizon agentic coding", "gpt-5.1-codex-max"),
    ("GPT-5.1 Codex - Agentic coding", "gpt-5.1-codex"),
    ("GPT-5 Codex - Tuned for agentic coding", "gpt-5-codex"),
]


# #region debug-point A:debug-report
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: Optional[dict[str, Any]] = None) -> None:
    env_path = Path(".dbg/codex-notification-timeout.env")
    debug_url = "http://127.0.0.1:7777/event"
    session_id = "codex-notification-timeout"
    try:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_url = line.split("=", 1)[1].strip() or debug_url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
            "ts": int(time.time() * 1000),
        }
        request = urllib.request.Request(
            debug_url,
            data=json.dumps(payload, default=str).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=1).read()
    except Exception:
        pass
# #endregion


#region debug-point codex-model-loading
def _model_loading_debug_emit(msg: str, data: Optional[dict[str, Any]] = None) -> None:
    env_path = Path(".dbg/codex-model-loading.env")
    debug_url = "http://127.0.0.1:7777/event"
    session_id = "codex-model-loading"
    try:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_url = line.split("=", 1)[1].strip() or debug_url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": "A",
            "location": "codex_client.py",
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
            "ts": int(time.time() * 1000),
        }
        request = urllib.request.Request(
            debug_url,
            data=json.dumps(payload, default=str).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=1).read()
    except Exception:
        pass
#endregion


def get_codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        auth_path = Path(codex_home).expanduser() / "auth.json"
        _model_loading_debug_emit(
            "resolved codex auth path from CODEX_HOME",
            {"codex_home": codex_home, "auth_path": str(auth_path)},
        )
        return auth_path
    auth_path = Path.home() / ".codex" / "auth.json"
    _model_loading_debug_emit(
        "resolved codex auth path from home directory",
        {"auth_path": str(auth_path)},
    )
    return auth_path


def has_codex_auth() -> bool:
    auth_path = get_codex_auth_path()
    exists = auth_path.exists()
    _model_loading_debug_emit(
        "checked codex auth existence",
        {"auth_path": str(auth_path), "exists": exists},
    )
    return exists


def can_use_codex(cwd: Optional[str] = None) -> bool:
    if has_codex_auth():
        _model_loading_debug_emit(
            "codex available via auth file",
            {"cwd": cwd},
        )
        return True
    try:
        models = list_codex_models(cwd=cwd)
        available = len(models) > 0
        _model_loading_debug_emit(
            "codex availability probe via model listing",
            {"cwd": cwd, "available": available, "models": models},
        )
        return available
    except Exception as error:
        _model_loading_debug_emit(
            "codex availability probe failed",
            {"cwd": cwd, "error_type": type(error).__name__, "error": str(error)},
        )
        return False


def build_codex_subprocess_env(codex_command: str) -> dict[str, str]:
    env = os.environ.copy()
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    codex_path = shutil.which(codex_command)
    if codex_path:
        codex_dir = str(Path(codex_path).resolve().parent)
        if codex_dir not in path_entries:
            path_entries.insert(0, codex_dir)
    git_exec_path = env.get("GIT_EXEC_PATH")
    if not git_exec_path:
        try:
            git_exec_path = subprocess.check_output(
                ["git", "--exec-path"],
                text=True,
            ).strip()
        except Exception:
            git_exec_path = ""
    if git_exec_path:
        env["GIT_EXEC_PATH"] = git_exec_path
        if git_exec_path not in path_entries:
            path_entries.insert(0, git_exec_path)
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    _model_loading_debug_emit(
        "prepared codex subprocess environment",
        {
            "codex_command": codex_command,
            "codex_path": codex_path,
            "git_exec_path": git_exec_path or None,
            "path_prefix": path_entries[:6],
        },
    )
    return env


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
    return str(value)


def _coerce_message(role: str, content: Any) -> BaseMessage:
    if role == "system":
        return SystemMessage(content=_extract_text(content))
    if role == "tool":
        return ToolMessage(content=_extract_text(content), tool_call_id="tool")
    if role == "assistant":
        return AIMessage(content=_extract_text(content))
    return HumanMessage(content=_extract_text(content))


def _coerce_messages(input_value: Any) -> list[BaseMessage]:
    if hasattr(input_value, "to_messages"):
        return list(input_value.to_messages())
    if isinstance(input_value, BaseMessage):
        return [input_value]
    if isinstance(input_value, str):
        return [HumanMessage(content=input_value)]
    if isinstance(input_value, list):
        messages: list[BaseMessage] = []
        for item in input_value:
            if isinstance(item, BaseMessage):
                messages.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                messages.append(_coerce_message(str(item[0]), item[1]))
            elif isinstance(item, dict):
                role = str(item.get("role", "user"))
                messages.append(_coerce_message(role, item.get("content")))
            else:
                messages.append(HumanMessage(content=_extract_text(item)))
        return messages
    return [HumanMessage(content=_extract_text(input_value))]


def _format_messages_as_prompt(messages: list[BaseMessage]) -> tuple[str, str]:
    developer_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        role = getattr(message, "type", "user")
        content = _extract_text(getattr(message, "content", ""))
        if not content.strip():
            continue
        if role == "system":
            developer_parts.append(content.strip())
            continue
        label = {
            "human": "User",
            "ai": "Assistant",
            "tool": "Tool",
        }.get(role, role.capitalize())
        user_parts.append(f"{label}:\n{content.strip()}")
    return "\n\n".join(developer_parts).strip(), "\n\n".join(user_parts).strip()


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema_source = getattr(tool, "args_schema", None)
    if schema_source is None and hasattr(tool, "get_input_schema"):
        try:
            schema_source = tool.get_input_schema()
        except Exception:
            schema_source = None
    if schema_source is not None:
        if hasattr(schema_source, "model_json_schema"):
            try:
                return schema_source.model_json_schema()
            except Exception:
                pass
        if hasattr(schema_source, "schema"):
            try:
                return schema_source.schema()
            except Exception:
                pass
    return {"type": "object", "properties": {}}


def _serialize_tool_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list, tuple)):
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            return str(result)
    return str(result)


def _read_turn_text(turn: dict[str, Any]) -> str:
    items = turn.get("items")
    if not isinstance(items, list):
        return ""
    texts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "agentMessage":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    if not texts:
        return ""
    return texts[-1]


def _notification_turn_id(params: dict[str, Any]) -> Optional[str]:
    turn_id = params.get("turnId")
    if isinstance(turn_id, str) and turn_id.strip():
        return turn_id
    turn = params.get("turn")
    if isinstance(turn, dict):
        nested_turn_id = turn.get("id")
        if isinstance(nested_turn_id, str) and nested_turn_id.strip():
            return nested_turn_id
    return None


class CodexAppServerRpcError(RuntimeError):
    pass


class CodexAppServerRpcClient:
    def __init__(
        self,
        command: str = DEFAULT_CODEX_COMMAND,
        args: tuple[str, ...] = DEFAULT_CODEX_ARGS,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        on_stderr=None,
    ):
        if shutil.which(command) is None:
            raise RuntimeError(f"Codex CLI not found: {command}. Install Codex and run `codex login`.")
        self._env = dict(env or os.environ.copy())
        self._stderr_tail: list[str] = []
        self._process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=cwd,
            env=self._env,
        )
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._notifications: queue.Queue = queue.Queue()
        self._request_handler = None
        self._on_stderr = on_stderr
        self._next_id = 1
        self._closed = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def _timeout_context(self) -> dict[str, Any]:
        env_path = self._env.get("PATH", "")
        stderr_tail = self._stderr_tail[-8:]
        return {
            "returncode": self._process.poll(),
            "node_path": shutil.which("node", path=env_path),
            "git_path": shutil.which("git", path=env_path),
            "git_exec_path": self._env.get("GIT_EXEC_PATH"),
            "stderr_tail": stderr_tail,
            "path_prefix": env_path.split(os.pathsep)[:8] if env_path else [],
        }

    def _build_process_exit_error(self, method: str) -> RuntimeError:
        timeout_context = self._timeout_context()
        stderr_text = " ".join(timeout_context["stderr_tail"])
        if "Missing optional dependency" in stderr_text:
            return RuntimeError(
                f"Codex app-server exited before responding to {method}; "
                f"returncode={timeout_context['returncode']}; "
                "Codex installation is missing an optional platform dependency. "
                "Reinstall Codex with `npm install -g @openai/codex@latest`; "
                f"stderr_tail={timeout_context['stderr_tail']}"
            )
        return RuntimeError(
            f"Codex app-server exited before responding to {method}; "
            f"returncode={timeout_context['returncode']}; "
            f"node={timeout_context['node_path']}; "
            f"git={timeout_context['git_path']}; "
            f"GIT_EXEC_PATH={timeout_context['git_exec_path']}; "
            f"stderr_tail={timeout_context['stderr_tail']}"
        )

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "tradingagents",
                    "title": "TradingAgents",
                    "version": "0.2.3",
                },
                "capabilities": {
                    "experimentalApi": True,
                },
            },
        )
        self.notify("initialized")

    def set_request_handler(self, handler) -> None:
        self._request_handler = handler

    def clear_request_handler(self) -> None:
        self._request_handler = None

    def request(self, method: str, params: Optional[dict[str, Any]] = None, timeout: float = 60.0) -> Any:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            response_queue: queue.Queue = queue.Queue(maxsize=1)
            self._pending[request_id] = response_queue
            self._write({"id": request_id, "method": method, "params": params})
        # #region debug-point A:request-sent
        _debug_emit("A", "codex_client.py:request", "rpc request sent", {"id": request_id, "method": method})
        # #endregion
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._pending.pop(request_id, None)
                timeout_context = self._timeout_context()
                # #region debug-point B:request-timeout
                _debug_emit("B", "codex_client.py:request", "rpc request timed out", {"id": request_id, "method": method, "timeout": timeout, **timeout_context})
                # #endregion
                raise RuntimeError(
                    f"Codex app-server request timed out: {method}; "
                    f"returncode={timeout_context['returncode']}; "
                    f"node={timeout_context['node_path']}; "
                    f"git={timeout_context['git_path']}; "
                    f"GIT_EXEC_PATH={timeout_context['git_exec_path']}; "
                    f"stderr_tail={timeout_context['stderr_tail']}"
                )
            try:
                response = response_queue.get(timeout=min(0.25, remaining))
                break
            except queue.Empty:
                if self._process.poll() is not None:
                    self._pending.pop(request_id, None)
                    process_error = self._build_process_exit_error(method)
                    _debug_emit(
                        "C",
                        "codex_client.py:request",
                        "codex process exited before rpc response",
                        {"id": request_id, "method": method, "error": str(process_error)},
                    )
                    raise process_error
        if "error" in response:
            message = response["error"].get("message", f"{method} failed")
            # #region debug-point C:request-error
            _debug_emit("C", "codex_client.py:request", "rpc response returned error", {"id": request_id, "method": method, "error": message})
            # #endregion
            raise CodexAppServerRpcError(message)
        # #region debug-point A:request-response
        _debug_emit("A", "codex_client.py:request", "rpc response received", {"id": request_id, "method": method})
        # #endregion
        return response.get("result")

    def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        self._write({"method": method, "params": params})

    def next_notification(self, timeout: float = 60.0) -> dict[str, Any]:
        try:
            notification = self._notifications.get(timeout=timeout)
            # #region debug-point A:notification-received
            _debug_emit("A", "codex_client.py:next_notification", "notification dequeued", {"method": notification.get("method") if isinstance(notification, dict) else type(notification).__name__})
            # #endregion
            return notification
        except queue.Empty as error:
            # #region debug-point B:notification-timeout
            _debug_emit("B", "codex_client.py:next_notification", "notification wait timed out", {"timeout": timeout, "pending_requests": list(self._pending.keys())})
            # #endregion
            raise RuntimeError("Timed out waiting for Codex app-server notification") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
        except Exception:
            pass
        try:
            self._process.wait(timeout=2)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass

    def _write(self, message: dict[str, Any]) -> None:
        if not self._process.stdin:
            raise RuntimeError("Codex app-server stdin is not available")
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _read_stdout(self) -> None:
        if not self._process.stdout:
            return
        for line in self._process.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except Exception:
                continue
            if not isinstance(message, dict):
                continue
            if "id" in message and "method" not in message:
                pending = self._pending.pop(message["id"], None)
                if pending is not None:
                    pending.put(message)
                # #region debug-point A:stdout-response
                _debug_emit("A", "codex_client.py:_read_stdout", "stdout rpc response parsed", {"id": message.get("id")})
                # #endregion
                continue
            if "id" in message and "method" in message:
                # #region debug-point D:server-request
                _debug_emit("D", "codex_client.py:_read_stdout", "server initiated request received", {"id": message.get("id"), "method": message.get("method")})
                # #endregion
                self._handle_server_request(message)
                continue
            if "method" in message:
                self._notifications.put(message)
                # #region debug-point A:stdout-notification
                _debug_emit("A", "codex_client.py:_read_stdout", "stdout notification queued", {"method": message.get("method")})
                # #endregion

    def _read_stderr(self) -> None:
        if not self._process.stderr:
            return
        for line in self._process.stderr:
            if self._closed:
                break
            if self._on_stderr is not None:
                text = line.strip()
                if text:
                    self._stderr_tail.append(text)
                    if len(self._stderr_tail) > 20:
                        self._stderr_tail = self._stderr_tail[-20:]
                    self._on_stderr(text)
                    # #region debug-point C:stderr-line
                    _debug_emit("C", "codex_client.py:_read_stderr", "stderr line received", {"text": text[:500]})
                    # #endregion

    def _handle_server_request(self, message: dict[str, Any]) -> None:
        result: Any = None
        if self._request_handler is not None:
            result = self._request_handler(message)
        # #region debug-point D:server-request-result
        _debug_emit("D", "codex_client.py:_handle_server_request", "server request handled", {"method": message.get("method"), "id": message.get("id")})
        # #endregion
        self._write({"id": message["id"], "result": result})


class CodexToolsBinding:
    def __init__(self, model: "CodexChatModel", tools: list[Any]):
        self._model = model
        self._tools = tools

    def __call__(self, input_value: Any, config: Any = None) -> AIMessage:
        return self.invoke(input_value, config=config)

    def invoke(self, input_value: Any, config: Any = None) -> AIMessage:
        return self._model.invoke(input_value, tools=self._tools)


class CodexChatModel:
    def __init__(
        self,
        model: str,
        cwd: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        codex_command: str = DEFAULT_CODEX_COMMAND,
        codex_args: tuple[str, ...] = DEFAULT_CODEX_ARGS,
        callbacks: Optional[list[Any]] = None,
    ):
        self.model = model
        self.cwd = cwd
        self.reasoning_effort = reasoning_effort
        self.codex_command = codex_command
        self.codex_args = codex_args
        self.callbacks = list(callbacks or [])
        self._rpc: Optional[CodexAppServerRpcClient] = None
        self._invoke_lock = threading.Lock()
        self._assistant_deltas: dict[str, str] = {}
        atexit.register(self.close)

    def bind_tools(self, tools: list[Any]) -> CodexToolsBinding:
        return CodexToolsBinding(self, tools)

    def list_models(self, timeout: float = 10.0) -> list[str]:
        rpc = self._get_rpc()
        payload = rpc.request(
            "model/list",
            {
                "limit": None,
                "cursor": None,
                "includeHidden": False,
            },
            timeout=timeout,
        )
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        models = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id") or entry.get("model")
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
        return sorted(dict.fromkeys(models))

    def invoke(self, input_value: Any, tools: Optional[list[Any]] = None) -> AIMessage:
        messages = _coerce_messages(input_value)
        developer_instructions, user_prompt = _format_messages_as_prompt(messages)
        rpc = self._get_rpc()
        self._emit_chat_model_start(messages)
        with self._invoke_lock:
            self._assistant_deltas = {}
            tool_map = {getattr(tool, "name", getattr(tool, "__name__", "tool")): tool for tool in (tools or [])}
            rpc.set_request_handler(lambda message: self._handle_server_request(message, tool_map))
            try:
                self._emit_codex_event({"type": "status", "text": f"Starting Codex thread with model {self.model}"})
                thread_result = rpc.request(
                    "thread/start",
                    {
                        "model": self.model,
                        "modelProvider": "openai",
                        "cwd": self.cwd,
                        "approvalPolicy": "never",
                        "sandbox": "workspace-write",
                        "developerInstructions": developer_instructions or None,
                        "dynamicTools": self._dynamic_tools(tool_map),
                        "experimentalRawEvents": True,
                        "persistExtendedHistory": False,
                    },
                )
                thread_id = self._extract_thread_id(thread_result)
                self._emit_codex_event({"type": "status", "text": f"Codex thread ready: {thread_id}"})
                turn_result = rpc.request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": user_prompt or developer_instructions or ""}],
                        "cwd": self.cwd,
                        "model": self.model,
                        "effort": self.reasoning_effort,
                    },
                )
                turn_id = self._extract_turn_id(turn_result)
                # #region debug-point B:turn-started
                _debug_emit("B", "codex_client.py:invoke", "turn started; waiting for completion", {"thread_id": thread_id, "turn_id": turn_id, "tool_count": len(tool_map)})
                # #endregion
                self._emit_codex_event({"type": "status", "text": f"Waiting for Codex turn {turn_id} to complete"})
                turn = self._wait_for_completion(rpc, thread_id, turn_id)
                content = _read_turn_text(turn)
                if not content:
                    content = self._latest_assistant_text()
                return AIMessage(content=content)
            finally:
                rpc.clear_request_handler()

    def close(self) -> None:
        if self._rpc is not None:
            self._rpc.close()
            self._rpc = None

    def _get_rpc(self) -> CodexAppServerRpcClient:
        if self._rpc is None:
            self._rpc = CodexAppServerRpcClient(
                command=self.codex_command,
                args=self.codex_args,
                cwd=self.cwd,
                env=build_codex_subprocess_env(self.codex_command),
                on_stderr=lambda text: self._emit_codex_event({"type": "stderr", "text": text}),
            )
            self._rpc.initialize()
        return self._rpc

    def _dynamic_tools(self, tool_map: dict[str, Any]) -> list[dict[str, Any]]:
        specs = []
        for name, tool in tool_map.items():
            specs.append(
                {
                    "name": name,
                    "description": getattr(tool, "description", "") or "",
                    "inputSchema": _tool_schema(tool),
                }
            )
        return specs

    def _handle_server_request(self, message: dict[str, Any], tool_map: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") != "item/tool/call":
            return {"success": False, "contentItems": [{"type": "inputText", "text": ""}]}
        params = message.get("params")
        if not isinstance(params, dict):
            return {"success": False, "contentItems": [{"type": "inputText", "text": "Invalid tool call"}]}
        tool_name = params.get("tool")
        tool = tool_map.get(tool_name)
        if tool is None:
            return {
                "success": False,
                "contentItems": [{"type": "inputText", "text": f"Tool not found: {tool_name}"}],
            }
        arguments = params.get("arguments")
        try:
            self._emit_codex_event(
                {
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": arguments if isinstance(arguments, dict) else {"value": arguments},
                }
            )
            if hasattr(tool, "invoke"):
                result = tool.invoke(arguments if arguments is not None else {})
            elif isinstance(arguments, dict):
                result = tool(**arguments)
            elif arguments is None:
                result = tool()
            else:
                result = tool(arguments)
            self._emit_codex_event({"type": "tool_result", "tool": tool_name, "status": "completed"})
            return {
                "success": True,
                "contentItems": [{"type": "inputText", "text": _serialize_tool_result(result)}],
            }
        except Exception as error:
            self._emit_codex_event({"type": "tool_result", "tool": tool_name, "status": f"error: {error}"})
            return {
                "success": False,
                "contentItems": [{"type": "inputText", "text": f"{type(error).__name__}: {error}"}],
            }

    def _extract_thread_id(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid thread/start response from Codex app-server")
        thread = payload.get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise RuntimeError("Codex app-server did not return a thread id")
        return thread["id"]

    def _extract_turn_id(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid turn/start response from Codex app-server")
        turn = payload.get("turn")
        if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
            raise RuntimeError("Codex app-server did not return a turn id")
        return turn["id"]

    def _wait_for_completion(
        self,
        rpc: CodexAppServerRpcClient,
        thread_id: str,
        turn_id: str,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        while True:
            notification = rpc.next_notification(timeout=timeout)
            self._handle_notification(notification, thread_id, turn_id)
            if notification.get("method") != "turn/completed":
                continue
            params = notification.get("params")
            if not isinstance(params, dict):
                continue
            if params.get("threadId") != thread_id:
                continue
            if _notification_turn_id(params) != turn_id:
                continue
            turn = params.get("turn")
            if not isinstance(turn, dict):
                raise RuntimeError("Codex app-server completed turn without turn payload")
            status = turn.get("status")
            # #region debug-point B:turn-completed
            _debug_emit("B", "codex_client.py:_wait_for_completion", "turn completed notification matched", {"thread_id": thread_id, "turn_id": turn_id, "status": status})
            # #endregion
            if status == "failed":
                error = turn.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message.strip():
                        raise RuntimeError(message.strip())
                raise RuntimeError("Codex app-server turn failed")
            return turn

    def _handle_notification(self, notification: dict[str, Any], thread_id: str, turn_id: str) -> None:
        params = notification.get("params")
        if not isinstance(params, dict):
            return
        notification_thread_id = params.get("threadId")
        notification_turn_id = _notification_turn_id(params)
        if notification_thread_id != thread_id:
            return
        if notification_turn_id is not None and notification_turn_id != turn_id:
            return
        method = notification.get("method")
        if method == "item/reasoning/textDelta" or method == "item/reasoning/summaryTextDelta":
            self._emit_codex_event(
                {
                    "type": "reasoning_delta",
                    "item_id": params.get("itemId"),
                    "delta": params.get("delta", ""),
                }
            )
        elif method == "item/plan/delta":
            self._emit_codex_event(
                {
                    "type": "plan_delta",
                    "item_id": params.get("itemId"),
                    "delta": params.get("delta", ""),
                }
            )
        elif method == "item/agentMessage/delta":
            item_id = params.get("itemId")
            delta = params.get("delta", "")
            if isinstance(item_id, str) and isinstance(delta, str):
                self._assistant_deltas[item_id] = self._assistant_deltas.get(item_id, "") + delta
            self._emit_codex_event(
                {
                    "type": "assistant_delta",
                    "item_id": item_id,
                    "delta": delta,
                }
            )
        elif method == "item/started":
            item_type = params.get("item", {}).get("type") if isinstance(params.get("item"), dict) else None
            if isinstance(item_type, str) and item_type:
                self._emit_codex_event({"type": "status", "text": f"Codex started {item_type}"})
        elif method == "item/completed":
            item = params.get("item") if isinstance(params.get("item"), dict) else None
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "agentMessage":
                item_id = item.get("id")
                item_text = item.get("text")
                if isinstance(item_id, str) and isinstance(item_text, str):
                    self._assistant_deltas[item_id] = item_text
            if isinstance(item_type, str) and item_type:
                self._emit_codex_event({"type": "status", "text": f"Codex completed {item_type}"})
        elif method == "turn/completed":
            self._emit_codex_event({"type": "status", "text": "Codex turn completed"})
        elif method == "error":
            message = params.get("message")
            if isinstance(message, str) and message.strip():
                self._emit_codex_event({"type": "error", "text": message.strip()})

    def _emit_chat_model_start(self, messages: list[BaseMessage]) -> None:
        payload = {
            "name": "codex",
            "id": ["tradingagents", "codex"],
        }
        for callback in self.callbacks:
            handler = getattr(callback, "on_chat_model_start", None)
            if callable(handler):
                try:
                    handler(payload, [messages])
                except Exception:
                    pass

    def _emit_codex_event(self, event: dict[str, Any]) -> None:
        for callback in self.callbacks:
            handler = getattr(callback, "on_codex_event", None)
            if callable(handler):
                try:
                    handler(event)
                except Exception:
                    pass

    def _latest_assistant_text(self) -> str:
        if not self._assistant_deltas:
            return ""
        last_key = next(reversed(self._assistant_deltas))
        return self._assistant_deltas[last_key]


def list_codex_models(
    codex_command: str = DEFAULT_CODEX_COMMAND,
    codex_args: tuple[str, ...] = DEFAULT_CODEX_ARGS,
    cwd: Optional[str] = None,
) -> list[str]:
    _model_loading_debug_emit(
        "starting codex model listing",
        {
            "codex_command": codex_command,
            "codex_args": list(codex_args),
            "cwd": cwd,
            "which_codex": shutil.which(codex_command),
        },
    )
    model = CodexChatModel(
        model="gpt-5.4",
        cwd=cwd,
        codex_command=codex_command,
        codex_args=codex_args,
    )
    try:
        models = model.list_models()
        _model_loading_debug_emit(
            "codex model listing succeeded",
            {"models": models},
        )
        return models
    except Exception as error:
        _model_loading_debug_emit(
            "codex model listing failed",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        model.close()


class CodexClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, provider: str = "codex", **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        return CodexChatModel(
            model=self.model,
            cwd=self.kwargs.get("cwd"),
            reasoning_effort=self.kwargs.get("reasoning_effort"),
            codex_command=self.kwargs.get("codex_command", DEFAULT_CODEX_COMMAND),
            codex_args=tuple(self.kwargs.get("codex_args", DEFAULT_CODEX_ARGS)),
            callbacks=self.kwargs.get("callbacks"),
        )

    def validate_model(self) -> bool:
        provider = "codex" if self.provider in ("codex", "openai_codex_oauth") else self.provider
        return validate_model(provider, self.model)
