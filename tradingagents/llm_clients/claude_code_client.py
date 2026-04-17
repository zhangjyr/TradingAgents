import getpass
import hashlib
import binascii
import json
import os
import subprocess
import time
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import requests
try:
    from langchain_anthropic import ChatAnthropic
    from pydantic import Field, SecretStr
    _CLAUDE_CODE_DEPS_AVAILABLE = True
except Exception:
    ChatAnthropic = object

    def Field(default=None, **_kwargs):
        return default

    class SecretStr(str):
        def get_secret_value(self) -> str:
            return str(self)

    _CLAUDE_CODE_DEPS_AVAILABLE = False

from .base_client import BaseLLMClient, normalize_content
from .anthropic_client import _emit_provider_event, _extract_text_delta, supports_anthropic_effort
from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message
from .validators import validate_model

CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_CODE_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLAUDE_CODE_OAUTH_BETA_HEADER = "oauth-2025-04-20"
CLAUDE_CODE_OAUTH_SCOPES = [
    "user:profile",
    "user:inference",
    "user:sessions:claude_code",
    "user:mcp_servers",
    "user:file_upload",
]

_PASSTHROUGH_KWARGS = (
    "timeout",
    "max_retries",
    "max_tokens",
    "callbacks",
    "http_client",
    "http_async_client",
    "effort",
)


def get_claude_config_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")).expanduser()


def get_claude_credentials_path() -> Path:
    return get_claude_config_home() / ".credentials.json"


def _get_claude_keychain_service_name() -> str:
    config_dir = str(get_claude_config_home())
    is_default_dir = "CLAUDE_CONFIG_DIR" not in os.environ
    suffix = ""
    if not is_default_dir:
        suffix = "-" + hashlib.sha256(config_dir.encode("utf-8")).hexdigest()[:8]
    return f"Claude Code-credentials{suffix}"


def _read_macos_keychain_credentials() -> Optional[dict[str, Any]]:
    if os.name != "posix" or sys_platform() != "darwin":
        return None
    username = os.environ.get("USER") or getpass.getuser()
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                username,
                "-w",
                "-s",
                _get_claude_keychain_service_name(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_claude_oauth_state() -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]:
    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        payload = {
            "claudeAiOauth": {
                "accessToken": env_token,
                "refreshToken": os.environ.get("CLAUDE_CODE_OAUTH_REFRESH_TOKEN"),
                "expiresAt": None,
                "scopes": ["user:inference"],
                "subscriptionType": None,
                "rateLimitTier": None,
            }
        }
        return payload, payload["claudeAiOauth"], "env"

    credentials_path = get_claude_credentials_path()
    if credentials_path.exists():
        try:
            payload = json.loads(credentials_path.read_text())
            if isinstance(payload, dict):
                oauth_data = payload.get("claudeAiOauth")
                if isinstance(oauth_data, dict):
                    return payload, oauth_data, "file"
        except Exception:
            pass
    payload = _read_macos_keychain_credentials()
    if isinstance(payload, dict):
        oauth_data = payload.get("claudeAiOauth")
        if isinstance(oauth_data, dict):
            return payload, oauth_data, "keychain"
    return None, None, None


def _token_is_expired(tokens: dict[str, Any], skew_ms: int = 60_000) -> bool:
    expires_at = tokens.get("expiresAt")
    if not isinstance(expires_at, (int, float)):
        return False
    return int(expires_at) <= int(time.time() * 1000) + skew_ms


def _refresh_claude_oauth_token(tokens: dict[str, Any]) -> dict[str, Any]:
    refresh_token = tokens.get("refreshToken")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return tokens
    response = requests.post(
        CLAUDE_CODE_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLAUDE_CODE_CLIENT_ID,
            "scope": " ".join(CLAUDE_CODE_OAUTH_SCOPES),
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return tokens
    expires_in = payload.get("expires_in")
    expires_at = None
    if isinstance(expires_in, (int, float)):
        expires_at = int(time.time() * 1000) + int(expires_in * 1000)
    scopes = payload.get("scope")
    if isinstance(scopes, str):
        parsed_scopes = [scope for scope in scopes.split() if scope]
    else:
        parsed_scopes = tokens.get("scopes")
    refreshed = dict(tokens)
    refreshed["accessToken"] = access_token.strip()
    refreshed["refreshToken"] = payload.get("refresh_token", refresh_token)
    refreshed["expiresAt"] = expires_at
    refreshed["scopes"] = parsed_scopes
    return refreshed


def _write_file_credentials_payload(payload: dict[str, Any]) -> None:
    credentials_path = get_claude_credentials_path()
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.chmod(credentials_path, 0o600)


def _write_macos_keychain_credentials(payload: dict[str, Any]) -> None:
    username = os.environ.get("USER") or getpass.getuser()
    json_string = json.dumps(payload, ensure_ascii=False)
    hex_value = binascii.hexlify(json_string.encode("utf-8")).decode("ascii")
    command = (
        f'add-generic-password -U -a "{username}" '
        f'-s "{_get_claude_keychain_service_name()}" -X "{hex_value}"\n'
    )
    subprocess.run(
        ["security", "-i"],
        input=command,
        capture_output=True,
        text=True,
        check=True,
    )


def _persist_refreshed_tokens(
    source: Optional[str],
    original_payload: Optional[dict[str, Any]],
    refreshed_tokens: dict[str, Any],
) -> bool:
    if source in (None, "env"):
        return False
    payload = dict(original_payload or {})
    payload["claudeAiOauth"] = dict(refreshed_tokens)
    try:
        if source == "keychain" and os.name == "posix" and sys_platform() == "darwin":
            _write_macos_keychain_credentials(payload)
        else:
            _write_file_credentials_payload(payload)
        return True
    except Exception:
        return False


def get_claude_code_oauth_token() -> Optional[str]:
    payload, tokens, source = _read_claude_oauth_state()
    if not tokens:
        return None
    if _token_is_expired(tokens):
        try:
            refreshed_tokens = _refresh_claude_oauth_token(tokens)
            if refreshed_tokens != tokens:
                _persist_refreshed_tokens(source, payload, refreshed_tokens)
            tokens = refreshed_tokens
        except Exception:
            pass
    access_token = tokens.get("accessToken")
    if isinstance(access_token, str) and access_token.strip():
        return access_token.strip()
    return None


def has_claude_code_auth() -> bool:
    return bool(get_claude_code_oauth_token())


def sys_platform() -> str:
    return os.uname().sysname.lower() if hasattr(os, "uname") else ""


class NormalizedClaudeCodeChatAnthropic(ChatAnthropic):
    auth_token: Optional[SecretStr] = Field(default=None, exclude=True)

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        default_headers = {"User-Agent": "TradingAgents-ClaudeCode"}
        if self.default_headers:
            default_headers.update(self.default_headers)
        if self.auth_token is not None:
            default_headers.setdefault("anthropic-beta", CLAUDE_CODE_OAUTH_BETA_HEADER)
        client_params: dict[str, Any] = {
            "base_url": self.anthropic_api_url,
            "max_retries": self.max_retries,
            "default_headers": default_headers,
        }
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_params["timeout"] = self.default_request_timeout
        if self.auth_token is not None:
            client_params["auth_token"] = (
                self.auth_token.get_secret_value()
                if hasattr(self.auth_token, "get_secret_value")
                else str(self.auth_token)
            )
        else:
            client_params["api_key"] = (
                self.anthropic_api_key.get_secret_value()
                if hasattr(self.anthropic_api_key, "get_secret_value")
                else str(self.anthropic_api_key)
            )
        return client_params

    def invoke(self, input, config=None, **kwargs):
        callbacks = getattr(self, "callbacks", None)
        _emit_provider_event(
            callbacks,
            {"type": "status", "text": f"Starting Claude Code response with {self.model}"},
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
                {"type": "status", "text": f"Claude Code response completed for {self.model}"},
            )
            return response
        except Exception as error:
            _emit_provider_event(callbacks, {"type": "error", "text": str(error)})
            raise


class ClaudeCodeClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = "claude_code"

    def get_llm(self) -> Any:
        if not _CLAUDE_CODE_DEPS_AVAILABLE:
            raise ImportError(
                "Claude Code provider requires langchain-anthropic and pydantic in the active environment."
            )
        self.warn_if_unknown_model()
        auth_token = self.kwargs.get("auth_token") or get_claude_code_oauth_token()
        if not auth_token:
            raise ValueError("No Claude Code OAuth credential found. Run `claude login` first.")

        llm_kwargs = {
            "model": self.model,
            "auth_token": auth_token,
            "anthropic_api_key": "unused",
        }

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                if key == "effort" and not supports_anthropic_effort(self.model):
                    continue
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedClaudeCodeChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
