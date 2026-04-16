from typing import Optional

from .codex_client import CodexClient, get_codex_auth_path, has_codex_auth, list_codex_models as list_codex_app_server_models


def list_codex_models(
    base_url: Optional[str] = None,
    oauth_token: Optional[str] = None,
    timeout: float = 30.0,
) -> list[str]:
    return list_codex_app_server_models()


class OpenAIOAuthClient(CodexClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, provider="openai_codex_oauth", **kwargs)
