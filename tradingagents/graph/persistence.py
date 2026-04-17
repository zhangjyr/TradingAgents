import hashlib
import json
import os
import pickle
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver


class PersistentInMemorySaver(InMemorySaver):
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._persist_lock = threading.Lock()
        super().__init__()
        self._load()

    def put(self, config, checkpoint, metadata, new_versions):
        next_config = super().put(config, checkpoint, metadata, new_versions)
        self._persist()
        return next_config

    def put_writes(self, config, writes, task_id, task_path=""):
        super().put_writes(config, writes, task_id, task_path)
        self._persist()

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        self._persist()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        with self.file_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.storage = defaultdict(lambda: defaultdict(dict))
        for thread_id, namespaces in payload["storage"].items():
            self.storage[thread_id] = defaultdict(dict, namespaces)
        self.writes = defaultdict(dict, payload["writes"])
        self.blobs = dict(payload["blobs"])

    def _persist(self) -> None:
        with self._persist_lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=self.file_path.parent,
                prefix=self.file_path.name,
                suffix=".tmp",
            ) as handle:
                pickle.dump(
                    {
                        "storage": {
                            thread_id: {
                                checkpoint_ns: dict(checkpoints)
                                for checkpoint_ns, checkpoints in namespaces.items()
                            }
                            for thread_id, namespaces in self.storage.items()
                        },
                        "writes": {key: dict(value) for key, value in self.writes.items()},
                        "blobs": dict(self.blobs),
                    },
                    handle,
                )
                temp_path = Path(handle.name)
            os.replace(temp_path, self.file_path)

    def persist(self) -> None:
        self._persist()


def build_run_signature(selections: dict[str, Any], selected_analysts: list[str]) -> str:
    payload = {
        "ticker": selections["ticker"],
        "analysis_date": selections["analysis_date"],
        "research_depth": selections["research_depth"],
        "provider": selections["llm_provider"].lower(),
        "backend_url": selections.get("backend_url"),
        "quick_model": selections["shallow_thinker"],
        "deep_model": selections["deep_thinker"],
        "anthropic_effort": selections.get("anthropic_effort"),
        "openai_reasoning_effort": selections.get("openai_reasoning_effort"),
        "google_thinking_level": selections.get("google_thinking_level"),
        "analysts": selected_analysts,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def is_resumable_run_state(
    run_state: dict[str, Any],
    signature: str,
    checkpoint_path: str | Path,
) -> bool:
    return (
        run_state.get("status") in {"paused_rate_limit", "paused_manual_exit"}
        and run_state.get("signature") == signature
        and Path(checkpoint_path).exists()
    )


def compute_retry_delay(
    retry_attempt: int,
    total_waited_seconds: int,
    max_total_wait_seconds: int = 300,
    base_delay_seconds: int = 5,
) -> int:
    if total_waited_seconds >= max_total_wait_seconds:
        return 0
    delay = min(base_delay_seconds * (2 ** max(retry_attempt - 1, 0)), max_total_wait_seconds)
    remaining = max_total_wait_seconds - total_waited_seconds
    return min(delay, remaining)


def load_json_file(file_path: str | Path) -> Optional[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_json_file(file_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def safe_serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe_serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_serialize(item) for item in value]
    if hasattr(value, "model_dump"):
        return safe_serialize(value.model_dump())
    if hasattr(value, "content"):
        payload = {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": safe_serialize(getattr(value, "content", "")),
        }
        msg_id = getattr(value, "id", None)
        if msg_id is not None:
            payload["id"] = msg_id
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = safe_serialize(tool_calls)
        return payload
    return repr(value)


def snapshot_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "company_of_interest",
        "trade_date",
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_debate_state",
        "investment_plan",
        "trader_investment_plan",
        "risk_debate_state",
        "final_trade_decision",
        "messages",
    ]
    return {key: safe_serialize(state.get(key)) for key in keys if key in state}


def detect_rate_limit_error(error: BaseException) -> bool:
    text = str(error).lower()
    name = type(error).__name__.lower()
    return "ratelimit" in name or "rate limit" in text or "429" in text or "rate_limit_error" in text


def summarize_graph_state(state: dict[str, Any]) -> str:
    if state.get("final_trade_decision"):
        return "portfolio_manager"
    risk_state = state.get("risk_debate_state") or {}
    if isinstance(risk_state, dict) and (
        risk_state.get("aggressive_history")
        or risk_state.get("conservative_history")
        or risk_state.get("neutral_history")
    ):
        return "risk_management"
    if state.get("trader_investment_plan"):
        return "trader"
    debate_state = state.get("investment_debate_state") or {}
    if isinstance(debate_state, dict) and (
        debate_state.get("bull_history")
        or debate_state.get("bear_history")
        or debate_state.get("judge_decision")
    ):
        return "research_manager"
    if state.get("market_report") or state.get("sentiment_report") or state.get("news_report") or state.get("fundamentals_report"):
        return "analysts"
    return "initializing"


def clear_resume_files(*paths: str | Path) -> None:
    for path in paths:
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
