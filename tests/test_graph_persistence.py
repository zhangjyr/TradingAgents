import tempfile
import unittest
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tradingagents.graph.persistence import (
    PersistentInMemorySaver,
    compute_retry_delay,
    detect_rate_limit_error,
    is_resumable_run_state,
)


class CounterState(TypedDict):
    value: int


class TestGraphPersistence(unittest.TestCase):
    def test_detect_rate_limit_error(self):
        self.assertTrue(detect_rate_limit_error(RuntimeError("429 rate_limit_error")))
        self.assertTrue(detect_rate_limit_error(RuntimeError("Rate limit exceeded")))
        self.assertFalse(detect_rate_limit_error(RuntimeError("network timeout")))

    def test_compute_retry_delay_caps_total_wait_at_300_seconds(self):
        self.assertEqual(compute_retry_delay(1, 0), 5)
        self.assertEqual(compute_retry_delay(2, 5), 10)
        self.assertEqual(compute_retry_delay(7, 155), 145)
        self.assertEqual(compute_retry_delay(8, 300), 0)

    def test_is_resumable_run_state_accepts_paused_statuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "graph_checkpoint.pkl"
            checkpoint_path.write_bytes(b"checkpoint")
            self.assertTrue(
                is_resumable_run_state(
                    {"status": "paused_rate_limit", "signature": "abc"},
                    "abc",
                    checkpoint_path,
                )
            )
            self.assertTrue(
                is_resumable_run_state(
                    {"status": "paused_manual_exit", "signature": "abc"},
                    "abc",
                    checkpoint_path,
                )
            )
            self.assertFalse(
                is_resumable_run_state(
                    {"status": "completed", "signature": "abc"},
                    "abc",
                    checkpoint_path,
                )
            )

    def test_persistent_saver_resumes_after_failure(self):
        attempts = {"count": 0}

        def first_step(state: CounterState):
            return {"value": state["value"] + 1}

        def second_step(state: CounterState):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("429 rate_limit_error")
            return {"value": state["value"] + 1}

        workflow = StateGraph(CounterState)
        workflow.add_node("first_step", first_step)
        workflow.add_node("second_step", second_step)
        workflow.add_edge(START, "first_step")
        workflow.add_edge("first_step", "second_step")
        workflow.add_edge("second_step", END)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "graph_checkpoint.pkl"
            config = {"configurable": {"thread_id": "resume-thread"}}

            saver = PersistentInMemorySaver(checkpoint_path)
            graph = workflow.compile(checkpointer=saver)

            with self.assertRaises(RuntimeError):
                list(graph.stream({"value": 0}, config=config, stream_mode="values"))

            interrupted_state = graph.get_state(config)
            saver.persist()
            self.assertEqual(interrupted_state.values["value"], 1)
            self.assertTrue(checkpoint_path.exists())

            resumed_graph = workflow.compile(checkpointer=PersistentInMemorySaver(checkpoint_path))
            chunks = list(resumed_graph.stream(None, config=config, stream_mode="values"))

            self.assertEqual(chunks[-1]["value"], 2)


if __name__ == "__main__":
    unittest.main()
