import unittest

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.graph.trading_graph import (
    TradingAgentsGraph,
    _debug_message_key,
    should_print_debug_message,
)


class ResettableModel:
    def __init__(self):
        self.reset_calls = 0

    def reset_thread(self):
        self.reset_calls += 1


class TradingGraphDebugTests(unittest.TestCase):
    def test_keeps_single_continue_message(self):
        self.assertTrue(should_print_debug_message(HumanMessage(content="Continue")))

    def test_keeps_non_placeholder_messages(self):
        self.assertTrue(should_print_debug_message(HumanMessage(content="Continue with more detail")))
        self.assertTrue(should_print_debug_message(AIMessage(content="Decision: BUY")))

    def test_debug_message_key_uses_content_when_id_missing(self):
        first = AIMessage(content="Decision: BUY")
        second = AIMessage(content="Decision: BUY")
        self.assertEqual(_debug_message_key(first), _debug_message_key(second))

    def test_debug_message_key_prefers_stable_id(self):
        first = AIMessage(content="Decision: BUY", id="msg-1")
        second = AIMessage(content="Decision: BUY updated", id="msg-1")
        self.assertEqual(_debug_message_key(first), _debug_message_key(second))

    def test_reset_reusable_model_threads_deduplicates_shared_instances(self):
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        shared_model = ResettableModel()
        other_model = ResettableModel()
        graph.deep_thinking_llm = shared_model
        graph.quick_thinking_llm = shared_model
        graph.research_manager_llm = other_model
        graph.portfolio_manager_llm = other_model
        graph._reset_reusable_model_threads()
        self.assertEqual(shared_model.reset_calls, 1)
        self.assertEqual(other_model.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
