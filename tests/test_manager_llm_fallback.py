import unittest

from tradingagents.graph.trading_graph import (
    RateLimitFallbackLLM,
    build_manager_llm,
)


class FakeLLM:
    def __init__(self, model, result=None, error=None):
        self.model = model
        self.result = result
        self.error = error
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class TestManagerLlmFallback(unittest.TestCase):
    def test_rate_limit_wrapper_uses_primary_when_successful(self):
        primary = FakeLLM("deep-model", result="deep")
        fallback = FakeLLM("quick-model", result="quick")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = llm.invoke("prompt")

        self.assertEqual(result, "deep")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 0)

    def test_rate_limit_wrapper_falls_back_on_rate_limit(self):
        primary = FakeLLM(
            "deep-model",
            error=RuntimeError("Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'Error'}}"),
        )
        fallback = FakeLLM("quick-model", result="quick")
        llm = RateLimitFallbackLLM(primary, fallback)

        result = llm.invoke("prompt")

        self.assertEqual(result, "quick")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)

    def test_rate_limit_wrapper_does_not_hide_non_rate_limit_errors(self):
        primary = FakeLLM("deep-model", error=RuntimeError("socket closed"))
        fallback = FakeLLM("quick-model", result="quick")
        llm = RateLimitFallbackLLM(primary, fallback)

        with self.assertRaises(RuntimeError):
            llm.invoke("prompt")

        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 0)

    def test_build_manager_llm_wraps_claude_code(self):
        primary = FakeLLM("deep-model", result="deep")
        fallback = FakeLLM("quick-model", result="quick")

        llm = build_manager_llm("claude_code", primary, fallback)

        self.assertIsInstance(llm, RateLimitFallbackLLM)

    def test_build_manager_llm_keeps_primary_for_other_providers(self):
        primary = FakeLLM("deep-model", result="deep")
        fallback = FakeLLM("quick-model", result="quick")

        llm = build_manager_llm("openai", primary, fallback)

        self.assertIs(llm, primary)


if __name__ == "__main__":
    unittest.main()
