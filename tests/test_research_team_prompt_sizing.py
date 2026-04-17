import json
import unittest
from pathlib import Path

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


class FakeMemory:
    def get_memories(self, curr_situation, n_matches=2):
        return []


class FakeLLM:
    def __init__(self):
        self.last_prompt = ""
        self.model = "claude-sonnet-4-6"

    def invoke(self, prompt):
        self.last_prompt = prompt

        class Response:
            content = "ok"

        return Response()


class TestResearchTeamPromptSizing(unittest.TestCase):
    def _build_large_state(self):
        huge = "X" * 12000
        return {
            "company_of_interest": "ADBE",
            "market_report": huge,
            "sentiment_report": huge,
            "news_report": huge,
            "fundamentals_report": huge,
            "investment_debate_state": {
                "history": huge * 2,
                "bull_history": huge,
                "bear_history": huge,
                "current_response": huge,
                "count": 2,
            },
        }

    def test_bull_researcher_compacts_large_prompt(self):
        llm = FakeLLM()
        node = create_bull_researcher(llm, FakeMemory())
        node(self._build_large_state())
        self.assertLess(len(llm.last_prompt), 25000)

    def test_bear_researcher_compacts_large_prompt(self):
        llm = FakeLLM()
        node = create_bear_researcher(llm, FakeMemory())
        node(self._build_large_state())
        self.assertLess(len(llm.last_prompt), 25000)

    def test_research_manager_compacts_saved_adbe_state(self):
        state_path = Path("results/ADBE/2026-04-16/latest_state.json")
        state = json.loads(state_path.read_text())
        llm = FakeLLM()
        node = create_research_manager(llm, FakeMemory())
        node(state)
        self.assertLess(len(llm.last_prompt), 32000)
        self.assertIn("Bull analyst case:", llm.last_prompt)
        self.assertIn("Bear analyst case:", llm.last_prompt)

    def test_portfolio_manager_compacts_saved_adbe_state(self):
        state_path = Path("results/ADBE/2026-04-16/latest_state.json")
        state = json.loads(state_path.read_text())
        llm = FakeLLM()
        node = create_portfolio_manager(llm, FakeMemory())
        node(state)
        self.assertLess(len(llm.last_prompt), 26000)
        self.assertIn("Risk Analysts Debate History:", llm.last_prompt)


if __name__ == "__main__":
    unittest.main()
