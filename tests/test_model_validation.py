import sys
import types
import unittest
import warnings

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

from tradingagents.llm_clients.base_client import BaseLLMClient
from tradingagents.llm_clients.model_catalog import get_known_models
from tradingagents.llm_clients.validators import validate_model


class DummyLLMClient(BaseLLMClient):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        super().__init__(model)

    def get_llm(self):
        self.warn_if_unknown_model()
        return object()

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)


class ModelValidationTests(unittest.TestCase):
    def test_cli_catalog_models_are_all_validator_approved(self):
        for provider, models in get_known_models().items():
            if provider in ("ollama", "openrouter"):
                continue

            for model in models:
                with self.subTest(provider=provider, model=model):
                    self.assertTrue(validate_model(provider, model))

    def test_unknown_model_emits_warning_for_strict_provider(self):
        client = DummyLLMClient("openai", "not-a-real-openai-model")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.get_llm()

        self.assertEqual(len(caught), 1)
        self.assertIn("not-a-real-openai-model", str(caught[0].message))
        self.assertIn("openai", str(caught[0].message))

    def test_openrouter_and_ollama_accept_custom_models_without_warning(self):
        for provider in ("openrouter", "ollama"):
            client = DummyLLMClient(provider, "custom-model-name")

            with self.subTest(provider=provider):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    client.get_llm()

                self.assertEqual(caught, [])
