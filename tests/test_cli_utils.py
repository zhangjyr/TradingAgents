import unittest
from unittest.mock import patch

from cli import utils as cli_utils


class TestCliUtils(unittest.TestCase):
    def setUp(self):
        cli_utils._CODEX_DISCOVERY_CACHE.clear()
        cli_utils._CODEX_DISCOVERY_FAILURES.clear()

    def tearDown(self):
        cli_utils._CODEX_DISCOVERY_CACHE.clear()
        cli_utils._CODEX_DISCOVERY_FAILURES.clear()

    @patch("cli.utils.console.print")
    @patch("cli.utils.get_model_options")
    @patch("cli.utils.list_codex_models")
    def test_resolve_model_options_reports_failure_and_uses_static_catalog(
        self,
        mock_list_models,
        mock_get_model_options,
        mock_print,
    ):
        mock_list_models.side_effect = RuntimeError("Codex app-server request timed out: model/list")
        mock_get_model_options.return_value = [("GPT-5.4", "gpt-5.4")]

        choices = cli_utils._resolve_model_options("codex", "quick")

        self.assertEqual(choices, [("GPT-5.4", "gpt-5.4")])
        mock_print.assert_called_once()
        self.assertIn("codex", cli_utils._CODEX_DISCOVERY_FAILURES)

    @patch("cli.utils.get_model_options")
    @patch("cli.utils.list_codex_models")
    def test_resolve_model_options_uses_cached_failure_without_second_probe(
        self,
        mock_list_models,
        mock_get_model_options,
    ):
        cli_utils._CODEX_DISCOVERY_FAILURES.add("codex")
        mock_get_model_options.return_value = [("GPT-5.4", "gpt-5.4")]

        choices = cli_utils._resolve_model_options("codex", "deep")

        self.assertEqual(choices, [("GPT-5.4", "gpt-5.4")])
        mock_list_models.assert_not_called()


if __name__ == "__main__":
    unittest.main()
