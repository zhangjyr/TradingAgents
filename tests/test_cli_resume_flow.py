import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli.main import build_resume_paths, prompt_for_resume_choice
from tradingagents.graph.persistence import clear_resume_files


class TestCliResumeFlow(unittest.TestCase):
    def test_prompt_for_resume_choice_accepts_resume(self):
        with patch("cli.main.typer.prompt", return_value="R"):
            self.assertTrue(
                prompt_for_resume_choice(
                    {"status": "paused_rate_limit", "last_stage": "research_manager"}
                )
            )

    def test_prompt_for_resume_choice_accepts_start_new(self):
        with patch("cli.main.typer.prompt", return_value="N"):
            self.assertFalse(
                prompt_for_resume_choice(
                    {"status": "paused_manual_exit", "last_stage": "trader"}
                )
            )

    def test_clear_resume_files_removes_saved_resume_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_resume_paths(Path(tmpdir))
            for path in paths.values():
                path.write_text("saved")
            clear_resume_files(*paths.values())
            self.assertFalse(any(path.exists() for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
