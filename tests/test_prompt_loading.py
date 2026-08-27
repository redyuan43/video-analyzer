#!/usr/bin/env python3
"""Tests for the prompt loading system (PromptLoader)."""
import tempfile
import unittest
from pathlib import Path

from video_analyzer.prompt import PromptLoader


class TestPromptLoading(unittest.TestCase):
    """Test prompt loading with package and custom prompts."""

    PROMPTS = [
        {"name": "Frame Analysis", "path": "frame_analysis/frame_analysis.txt"},
        {"name": "Video Reconstruction", "path": "frame_analysis/describe.txt"},
    ]

    def test_default_package_prompts(self):
        loader = PromptLoader("", self.PROMPTS)
        self.assertTrue(
            loader.get_by_name("Frame Analysis"),
            "Failed to load package frame analysis prompt",
        )
        self.assertTrue(
            loader.get_by_name("Video Reconstruction"),
            "Failed to load package describe prompt",
        )

    def test_custom_prompts_in_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_prompts"
            frame_dir = test_dir / "frame_analysis"
            frame_dir.mkdir(parents=True)

            (frame_dir / "frame_analysis.txt").write_text("Test frame analysis")
            (frame_dir / "describe.txt").write_text("Test description")

            loader = PromptLoader(str(test_dir), self.PROMPTS)
            self.assertEqual(loader.get_by_name("Frame Analysis"), "Test frame analysis")
            self.assertEqual(loader.get_by_name("Video Reconstruction"), "Test description")


if __name__ == "__main__":
    unittest.main()
