import importlib.util
import unittest
from pathlib import Path

from video_analyzer.config import Config

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tools' / 'run_audio_template_analysis.py'
SPEC = importlib.util.spec_from_file_location('run_audio_template_analysis', MODULE_PATH)
run_audio_template_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_audio_template_analysis)


class AudioTemplateAnalysisModelTests(unittest.TestCase):
    def test_template_selector_uses_study_cards_qwen_model(self):
        config = Config('config')
        _client, model, base_url, temperature = run_audio_template_analysis.build_template_selector_client(config)

        self.assertEqual(model, 'qwen3:4b-instruct')
        self.assertEqual(base_url, 'http://agx.taild500c8.ts.net:11434/v1')
        self.assertEqual(temperature, 0.1)

    def test_content_analysis_uses_deepseek_runtime_profile(self):
        config = Config('config')
        _client, model, base_url, temperature = run_audio_template_analysis.build_content_analysis_client(
            config,
            'deepseek_v4_pro',
        )

        self.assertEqual(model, 'deepseek-v4-pro')
        self.assertEqual(base_url, 'https://api.deepseek.com')
        self.assertEqual(temperature, 1.0)

    def test_recording_time_from_source_filename(self):
        self.assertEqual(
            run_audio_template_analysis.recording_time_from_source('20260709113245.mp3'),
            '2026年7月9日 11:32:45',
        )


if __name__ == '__main__':
    unittest.main()
