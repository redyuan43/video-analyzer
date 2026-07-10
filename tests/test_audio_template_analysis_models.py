import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from video_analyzer.asr_providers import ASRStrategyResult
from video_analyzer.audio_processor import AudioTranscript
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

    def test_auto_asr_uses_deep_strategy_when_speaker_diarization_is_enabled(self):
        config = Config('config')
        config.config.setdefault('asr', {})['provider'] = 'auto'
        config.config.setdefault('asr', {})['strategy'] = 'balanced'
        config.config.setdefault('speaker_diarization', {})['enabled'] = True
        transcript = AudioTranscript(text='hello', segments=[], language='zh', metadata={})
        result = ASRStrategyResult(strategy='deep', transcript=transcript)

        with patch.object(run_audio_template_analysis, 'analyzer_resource_lock') as lock_factory:
            lock_factory.return_value.__enter__.return_value = None
            lock_factory.return_value.__exit__.return_value = None
            with patch.object(run_audio_template_analysis, 'local_model_runtime_session') as runtime_factory:
                runtime_factory.return_value.__enter__.return_value = None
                runtime_factory.return_value.__exit__.return_value = None
                with patch.object(run_audio_template_analysis, 'local_model_stage') as stage_factory:
                    stage_factory.return_value.__enter__.return_value = None
                    stage_factory.return_value.__exit__.return_value = None
                    with patch.object(run_audio_template_analysis, 'transcribe_with_strategy', return_value=result) as strategy_mock:
                        got_transcript, got_result = run_audio_template_analysis.transcribe_audio(
                            Path('/tmp/fake.wav'),
                            Path('/tmp/out'),
                            config,
                        )

        self.assertIs(got_transcript, transcript)
        self.assertIs(got_result, result)
        self.assertEqual(strategy_mock.call_args.kwargs['strategy'], 'deep')


if __name__ == '__main__':
    unittest.main()
