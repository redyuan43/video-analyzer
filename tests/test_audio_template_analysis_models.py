import importlib.util
import json
import tempfile
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
            'deepseek_v4_flash',
        )

        self.assertEqual(model, 'deepseek-v4-flash')
        self.assertEqual(base_url, 'https://api.deepseek.com')
        self.assertEqual(temperature, 1.0)

    def test_audio_deepseek_flash_profile_uses_audio_workflow(self):
        config = Config('config')
        profile = config.get_runtime_profile('audio_nx1_deepseek_flash')
        _client, model, base_url, temperature = (
            run_audio_template_analysis.build_content_analysis_client(
                config,
                'audio_nx1_deepseek_flash',
            )
        )

        self.assertEqual(profile['workflow_id'], 'audio_nx1')
        self.assertEqual(model, 'deepseek-v4-flash')
        self.assertEqual(base_url, 'https://api.deepseek.com')
        self.assertEqual(temperature, 1.0)

    def test_audio_local_quality_profile_uses_tuned_bonsai_settings(self):
        config = Config('config')
        profile = config.get_runtime_profile('audio_nx1_local_quality')
        client, model, base_url, temperature = (
            run_audio_template_analysis.build_content_analysis_client(
                config,
                'audio_nx1_local_quality',
            )
        )

        self.assertEqual(profile['workflow_id'], 'audio_nx1')
        self.assertEqual(model, 'prism-ml/bonsai-27b')
        self.assertEqual(base_url, 'http://127.0.0.1:18103/v1')
        self.assertEqual(temperature, 0.2)
        self.assertEqual(client.extra_body['repeat_penalty'], 1.1)
        self.assertEqual(profile['summary_single_pass_chars'], 12000)
        self.assertEqual(profile['summary_map_chunk_chars'], 8000)

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

    def test_main_with_provided_transcript_skips_audio_asr_and_diarization(self):
        transcript = AudioTranscript(
            text='provided words',
            segments=[{'start': 0.1, 'end': 1.2, 'speaker': 'A', 'text': 'provided words'}],
            language='en',
            metadata={'source': 'provided'},
        )
        result = ASRStrategyResult(
            strategy='provided_transcript',
            transcript=transcript,
            providers_run=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / 'demo.mp3'
            media.write_bytes(b'audio is intentionally unused')
            provided = root / 'provided.json'
            provided.write_text(json.dumps({'text': 'provided words'}), encoding='utf-8')
            output = root / 'output'
            argv = [
                'run_audio_template_analysis.py', str(media), '--output', str(output),
                '--transcript-json', str(provided),
            ]
            selected = {'id': 'test', 'title': 'Test', 'prompt_original': '{transcript}'}
            with (
                patch('sys.argv', argv),
                patch.object(run_audio_template_analysis, 'load_operation_config'),
                patch.object(run_audio_template_analysis, 'load_templates', return_value=[selected]),
                patch.object(run_audio_template_analysis, 'load_provided_transcript', return_value=(transcript, result)),
                patch.object(run_audio_template_analysis, 'extract_audio_to_wav') as extract_mock,
                patch.object(run_audio_template_analysis, 'transcribe_audio') as asr_mock,
                patch.object(run_audio_template_analysis, 'refine_audio_speakers') as diarization_mock,
                patch.object(run_audio_template_analysis, 'build_template_selector_client', return_value=(object(), 'selector', 'http://selector', 0.1)),
                patch.object(run_audio_template_analysis, 'build_content_analysis_client', return_value=(object(), 'content', 'http://content', 0.1)),
                patch.object(run_audio_template_analysis, 'choose_template', return_value=(selected, {'method': 'test'})),
                patch.object(run_audio_template_analysis, 'summarize_with_template', return_value='summary'),
                patch.object(run_audio_template_analysis, 'write_audio_only_manifest'),
                patch.object(run_audio_template_analysis, 'build_light_study_guide'),
                patch.object(run_audio_template_analysis, 'write_operation_manual', return_value=output / 'operation_manual.md'),
                patch.object(run_audio_template_analysis, 'write_manual_evidence', return_value=output / 'manual_evidence.md'),
                patch.object(run_audio_template_analysis, 'write_analysis_json', return_value=output / 'analysis.json'),
                patch.object(run_audio_template_analysis, 'local_model_stage') as text_stage,
            ):
                self.assertEqual(run_audio_template_analysis.main(), 0)

        extract_mock.assert_not_called()
        asr_mock.assert_not_called()
        diarization_mock.assert_not_called()
        self.assertEqual(text_stage.call_args.args[0], 'text')
        self.assertEqual(result.strategy, 'provided_transcript')
        self.assertEqual(result.providers_run, [])


if __name__ == '__main__':
    unittest.main()
