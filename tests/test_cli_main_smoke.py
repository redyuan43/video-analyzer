"""cli.main() 流水线冒烟测试。

纯 mock 边界：不触网、不占 GPU、不依赖本地模型服务。
覆盖最常回归的编排逻辑：resume-existing 转录复用、阶段跳过、产物落盘。
"""

import json
import sys
import unittest
import unittest.mock as mock
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analyzer import cli  # noqa: E402

TRANSCRIPT_MD = """# Transcript

- Language: zh

- [00:00:00 - 00:00:05] 大家好，这是一段测试转录。
- [00:00:05 - 00:00:10] 第二句用于验证解析。
"""


class _FakeTextClient:
    def chat(self, *args, **kwargs):
        return {"response": "# 操作手册（测试桩）\n\n1. 测试步骤。"}


def _write_base_config(config_dir: Path, output_dir: Path) -> None:
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "task": "operation_manual",
                "asr": {"provider": "none"},
                "ocr": {"provider": "none"},
                "vl": {"frame_policy": "none"},
                "clients": {"default": "openai"},
            }
        ),
        encoding="utf-8",
    )


def _run_main(argv: list[str]) -> None:
    with mock.patch.object(sys, "argv", ["cli"] + argv):
        cli.main()


def _common_patches():
    """关闭本地模型运行时与外部 client 构造。"""
    return [
        mock.patch.object(cli, "local_model_runtime_session", lambda *a, **k: _null_ctx()),
        mock.patch.object(cli, "local_model_stage", lambda *a, **k: _null_ctx()),
        mock.patch.object(cli, "create_client", mock.Mock(return_value=_FakeTextClient())),
        mock.patch.object(
            cli,
            "create_operation_manual_text_client",
            mock.Mock(return_value=_FakeTextClient()),
        ),
        mock.patch.object(
            cli,
            "create_operation_manual_fallback_client",
            mock.Mock(return_value=(None, None, None)),
        ),
    ]


import contextlib  # noqa: E402


def _null_ctx():
    return contextlib.nullcontext()


class CliMainSmokeTests(unittest.TestCase):
    def test_audio_only_operation_manual_run_writes_artifacts(self):
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config_dir = tmp_dir / "config"
            output_dir = tmp_dir / "out"
            config_dir.mkdir()
            _write_base_config(config_dir, output_dir)
            media = tmp_dir / "audio.m4a"
            media.write_bytes(b"\x00" * 64)
            transcript_file = tmp_dir / "transcript.md"
            transcript_file.write_text(TRANSCRIPT_MD, encoding="utf-8")

            gen = mock.Mock(
                return_value={
                    "response": "# 操作手册\n\n## 步骤一\n测试内容。",
                }
            )
            patches = _common_patches() + [
                mock.patch.object(cli, "media_has_video_stream", mock.Mock(return_value=False)),
                mock.patch.object(cli, "generate_operation_manual", gen),
            ]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                _run_main(
                    [
                        str(media),
                        "--config",
                        str(config_dir),
                        "--output",
                        str(output_dir),
                        "--asr-provider",
                        "none",
                        "--transcript-file",
                        str(transcript_file),
                        "--task",
                        "operation_manual",
                        "--vl-frame-policy",
                        "none",
                    ]
                )

            self.assertTrue((output_dir / "transcript.md").is_file())
            manual = output_dir / "operation_manual.md"
            self.assertTrue(manual.is_file(), f"manual missing; out={list(output_dir.glob('*'))}")
            content = manual.read_text(encoding="utf-8")
            self.assertIn("操作手册", content)
            progress_path = output_dir / "progress.json"
            self.assertTrue(
                progress_path.is_file(),
                f"progress.json missing; out={sorted(p.name for p in output_dir.glob('*'))}",
            )
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress.get("current_step"), "write")
            self.assertEqual(progress.get("status"), "succeeded")

            call = gen.call_args
            self.assertEqual(call.kwargs.get("transcript").text.splitlines()[0].strip() if call.kwargs.get("transcript") else "", "大家好，这是一段测试转录。")

    def test_resume_existing_reuses_output_transcript(self):
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config_dir = tmp_dir / "config"
            output_dir = tmp_dir / "out"
            config_dir.mkdir()
            output_dir.mkdir()
            _write_base_config(config_dir, output_dir)
            media = tmp_dir / "audio.m4a"
            media.write_bytes(b"\x00" * 64)
            (output_dir / "transcript.md").write_text(TRANSCRIPT_MD, encoding="utf-8")

            gen = mock.Mock(return_value={"response": "# 手册\n内容。"})
            read_mock = mock.Mock(wraps=cli.read_transcript_markdown)
            patches = _common_patches() + [
                mock.patch.object(cli, "media_has_video_stream", mock.Mock(return_value=False)),
                mock.patch.object(cli, "generate_operation_manual", gen),
                mock.patch.object(cli, "read_transcript_markdown", read_mock),
            ]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                _run_main(
                    [
                        str(media),
                        "--config",
                        str(config_dir),
                        "--output",
                        str(output_dir),
                        "--resume-existing",
                        "--task",
                        "operation_manual",
                        "--vl-frame-policy",
                        "none",
                    ]
                )

            # resume 模式应自动复用 output_dir/transcript.md
            read_mock.assert_called_once_with(output_dir / "transcript.md")
            self.assertTrue((output_dir / "operation_manual.md").is_file())


if __name__ == "__main__":
    unittest.main()
