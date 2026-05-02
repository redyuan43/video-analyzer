import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

import requests

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    AudioSegment = None

from .audio_processor import AudioProcessor, AudioTranscript

logger = logging.getLogger(__name__)

VIBEVOICE_PYTHONS = [
    Path("/home/ivan/github/VibeVoice/.venv-vvasr4bit/bin/python"),
    Path("/home/ivan/github/VibeVoice/.venv-rocm71/bin/python"),
]
VIBEVOICE_SCRIPT = Path("/home/ivan/github/VibeVoice/demo/vibevoice_asr_inference_from_file.py")
VIBEVOICE_MODEL = "microsoft/VibeVoice-ASR"
CAPSWRITER_URL = "http://127.0.0.1:8001/api/asr/transcribe"
REMOTE_ASR_URLS = [
    "http://edge.taild500c8.ts.net:8001/api/asr/transcribe",
    "http://spark-31d6.taild500c8.ts.net:8001/api/asr/transcribe",
]


def extract_audio_to_wav(video_path: Path, output_dir: Path) -> Optional[Path]:
    audio_path = output_dir / "audio.wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        return audio_path
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr.decode(errors="replace")
        if "Output file does not contain any stream" in error_output:
            return None
        if AudioSegment is None:
            raise RuntimeError("ffmpeg failed and pydub is not installed") from exc
        logger.info("Falling back to pydub for audio extraction")
        video = AudioSegment.from_file(str(video_path))
        audio = video.set_channels(1).set_frame_rate(16000)
        audio.export(str(audio_path), format="wav")
        return audio_path


def transcribe_with_http_asr(audio_path: Path, url: str, hotword: str = "") -> Optional[AudioTranscript]:
    try:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                url,
                files={"audio": (audio_path.name, audio_file, "audio/wav")},
                data={"hotword": hotword},
                timeout=900,
            )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            logger.warning("HTTP ASR failed from %s: %s", url, payload)
            return None
        return AudioTranscript(
            text=payload.get("text", ""),
            segments=payload.get("segments") or [],
            language=payload.get("language") or "unknown",
        )
    except Exception as exc:
        logger.warning("HTTP ASR unavailable from %s: %s", url, exc)
        return None


def transcribe_with_capswriter(audio_path: Path, hotword: str = "") -> Optional[AudioTranscript]:
    return transcribe_with_http_asr(audio_path, CAPSWRITER_URL, hotword=hotword)


def transcribe_with_remote_http(audio_path: Path, urls: Optional[list[str]] = None) -> Optional[AudioTranscript]:
    for url in urls or REMOTE_ASR_URLS:
        transcript = transcribe_with_http_asr(audio_path, url)
        if transcript and transcript.text.strip():
            transcript.segments = transcript.segments or [{"provider_url": url}]
            return transcript
    return None


def _vibevoice_python_candidates(config: Dict[str, str]) -> list[Path]:
    configured_python = config.get("python")
    if configured_python:
        return [Path(configured_python).expanduser()]
    return VIBEVOICE_PYTHONS


def transcribe_with_vibevoice(audio_path: Path, config: Optional[Dict[str, str]] = None) -> Optional[AudioTranscript]:
    config = config or {}
    python_candidates = [path for path in _vibevoice_python_candidates(config) if path.exists()]
    if not python_candidates or not VIBEVOICE_SCRIPT.exists():
        logger.warning("VibeVoice ASR environment is not available")
        return None

    failures = []
    for python_path in python_candidates:
        command = [
            str(python_path),
            str(VIBEVOICE_SCRIPT),
            "--model_path",
            config.get("model_path", VIBEVOICE_MODEL),
            "--audio_files",
            str(audio_path),
            "--batch_size",
            "1",
            "--temperature",
            "0",
            "--num_beams",
            "1",
            "--attn_implementation",
            config.get("attn_implementation", "auto"),
        ]
        if config.get("device"):
            command.extend(["--device", config["device"]])

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            failures.append(f"{python_path}: exit {exc.returncode}; stderr={stderr[-2000:]}; stdout={stdout[-1000:]}")
            logger.warning("VibeVoice ASR failed with %s: %s", python_path, failures[-1])
            continue
        except Exception as exc:
            failures.append(f"{python_path}: {exc}")
            logger.warning("VibeVoice ASR failed with %s: %s", python_path, exc)
            continue

        raw = result.stdout.strip()
        text = _extract_vibevoice_text(raw)
        if not text:
            text = raw
        return AudioTranscript(text=text, segments=[{"raw_output": raw}], language="unknown")

    logger.warning("All VibeVoice ASR candidates failed: %s", failures)
    return None


def _extract_vibevoice_text(raw: str) -> str:
    lines = []
    capture = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "--- Raw Output ---":
            capture = True
            continue
        if capture and stripped.startswith("--- Structured Output"):
            break
        if capture and stripped and not stripped.startswith("="):
            lines.append(stripped)
    return "\n".join(lines).strip()


def transcribe_with_provider(
    provider: str,
    audio_path: Path,
    language: str,
    whisper_model: str,
    device: str,
    vibevoice_config: Optional[Dict[str, str]] = None,
) -> Optional[AudioTranscript]:
    vibevoice_config = vibevoice_config or {}
    providers = ["remote_http", "vibevoice", "capswriter_http", "faster_whisper"] if provider == "auto" else [provider]
    for candidate in providers:
        if candidate == "none":
            return None
        if candidate == "remote_http":
            transcript = transcribe_with_remote_http(audio_path, vibevoice_config.get("remote_urls"))
        elif candidate == "capswriter_http":
            transcript = transcribe_with_capswriter(audio_path)
        elif candidate == "vibevoice":
            transcript = transcribe_with_vibevoice(audio_path, vibevoice_config)
        elif candidate == "faster_whisper":
            processor = AudioProcessor(language=language, model_size_or_path=whisper_model, device=device)
            transcript = processor.transcribe(audio_path)
        else:
            raise ValueError(f"Unknown ASR provider: {candidate}")

        if transcript and transcript.text.strip():
            logger.info("ASR succeeded with provider: %s", candidate)
            return transcript
        logger.info("ASR provider did not produce transcript: %s", candidate)
    return None
