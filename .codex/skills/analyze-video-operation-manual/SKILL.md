---
name: analyze-video-operation-manual
description: Use when the user says to analyze a video, 分析视频, 生成视频操作手册, or gives a YouTube/Bilibili URL or local video path and wants the full video-analyzer operation-manual pipeline run end-to-end. Downloads URL videos, captures page description, uses edge VibeVoice ASR, spark DotsMOCR OCR, LM Studio VL/text models, and returns operation_manual.md plus analysis.json.
---

# Analyze Video Operation Manual

Run the `video-analyzer` operation-manual pipeline end to end from either an online video URL or a local video file.

## Default Runtime Policy

- URL input: use `tools/run_operation_manual_from_url.sh`.
- Local file input: call `.venv/bin/python -m video_analyzer.cli` directly.
- ASR: VibeVoice on edge only by default: `http://192.168.100.236:8003/api/asr/transcribe`.
- OCR: DotsMOCR on spark: `http://192.168.100.169:8000/v1`.
- LM Studio: `http://127.0.0.1:1234/v1`.
- Vision model: `sayanything-hauhaucs-aggressive@?`.
- Text model: `redhatai_qwen3.6-35b-a3b-nvfp4`.
- Do not start VibeVoice on spark for normal 7-8 minute videos; spark is reserved for OCR unless the user explicitly asks otherwise.
- Do not use AGX/Qwen3-ASR unless the user explicitly asks for a fast ASR endpoint.

## Workflow

1. Identify the input:
   - If it starts with `http://` or `https://`, treat it as a URL.
   - Otherwise treat it as a local video path.
2. Ensure required local services are reachable when practical:
   - `curl http://192.168.100.236:8003/api/health`
   - `curl http://192.168.100.169:8000/v1/models`
   - LM Studio at `http://127.0.0.1:1234/v1`
3. For URL input, run:

   ```bash
   tools/run_operation_manual_from_url.sh "URL"
   ```

   Useful options:

   ```bash
   tools/run_operation_manual_from_url.sh "URL" --cookies-from-browser chrome
   tools/run_operation_manual_from_url.sh "URL" --max-frames 48
   tools/run_operation_manual_from_url.sh "URL" --download-only
   ```

4. For local video input, create or reuse a context file if provided by the user, then run:

   ```bash
   .venv/bin/python -m video_analyzer.cli VIDEO.mp4 \
     --task operation_manual \
     --output OUTPUT_DIR \
     --context-file CONTEXT.md \
     --asr-provider vibevoice \
     --vibevoice-url http://192.168.100.236:8003/api/asr/transcribe \
     --ocr-provider auto \
     --ocr-base-url http://192.168.100.169:8000/v1 \
     --llm-base-url http://127.0.0.1:1234/v1 \
     --vision-model 'sayanything-hauhaucs-aggressive@?' \
     --text-model redhatai_qwen3.6-35b-a3b-nvfp4 \
     --max-frames 24 \
     --keep-frames \
     --log-level INFO
   ```

5. Verify completion before reporting:
   - `analysis.json` exists.
   - `operation_manual.md` or `operation_manual.quality_failed.md` exists.
   - `operation_manual.quality_review` has no errors, or report the quality-failed path clearly.
   - `ocr_events` count matches extracted frames and most/all statuses are `ok`.
   - `visual_events` are non-empty.
   - `asr.providers_run` includes `vibevoice`, not Qwen3-ASR by accident.

## Reporting

Return the manual path, analysis path, ASR provider/elapsed time, OCR success count, visual frame count, and any quality warnings. Keep it concise.

