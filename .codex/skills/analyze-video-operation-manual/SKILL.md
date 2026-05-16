---
name: analyze-video-operation-manual
description: Use when the user says to analyze a video, 分析视频, 生成视频操作手册, or gives a YouTube/Bilibili URL or local video path and wants the full video-analyzer operation-manual pipeline run end-to-end. Downloads URL videos, captures page context including description/subtitles/selected comments, uses Spark/Edge VibeVoice ASR, Spark/Edge DotsMOCR OCR, AMD Fast VL/text models, and returns operation_manual.md plus analysis.json.
---

# Analyze Video Operation Manual

Run the `video-analyzer` operation-manual pipeline end to end from either an online video URL or a local video file.

## Default Runtime Policy

- URL input: use `tools/run_operation_manual_from_url.sh`.
- URL context: default to `page_context.md`, which combines description,
  metadata, subtitles, and selected comments.
- Local file input: call `.venv/bin/python -m video_analyzer.cli` directly.
- Runtime profile: use `local_lan` from `video_analyzer/config/default_config.json`; put machine-specific overrides in `config/config.json`.
- The profile owns ASR URL, OCR URL, AMD Fast URL, vision model, text model, subtitle languages, comment budget, and multi-doc defaults.
- Use the project VibeVoice HTTP endpoint for required ASR. Spark and Edge expose lazy `8012` services; a cold first request can take several minutes while the backend loads.
- Use Spark/Edge DotsMOCR on `:8000/v1` for OCR. A short `/v1/models` timeout can be lazy cold start rather than OCR failure.
- Use the project AMD Fast OpenAI-compatible endpoint for text and vision unless the user explicitly asks for a different backend.
- Do not use AGX/Qwen3-ASR unless the user explicitly asks for a fast ASR endpoint.

## Workflow

1. Identify the input:
   - If it starts with `http://` or `https://`, treat it as a URL.
   - Otherwise treat it as a local video path.
2. Ensure required local services are reachable when practical:
   - `curl http://spark-31d6.taild500c8.ts.net:8012/api/health`
   - `curl http://spark-31d6.taild500c8.ts.net:8000/v1/models`
   - `curl http://edgexpert-4353.taild500c8.ts.net:8000/v1/models`
   - AMD Fast at `http://100.90.114.26:18081/v1`
3. For URL input, run:

   ```bash
   tools/run_operation_manual_from_url.sh "URL" --profile local_lan
   ```

   Useful options:

   ```bash
   tools/run_operation_manual_from_url.sh "URL" --cookies-from-browser chrome
   tools/run_operation_manual_from_url.sh "URL" --max-frames 48
   tools/run_operation_manual_from_url.sh "URL" --download-only
   tools/run_operation_manual_from_url.sh "URL" --no-include-comments
   tools/run_operation_manual_from_url.sh "URL" --subtitle-langs zh-CN,zh-Hans,zh,en
   ```

   After the manual run, optional multi-document analysis can be generated with:

   ```bash
   tools/run_multidoc_analysis.sh RUN_DIR --profile local_lan
   ```

4. For local video input, create or reuse a context file if provided by the user, then run:

   ```bash
   .venv/bin/python -m video_analyzer.cli VIDEO.mp4 \
     --task operation_manual \
     --output OUTPUT_DIR \
     --context-file CONTEXT.md \
     --asr-provider vibevoice \
     --vibevoice-url http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe \
     --ocr-provider auto \
     --ocr-base-url http://spark-31d6.taild500c8.ts.net:8000/v1 \
     --ocr-base-url http://edgexpert-4353.taild500c8.ts.net:8000/v1 \
     --llm-base-url http://100.90.114.26:18081/v1 \
     --vision-base-url http://100.90.114.26:18081/v1 \
     --text-base-url http://100.90.114.26:18081/v1 \
     --vision-model 'hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive' \
     --text-model 'hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive' \
     --max-frames 24 \
     --keep-frames \
     --log-level INFO
   ```

5. Verify completion before reporting:
   - `analysis.json` exists.
   - `operation_manual.md` or `operation_manual.quality_failed.md` exists.
   - URL runs have `page_context.md`; `analysis.json.metadata.page_context`
     records subtitle/comment success or diagnostics.
   - `operation_manual.quality_review` has no errors, or report the quality-failed path clearly.
   - `ocr_events` count matches extracted frames and most/all statuses are `ok`.
   - `visual_events` are non-empty.
   - `asr.providers_run` includes `vibevoice`, not Qwen3-ASR by accident.

## Failure Recovery

- If a URL run has already downloaded the video and extracted `audio.wav` but fails with `Required ASR transcript was not produced`, treat it as ASR failure, not OCR failure.
- Retry only VibeVoice ASR against the existing `audio.wav`, with proxy variables cleared for LAN/Tailscale endpoints, and write `transcript.md` with `video_analyzer.artifacts.write_transcript_markdown()`.
- Do not redownload the video or rerun OCR/frame/VL work that already exists. Once `transcript.md` exists, resume with `--transcript-file` so ASR is skipped.
- During VibeVoice cold start, no output for a few minutes can be normal. Confirm progress by checking `vibevoice-asr-backend.service` logs or GPU memory usage before aborting.
- If DotsMOCR `:8000/v1/models` times out briefly, check `dots-mocr-lazy-proxy.service`, `/proxy/health`, and `docker start dots-mocr-vllm` on Spark before declaring OCR unavailable.

## Reporting

Return the manual path, analysis path, ASR provider/elapsed time, OCR success count, visual frame count, and any quality warnings. Keep it concise.

## Evidence Policy

Use evidence in this order: OCR/VL frame evidence, author subtitles, VibeVoice
ASR, automatic subtitles, page description/metadata, pinned or uploader
comments, ordinary comments. Comments are low-confidence; keep comment-only
information in community supplements or FAQ, not main deterministic steps.
