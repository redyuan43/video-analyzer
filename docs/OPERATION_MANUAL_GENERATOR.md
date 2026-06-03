# Operation Manual Generator

This guide is for users and AI agents that need to turn an installation,
operation, tutorial, or screen-recording video into a readable illustrated
manual.

## What It Is Good For

The `operation_manual` task works best for videos with visible procedures:

- software installation and setup walkthroughs
- IDE, terminal, browser, SaaS, or app operation videos
- plugin or workflow demos
- product feature tutorials
- lecture-style videos that show slides, commands, diagrams, or UI text

It is less reliable for videos where the main information is not visible on
screen:

- talking-head videos with few visual anchors
- fast-cut entertainment footage
- videos where important details are only implied, not spoken or shown
- low-resolution recordings with unreadable UI text
- videos requiring frame-perfect physical action analysis

For mixed videos, the manual should mark uncertain details rather than invent
missing steps.

## Quick Start

For YouTube, Bilibili, or any URL supported by `yt-dlp`, run from the
repository root:

```bash
tools/run_operation_manual_from_url.sh "https://www.bilibili.com/video/BVxxxx"
```

For Bilibili, prefer the canonical video URL without share parameters:

```bash
./start_example.sh https://www.bilibili.com/video/BVxxxx/
```

If you paste a full browser/share URL containing `&`, quote it. Otherwise bash
treats `&` as "run in background" before the script can see the full URL:

```bash
./start_example.sh 'https://www.bilibili.com/video/BVxxxx/?share_source=copy_web&vd_source=...'
```

The one-command runner downloads the video, saves the page metadata and
description to `description.md`, collects subtitles/comments when available,
builds `page_context.md`, then runs the full operation-manual pipeline.
Defaults come from the `spark` runtime profile in
`video_analyzer/config/default_config.json`. Put local overrides in
`config/config.json` rather than editing scripts.

Useful URL-runner variants:

```bash
# Use browser cookies for Bilibili/YouTube login or age-gated content.
tools/run_operation_manual_from_url.sh "URL" --cookies-from-browser chrome

# Choose pipeline depth. balanced is the default.
tools/run_operation_manual_from_url.sh "URL" --pipeline-mode fast
tools/run_operation_manual_from_url.sh "URL" --pipeline-mode balanced
tools/run_operation_manual_from_url.sh "URL" --pipeline-mode deep

# Override dynamic frame budgets only when you need a hard cap or fixed pool.
tools/run_operation_manual_from_url.sh "URL" --candidate-frames auto --max-vl-frames 80

# Use two DotsMOCR endpoints and keep OCR cache enabled.
tools/run_operation_manual_from_url.sh "URL" \
  --ocr-base-url http://spark-31d6.taild500c8.ts.net:8000/v1 \
  --ocr-base-url http://edge.taild500c8.ts.net:8000/v1 \
  --ocr-concurrency auto \
  --ocr-cache on

# Offload candidate frame extraction to the AGX Ray dual worker path.
.venv/bin/python -m video_analyzer.cli VIDEO.mp4 \
  --task operation_manual \
  --pipeline-mode fast \
  --frame-extractor jetson \
  --jetson-frame-hosts agx,agx \
  --jetson-frame-backend ray \
  --jetson-sample-fps 0.5

# Only download video and page context.
tools/run_operation_manual_from_url.sh "URL" --download-only

# Disable low-trust community comments, or tune the comment budget.
tools/run_operation_manual_from_url.sh "URL" --no-include-comments
tools/run_operation_manual_from_url.sh "URL" --max-comments 10

# Override subtitle language priority.
tools/run_operation_manual_from_url.sh "URL" --subtitle-langs zh-CN,zh-Hans,zh,en
```

To switch or customize endpoints/models, use a runtime profile:

```json
{
  "active_runtime_profile": "deepseek_v4_pro",
  "runtime_profiles": {
    "deepseek_v4_pro": {
      "llm_base_url": "https://api.deepseek.com",
      "vision_base_url": "http://100.96.79.21:18082/v1",
      "text_base_url": "https://api.deepseek.com",
      "vision_model": "minicpm-v-4.5-v100",
      "text_model": "deepseek-v4-pro",
      "vibevoice_url": "http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe",
      "ocr_base_url": "http://spark-31d6.taild500c8.ts.net:8000/v1",
      "ocr_base_urls": [
        "http://spark-31d6.taild500c8.ts.net:8000/v1",
        "http://edge.taild500c8.ts.net:8000/v1"
      ],
      "ocr_concurrency": "auto",
      "ocr_cache": "on",
      "download_device": "local",
      "max_comments": 3000,
      "subtitle_langs": "zh-CN,zh-Hans,zh,en"
    }
  }
}
```

Both URL and multi-document runners accept `--profile deepseek_v4_pro`. Command-line
arguments still override the profile for one-off runs.

For URL downloads, `--download-device local` is the default. Use
`--download-device mi` when the local host cannot download the video; the MI
device runs `yt-dlp`, then the downloaded video, metadata, subtitles, and
comments are synced back before the normal analysis stages continue.

### Remote Runtime Installation Notes

For URL runs, install both the Python package and URL downloader in the project
virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install yt-dlp
sudo apt-get update && sudo apt-get install -y ffmpeg
```

`tools/run_operation_manual_from_url.sh` adds `.venv/bin` to `PATH` before
starting Python, so commands such as `yt-dlp` are found even when the shell has
not manually activated the virtual environment.

When the LLM endpoint is a shared remote LM Studio server with limited VRAM,
only configure models that are already loaded on that server. Verify with the
LM Studio model API before running:

```bash
curl http://HOST:1234/api/v0/models
```

Use a model whose `state` is `loaded` for both `vision_model` and `text_model`
unless there is enough free VRAM for a deliberate model switch. Do not point a
runtime profile at an unloaded large model during normal operation; that can
force LM Studio to load it and exhaust the remote machine.

For an existing local video:

```bash
.venv/bin/python -m video_analyzer.cli VIDEO.mp4 \
  --task operation_manual \
  --output output/manual-run \
  --context-file optional-page-description.md \
  --asr-strategy balanced \
  --ocr-provider auto \
  --pipeline-mode balanced \
  --candidate-frames auto \
  --min-vl-frames auto \
  --max-vl-frames auto \
  --vl-context-before 3 \
  --vl-context-after 2 \
  --keep-frames \
  --log-level INFO
```

Recommended current local model setup:

- Vision / VL model, text model, LLM endpoints, VibeVoice URL, OCR URL, subtitle
  languages, and comment budget are managed by the active runtime profile.
- The default visual frame analysis path uses Ivan MiniCPM-V-4.5:
  `http://100.96.79.21:18082/v1`, model `minicpm-v-4.5-v100`, with context
  `40960`. The final manual text path continues to use AMD Fast:
  `http://100.90.114.26:18081/v1`, model
  `hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`.
- MiniCPM is used for visual understanding only. OCR remains the hard evidence
  source for UI text, commands, filenames, labels, and parameters.
- ASR strategy: `balanced` by default for operation manuals. The default path
  stays on Spark services; if Spark ASR/VibeVoice is unavailable, fix that
  service instead of silently falling back to AGX or the caller machine.
- VibeVoice endpoint: spark
  `http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe`.
- OCR order: spark/Edge DotsMOCR vLLM first, then AMD Fast OpenAI-compatible
  vision OCR fallback with
  `hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`.

Current OCR deployment:

| Field | Value |
|-------|-------|
| Device | `spark-31d6` |
| GPU | NVIDIA GB10 |
| Container | `dots-mocr-vllm` |
| Image | `vllm/vllm-openai:v0.17.1-cu130` |
| Model path | `/workspace/dots.mocr/weights/DotsMOCR` |
| API base URL | `http://spark-31d6.taild500c8.ts.net:8000/v1` |
| Served model name | `model` |
| max_model_len | `16384` |

The local machine is only the caller for OCR. The DotsMOCR model itself runs on
the remote spark device. Local Gradio on `127.0.0.1:7861` is not required for the
manual generator.

The same command also works on AV1 videos. If OpenCV cannot decode the video,
the extractor automatically falls back to ffmpeg.

## Outputs

The output directory contains:

- `operation_manual.md`: the user-facing illustrated manual.
- `manual_evidence.md`: the full frame/OCR/vision evidence index for review.
- `analysis.json`: structured transcript, OCR events, visual events, metadata,
  uncertainties, and manual response.
- `manual_assets/`: screenshots embedded inside the manual.
- `frames/`: extracted keyframes when `--keep-frames` is used.
- `audio.wav`: extracted audio when `--keep-frames` is used.

For URL runs, the download directory also contains page context artifacts:

- `description.md`: the original page metadata and description from `yt-dlp`.
- `subtitles/`: raw subtitle files plus cleaned timestamped text when subtitles are available.
- `comments.json`: selected raw comment records.
- `comments.md`: cleaned low-trust comment notes.
- `page_context.md`: the final context evidence package passed through `--context-file`.
- `page_context.json`: diagnostics and metadata recorded into `analysis.json.metadata.page_context`.

Subtitle and comment collection are best-effort. Missing subtitles, blocked
comments, insufficient cookies, or platform limits are recorded in
`page_context.md` diagnostics and do not stop the video/OCR/ASR/VL pipeline.

The user-facing manual should be written as a readable "overview -> illustrated
steps -> checks/caveats" document. Full screenshot dumps belong in
`manual_evidence.md`, not in the main manual.

## How The Video Is Split

The tool does not analyze every source frame. A 30 fps, 165 second video has
roughly 4,956 frames, which is too expensive to send to a vision model.

Instead, `operation_manual` uses the staged OCR keyframe funnel documented in
`docs/VIDEO_OCR_KEYFRAME_STRATEGY.md`:

```text
low-resolution scan frames
-> textness/change OCR candidates
-> high-resolution OCR keyframes
-> deduplicated OCR text events
-> small MiniCPM/VL explanation set
```

1. **Decode and scan**
   - OpenCV is tried first.
   - If OpenCV cannot open or decode useful frames, ffmpeg extracts preview
     frames. This handles AV1 and other codecs that OpenCV may not support.
   - Jetson/Ray workers should be preferred for long videos so the scan can use
     hardware decode and low-resolution previews.

2. **Size the candidate frame pool**
   - `--candidate-frames auto` scales with video duration and pipeline mode.
   - Short videos keep a smaller candidate pool to avoid over-analysis.
   - Long videos naturally expand to hundreds of candidate frames when needed;
     a 1-hour balanced run is not constrained to 12 or 24 frames.
   - `--max-frames` remains as a legacy explicit cap for the candidate pool.

3. **Keep coverage**
   - The extractor preserves broad timeline coverage so long videos do not lose
     entire sections.
   - The density-budget selector keeps at least representative frames across
     coarse time buckets.

4. **Spend remaining budget on information density**
   - Candidate frames are scored by visual change and textness.
   - After coverage frames are reserved, the remaining candidate budget is
     assigned to high-change frames.
   - This means dense sections, such as UI transitions, code blocks, tables, or
     slide changes, get more screenshots than static sections.

5. **Remove near duplicates**
   - Very similar neighboring frames are skipped.
   - This avoids wasting model calls on unchanged screens.

6. **Select OCR keyframes**
   - `--ocr-keyframe-strategy auto|scan-text|legacy` controls OCR frame
     selection.
   - All normal URL operation-manual runs default to `scan-text`, regardless of
     `fast`, `balanced`, `deep`, or `long-talk-fast`. `auto` is backward
     compatible but should not be the default in production profiles.
   - `--ocr-keyframe-budget auto|N` controls the actual number of frames sent to
     DotsMOCR.
   - OCR text is deduplicated into text events before VL selection.

7. **Select VL frames**
   - `--pipeline-mode fast` skips VL but still keeps dynamic candidate frames
     and OCR evidence.
   - `--pipeline-mode balanced` sends a dynamic subset to VL, based on OCR text
     density, ASR chapter density, visual change, and time coverage.
   - `--pipeline-mode deep` sends the whole dynamic candidate pool to VL.
   - `--vl-frame-policy all|none` can force the VL pass on or off.
   - `--vl-concurrency N` controls how many selected VL frame requests may run
     at the same time.
   - `--vl-context-before 3 --vl-context-after 2` sends nearby candidate frames
     as multi-image context for each selected VL frame.
   - `--vl-context-max-gap auto` uses the median candidate-frame interval times
     3, clamped to 8-45 seconds, so context does not cross obvious time breaks.

`analysis.json` records `metadata.ocr_keyframes` with scan frame count, OCR
candidate count, actual OCR frame count, OCR text event count, per-frame OCR
selection reasons, and text-event dedupe results. It also records
`metadata.frame_selection` with candidate/VL counts and per-frame VL selection
reasons. Skipped VL frames are still present in `visual_events` with
`status: "skipped"`, an OCR summary, and the selection score. Runtime breakdowns
are written to `metadata.timings`; `metadata.vl_context` records the
before/after window and resolved time-gap threshold. OCR endpoint selection,
cache mode, cache hit counts, and effective worker count are written to
`metadata.ocr`.

Candidate frame extraction can run locally or on Jetson workers. For long
teaching/talk videos, the default Jetson path is the AGX Ray dual worker setup
(`agx,agx`); NX1-NX4 remain manual override workers and are not part of the
default path. With `--frame-extractor jetson`, the video is cached on each
listed Jetson host, split into overlapping chunks, processed in parallel, then
merged locally with the same global density/coverage budget. Worker health,
per-host timings, sample FPS, and transport are written to
`metadata.frame_extraction`.

## OCR vs Vision Model Efficiency

OCR and VL solve different parts of the job:

- **OCR** extracts exact visible text: commands, filenames, labels, URLs,
  parameters, button names, and small captions.
- **VL/frame analysis** explains what the screen means: workflow state, layout,
  before/after relationships, and visual context.
- **Text/manual generation** combines ASR, OCR, page context, and visual analysis
  into a readable manual.

Dedicated OCR is usually faster and more accurate for text-heavy UI frames than
a general vision-language model. However, if the pipeline runs both OCR and VL
for every frame, total runtime is slower than VL-only.

Recommended policy:

- Use spark DotsMOCR when available. It is preferred for exact screen text.
- Pass `--ocr-base-url` more than once, or set `ocr_base_urls`, to distribute
  DotsMOCR frames across multiple healthy endpoints. `--ocr-concurrency auto`
  means one in-flight OCR request per endpoint.
- OCR cache is enabled by default. Use `--ocr-cache refresh` to recompute and
  rewrite cached entries, or `--ocr-cache off` for a cold measurement.
- Use the AMD Fast OpenAI-compatible vision OCR fallback only when DotsMOCR is
  down.
- Keep OCR for operation manuals because exact text is high-value evidence.
- Reduce total cost by selecting OCR keyframes and then using the balanced
  dynamic VL selector instead of uniformly increasing frame caps.

## ASR Policy

Manual generation uses strategy-level ASR by default:

- `--asr-strategy fast`: run remote HTTP ASR only. Use this for quick
  iteration and smoke tests when a fast ASR endpoint was explicitly configured.
- `--asr-strategy balanced`: use VibeVoice when no fast transcript exists, when
  the audio is long, or when the fast transcript looks too weak. This is the
  default for `operation_manual`. If VibeVoice succeeds, the tool does not fall
  back to Qwen3-ASR/CapsWriter just to manufacture a second transcript.
- `--asr-strategy deep`: run VibeVoice and, when `--remote-asr-url` is
  configured, also run remote HTTP ASR for timestamp anchoring. Use this for
  final manuals where long-audio terminology and chapter consistency matter.

When explicitly configured, remote HTTP ASR is treated as the timestamp anchor.
VibeVoice is treated as the long-context semantic pass: it helps correct
terminology, infer chapter structure, and resolve audio-related uncertainties.
When both are available, the merged transcript keeps remote HTTP timestamps and
uses VibeVoice text for higher-quality wording and terminology.

VibeVoice is remote-GPU only. The tool tries configured
`vibevoice.deep_remote_urls`, such as a spark/edge GPU service, and never starts
a local VibeVoice subprocess. This prevents accidental long CPU or inefficient
local ROCm runs.

Balanced mode is VibeVoice-first by default. If VibeVoice fails to produce text,
the tool may still fall back to CapsWriter HTTP or `faster_whisper` so a manual
can be generated with an explicit ASR warning. For a strict no-Qwen/no-CapsWriter
run, force `--asr-provider vibevoice`.

There are two different kinds of VibeVoice parallelism:

- **Remote service chunking:** the VibeVoice HTTP wrapper can call VibeVoice's
  native meeting workflow on one machine. Keep
  `VIBEVOICE_CHUNK_PARALLEL_WORKERS=1` on a single GB10-class device unless you
  have verified memory headroom. Setting it to `2` starts two model-loading
  worker processes on the same host and can overload the machine.
- **Distributed workers:** for long audio and multiple `--vibevoice-url`
  endpoints, `video-analyzer` splits the audio timeline across endpoints, for
  example spark handles one slice and edge handles another. Each remote request
  uses `use_native_chunking=false` so every machine runs only one VibeVoice
  model instance.

Example remote VibeVoice override:

```bash
--asr-strategy deep \
--vibevoice-url http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe
```

Add `--remote-asr-url ...` only when you intentionally want a fast timestamp
anchor in addition to VibeVoice.

Use `--asr-provider remote_http`, `--asr-provider vibevoice`, or another
provider only when you want to force one provider and bypass strategy fusion.

## Page Context Evidence Policy

URL runs pass `page_context.md` to the manual generator instead of the older
single-purpose `description.md`. The generator should weigh evidence in this
order:

- OCR/VL frame evidence: highest confidence for visible operations.
- Author subtitles: high-confidence speech/timeline evidence.
- VibeVoice ASR: high-confidence semantic audio evidence.
- Automatic subtitles: useful but may contain recognition errors.
- Page description and metadata: contextual evidence.
- Pinned or uploader comments: low-confidence supplemental evidence.
- Ordinary comments: lowest-confidence community notes.

When subtitles and ASR conflict, mark the claim as `需复核` unless OCR/VL
evidence clearly resolves it. Comment-only information belongs in
`社区补充/常见问题`; it should not create deterministic operation steps,
commands, or parameters unless supported by video, OCR, ASR, subtitles, or page
description.

## Current Real Sample Command

The Bilibili sample used during validation:

```bash
.venv/bin/python -m video_analyzer.cli \
  downloads/test-videos/BV12moMBrELB/video.mp4 \
  --task operation_manual \
  --output downloads/test-videos/BV12moMBrELB/run-adaptive-vibevoice-deep \
  --context-file downloads/test-videos/BV12moMBrELB/description.md \
  --asr-strategy deep \
  --ocr-provider auto \
  --pipeline-mode deep \
  --candidate-frames auto \
  --keep-frames \
  --log-level INFO
```

Validated characteristics:

- source video codec: AV1
- source frames: about 4,956
- extracted keyframes: 24
- ASR: `deep` mode runs VibeVoice for long-context terminology and chapter
  consistency; it uses remote HTTP timestamps only when `--remote-asr-url` is
  configured
- OCR events: 24 successful events
- user manual: `operation_manual.md`
- full evidence: `manual_evidence.md`

## AI Agent Checklist

When an agent is asked to generate a manual:

1. Confirm LM Studio `/v1/models` has the configured vision and text models.
2. Confirm remote ASR endpoint health if using `remote_http`.
3. Run the CLI with `--task operation_manual`.
4. Use a page/context file when the video page description contains important
   URLs, install commands, release notes, subtitles, or useful uploader comments.
5. For URL runs, prefer `page_context.md` over `description.md`.
6. Inspect `analysis.json` metadata for frame count, ASR success, OCR statuses,
   and model names.
7. Inspect `analysis.json.metadata.page_context` for subtitle/comment diagnostics.
8. Inspect `operation_manual.md` for step-level images, not just text.
9. Inspect `manual_evidence.md` only when verifying uncertain claims.
10. If the manual is text-heavy or screenshots are appended as a gallery, fix the
   generation/post-processing before reporting completion.

## Quality Notes

The best manual is not a transcript summary. It should:

- describe the video's structure first
- use screenshots near the exact step they support
- include commands and parameters exactly
- mark conflicts between OCR, ASR, and visual analysis as "需复核"
- include a short flowchart when the video has a clear process
- avoid dumping every frame into the user-facing document
