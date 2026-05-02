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

Run from the repository root:

```bash
.venv/bin/python -m video_analyzer.cli VIDEO.mp4 \
  --task operation_manual \
  --output output/manual-run \
  --context-file optional-page-description.md \
  --asr-provider remote_http \
  --ocr-provider auto \
  --max-frames 24 \
  --keep-frames \
  --log-level INFO
```

Recommended current local model setup:

- Vision / VL model: `sayanything-hauhaucs-aggressive@?`
- Text/manual model: `redhatai_qwen3.6-35b-a3b-nvfp4`
- LLM endpoint: `http://127.0.0.1:1234/v1`
- Remote ASR endpoint: `http://edge.taild500c8.ts.net:8001/api/asr/transcribe`
- OCR order: spark DotsMOCR vLLM first, then other shared endpoints, then local
  LM Studio vision OCR fallback

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

The user-facing manual should be written as a readable "overview -> illustrated
steps -> checks/caveats" document. Full screenshot dumps belong in
`manual_evidence.md`, not in the main manual.

## How The Video Is Split

The tool does not analyze every source frame. A 30 fps, 165 second video has
roughly 4,956 frames, which is too expensive to send to a vision model.

Instead, `operation_manual` uses screen-recording keyframe extraction:

1. **Decode and sample**
   - OpenCV is tried first.
   - If OpenCV cannot open or decode useful frames, ffmpeg extracts preview
     frames. This handles AV1 and other codecs that OpenCV may not support.

2. **Keep coverage**
   - The extractor preserves broad timeline coverage so long videos do not lose
     entire sections.
   - The density-budget selector keeps at least representative frames across
     coarse time buckets.

3. **Spend remaining budget on information density**
   - Candidate frames are scored by visual change.
   - After coverage frames are reserved, the remaining `--max-frames` budget is
     assigned to high-change frames.
   - This means dense sections, such as UI transitions, code blocks, tables, or
     slide changes, get more screenshots than static sections.

4. **Remove near duplicates**
   - Very similar neighboring frames are skipped.
   - This avoids wasting model calls on unchanged screens.

`--max-frames` is a budget, not the number of original video frames. For a short
operation video, use 24-40. For long or dense tutorials, use 60-120 if latency
and model cost are acceptable.

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
- Use LM Studio vision OCR fallback only when shared OCR endpoints are down.
- Keep OCR for operation manuals because exact text is high-value evidence.
- Reduce total cost by using density-based frame selection instead of uniformly
  increasing `--max-frames`.

## ASR Policy

ASR provider order for manual generation should prefer remote or stronger
models:

1. `remote_http`: remote ASR service on edge/spark when reachable.
2. `vibevoice`: local VibeVoice script fallback.
3. `capswriter_http`: local HTTP fallback.
4. `faster_whisper`: final local fallback.

Use `--asr-provider remote_http` when you specifically want the remote service.
Use `--asr-provider auto` when you want the fallback chain.

## Current Real Sample Command

The Bilibili sample used during validation:

```bash
.venv/bin/python -m video_analyzer.cli \
  downloads/test-videos/BV12moMBrELB/video.mp4 \
  --task operation_manual \
  --output downloads/test-videos/BV12moMBrELB/run-final-av1-remoteasr-sayanything-24f \
  --context-file downloads/test-videos/BV12moMBrELB/description.md \
  --asr-provider remote_http \
  --ocr-provider auto \
  --max-frames 24 \
  --keep-frames \
  --log-level INFO
```

Validated characteristics:

- source video codec: AV1
- source frames: about 4,956
- extracted keyframes: 24
- ASR: remote HTTP success
- OCR events: 24 successful events
- user manual: `operation_manual.md`
- full evidence: `manual_evidence.md`

## AI Agent Checklist

When an agent is asked to generate a manual:

1. Confirm LM Studio `/v1/models` has the configured vision and text models.
2. Confirm remote ASR endpoint health if using `remote_http`.
3. Run the CLI with `--task operation_manual`.
4. Use a page/context file when the video page description contains important
   URLs, install commands, or release notes.
5. Inspect `analysis.json` metadata for frame count, ASR success, OCR statuses,
   and model names.
6. Inspect `operation_manual.md` for step-level images, not just text.
7. Inspect `manual_evidence.md` only when verifying uncertain claims.
8. If the manual is text-heavy or screenshots are appended as a gallery, fix the
   generation/post-processing before reporting completion.

## Quality Notes

The best manual is not a transcript summary. It should:

- describe the video's structure first
- use screenshots near the exact step they support
- include commands and parameters exactly
- mark conflicts between OCR, ASR, and visual analysis as "需复核"
- include a short flowchart when the video has a clear process
- avoid dumping every frame into the user-facing document
