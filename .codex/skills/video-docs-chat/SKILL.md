---
name: video-docs-chat
description: Use when the user wants to ask questions or have a multi-turn conversation over generated video-analyzer documents from an operation-manual run directory. Supports single-question CLI use, interactive chat, and evidence-grounded summaries over operation_manual.md, transcript.md, manual_evidence.md, page_context, comments, and docs_analysis outputs. For audio narration, use the full Markdown narration workflow backed by audio-narration-script/Ivan TTS instead of the old 30-second AGX recap path.
---

# Video Docs Chat

Use this skill to answer questions over a completed `video-analyzer` operation-manual run directory.

## Inputs

The user should provide an operation-manual run directory, usually:

```bash
downloads/url-videos/BVxxxx/operation-manual
```

The directory may contain:

- `operation_manual.md`
- `transcript.md`
- `manual_evidence.md`
- `orin/page_context.md`
- `orin/comments.md`
- `docs_analysis/*.md`
- `analysis.json`

## Single question

Run:

```bash
tools/pipelines/ask_video_docs.sh RUN_DIR "QUESTION" --profile local_lan
```

Example:

```bash
tools/pipelines/ask_video_docs.sh downloads/url-videos/BVxxxx/operation-manual "这个视频的核心观点是什么？" --profile local_lan
```

## Multi-turn chat

Run:

```bash
tools/pipelines/chat_with_video_docs.sh RUN_DIR --profile local_lan
```

Then ask questions interactively. Use `/exit` to quit.

## Audio narration

When the user asks for `音频讲解`, `讲解稿`, `Markdown 转音频`, `生成 WAV`, `朗读音频`, or asks to turn one of the exported PDFs into audio, use the full narration workflow instead of the old short AGX recap path.

Run:

```bash
tools/generate_audio_narration.sh RUN_DIR --profile local_lan
```

The tool defaults to `--tts-concurrency 2` so the Ivan gateway can use both ready Qwen3-TTS workers. Lower it only when the user explicitly asks to keep TTS load minimal.

If the user names an exported PDF, pass that file or basename with `--source`; the tool maps `*.pdf` to the matching Markdown file before writing narration:

```bash
tools/pipelines/generate_audio_narration.sh RUN_DIR --source knowledge_notes_v2.pdf --profile local_lan
```

Outputs are written under `RUN_DIR/audio_narration/`:

- `narration_outline.md`
- `narration_script.md`
- `narration_script.txt`
- `audio_output/narration_full.wav`
- `audio_output/manifest.json`

Do not call `tools/generate_30s_agx_tts.sh` for audio narration requests. That script is now only a deprecated compatibility wrapper and forwards to `tools/pipelines/generate_audio_narration.sh`.

## Evidence policy

- Prefer `manual_evidence.md`, OCR, VL, and screenshots for visible UI or exact operations.
- Use `transcript.md` for spoken claims and timestamped explanations.
- Use `operation_manual.md` and `docs_analysis/*.md` as derived summaries.
- Use `orin/page_context.md` for page description and metadata.
- Treat `orin/comments.md` as low-confidence community supplement only.
- If evidence conflicts or is absent, say `需复核` instead of inventing.

## Reporting style

Answer in Chinese by default. Cite source filenames and timestamps/frame IDs when useful, for example `transcript.md 03:20` or `manual_evidence.md frame_012`.
