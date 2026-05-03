---
name: video-docs-chat
description: Use when the user wants to ask questions or have a multi-turn conversation over generated video-analyzer documents from an operation-manual run directory. Supports single-question CLI use, interactive chat, evidence-grounded summaries, and 30-second spoken recap audio via AGX local TTS over operation_manual.md, transcript.md, manual_evidence.md, page_context, comments, and docs_analysis outputs.
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
tools/ask_video_docs.sh RUN_DIR "QUESTION" --profile local_lan
```

Example:

```bash
tools/ask_video_docs.sh downloads/url-videos/BVxxxx/operation-manual "这个视频的核心观点是什么？" --profile local_lan
```

## Multi-turn chat

Run:

```bash
tools/chat_with_video_docs.sh RUN_DIR --profile local_lan
```

Then ask questions interactively. Use `/exit` to quit.

## 30-second spoken recap with AGX TTS

When the user asks for a `30秒口播`, `口播音频`, `TTS`, or asks for audio from the video docs, produce a concise Chinese spoken recap and synthesize it with the AGX local TTS service by default.

Default assumptions:

- Target host: SSH alias `agx`.
- TTS service: CapsWriter/Qwen3-TTS on AGX at `http://127.0.0.1:8002`.
- Default speaker: `vivian` unless the user specifies another voice.
- Output location: the provided `RUN_DIR`, with filenames ending in `_agx` to show provenance.
- Do not use online TTS or local non-AGX fallback unless AGX is unreachable or unavailable; if falling back, say so explicitly.

Workflow:

1. Draft a natural spoken recap from the evidence files, not a written abstract. Keep it short enough for about 30 seconds in the AGX voice; if AGX output is long, shorten the script first.
2. Check AGX reachability and health:

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=8 agx \
     'hostname; uname -m; curl -s --max-time 10 http://127.0.0.1:8002/api/health'
   ```

3. If `tts_model_loaded` is not `true`, load and poll:

   ```bash
   ssh -o BatchMode=yes agx '
     curl -s -X POST http://127.0.0.1:8002/api/tts/load
     for i in $(seq 1 30); do
       h=$(curl -s http://127.0.0.1:8002/api/health)
       loaded=$(printf "%s" "$h" | grep -o "\"tts_model_loaded\":[^,}]*" | grep -o "true\|false")
       workers=$(printf "%s" "$h" | grep -o "\"tts_parallel_workers_ready\":[0-9]*" | grep -o "[0-9]*$" || true)
       [ "$loaded" = true ] && [ "${workers:-0}" -gt 0 ] && exit 0
       sleep 3
     done
     exit 1
   '
   ```

4. Synthesize on AGX and copy the WAV back to `RUN_DIR`:

   ```bash
   RUN_DIR="downloads/url-videos/BVxxxx/operation-manual"
   OUT_BASENAME="video_30s_summary_agx"
   TEXT="这里填入约30秒中文口播稿"

   printf '%s' "$TEXT" | ssh -o BatchMode=yes agx 'cat > /tmp/video_30s_summary_agx.txt'

   ssh -o BatchMode=yes agx 'bash -s' <<'REMOTE'
   set -euo pipefail
   OUT=/tmp/video_30s_summary_agx.wav
   PAYLOAD=$(python3 -c 'import json, pathlib; text = pathlib.Path("/tmp/video_30s_summary_agx.txt").read_text(encoding="utf-8"); print(json.dumps({"text": text, "speaker": "vivian", "speed": 1.0}, ensure_ascii=False))')
   printf '%s' "$PAYLOAD" | curl -sS -X POST http://127.0.0.1:8002/api/tts/speak \
     -H 'Content-Type: application/json' \
     --data-binary @- \
     -o "$OUT"
   file "$OUT"
   REMOTE

   scp -q agx:/tmp/video_30s_summary_agx.wav "$RUN_DIR/$OUT_BASENAME.wav"
   ```

5. Verify duration locally:

   ```bash
   ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$RUN_DIR/$OUT_BASENAME.wav"
   ```

6. If the duration is outside roughly 27-33 seconds, shorten or expand the recap and regenerate. Use `ffmpeg atempo` only as a final polish when the wording is already correct and the user wants a strict duration.

## Evidence policy

- Prefer `manual_evidence.md`, OCR, VL, and screenshots for visible UI or exact operations.
- Use `transcript.md` for spoken claims and timestamped explanations.
- Use `operation_manual.md` and `docs_analysis/*.md` as derived summaries.
- Use `orin/page_context.md` for page description and metadata.
- Treat `orin/comments.md` as low-confidence community supplement only.
- If evidence conflicts or is absent, say `需复核` instead of inventing.

## Reporting style

Answer in Chinese by default. Cite source filenames and timestamps/frame IDs when useful, for example `transcript.md 03:20` or `manual_evidence.md frame_012`.
