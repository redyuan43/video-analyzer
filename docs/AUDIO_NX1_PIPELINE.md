# Audio NX1 Pipeline

## Responsibility Boundary

- AI runs the complete `audio_nx1` pipeline: ASR, speaker diarization and
  alignment, prompt-template selection, summarization, and artifact generation.
- NX1 owns the source audio, mirrored intermediate artifacts, final artifacts,
  retention, and a read-only monitoring interface.
- NX1 must not run, tune, reorder, or fall back to local model stages.
- Video jobs keep using the `core` resource. Full `audio_nx1` jobs use the
  serial `audio-analysis` resource.
- Transcription-only compatibility jobs use the separate `asr` resource.

## Compatibility Contract

- New full audio jobs use `pipeline_profile=audio_nx1`.
- Legacy `pipeline_kind=analysis` requests are accepted and normalized to
  `audio_nx1`.
- `transcription` remains available only for compatibility with a split
  transcription workflow.
- The runtime/model profile remains separate from the pipeline profile.
- The prompt-template count is read from the active catalog. The pipeline does
  not depend on a hard-coded template count.

## AI API

- Upload a full audio job: `POST /api/mobile/audio-jobs/upload`
- Upload an existing transcript: `POST /api/mobile/audio-jobs/from-transcript`
- List jobs: `GET /api/mobile/audio-jobs`
- Read one job: `GET /api/mobile/audio-jobs/<job_id>`
- Read template metadata: `GET /api/mobile/audio-templates`
- Download result resources through the paths returned in `result_resources`.
- Acknowledge a mirrored result: `POST /api/mobile/audio-jobs/<job_id>/ack`

All mobile audio requests use `X-Audio-Pipeline-Token`. The AI service reads
the expected value from `VIDEO_ANALYZER_AUDIO_PIPELINE_TOKEN`.

## NX1 Deployment Checklist

1. Configure the same non-empty pipeline token on AI and NX1.
2. Point NX1 uploads and polling to the AI Video Analyzer endpoint.
3. Disable NX1 processing workers while keeping storage and monitoring online.
4. Mirror every returned resource into the NX1 attempt directory.
5. Acknowledge the AI job only after NX1 verifies the mirrored files.
6. Keep local audio work queued when AI local model resources are occupied.
7. Enable cloud fallback only after cloud ASR and diarization endpoints are
   configured and validated; never use NX1 as a model fallback.

Service restart, NX1 worker shutdown, token changes, and endpoint cutover are
deployment operations and require explicit confirmation before execution.
