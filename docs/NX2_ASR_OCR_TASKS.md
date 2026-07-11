# NX2 ASR/OCR Task List

This task list implements the decisions in `NX2_ASR_OCR_MIGRATION.md`.
Each task must pass its acceptance criteria before the next service becomes a
candidate for the `video-analyzer` runtime profile.

## T0 - Prepare Isolated NX2 Runtime

Status: complete

Goal: reserve an isolated runtime location without changing existing `nx2`
services or the `ai` ASR/OCR deployment.

Deliverables:

- `/home/nx/github/video-analyzer-nx2-runtime/`
- `asr/`, `ocr/`, `models/`, `scripts/`, `tests/`, and `logs/`

Acceptance:

- No existing service is stopped or reconfigured.
- All later artifacts are kept below the isolated runtime directory.

## T1 - Provision Jetson ASR Base Environment

Status: pending

Goal: create an isolated ARM64 Python environment for SenseVoice/FunASR.

Work:

1. Select a JetPack R36.5-compatible PyTorch installation route.
2. Install only ASR dependencies into the isolated environment.
3. Verify CUDA is visible to PyTorch and FFmpeg can decode a WAV input.

Acceptance:

- `torch.cuda.is_available()` is true.
- The environment uses the Jetson GPU, not CPU fallback.
- No global Python package is modified.
- Startup and import logs are saved under `logs/asr/`.

## T2 - Run SenseVoice/FunASR ASR Smoke Test

Status: pending

Goal: verify SenseVoice-Small can transcribe Chinese audio on `nx2`.

Work:

1. Download or sync the required model into `models/asr/`.
2. Run a short Chinese audio file through ASR with VAD and punctuation.
3. Record elapsed time, peak memory, swap use, and transcript output.

Acceptance:

- The transcript is non-empty and usable.
- The process does not OOM or increase swap usage materially.
- The result records language, text, and timestamp-compatible segments.

## T3 - Add Local ASR HTTP Adapter

Status: pending

Goal: expose SenseVoice through the analyzer-compatible endpoint.

Work:

1. Implement `POST /api/asr/transcribe`.
2. Return `success`, `text`, `segments`, and `language`.
3. Add health reporting and request logging.
4. Keep the service single-model and idle-unloadable.

Acceptance:

- A multipart WAV request succeeds through the HTTP endpoint.
- The response is accepted by `video_analyzer.asr_providers.transcribe_with_http_asr`.
- A cold start and an idle reload both succeed.

## T4 - Validate ASR Against Real Operation-Manual Audio

Status: pending

Goal: establish whether SenseVoice is good enough for the target workflow.

Work:

1. Run a 10-minute Chinese tutorial.
2. Run a Chinese/English terminology sample.
3. Compare wording, timestamps, and terminology against the current
   VibeVoice result.

Acceptance:

- RTF and peak memory are recorded.
- No swap growth or OOM occurs.
- The comparison identifies whether SenseVoice can become the nx2 default or
  whether `faster-whisper` remains the local fallback.

## T5 - Provision Unlimited-OCR Single-Worker Environment

Status: pending

Goal: run the current Unlimited-OCR model on `nx2` without changing the model
or the analyzer's OCR protocol.

Work:

1. Reuse the validated Jetson PyTorch base where compatible.
2. Sync the current Unlimited-OCR model into `models/ocr/`.
3. Copy or adapt only the existing proxy and single worker scripts.
4. Set one GPU worker and `ocr_concurrency=1`.

Acceptance:

- The worker loads with CUDA and returns `/health`.
- No VibeVoice or ASR model is resident during OCR.
- Logs are stored under `logs/ocr/`.

## T6 - Validate OCR HTTP Contract And Real Frames

Status: pending

Goal: validate the existing OpenAI-compatible OCR integration on `nx2`.

Work:

1. Run a simple UI screenshot through `/v1/chat/completions`.
2. Run ten representative operation-manual frames.
3. Measure text quality, wall time, memory, and swap.

Acceptance:

- The response is consumable by the current OCR provider.
- All ten frames complete without OOM or swap growth.
- Results are compared against the `ai` Unlimited-OCR baseline.

## T7 - Integrate Only Passed Services

Status: pending

Goal: connect proven nx2 services to a local runtime profile.

Work:

1. Add host-local endpoint overrides only after T4 and T6 pass.
2. Configure ASR and OCR as mutually exclusive stages.
3. Run one end-to-end operation-manual smoke on nx2.

Acceptance:

- `config/config.json` remains host-local and uncommitted.
- The end-to-end run records which nx2 service handled ASR and OCR.
- `ai` remains unchanged and available as a rollback path.
