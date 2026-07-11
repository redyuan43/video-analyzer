# NX2 ASR/OCR Migration Decisions

## Scope

Move the video-analyzer runtime to `nx2` incrementally. Validate ASR and OCR
as isolated services before changing a runtime profile or starting the unified
UI on `nx2`.

This document records the decisions made on 2026-07-11 and the acceptance
gates for each capability.

## Target Selection

`nx2` is the selected Jetson target because its observed system memory use was
about 1.1 GiB of 15.6 GiB, with no swap in use. `nx1`, `nx3`, and `nx4` had
higher active memory use at the same check.

The target is an ARM64 Jetson Orin NX running JetPack/L4T R36.5. It has one
shared-memory GPU. ASR and OCR must not be loaded concurrently.

## ASR Decision

- Do not migrate the existing VibeVoice P40 service to `nx2`.
  Its model files are about 17 GiB and the existing runtime depends on the
  P40/Pascal-specific vLLM stack. This exceeds the usable memory budget of
  the 16 GiB Orin NX before runtime and KV-cache allocations.
- Use SenseVoice-Small through FunASR as the first local ASR candidate.
- Expose a small service compatible with the analyzer's existing
  `POST /api/asr/transcribe` contract:
  `success`, `text`, `segments`, and `language`.
- Use `remote_http` while validating the service so the analyzer does not need
  a new ASR provider immediately.
- Do not introduce `faster-whisper` as an NX2 fallback. Keep VibeVoice on `ai`
  as the production-quality rollback path while SenseVoice is evaluated.

ASR acceptance gates:

1. The service starts in an isolated `nx2` environment without changing global
   Python packages.
2. A short Chinese audio request returns the expected response schema.
3. A 10-minute Chinese tutorial produces usable text, timestamps, and a
   measured RTF without OOM or swap growth.
4. A mixed Chinese/English terminology sample is compared against the current
   VibeVoice output before making it the default. Current result: SenseVoice
   passes the runtime and timestamp gates, but terminology accuracy is not
   sufficient to replace VibeVoice as the default.

## OCR Decision

- Keep Unlimited-OCR as the first OCR candidate.
- Do not migrate DotsMOCR: its P40/vLLM runtime is harder to reproduce and it
  is not the current project default.
- Run exactly one Unlimited-OCR worker on `nx2`.
- Keep `ocr_concurrency=1` and use the existing OpenAI-compatible endpoint
  contract on port `18088`.
- Preserve the current lazy idle unload behavior.

OCR acceptance gates:

1. The model loads with Jetson-compatible PyTorch and Transformers in an
   isolated environment.
2. A simple UI screenshot returns correct text through `/v1/chat/completions`.
3. A 10-frame operation-manual sample completes without OOM or swap growth.
4. Results and wall time are compared with the current `ai` Unlimited-OCR
   baseline before enabling an `nx2` endpoint in a runtime profile.

Validation result on 2026-07-11: rejected on NX2. The 6.78 GiB model completed
loading in the isolated environment, but the first simple-image inference
failed in the vision encoder with `NvMapMemAllocInternalTagged ... error 12`
and a PyTorch CUDA caching-allocator assertion. Swap grew from 2 MiB to
435 MiB. Do not attempt the 10-frame or HTTP acceptance gates with
Unlimited-OCR on the 16 GiB Orin NX.

The next NX2 OCR candidate must use a substantially smaller runtime. It must
be evaluated as a new decision; DotsMOCR is not an appropriate fallback
because it has a larger, P40/vLLM-oriented deployment footprint.

## Runtime Rules

- ASR and OCR are mutually exclusive stages on `nx2`.
- Do not edit or commit machine-specific endpoints. Keep validated `nx2`
  endpoint overrides in that host's local `config/config.json`.
- Do not alter existing `ai` ASR/OCR services while validating `nx2`.
- Switch the analyzer only after each service independently passes its gates.
