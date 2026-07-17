# NX2 ASR/OCR Validation Report

Date: 2026-07-11

## Environment

- Target: Jetson Orin NX (`nx2`), JetPack/L4T R36.5, CUDA 12.6, 16 GiB shared
  memory.
- Runtime root: `/home/nx/github/video-analyzer-nx2-runtime/`.
- ASR and OCR were loaded sequentially only. No `ai` ASR/OCR service was
  stopped, reconfigured, or used as an execution backend during these tests.

## Storage

| Item | NX2 disk use | Result |
| --- | ---: | --- |
| SenseVoice, VAD, and punctuation models | about 1.2 GiB | Validated |
| Unlimited-OCR model | 6.78 GiB | Cannot infer on NX2 |
| EasyOCR Chinese and English models | 101 MiB | Validated |
| OCR Python environment | about 2.0 GiB | Validated |

## ASR: SenseVoice/FunASR

The validated runtime uses CUDA SenseVoice-Small with FSMN VAD, punctuation,
and native timestamp output from the current FunASR implementation.

| Sample | Duration | Inference | RTF | Peak RSS | Timestamp tokens | VibeVoice character similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed Chinese/English operation-manual clip | 215.667s | 15.298s | 0.0709 | 5.76 GiB | 1,380 | 0.898 |
| Chinese interview clip | 900s | 54.851s | 0.0609 | 5.39 GiB | 5,226 | 0.914 |

The timestamp ranges were strictly ordered and reached the end of each audio
sample. The local HTTP adapter was also validated through
`video_analyzer.asr_providers.transcribe_with_http_asr`:

- Endpoint: `POST /api/asr/transcribe`
- Response: `success`, `text`, `segments`, and `language`
- Short operation-manual request: 1,567 text characters and 58 timestamped
  segments.
- Idle unload: model unloaded after the configured 2-second validation window;
  a later request cold-starts it again.

Verdict: runtime and endpoint compatibility pass. Proper nouns and mixed
Chinese/English terminology are visibly weaker than VibeVoice, so this is an
on-demand NX2 fallback, not the default production ASR.

## OCR: Unlimited-OCR

Unlimited-OCR was tested with its documented lower-cost `gundam` single-image
configuration: `base_size=1024`, `image_size=640`, and `crop_mode=True`.

- Model load completed.
- The first simple text image failed in the SAM vision encoder with
  `NvMapMemAllocInternalTagged ... error 12` and a PyTorch CUDA caching
  allocator assertion.
- Swap grew from 2 MiB to about 435 MiB before process exit.

Verdict: rejected on the 16 GiB Orin NX. The failure occurs before text
generation, so lowering generation token limits does not address it.

## OCR: EasyOCR

EasyOCR was validated with CUDA and `ch_sim,en`.

| Test | Result | Timing | Peak RSS | Swap |
| --- | --- | ---: | ---: | --- |
| Simple text image | Exact expected text | 25.985s cold load, 0.796s inference | 1.32 GiB | No growth |
| Ten real operation-manual frames, direct reader | 10/10 complete | 24.840s total, 2.484s/frame | 2.82 GiB | No growth |
| Ten real frames, project HTTP client | 10/10 complete | 19.047s total, 1.506s median, 4.223s max | Service stayed within NX2 budget | No new growth |

The temporary adapter passed both `/v1/models` and `/v1/chat/completions`.
It returns OpenAI-compatible JSON text items, and the existing project OCR
client consumed all ten responses successfully.

Quality observations:

- English UI text and large Chinese headings are generally usable.
- Small Chinese UI text, diagram labels, and product names have substantial
  recognition errors.
- The correct role is sparse frame-text evidence extraction, not a
  quality-equivalent replacement for Unlimited-OCR or DotsMOCR.

## Integration Decision

The validated NX2 services are intentionally not configured as the default
runtime profile:

- SenseVoice is fast and timestamped but fails the terminology-quality bar for
  default operation-manual ASR.
- EasyOCR is low-memory and API-compatible but fails the fine-text and
  structured-document quality bar for default OCR.

For a full production-quality operation-manual run, keep VibeVoice and the
existing `ai` OCR path as the default. Use NX2 SenseVoice and EasyOCR only as
explicit, host-local fallback services where their documented quality limits
are acceptable.

## Sequential Fallback Smoke

On 2026-07-12, an uncommitted NX2-local `nx2_fallback` profile was loaded by
the project configuration system. It defined:

- SenseVoice: `http://127.0.0.1:18013/api/asr/transcribe`
- EasyOCR: `http://127.0.0.1:18089/v1`
- OCR concurrency: `1`

The services were started and stopped sequentially:

1. The project HTTP ASR provider transcribed the 215.667-second real
   operation-manual sample in 21.833 seconds, returning 1,567 characters and
   58 timestamped segments.
2. The ASR service stopped. The project OCR provider then processed ten real
   operation-manual frames in 18.997 seconds, with all ten requests
   succeeding.
3. Both temporary services were confirmed stopped. NX2 returned to about
   11 GiB available memory, and no `ai` service was modified.

This demonstrates the intended mutually exclusive fallback integration. It is
not a replacement-quality full publishing run because both NX2 model choices
remain below the production-quality gates described above.
