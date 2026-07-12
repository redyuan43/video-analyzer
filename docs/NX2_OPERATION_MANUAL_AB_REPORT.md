# NX2 Operation-Manual A/B Report

Date: 2026-07-12

## Scope

This report compares the existing `ai` baseline with a complete NX2 fallback
core-analysis run for the same 215.667-second Bilibili video:

- Baseline: `downloads/url-videos/BV1DxJH6YEyU/operation-manual-001/`
- NX2 A/B run: `downloads/url-videos/BV1DxJH6YEyU/operation-manual-nx2-deep-compare/`
- Identical extracted-frame set: 29 frames with timestamps recovered from the
  baseline `analysis.json`.
- Identical operation-manual text model: `deepseek-v4-pro`.
- NX2 model sequence: SenseVoice ASR -> EasyOCR -> MiniCPM-V 4.5.
  The services were stopped between stages and never co-resident.

The NX2 run resumed from the same local video, page context, and extracted
frames. Download and frame extraction were intentionally excluded from the
direct A/B because they do not depend on the ASR/OCR/VL model choice.

## Runtime Decision

| Capability | NX2 choice | Decision |
| --- | --- | --- |
| ASR | SenseVoice/FunASR | Technically usable, but not default-quality |
| OCR | EasyOCR `ch_sim,en` | Technically usable, but not default-quality |
| VL | MiniCPM-V 4.5 Q4_K_M with GGUF projector | Technically usable, but not default-quality |
| Unlimited-OCR | Unlimited-OCR | Rejected: inference fails on 16 GiB Orin NX |

MiniCPM-V required `llama-server --reasoning off`. Without this, the model put
all generated text in `reasoning_content`, leaving the OpenAI-compatible
`content` field empty. The project already sends `/no_think`; server-side
reasoning disable is also required for a reliable endpoint contract.

## Coverage And Output

| Check | AI baseline | NX2 deep A/B |
| --- | ---: | ---: |
| Candidate frames | 29 | 29 |
| OCR success events | 18 | 29 |
| Frames selected for VL | 29 | 29 |
| Non-empty VL analyses | 29 | 29 |
| Operation-manual quality gate | Passed, one warning | Passed, no warnings |
| Operation-manual response characters | 4,195 | 4,025 |
| VL response characters | 43,252 | 37,708 |
| Final PDFs | 4 | 4 |

The initial NX2 `balanced` run selected only 13 visual frames. It is retained
as a throughput reference only. The final A/B uses `--pipeline-mode deep`,
which resolves the visual policy to `all`, matching the AI baseline's 29/29
visual coverage.

## Timing

| Stage | AI baseline | NX2 | Relative result |
| --- | ---: | ---: | --- |
| ASR | 142.214s | 22.637s | NX2 6.3x faster |
| OCR | 108.514s | 37.057s | NX2 2.9x faster |
| VL | 134.275s | 1,296.205s | NX2 9.7x slower |
| Operation manual | 36.215s | 31.849s | Comparable |
| Core model stages, excluding extraction and service cold start | 421.218s | 1,387.748s | NX2 about 3.3x slower |

Adding the baseline's 52.660-second candidate-frame extraction gives an
estimated NX2 full-run total of 1,440.408 seconds, about 3.0x the observed
AI baseline total. The NX2 VL server also has an approximately 55-second cold
load before it becomes healthy; it was kept separate from the request-stage
timings above.

The bottleneck is VL. The NX2 MiniCPM-V service held about 7.2-7.7 GiB RSS and
generated at about 8-13 tokens per second with the project's frame-analysis
prompt. After stopping it, NX2 returned to about 11 GiB available memory.

## Quality Findings

The NX2 visual model correctly recovered the key structure of the video:

- GitHub `obra/superpowers` and
  `superpowers/skills/brainstorming/SKILL.md`.
- The `Brainstorming`, `Goal: 实现`, `Goal: 验证`, and `人工验证` flow.
- `Agent Sprite Forge Skill`, `GPT Image`, and `2d-tank-v2`.
- The final workflow of design, Goal-driven implementation, iteration, and
  human validation.

The generated operation manual has the same high-level workflow and all core
steps as the AI baseline. It is shorter and less detailed in visual
descriptions. EasyOCR produces more text than the baseline OCR path but has
clear Chinese-character, UI-label, and proper-noun errors; MiniCPM-V often
compensates for these errors, but it also makes occasional unsupported
interaction inferences such as treating a static cursor as active hovering.

## Final Recommendation

NX2 is a valid on-demand fallback host for a complete operation-manual run:
the full core analysis, multidocument analysis, and four final PDFs all
completed successfully with ASR/OCR/VL running sequentially.

Do not migrate the default production-quality path from `ai` to NX2:

- The full deep run is about 3x slower after including frame extraction.
- VL alone is about 9.7x slower.
- SenseVoice terminology quality and EasyOCR small-text quality are visibly
  below the current VibeVoice and `ai` OCR paths.
- MiniCPM-V is adequate for workflow-level scene understanding, but it is
  less detailed and more prone to unsupported UI-action claims.

Use NX2 when `ai` is unavailable, when low-priority asynchronous processing is
acceptable, or when the user explicitly accepts the documented quality loss.
Keep `ai` VibeVoice, current OCR, and V100 MiniCPM-V as the normal
operation-manual runtime.
