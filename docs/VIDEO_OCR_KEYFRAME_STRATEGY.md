# Video OCR Keyframe Strategy

This project must treat long-video OCR as a staged evidence funnel, not as a
small fixed-frame sample. The required flow is:

```text
low-resolution scan frames
-> textness/change OCR candidates
-> high-resolution OCR keyframes
-> deduplicated OCR text events
-> small MiniCPM/VL explanation set
```

## Non-Negotiable Rules

- Do not OCR every source video frame.
- Do not rely on a fixed 24 or 48 frame cap for long videos.
- Do not use MiniCPM/VL as the primary mechanism for finding visible text.
- Every selected OCR frame must have a machine-readable selection reason.
- Every run must record separate counts for scan frames, OCR candidate frames,
  actual OCR frames, OCR text events, and VL frames.

## Selection Policy

The first pass scans cheap preview frames with hardware decode when available.
For long videos this should usually be Jetson/Ray with NVDEC at about `0.5fps`
for talk videos and higher rates for dense screen recordings. The scan stage
keeps frames that have one or more of:

- visual change from nearby frames;
- likely text regions (`textness`);
- timeline coverage value;
- proximity to ASR/subtitle segment boundaries.

The OCR keyframe selector then chooses a bounded set of high-resolution frames.
The budget is dynamic and grows with duration and pipeline mode. It must keep
coverage before spending the remaining budget on textness and change density.

## OCR And VL Responsibilities

OCR is hard evidence for exact visible text: commands, paths, filenames, labels,
URLs, button names, errors, parameters, and code. DotsMOCR remains the preferred
OCR provider.

VL/frame analysis explains visual context: workflow state, relationships between
screens, layout, and what changed. MiniCPM/VL should receive only selected frames
after OCR evidence exists. It must not be used to replace OCR discovery.

## Required Metadata

`analysis.json` must include `metadata.ocr_keyframes` with at least:

- `strategy` and `strategy_resolved`
- `scan_frames_count`
- `ocr_candidate_frames_count`
- `ocr_frames_count`
- `ocr_text_events_count`
- per-frame decisions under `frames`
- deduplicated text events under `text_events`

The web status UI should surface:

- scanned preview frames;
- OCR candidate frames;
- actual OCR frames;
- unique OCR text events;
- VL frames.

## Current Implementation Notes

The implementation lives in `video_analyzer/ocr_keyframes.py` and is wired into
`video_analyzer.cli` before `run_ocr`. URL runs forward the public flags:

```bash
--ocr-keyframe-strategy auto|scan-text|legacy
--ocr-keyframe-budget auto|N
--ocr-scan-sample-fps auto|N
```

`legacy` exists only for debugging and regression isolation. Normal operation
must use `scan-text` for every operation-manual URL run. `auto` is accepted for
backward compatibility and currently resolves to the same text-aware scan
policy; it should not be used as the default because it makes the intended
pipeline ambiguous in logs and reviews.
