---
name: jetson-frame-extraction
description: Use when operation-manual candidate frame extraction is slow, when the user asks to use Jetson/NX/Orin devices for video frame extraction, or when comparing local CPU extraction with Jetson dual-worker extraction in video-analyzer.
---

# Jetson Frame Extraction

Use this skill when long-video candidate frame extraction is the bottleneck in
`video-analyzer` operation-manual runs.

## Default Policy

- Prefer Jetson offload for long videos instead of local OpenCV full-frame scanning.
- Use `agx,agx` as the default AGX dual worker pair. NX1-NX4 are manual override workers only.
- Use `--frame-extractor jetson` for strict Jetson mode; do not silently fall back to local extraction when the user explicitly wants Jetson.
- Use `--jetson-frame-backend ray` for the default AGX long-video path.
- For detailed operations and maintenance procedures, read `docs/JETSON_FRAME_WORKERS.md`.

## Operating Model

- The AGX Ray head can be kept warm; extraction jobs are still created per run.
- The local pipeline pushes `worker.py` on demand, caches the source video,
  runs both workers concurrently, pulls candidate frames back, and merges them.
- Human operators should normally use the local one-command script rather than
  SSHing into the Jetsons manually.

## Known Device State

- AGX is the default frame extraction device and runs two Ray frame workers by default.
- NX1-NX4 are not used by the default operation-manual path.

## Commands

Run the long-video fast script with Jetson workers:

```bash
cd /home/ai/github/video-analyzer
tools/pipelines/run_long_talk_fast_from_url.sh URL --keep-existing
```

Core CLI flags for manual runs:

```bash
--frame-extractor jetson \
--jetson-frame-hosts agx,agx \
--jetson-frame-backend ray \
--jetson-sample-fps 0.5 \
--jetson-chunk-overlap-seconds 2
```

Check Jetson prerequisites:

```bash
tools/check_jetson_frame_workers.sh
```

Manual check:

```bash
ssh -o HostKeyAlias=agx-lan agx@192.168.2.142 'command -v ffmpeg; python3 - << "PY"
import importlib.util
for name in ["cv2", "numpy", "PIL"]:
    print(name, bool(importlib.util.find_spec(name)))
PY'
```

## Benchmark Baseline

Validated on:

```text
downloads/url-videos/3W36pd50Wqw/video.mp4
duration: 3472.181 seconds
candidate budget: 232
```

Measured extraction-only comparison:

```text
AGX 1 worker, 10-minute window: 104.358s, 5.75x realtime
AGX 2 workers, 10-minute window: 85.966s, 6.98x realtime
AGX 2 workers, full 58-minute video: 467.121s, 7.43x realtime
```

Rule of thumb: keep AGX at two workers by default; increase worker count only
for explicit benchmarking.

## Verification

After a run, inspect:

```bash
jq '.metadata.timings, .metadata.frame_extraction, .metadata.frame_selection' RUN_DIR/analysis.json
```

Expected signs:

- `metadata.frame_extraction.backend == "jetson"`
- `metadata.frame_extraction.per_host` contains AGX workers only
- `metadata.frame_extraction.transport == "ray"`
- `metadata.frame_extraction.sample_fps == 0.5` for the long-talk fast wrapper
- `metadata.frame_selection.candidate_frames_count` matches the dynamic budget
- Fast mode keeps `metadata.frame_selection.vl_frames_count == 0`

## Public API

Use these CLI flags to connect the local pipeline to Jetson workers:

```bash
--frame-extractor local|jetson|auto
--jetson-frame-hosts agx,agx
--jetson-frame-backend auto|ssh|ray
--jetson-sample-fps auto|N
--jetson-chunk-overlap-seconds N
```

`jetson` is strict and should fail if a requested worker is unhealthy. `auto`
may fall back to local extraction. `ray` is the default AGX long-video transport.

## Failure Handling

- If strict Jetson mode fails health checks, fix the worker dependency instead of falling back to local.
- If a benchmark seems slow, distinguish:
  - remote extraction time: `metadata.frame_extraction.per_host[].timings.total_seconds`
  - end-to-end extraction time: `metadata.frame_extraction.total_seconds`
  The difference is sync, SSH orchestration, pullback, and merge overhead.
