---
name: jetson-frame-extraction
description: Use when operation-manual candidate frame extraction is slow, when the user asks to use Jetson/NX/Orin devices for video frame extraction, or when comparing local CPU extraction with Jetson dual-worker extraction in video-analyzer.
---

# Jetson Frame Extraction

Use this skill when long-video candidate frame extraction is the bottleneck in
`video-analyzer` operation-manual runs.

## Default Policy

- Prefer Jetson offload for long videos instead of local OpenCV full-frame scanning.
- Use `nx2,nx3` as the default dual worker pair.
- Use `--frame-extractor jetson` for strict Jetson mode; do not silently fall back to local extraction when the user explicitly wants Jetson.
- Use `--jetson-frame-backend auto` or `ssh`. Ray is only a future transport path here; the current validated path is SSH concurrent workers.
- `nx3` should be addressed as `nx3` only after verifying the SSH alias. If alias issues appear, use `nx@nx3.taild500c8.ts.net` with `ProxyCommand=none`.
- For detailed operations and maintenance procedures, read `docs/JETSON_FRAME_WORKERS.md`.

## Operating Model

- There is no persistent service to start on the NX devices.
- The local pipeline pushes `worker.py` on demand, caches the source video,
  runs both workers concurrently, pulls candidate frames back, and merges them.
- Human operators should normally use the local one-command script rather than
  SSHing into the Jetsons manually.

## Known Device State

- `nx2` has Jetson Linux R36.5, `ffmpeg`, Python OpenCV, GStreamer, `nvv4l2decoder`, and `nvjpegenc`.
- `nx3` was missing extraction dependencies and has been provisioned with:
  - `ffmpeg`
  - `python3-opencv`
  - `gstreamer1.0-tools`
  - `gstreamer1.0-plugins-good`
  - `gstreamer1.0-plugins-bad`
  - `gstreamer1.0-libav`
- Both workers currently use the SSH backend and `ffmpeg` decode path.

## Commands

Run the long-video fast script with Jetson workers:

```bash
cd /home/ivan/github/video-analyzer
OCR_CACHE=refresh tools/run_s36ri23_fast_full.sh
```

Core CLI flags for manual runs:

```bash
--frame-extractor jetson \
--jetson-frame-hosts nx2,nx3 \
--jetson-frame-backend auto \
--jetson-sample-fps auto \
--jetson-chunk-overlap-seconds 2
```

Check Jetson prerequisites:

```bash
tools/check_jetson_frame_workers.sh
```

Manual check:

```bash
ssh nx2 'command -v ffmpeg; python3 - << "PY"
import importlib.util
for name in ["cv2", "numpy", "PIL"]:
    print(name, bool(importlib.util.find_spec(name)))
PY'

ssh -o ProxyCommand=none nx@nx3.taild500c8.ts.net 'command -v ffmpeg; python3 - << "PY"
import importlib.util
for name in ["cv2", "numpy", "PIL"]:
    print(name, bool(importlib.util.find_spec(name)))
PY'
```

## Benchmark Baseline

Validated on:

```text
downloads/url-videos/S36ri23-l60/video.mp4
duration: 1386.121 seconds
candidate budget: 93
```

Measured extraction-only comparison:

```text
local CPU/OpenCV candidate_frame_extraction_seconds: 648.495s
Jetson dual worker first run: 149.467s
Jetson dual worker warm run: 105.583s
nx2 remote extraction: ~54.7s
nx3 remote extraction: ~55.9s
```

Rule of thumb: expect about `4x` speedup including first video sync and about
`6x` speedup after the video is cached on Jetsons.

## Verification

After a run, inspect:

```bash
jq '.metadata.timings, .metadata.frame_extraction, .metadata.frame_selection' RUN_DIR/analysis.json
```

Expected signs:

- `metadata.frame_extraction.backend == "jetson"`
- `metadata.frame_extraction.per_host` contains both `nx2` and `nx3`
- `metadata.frame_extraction.sample_fps == 1.0` for fast mode with `auto`
- `metadata.frame_selection.candidate_frames_count` matches the dynamic budget
- Fast mode keeps `metadata.frame_selection.vl_frames_count == 0`

## Public API

Use these CLI flags to connect the local pipeline to Jetson workers:

```bash
--frame-extractor local|jetson|auto
--jetson-frame-hosts nx2,nx3
--jetson-frame-backend auto|ssh|ray
--jetson-sample-fps auto|N
--jetson-chunk-overlap-seconds N
```

`jetson` is strict and should fail if a requested worker is unhealthy. `auto`
may fall back to local extraction. `ray` is reserved and currently uses the SSH
worker path.

## Failure Handling

- If strict Jetson mode fails health checks, fix the worker dependency instead of falling back to local.
- If `nx3` alias fails with `%` expansion or ProxyCommand errors, use direct host:
  `ssh -o ProxyCommand=none nx@nx3.taild500c8.ts.net`.
- If a benchmark seems slow, distinguish:
  - remote extraction time: `metadata.frame_extraction.per_host[].timings.total_seconds`
  - end-to-end extraction time: `metadata.frame_extraction.total_seconds`
  The difference is sync, SSH orchestration, pullback, and merge overhead.
