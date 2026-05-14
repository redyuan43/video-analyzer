# Jetson Frame Workers

This project can offload operation-manual candidate frame extraction to Jetson
NX workers. The validated worker pair is `nx2,nx3`.

## Operating Model

The Jetson frame workers are **not** a long-running service. There is no daemon
to start or stop on the NX devices.

The local pipeline does the orchestration on demand:

1. installs or refreshes `~/.cache/video-analyzer/frame-worker/worker.py` on each
   Jetson host;
2. caches the source video under `~/.cache/video-analyzer/frame-worker/videos/`;
3. splits the video into overlapping chunks;
4. runs both chunks concurrently over SSH;
5. pulls selected candidate frames back and merges them locally.

This keeps manual operation simple: run one local command and inspect
`analysis.json`.

## Human One-Command Usage

For the current long sample:

```bash
cd /home/ivan/github/video-analyzer
OCR_CACHE=refresh tools/run_s36ri23_fast_full.sh
```

The script already includes:

```bash
--frame-extractor jetson
--jetson-frame-hosts nx2,nx3
--jetson-frame-backend auto
--jetson-sample-fps auto
--jetson-chunk-overlap-seconds 2
```

Use `OCR_CACHE=on` for normal reruns. Use `OCR_CACHE=refresh` when measuring
fresh OCR performance.

## API / CLI Contract

Public CLI flags:

```bash
--frame-extractor local|jetson|auto
--jetson-frame-hosts nx2,nx3
--jetson-frame-backend auto|ssh|ray
--jetson-sample-fps auto|N
--jetson-chunk-overlap-seconds N
```

Behavior:

- `local`: use the original local OpenCV extraction path.
- `jetson`: strict Jetson mode. If any requested worker fails health checks,
  fail loudly instead of falling back to local.
- `auto`: try Jetson first, then fall back to local if Jetson orchestration
  fails.
- `ssh`: current validated transport. It runs one SSH command per worker
  concurrently.
- `ray`: reserved for a future Ray transport. Current implementation logs a
  warning and uses the SSH worker path.
- `jetson-sample-fps auto`: maps to `fast=1`, `balanced=2`, `deep=3`.

Output metadata:

```bash
jq '.metadata.frame_extraction' RUN_DIR/analysis.json
```

Important fields:

- `backend`: expected `jetson`
- `transport`: expected `ssh`
- `hosts`: requested hosts
- `health`: per-host tool/decode readiness
- `per_host`: chunk range, decode backend, preview frame count, candidate count,
  and remote extraction timing
- `total_seconds`: end-to-end extraction orchestration time, including sync,
  worker execution, pullback, and merge

## Health Check

Run:

```bash
tools/check_jetson_frame_workers.sh
```

Expected minimum:

- both `nx2` and `nx3` reachable;
- `ffmpeg` present;
- Python can import `cv2`, `numpy`, and `PIL`;
- `rsync` present.

Known current state:

- `nx2`: has NVIDIA GStreamer decode/encode plugins and OpenCV.
- `nx3`: provisioned with `ffmpeg`, `python3-opencv`, and GStreamer plugins;
  uses `ffmpeg` decode path.

If `ssh nx3` fails because of local SSH alias/proxy expansion, use:

```bash
ssh -o ProxyCommand=none nx@nx3.taild500c8.ts.net 'hostname'
```

## Performance Baseline

Validated source:

```text
downloads/url-videos/S36ri23-l60/video.mp4
duration: 1386.121 seconds
candidate budget: 93
```

Measured extraction-only results:

```text
local CPU/OpenCV extraction: 648.495s
Jetson dual worker first run: 149.467s
Jetson dual worker warm run: 105.583s
nx2 remote extraction: ~54.7s
nx3 remote extraction: ~55.9s
```

Interpretation:

- first run includes video sync to both Jetsons;
- warm run uses cached video on each Jetson;
- per-host timings show real remote extraction work;
- total time also includes SSH orchestration, candidate pullback, and local merge.

## Maintenance

The worker code is generated from `video_analyzer/jetson_frames.py` and pushed on
demand. To update worker behavior, change the repo code locally; the next run
refreshes `worker.py` on each Jetson automatically.

Remote cache locations:

```text
~/.cache/video-analyzer/frame-worker/worker.py
~/.cache/video-analyzer/frame-worker/videos/
~/.cache/video-analyzer/frame-worker/runs/
```

To reclaim Jetson disk space:

```bash
ssh nx2 'rm -rf ~/.cache/video-analyzer/frame-worker/runs/*'
ssh -o ProxyCommand=none nx@nx3.taild500c8.ts.net 'rm -rf ~/.cache/video-analyzer/frame-worker/runs/*'
```

Do not delete `videos/` during performance comparisons unless you intentionally
want to measure cold sync time again.
