# Jetson Frame Workers

This project can offload operation-manual candidate frame extraction to Jetson
devices. For long teaching/talk videos, the current default is one AGX device
split into two Ray frame workers: `agx,agx`.

## Operating Model

The frame extraction worker itself is pushed on demand. The AGX Ray head may be
kept warm, but extraction jobs are still created per run.

The local pipeline does the orchestration on demand:

1. installs or refreshes `~/.cache/video-analyzer/frame-worker/worker.py` on each
   Jetson host;
2. caches the source video under `~/.cache/video-analyzer/frame-worker/videos/`;
3. splits the video into overlapping chunks;
4. runs all chunks concurrently over SSH/Ray;
5. uses visual change plus lightweight textness to select OCR candidates;
6. pulls selected candidate frames back and merges them locally.

This keeps manual operation simple: run one local command and inspect
`analysis.json`.

## Human One-Command Usage

For current long teaching/talk samples:

```bash
cd /home/ai/github/video-analyzer
tools/pipelines/run_long_talk_fast_from_url.sh URL --keep-existing
```

The script already includes:

```bash
--frame-extractor jetson
--jetson-frame-hosts agx,agx
--jetson-frame-backend ray
--jetson-sample-fps 0.5
--jetson-chunk-overlap-seconds 2
```

Start or refresh the AGX Ray head with:

```bash
tools/start_jetson_frame_ray.sh
```

The default Ray resource shape is `frame_worker=2` on AGX. The startup script
uses the control host `agx` first, then resolves the current private LAN address
from AGX itself for Ray's `--node-ip-address`. Do not hard-code transient DHCP
addresses in the default path. The video-link status launcher probes
`agx-lan,agx.local,ubuntu.local` by default and exports
`JETSON_AGX_LAN_HOST` only when a candidate resolves and accepts
`ssh agx@<name>`. Override candidates with
`VIDEO_LINK_AGX_LAN_HOST_CANDIDATES`, or set `JETSON_AGX_LAN_HOST=agx-lan`
after adding a stable LAN DNS/DHCP hostname for AGX. Use
`JETSON_RAY_HEAD_FRAME_WORKERS=N` only for explicit benchmarking.

## API / CLI Contract

Public CLI flags:

```bash
--frame-extractor local|jetson|auto
--jetson-frame-hosts agx,agx
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
- `ssh`: runs one SSH command per worker concurrently.
- `ray`: current validated long-talk transport for AGX.
- `jetson-sample-fps auto`: maps to `fast=1`, `balanced=2`, `deep=3`; the
  long-talk fast script intentionally overrides this to `0.5`.
- Long-video OCR should still use the OCR keyframe funnel in
  `docs/VIDEO_OCR_KEYFRAME_STRATEGY.md`; Jetson workers provide the cheap
  preview scan and high-resolution candidate materialization.

Output metadata:

```bash
jq '.metadata.frame_extraction' RUN_DIR/analysis.json
```

Important fields:

- `backend`: expected `jetson`
- `transport`: expected `ray` for the long-talk AGX path
- `hosts`: requested hosts
- `health`: per-host tool/decode readiness
- `per_host`: chunk range, decode backend, preview frame count, candidate count,
  and remote extraction timing
- OCR keyframe counts are recorded later under `metadata.ocr_keyframes`.
- `total_seconds`: end-to-end extraction orchestration time, including sync,
  worker execution, pullback, and merge

## Health Check

Run:

```bash
tools/check_jetson_frame_workers.sh
```

Expected minimum for the AGX long-talk path:

- `agx` reachable from the ai host as the control-plane SSH target;
- AGX has a private LAN IPv4 address discoverable through
  `ip -4 -o addr show scope global`, or `JETSON_RAY_HEAD_IP` is set explicitly;
- if `JETSON_AGX_LAN_HOST` is set, it must point to a stable reachable LAN DNS
  name such as `agx-lan`, not a transient DHCP address. The video-link launcher
  can detect this automatically when `agx-lan`, `agx.local`, or `ubuntu.local`
  is resolvable and reachable;
- `ffmpeg` present;
- Python can import `numpy` and `PIL`;
- `rsync` present.

The NX devices are not part of the default long-talk path because they have
been operationally unstable.

## Performance Baseline

Validated source:

```text
downloads/url-videos/3W36pd50Wqw/video.mp4
duration: 3472.181 seconds
candidate budget: 232
sample_fps: 0.5
```

Measured extraction-only results:

```text
AGX 1 worker, 10-minute window: 104.358s, 5.75x realtime
AGX 2 workers, 10-minute window: 85.966s, 6.98x realtime
AGX 2 workers, full 58-minute video: 467.121s, 7.43x realtime
AGX 4 workers, full 58-minute video: 412.280s, 8.42x realtime
```

Interpretation:

- 2 workers are the default: materially faster than 1 worker, while leaving
  headroom on the single AGX GPU/VIC/NVDEC path.
- 4 workers were faster in the benchmark, but only by another 11.74% over
  2 workers and consume more AGX concurrency headroom.
- total time includes Ray orchestration, candidate pullback, and local merge.

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
ssh agx 'rm -rf ~/.cache/video-analyzer/frame-worker/runs/*'
```

Do not delete `videos/` during performance comparisons unless you intentionally
want to measure cold sync time again.
