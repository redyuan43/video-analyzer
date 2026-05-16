# AGENTS.md

## Project Rules

- Always work from the repository root unless a command explicitly requires another directory.
- Read existing code and docs before changing behavior. Keep changes small, testable, and aligned with current patterns.
- Do not commit or push unless the user explicitly asks for it.
- Keep local runtime overrides in `config/config.json`; do not commit machine-specific endpoint or model configuration.
- Prefer `rg` for code and documentation search.
- Operation-manual run scripts must bypass local proxy variables for LAN/Tailscale endpoints. Source `tools/operation_manual_no_proxy_env.sh` instead of letting `HTTP_PROXY`/`ALL_PROXY` route Spark, Edge, AMD Fast, or Jetson traffic through local proxies such as `127.0.0.1:10808`.
- Keep DotsMOCR OCR endpoint configuration on stable MagicDNS names. The OCR client has a runtime fallback that uses `tailscale status --json` to resolve the current Tailscale IP if MagicDNS lookup fails.
- If the user's request is clearly part of an ongoing implementation or says to continue, keep executing the next required step instead of stopping at a status update. Report concise progress, then continue until the task is genuinely blocked or complete.

## Operation Manual Runtime

- The default operation-manual ASR path should use remote VibeVoice on Spark/Edge services. Do not add local Whisper or CPU fallback just to hide remote service failures.
- The project-wide LLM/VL endpoint is AMD Fast:
  - Base URL: `http://100.90.114.26:18081/v1`
  - Model: `hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`
  Use this same OpenAI-compatible model for frame vision analysis and final text/manual generation. Do not route this project through the generic SayAnything Gateway unless the user explicitly asks for a cross-service comparison.
- For long or strict ASR runs, use the VibeVoice HTTP endpoint on either:
  - `http://spark-31d6.taild500c8.ts.net:8012/api/asr/transcribe`
  - `http://edgexpert-4353.taild500c8.ts.net:8012/api/asr/transcribe`
- Both `8012` endpoints are lazy proxies. They should keep the VibeVoice backend unloaded until a request arrives.
- The persistent VibeVoice Ray pool is:
  - Spark Ray head: `vibevoice-ray-head.service` on `spark-31d6`, `10.31.36.1:6379`
  - Edge Ray worker: `vibevoice-ray-worker.service` on `edgexpert-4353`, `10.31.36.2`
  - Ray resources: `vibevoice_spark:1`, `vibevoice_edge:1`, `GPU:2`
- Both VibeVoice backend services should use Ray mode:
  - `VIBEVOICE_WORKER_BACKEND=ray`
  - `VIBEVOICE_RAY_ADDRESS=10.31.36.1:6379`
  - `VIBEVOICE_RAY_REQUIRED_ACTORS=spark,edge`
  - `VIBEVOICE_CHUNK_PARALLEL_WORKERS=2`
  - `VIBEVOICE_RAY_FALLBACK=raise`
- Both machines use shared logical model paths:
  - `/tmp/vibevoice-model`
  - `/tmp/qwen-tokenizer`
  These are symlinks to each machine's own Hugging Face cache. Do not pass `/home/admin/...` paths to Spark or `/home/dgx/...` paths to Edge.
- Ray worker import requires `PYTHONPATH` to include the local `VibeVoice-bench` checkout on each machine.

## Verification Commands

Use these checks before claiming the dual-worker path is healthy:

```bash
ssh dgx@spark-31d6.taild500c8.ts.net \
  "/home/dgx/github/VibeVoice-bench/.venv/bin/ray status"
```

Expected resources include both `vibevoice_spark` and `vibevoice_edge`.

```bash
ssh admin@edgexpert-4353.taild500c8.ts.net \
  "curl -fsS http://127.0.0.1:8012/api/health | python3 -m json.tool"
ssh dgx@spark-31d6.taild500c8.ts.net \
  "curl -fsS http://127.0.0.1:8012/api/health | python3 -m json.tool"
```

Expected health contains `"ray": {"enabled": true, ...}`. `connected` may remain false while idle; it becomes connected when a transcription request creates Ray actors.

For a real smoke test, send a short audio file and force chunking:

```bash
curl -fsS -X POST http://127.0.0.1:8012/api/asr/transcribe \
  -F audio=@/tmp/vibevoice-smoke-4min.wav \
  -F use_native_chunking=true \
  -F single_pass_max_duration_sec=1 \
  -F chunk_duration_sec=120 \
  -F chunk_overlap_sec=10 \
  -F chunk_parallel_workers=2
```

A passing dual-worker response includes:

```json
{
  "success": true,
  "mode": "ray_chunk_reconcile",
  "chunk_parallel_workers": 2,
  "ray_enabled": true,
  "ray_fallback_active": false,
  "ray_actor_names": ["spark", "edge"]
}
```

## Operational Notes

- Keep only one long VibeVoice ASR job active at a time. A single dual-worker request consumes both GPUs.
- After a smoke test, stop only `vibevoice-asr-backend.service` if you want to unload model actors. Keep Ray head/worker and lazy proxies active.
- If `8012` health times out during a request, check GPU processes before declaring failure; model/actor initialization can occupy the backend until the request completes.

## Video Link Status Server

- The local status server starts durable `video-link` background runs from `http://127.0.0.1:18120/video-link`, then shows progress for:
  `探测时长 -> 下载/上下文 -> 核心分析 -> 校验产物 -> 多文档分析 -> 章节深度报告 -> 导出 PDF/长图 -> 生成配图提示词`.
- The server is `tools/video_link_status_server.py`, managed by `tools/run_video_link_status_server.sh`. It exposes `POST /api/video-link/jobs`, `POST /api/video-link/jobs/<job_id>/run`, manual stage endpoints under `/stages/<stage>`, a dashboard at `/video-link/jobs/<job_id>`, and stores job state/logs under `tmp/video-link-status/jobs/`.
- Keep runtime progress and service failures visible on the status page, including failed stage, error message, log path, current log tail, and artifact counts.
- The background runner skips stages already marked `succeeded` or `skipped`, then resumes from the first incomplete stage.
- The create page should expose only common and collection options: URL, analysis mode, profile, run name, browser cookie source, skip images, keep existing, subtitles, subtitle transcript preference, comments, max comments, subtitle languages, and refresh context.
- Model endpoint/model overrides should stay in runtime profiles, not page fields. The default page profile should prefer `deepseek_v4_flash` when available.
- Start or restart the server with:
  `tools/run_video_link_status_server.sh restart`
  The launcher defaults to this repo's `.venv/bin/python`; use `VIDEO_LINK_STATUS_PYTHON=...` only for an intentional override. It must use `setsid ... < /dev/null` so the server survives Codex command-session cleanup.
- Server child commands must prepend `.venv/bin` to `PATH` and set `PYTHON=.venv/bin/python`, otherwise URL analysis can accidentally run with system Python instead of the prepared project environment.
- YouTube URLs can fail instantly with `Requested format is not available` when yt-dlp cannot solve YouTube's JS challenge. Keep `yt-dlp[default]`/`yt-dlp-ejs` installed in `.venv`, keep local `node` available, and let the URL runner's default `--ytdlp-js-runtimes auto` pass `--js-runtimes node`.
- After changing the status server, run:
  `python3 -m py_compile tools/video_link_status_server.py tests/test_video_link_status_server.py`
  and `python3 -m unittest tests.test_video_link_status_server`.

## Jetson Frame Extraction

- For long operation-manual videos, prefer Jetson candidate-frame extraction instead of local CPU/OpenCV scanning.
- For long podcast/talk videos, do not scan at `1fps` by default. Use the scripted fast path with subtitles and a sparse visual scan:
  `tools/run_long_talk_fast_from_url.sh URL --keep-existing`
  This path should use subtitles as transcript when available, skip audio ASR with `--asr-provider none`, disable VL with `--vl-frame-policy none`, use Jetson workers, require hardware decode, and sample at `--jetson-sample-fps 0.5` (one preview frame every 2 seconds).
- The current validated long-talk worker set is:
  `--jetson-frame-hosts nx1,nx2,nx3,nx4,agx`
  Use equal-weight splitting across the five devices. Do not give AGX double segment weight by default; the measured 3.8-hour sample showed AGX became the tail when assigned two slices.
- AGX can run multiple NVDEC sessions concurrently; a two-way `h264_nvv4l2dec` smoke improved two 300s segments from about 138s sequential to about 48s parallel. The best AGX internal subworker count is not fixed yet, so do not enable a default without a dedicated benchmark.
- Hardware decode is mandatory for long-video Jetson extraction. Before claiming a worker is healthy, verify the worker `health` result reports `decode_backend` containing `nvdec`. The current expected backend is `ffmpeg-nvdec` on `nx1`, `nx2`, `nx3`, `nx4`, and `agx`. Do not silently fall back to software `ffmpeg` for long videos.
- Jetson workers should use hardware low-res previews for frame-difference scoring when available: `nvv4l2decoder ! nvvidconv ! video/x-raw,format=GRAY8,width=320,height=180`. The selected candidate timestamps are then materialized as high-resolution JPEG stills for OCR/VL. This keeps OCR quality while moving grayscale/resize work to VIC.
- The current validated long-talk transport is Ray. Use `tools/start_jetson_frame_ray.sh` before long-video runs; it verifies the AGX Ray head and all five host resources, and restarts the cluster only when the resource set is incomplete.
- Ray worker subprocesses must not inherit an empty `CUDA_VISIBLE_DEVICES`. On AGX, `ffmpeg -c:v h264_nvv4l2dec` can SIGSEGV under Ray when `CUDA_VISIBLE_DEVICES=""`; remove that variable before launching the frame worker subprocess.
- The SSH workers are on-demand, not daemons: the local pipeline pushes `worker.py`, syncs/caches the video, runs chunks, pulls candidates back, and merges locally.
- For Ray conversion, do not rely on the local `.venv` as the driver because it uses Python 3.14 and Ray wheels may be unavailable. The Jetson devices use Python 3.10, so run the Ray head/driver on a device, preferably AGX when available, and have NX devices join as Ray workers. If the Ray head disappears during a job, expect the job to fail; scripts may choose a new head before a run, but Ray will not automatically keep the current job alive by electing a replacement head.
- Human one-command path for the current long sample is:
  `OCR_CACHE=refresh tools/run_s36ri23_fast_full.sh`
- Check worker readiness with:
  `tools/check_jetson_frame_workers.sh`
- Detailed operations, public CLI/API flags, maintenance commands, and the measured baseline live in `docs/JETSON_FRAME_WORKERS.md` and `.codex/skills/jetson-frame-extraction/SKILL.md`.

## Jetson LAN Sync

- Use Tailscale/MagicDNS as the control plane to reach devices and repair SSH state, but use `192.168.31.x` LAN addresses as the data plane for large video transfers between Jetson workers.
- Current LAN identities:
  - `nx1`: `nx@192.168.31.40`, Tailscale `100.119.5.57`
  - `nx2`: `nx@192.168.31.68`, Tailscale `100.123.222.45`
  - `nx3`: `nx@192.168.31.35`, Tailscale `100.127.71.86`
  - `nx4`: `nx@192.168.31.10`, Tailscale `100.82.227.71`
  - `agx`: `agx@192.168.31.201`, Tailscale `100.103.199.121`
- All `nx*` device passwords are `nx`; AGX password is `agx`. Prefer using those only to bootstrap public-key auth, then keep automated runs passwordless.
- Before large syncs, validate LAN mesh with direct device-to-device SSH, for example:
  `ssh nx1 "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new nx@192.168.31.10 hostname"`
  Every source device should be able to SSH to every peer LAN IP with `BatchMode=yes`.
- If LAN SSH fails, fix the root cause instead of falling back to local retransmission:
  - stale host key: run `ssh-keygen -f ~/.ssh/known_hosts -R <LAN_IP>` on the source device, then reconnect with `StrictHostKeyChecking=accept-new`;
  - missing auth: collect the source device public key through the working Tailscale alias and append it to the target user's `~/.ssh/authorized_keys`;
  - wrong user: use `nx@...` for NX devices and `agx@...` for AGX.
- The video cache sync should be seed-to-peer over LAN. Do not repeatedly upload the same multi-GB video from the local machine to every worker. For multiple missing peers, sync in parallel from the seed where possible.
- A full cached video should appear under:
  `~/.cache/video-analyzer/frame-worker/videos/video-<size>-<mtime>.mp4`
  Partial rsync files such as `.video-*.mp4.*` mean the cache is not ready yet.

## Network And Proxy Notes

- Keep proxy behavior split by domain:
  - YouTube/yt-dlp metadata and subtitle/comment download may use the local proxy if direct access fails.
  - LAN/Tailscale model endpoints, Jetson workers, Ray nodes, and rsync must bypass proxy variables.
- When installing or starting distributed runtime components on Jetson devices, clear proxy variables unless intentionally needed:
  `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy ...`
- If `pip` or another installer hangs on a Jetson device, check for inherited proxy settings and stuck `python3 -m pip` processes before waiting indefinitely.
