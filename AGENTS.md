# AGENTS.md

## Project Rules

- Always work from the repository root unless a command explicitly requires another directory.
- Read existing code and docs before changing behavior. Keep changes small, testable, and aligned with current patterns.
- Do not commit or push unless the user explicitly asks for it.
- Keep local runtime overrides in `config/config.json`; do not commit machine-specific endpoint or model configuration.
- Prefer `rg` for code and documentation search.
- Operation-manual run scripts must bypass local proxy variables for LAN/Tailscale endpoints. Source `tools/operation_manual_no_proxy_env.sh` instead of letting `HTTP_PROXY`/`ALL_PROXY` route local loopback services, AMD Fast, Jetson, or other LAN/Tailscale traffic through local proxies such as `127.0.0.1:10808`.
- Keep runtime endpoints aligned with the current `ai` host by default. Do not switch ASR/OCR/VL work to Spark or Edge unless the user explicitly asks for a cross-machine comparison or fallback.
- If the user's request is clearly part of an ongoing implementation or says to continue, keep executing the next required step instead of stopping at a status update. Report concise progress, then continue until the task is genuinely blocked or complete.

## Operation Manual Runtime

- On the `ai` host, operation-manual model work is local-first by default.
  Before using any local loopback model endpoint such as VibeVoice ASR,
  DotsMOCR OCR, or MiniCPM VL, the analyzer must hold the global
  `local-model-runtime` lock for the whole model-using stage. A second task
  must wait on that lock and must not unload or replace the currently active
  local model until the first task finishes its ASR/OCR/VL stage and releases
  the lock. Stage switching uses `tools/prepare_ai_local_model_stage.sh` for
  loopback endpoints only; remote endpoints should not trigger
  local service switching.
- The default operation-manual ASR path should use local VibeVoice on the current `ai` machine:
  `http://127.0.0.1:18012/api/asr/transcribe`. Do not add local Whisper or CPU fallback just to hide VibeVoice failures.
- The current `ai` machine has five Tesla P40 cards plus one Tesla V100. VibeVoice uses the P40/Pascal runtime and must exclude the V100 until the user explicitly asks to validate a V100 path.
- When starting VibeVoice workers, choose only P40 GPU indices from current `nvidia-smi` output. As of 2026-06-01, the expected P40 indices are `0,1,2,4,5`; GPU `3` is `Tesla V100-SXM2-16GB` and should not be included in VibeVoice worker mapping.
- If VibeVoice startup fails with an apparent CUDA OOM on a 15-16 GiB device, first suspect that the V100 was selected accidentally. Check `nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv,noheader` and fix the GPU mapping before changing model length, memory utilization, or ASR provider.
- As of 2026-08-20, Foundation-Sec is temporarily disabled and GPU `3` (Tesla V100) is assigned first to the six-worker Qwen3.8 text pool, followed by P40 GPUs `0,1,2,4,5`. MiniCPM VL must still use only the five P40 GPUs with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; its normal local worker set and VL concurrency are both five.
- MiniCPM VL keeps the OpenAI-compatible proxy on `http://127.0.0.1:18082/v1` alive while worker `llama-server` processes are lazy and unloadable. `tools/ocr_servers/minicpm_p40_proxy.py` starts workers only on `POST /v1/...`; `/api/health` and `/v1/models` must not load the model. Workers unload after `MINICPM_IDLE_UNLOAD_SECONDS` seconds of no inflight work, default `600`, while the proxy API remains available for the next call.
- MiniCPM proxy autostart is installed as the user service `minicpm-p40-proxy.service`. It starts only the proxy after reboot; it sets `MINICPM_STOP_CONFLICTS=0`, so boot-time proxy recovery does not stop VibeVoice or DotsMOCR. Check it with `systemctl --user status minicpm-p40-proxy.service --no-pager`.
- DotsMOCR OCR must remain P40-only for now: use `0,1,2,4,5`. A 2026-06-01 V100 smoke failed in the current vLLM/Pascal runtime with `CUDA error: no kernel image is available for execution on the device`, so do not include GPU `3` in OCR until the OCR runtime is rebuilt or otherwise validated for sm_70.
- The project-wide text/VL endpoint is configured by the active runtime profile. Current local overrides may intentionally use loopback services such as MiniCPM or DotsMOCR; keep those on the current machine unless the user asks otherwise. Historical AMD Fast settings are reference-only, not a reason to move work off `ai`.
- Historical AMD Fast reference:
  - Base URL: `http://100.90.114.26:18081/v1`
  - Model: `hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`
  Use this OpenAI-compatible model only when the selected runtime profile points to AMD Fast or the user explicitly requests it. Do not route this project through the generic SayAnything Gateway unless the user explicitly asks for a cross-service comparison.
- Spark/Edge VibeVoice and Ray notes are historical fallback context only. Do not probe, SSH into, or route operation-manual ASR to Spark/Edge during normal work on this repo.

## Verification Commands

Use these checks before claiming the local VibeVoice path is healthy:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv,noheader
curl --noproxy "*" -fsS http://127.0.0.1:18012/api/health | python3 -m json.tool
```

Expected GPU selection includes only P40 cards for VibeVoice workers. The V100 must remain excluded unless explicitly requested.

For a real smoke test, send a short audio file and force chunking:

```bash
curl --noproxy "*" -fsS -X POST http://127.0.0.1:18012/api/asr/transcribe \
  -F audio=@/tmp/vibevoice-smoke-4min.wav \
  -F use_native_chunking=true \
  -F single_pass_max_duration_sec=1 \
  -F chunk_duration_sec=120 \
  -F chunk_overlap_sec=10
```

A passing local response includes:

```json
{
  "success": true,
  "provider": "vibevoice_remote"
}
```

## Operational Notes

- Keep only one long VibeVoice ASR job active at a time. A local multi-worker request consumes the selected P40 cards.
- After a smoke test, unload only the local VibeVoice backend if needed; keep unrelated local services untouched.
- If `18012` health times out during a request, check GPU processes before declaring failure; model initialization can occupy the backend until the request completes.

## AI Host VNC Operations

- The quick local helper for recreating the current x11vnc setup is `/home/ai/create-vnc-server.sh`. Do not record the VNC password in repo files. Keep the active password in the local password file used by the service.
- To refresh the VNC password and restart the existing service, run:
  `/home/ai/create-vnc-server.sh restart`
  Override only when needed with `VNC_PASSWORD=... /home/ai/create-vnc-server.sh restart`.
- The persistent service is `x11vnc.service`, defined at `/etc/systemd/system/x11vnc.service`. It serves the logged-in `ai` desktop on `:1` with auth `/run/user/1000/gdm/Xauthority`.
- If VNC works on the GDM login screen and then fails after login, check whether the graphical session switched from `:0` to `:1` and whether `x11vnc.service` took over port `5900`. In that state, changing a temporary password file such as `/tmp/x11vnc-ai.pass` will not help; update `/home/ai/.vnc/passwd` with `vncpasswd -f` and restart `x11vnc.service`.
- For TigerVNC password failures, inspect:
  `journalctl -u x11vnc.service --since '10 minutes ago' --no-pager`
  and look for `password check failed`, the active `-rfbauth` path, and the active display. Use `vncpasswd -f` to write the password file; do not use `x11vnc -storepasswd - file` as if it read from stdin.

## Video Link Status Server

- The local status server starts durable `video-link` background runs from the unified UI at `http://127.0.0.1:5000/`, then shows progress for:
  `探测时长 -> 下载/上下文 -> 核心分析 -> 校验产物 -> 多文档分析 -> 章节深度报告 -> 生成配图提示词 -> 最终定稿/发布`.
- The reusable job engine is `video_analyzer/jobengine/video_link_status_server.py`; the human entrypoint is the Flask UI under `video-analyzer-ui/video_analyzer_ui`, managed by `tools/run_video_link_status_server.sh`. It exposes `POST /api/video-link/jobs`, `GET /api/video-link/jobs`, `POST /api/video-link/jobs/<job_id>/run`, manual stage endpoints under `/stages/<stage>`, and stores job state/logs under `tmp/video-link-status/jobs/`.
- Keep runtime progress and service failures visible on the home page, including failed stage, queue/resource state, error message, log path, selected log tail, full-log copy, core-analysis substeps, and artifact counts.
- The background runner skips stages already marked `succeeded` or `skipped`, then resumes from the first incomplete stage. If a resource is busy, the stage should become `queued` instead of failing with a lock conflict.
- Treat transient video-link failures as retry-first conditions before reporting a final failure. This includes YouTube/yt-dlp incomplete data, subtitle/comment fetch errors, HTTP 429/rate limits, temporary network errors, local model cold-start races, and queued/resource-lock contention. Retry the failed stage or resume the job from the first incomplete stage at least once when the operation is idempotent or already uses `keep_existing`; only escalate after the retry also fails, and keep the first error plus retry result in the report. Do not auto-retry destructive or high-risk operations without explicit user confirmation.
- The home page should expose only common and collection options: URL, analysis mode, profile, run name, browser cookie source, skip images, keep existing, subtitles, subtitle transcript preference, comments, max comments, subtitle languages, and refresh context.
- If a job has `keep_existing=true` and `refresh_context=true`, the core stage may intentionally rerun the URL context/download path instead of only reusing the successful prepare artifacts. With `download_device=mi` and subtitles enabled, a repeated remote `yt-dlp` attempt can fail on YouTube subtitle fetch with `HTTP Error 429: Too Many Requests`; treat that as a network/rate-limit retry condition, not an ASR/OCR/VL/LLM failure.
- Model endpoint/model overrides should stay in runtime profiles, not page fields. The default page profile should prefer `deepseek_v4_pro` when available.
- Start or restart the server with:
  `tools/run_video_link_status_server.sh restart`
  The launcher defaults to this repo's `.venv/bin/python`; use `VIDEO_LINK_STATUS_PYTHON=...` only for an intentional override. It must use `setsid ... < /dev/null` so the server survives Codex command-session cleanup.
- Server child commands must prepend `.venv/bin` to `PATH` and set `PYTHON=.venv/bin/python`, otherwise URL analysis can accidentally run with system Python instead of the prepared project environment.
- YouTube URLs can fail instantly with `Requested format is not available` when yt-dlp cannot solve YouTube's JS challenge. Keep `yt-dlp[default]`/`yt-dlp-ejs` installed in `.venv`, keep local `node` available, and let the URL runner's default `--ytdlp-js-runtimes auto` pass `--js-runtimes node`.
- After changing the status server, run:
  `.venv/bin/python -m py_compile video_analyzer/jobengine/video_link_status_server.py video-analyzer-ui/video_analyzer_ui/server.py tests/test_video_link_status_server.py tests/test_video_analyzer_ui.py`
  and `.venv/bin/python -m unittest tests.test_video_link_status_server tests.test_video_analyzer_ui`.

## Video Link Resume And Publishing Notes

- `tools/run_operation_manual_from_url.py` deletes the target `run_dir` before launching analyzer work, even when `--keep-existing` reuses the downloaded video and page context. Do not resume by passing `--transcript-file` that points inside the same target `run_dir`; the wrapper can delete the transcript before the analyzer reads it.
- If a URL run is interrupted after ASR succeeds, first check whether `transcript.md` survived. If it exists, resume with absolute paths and skip ASR:
  `python -m video_analyzer.cli VIDEO.mp4 --output NEW_RUN_DIR --context-file PAGE_CONTEXT.md --asr-provider none --transcript-file /abs/path/transcript.md ...`
  Prefer a new resume output directory when in doubt, so existing ASR/transcript artifacts are not destroyed.
- If the requested text LLM changes while ASR is already running, let VibeVoice finish and write `transcript.md`, then stop before OCR/VL/manual generation and restart from that transcript with the new runtime profile/model. Do not rerun download or ASR just to change the final LLM.
- When the user asks for DeepSeek V4 Pro output, use the `deepseek_v4_pro` runtime profile for text/manual/multidoc stages. Treat DeepSeek V4 Pro as a text/review path; keep visual frame analysis on the configured vision model such as MiniCPM unless the user explicitly asks to change the visual model.
- For publisher resume after operation-manual artifacts already exist, use:
  `~/.codex/skills/video-link/scripts/run_video_link_analysis_publisher.sh URL --profile deepseek_v4_pro --run-dir "$RUN_DIR" --skip-operation`
- Current final publish is Markdown-first and does not generate PDF by default. The four final documents are `operation_manual.md`, `docs_analysis_chapters/knowledge_notes_v2.md`, `docs_analysis_chapters/deep_report_v2.md`, and `manual_evidence.md`.
- PDF export is opt-in only. Use `tools/publish/run_video_doc_final_publish.sh RUN_DIR --pdf` or `tools/publish/export_video_docs.sh` only when the user explicitly requests PDF delivery.
- The optional PDF backend is `tools/publish/md_to_mobile_pdf.py` through `tools/publish/export_video_docs.sh`. It renders prepared Markdown to narrow mobile-readable PDF with WeasyPrint. Simple acyclic `flowchart TB/TD` and `graph TB/TD/LR/RL` Mermaid blocks should render as native mobile HTML flowcharts, including branch/merge flows and `<br/>` label breaks; more complex Mermaid blocks rely on `@mermaid-js/mermaid-cli` plus Chrome/Chromium PNG rendering.
- If final publish appears stuck during PDF export, first identify the exact document being rendered. Check the stage log, live process, and export directory before suspecting OCR/GPU/model work:
  `tail -n 160 tmp/video-link-status/jobs/<job_id>/logs/final-publish.log`,
  `pgrep -af 'run_video_doc_final_publish|export_video_docs|md_to_mobile_pdf'`,
  and `ls -lh <run_dir>/exports`.
  A common failure mode is `md_to_mobile_pdf.py ... manual_evidence.pdf` consuming 100% CPU with no output PDF. This is usually WeasyPrint/Pango struggling with `manual_evidence.md` evidence tables on the narrow mobile page, not a DeepSeek/OCR/VL/GPU issue.
- For `manual_evidence.pdf`, keep the source Markdown/JSON evidence complete, but make the export-prepared Markdown PDF-friendly. `tools/publish/prepare_video_doc_export.py` should summarize or card-ify pathological evidence tables and avoid passing huge OCR/VL cells, inline HTML tables, and dense frame evidence maps directly to WeasyPrint. Validate with a focused smoke before rerunning final publish:
  `.venv/bin/python tools/publish/prepare_video_doc_export.py RUN_DIR RUN_DIR/manual_evidence.md /tmp/manual_evidence_test.md`
  then `timeout 60 .venv/bin/python tools/publish/md_to_mobile_pdf.py /tmp/manual_evidence_test.md /tmp/manual_evidence_test.pdf --title manual_evidence`.
- If `final-publish` was interrupted, verify the four final Markdown documents and `final_publish_summary.json`; incomplete final documents must leave the stage and top-level runner failed.
- Optional long-image delivery requires explicit PDF export and uses `tools/publish/export_video_docs.sh --long-png` or `tools/publish/run_video_doc_final_publish.sh --pdf --long-png`.
- Final publish should generate the configured Baoyu final images when `skip_images` is false and run `tools/publish/augment_video_docs_images.py`; it must not generate PDFs unless explicitly requested.
- Before reporting video-link completion, verify the four final Markdown documents exist and are non-empty, and verify configured final image PNGs exist unless `skip_images` is true.
  Keep `knowledge_notes`, `deep_report`, and `deep_report_v2.review` as intermediate or QA artifacts unless the user explicitly asks for them.

## Jetson Frame Extraction

- For long operation-manual videos, prefer Jetson candidate-frame extraction instead of local CPU/OpenCV scanning.
- For long podcast/talk videos, do not scan at `1fps` by default. Use the scripted fast path with subtitles and a sparse visual scan:
  `tools/run_long_talk_fast_from_url.sh URL --keep-existing`
  This path should use subtitles as transcript when available, skip audio ASR with `--asr-provider none`, disable VL with `--vl-frame-policy none`, use Jetson workers, require hardware decode, and sample at `--jetson-sample-fps 0.5` (one preview frame every 2 seconds).
- The current validated long-talk worker set is one physical AGX exposed as two
  logical frame workers:
  `--jetson-frame-hosts agx,agx`
  NX1-NX4 are manual override workers only and must not join the default Ray
  cluster.
- AGX can run two NVDEC sessions concurrently; a two-way `h264_nvv4l2dec` smoke improved two 300s segments from about 138s sequential to about 48s parallel. Keep the default at two AGX frame workers unless a dedicated benchmark proves a better count.
- Hardware decode is mandatory for long-video Jetson extraction. Before claiming the AGX worker is healthy, verify its `health` result reports `decode_backend` containing `nvdec`. Do not silently fall back to software `ffmpeg` for long videos.
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

- Use Tailscale/MagicDNS as the control plane to reach devices and repair SSH state. Use dynamically discovered private LAN addresses as the data plane for large video transfers between Jetson workers; do not hard-code transient DHCP IPs in default startup paths.
- Historical LAN identities for diagnostics only:
  - `nx1`: `nx@192.168.31.40`, Tailscale `100.119.5.57`
  - `nx2`: `nx@192.168.31.68`, Tailscale `100.123.222.45`
  - `nx3`: `nx@192.168.31.35`, Tailscale `100.127.71.86`
  - `nx4`: `nx@192.168.31.10`, Tailscale `100.82.227.71`
  - `agx`: `agx@192.168.31.201`, Tailscale `100.103.199.121`
- For AGX Ray startup, default to the `agx` control host and let `tools/start_jetson_frame_ray.sh` resolve the current private LAN IP from AGX before starting Ray. The video-link status launcher probes LAN device names (`agx-lan,agx.local,ubuntu.local` by default) and exports `JETSON_AGX_LAN_HOST` only when the name resolves and `ssh agx@<name>` works. Set `JETSON_AGX_LAN_HOST=agx-lan` only when a stable LAN DNS/DHCP hostname exists; customize probe names with `VIDEO_LINK_AGX_LAN_HOST_CANDIDATES`; set `JETSON_RAY_HEAD_IP` only as an explicit temporary override.
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
