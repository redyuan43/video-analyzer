# AGENTS.md

## Project Rules

- Always work from the repository root unless a command explicitly requires another directory.
- Read existing code and docs before changing behavior. Keep changes small, testable, and aligned with current patterns.
- Do not commit or push unless the user explicitly asks for it.
- Keep local runtime overrides in `config/config.json`; do not commit machine-specific endpoint or model configuration.
- Prefer `rg` for code and documentation search.
- Operation-manual run scripts must bypass local proxy variables for LAN/Tailscale endpoints. Source `tools/operation_manual_no_proxy_env.sh` instead of letting `HTTP_PROXY`/`ALL_PROXY` route Spark, Edge, AMD Fast, or Jetson traffic through local proxies such as `127.0.0.1:10808`.
- Keep DotsMOCR OCR endpoint configuration on stable MagicDNS names. The OCR client has a runtime fallback that uses `tailscale status --json` to resolve the current Tailscale IP if MagicDNS lookup fails.

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

## Jetson Frame Extraction

- For long operation-manual videos, prefer Jetson candidate-frame extraction instead of local CPU/OpenCV scanning.
- Use `--frame-extractor jetson --jetson-frame-hosts nx2,nx3` for strict dual-worker mode. The current validated transport is SSH concurrent workers; Ray is only a reserved backend name for this path.
- The NX workers are on-demand, not daemons: the local pipeline pushes `worker.py`, syncs/caches the video, runs both chunks, pulls candidates back, and merges locally.
- Human one-command path for the current long sample is:
  `OCR_CACHE=refresh tools/run_s36ri23_fast_full.sh`
- Check worker readiness with:
  `tools/check_jetson_frame_workers.sh`
- Detailed operations, public CLI/API flags, maintenance commands, and the measured baseline live in `docs/JETSON_FRAME_WORKERS.md` and `.codex/skills/jetson-frame-extraction/SKILL.md`.
