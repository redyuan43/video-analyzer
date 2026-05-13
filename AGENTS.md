# AGENTS.md

## Project Rules

- Always work from the repository root unless a command explicitly requires another directory.
- Read existing code and docs before changing behavior. Keep changes small, testable, and aligned with current patterns.
- Do not commit or push unless the user explicitly asks for it.
- Keep local runtime overrides in `config/config.json`; do not commit machine-specific endpoint or model configuration.
- Prefer `rg` for code and documentation search.

## Operation Manual Runtime

- The default operation-manual ASR path should use remote VibeVoice on Spark/Edge services. Do not add local Whisper or CPU fallback just to hide remote service failures.
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
