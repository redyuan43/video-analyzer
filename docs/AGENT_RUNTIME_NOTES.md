# Agent Runtime Notes

This document records operational knowledge for AI agents working on this
repository. It is intentionally practical: prefer these checks over assumptions
when debugging the operation-manual pipeline.

## LLM and Vision Runtime

The operation-manual pipeline should use Ivan MiniCPM-V-4.5 for visual frame
analysis and AMD Fast for final text/manual generation:

```text
Vision base URL: http://100.96.79.21:18082/v1
Vision model: minicpm-v-4.5-v100
Text base URL: http://100.90.114.26:18081/v1
Text model: hauhaucs/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive
```

Do not use the generic SayAnything Gateway as the default for this repository.
For Tailscale/LAN endpoints, bypass local proxy environment variables; routing
through `127.0.0.1:10808` can cause long generation requests to time out.

## VibeVoice Dual-Worker Deployment

The Spark/Edge VibeVoice ASR deployment is persisted as a shared Ray pool:

| Role | Host | Service | Address |
|------|------|---------|---------|
| Ray head | `spark-31d6` | `vibevoice-ray-head.service` | `10.31.36.1:6379` |
| Ray worker | `edgexpert-4353` | `vibevoice-ray-worker.service` | `10.31.36.2` |
| Lazy API | `spark-31d6` | `vibevoice-asr-lazy-proxy.service` | `:8012` |
| Lazy API | `edgexpert-4353` | `vibevoice-asr-lazy-proxy.service` | `:8012` |

Either `spark:8012` or `edgexpert:8012` can be used as the API entry point.
The backend should connect to the same Ray cluster and create two actors named
`spark` and `edge`.

## Required Environment

The VibeVoice backend must run with:

```text
VIBEVOICE_WORKER_BACKEND=ray
VIBEVOICE_RAY_ADDRESS=10.31.36.1:6379
VIBEVOICE_RAY_REQUIRED_ACTORS=spark,edge
VIBEVOICE_CHUNK_PARALLEL_WORKERS=2
VIBEVOICE_RAY_FALLBACK=raise
PYTHONPATH=/home/<user>/github/VibeVoice-bench
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Use shared logical model paths on both machines:

```text
/tmp/vibevoice-model
/tmp/qwen-tokenizer
```

These are symlinks to each machine's own local Hugging Face cache. Do not pass
machine-specific absolute paths between nodes.

## Verified Smoke Test

Both production endpoints were verified with a 4-minute audio sample and forced
chunking.

Passing response shape:

```json
{
  "success": true,
  "mode": "ray_chunk_reconcile",
  "chunk_parallel_workers": 2,
  "ray_enabled": true,
  "ray_fallback_active": false,
  "ray_error": null,
  "ray_actor_names": ["spark", "edge"]
}
```

Observed timings:

- `edgexpert:8012`: about 160 seconds end to end.
- `spark:8012`: about 159 seconds end to end.

## Debug Checklist

1. Confirm Ray resources on Spark:

```bash
ssh dgx@spark-31d6.taild500c8.ts.net \
  "/home/dgx/github/VibeVoice-bench/.venv/bin/ray status"
```

Expected: `GPU: 2`, `vibevoice_spark: 1`, and `vibevoice_edge: 1`.

2. Confirm both lazy proxies are alive:

```bash
ssh dgx@spark-31d6.taild500c8.ts.net \
  "curl -fsS http://127.0.0.1:8012/api/health | python3 -m json.tool"
ssh admin@edgexpert-4353.taild500c8.ts.net \
  "curl -fsS http://127.0.0.1:8012/api/health | python3 -m json.tool"
```

Expected: `ray.enabled` is `true`. Idle health may show `connected: false`.

3. During a real ASR request, confirm both GPUs are active:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

Expected process names include `ray::VibeVoiceChunkWorker.__init__` during
cold start and `ray::VibeVoiceChunkWorker.transcribe_chunk` during inference.

## Common Failure Modes

- `ModuleNotFoundError: No module named 'vibevoice_ray_workers'`: Ray worker
  processes do not have `VibeVoice-bench` on `PYTHONPATH`.
- Hugging Face path errors using `/home/admin/...` on Spark: the driver passed a
  machine-specific model path. Use `/tmp/vibevoice-model`.
- `ray.enabled=false` on edgexpert: check
  `~/.config/systemd/user/vibevoice-asr-backend.service.d/override.conf`; an
  old override may still force `VIBEVOICE_WORKER_BACKEND=local_single`.
- `8012` health timeout while a request is running: first inspect GPU processes
  and the request result. The backend may be busy loading actors.

## Cleanup After Tests

To unload VibeVoice model actors after a smoke test:

```bash
systemctl --user stop vibevoice-asr-backend.service
systemctl --user reset-failed vibevoice-asr-backend.service
```

Do not stop `vibevoice-ray-head.service`, `vibevoice-ray-worker.service`, or the
lazy proxy unless intentionally taking the ASR pool offline.
