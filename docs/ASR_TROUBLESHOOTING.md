# ASR Troubleshooting

## Endpoint Updates

Local endpoint hostnames are centralized in `endpoints.hosts` in `config/default_config.json`.
For machine-local overrides, edit `config/config.json` instead of changing profiles or code.

Common host keys:

- `spark`: Spark VibeVoice/Ray head host.
- `edge`: Edge VibeVoice worker/API host.
- `amd_fast`: AMD Fast OpenAI-compatible endpoint host.
- `minicpm_v100`: MiniCPM vision endpoint host.

Runtime profiles and default ASR/OCR endpoints expand these host placeholders at config load time.
Unknown placeholders fail fast during config loading.

## VibeVoice Half-Hang Recovery

Do not kill queued local jobs just because multiple jobs were submitted. The local `resource-lock`
should allow only one active ASR request while later jobs wait.

Use this order:

1. Check the local ASR lock and logs. One job should show `resource-lock acquired resource=asr`;
   queued jobs should show `resource-lock waiting resource=asr`.
2. Check Ray resources on Spark:
   `ssh dgx@spark-31d6.taild500c8.ts.net "/home/dgx/github/VibeVoice-bench/.venv/bin/ray status"`.
3. Check GPU processes on both machines. A live ASR request usually shows
   `ray::VibeVoiceChunkWorker` or `ray::VibeVoiceChunkWorker.transcribe_chunk` consuming GPU memory.
4. Treat `/api/health` timeouts during an active request as suspicious but not sufficient by itself.
   Confirm whether the request has exceeded the audio-duration-based timeout or historical runtime.
5. If an entrypoint is half-hung (GPU worker exists, health times out, no POST completion in logs),
   restart only that entrypoint's `vibevoice-asr-backend.service`. Keep Ray head/worker running.
6. After restart, the local client should log the failed endpoint and automatically try the next
   configured VibeVoice endpoint. Confirm `ASR succeeded with provider: vibevoice` and
   `resource-lock released resource=asr`.
7. Restart the Ray cluster only if Ray resources are missing, actors cannot be created, or both
   VibeVoice entrypoints are stuck.
