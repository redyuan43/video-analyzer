from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .frame import Frame, VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class JetsonFrameWorker:
    host: str
    start_seconds: float
    duration_seconds: float
    output_dir: Path


@dataclass
class JetsonFrameExtractionResult:
    frames: List[Frame]
    metadata: dict[str, Any]


REMOTE_WORKER_SCRIPT = r"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def has_command(name):
    return shutil.which(name) is not None


def has_gst_plugin(name):
    if not has_command("gst-inspect-1.0"):
        return False
    return subprocess.run(
        ["gst-inspect-1.0", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def health():
    tools = {
        "python": True,
        "numpy": True,
        "pillow": True,
        "ffmpeg": has_command("ffmpeg"),
        "ffprobe": has_command("ffprobe"),
        "gst-launch-1.0": has_command("gst-launch-1.0"),
        "gst-inspect-1.0": has_command("gst-inspect-1.0"),
        "h264parse": has_gst_plugin("h264parse"),
        "avdec_h264": has_gst_plugin("avdec_h264"),
        "nvv4l2decoder": has_gst_plugin("nvv4l2decoder"),
        "jpegenc": has_gst_plugin("jpegenc"),
        "nvjpegenc": has_gst_plugin("nvjpegenc"),
        "multifilesink": has_gst_plugin("multifilesink"),
    }
    has_decode_path = tools["ffmpeg"] or (
        tools["gst-launch-1.0"]
        and tools["h264parse"]
        and (tools["nvv4l2decoder"] or tools["avdec_h264"])
        and tools["multifilesink"]
        and (tools["nvjpegenc"] or tools["jpegenc"])
    )
    return {
        "ok": has_decode_path,
        "tools": tools,
        "decode_backend": "ffmpeg" if tools["ffmpeg"] else "gstreamer" if has_decode_path else "unavailable",
    }


def diff_score(path, previous):
    current = np.asarray(Image.open(path).convert("L").resize((320, 180)), dtype=np.float32)
    if previous is None:
        return 255.0, current
    return float(np.mean(np.abs(current - previous))), current


def extract_with_ffmpeg(video, output_dir, start, duration, sample_fps):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video),
        "-vf",
        f"fps={sample_fps}",
        "-q:v",
        "3",
        str(output_dir / "preview_%06d.jpg"),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return "ffmpeg"


def extract_with_gstreamer(video, output_dir, start, duration, sample_fps):
    # GStreamer seeking in gst-launch is less portable than ffmpeg. Decode the
    # file and keep the segment frames in the scoring pass below.
    encoder = "nvjpegenc" if has_gst_plugin("nvjpegenc") else "jpegenc"
    decoder = "nvv4l2decoder" if has_gst_plugin("nvv4l2decoder") else "avdec_h264"
    pipeline = (
        f"filesrc location={shlex_quote(str(video))} ! qtdemux ! h264parse ! "
        f"{decoder} ! videorate ! video/x-raw,framerate={int(sample_fps)}/1 ! "
        f"videoconvert ! {encoder} ! multifilesink location={shlex_quote(str(output_dir / 'preview_%06d.jpg'))}"
    )
    subprocess.run(["gst-launch-1.0", "-q", *pipeline.split()], check=True, capture_output=True)
    return "gstreamer"


def shlex_quote(value):
    import shlex
    return shlex.quote(value)


def select_candidates(image_paths, segment_start, segment_duration, sample_fps, max_frames, change_threshold, min_gap_seconds):
    candidates = []
    previous = None
    last_selected_ts = -min_gap_seconds
    segment_end = segment_start + segment_duration
    for index, path in enumerate(image_paths):
        timestamp = segment_start + (index / sample_fps if sample_fps else 0.0)
        if timestamp < segment_start or timestamp > segment_end:
            continue
        score, current = diff_score(path, previous)
        if not candidates or (score >= change_threshold and timestamp - last_selected_ts >= min_gap_seconds):
            candidates.append({"path": str(path), "timestamp": timestamp, "score": score})
            last_selected_ts = timestamp
        previous = current

    if max_frames and len(candidates) > max_frames:
        coverage = {}
        for idx, item in enumerate(candidates):
            bucket = int(item["timestamp"] // 20.0)
            coverage.setdefault(bucket, idx)
        selected = set(coverage.values())
        remaining = max(max_frames - len(selected), 0)
        ranked = sorted(
            (idx for idx in range(len(candidates)) if idx not in selected),
            key=lambda idx: candidates[idx]["score"],
            reverse=True,
        )
        selected.update(ranked[:remaining])
        candidates = [candidates[idx] for idx in sorted(selected)[:max_frames]]
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--output")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--change-threshold", type=float, default=6.0)
    parser.add_argument("--min-gap-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.health:
        print(json.dumps(health(), ensure_ascii=False))
        return 0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = health()
    if not status["ok"]:
        print(json.dumps({"success": False, "health": status}, ensure_ascii=False))
        return 2

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if status["decode_backend"] == "ffmpeg":
        backend = extract_with_ffmpeg(Path(args.video), preview_dir, args.start, args.duration, args.sample_fps)
    else:
        backend = extract_with_gstreamer(Path(args.video), preview_dir, args.start, args.duration, args.sample_fps)

    image_paths = sorted(preview_dir.glob("preview_*.jpg"))
    candidates = select_candidates(
        image_paths,
        args.start,
        args.duration,
        args.sample_fps,
        args.max_frames,
        args.change_threshold,
        args.min_gap_seconds,
    )
    manifest = {
        "success": True,
        "decode_backend": backend,
        "segment_start": args.start,
        "segment_duration": args.duration,
        "sample_fps": args.sample_fps,
        "preview_frames": len(image_paths),
        "candidate_frames": len(candidates),
        "candidates": candidates,
        "timings": {"total_seconds": round(time.perf_counter() - started, 3)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def resolve_jetson_sample_fps(value: str | float, pipeline_mode: str) -> float:
    if value == "auto":
        return {"fast": 1.0, "balanced": 2.0, "deep": 3.0}.get(pipeline_mode, 2.0)
    return max(float(value), 0.2)


def split_jetson_workers(
    hosts: Iterable[str],
    video_duration_seconds: float,
    output_dir: Path,
    overlap_seconds: float,
) -> List[JetsonFrameWorker]:
    host_list = [host.strip() for host in hosts if host.strip()]
    if not host_list:
        return []
    chunk_seconds = video_duration_seconds / len(host_list)
    workers: List[JetsonFrameWorker] = []
    for index, host in enumerate(host_list):
        start = max(0.0, index * chunk_seconds - (overlap_seconds if index else 0.0))
        end = video_duration_seconds if index == len(host_list) - 1 else min(video_duration_seconds, (index + 1) * chunk_seconds + overlap_seconds)
        workers.append(
            JetsonFrameWorker(
                host=host,
                start_seconds=start,
                duration_seconds=max(0.1, end - start),
                output_dir=output_dir / f"jetson_{index:02d}_{_safe_host_name(host)}",
            )
        )
    return workers


def extract_frames_with_jetson_workers(
    video_path: Path,
    output_dir: Path,
    hosts: List[str],
    video_duration_seconds: float,
    pipeline_mode: str,
    candidate_budget: int,
    sample_fps: str | float = "auto",
    backend: str = "auto",
    overlap_seconds: float = 2.0,
    strict: bool = True,
) -> JetsonFrameExtractionResult:
    if backend not in {"auto", "ssh", "ray"}:
        raise ValueError(f"Unknown Jetson frame backend: {backend}")
    if backend == "ray":
        logger.warning("Ray frame backend is not installed in this environment yet; using SSH worker backend")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_sample_fps = resolve_jetson_sample_fps(sample_fps, pipeline_mode)
    workers = split_jetson_workers(hosts, video_duration_seconds, output_dir, overlap_seconds)
    if not workers:
        raise RuntimeError("Jetson frame extraction requires at least one host")

    remote_video = f".cache/video-analyzer/frame-worker/videos/{_video_cache_name(video_path)}"
    health = []
    for worker in workers:
        _install_remote_worker(worker.host)
        _sync_video_to_host(worker.host, video_path, remote_video)
        host_health = _remote_health(worker.host)
        health.append({"host": worker.host, **host_health})
    unhealthy = [item for item in health if not item.get("ok")]
    if unhealthy and strict:
        raise RuntimeError(f"Jetson frame worker health check failed: {json.dumps(unhealthy, ensure_ascii=False)}")

    active_workers = [worker for worker in workers if next(item for item in health if item["host"] == worker.host).get("ok")]
    if not active_workers:
        raise RuntimeError(f"No healthy Jetson frame workers: {json.dumps(health, ensure_ascii=False)}")
    manifests = []
    with tempfile.TemporaryDirectory(prefix="jetson_frame_pull_") as pull_root:
        max_frames_per_worker = max(candidate_budget * 2 // max(len(active_workers), 1), 8)
        with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
            futures = {
                executor.submit(
                    _run_remote_extraction,
                    worker=worker,
                    remote_video=remote_video,
                    sample_fps=resolved_sample_fps,
                    max_frames=max_frames_per_worker,
                ): worker
                for worker in active_workers
            }
            for future in as_completed(futures):
                worker = futures[future]
                manifest = future.result()
                local_worker_dir = Path(pull_root) / _safe_host_name(worker.host)
                local_worker_dir.mkdir(parents=True, exist_ok=True)
                _pull_remote_candidates(worker.host, manifest, local_worker_dir)
                manifests.append((worker, manifest, local_worker_dir))

        merged = _merge_jetson_candidates(manifests, output_dir, candidate_budget)

    metadata = {
        "backend": "jetson",
        "transport": "ssh",
        "requested_backend": backend,
        "hosts": hosts,
        "health": health,
        "sample_fps": resolved_sample_fps,
        "overlap_seconds": overlap_seconds,
        "candidate_budget": candidate_budget,
        "per_host": [
            {
                "host": worker.host,
                "segment_start": worker.start_seconds,
                "segment_duration": worker.duration_seconds,
                "decode_backend": manifest.get("decode_backend"),
                "preview_frames": manifest.get("preview_frames"),
                "candidate_frames": manifest.get("candidate_frames"),
                "timings": manifest.get("timings", {}),
            }
            for worker, manifest, _ in manifests
        ],
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    return JetsonFrameExtractionResult(frames=merged, metadata=metadata)


def extract_local_screen_keyframes(
    processor: VideoProcessor,
    frames_per_minute: int,
    duration: Optional[float],
    max_frames: int,
) -> JetsonFrameExtractionResult:
    started = time.perf_counter()
    frames = processor.extract_screen_keyframes(
        frames_per_minute=frames_per_minute,
        duration=duration,
        max_frames=max_frames,
    )
    return JetsonFrameExtractionResult(
        frames=frames,
        metadata={"backend": "local", "total_seconds": round(time.perf_counter() - started, 3)},
    )


def _merge_jetson_candidates(
    manifests: list[tuple[JetsonFrameWorker, dict[str, Any], Path]],
    output_dir: Path,
    candidate_budget: int,
) -> List[Frame]:
    raw_candidates = []
    for worker, manifest, local_dir in manifests:
        for item in manifest.get("candidates", []):
            source = local_dir / Path(item["path"]).name
            if not source.exists():
                continue
            raw_candidates.append((len(raw_candidates), source, float(item["timestamp"]), float(item["score"])))
    raw_candidates.sort(key=lambda item: item[2])

    deduped = []
    last_timestamp = -math.inf
    for candidate in raw_candidates:
        if candidate[2] - last_timestamp < 0.5:
            continue
        deduped.append(candidate)
        last_timestamp = candidate[2]

    processor = VideoProcessor(Path("jetson-remote"), output_dir, "jetson")
    selected = processor._select_density_budget(deduped, candidate_budget)
    frames = []
    for index, (_source_index, source, timestamp, score) in enumerate(selected):
        frame_path = output_dir / f"frame_{index}.jpg"
        frame_path.write_bytes(source.read_bytes())
        frames.append(Frame(index, frame_path, timestamp, score))
    return frames


def _install_remote_worker(host: str) -> None:
    remote_dir = "~/.cache/video-analyzer/frame-worker"
    _run_ssh(host, f"mkdir -p {remote_dir}")
    script = textwrap.dedent(REMOTE_WORKER_SCRIPT).strip() + "\n"
    subprocess.run(
        ["ssh", *(_ssh_host_args(host)), "cat > ~/.cache/video-analyzer/frame-worker/worker.py"],
        input=script,
        text=True,
        check=True,
    )


def _sync_video_to_host(host: str, video_path: Path, remote_video: str) -> None:
    _run_ssh(host, "mkdir -p ~/.cache/video-analyzer/frame-worker/videos")
    subprocess.run(
        ["rsync", "-a", "--info=progress2", str(video_path), f"{_rsync_host(host)}:{remote_video}"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _remote_health(host: str) -> dict[str, Any]:
    result = _run_ssh(host, "python3 ~/.cache/video-analyzer/frame-worker/worker.py --health")
    return json.loads(result.stdout.strip())


def _run_remote_extraction(
    worker: JetsonFrameWorker,
    remote_video: str,
    sample_fps: float,
    max_frames: int,
) -> dict[str, Any]:
    remote_output = f".cache/video-analyzer/frame-worker/runs/{worker.output_dir.name}"
    command = " ".join(
        [
            "rm -rf",
            remote_output,
            "&&",
            "python3 ~/.cache/video-analyzer/frame-worker/worker.py",
            "--video",
            shlex.quote(remote_video),
            "--output",
            shlex.quote(remote_output),
            "--start",
            f"{worker.start_seconds:.3f}",
            "--duration",
            f"{worker.duration_seconds:.3f}",
            "--sample-fps",
            f"{sample_fps:.3f}",
            "--max-frames",
            str(max_frames),
        ]
    )
    result = _run_ssh(worker.host, command, timeout=1800)
    return json.loads(result.stdout.strip().splitlines()[-1])


def _pull_remote_candidates(host: str, manifest: dict[str, Any], local_dir: Path) -> None:
    paths = [Path(item["path"]) for item in manifest.get("candidates", [])]
    if not paths:
        return
    remote_dir = str(paths[0].parent)
    subprocess.run(
        ["rsync", "-a", f"{_rsync_host(host)}:{remote_dir}/", str(local_dir) + "/"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _run_ssh(host: str, command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *(_ssh_host_args(host)), command],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ssh_host_args(host: str) -> list[str]:
    if host == "nx3":
        return ["-o", "ProxyCommand=none", "nx@nx3.taild500c8.ts.net"]
    return [host]


def _rsync_host(host: str) -> str:
    if host == "nx3":
        return "nx@nx3.taild500c8.ts.net"
    return host


def _safe_host_name(host: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in host)


def _video_cache_name(video_path: Path) -> str:
    stat = video_path.stat()
    return f"{video_path.stem}-{stat.st_size}-{int(stat.st_mtime)}{video_path.suffix}"
