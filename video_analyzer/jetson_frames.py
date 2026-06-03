from __future__ import annotations

import json
import logging
import math
import ipaddress
import shlex
import subprocess
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .audio_processor import AudioTranscript
from .candidate_frame_strategies import select_candidate_frames_with_strategy
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
import importlib.util
import os
from fractions import Fraction
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


def has_module(name):
    return importlib.util.find_spec(name) is not None


def has_gst_plugin(name):
    if not has_command("gst-inspect-1.0"):
        return False
    result = subprocess.run(
        ["gst-inspect-1.0", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and "Factory Details:" in output and "No such element" not in output


def has_ffmpeg_decoder(name):
    if not has_command("ffmpeg"):
        return False
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0 and name in result.stdout


def health():
    tools = {
        "python": True,
        "numpy": True,
        "pillow": True,
        "torch": has_module("torch"),
        "open_clip": has_module("open_clip"),
        "transnetv2_pytorch": has_module("transnetv2_pytorch"),
        "sklearn": has_module("sklearn"),
        "ffmpeg": has_command("ffmpeg"),
        "ffprobe": has_command("ffprobe"),
        "ffmpeg_h264_nvv4l2dec": has_ffmpeg_decoder("h264_nvv4l2dec"),
        "gst-launch-1.0": has_command("gst-launch-1.0"),
        "gst-inspect-1.0": has_command("gst-inspect-1.0"),
        "h264parse": has_gst_plugin("h264parse"),
        "avdec_h264": has_gst_plugin("avdec_h264"),
        "nvv4l2decoder": has_gst_plugin("nvv4l2decoder"),
        "nvvidconv": has_gst_plugin("nvvidconv"),
        "jpegenc": has_gst_plugin("jpegenc"),
        "nvjpegenc": has_gst_plugin("nvjpegenc"),
        "multifilesink": has_gst_plugin("multifilesink"),
    }
    has_gstreamer_nvdec = (
        tools["ffmpeg"]
        and tools["gst-launch-1.0"]
        and tools["h264parse"]
        and tools["nvv4l2decoder"]
        and tools["nvvidconv"]
        and tools["multifilesink"]
        and (tools["nvjpegenc"] or tools["jpegenc"])
    )
    has_decode_path = tools["ffmpeg_h264_nvv4l2dec"] or has_gstreamer_nvdec or tools["ffmpeg"] or (
        tools["gst-launch-1.0"]
        and tools["h264parse"]
        and (tools["nvv4l2decoder"] or tools["avdec_h264"])
        and tools["multifilesink"]
        and (tools["nvjpegenc"] or tools["jpegenc"])
    )
    if tools["ffmpeg_h264_nvv4l2dec"]:
        decode_backend = "ffmpeg-nvdec"
    elif has_gstreamer_nvdec:
        decode_backend = "gstreamer-nvdec"
    elif tools["ffmpeg"]:
        decode_backend = "ffmpeg"
    elif has_decode_path:
        decode_backend = "gstreamer"
    else:
        decode_backend = "unavailable"
    return {
        "ok": has_decode_path,
        "tools": tools,
        "decode_backend": decode_backend,
    }


def diff_score(path, previous):
    with Image.open(path) as image:
        if image.mode != "L":
            image = image.convert("L")
        if image.size != (320, 180):
            image = image.resize((320, 180))
        current = np.asarray(image, dtype=np.float32)
    if previous is None:
        return 255.0, current
    return float(np.mean(np.abs(current - previous))), current


def textness_score(path):
    with Image.open(path) as image:
        if image.mode != "L":
            image = image.convert("L")
        if image.size != (320, 180):
            image = image.resize((320, 180))
        current = np.asarray(image, dtype=np.float32)
    horizontal = np.abs(np.diff(current, axis=1))
    vertical = np.abs(np.diff(current, axis=0))
    edge_density = min((float(np.mean(horizontal > 28)) + float(np.mean(vertical > 28))) * 3.0, 1.0)
    dark = current < max(30.0, float(np.mean(current) - np.std(current) * 0.55))
    ink_density = min(float(np.mean(dark)) * 10.0, 1.0)
    sharpness = min(float(np.var(horizontal)) / 1500.0, 1.0)
    return min((0.45 * edge_density) + (0.35 * ink_density) + (0.20 * sharpness), 1.0)


def gffv_feature(path):
    with Image.open(path) as image:
        if image.mode != "L":
            image = image.convert("L")
        if image.size != (320, 180):
            image = image.resize((320, 180))
        current = np.asarray(image, dtype=np.float32)
    gy, gx = np.gradient(current)
    magnitude = np.sqrt((gx * gx) + (gy * gy))
    angles = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    bins = np.minimum((angles * 8).astype(np.int32), 7)
    histogram = np.zeros(8, dtype=np.float64)
    for index in range(8):
        histogram[index] = float(np.sum(magnitude[bins == index]))
    total = float(np.sum(histogram))
    if total <= 0:
        return [0.0] * 8
    return [round(float(value / total), 6) for value in histogram]


def sspa_projection(path):
    with Image.open(path) as image:
        if image.mode != "L":
            image = image.convert("L")
        if image.size != (320, 180):
            image = image.resize((320, 180))
        current = np.asarray(image, dtype=np.float32)
    # Use the lower-middle screen band where captions and slide bullets usually live.
    band = current[int(current.shape[0] * 0.45): int(current.shape[0] * 0.92), int(current.shape[1] * 0.05): int(current.shape[1] * 0.95)]
    if band.size == 0:
        return 0.0
    horizontal_projection = np.mean(np.abs(np.diff(band, axis=1)) > 28, axis=1)
    return round(float(np.mean(horizontal_projection)), 6)


def attach_clip_embeddings(candidates):
    if not candidates or not has_module("open_clip") or not has_module("torch"):
        return {"status": "fail", "reason": "open_clip_or_torch_missing"}
    try:
        import torch
        import open_clip

        if not torch.cuda.is_available():
            return {"status": "fail", "reason": "cuda_unavailable"}
        model_name = os.environ.get("VIDEO_ANALYZER_CLIP_MODEL", "ViT-B-32")
        pretrained = os.environ.get("VIDEO_ANALYZER_CLIP_PRETRAINED", "")
        if not pretrained:
            return {"status": "unavailable", "reason": "VIDEO_ANALYZER_CLIP_PRETRAINED missing"}
        device = "cuda" if torch.cuda.is_available() and os.environ.get("VIDEO_ANALYZER_CLIP_DEVICE", "auto") != "cpu" else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()
        batch_size = max(1, int(os.environ.get("VIDEO_ANALYZER_CLIP_BATCH", "8")))
        with torch.no_grad():
            for offset in range(0, len(candidates), batch_size):
                rows = candidates[offset: offset + batch_size]
                images = []
                for item in rows:
                    with Image.open(item["path"]) as image:
                        images.append(preprocess(image.convert("RGB")))
                tensor = torch.stack(images).to(device)
                features = model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                for item, feature in zip(rows, features.detach().cpu().tolist()):
                    item["clip_embedding"] = [round(float(value), 6) for value in feature]
        return {"status": "ok", "model": model_name, "pretrained": pretrained, "device": device, "candidate_count": len(candidates)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def attach_transnet_shot_boundaries(video, candidates):
    if not candidates or not has_module("transnetv2_pytorch") or not has_module("torch"):
        return {"status": "fail", "reason": "transnetv2_pytorch_or_torch_missing"}
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "fail", "reason": "cuda_unavailable"}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"torch_unavailable: {exc}"}
    weights_path = os.environ.get("VIDEO_ANALYZER_TRANSNET_WEIGHTS", "")
    if not weights_path:
        try:
            import transnetv2_pytorch
            package_root = Path(transnetv2_pytorch.__file__).resolve().parent
            bundled = package_root / "transnetv2-pytorch-weights.pth"
            if bundled.exists():
                weights_path = str(bundled)
        except Exception:
            weights_path = ""
    if not weights_path or not Path(weights_path).exists():
        return {"status": "unavailable", "reason": "VIDEO_ANALYZER_TRANSNET_WEIGHTS missing"}
    try:
        from transnetv2_pytorch import TransNetV2

        threshold = float(os.environ.get("VIDEO_ANALYZER_TRANSNET_THRESHOLD", "0.5"))
        model = TransNetV2(device=os.environ.get("VIDEO_ANALYZER_TRANSNET_DEVICE", "auto"))
        model.eval()
        state_dict = torch.load(weights_path, map_location=model.device)
        model.load_state_dict(state_dict)
        with torch.no_grad():
            scenes = model.detect_scenes(str(video), threshold=threshold)
        boundaries = []
        for scene in scenes:
            start = float(scene.get("start_time", 0.0))
            boundaries.append(start)
        for item in candidates:
            timestamp = float(item.get("timestamp", 0.0))
            item["shot_boundary"] = any(abs(timestamp - boundary) <= 1.0 for boundary in boundaries if boundary > 0.0)
        return {"status": "ok", "scene_count": len(scenes), "threshold": threshold, "weights": weights_path}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def gst_framerate(value):
    fraction = Fraction(float(value)).limit_denominator(1000)
    return f"{fraction.numerator}/{fraction.denominator}"


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


def extract_with_ffmpeg_nvdec(video, output_dir, start, duration, sample_fps):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-c:v",
        "h264_nvv4l2dec",
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
    return "ffmpeg-nvdec"


def copy_h264_segment(video, output_dir, start, duration):
    segment = output_dir / "segment.h264"
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
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "h264",
        "-y",
        str(segment),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return segment


def extract_with_gstreamer(video, output_dir, start, duration, sample_fps):
    segment = copy_h264_segment(video, output_dir, start, duration)
    decoder = "nvv4l2decoder" if has_gst_plugin("nvv4l2decoder") else "avdec_h264"
    pipeline = [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        f"location={segment}",
        "!",
        "h264parse",
        "!",
        decoder,
        "!",
    ]
    if decoder == "nvv4l2decoder":
        pipeline.extend(["nvvidconv", "!", "video/x-raw", "!"])
    pipeline.extend(
        [
            "videorate",
            "!",
            f"video/x-raw,framerate={int(sample_fps)}/1",
            "!",
            "videoconvert",
            "!",
            "jpegenc",
            "!",
            "multifilesink",
            f"location={output_dir / 'preview_%06d.jpg'}",
        ]
    )
    subprocess.run(pipeline, check=True, capture_output=True)
    return "gstreamer-nvdec" if decoder == "nvv4l2decoder" else "gstreamer"


def extract_gray_preview_with_gstreamer(video, output_dir, start, duration, sample_fps):
    segment = copy_h264_segment(video, output_dir, start, duration)
    try:
        pipeline = [
            "gst-launch-1.0",
            "-q",
            "filesrc",
            f"location={segment}",
            "!",
            "h264parse",
            "!",
            "nvv4l2decoder",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw,format=GRAY8,width=320,height=180",
            "!",
            "videorate",
            "!",
            f"video/x-raw,format=GRAY8,width=320,height=180,framerate={gst_framerate(sample_fps)}",
            "!",
            "jpegenc",
            "!",
            "multifilesink",
            f"location={output_dir / 'preview_%06d.jpg'}",
        ]
        subprocess.run(pipeline, check=True, capture_output=True)
    finally:
        segment.unlink(missing_ok=True)
    return "gstreamer-nvdec-vic-gray"


def extract_frames(video, output_dir, start, duration, sample_fps, backend):
    if backend == "ffmpeg-nvdec":
        return extract_with_ffmpeg_nvdec(video, output_dir, start, duration, sample_fps)
    if backend in {"gstreamer-nvdec", "gstreamer"}:
        return extract_with_gstreamer(video, output_dir, start, duration, sample_fps)
    return extract_with_ffmpeg(video, output_dir, start, duration, sample_fps)


def extract_preview_frames(video, output_dir, start, duration, sample_fps, status):
    tools = status.get("tools", {})
    if (
        tools.get("gst-launch-1.0")
        and tools.get("h264parse")
        and tools.get("nvv4l2decoder")
        and tools.get("nvvidconv")
        and tools.get("jpegenc")
        and tools.get("multifilesink")
    ):
        try:
            return extract_gray_preview_with_gstreamer(video, output_dir, start, duration, sample_fps), None
        except Exception as exc:
            return extract_frames(video, output_dir, start, duration, sample_fps, status["decode_backend"]), str(exc)
    return extract_frames(video, output_dir, start, duration, sample_fps, status["decode_backend"]), None


def extract_highres_candidate(video, timestamp, output_path):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-c:v",
        "h264_nvv4l2dec",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return "ffmpeg-nvdec"
    except Exception:
        fallback = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output_path),
        ]
        subprocess.run(fallback, check=True, capture_output=True)
        return "ffmpeg"


def materialize_highres_candidates(video, output_dir, candidates):
    highres_dir = output_dir / "candidates"
    highres_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    still_backends = set()
    for index, item in enumerate(candidates):
        timestamp = float(item["timestamp"])
        highres_path = highres_dir / f"candidate_{index:06d}.jpg"
        still_backends.add(extract_highres_candidate(video, timestamp, highres_path))
        materialized.append(
            {
                **item,
                "preview_path": item.get("path", ""),
                "path": str(highres_path),
            }
        )
    return materialized, sorted(still_backends)


def read_candidates_json(value):
    if not value:
        return []
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def candidate_observation_metrics(candidates, segment_start, segment_duration):
    if not candidates:
        return {
            "candidate_count": 0,
            "coverage_60s_bucket_ratio": 0.0,
            "near_duplicate_gap_count": 0,
            "stable_text_candidate_count": 0,
            "textness_mean": 0.0,
            "visual_score_mean": 0.0,
        }
    timestamps = [float(item.get("timestamp", 0.0)) for item in candidates]
    textness_values = [float(item.get("textness_score", 0.0)) for item in candidates]
    visual_values = [float(item.get("visual_score", 0.0)) for item in candidates]
    total_buckets = max(1, int(np.ceil(float(segment_duration or 0.0) / 60.0)))
    covered_buckets = {
        int(max(0.0, timestamp - float(segment_start or 0.0)) // 60.0)
        for timestamp in timestamps
    }
    sorted_timestamps = sorted(timestamps)
    near_duplicates = sum(
        1
        for previous, current in zip(sorted_timestamps, sorted_timestamps[1:])
        if current - previous < 2.0
    )
    stable_text = sum(
        1
        for textness, visual in zip(textness_values, visual_values)
        if textness >= 0.28 and visual <= 8.0
    )
    return {
        "candidate_count": len(candidates),
        "coverage_60s_bucket_ratio": round(min(len(covered_buckets) / total_buckets, 1.0), 4),
        "near_duplicate_gap_count": near_duplicates,
        "stable_text_candidate_count": stable_text,
        "textness_mean": round(float(np.mean(textness_values)), 4) if textness_values else 0.0,
        "textness_max": round(float(np.max(textness_values)), 4) if textness_values else 0.0,
        "visual_score_mean": round(float(np.mean(visual_values)), 4) if visual_values else 0.0,
        "visual_score_max": round(float(np.max(visual_values)), 4) if visual_values else 0.0,
    }


def shlex_quote(value):
    import shlex
    return shlex.quote(value)


def add_uniform_coverage_candidates(candidates, image_paths, segment_start, segment_duration, sample_fps, max_frames):
    if not max_frames or not image_paths:
        return candidates
    target = min(max_frames, len(image_paths))
    if len(candidates) >= target:
        return candidates

    by_path = {item["path"]: item for item in candidates}
    needed = target - len(by_path)
    if needed <= 0:
        return candidates

    denominator = max(target - 1, 1)
    uniform_indexes = [round(index * (len(image_paths) - 1) / denominator) for index in range(target)]
    for index in uniform_indexes:
        if needed <= 0:
            break
        path = image_paths[index]
        path_text = str(path)
        if path_text in by_path:
            continue
        timestamp = segment_start + (index / sample_fps if sample_fps else 0.0)
        by_path[path_text] = {
            "path": path_text,
            "timestamp": min(timestamp, segment_start + segment_duration),
            "score": 0.0,
        }
        needed -= 1

    return sorted(by_path.values(), key=lambda item: item["timestamp"])


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
        textness = textness_score(path)
        combined_score = score + (textness * 30.0)
        if not candidates or ((score >= change_threshold or textness >= 0.28) and timestamp - last_selected_ts >= min_gap_seconds):
            candidates.append({
                "path": str(path),
                "timestamp": timestamp,
                "score": combined_score,
                "visual_score": score,
                "textness_score": textness,
                "gffv": gffv_feature(path),
                "sspa_projection": sspa_projection(path),
            })
            last_selected_ts = timestamp
        previous = current

    candidates = add_uniform_coverage_candidates(
        candidates,
        image_paths,
        segment_start,
        segment_duration,
        sample_fps,
        max_frames,
    )

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
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--materialize-candidates-json")
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

    if args.materialize_candidates_json:
        video = Path(args.video)
        candidates = read_candidates_json(args.materialize_candidates_json)
        materialize_started = time.perf_counter()
        candidates, still_backends = materialize_highres_candidates(video, output_dir, candidates)
        materialize_seconds = round(time.perf_counter() - materialize_started, 3)
        manifest = {
            "success": True,
            "materialize_only": True,
            "candidate_still_backends": still_backends,
            "candidate_frames": len(candidates),
            "candidates": candidates,
            "timings": {
                "preview_seconds": 0.0,
                "selection_seconds": 0.0,
                "materialize_seconds": materialize_seconds,
                "total_seconds": round(time.perf_counter() - started, 3),
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False))
        return 0

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    preview_started = time.perf_counter()
    backend, preview_fallback = extract_preview_frames(
        video,
        preview_dir,
        args.start,
        args.duration,
        args.sample_fps,
        status,
    )
    preview_seconds = round(time.perf_counter() - preview_started, 3)

    image_paths = sorted(preview_dir.glob("preview_*.jpg"))
    selection_started = time.perf_counter()
    candidates = select_candidates(
        image_paths,
        args.start,
        args.duration,
        args.sample_fps,
        args.max_frames,
        args.change_threshold,
        args.min_gap_seconds,
    )
    paper_backends = {
        "transnetv2": attach_transnet_shot_boundaries(video, candidates),
        "open_clip": attach_clip_embeddings(candidates),
        "sspa": {"status": "ok", "projection": "lower_middle_text_band"},
        "mskvs": {"status": "ok", "feature": "gffv_global_orientation_histogram"},
    }
    selection_seconds = round(time.perf_counter() - selection_started, 3)
    still_backends = []
    materialize_seconds = 0.0
    if not args.metadata_only:
        materialize_started = time.perf_counter()
        candidates, still_backends = materialize_highres_candidates(video, output_dir, candidates)
        materialize_seconds = round(time.perf_counter() - materialize_started, 3)
    manifest = {
        "success": True,
        "decode_backend": backend,
        "preview_fallback": preview_fallback,
        "candidate_still_backends": still_backends,
        "metadata_only": args.metadata_only,
        "paper_backends": paper_backends,
        "segment_start": args.start,
        "segment_duration": args.duration,
        "sample_fps": args.sample_fps,
        "preview_frames": len(image_paths),
        "candidate_frames": len(candidates),
        "observation_metrics": candidate_observation_metrics(candidates, args.start, args.duration),
        "candidates": candidates,
        "timings": {
            "preview_seconds": preview_seconds,
            "selection_seconds": selection_seconds,
            "materialize_seconds": materialize_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
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
    host_weights: dict[str, float] | None = None,
) -> List[JetsonFrameWorker]:
    host_list = [host.strip() for host in hosts if host.strip()]
    if not host_list:
        return []
    weights = [max(float((host_weights or {}).get(host, 1.0)), 0.1) for host in host_list]
    total_weight = sum(weights)
    workers: List[JetsonFrameWorker] = []
    cursor = 0.0
    for index, (host, weight) in enumerate(zip(host_list, weights)):
        nominal_start = cursor
        cursor += video_duration_seconds * weight / total_weight
        nominal_end = video_duration_seconds if index == len(host_list) - 1 else cursor
        start = max(0.0, nominal_start - (overlap_seconds if index else 0.0))
        end = min(video_duration_seconds, nominal_end + (overlap_seconds if index != len(host_list) - 1 else 0.0))
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
    candidate_strategy: str = "auto",
    transcript: AudioTranscript | None = None,
    sample_fps: str | float = "auto",
    backend: str = "auto",
    overlap_seconds: float = 2.0,
    host_weights: dict[str, float] | None = None,
    require_hardware_decode: bool = False,
    strict: bool = True,
) -> JetsonFrameExtractionResult:
    if backend not in {"auto", "ssh", "ray"}:
        raise ValueError(f"Unknown Jetson frame backend: {backend}")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_sample_fps = resolve_jetson_sample_fps(sample_fps, pipeline_mode)
    workers = split_jetson_workers(hosts, video_duration_seconds, output_dir, overlap_seconds, host_weights)
    if not workers:
        raise RuntimeError("Jetson frame extraction requires at least one host")

    remote_video = f".cache/video-analyzer/frame-worker/videos/{_video_cache_name(video_path)}"
    for worker in workers:
        _install_remote_worker(worker.host)

    health = []
    for worker in workers:
        host_health = _remote_health(worker.host)
        health.append({"host": worker.host, **host_health})
    unhealthy = [item for item in health if not item.get("ok")]
    if unhealthy and strict:
        raise RuntimeError(f"Jetson frame worker health check failed: {json.dumps(unhealthy, ensure_ascii=False)}")

    active_workers = []
    skipped_workers = []
    for worker in workers:
        host_health = next(item for item in health if item["host"] == worker.host)
        if not host_health.get("ok"):
            skipped_workers.append({"host": worker.host, "reason": "unhealthy", "health": host_health})
            continue
        if require_hardware_decode and "nvdec" not in str(host_health.get("decode_backend", "")):
            skipped_workers.append(
                {
                    "host": worker.host,
                    "reason": "hardware_decode_unavailable",
                    "decode_backend": host_health.get("decode_backend"),
                }
            )
            logger.warning(
                "Skipping Jetson worker %s because hardware decode is required but backend is %s",
                worker.host,
                host_health.get("decode_backend"),
            )
            continue
        active_workers.append(worker)
    if not active_workers:
        raise RuntimeError(f"No healthy Jetson frame workers: {json.dumps(health, ensure_ascii=False)}")

    _sync_video_to_workers([worker.host for worker in active_workers], video_path, remote_video)
    manifests = []
    pullback_seconds = 0.0
    materialize_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="jetson_frame_pull_") as pull_root:
        max_frames_per_worker = max(candidate_budget * 2 // max(len(active_workers), 1), 8)
        if backend == "ray":
            remote_manifests = _run_ray_extractions(
                active_workers,
                remote_video=remote_video,
                sample_fps=resolved_sample_fps,
                max_frames=max_frames_per_worker,
                metadata_only=True,
            )
            for worker, manifest in remote_manifests:
                local_worker_dir = Path(pull_root) / worker.output_dir.name
                local_worker_dir.mkdir(parents=True, exist_ok=True)
                manifests.append((worker, manifest, local_worker_dir))
        else:
            with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
                futures = {
                    executor.submit(
                        _run_remote_extraction,
                        worker=worker,
                        remote_video=remote_video,
                        sample_fps=resolved_sample_fps,
                        max_frames=max_frames_per_worker,
                        metadata_only=True,
                    ): worker
                    for worker in active_workers
                }
                for future in as_completed(futures):
                    worker = futures[future]
                    manifest = future.result()
                    local_worker_dir = Path(pull_root) / worker.output_dir.name
                    local_worker_dir.mkdir(parents=True, exist_ok=True)
                    manifests.append((worker, manifest, local_worker_dir))

        selected_by_worker, selection_observation = _select_jetson_candidate_metadata(
            manifests,
            candidate_budget,
            video_duration_seconds=video_duration_seconds,
            pipeline_mode=pipeline_mode,
            candidate_strategy=candidate_strategy,
            transcript=transcript,
        )
        materialize_started = time.perf_counter()
        materialized_manifests = _materialize_selected_jetson_candidates(
            selected_by_worker,
            remote_video=remote_video,
            backend=backend,
        )
        materialize_seconds = round(time.perf_counter() - materialize_started, 3)
        manifest_by_worker = {worker.output_dir.name: manifest for worker, manifest in materialized_manifests}
        for index, (worker, manifest, local_worker_dir) in enumerate(manifests):
            materialized_manifest = manifest_by_worker.get(worker.output_dir.name) or {}
            combined_timings = {
                **dict(manifest.get("timings") or {}),
                "materialize_seconds": (materialized_manifest.get("timings") or {}).get("materialize_seconds", 0.0),
                "materialize_total_seconds": (materialized_manifest.get("timings") or {}).get("total_seconds", 0.0),
            }
            combined_manifest = {
                **manifest,
                **materialized_manifest,
                "decode_backend": manifest.get("decode_backend"),
                "preview_fallback": manifest.get("preview_fallback"),
                "metadata_only": manifest.get("metadata_only"),
                "raw_candidate_frames": manifest.get("candidate_frames"),
                "candidate_still_backends": materialized_manifest.get("candidate_still_backends", []),
                "candidate_frames": materialized_manifest.get("candidate_frames", 0),
                "candidates": materialized_manifest.get("candidates", []),
                "observation_metrics": manifest.get("observation_metrics", {}),
                "timings": combined_timings,
            }
            manifests[index] = (worker, combined_manifest, local_worker_dir)
        pullback_started = time.perf_counter()
        for worker, manifest, local_worker_dir in manifests:
            _pull_remote_candidates(worker.host, manifest, local_worker_dir)
        pullback_seconds = round(time.perf_counter() - pullback_started, 3)
        merged = _merge_jetson_candidates(manifests, output_dir, candidate_budget)

    metadata = {
        "backend": "jetson",
        "transport": "ray" if backend == "ray" else "ssh",
        "requested_backend": backend,
        "hosts": hosts,
        "host_weights": host_weights or {},
        "active_hosts": [worker.host for worker in active_workers],
        "skipped_hosts": skipped_workers,
        "require_hardware_decode": require_hardware_decode,
        "health": health,
        "sample_fps": resolved_sample_fps,
        "overlap_seconds": overlap_seconds,
        "candidate_budget": candidate_budget,
        "candidate_strategy": candidate_strategy,
        "pullback_seconds": pullback_seconds,
        "materialize_seconds": materialize_seconds,
        "quality_guard": {
            "status": selection_observation.get("paper_algorithm_status", "unknown"),
            "selection_strategy_changed": selection_observation.get("selected_candidate_strategy") not in {None, "legacy"},
            "final_budget_preserved": len(merged) <= candidate_budget,
            "candidate_strategy_guard": selection_observation.get("quality_guard", {}),
        },
        "observation_metrics": selection_observation,
        "video_profile": selection_observation.get("video_profile"),
        "video_profile_confidence": selection_observation.get("video_profile_confidence"),
        "selected_candidate_strategy": selection_observation.get("selected_candidate_strategy"),
        "paper_algorithm": selection_observation.get("paper_algorithm"),
        "paper_algorithm_status": selection_observation.get("paper_algorithm_status"),
        "paper_backends": _merge_paper_backend_status(manifests),
        "strategy_observations": selection_observation.get("strategy_observations", {}),
        "per_host": [
            {
                "host": worker.host,
                "segment_start": worker.start_seconds,
                "segment_duration": worker.duration_seconds,
                "decode_backend": manifest.get("decode_backend"),
                "preview_fallback": manifest.get("preview_fallback"),
                "candidate_still_backends": manifest.get("candidate_still_backends", []),
                "preview_frames": manifest.get("preview_frames"),
                "candidate_frames": manifest.get("candidate_frames"),
                "raw_candidate_frames": manifest.get("raw_candidate_frames", manifest.get("candidate_frames")),
                "paper_backends": manifest.get("paper_backends", {}),
                "observation_metrics": manifest.get("observation_metrics", {}),
                "timings": manifest.get("timings", {}),
            }
            for worker, manifest, _ in manifests
        ],
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    return JetsonFrameExtractionResult(frames=merged, metadata=metadata)


def _merge_paper_backend_status(manifests: list[tuple[JetsonFrameWorker, dict[str, Any], Path]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for worker, manifest, _local_dir in manifests:
        for name, status in (manifest.get("paper_backends") or {}).items():
            bucket = merged.setdefault(name, {"ok_hosts": [], "unavailable_hosts": [], "details": {}})
            if isinstance(status, dict) and status.get("status") == "ok":
                bucket["ok_hosts"].append(worker.host)
            else:
                bucket["unavailable_hosts"].append(worker.host)
            bucket["details"][worker.output_dir.name] = status
    return merged


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
    metadata = {"backend": "local", "total_seconds": round(time.perf_counter() - started, 3)}
    metadata.update(getattr(processor, "last_extraction_metadata", {}) or {})
    metadata.setdefault("scan_frames_count", len(frames))
    metadata.setdefault("candidate_budget", max_frames)
    return JetsonFrameExtractionResult(
        frames=frames,
        metadata=metadata,
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


def _select_jetson_candidate_metadata(
    manifests: list[tuple[JetsonFrameWorker, dict[str, Any], Path]],
    candidate_budget: int,
    video_duration_seconds: float | None = None,
    pipeline_mode: str = "balanced",
    candidate_strategy: str = "auto",
    transcript: AudioTranscript | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_candidates = []
    for worker, manifest, _local_dir in manifests:
        for item_index, item in enumerate(manifest.get("candidates", [])):
            timestamp = float(item["timestamp"])
            score = float(item["score"])
            raw_candidates.append(
                (
                    len(raw_candidates),
                    {"worker": worker, "item": item, "item_index": item_index},
                    timestamp,
                    score,
                )
            )
    raw_candidates.sort(key=lambda item: item[2])

    deduped = []
    last_timestamp = -math.inf
    for candidate in raw_candidates:
        if candidate[2] - last_timestamp < 0.5:
            continue
        deduped.append(candidate)
        last_timestamp = candidate[2]

    processor = VideoProcessor(Path("jetson-remote"), Path("."), "jetson")
    strategy_result = select_candidate_frames_with_strategy(
        deduped,
        candidate_budget=candidate_budget,
        video_duration_seconds=float(video_duration_seconds or (deduped[-1][2] if deduped else 0.0)),
        pipeline_mode=pipeline_mode,
        strategy=candidate_strategy,
        transcript_segments=(transcript.segments if transcript else None),
        legacy_selector=processor._select_density_budget,
    )
    selected = strategy_result.selected
    selected_by_worker: dict[str, dict[str, Any]] = {}
    for _source_index, payload, _timestamp, _score in selected:
        worker = payload["worker"]
        selected_by_worker.setdefault(worker.output_dir.name, {"worker": worker, "candidates": []})["candidates"].append(payload["item"])

    raw_metrics = _candidate_observation_metrics_from_items([item[1]["item"] for item in raw_candidates])
    selected_metrics = _candidate_observation_metrics_from_items([item[1]["item"] for item in selected])
    return selected_by_worker, {
        "raw_candidate_points": len(raw_candidates),
        "deduped_candidate_points": len(deduped),
        "final_candidate_points": len(selected),
        "candidate_budget": candidate_budget,
        "raw_metrics": raw_metrics,
        "final_metrics": selected_metrics,
        **strategy_result.metadata,
    }


def _candidate_observation_metrics_from_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "candidate_count": 0,
            "coverage_60s_bucket_ratio": 0.0,
            "near_duplicate_gap_count": 0,
            "stable_text_candidate_count": 0,
            "textness_mean": 0.0,
            "visual_score_mean": 0.0,
        }
    timestamps = [float(item.get("timestamp", 0.0)) for item in items]
    start = min(timestamps)
    duration = max(timestamps) - start if len(timestamps) > 1 else 0.0
    total_buckets = max(1, int(math.ceil(duration / 60.0)))
    covered_buckets = {int(max(0.0, timestamp - start) // 60.0) for timestamp in timestamps}
    sorted_timestamps = sorted(timestamps)
    textness_values = [float(item.get("textness_score", 0.0)) for item in items]
    visual_values = [float(item.get("visual_score", 0.0)) for item in items]
    return {
        "candidate_count": len(items),
        "coverage_60s_bucket_ratio": round(min(len(covered_buckets) / total_buckets, 1.0), 4),
        "near_duplicate_gap_count": sum(
            1
            for previous, current in zip(sorted_timestamps, sorted_timestamps[1:])
            if current - previous < 2.0
        ),
        "stable_text_candidate_count": sum(
            1
            for textness, visual in zip(textness_values, visual_values)
            if textness >= 0.28 and visual <= 8.0
        ),
        "textness_mean": round(sum(textness_values) / len(textness_values), 4),
        "textness_max": round(max(textness_values), 4),
        "visual_score_mean": round(sum(visual_values) / len(visual_values), 4),
        "visual_score_max": round(max(visual_values), 4),
    }


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
    logger.info("Syncing video to Jetson worker %s via local rsync", host)
    _run_ssh(host, "mkdir -p ~/.cache/video-analyzer/frame-worker/videos")
    command = ["rsync", "-a", "--info=progress2", *_rsync_ssh_args(host), str(video_path), f"{_rsync_host(host)}:{remote_video}"]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def _sync_video_to_workers(hosts: list[str], video_path: Path, remote_video: str) -> None:
    if not hosts:
        return
    size = video_path.stat().st_size
    ready_hosts = [host for host in hosts if _remote_file_size(host, remote_video) == size]
    seed = ready_hosts[0] if ready_hosts else hosts[0]
    logger.info("Jetson video cache seed is %s; ready hosts: %s", seed, ready_hosts or "none")
    if seed not in ready_hosts:
        _sync_video_to_host(seed, video_path, remote_video)

    missing_hosts = [
        host
        for host in hosts
        if host != seed and _remote_file_size(host, remote_video) != size
    ]
    if not missing_hosts:
        return

    lan_targets = _resolve_lan_peer_targets(seed, missing_hosts)

    def sync_missing_host(host: str) -> None:
        peer_target = lan_targets.get(host)
        if peer_target:
            logger.info("Syncing video from Jetson seed %s to %s over LAN peer %s", seed, host, peer_target)
            try:
                _sync_video_between_workers(seed, peer_target, remote_video)
                return
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "LAN peer sync from %s to %s failed with code %s; falling back to local rsync",
                    seed,
                    host,
                    exc.returncode,
                )
        else:
            logger.info("No LAN peer target for %s from seed %s; falling back to local rsync", host, seed)
        _sync_video_to_host(host, video_path, remote_video)

    with ThreadPoolExecutor(max_workers=len(missing_hosts)) as executor:
        futures = [executor.submit(sync_missing_host, host) for host in missing_hosts]
        for future in as_completed(futures):
            future.result()


def _remote_file_size(host: str, remote_path: str) -> int:
    result = _run_ssh(host, f"stat -c %s {shlex.quote(remote_path)} 2>/dev/null || true")
    value = result.stdout.strip()
    return int(value) if value.isdigit() else -1


def _resolve_lan_peer_targets(seed: str, hosts: list[str]) -> dict[str, str]:
    seed_interfaces = _remote_lan_interfaces(seed)
    if not seed_interfaces:
        return {}
    targets: dict[str, str] = {}
    for host in hosts:
        if host == seed:
            continue
        for peer_interface in _remote_lan_interfaces(host):
            if any(peer_interface.ip in seed_interface.network for seed_interface in seed_interfaces):
                targets[host] = f"{_ssh_user(host)}@{peer_interface.ip}"
                break
    return targets


def _ssh_user(host: str) -> str:
    result = subprocess.run(
        ["ssh", "-G", host],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("user "):
            return line.split(None, 1)[1]
    return "nx"


def _remote_lan_interfaces(host: str) -> list[ipaddress.IPv4Interface]:
    result = _run_ssh(host, "ip -4 -o addr show scope global | awk '{print $4}'")
    interfaces: list[ipaddress.IPv4Interface] = []
    for line in result.stdout.splitlines():
        try:
            interface = ipaddress.ip_interface(line.strip())
        except ValueError:
            continue
        if not isinstance(interface, ipaddress.IPv4Interface):
            continue
        if interface.ip.is_loopback or str(interface.ip).startswith("100."):
            continue
        interfaces.append(interface)
    return interfaces


def _sync_video_between_workers(seed_host: str, peer_target: str, remote_video: str) -> None:
    remote_dir = shlex.quote(str(Path(remote_video).parent))
    remote_path = shlex.quote(remote_video)
    peer = shlex.quote(f"{peer_target}:{remote_video}")
    ssh_cmd = "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    command = " ".join(
        [
            f"{ssh_cmd} {shlex.quote(peer_target)} mkdir -p {remote_dir}",
            "&&",
            "rsync -a --info=progress2",
            "-e",
            shlex.quote(ssh_cmd),
            remote_path,
            peer,
        ]
    )
    _run_ssh(seed_host, command, timeout=1800)


def _remote_health(host: str) -> dict[str, Any]:
    result = _run_ssh(host, f"{_remote_worker_python()} ~/.cache/video-analyzer/frame-worker/worker.py --health")
    return json.loads(result.stdout.strip())


def _run_remote_extraction(
    worker: JetsonFrameWorker,
    remote_video: str,
    sample_fps: float,
    max_frames: int,
    metadata_only: bool = False,
) -> dict[str, Any]:
    remote_output = f".cache/video-analyzer/frame-worker/runs/{worker.output_dir.name}"
    metadata_arg = " --metadata-only" if metadata_only else ""
    command = " ".join(
        [
            "rm -rf",
            remote_output,
            "&&",
            _remote_worker_python(),
            "~/.cache/video-analyzer/frame-worker/worker.py",
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
            metadata_arg,
        ]
    )
    result = _run_ssh(worker.host, command, timeout=1800)
    return json.loads(result.stdout.strip().splitlines()[-1])


def _run_ray_extractions(
    workers: list[JetsonFrameWorker],
    remote_video: str,
    sample_fps: float,
    max_frames: int,
    metadata_only: bool = False,
) -> list[tuple[JetsonFrameWorker, dict[str, Any]]]:
    head = "agx" if any(worker.host == "agx" for worker in workers) else workers[0].host
    specs = [
        {
            "id": str(index),
            "host": worker.host,
            "host_resource": f"host_{_safe_host_name(worker.host)}",
            "remote_video": remote_video,
            "remote_output": f".cache/video-analyzer/frame-worker/runs/{worker.output_dir.name}",
            "start": worker.start_seconds,
            "duration": worker.duration_seconds,
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "metadata_only": metadata_only,
            "python": _remote_worker_python(),
        }
        for index, worker in enumerate(workers)
    ]
    driver = r"""
import json
import os
import subprocess
import sys

import ray


@ray.remote(num_cpus=1)
def run_frame_worker(spec):
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        "bash",
        "-lc",
        "rm -rf {out} && {python} ~/.cache/video-analyzer/frame-worker/worker.py "
        "--video {video} --output {out} --start {start:.3f} --duration {duration:.3f} "
        "--sample-fps {sample_fps:.3f} --max-frames {max_frames}{metadata_arg}".format(
            python=spec["python"],
            out=spec["remote_output"],
            video=spec["remote_video"],
            start=float(spec["start"]),
            duration=float(spec["duration"]),
            sample_fps=float(spec["sample_fps"]),
            max_frames=int(spec["max_frames"]),
            metadata_arg=" --metadata-only" if spec.get("metadata_only") else "",
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "frame worker failed on {host} with code {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                host=spec["host"],
                code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    manifest = json.loads(result.stdout.strip().splitlines()[-1])
    return {"id": spec["id"], "host": spec["host"], "manifest": manifest}


def main():
    specs = json.loads(os.environ["JETSON_RAY_SPECS"])
    ray.init(address="auto")
    refs = [
        run_frame_worker.options(resources={spec["host_resource"]: 0.01, "frame_worker": 1}).remote(spec)
        for spec in specs
    ]
    print(json.dumps(ray.get(refs), ensure_ascii=False))


if __name__ == "__main__":
    main()
"""
    command = "JETSON_RAY_SPECS=" + shlex.quote(json.dumps(specs)) + " python3 - <<'PY'\n" + driver + "\nPY"
    result = subprocess.run(
        ["ssh", *(_ssh_host_args(head)), command],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Ray frame driver failed on {head} with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    by_id = {row["id"]: row["manifest"] for row in rows}
    return [(worker, by_id[str(index)]) for index, worker in enumerate(workers)]


def _materialize_selected_jetson_candidates(
    selected_by_worker: dict[str, dict[str, Any]],
    remote_video: str,
    backend: str,
) -> list[tuple[JetsonFrameWorker, dict[str, Any]]]:
    materialize_specs = []
    for worker_name, payload in selected_by_worker.items():
        worker = payload["worker"]
        candidates = payload["candidates"]
        if not candidates:
            continue
        materialize_specs.append((worker, worker_name, candidates))
    if not materialize_specs:
        return []
    with ThreadPoolExecutor(max_workers=len(materialize_specs)) as executor:
        futures = {
            executor.submit(_run_remote_materialization, worker.host, worker_name, remote_video, candidates): worker
            for worker, worker_name, candidates in materialize_specs
        }
        rows = []
        for future in as_completed(futures):
            worker = futures[future]
            rows.append((worker, future.result()))
        return rows


def _run_remote_materialization(
    host: str,
    worker_name: str,
    remote_video: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    remote_output = f".cache/video-analyzer/frame-worker/runs/{worker_name}/final"
    remote_payload = f"{remote_output}/selected_candidates.json"
    payload = json.dumps(candidates, ensure_ascii=False)
    _run_ssh(host, f"rm -rf {shlex.quote(remote_output)} && mkdir -p {shlex.quote(remote_output)}")
    subprocess.run(
        ["ssh", *(_ssh_host_args(host)), f"cat > {shlex.quote(remote_payload)}"],
        input=payload,
        text=True,
        check=True,
    )
    command = (
        "{python} ~/.cache/video-analyzer/frame-worker/worker.py "
        "--video {video} --output {out} --materialize-candidates-json {payload}"
    ).format(
        python=_remote_worker_python(),
        out=shlex.quote(remote_output),
        payload=shlex.quote(remote_payload),
        video=shlex.quote(remote_video),
    )
    result = _run_ssh(host, command, timeout=1800)
    return json.loads(result.stdout.strip().splitlines()[-1])


def _pull_remote_candidates(host: str, manifest: dict[str, Any], local_dir: Path) -> None:
    paths = [Path(item["path"]) for item in manifest.get("candidates", [])]
    if not paths:
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        subprocess.run(
            ["rsync", "-a", *_rsync_ssh_args(host), f"{_rsync_host(host)}:{path.as_posix()}", str(local_dir / path.name)],
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


def _remote_worker_python() -> str:
    return "$(if [ -x ~/.cache/video-analyzer/frame-worker/paper-venv/bin/python ]; then printf %s ~/.cache/video-analyzer/frame-worker/paper-venv/bin/python; else printf %s python3; fi)"


def _ssh_host_args(host: str) -> list[str]:
    base = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if host == "agx":
        return [*base, "-o", "HostKeyAlias=agx-lan", "agx@192.168.2.110"]
    if host == "nx3":
        return [*base, "-o", "ProxyCommand=none", "nx@nx3.taild500c8.ts.net"]
    return [*base, host]


def _rsync_host(host: str) -> str:
    if host == "agx":
        return "agx@192.168.2.110"
    if host == "nx3":
        return "nx@nx3.taild500c8.ts.net"
    return host


def _rsync_ssh_args(host: str) -> list[str]:
    if host == "agx":
        return [
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 "
            "-o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o HostKeyAlias=agx-lan",
        ]
    return []


def _safe_host_name(host: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in host)


def _video_cache_name(video_path: Path) -> str:
    stat = video_path.stat()
    return f"{video_path.stem}-{stat.st_size}-{int(stat.st_mtime)}{video_path.suffix}"
