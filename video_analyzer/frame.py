from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging
import subprocess
import tempfile

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None

logger = logging.getLogger(__name__)

@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float


@dataclass
class FrameCandidate:
    source_index: int
    image_path: Path
    timestamp: float
    density_score: float
    coverage_score: float

class VideoProcessor:
    # Class constants
    FRAME_DIFFERENCE_THRESHOLD = 10.0
    
    def __init__(self, video_path: Path, output_dir: Path, model: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.model = model
        self.frames: List[Frame] = []
        
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames using absolute difference."""
        if cv2 is None or np is None:
            return 0.0
        if frame1 is None or frame2 is None:
            return 0.0
        
        # Convert to grayscale for simpler comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference and mean
        diff = cv2.absdiff(gray1, gray2)
        score = np.mean(diff)
        
        return float(score)

    def _is_keyframe(self, current_frame: np.ndarray, prev_frame: np.ndarray, threshold: float = FRAME_DIFFERENCE_THRESHOLD) -> bool:
        """Determine if frame is significantly different from previous frame."""
        if prev_frame is None:
            return True
            
        score = self._calculate_frame_difference(current_frame, prev_frame)
        return score > threshold

    def extract_keyframes(self, frames_per_minute: int = 10, duration: Optional[float] = None, max_frames: Optional[int] = None) -> List[Frame]:
        """Extract keyframes from video targeting a specific number of frames per minute."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if cv2 is None or np is None:
            return self._extract_ffmpeg_keyframes(frames_per_minute, duration, max_frames)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.warning("OpenCV could not open %s; falling back to ffmpeg extraction", self.video_path)
            return self._extract_ffmpeg_keyframes(frames_per_minute, duration, max_frames)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if duration:
            video_duration = min(duration, video_duration)
            total_frames = int(min(total_frames, duration * fps))
        
        # Calculate target number of frames
        target_frames = max(1, min(
            int((video_duration / 60) * frames_per_minute),
            total_frames,
            max_frames if max_frames is not None else float('inf')
        ))
        
        # Calculate adaptive sampling interval
        sample_interval = max(1, total_frames // (target_frames * 2))
        
        frame_candidates = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                score = self._calculate_frame_difference(frame, prev_frame)
                if score > self.FRAME_DIFFERENCE_THRESHOLD:
                    frame_candidates.append((frame_count, frame, score))
                prev_frame = frame.copy()
                
            frame_count += 1
            
        cap.release()
        if not frame_candidates:
            logger.warning("OpenCV decoded no useful frames from %s; falling back to ffmpeg extraction", self.video_path)
            return self._extract_ffmpeg_keyframes(frames_per_minute, duration, max_frames)
        
        # Select the most significant frames by score, then restore chronological order
        selected_candidates = sorted(frame_candidates, key=lambda x: x[2], reverse=True)[:target_frames]

        # If max_frames is specified, sample evenly across the candidates
        if max_frames is not None and max_frames < len(selected_candidates):
            step = len(selected_candidates) / max_frames
            selected_frames = [selected_candidates[int(i * step)] for i in range(max_frames)]
        else:
            selected_frames = selected_candidates

        # Re-sort by frame number so frames on disk and in the JSON are chronological
        selected_frames = sorted(selected_frames, key=lambda x: x[0])

        self.frames = []
        for idx, (frame_num, frame, score) in enumerate(selected_frames):
            frame_path = self.output_dir / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            timestamp = frame_num / fps
            self.frames.append(Frame(idx, frame_path, timestamp, score))
        
        logger.info(f"Extracted {len(self.frames)} frames from video (target was {target_frames})")
        return self.frames

    def _probe_duration(self, duration: Optional[float]) -> float:
        if duration:
            return duration
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(self.video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return max(float(result.stdout.strip()), 0.0)
        except Exception:
            return 0.0

    def _extract_ffmpeg_images(
        self,
        frames_per_minute: int,
        duration: Optional[float],
        temp_dir: Path,
    ) -> tuple[List[Path], float]:
        sample_rate = max(frames_per_minute / 60.0, 0.2)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.video_path),
        ]
        if duration:
            command.extend(["-t", str(duration)])
        command.extend(
            [
                "-vf",
                f"fps={sample_rate}",
                "-q:v",
                "2",
                str(temp_dir / "frame_%06d.jpg"),
            ]
        )
        subprocess.run(command, check=True, capture_output=True)
        return sorted(temp_dir.glob("frame_*.jpg")), sample_rate

    def _select_ffmpeg_candidates(
        self,
        image_paths: List[Path],
        sample_rate: float,
        max_frames: Optional[int],
        similarity_threshold: float,
    ) -> List[tuple[int, Path, float, float]]:
        candidates = []
        last_frame = None
        for idx, image_path in enumerate(image_paths):
            timestamp = idx / sample_rate if sample_rate else 0.0
            score = 1.0
            if cv2 is not None:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                score = self._calculate_frame_difference(frame, last_frame)
                if last_frame is not None and score < similarity_threshold:
                    continue
                last_frame = frame
            candidates.append((idx, image_path, timestamp, score))

        if max_frames is not None:
            candidates = self._select_density_budget(candidates, max_frames)
        return candidates

    def _select_density_budget(
        self,
        candidates: List[tuple[int, Path, float, float]],
        max_frames: int,
        coverage_interval_seconds: float = 20.0,
    ) -> List[tuple[int, Path, float, float]]:
        """Keep coverage frames, then spend the rest on high-change dense moments."""
        if len(candidates) <= max_frames:
            return candidates

        selected_indexes = set()
        last_bucket = None
        for idx, (_, _, timestamp, _) in enumerate(candidates):
            bucket = int(timestamp // coverage_interval_seconds)
            if bucket != last_bucket:
                selected_indexes.add(idx)
                last_bucket = bucket

        remaining_slots = max(max_frames - len(selected_indexes), 0)
        ranked_by_density = sorted(
            (idx for idx in range(len(candidates)) if idx not in selected_indexes),
            key=lambda idx: candidates[idx][3],
            reverse=True,
        )
        selected_indexes.update(ranked_by_density[:remaining_slots])

        if len(selected_indexes) > max_frames:
            selected_indexes = set(sorted(selected_indexes)[:max_frames])

        return [candidates[idx] for idx in sorted(selected_indexes)]

    def _extract_ffmpeg_keyframes(
        self,
        frames_per_minute: int,
        duration: Optional[float],
        max_frames: Optional[int],
        similarity_threshold: float = 2.0,
    ) -> List[Frame]:
        """Decode frames through ffmpeg for codecs OpenCV cannot handle, such as AV1."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            image_paths, sample_rate = self._extract_ffmpeg_images(frames_per_minute, duration, Path(temp))
            candidates = self._select_ffmpeg_candidates(
                image_paths=image_paths,
                sample_rate=sample_rate,
                max_frames=max_frames,
                similarity_threshold=similarity_threshold,
            )

            self.frames = []
            for idx, (_, image_path, timestamp, score) in enumerate(candidates):
                frame_path = self.output_dir / f"frame_{idx}.jpg"
                frame_path.write_bytes(image_path.read_bytes())
                self.frames.append(Frame(idx, frame_path, timestamp, score))

        logger.info("Extracted %s frames with ffmpeg fallback", len(self.frames))
        return self.frames

    def extract_screen_keyframes(
        self,
        frames_per_minute: int = 60,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        change_threshold: float = 6.0,
        min_gap_seconds: float = 1.0,
        similarity_threshold: float = 2.0,
    ) -> List[Frame]:
        """Extract keyframes with a screen-recording friendly fixed+change strategy."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if cv2 is None or np is None:
            return self._extract_ffmpeg_keyframes(frames_per_minute, duration, max_frames, similarity_threshold)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.warning("OpenCV could not open %s; falling back to ffmpeg extraction", self.video_path)
            return self._extract_ffmpeg_keyframes(frames_per_minute, duration, max_frames, similarity_threshold)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps else 0
        if duration:
            video_duration = min(duration, video_duration)
            total_frames = int(min(total_frames, duration * fps))

        sample_interval = max(1, int((60 / max(frames_per_minute, 1)) * fps))
        min_gap_frames = max(1, int(min_gap_seconds * fps))
        candidates = []
        prev_sample = None
        last_selected_num = -min_gap_frames
        frame_num = 0

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            should_sample = frame_num == 0 or frame_num % sample_interval == 0
            score = self._calculate_frame_difference(frame, prev_sample)
            changed = prev_sample is not None and score >= change_threshold

            if (should_sample or changed) and frame_num - last_selected_num >= min_gap_frames:
                if not candidates:
                    candidates.append((frame_num, frame.copy(), score))
                    last_selected_num = frame_num
                else:
                    last_frame = candidates[-1][1]
                    similarity = self._calculate_frame_difference(frame, last_frame)
                    if similarity >= similarity_threshold:
                        candidates.append((frame_num, frame.copy(), score))
                        last_selected_num = frame_num

            if should_sample or changed:
                prev_sample = frame.copy()
            frame_num += 1

        cap.release()
        if not candidates:
            logger.warning("OpenCV decoded no screen keyframes from %s; falling back to ffmpeg extraction", self.video_path)
            return self._extract_ffmpeg_keyframes(frames_per_minute, duration, max_frames, similarity_threshold)

        if max_frames is not None:
            candidates = self._select_opencv_density_budget(candidates, fps, max_frames)

        self.frames = []
        for idx, (source_frame_num, frame, score) in enumerate(candidates):
            frame_path = self.output_dir / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            self.frames.append(Frame(idx, frame_path, source_frame_num / fps, score))

        logger.info("Extracted %s screen keyframes", len(self.frames))
        return self.frames

    def _select_opencv_density_budget(
        self,
        candidates: List[tuple[int, np.ndarray, float]],
        fps: float,
        max_frames: int,
        coverage_interval_seconds: float = 20.0,
    ) -> List[tuple[int, np.ndarray, float]]:
        if len(candidates) <= max_frames:
            return candidates

        selected_indexes = set()
        last_bucket = None
        for idx, (source_frame_num, _, _) in enumerate(candidates):
            timestamp = source_frame_num / fps if fps else 0.0
            bucket = int(timestamp // coverage_interval_seconds)
            if bucket != last_bucket:
                selected_indexes.add(idx)
                last_bucket = bucket

        remaining_slots = max(max_frames - len(selected_indexes), 0)
        ranked_by_density = sorted(
            (idx for idx in range(len(candidates)) if idx not in selected_indexes),
            key=lambda idx: candidates[idx][2],
            reverse=True,
        )
        selected_indexes.update(ranked_by_density[:remaining_slots])

        if len(selected_indexes) > max_frames:
            selected_indexes = set(sorted(selected_indexes)[:max_frames])

        return [candidates[idx] for idx in sorted(selected_indexes)]
