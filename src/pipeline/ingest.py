from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .export import write_json
from .models import ReconstructionJob, RunPaths


def _iter_image_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted({item.resolve() for item in files})


def _sample_paths(paths: list[Path], limit: int = 10) -> list[str]:
    return [str(path) for path in paths[:limit]]


def _infer_ffprobe_binary(ffmpeg_binary: str) -> str | None:
    ffmpeg_path = shutil.which(ffmpeg_binary)
    if not ffmpeg_path:
        return None
    candidate = Path(ffmpeg_path).with_name("ffprobe")
    if candidate.exists():
        return str(candidate)
    return shutil.which("ffprobe")


def probe_video(video_path: Path, ffmpeg_binary: str = "ffmpeg") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(video_path),
        "exists": video_path.exists(),
        "size_bytes": video_path.stat().st_size if video_path.exists() else None,
        "ffprobe_available": False,
    }
    ffprobe_binary = _infer_ffprobe_binary(ffmpeg_binary)
    if not ffprobe_binary or not video_path.exists():
        return summary

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        summary["ffprobe_error"] = completed.stderr.strip() or completed.stdout.strip()
        return summary

    payload = json.loads(completed.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    summary.update(
        {
            "ffprobe_available": True,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "duration_seconds": stream.get("duration"),
            "num_frames": stream.get("nb_frames"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "raw_frame_rate": stream.get("r_frame_rate"),
        }
    )
    return summary


def ingest_images(job: ReconstructionJob) -> dict[str, Any]:
    input_dir = Path(job.config.input_path)
    image_files = _iter_image_files(input_dir, job.config.image_prep.image_glob_patterns)
    return {
        "scene_name": job.config.scene_name,
        "input_type": "images",
        "input_path": str(input_dir),
        "image_count": len(image_files),
        "sample_images": _sample_paths(image_files),
    }


def ingest_video(job: ReconstructionJob) -> dict[str, Any]:
    input_path = Path(job.config.input_path)
    summary = probe_video(input_path, ffmpeg_binary=job.config.image_prep.ffmpeg_binary)
    summary.update(
        {
            "scene_name": job.config.scene_name,
            "input_type": "video",
        }
    )
    return summary


def run_ingest(job: ReconstructionJob, paths: RunPaths) -> dict[str, Any]:
    input_type = job.config.input_type.lower()
    if input_type == "video":
        summary = ingest_video(job)
    elif input_type == "images":
        summary = ingest_images(job)
    else:
        raise ValueError(f"Unsupported input_type '{job.config.input_type}'. Expected 'images' or 'video'.")

    summary["workspace_dir"] = str(paths.workspace_dir)
    summary["frames_dir"] = str(paths.frames_dir)
    summary["prepared_images_dir"] = str(paths.prepared_images_dir)
    write_json(paths.artifacts_dir / "ingest_summary.json", summary)
    return summary
