from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .export import write_json
from .models import ImagePrepConfig, ReconstructionJob, RunPaths
from .ingest import _iter_image_files


def clean_directory(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def resize_and_copy_image(source: Path, destination: Path, resize_max: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for resize_max support. Install it with 'pip install pillow'."
        ) from exc

    with Image.open(source) as image:
        image.thumbnail((resize_max, resize_max))
        image.save(destination)


def link_or_copy_images(
    image_files: list[Path],
    output_dir: Path,
    image_prep: ImagePrepConfig,
) -> None:
    clean_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in image_files:
        destination = output_dir / source.name
        if image_prep.resize_max:
            resize_and_copy_image(source, destination, image_prep.resize_max)
            continue

        if image_prep.copy_mode == "copy" or sys.platform.startswith("win"):
            shutil.copy2(source, destination)
            continue

        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)


def extract_video_frames(
    input_video: Path,
    output_dir: Path,
    image_prep: ImagePrepConfig,
    dry_run: bool,
) -> dict[str, Any]:
    ffmpeg_binary = shutil.which(image_prep.ffmpeg_binary)
    if not ffmpeg_binary:
        raise RuntimeError(
            f"ffmpeg binary '{image_prep.ffmpeg_binary}' was not found on PATH."
        )

    clean_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = []
    stride = max(1, image_prep.frame_stride)
    if stride > 1:
        vf_parts.append(f"select='not(mod(n\\,{stride}))'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")
    if image_prep.resize_max:
        resize = image_prep.resize_max
        vf_parts.append(
            f"scale='if(gt(iw,ih),min(iw,{resize}),-2)':'if(gt(ih,iw),min(ih,{resize}),-2)'"
        )

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_video),
    ]
    if vf_parts:
        command.extend(["-vf", ",".join(vf_parts)])
    command.append(str(output_dir / image_prep.video_frame_pattern))

    if dry_run:
        return {
            "dry_run": True,
            "command": command,
            "frames_dir": str(output_dir),
        }

    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg frame extraction failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return {
        "dry_run": False,
        "command": command,
        "frames_dir": str(output_dir),
    }


def summarize_prepared_images(prepared_images_dir: Path, patterns: tuple[str, ...]) -> dict[str, Any]:
    images = _iter_image_files(prepared_images_dir, patterns) if prepared_images_dir.exists() else []
    return {
        "prepared_images_dir": str(prepared_images_dir),
        "image_count": len(images),
        "sample_images": [str(path) for path in images[:10]],
    }


def prepare_images(job: ReconstructionJob, paths: RunPaths) -> Path:
    config = job.config
    input_path = Path(config.input_path)
    input_type = config.input_type.lower()

    if input_type == "video":
        extract_video_frames(
            input_video=input_path,
            output_dir=paths.frames_dir,
            image_prep=config.image_prep,
            dry_run=config.dry_run,
        )
        if config.dry_run:
            return paths.prepared_images_dir
        frame_files = _iter_image_files(paths.frames_dir, config.image_prep.image_glob_patterns)
        if not frame_files:
            raise RuntimeError(f"No frames were extracted from {input_path}")
        staged_prep = ImagePrepConfig(
            frame_stride=config.image_prep.frame_stride,
            resize_max=None,
            copy_mode=config.image_prep.copy_mode,
            ffmpeg_binary=config.image_prep.ffmpeg_binary,
            video_frame_pattern=config.image_prep.video_frame_pattern,
            image_glob_patterns=config.image_prep.image_glob_patterns,
        )
        link_or_copy_images(frame_files, paths.prepared_images_dir, staged_prep)
        return paths.prepared_images_dir

    if not input_path.is_dir():
        raise FileNotFoundError(f"Expected input_path to be a directory, got {input_path}")

    image_files = _iter_image_files(input_path, config.image_prep.image_glob_patterns)
    if not image_files:
        raise RuntimeError(f"No images found in {input_path}")

    if config.dry_run:
        return paths.prepared_images_dir

    link_or_copy_images(image_files, paths.prepared_images_dir, config.image_prep)
    return paths.prepared_images_dir


def run_prepare(job: ReconstructionJob, paths: RunPaths) -> dict[str, Any]:
    prepared_dir = prepare_images(job, paths)
    summary = summarize_prepared_images(prepared_dir, job.config.image_prep.image_glob_patterns)
    summary.update(
        {
            "scene_name": job.config.scene_name,
            "input_type": job.config.input_type,
            "frames_dir": str(paths.frames_dir),
            "resize_max": job.config.image_prep.resize_max,
            "frame_stride": job.config.image_prep.frame_stride,
        }
    )
    write_json(paths.artifacts_dir / "prepare_summary.json", summary)
    return summary
