from __future__ import annotations

import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ColmapConfig, CommandExecution


class ColmapError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _options_to_args(options: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in options.items():
        if value is None:
            continue
        args.extend([f"--{key}", _coerce_option_value(value)])
    return args


def ensure_colmap_available(binary: str) -> None:
    if shutil.which(binary):
        return
    raise ColmapError(
        f"COLMAP binary '{binary}' was not found on PATH. "
        "Set colmap.binary in the config or install COLMAP on the instance."
    )


def run_colmap_command(
    *,
    binary: str,
    command_name: str,
    option_map: dict[str, Any],
    log_path: Path,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> CommandExecution:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [binary, command_name, *_options_to_args(option_map)]
    started_at = _utc_now()
    started = time.perf_counter()

    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        finished_at = _utc_now()
        return CommandExecution(
            name=command_name,
            command=command,
            log_path=log_path,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=0.0,
            return_code=0,
            dry_run=True,
        )

    ensure_colmap_available(binary)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    finished_at = _utc_now()
    duration_seconds = time.perf_counter() - started
    execution = CommandExecution(
        name=command_name,
        command=command,
        log_path=log_path,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        return_code=completed.returncode,
        dry_run=False,
    )
    if completed.returncode != 0:
        raise ColmapError(
            f"COLMAP command '{command_name}' failed with exit code "
            f"{completed.returncode}. See log: {log_path}"
        )
    return execution


def matcher_command_name(matcher: str) -> str:
    suffix = "_matcher"
    if matcher.endswith(suffix):
        return matcher
    return f"{matcher}{suffix}"


def run_feature_extractor(
    cfg: ColmapConfig,
    *,
    images_path: Path,
    database_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> CommandExecution:
    options: dict[str, Any] = {
        "database_path": str(database_path),
        "image_path": str(images_path),
        "ImageReader.camera_model": cfg.camera_model,
        "ImageReader.single_camera": cfg.single_camera,
    }
    if cfg.use_gpu is not None:
        options["FeatureExtraction.use_gpu"] = cfg.use_gpu
    options.update(cfg.feature_extractor_options)
    return run_colmap_command(
        binary=cfg.binary,
        command_name="feature_extractor",
        option_map=options,
        log_path=log_path,
        dry_run=dry_run,
    )


def run_matcher(
    cfg: ColmapConfig,
    *,
    database_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> CommandExecution:
    options: dict[str, Any] = {
        "database_path": str(database_path),
    }
    if cfg.use_gpu is not None:
        options["FeatureMatching.use_gpu"] = cfg.use_gpu
    options.update(cfg.matcher_options)
    return run_colmap_command(
        binary=cfg.binary,
        command_name=matcher_command_name(cfg.matcher),
        option_map=options,
        log_path=log_path,
        dry_run=dry_run,
    )


def run_view_graph_calibrator(
    cfg: ColmapConfig,
    *,
    database_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> CommandExecution:
    options: dict[str, Any] = {
        "database_path": str(database_path),
    }
    options.update(cfg.view_graph_calibrator_options)
    return run_colmap_command(
        binary=cfg.binary,
        command_name="view_graph_calibrator",
        option_map=options,
        log_path=log_path,
        dry_run=dry_run,
    )


def run_mapper(
    cfg: ColmapConfig,
    *,
    images_path: Path,
    database_path: Path,
    output_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> CommandExecution:
    options: dict[str, Any] = {
        "database_path": str(database_path),
        "image_path": str(images_path),
        "output_path": str(output_path),
    }
    options.update(cfg.mapper_options)
    return run_colmap_command(
        binary=cfg.binary,
        command_name="mapper",
        option_map=options,
        log_path=log_path,
        dry_run=dry_run,
    )


def run_global_mapper(
    cfg: ColmapConfig,
    *,
    images_path: Path,
    database_path: Path,
    output_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> CommandExecution:
    options: dict[str, Any] = {
        "database_path": str(database_path),
        "image_path": str(images_path),
        "output_path": str(output_path),
    }
    options.update(cfg.global_mapper_options)
    return run_colmap_command(
        binary=cfg.binary,
        command_name="global_mapper",
        option_map=options,
        log_path=log_path,
        dry_run=dry_run,
    )


def export_model_to_text(
    cfg: ColmapConfig,
    *,
    input_path: Path,
    output_path: Path,
    log_path: Path,
    dry_run: bool = False,
) -> CommandExecution:
    options: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_type": "TXT",
    }
    options.update(cfg.model_converter_options)
    return run_colmap_command(
        binary=cfg.binary,
        command_name="model_converter",
        option_map=options,
        log_path=log_path,
        dry_run=dry_run,
    )


def locate_sparse_model(root: Path) -> Path:
    markers = (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    )
    if any((root / marker).exists() for marker in markers):
        return root

    for child in sorted(root.iterdir()):
        if child.is_dir() and any((child / marker).exists() for marker in markers):
            return child

    raise ColmapError(f"Could not find a sparse model under {root}")
