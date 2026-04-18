from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import GsplatConfig, RunPaths


class GsplatTrainingError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_command(command: list[str], values: dict[str, str]) -> list[str]:
    return [item.format(**values) for item in command]


def _collect_artifacts(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file()
    )


def run_gsplat_training(
    cfg: GsplatConfig,
    *,
    paths: RunPaths,
    sparse_model_path: Path,
    sparse_text_path: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    log_path = paths.logs_dir / "gsplat_train.log"
    values = {
        "workspace_dir": str(paths.workspace_dir),
        "images_dir": str(paths.prepared_images_dir),
        "sparse_model_dir": str(sparse_model_path),
        "sparse_text_dir": str(sparse_text_path),
        "gsplat_dir": str(paths.gsplat_dir),
        "artifacts_dir": str(paths.artifacts_dir),
    }

    if not cfg.enabled:
        return {
            "enabled": False,
            "skipped": True,
            "log_path": str(log_path),
            "artifacts": [],
        }

    if not cfg.command:
        raise GsplatTrainingError(
            "gsplat.enabled is true, but no gsplat.command was provided in the config."
        )

    command = _format_command(cfg.command, values)
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_WORKSPACE_DIR": str(paths.workspace_dir),
            "PIPELINE_IMAGES_DIR": str(paths.prepared_images_dir),
            "PIPELINE_COLMAP_MODEL_DIR": str(sparse_model_path),
            "PIPELINE_COLMAP_TEXT_DIR": str(sparse_text_path),
            "PIPELINE_GSPLAT_DIR": str(paths.gsplat_dir),
            "PIPELINE_ARTIFACTS_DIR": str(paths.artifacts_dir),
        }
    )
    env.update(cfg.env)

    started_at = _utc_now()
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        return {
            "enabled": True,
            "skipped": False,
            "dry_run": True,
            "command": command,
            "working_dir": cfg.working_dir,
            "log_path": str(log_path),
            "started_at": started_at,
            "finished_at": started_at,
            "duration_seconds": 0.0,
            "artifacts": _collect_artifacts(paths.gsplat_dir),
        }

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=cfg.working_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    finished_at = _utc_now()
    duration_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise GsplatTrainingError(
            f"gsplat command failed with exit code {completed.returncode}. "
            f"See log: {log_path}"
        )

    return {
        "enabled": True,
        "skipped": False,
        "command": command,
        "working_dir": cfg.working_dir,
        "log_path": str(log_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "artifacts": _collect_artifacts(paths.gsplat_dir),
    }
