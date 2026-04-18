from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_colmap import parse_cameras_text, parse_images_text, parse_points3d_text, summarize_sparse_model
from .models import ReconstructionSummary, RunPaths


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def export_sparse_artifacts(sparse_text_dir: Path, artifacts_dir: Path) -> tuple[Path, dict[str, Any]]:
    cameras = parse_cameras_text(sparse_text_dir)
    images = parse_images_text(sparse_text_dir)
    points3d = parse_points3d_text(sparse_text_dir)
    summary = summarize_sparse_model(cameras, images, points3d)

    cameras_path = artifacts_dir / "cameras.json"
    images_path = artifacts_dir / "images.json"
    points_path = artifacts_dir / "points3D.json"
    summary_path = artifacts_dir / "reconstruction_summary.json"
    manifest_path = artifacts_dir / "artifact_manifest.json"

    write_json(cameras_path, cameras)
    write_json(images_path, images)
    write_json(points_path, points3d)
    write_json(summary_path, summary.to_dict())

    manifest = {
        "cameras_path": str(cameras_path),
        "images_path": str(images_path),
        "points3d_path": str(points_path),
        "summary_path": str(summary_path),
        "summary": summary.to_dict(),
    }
    write_json(manifest_path, manifest)

    return summary_path, {
        **manifest,
        "manifest_path": str(manifest_path),
        "summary_object": summary,
    }


def collect_run_artifacts(paths: RunPaths) -> dict[str, Any]:
    files = sorted(str(path) for path in paths.artifacts_dir.rglob("*") if path.is_file())
    return {
        "workspace_dir": str(paths.workspace_dir),
        "artifacts_dir": str(paths.artifacts_dir),
        "artifact_files": files,
        "run_log_path": str(paths.run_log_path),
    }


def load_reconstruction_summary(paths: RunPaths) -> ReconstructionSummary:
    payload = read_json(paths.artifacts_dir / "reconstruction_summary.json")
    return ReconstructionSummary(**payload)
