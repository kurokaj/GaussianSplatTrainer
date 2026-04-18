from __future__ import annotations

from pathlib import Path

from .models import ReconstructionSummary


def _read_non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_cameras_text(model_dir: Path) -> list[dict[str, object]]:
    cameras_path = model_dir / "cameras.txt"
    if not cameras_path.exists():
        return []

    cameras: list[dict[str, object]] = []
    for line in _read_non_comment_lines(cameras_path):
        parts = line.split()
        cameras.append(
            {
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(item) for item in parts[4:]],
            }
        )
    return cameras


def parse_images_text(model_dir: Path) -> list[dict[str, object]]:
    images_path = model_dir / "images.txt"
    if not images_path.exists():
        return []

    lines = _read_non_comment_lines(images_path)
    images: list[dict[str, object]] = []
    for index in range(0, len(lines), 2):
        parts = lines[index].split()
        images.append(
            {
                "image_id": int(parts[0]),
                "qvec": [float(value) for value in parts[1:5]],
                "tvec": [float(value) for value in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        )
    return images


def parse_points3d_text(model_dir: Path) -> list[dict[str, object]]:
    points_path = model_dir / "points3D.txt"
    if not points_path.exists():
        return []

    points: list[dict[str, object]] = []
    for line in _read_non_comment_lines(points_path):
        parts = line.split()
        track = parts[8:]
        points.append(
            {
                "point3d_id": int(parts[0]),
                "xyz": [float(value) for value in parts[1:4]],
                "rgb": [int(value) for value in parts[4:7]],
                "error": float(parts[7]),
                "track_length": len(track) // 2,
            }
        )
    return points


def summarize_sparse_model(
    cameras: list[dict[str, object]],
    images: list[dict[str, object]],
    points3d: list[dict[str, object]],
) -> ReconstructionSummary:
    mean_track_length = None
    mean_reprojection_error = None

    if points3d:
        mean_track_length = sum(point["track_length"] for point in points3d) / len(points3d)
        mean_reprojection_error = sum(point["error"] for point in points3d) / len(points3d)

    return ReconstructionSummary(
        num_cameras=len(cameras),
        num_images=len(images),
        num_points3d=len(points3d),
        mean_track_length=mean_track_length,
        mean_reprojection_error=mean_reprojection_error,
    )
