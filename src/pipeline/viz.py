from __future__ import annotations

import random
from pathlib import Path

from .io_colmap import parse_images_text, parse_points3d_text


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install it with 'pip install matplotlib'."
        ) from exc
    return plt


def show_image_samples(images_dir: str | Path, max_images: int = 9):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image previews. Install it with 'pip install pillow'."
        ) from exc

    images_dir = Path(images_dir)
    image_files = sorted(path for path in images_dir.iterdir() if path.is_file())[:max_images]
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    plt = _require_matplotlib()
    columns = min(3, len(image_files))
    rows = (len(image_files) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    if hasattr(axes, "flatten"):
        axes = list(axes.flatten())
    else:
        axes = [axes]

    for axis, image_path in zip(axes, image_files):
        with Image.open(image_path) as image:
            axis.imshow(image)
        axis.set_title(image_path.name)
        axis.axis("off")

    for axis in axes[len(image_files):]:
        axis.axis("off")

    figure.tight_layout()
    return figure


def show_sparse_model(model_dir: str | Path, max_points: int = 50000, point_size: float = 1.0):
    plt = _require_matplotlib()
    model_dir = Path(model_dir)
    points = parse_points3d_text(model_dir)
    images = parse_images_text(model_dir)
    if not points:
        raise RuntimeError(f"No sparse points found in {model_dir}")

    if len(points) > max_points:
        points = random.sample(points, max_points)

    xs = [point["xyz"][0] for point in points]
    ys = [point["xyz"][1] for point in points]
    zs = [point["xyz"][2] for point in points]
    colors = [
        [channel / 255.0 for channel in point["rgb"]]
        for point in points
    ]
    camera_centers = [image["tvec"] for image in images]

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(xs, ys, zs, s=point_size, c=colors, alpha=0.8)
    if camera_centers:
        axis.scatter(
            [center[0] for center in camera_centers],
            [center[1] for center in camera_centers],
            [center[2] for center in camera_centers],
            c="red",
            s=12,
            label="Cameras",
        )
        axis.legend(loc="upper right")
    axis.set_title(f"Sparse Model: {model_dir}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    return figure


def show_camera_trajectory(model_dir: str | Path):
    plt = _require_matplotlib()
    model_dir = Path(model_dir)
    images = parse_images_text(model_dir)
    if not images:
        raise RuntimeError(f"No reconstructed images found in {model_dir}")

    centers = [image["tvec"] for image in images]
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    zs = [center[2] for center in centers]

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(xs, ys, zs, color="tab:red")
    axis.scatter(xs, ys, zs, color="tab:red", s=10)
    axis.set_title("Camera Trajectory")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    return figure
