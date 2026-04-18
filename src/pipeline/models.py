from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value


@dataclass
class ImagePrepConfig:
    frame_stride: int = 1
    resize_max: int | None = None
    copy_mode: str = "symlink"
    ffmpeg_binary: str = "ffmpeg"
    video_frame_pattern: str = "frame_%06d.png"
    image_glob_patterns: tuple[str, ...] = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
        "*.BMP",
        "*.TIF",
        "*.TIFF",
    )


@dataclass
class ColmapConfig:
    binary: str = "colmap"
    mode: str = "global"
    matcher: str = "exhaustive"
    camera_model: str = "OPENCV"
    single_camera: bool = True
    use_gpu: bool | None = None
    use_view_graph_calibrator: bool = False
    feature_extractor_options: dict[str, Any] = field(default_factory=dict)
    matcher_options: dict[str, Any] = field(default_factory=dict)
    mapper_options: dict[str, Any] = field(default_factory=dict)
    global_mapper_options: dict[str, Any] = field(default_factory=dict)
    view_graph_calibrator_options: dict[str, Any] = field(default_factory=dict)
    model_converter_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GsplatConfig:
    enabled: bool = False
    command: list[str] = field(default_factory=list)
    working_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    scene_name: str
    input_type: str
    input_path: str
    workspace_dir: str
    frames_dir: str | None = None
    prepared_images_dir: str | None = None
    dry_run: bool = False
    keep_existing_outputs: bool = True
    image_prep: ImagePrepConfig = field(default_factory=ImagePrepConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    gsplat: GsplatConfig = field(default_factory=GsplatConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _stringify_paths(asdict(self))


@dataclass
class ReconstructionJob:
    config_path: Path
    config: PipelineConfig
    raw_config: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReconstructionJob":
        from .reconstruct import load_reconstruction_job

        return load_reconstruction_job(path)

    @property
    def scene_name(self) -> str:
        return self.config.scene_name


@dataclass
class RunPaths:
    workspace_dir: Path
    frames_dir: Path
    prepared_images_dir: Path
    colmap_db_dir: Path
    sparse_dir: Path
    sparse_text_dir: Path
    dense_dir: Path
    gsplat_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    database_path: Path
    global_database_path: Path
    run_config_path: Path
    run_log_path: Path

    def to_dict(self) -> dict[str, str]:
        return _stringify_paths(asdict(self))


@dataclass
class CommandExecution:
    name: str
    command: list[str]
    log_path: Path
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _stringify_paths(asdict(self))


@dataclass
class ReconstructionSummary:
    num_cameras: int
    num_images: int
    num_points3d: int
    mean_track_length: float | None = None
    mean_reprojection_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReconstructionResult:
    scene_name: str
    workspace_dir: Path
    images_dir: Path
    sparse_model_path: Path
    sparse_text_path: Path
    summary_path: Path
    run_log_path: Path
    summary: ReconstructionSummary
    commands: list[CommandExecution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "workspace_dir": str(self.workspace_dir),
            "images_dir": str(self.images_dir),
            "sparse_model_path": str(self.sparse_model_path),
            "sparse_text_path": str(self.sparse_text_path),
            "summary_path": str(self.summary_path),
            "run_log_path": str(self.run_log_path),
            "summary": self.summary.to_dict(),
            "commands": [command.to_dict() for command in self.commands],
        }
