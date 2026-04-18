from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .colmap_cli import (
    export_model_to_text,
    locate_sparse_model,
    run_feature_extractor,
    run_global_mapper,
    run_mapper,
    run_matcher,
    run_view_graph_calibrator,
)
from .export import export_sparse_artifacts, write_json
from .ingest import run_ingest
from .models import (
    ColmapConfig,
    GsplatConfig,
    ImagePrepConfig,
    PipelineConfig,
    ReconstructionJob,
    ReconstructionResult,
    ReconstructionSummary,
    RunPaths,
)
from .prepare import run_prepare
from .train_gsplat import run_gsplat_training


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load config files. Install it with 'pip install pyyaml'."
        ) from exc
    return yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(config_path: Path) -> dict[str, Any]:
    yaml = _require_yaml()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    extends = data.pop("extends", None)
    if not extends:
        return data

    if isinstance(extends, str):
        base_paths = [extends]
    else:
        base_paths = list(extends)

    merged: dict[str, Any] = {}
    for item in base_paths:
        base_path = (config_path.parent / item).resolve()
        merged = _deep_merge(merged, _load_yaml_with_extends(base_path))
    return _deep_merge(merged, data)


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _job_from_data(config_path: Path, data: dict[str, Any]) -> ReconstructionJob:
    base_dir = config_path.parent.resolve()

    image_prep_data = data.get("image_prep", {})
    colmap_data = data.get("colmap", {})
    gsplat_data = data.get("gsplat", {})

    config = PipelineConfig(
        scene_name=data["scene_name"],
        input_type=data["input_type"],
        input_path=_resolve_path(data["input_path"], base_dir) or data["input_path"],
        workspace_dir=_resolve_path(data["workspace_dir"], base_dir) or data["workspace_dir"],
        frames_dir=_resolve_path(data.get("frames_dir"), base_dir),
        prepared_images_dir=_resolve_path(data.get("prepared_images_dir"), base_dir),
        dry_run=bool(data.get("dry_run", False)),
        keep_existing_outputs=bool(data.get("keep_existing_outputs", True)),
        image_prep=ImagePrepConfig(
            frame_stride=int(image_prep_data.get("frame_stride", 1)),
            resize_max=image_prep_data.get("resize_max"),
            copy_mode=image_prep_data.get("copy_mode", "symlink"),
            ffmpeg_binary=image_prep_data.get("ffmpeg_binary", "ffmpeg"),
            video_frame_pattern=image_prep_data.get("video_frame_pattern", "frame_%06d.png"),
        ),
        colmap=ColmapConfig(
            binary=colmap_data.get("binary", "colmap"),
            mode=colmap_data.get("mode", "global"),
            matcher=colmap_data.get("matcher", "exhaustive"),
            camera_model=colmap_data.get("camera_model", "OPENCV"),
            single_camera=bool(colmap_data.get("single_camera", True)),
            use_gpu=_coerce_bool(colmap_data.get("use_gpu")),
            use_view_graph_calibrator=bool(colmap_data.get("use_view_graph_calibrator", False)),
            feature_extractor_options=dict(colmap_data.get("feature_extractor_options", {})),
            matcher_options=dict(colmap_data.get("matcher_options", {})),
            mapper_options=dict(colmap_data.get("mapper_options", {})),
            global_mapper_options=dict(colmap_data.get("global_mapper_options", {})),
            view_graph_calibrator_options=dict(colmap_data.get("view_graph_calibrator_options", {})),
            model_converter_options=dict(colmap_data.get("model_converter_options", {})),
        ),
        gsplat=GsplatConfig(
            enabled=bool(gsplat_data.get("enabled", False)),
            command=list(gsplat_data.get("command", [])),
            working_dir=_resolve_path(gsplat_data.get("working_dir"), base_dir),
            env={str(key): str(value) for key, value in dict(gsplat_data.get("env", {})).items()},
        ),
        metadata=dict(data.get("metadata", {})),
    )
    return ReconstructionJob(
        config_path=config_path.resolve(),
        config=config,
        raw_config=data,
    )


def load_reconstruction_job(config_path: str | Path) -> ReconstructionJob:
    path = Path(config_path).resolve()
    data = _load_yaml_with_extends(path)
    return _job_from_data(path, data)


def build_run_paths(config: PipelineConfig) -> RunPaths:
    workspace_dir = Path(config.workspace_dir)
    frames_dir = Path(config.frames_dir) if config.frames_dir else workspace_dir / "frames"
    prepared_images_dir = (
        Path(config.prepared_images_dir)
        if config.prepared_images_dir
        else workspace_dir / "prepared"
    )
    colmap_db_dir = workspace_dir / "colmap_db"
    sparse_dir = workspace_dir / "sparse"
    sparse_text_dir = workspace_dir / "artifacts" / "colmap_text"
    dense_dir = workspace_dir / "dense"
    gsplat_dir = workspace_dir / "gsplat"
    artifacts_dir = workspace_dir / "artifacts"
    logs_dir = workspace_dir / "logs"
    return RunPaths(
        workspace_dir=workspace_dir,
        frames_dir=frames_dir,
        prepared_images_dir=prepared_images_dir,
        colmap_db_dir=colmap_db_dir,
        sparse_dir=sparse_dir,
        sparse_text_dir=sparse_text_dir,
        dense_dir=dense_dir,
        gsplat_dir=gsplat_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        database_path=colmap_db_dir / "database.db",
        global_database_path=colmap_db_dir / "database_global.db",
        run_config_path=workspace_dir / "run_config.yaml",
        run_log_path=workspace_dir / "run_log.json",
    )


def _ensure_workspace(paths: RunPaths) -> None:
    for path in (
        paths.workspace_dir,
        paths.frames_dir,
        paths.prepared_images_dir,
        paths.colmap_db_dir,
        paths.sparse_dir,
        paths.sparse_text_dir,
        paths.dense_dir,
        paths.gsplat_dir,
        paths.artifacts_dir,
        paths.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    yaml = _require_yaml()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_reconstruction(job: ReconstructionJob) -> ReconstructionResult:
    config = job.config
    paths = build_run_paths(config)
    _ensure_workspace(paths)
    _write_yaml(paths.run_config_path, job.raw_config)

    ingest_summary = run_ingest(job, paths)
    prepare_summary = run_prepare(job, paths)
    prepared_images_dir = Path(prepare_summary["prepared_images_dir"])
    commands = []

    commands.append(
        run_feature_extractor(
            config.colmap,
            images_path=prepared_images_dir,
            database_path=paths.database_path,
            log_path=paths.logs_dir / "01_feature_extractor.log",
            dry_run=config.dry_run,
        )
    )

    commands.append(
        run_matcher(
            config.colmap,
            database_path=paths.database_path,
            log_path=paths.logs_dir / "02_matcher.log",
            dry_run=config.dry_run,
        )
    )

    mapping_database = paths.database_path
    if config.colmap.mode.lower() == "global" and config.colmap.use_view_graph_calibrator:
        if not config.dry_run:
            shutil.copy2(paths.database_path, paths.global_database_path)
        mapping_database = paths.global_database_path
        commands.append(
            run_view_graph_calibrator(
                config.colmap,
                database_path=mapping_database,
                log_path=paths.logs_dir / "03_view_graph_calibrator.log",
                dry_run=config.dry_run,
            )
        )

    if config.colmap.mode.lower() == "global":
        commands.append(
            run_global_mapper(
                config.colmap,
                images_path=prepared_images_dir,
                database_path=mapping_database,
                output_path=paths.sparse_dir,
                log_path=paths.logs_dir / "04_global_mapper.log",
                dry_run=config.dry_run,
            )
        )
    elif config.colmap.mode.lower() == "incremental":
        commands.append(
            run_mapper(
                config.colmap,
                images_path=prepared_images_dir,
                database_path=mapping_database,
                output_path=paths.sparse_dir,
                log_path=paths.logs_dir / "04_mapper.log",
                dry_run=config.dry_run,
            )
        )
    else:
        raise ValueError(
            f"Unsupported colmap.mode '{config.colmap.mode}'. "
            "Expected 'global' or 'incremental'."
        )

    sparse_model_path = paths.sparse_dir if config.dry_run else locate_sparse_model(paths.sparse_dir)
    commands.append(
        export_model_to_text(
            config.colmap,
            input_path=sparse_model_path,
            output_path=paths.sparse_text_dir,
            log_path=paths.logs_dir / "05_model_converter.log",
            dry_run=config.dry_run,
        )
    )

    if config.dry_run:
        summary_path = paths.artifacts_dir / "reconstruction_summary.json"
        write_json(
            summary_path,
            {
                "dry_run": True,
                "scene_name": config.scene_name,
                "mode": config.colmap.mode,
                "prepared_images_dir": str(prepared_images_dir),
            },
        )
        summary = {
            "num_cameras": 0,
            "num_images": 0,
            "num_points3d": 0,
            "mean_track_length": None,
            "mean_reprojection_error": None,
        }
        summary_object = ReconstructionSummary(**summary)
    else:
        summary_path, exported = export_sparse_artifacts(paths.sparse_text_dir, paths.artifacts_dir)
        summary = exported["summary"]
        summary_object = exported["summary_object"]

    result = ReconstructionResult(
        scene_name=config.scene_name,
        workspace_dir=paths.workspace_dir,
        images_dir=prepared_images_dir,
        sparse_model_path=sparse_model_path,
        sparse_text_path=paths.sparse_text_dir,
        summary_path=summary_path,
        run_log_path=paths.run_log_path,
        summary=summary_object,
        commands=commands,
    )

    run_log_payload = {
        "scene_name": config.scene_name,
        "config_path": str(job.config_path),
        "run_paths": paths.to_dict(),
        "config": config.to_dict(),
        "ingest": ingest_summary,
        "prepare": prepare_summary,
        "summary": summary,
        "commands": [command.to_dict() for command in commands],
    }
    write_json(paths.run_log_path, run_log_payload)
    return result


def run_pipeline(job: ReconstructionJob, *, skip_gsplat: bool = False) -> dict[str, Any]:
    reconstruction = run_reconstruction(job)
    paths = build_run_paths(job.config)

    gsplat_result = {
        "enabled": job.config.gsplat.enabled,
        "skipped": True,
        "reason": "skip_gsplat flag was provided" if skip_gsplat else "gsplat disabled in config",
    }
    if not skip_gsplat:
        gsplat_result = run_gsplat_training(
            job.config.gsplat,
            paths=paths,
            sparse_model_path=reconstruction.sparse_model_path,
            sparse_text_path=reconstruction.sparse_text_path,
            dry_run=job.config.dry_run,
        )

    pipeline_result = {
        "reconstruction": reconstruction.to_dict(),
        "gsplat": gsplat_result,
    }
    write_json(paths.artifacts_dir / "pipeline_result.json", pipeline_result)
    return pipeline_result
