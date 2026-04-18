#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from pipeline.reconstruct import load_reconstruction_job, run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the step-1 COLMAP -> gsplat pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--skip-gsplat",
        action="store_true",
        help="Run through COLMAP/export only and skip the gsplat stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and log commands without executing COLMAP or gsplat.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job = load_reconstruction_job(args.config)
    if args.dry_run:
        job.config.dry_run = True

    result = run_pipeline(job, skip_gsplat=args.skip_gsplat)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
