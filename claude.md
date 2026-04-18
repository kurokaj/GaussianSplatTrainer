# Claude Notes

## Repo Purpose

This repo is a step-1 pipeline scaffold for:

`images / video -> COLMAP sparse -> exported reconstruction -> gsplat -> artifacts`

The design goal is to keep notebooks thin and put real logic in Python modules and scripts.

## Important Repo Conventions

- Keep pipeline logic in `src/pipeline/`, not in notebooks.
- Use config-driven paths and behavior.
- Treat COLMAP CLI integration as a narrow boundary in `src/pipeline/colmap_cli.py`.
- Prefer adding new pipeline stages as modules plus script/notebook entrypoints.
- Keep scene outputs self-contained under one run workspace.

## Expected Verda Layout

```text
/workspace
  /repo
  /envs
  /opt
  /data
  /runs
```

Assumptions:
- repo root is `/workspace/repo`
- datasets live under `/workspace/data`
- run outputs live under `/workspace/runs`
- COLMAP is usually under `/workspace/opt/colmap/bin/colmap`
- Python environments usually live under `/workspace/envs`

## Current Entry Points

- CLI: `python3 scripts/run_pipeline.py --config configs/house_global.yaml`
- Notebooks:
  - `notebooks/01_ingest_and_prepare.ipynb`
  - `notebooks/02_reconstruct.ipynb`
  - `notebooks/03_train_gsplat.ipynb`
  - `notebooks/04_inspect_outputs.ipynb`

## Editing Guidance

- Preserve the separation between:
  - ingest
  - prepare
  - reconstruct
  - export
  - visualization
  - gsplat training
- Keep config path defaults aligned with the Verda workspace layout.
- Prefer absolute or config-resolved paths over hardcoded local-machine assumptions.
- If adding COLMAP flags, change `src/pipeline/colmap_cli.py` first instead of scattering CLI details elsewhere.
- If adding visualization, keep it notebook-friendly and optional.

## Verda-Specific Notes

- The current workflow assumes an interactive Verda VM first, not a production job runner.
- Jupyter should be run privately and reached through SSH tunneling.
- Persistent storage matters: repo, envs, tools, datasets, and runs should survive instance restarts.
- Dockerization and reproducible startup belong to the next phase, not this step-1 scaffold.
