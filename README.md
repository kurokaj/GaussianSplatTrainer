# GaussianSplatTrainer

Step 1 scaffold for a reconstruction and Gaussian Splatting workflow on Verda:

`images / video -> COLMAP sparse reconstruction -> exported cameras + points -> gsplat training -> artifacts`

The current repo is intentionally hybrid:
- Python owns orchestration, config loading, ingest/prepare, export, and visualization
- COLMAP CLI owns feature extraction, matching, and sparse mapping
- gsplat is called from Python as a configurable training step

## Repo Layout

Inside the repo:
- `configs/`: YAML configs for scenes and modes
- `src/pipeline/`: pipeline modules
- `scripts/run_pipeline.py`: main CLI entrypoint
- `notebooks/`: thin interactive notebooks for stages 01-04

Expected Verda workspace layout:
```text
/workspace
  /repo
  /envs
  /opt
  /data
  /runs
```

This repo is expected to live at `/workspace/repo`, with:
- datasets under `/workspace/data`
- outputs under `/workspace/runs`
- COLMAP under `/workspace/opt/colmap/bin/colmap`

## High-Level Usage On Verda

1. Start a Verda GPU VM with a persistent volume.
2. Clone this repo into `/workspace/repo`.
3. Create or activate your Python environment under `/workspace/envs`.
4. Install COLMAP under `/workspace/opt` or make `colmap` available on `PATH`.
5. Install Python dependencies.
6. Edit a config in `configs/`.
7. Run the pipeline from the repo root:

```bash
python3 scripts/run_pipeline.py --config configs/house_global.yaml
```

For notebook-driven work:
- `notebooks/01_ingest_and_prepare.ipynb`
- `notebooks/02_reconstruct.ipynb`
- `notebooks/03_train_gsplat.ipynb`
- `notebooks/04_inspect_outputs.ipynb`

Recommended Verda dev pattern:
- run JupyterLab on the instance
- keep it private
- connect through SSH tunneling from local VS Code

## Current Scope

Phase 1 is about getting one repeatable path working:
- image folder or video input
- frame extraction and image preparation
- incremental or global COLMAP sparse reconstruction
- export of sparse cameras and points
- optional gsplat training hook
- notebook-based inspection

## Future Steps

### Phase 2: Freeze the environment

- Write a Dockerfile for the exact stack.
- Push it to Verda’s private registry or GHCR.
- Keep a minimal startup script that mounts storage and starts Jupyter and other dev services.
- Save the VM config as a Verda template so you can recreate it quickly.

### Phase 3: Automate jobs

Split the pipeline into explicit stages:
- ingest video/images
- frame extraction / cleanup
- COLMAP sparse
- optional dense / fusion
- gsplat train
- export artifacts

Turn that into a Python CLI or Makefile first. Only after that, connect it to a database or object-store trigger.

### Phase 4: Productionize

- Provision with Verda CLI or Terraform/OpenTofu so the instance, volume, and startup script are reproducible.
- For ephemeral jobs: spin up the VM, run the pipeline, upload artifacts, and shut it down.
- Keep the environment and infrastructure declarative.

