# Arrhythmia Detector — Backend

[![Tests](https://github.com/Ivan-LB/arrhythmia-detector-backend/actions/workflows/tests.yml/badge.svg)](https://github.com/Ivan-LB/arrhythmia-detector-backend/actions/workflows/tests.yml)

ECG heartbeat classifier: a windowed-feature-engineering pipeline (FFT/PSD, wavelet, and statistical features around each beat) feeding a neural network trained on the MIT-BIH Arrhythmia Database, exposed over a FastAPI inference service. Classifies individual heartbeats into the 5 AAMI EC57 superclasses (N/S/V/F/Q) — it does not attempt rhythm diagnosis.

This is a from-scratch rebuild of an earlier degree-project version (tagged `v1.0`), redone with a leakage-free inter-patient evaluation protocol and a properly decoupled architecture. See [handoff.md](handoff.md) for the current project state and [docs/](docs/) for the full design record.

## Repo layout

| Path | What it is |
|---|---|
| `ecg_pipeline/` | Shared preprocessing, feature extraction, AAMI labeling, and DS1/DS2 split logic — the single source of truth used by both training and the API |
| `training/` | Dataset build, model training, and evaluation scripts |
| `api/` | FastAPI inference service (`api/main.py`) |
| `Data/Dataset/` | Vendored MIT-BIH raw records (WFDB format) |
| `Models/`, `UI/`, `ModelCreation/` | Original pre-rebuild PyQt desktop app and its model — kept as-is until the new web frontend reaches feature parity, not part of the rebuilt pipeline |
| `docs/` | Architecture docs: data pipeline, neural network, system design, plan, dated progress log |
| `tests/` | pytest suite (168 tests) |

The new web frontend (`arrhythmia-detector-web`) lives in a separate repo and talks to this API — see [docs/system-design.md](docs/system-design.md) for the split rationale and API contract.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Ivan-LB/arrhythmia-detector-backend.git
cd arrhythmia-detector-backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Run the tests

```bash
python -m pytest tests/ -q
```

### Train a model

The raw MIT-BIH records are already vendored under `Data/Dataset/` (see [Dataset](#dataset) below), so training runs directly against the committed data:

```bash
# Raw records -> leakage-free DS1/DS2 feature datasets (de Chazal et al. 2004 inter-patient split)
python -m training.build_dataset Data/Dataset Data

# Train on DS1, evaluate on DS2 (fully held-out patients)
python -m training.train Data/dataset_ds1.csv models
python -m training.evaluate models/<generated-model-dir> Data/dataset_ds2.csv
```

Both `Data/dataset_ds1.csv` / `Data/dataset_ds2.csv` and everything under `models/` are generated, deterministic (fixed seed) artifacts — intentionally not committed to git (see `.gitignore`). Regenerate them with the commands above instead of expecting them in a fresh clone.

Current DS2 (held-out) results: **63.22%** overall accuracy — a first correct, methodologically honest baseline (real inter-patient evaluation, no synthetic oversampling), not a tuned final model. Per-class Se/Sp/PPV/NPV and the full reasoning trail (including why SMOTE was tried and rejected for fabricating data) are in [docs/neural-network-architecture.md](docs/neural-network-architecture.md) and [docs/progress.md](docs/progress.md).

### Run the API

```bash
MODEL_DIR="$(ls -d models/beat-classifier-*)" uvicorn api.main:app --reload
```

`MODEL_DIR` is required — there's no implicit "latest model" default (see [docs/neural-network-architecture.md](docs/neural-network-architecture.md) §8).

Endpoints: `GET /health`, `POST /records` (multipart WFDB upload: `hea_file`/`dat_file`/`atr_file`), `GET /records/{id}/beats`, `GET /records/{id}/signal`. Full contract in [docs/system-design.md](docs/system-design.md) §4.

Relevant environment variables, all optional:

| Variable | Default | Purpose |
|---|---|---|
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed browser origins |
| `RATE_LIMIT_MAX_REQUESTS` | `10` | Max `POST /records` uploads per client per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window size, in seconds |
| `RATE_LIMIT_MAX_TRACKED_CLIENTS` | `1000` | LRU cap on tracked client entries |

## Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet `mitdb`), vendored directly under `Data/Dataset/` rather than fetched at build time. To re-fetch a clean copy from the canonical source instead (e.g. to verify provenance, or if a local copy is ever suspected corrupted):

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'Data/Dataset/Train')"
```

Citation: Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database." *IEEE Eng in Med and Biol* 20(3):45-50, 2001. Goldberger AL, et al. "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation* 101(23):e215–e220, 2000.

## Docs

- [handoff.md](handoff.md) — live state, read this first when picking the project back up
- [docs/plan.md](docs/plan.md) — the fixed roadmap and phase checklists
- [docs/progress.md](docs/progress.md) — dated log of what was actually done and why
- [docs/data-pipeline-architecture.md](docs/data-pipeline-architecture.md) — dataset, splits, feature engineering, AAMI label mapping
- [docs/neural-network-architecture.md](docs/neural-network-architecture.md) — model architecture, training, real evaluation results
- [docs/system-design.md](docs/system-design.md) — overall architecture, API contract, repo-split rationale
