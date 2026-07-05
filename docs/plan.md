# Project Plan — Arrhythmia Detector Rebuild

This is the roadmap for the rebuild. For the "what" and "why" of each area, see the architecture docs it links to. For "what's actually done so far," see [progress.md](progress.md) — that file is the living tracker; this one is the fixed plan.

## Goal

Redo the ECG arrhythmia detector with a literature-defensible ML methodology and a properly decoupled application architecture, while keeping the original windowed-feature-engineering approach recognizable. Dual purpose: portfolio piece + academic-adjacent deliverable, so both polish and methodological rigor matter.

## Scope decisions already made

| Decision | Choice | Doc |
|---|---|---|
| Dataset | MIT-BIH Arrhythmia Database (already vendored), evaluated correctly | [data-pipeline-architecture.md](data-pipeline-architecture.md) |
| Train/test split | DS1/DS2 inter-patient split (de Chazal 2004) | [data-pipeline-architecture.md](data-pipeline-architecture.md) §3 |
| Label scheme | AAMI EC57 5-superclass (N/S/V/F/Q) | [data-pipeline-architecture.md](data-pipeline-architecture.md) §6 |
| UI stack | FastAPI + React/Next.js (Phase 1), SwiftUI macOS app (Phase 2) | [system-design.md](system-design.md) |
| Rhythm classification | Deferred — different task, needs sequence context | [data-pipeline-architecture.md](data-pipeline-architecture.md) §11 |
| MIT-BIH-SVDB augmentation | Deferred — revisit only if S-class recall is poor | [data-pipeline-architecture.md](data-pipeline-architecture.md) §2 |

## Phases

### Phase 0 — Research & architecture docs
- [x] Full repo audit (code quality, security, ML correctness, repo hygiene)
- [x] Dataset/methodology research (AAMI mapping, DS1/DS2 split, dataset landscape, metrics) — verified against primary sources
- [x] `data-pipeline-architecture.md`
- [x] `system-design.md`
- [x] `neural-network-architecture.md`

### Phase 1 — Shared pipeline package (`ecg_pipeline/`) ✅ done — see PR `phase-1-ecg-pipeline` → `v2.0.0`
- [x] `ecg_pipeline/preprocessing.py` — channel selection, notch filter (zero-phase), annotation-driven windowing
- [x] `ecg_pipeline/features.py` — 9-feature extraction (FFT/PSD, wavelet, statistical)
- [x] `ecg_pipeline/labels.py` — AAMI EC57 symbol→class mapping
- [x] `ecg_pipeline/splits.py` — DS1/DS2 record lists as a single source of truth
- [x] Unit tests for all of the above — 63 tests, 100% statement coverage
- [x] Independent code review (python-reviewer) — 2 CRITICAL + 2 HIGH findings, all fixed before merge

### Phase 2 — Dataset build + training ✅ done — see PR `phase-2-dataset-training` → `v2.0.0`
- [x] `training/build_dataset.py` — raw records → `dataset_ds1.csv` / `dataset_ds2.csv` (50,995 / 49,687 real rows)
- [x] `training/train.py` — scaler (fit on train-only) + capped class weights (not SMOTE — see progress.md) + model training, seeded
- [x] `training/evaluate.py` — DS2 confusion matrix + per-class Se/Sp/PPV/NPV table
- [x] First versioned model artifact under `models/` — real DS2 accuracy 63.22%
- [x] Independent code review (python-reviewer) — 1 CRITICAL + 2 HIGH findings, all fixed before merge
- [x] Fixed a real data-completeness gap: 19 of 20 required 100-series MIT-BIH records had no `.atr` annotations committed at all, and one had no `.hea` header — re-fetched clean copies from PhysioNet

### Phase 3 — FastAPI backend ✅ done — see PR `phase-3-fastapi-backend` → `v2.0.0`
- [x] `api/main.py`, `api/inference.py`, `api/records.py`, `api/schemas.py`
- [x] `/records`, `/records/{id}/beats`, `/records/{id}/signal`, `/health`
- [x] Config via env vars (`MODEL_DIR`, required, no implicit "latest" default)
- [x] API tests — 32 tests, verified against the real running server (uvicorn), not just in-process `TestClient`
- [x] `ecg_pipeline.preprocessing.detect_r_peaks` — new R-peak detector for live-uploaded recordings with no ground-truth annotations
- [x] Two parallel code reviews (python-reviewer + security-reviewer) — 1 HIGH security finding (memory-exhaustion DoS), 2 HIGH correctness findings (crash on non-MLII records, no caching/eviction), several MEDIUM fixes, all resolved before merge

### Phase 4 — React/Next.js frontend
- [ ] Project scaffold (TypeScript)
- [ ] File upload → record metadata
- [ ] ECG trace rendering + per-beat classification overlay
- [ ] Summary view (class distribution, confidence)

### Phase 5 — Repo hygiene
- [ ] `pyproject.toml` with pinned dependencies
- [ ] Real `README.md` (setup, usage, dataset download step, architecture doc links)
- [ ] `.gitignore` with proper globs
- [ ] Dataset/model artifacts documented as a download/regeneration step instead of committed to git
- [ ] CI (test run on push, given this is also a portfolio piece)
- [ ] Delete `ModelCreation/sineWave.py` — unused DSP coursework utility (different author); its one relevant idea (Hann windowing before FFT) is absorbed directly into `ecg_pipeline/features.py` via a plain `np.hanning()` call, so the file itself adds nothing

### Phase 6 — SwiftUI macOS app (future)
- [ ] Deferred until Phases 1–5 are solid. Consumes the same API contract as the web app.

## Out of scope (this rebuild)

- Rhythm classification
- MIT-BIH-SVDB augmentation (unless S-class recall demands it)
- Any raw-signal deep learning architecture (CNN/RNN on the raw waveform) — the windowed-feature-engineering approach is being kept, not replaced

## References

- Repo audit (2026-07-05, conversation record) — original bug list that motivated this rebuild: scaler double-fit bug, hardcoded machine-specific paths, non-contiguous label encoding, data leakage via random beat-level split, duplicated feature-extraction code between training and UI. Not yet written up as a standalone doc — worth doing if this project needs a self-contained record independent of chat history.
- de Chazal, O'Dwyer & Reilly, "Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features," IEEE Trans. Biomed. Eng., 2004
- Moody GB, Mark RG, "The impact of the MIT-BIH Arrhythmia Database," IEEE Eng in Med and Biol 20(3):45-50, 2001
