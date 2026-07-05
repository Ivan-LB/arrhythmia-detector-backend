# Progress Log

Living tracker for the rebuild. Update this as work happens — check items off, add dated entries below. For the fixed roadmap this tracks against, see [plan.md](plan.md).

## Current status

**Phase 0 merged into `v2.0.0`. Phase 1 merged into `v2.0.0`. Phase 2 done, PR open awaiting review** (not yet merged). `v1.0` tag marks the pre-rebuild ("end of degree project") state. Repo renamed to `arrhythmia-detector-backend`. Real trained model + DS2 evaluation exist (63.22% accuracy — see Phase 2 log entry below for what that number does and doesn't mean). Phase 3 (FastAPI backend) not started; waits for Phase 2 to merge first.

## Checklist

### Phase 0 — Research & architecture docs
- [x] Full repo audit
- [x] Dataset/methodology research
- [x] `data-pipeline-architecture.md`
- [x] `system-design.md`
- [x] `neural-network-architecture.md`
- [x] `plan.md`
- [x] `progress.md` (this file)

### Phase 1 — Shared pipeline package ✅ merged
- [x] `ecg_pipeline/preprocessing.py`
- [x] `ecg_pipeline/features.py`
- [x] `ecg_pipeline/labels.py`
- [x] `ecg_pipeline/splits.py`
- [x] Unit tests (63 tests, 100% coverage)

### Phase 2 — Dataset build + training ✅ done, PR open
- [x] `training/build_dataset.py`
- [x] `training/train.py`
- [x] `training/evaluate.py`
- [x] `training/class_encoding.py` (added during review — shared, validated class<->index mapping)
- [x] First versioned model artifact (`beat-classifier-20260705-dc9f983`, DS2 accuracy 63.22%)

### Phase 3 — FastAPI backend
- [ ] `api/main.py` / `inference.py` / `schemas.py`
- [ ] Endpoints implemented
- [ ] Config via env vars
- [ ] API tests

### Phase 4 — React/Next.js frontend
- [ ] Project scaffold
- [ ] Upload flow
- [ ] ECG trace + beat overlay
- [ ] Summary view

### Phase 5 — Repo hygiene
- [ ] `pyproject.toml`
- [ ] `README.md`
- [ ] `.gitignore` fix
- [ ] Dataset/model download step documented
- [ ] CI

### Phase 6 — SwiftUI app (future)
- [ ] Not started (deferred)

## Log

### 2026-07-05 — Audit
Ran a full repo audit across code quality, security, ML correctness, and repo hygiene. 34 confirmed findings, all independently re-verified against the actual source before being reported. Headline result: the current model's reported accuracy/confusion-matrix figures are built on a leaky random beat-level train/test split (patient identity leaks across the split), so they don't reflect true generalization. Also found: a scaler fit-twice bug in `ANNModel_Prueba.py`, hardcoded machine-specific paths in four different files (one of which doesn't exist on this machine, breaking the app on import), non-contiguous/disagreeing label encodings across the two training scripts, and duplicated feature-extraction code between the training pipeline and the UI.

### 2026-07-05 — Dataset & methodology research
Decided to redo the project rather than patch it. Confirmed via direct source research (not memory):
- AAMI EC57 5-superclass mapping (N/S/V/F/Q), cross-checked against the canonical WFDB `ecgcodes.h` annotation reference after an initial source gave a wrong symbol ("U" instead of "Q" for unclassifiable).
- The DS1/DS2 inter-patient split (de Chazal et al., 2004) — confirmed identically across 3 independent sources, including the widely-used `mondejar/ecg-classification` reference implementation.
- MIT-BIH stays the primary dataset; MIT-BIH-SVDB flagged as an optional future augmentation for the underrepresented S class.
- Explicitly ruled out the popular Kaggle "ECG Heartbeat Categorization" dataset — confirmed its shipped train/test split is a random intra-patient split, i.e. the same leakage bug we're trying to fix.
- Standard evaluation practice confirmed: report per-class Sensitivity/Positive Predictivity/Specificity, not just accuracy.

Note on process: the first attempt at this research used a large automated multi-agent workflow, which stalled twice (once from a schema bug causing a retry loop, once from exhausting session credits) before being abandoned in favor of doing the research directly with targeted, manually cross-checked searches.

### 2026-07-05 — Architecture docs
Wrote `data-pipeline-architecture.md`, `system-design.md`, and `neural-network-architecture.md`, grounded in a direct re-read of the current source (`ModelPreparation.py`, `ANNModel.py`, `ecg_feature_extractor.py`), not just the earlier audit summary. Found one additional issue while doing this: `apply_window()` is called with its arguments in a different order than its own signature in `ModelPreparation.py` — currently harmless only because the specific arithmetic involved happens to be commutative, but fragile and worth fixing in the rewrite regardless.

Key scope decisions made and flagged for confirmation: rhythm classification dropped from this rebuild (different task, needs multi-beat sequence context); feature set expanded from 6 to 9 (including skew/kurtosis/variance, already computed today but discarded before training); beat-window centers move from self-detected R-peaks to the annotation's own sample index (fixes a real blind spot — the current peak detector can't detect beats faster than ~86 bpm, which would miss tachycardia).

### 2026-07-05 — Windowing-before-FFT gap (found while reviewing with the original author)
While walking through `ModelPreparation.py` history, surfaced that `calculate_fft_and_wavelet()` has always run the FFT on a raw truncated segment with no analysis window applied — meaning `SpectralEnergy`/`TotalPSD` have been carrying avoidable spectral leakage since the feature was first written. `ModelCreation/sineWave.py` (DSP coursework utilities, different author, never wired into the real pipeline) already had Hann/Hamming/Blackman window generators sitting unused. `data-pipeline-architecture.md` §5 updated to apply a Hann window before the FFT/PSD step — chosen over Blackman because these features measure aggregate spectral shape rather than resolving closely-spaced frequencies, so leakage suppression matters more than main-lobe narrowness, and Hann is the standard default for this in ECG spectral analysis. Wavelet-based features are unaffected — CWT doesn't carry FFT's periodicity assumption.

### 2026-07-05 — v1.0 tagged, docs pushed, repo split decided
Tagged the pre-rebuild commit (`51e9924`, "AD_35: End of Semester") as `v1.0` and pushed it — this is the permanent snapshot of the original degree project. Committed and pushed the five `docs/` files as the first commit of v2.0 (`190f384`). Repo is confirmed already public on GitHub.

Decided to split into separate repos rather than one monorepo: this repo becomes backend-only (`ecg_pipeline` + `training` + `api`), the web frontend and the future SwiftUI app each get their own repo and talk to this one purely over HTTP. `system-design.md` §3 updated accordingly. Explicitly decided *against* generalizing `ecg_pipeline` into a reusable library for other projects — scoped to what this app needs, revisit only if a real second consumer shows up. Repo rename pending a name decision.

`UI/` stays in this repo and stays working until the web app has visible functional parity — not removed as part of the restructure.

Repo renamed on GitHub: `Ivan-LB/Arrhythmia-Detector` → `Ivan-LB/arrhythmia-detector-backend` (via `gh repo rename`; local `origin` remote updated to match). GitHub auto-redirects the old URL.

Also established the git workflow going forward: `v2.0.0` is the long-lived dev/integration branch (branched off `main`); each phase gets its own branch off `v2.0.0` (e.g. `phase-1-ecg-pipeline`), opened as a PR for review, merged into `v2.0.0` once approved. `main` only gets the final merge once all phases are done.

### 2026-07-05 — Phase 1: `ecg_pipeline/` package built, reviewed, and fixed
Built `ecg_pipeline/` (`splits.py`, `labels.py`, `preprocessing.py`, `features.py`) test-first: wrote all 4 test files, confirmed they failed with `ImportError` (RED) since no implementation existed, then implemented each module to make them pass (GREEN). Added a minimal `pyproject.toml` scoped to Phase 1's actual dependencies (numpy, scipy, PyWavelets, pytest) — the rest (tensorflow, fastapi, wfdb, etc.) gets added incrementally as later phases need them, not all upfront. Created a `.venv` and a handful of necessary `.gitignore` entries (venv/pycache/egg-info) pulled forward out of necessity; the full hygiene overhaul stays Phase 5.

Ran an independent `python-reviewer` pass before considering this done, per the project's mandatory code-review rule. It found:

- **CRITICAL** — `_wavelet_features`'s second probability-normalization step divided 0/0 → `NaN` whenever an entire wavelet scale row was all-zero (reachable via a fully degenerate/flatlined window, which `preprocessing.min_max_normalize` explicitly documents as a real output it can produce). Fixed by guarding that division the same way the first one already was.
- **CRITICAL** — `scipy.stats.skew`/`kurtosis` return `NaN` for any perfectly flat window, via the same flatlined-segment trigger, completely unguarded. Rather than fabricate a placeholder (e.g. silently returning `0.0`) or let `NaN` reach a training matrix, `extract_features` now raises a clear `ValueError` for any window with no real signal variation — the caller (Phase 2's dataset builder) must explicitly exclude that beat, not silently train on an invented number for it.
- **HIGH** — `TotalPSD` used a hand-rolled power-spectral-density formula off by a factor of N² from the standard periodogram-density formula, and its manual one-sided-spectrum doubling was only valid for even-length windows. Fixed by delegating to `scipy.signal.periodogram(..., scaling="density")` directly, which gets both the scaling and the even/odd-length handling right for free.
- **HIGH** — `min_max_normalize`'s flat-signal guard used exact floating-point equality (`value_range == 0`), so a window that's flat except for ~1e-14 floating-point residue (plausible upstream-filter artifact on a genuinely flatlined segment) bypassed the guard and produced a spurious full-scale spike at a single sample. Fixed with `np.isclose` instead of exact equality.
- **MEDIUM** (fixed) — notch filter used causal `lfilter` instead of zero-phase `filtfilt`, which would shift signal content relative to the annotation-centered window this rebuild specifically introduced; added an `fs` precondition check; fixed a `kurtosis`/`skew` import-name shadowing readability hazard; added validation on `window_around_sample`'s `width_seconds`/`fs`.
- Noted but not fixed (documented rationale, not silently dropped): a `np.hanning(2)` degenerate-window edge case and an `lfilter`-derived `Any` return-type gap for `mypy --strict` — both require pathological inputs this pipeline's actual usage never produces.

All fixes verified: 63 tests passing, 100% statement coverage, zero warnings even with `RuntimeWarning` promoted to a hard error. `data-pipeline-architecture.md` updated to reflect the `filtfilt` change, the corrected `TotalPSD` description, and the new degenerate-window rejection policy.

**Next up:** Phase 2 — dataset build + training.

### 2026-07-05 — Phase 2: dataset build, training, evaluation — and a real data-fabrication bug caught before it shipped

**Data completeness gap found and fixed first.** Before writing any Phase 2 code, verified the DS1/DS2 record files were actually complete (per the "verify in the source" rule) rather than assuming the committed `Data/Dataset/` folders had everything. They didn't: 19 of the 20 required 100-series records had no `.atr` annotation file at all (only `.dat`/`.hea`), and record 113 was missing even its `.hea` header — unreadable as-is. Only the 200-series records (already `.atr`-complete) were usable. Re-fetched clean copies of all 20 100-series records directly from PhysioNet's `mitdb` via `wfdb.dl_database()`, removed a stray corrupted `108.at_` file, and verified all 44 DS1/DS2 records now load correctly with MLII present. This corrects something said earlier in this project: MIT-BIH-only does *not* mean "zero new downloads" — it needed this one.

**Built `ecg_pipeline`-consuming pipeline test-first**: `training/build_dataset.py` (raw records → labeled feature CSVs), `training/train.py` (DS1 training with a group-aware internal validation split), `training/evaluate.py` (DS2 confusion matrix + per-class Se/Sp/PPV/NPV). Ran the real dataset build against all 44 records: 50,995 DS1 rows / 49,687 DS2 rows, zero NaN, class distribution matching the published de Chazal 2004 inter-patient counts closely (N ~90%, S/V/F/Q all in the same ballpark as the literature) — strong confirmation the whole Phase 1 pipeline behaves correctly on real data.

**Caught SMOTE fabricating ~38,000 synthetic rows from 2 real ones.** First real training run used SMOTE (as originally planned) to balance classes. The Q class had only **2 real examples** in the training partition (of 41,848 total rows) — SMOTE oversampled that to match the ~37,782-row N class, an **18,891x amplification**, meaning every "Q" training row beyond the original 2 was a point mathematically interpolated on a single line segment between them. That's fabricating data to reach a number, which directly violates the standing rule against inventing data — caught by actually running the training and inspecting the real numbers, not by inspection alone.

**First fix (sklearn "balanced" class_weight) had the same problem in a different shape.** Replaced SMOTE with `class_weight`, which doesn't invent rows — but sklearn's `"balanced"` heuristic computed a weight of **~4185** for Q (vs ~20 for F, the next-rarest class), and empirically that blew up training just as badly: `val_loss` around 16 versus the ~1.6 random-guessing baseline for 5 classes, because a single Q example in a batch dominated the gradient.

**Final fix: cap the class weight, derived from the data, not memorized.** Capped the largest weight at the second-highest naturally-occurring weight in that run's own distribution (not a hardcoded constant) — self-adjusting if the class distribution ever shifts, and only applied when 3+ classes are present (with exactly 2, "second-highest" is the smaller one, which would collapse both weights to be equal — a real bug caught by the test suite before it shipped). Retrained: `val_loss` back to a sane 1.1-1.8 range, converges in ~12-18 epochs.

**Independent code review caught one more real bug and two real design gaps**, all fixed before merging:
- **CRITICAL** — `prepare_training_data` mapped `AAMIClass` strings to indices with a bare, unguarded `.map()`, unlike `encode_labels` right next to it which already validated. An unrecognized class value would silently become `NaN` → get cast to integer `0` by `to_categorical` → silently train as class "N" with no error at all. Consolidated the mapping into a new shared `training/class_encoding.py` (both `train.py` and `evaluate.py` had independently reimplemented `CLASS_TO_INDEX`) so the validation can't drift out of sync between files again.
- **HIGH** — the group-aware validation split is seed-dependent, and on the real run only 2 of DS1's 8 total Q rows landed in training purely because of which patient records the split drew — a different seed could zero Q out of training entirely with nothing surfacing that fact. Added `warn_on_missing_classes()` plus per-class train/validation counts in `training_config.json`.
- **HIGH** (the cap-derivation fix above) — flagged as a defensible-but-brittle hardcoded constant; fixed by deriving it from the run's own data.

**Final real result, after all fixes**: DS2 accuracy **63.22%**. Honestly far below the original (leaky) pipeline's ~98-99%, and that's the entire point — this reflects true inter-patient generalization, not memorized patient identity. Per-class breakdown: N Se=64.8%/PPV=95.1%, S Se=24.5%/PPV=13.1%, V Se=64.6%/PPV=21.8%, F Se=61.3%/PPV=3.6%, Q Se=0%/PPV=0% (expected — only 2 training examples exist for it). Minority-class precision is weak (lots of false alarms) because of the aggressive class weighting needed to get any recall at all on such rare classes — a real, expected precision/recall tradeoff, not a bug. This is a first correct baseline, not a tuned final model; comparing against the published benchmarks researched in Phase 0 suggests real room for improvement via richer features or a different architecture, which is future work, not something silently attempted here without saying so.

108 tests, 94% combined coverage. PR open (`phase-2-dataset-training` → `v2.0.0`), awaiting review.

**Next up:** Phase 3 — FastAPI backend (once Phase 2 is reviewed and merged).
