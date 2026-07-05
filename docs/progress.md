# Progress Log

Living tracker for the rebuild. Update this as work happens — check items off, add dated entries below. For the fixed roadmap this tracks against, see [plan.md](plan.md).

## Current status

**Phase 0 (Research & architecture docs) — complete and pushed.** `v1.0` tag marks the pre-rebuild ("end of degree project") state. Repo renamed to `arrhythmia-detector-backend`. Phase 1 (`ecg_pipeline/` package) not yet started.

## Checklist

### Phase 0 — Research & architecture docs
- [x] Full repo audit
- [x] Dataset/methodology research
- [x] `data-pipeline-architecture.md`
- [x] `system-design.md`
- [x] `neural-network-architecture.md`
- [x] `plan.md`
- [x] `progress.md` (this file)

### Phase 1 — Shared pipeline package
- [ ] `ecg_pipeline/preprocessing.py`
- [ ] `ecg_pipeline/features.py`
- [ ] `ecg_pipeline/labels.py`
- [ ] `ecg_pipeline/splits.py`
- [ ] Unit tests

### Phase 2 — Dataset build + training
- [ ] `training/build_dataset.py`
- [ ] `training/train.py`
- [ ] `training/evaluate.py`
- [ ] First versioned model artifact

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
