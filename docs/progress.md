# Progress Log

Living tracker for the rebuild. Update this as work happens — check items off, add dated entries below. For the fixed roadmap this tracks against, see [plan.md](plan.md).

## Current status

**Phases 0-3 merged into `v2.0.0`**, plus two follow-up fixes on the API surface (CORS middleware, upload rate limiting), both merged. `v1.0` tag marks the pre-rebuild ("end of degree project") state. Repo renamed to `arrhythmia-detector-backend`. Real trained model + DS2 evaluation exist (63.22% accuracy — see Phase 2 log entry for what that number does and doesn't mean). A working FastAPI service sits in front of that model, verified against a real running server, not just in-process tests. **Phase 4 (React/Next.js frontend) is done**, in its own new repo (`arrhythmia-detector-web`) per the polyrepo decision — upload flow, ECG trace + per-beat classification overlay, and a class-distribution/confidence summary view all built and browser-verified against this real API. **Phase 5 (repo hygiene, this repo) in progress.**

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

### Phase 3 — FastAPI backend ✅ merged
- [x] `api/main.py` / `inference.py` / `records.py` / `schemas.py`
- [x] Endpoints implemented (`/health`, `POST /records`, `GET /records/{id}/beats`, `GET /records/{id}/signal`)
- [x] Config via env vars (`MODEL_DIR`, required)
- [x] API tests (32 tests) + verified against a real running uvicorn server
- [x] `ecg_pipeline.preprocessing.detect_r_peaks` (new, for live-uploaded recordings with no ground truth)
- [x] Two parallel code reviews (correctness + security) — all findings fixed
- [x] Follow-up: CORS middleware (PR #4) and upload rate limiting (PR #5), both merged

### Phase 4 — React/Next.js frontend ✅ done — own repo (`arrhythmia-detector-web`)
- [x] Project scaffold (Next.js/TypeScript, designed via `/impeccable`)
- [x] Upload flow
- [x] ECG trace rendering + per-beat classification overlay
- [x] Summary view (class distribution, confidence) + trace-view UX redesign (class breakdown selector, 60s context window)

### Phase 5 — Repo hygiene
- [x] `pyproject.toml` pinned dependencies
- [x] `README.md`
- [x] `.gitignore` fix
- [x] Dataset/model download step documented
- [x] CI

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

### 2026-07-05 — Phase 3: FastAPI backend, plus a real memory-exhaustion DoS caught before it shipped

**New capability needed first: `ecg_pipeline.preprocessing.detect_r_peaks`.** Training/evaluation use the MIT-BIH annotation's own sample index as ground truth for where each beat is — a live-uploaded recording has no such annotations, so something has to find the beats. Added a corrected R-peak detector (`min_distance_seconds=0.2`, ~300bpm max) fixing the original pipeline's `distance=0.7*fs` bug (~86bpm max, structurally blind to tachycardia). Validated against a real record: 91.5% of the true annotated beat count detected — a reasonable approximation, not expected to match ground truth exactly.

**Built `api/records.py`, `api/inference.py`, `api/main.py`, `api/schemas.py` test-first**, wiring `/health`, `POST /records` (multipart `.hea`/`.dat`, optional `.atr` for ground truth), `GET /records/{id}/beats`, `GET /records/{id}/signal`. `create_app()` is a factory (not a fixed module instance) so tests can point it at a small/fast model; the real `MODEL_DIR` env var is only required when actually running the service (via a PEP 562 module `__getattr__`), not merely importing the module.

**Manually verified against a real running server** (`uvicorn`, not just FastAPI's in-process `TestClient`): uploaded a real MIT-BIH record, got 2255 classified beats back, confirmed annotation-based vs. self-detected `beat_source`, 400s/404s behave correctly, `/docs` (OpenAPI) loads, zero errors in the server log.

**Two parallel code reviews** (python-reviewer for correctness/design, security-reviewer for the file-upload surface specifically) found real, fixable problems:

- **HIGH (security) — memory-exhaustion DoS, exploitable today with a single unauthenticated request.** Uploads were read fully into memory (`await file.read()`) before any size check ran; Starlette's own multipart parser has no total-body-size ceiling of its own (confirmed by reading its source directly), so a multi-GB request body would be fully materialized in process memory before the 200MB app-level check was ever consulted. Fixed: read in bounded 1MB chunks with an early abort as soon as the limit is crossed.
- **HIGH (correctness) — a record with no MLII lead crashed every read, not the upload.** Upload only validated that the file parsed as *some* wfdb record, not that it had an MLII channel — accepted with 200, then an unhandled `ValueError` on every subsequent `GET /beats`/`GET /signal`. None of the test fixtures (all using record 230, which has MLII) caught this. Fixed: validate MLII presence at upload time, one clear 400 instead.
- **HIGH (correctness) — zero caching, and concurrent requests effectively serialized.** Every `GET /beats` recomputed the entire filter → detect/read-annotations → extract-features → predict pipeline from scratch — measured at ~2.1s/call on a real record, repeated identically on every call. 8 concurrent calls took ~24s (roughly 8x, not parallelized) due to GIL contention on the CPU-bound per-beat feature-extraction loop. Fixed: cache the computed response per record after first computation. Confirmed against the real server: first call 2.3s, every call after ~10ms.
- **HIGH (correctness) — no eviction.** Uploaded records and their on-disk directories lived for the entire process lifetime with nothing ever removing them. Added a configurable FIFO cap (default 50) evicting the oldest record (dict entry + directory together) once exceeded — proportionate for a portfolio/demo-scale service; still no idle-TTL, which would need revisiting for a real deployment.
- **MEDIUM (security) — a NUL byte in a filename crashed with an unhandled `ValueError`** instead of the clean 400 every other invalid-upload path already produced (confirmed reachable via a hand-crafted multipart request, not through httpx/browsers, which won't produce a raw NUL). Fixed with an explicit control-character check.
- **MEDIUM (security) — blocking disk I/O and wfdb parsing inside an `async def` handler** blocks the event loop (including `/health`) for the duration of every upload. Offloaded to a thread pool.
- **MEDIUM (mypy type-safety)** — `InferenceBundle.scaler` was typed as bare `object` (now `StandardScaler`), `beat_source` was inferred as plain `str` (now `Literal["annotations", "detected"]`, matching the response schema), `UploadFile.filename` narrowing made explicit. mypy is now clean across `api/` and `ecg_pipeline/`.
- **Verified NOT exploitable** (no fix needed): path traversal via filename or wfdb-header-embedded paths (wfdb's own field regexes structurally can't contain `/`), and injection/RCE via malicious `.hea`/`.dat`/`.atr` content (wfdb parses through fixed regexes and `int()`/`float()`, no `eval`/dynamic code paths anywhere in its parsing).

151 tests, 94% combined coverage, mypy clean. PR open (`phase-3-fastapi-backend` → `v2.0.0`), awaiting review.

**Next up:** Phase 4 — React/Next.js frontend (its own new repo, per the polyrepo decision), once Phase 3 is reviewed and merged.

### 2026-07-05/06 — CORS + upload rate limiting (two follow-up fixes on the merged API)

Phase 3 merged. Two gaps surfaced next, each fixed test-first on its own branch off `v2.0.0`, independently reviewed, then merged via PR.

**CORS (PR #4).** The API had no `CORSMiddleware` at all — confirmed via a real browser (`TypeError: Failed to fetch`, while `curl` against the same endpoint worked fine, so this was a browser-enforced CORS rejection, not a server bug). Added `CORSMiddleware` with a `CORS_ALLOWED_ORIGINS` env var (comma-separated, defaults to `http://localhost:3000`).

**Upload rate limiting (PR #5).** Flagged during the CORS review as a gap: `POST /records` had no rate limiting, so it was open to unbounded upload spam. Added a hand-rolled sliding-window limiter (`_UploadRateLimiter`) rather than adopting `slowapi` — research turned up that `slowapi` has carried an "alpha quality" disclaimer in its own docs without reaching a 1.0 release, not something to depend on for a real control. Two independent review passes converged on the same two real issues, both fixed before merge:
- Malformed numeric env vars (`RATE_LIMIT_MAX_REQUESTS=abc`) raised an unhandled `ValueError` instead of a clean startup error — fixed via `_parse_numeric_env()`.
- The original FIFO eviction policy was gameable: an attacker could churn through >1000 distinct client keys to evict a legitimate client's tracked state and reset their own limit. Fixed by switching to LRU eviction (`OrderedDict.move_to_end()` on every access) so eviction always removes the least-recently-active client, not just the oldest-inserted one.

168 tests passing after both merges.

### 2026-07-05 — Phase 4: React/Next.js frontend (own repo)

Built in a new, separate repo (`arrhythmia-detector-web`) per the polyrepo decision in `system-design.md`. Full detail — design-system work (`PRODUCT.md`/`DESIGN.md` via `/impeccable`), TDD history, and real bugs found via browser verification (WCAG contrast failures, a sub-pixel click-target overlap only fixable by reverting to direct per-button handlers, a JS floating-point boundary bug in window-slicing, a two-way filter/selection state-consistency gap) — lives in that repo's own docs, not duplicated here. Summary: upload flow → full-record ECG trace with per-beat AAMI classification → a class breakdown bar (count, %, avg confidence per class, filterable) → a 60-second real-data "context window" around a selected beat (built instead of a fabricated high-resolution zoom, since the `/signal` endpoint's ~2000-point downsampling genuinely doesn't support one) → a beat detail panel scoped to the active filter. Verified throughout against this repo's real running API and real MIT-BIH record 230, not mocks.

### 2026-07-05/06 — Phase 5: repo hygiene

Branched `phase-5-repo-hygiene` off `v2.0.0` (had to catch up a stale local `v2.0.0` to `origin/v2.0.0` first — PR #5 had merged upstream but hadn't been pulled locally yet, a good reminder to verify branch state against the source rather than assume). Worked through the fixed checklist in `plan.md`:

- **`pyproject.toml`** — replaced every loose `>=` bound with an exact pin matching the actual working `.venv` (`numpy==2.5.1`, `tensorflow==2.21.0`, `fastapi==0.139.0`, etc.); `pip check` confirms no broken requirements.
- **`.gitignore`** — replaced the ad hoc, phase-by-phase entries (including three now-stale Python 3.9-specific literal `.pyc` paths) with proper globs: caches (`.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`), editors, OS files, build artifacts — on top of the existing derived-CSV and `models/` rules.
- **CI** — added `.github/workflows/tests.yml`, running the full pytest suite on push/PR against `main` and `v2.0.0`.
- **`README.md`** — replaced the 2-line placeholder with real setup/usage/dataset docs. Cross-checked every command against the actual scripts before writing it down rather than trusting recall: caught and fixed a wrong argument order for `training/evaluate.py` (`model_dir` first, not the CSV) this way.
- **Deleted `ModelCreation/sineWave.py`** — confirmed unreferenced anywhere else first. Also found and removed a stale tracked `.pyc` for the same file (`ModelCreation/__pycache__/sineWave.cpython-38.pyc`) that had been accidentally committed before any pycache ignore rule existed.
- **Dataset/model artifact download step** — documented in the README (`wfdb.dl_database('mitdb', ...)` to re-fetch, `training/build_dataset.py` + `training/train.py` to regenerate the derived CSVs/model). Deliberately did **not** untrack the already-committed raw `Data/Dataset/` records or the legacy `Models/*.h5`/`.pk1` binaries: the test suite genuinely reads real files from `Data/Dataset/Train/` (so untracking would break a fresh CI checkout), and the legacy binaries are still load-bearing for the original PyQt app under `UI/`/`ModelCreation/`, which is explicitly being kept as-is until the new web frontend reaches parity. Flagged for a separate, explicit decision rather than silently deleted.

168 tests still passing throughout (no application code touched, only packaging/docs/CI).
