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

### Phase 4 — React/Next.js frontend ✅ done — own repo (`arrhythmia-detector-web`)
- [x] Project scaffold (TypeScript)
- [x] File upload → record metadata
- [x] ECG trace rendering + per-beat classification overlay
- [x] Summary view (class distribution, confidence)

### Phase 5 — Repo hygiene ✅ done — see PR `phase-5-repo-hygiene` → `v2.0.0`
- [x] `pyproject.toml` with pinned dependencies
- [x] Real `README.md` (setup, usage, dataset download step, architecture doc links)
- [x] `.gitignore` with proper globs
- [x] Dataset/model artifacts documented as a download/regeneration step instead of committed to git
- [x] CI (test run on push, given this is also a portfolio piece)
- [x] Delete `ModelCreation/sineWave.py` — unused DSP coursework utility (different author); its one relevant idea (Hann windowing before FFT) is absorbed directly into `ecg_pipeline/features.py` via a plain `np.hanning()` call, so the file itself adds nothing

### Phase 6 — SwiftUI macOS app (future)
- [ ] Deferred until Phases 1–5 are solid. Consumes the same API contract as the web app.

## Out of scope (the v2.0 rebuild)

- Rhythm classification
- MIT-BIH-SVDB augmentation (unless S-class recall demands it)
- Any raw-signal deep learning architecture (CNN/RNN on the raw waveform) — the windowed-feature-engineering approach is being kept, not replaced.
  **Revisited in v2.1** (see below): this boundary was drawn around the *beat classifier*, and still holds for it. v2.1's *detector* is a different task on a different label space (where is the beat, not what kind of beat is it), so a raw-signal model there doesn't contradict the classifier decision. Recording the distinction explicitly rather than quietly reversing a documented call.

---

# v2.1 — Beat detection

Tagged `v2.0` marks the end of the rebuild. v2.1 is a focused release with one theme: **proposing and evaluating a real beat-detection model**, and making the UI actually show beats at beat resolution.

## Why this, why now

Two concrete gaps left by v2.0, both verified in the code rather than assumed:

1. **There is no detection model.** `ecg_pipeline.preprocessing.detect_r_peaks` is a `scipy.signal.find_peaks` call with an amplitude threshold (mean + 2σ) and a minimum-distance constraint. It was written in Phase 3 as an honest stopgap for uploaded records with no `.atr` annotations, and it has never been evaluated with QRS-detection metrics — the one number ever recorded for it is "91.5% of the true annotated beat *count* on record 230," which is a count ratio, not sensitivity. It could be finding the wrong 91.5%.

2. **The UI physically cannot show a beat.** `GET /records/{id}/signal` downsamples the whole record to `SIGNAL_DOWNSAMPLE_TARGET_POINTS = 2000` (`api/main.py`). For a 30-minute record at 360 Hz that is `downsample_factor = 325` — one sample every ~0.9 s, against a beat period of ~0.8 s. Roughly **one point per beat**. No frontend work can render morphology from that; the endpoint has to change first.

There is also a subtler correctness angle worth stating, since it's the real end-to-end argument for this release: the classifier is *trained* on windows centered on annotation-provided R-peak indices, but at inference on an un-annotated upload it sees windows centered on `detect_r_peaks` output. Any systematic offset between the two is a train/inference distribution mismatch feeding the existing classifier. Better detection is therefore not only its own deliverable — it plausibly improves classification on exactly the inputs the web app actually serves. Whether it *does* is an empirical question this release should measure, not assert.

## Scope decisions

| Decision | Choice |
|---|---|
| Detector approach | Classical baseline **and** a proposed learned model, compared head-to-head |
| Classical baseline | Pan-Tompkins — the canonical QRS detector, so the comparison is against something defensible rather than against the current stopgap |
| Evaluation protocol | Peak-matching within a fixed tolerance → Se / PPV, on held-out patients. Exact tolerance and metric definitions to be pinned to the primary standard in Phase 1, not from memory |
| Split | DS1/DS2 inter-patient, same as v2.0 — consistency with the rest of the project wins over matching the all-48-records convention common in detector papers. Report both where cheap |
| UI | Both a real-resolution analysis window **and** a single-beat view |
| Beat markers | Only non-Normal beats get marked; Normal beats stay detected-but-unmarked |

**Honest risk, stated upfront:** Pan-Tompkins is a strong baseline and published QRS detectors sit very high on MIT-BIH. A learned detector may well *not* beat it. That is a legitimate result and will be reported as one — the project's standing rule is that an honest negative is correct, and this release must not turn into tuning-until-the-number-looks-good.

## Phases

Two independent threads again: Phase 2 (endpoint) + Phase 5 (UI) can proceed alongside Phases 3–4 (the model work), since only the endpoint couples them.

### v2.1 Phase 1 — Research & evaluation protocol
- [ ] Pin the QRS-detection evaluation standard to its primary source (matching tolerance, TP/FP/FN definitions, Se/PPV, and whether a detection error rate is worth reporting) — verified at the source, per project rule, not recalled
- [ ] Confirm the Pan-Tompkins algorithm's stages against the original paper rather than a blog reimplementation
- [ ] Decide the learned model's label formulation (sample-level mask vs. offset regression vs. peak heatmap) and write it down with the reasoning
- [ ] `docs/beat-detection-architecture.md`

### v2.1 Phase 2 — Windowed high-resolution signal endpoint
- [ ] `GET /records/{id}/signal` gains a time-window + resolution contract, keeping the current whole-record downsampled response as the default (no breaking change for the deployed frontend)
- [ ] Bounds validation and an explicit cap on returned points — a full-resolution 30-minute request is ~650k floats and is a trivially reachable memory/bandwidth DoS otherwise
- [ ] Revisit caching: the current per-record cached response can't cover arbitrary windows; cache the filtered signal and slice per request instead
- [ ] Tests, including the window-boundary and cap-enforcement cases

### v2.1 Phase 3 — Classical baseline + detection metrics harness
- [ ] `ecg_pipeline/qrs_detection.py` — Pan-Tompkins
- [ ] `training/evaluate_detection.py` — peak matching against ground-truth annotations, Se/PPV per record and aggregated
- [ ] Real DS2 numbers for both Pan-Tompkins **and** the incumbent `detect_r_peaks`, so the stopgap finally gets measured instead of assumed

### v2.1 Phase 4 — Proposed learned detector
- [ ] Dataset builder for the detection task (sliding windows + labels derived from annotation R-peak positions)
- [ ] Model + training, DS1 only, seeded
- [ ] DS2 evaluation through the same harness as Phase 3 — identical protocol, or the comparison is meaningless
- [ ] Head-to-head write-up, including a negative result if that's what the numbers say

### v2.1 Phase 5 — Frontend: beat-resolution viewing
- [ ] Analysis window consumes the windowed endpoint and renders real morphology
- [ ] Single-beat view at full resolution with prev/next navigation
- [ ] Normal beats detected but unmarked; only non-Normal beats carry a marker
- [ ] Stretch, only if the detector work makes it cheap: a detected-vs-annotated overlay, which is the most direct way to *show* a detection release rather than describe it

### v2.1 Phase 6 — Wire the winning detector into the API
- [ ] Replace the `detect_r_peaks` call in the `beat_source="detected"` path
- [ ] Surface which detector produced the beats, so an API consumer can tell
- [ ] Re-measure end-to-end classification on un-annotated uploads — does better detection actually move the classifier, or not?

## References

- Repo audit (2026-07-05, conversation record) — original bug list that motivated this rebuild: scaler double-fit bug, hardcoded machine-specific paths, non-contiguous label encoding, data leakage via random beat-level split, duplicated feature-extraction code between training and UI. Not yet written up as a standalone doc — worth doing if this project needs a self-contained record independent of chat history.
- de Chazal, O'Dwyer & Reilly, "Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features," IEEE Trans. Biomed. Eng., 2004
- Moody GB, Mark RG, "The impact of the MIT-BIH Arrhythmia Database," IEEE Eng in Med and Biol 20(3):45-50, 2001
