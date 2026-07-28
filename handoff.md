# Handoff

Live state for picking this project back up cold. For the fixed roadmap see [docs/plan.md](docs/plan.md); for the full dated history/reasoning behind every decision see [docs/progress.md](docs/progress.md) — this file is deliberately short, that one is the deep record.

## TL;DR

Solo degree-project ECG arrhythmia classifier (`Ivan-LB/Arrhythmia-Detector`, now renamed **`Ivan-LB/arrhythmia-detector-backend`**), rebuilt into a methodologically-correct, portfolio + academic-grade project. `v1.0` marks the frozen pre-rebuild state; **`v2.0` is tagged on `main`** and marks the completed rebuild.

**Current work: v2.1 — beat detection.** Long-lived branch `v2.1` off `main`, one phase per branch, PR'd and reviewed before merging into `v2.1`, same pattern the v2.0 rebuild used.

**v2.0 is shipped.** Phases 0-5 all merged (PRs #1-#9), `v2.0.0` merged into `main`, tag `v2.0` pushed, all phase branches deleted. A real trained model exists, a working FastAPI service sits in front of it (verified against a real running server), a React/Next.js frontend in its own repo (`arrhythmia-detector-web`, polyrepo decision, see below) consumes it, and the original PyQt app plus all residual legacy artifacts have been retired.

**Now starting v2.1 — beat detection.** Full scope and phase checklist in [docs/plan.md](docs/plan.md) under the `v2.1` heading. The short version: `detect_r_peaks` is a `find_peaks` heuristic that has never been properly evaluated, and `/signal` downsamples to ~1 point per beat so the UI physically can't show beat morphology. v2.1 fixes both — a Pan-Tompkins baseline plus a proposed learned detector compared head-to-head on DS2, a windowed high-resolution signal endpoint, and a UI that shows real beats.

**Two user decisions already made for v2.1** (don't relitigate): (1) both a classical baseline *and* a proposed learned model, compared — not one or the other; (2) both a high-resolution analysis window *and* a single-beat view.

Phase 6 of v2.0 (SwiftUI macOS app) remains future work, its own repo, explicitly deferred.

## Environment quickstart

```bash
cd /Users/ivanlorenzanabelli/Projects/Python/Arrhythmia-Detector
source .venv/bin/activate          # already set up, don't recreate
python -m pytest tests/ -q         # 168 tests, ~94% coverage, should be green
MODEL_DIR="$(ls -d models/beat-classifier-*)" uvicorn api.main:app --reload   # run the real API
```

`MODEL_DIR` is required to run the API — no implicit "latest model" default, deliberately (see `docs/neural-network-architecture.md` §8). There's currently one trained model artifact: `models/beat-classifier-20260705-dc9f983/`.

## Standing workflow rules (don't relitigate these)

- **Git identity**: commit as the user (`Belli <ivanlorenzana@outlook.com>`, already the configured git user) — **never add a `Co-Authored-By` line or any AI attribution.**
- **Branch pattern**: a long-lived version branch (`v2.1` now; `v2.0.0` was the v2.0 one) with each phase on its own branch off it (e.g. `v2.1-phase-2-...`), pushed, PR'd against the version branch. The user reviews and merges on GitHub themselves — after they say "merged", pull the version branch, delete the now-merged branch locally and on origin, then start the next phase's branch. The version branch merges to `main` and gets tagged once the release is complete.
- **TDD, every phase**: write the test, confirm it fails (RED) for the right reason, implement, confirm it passes (GREEN). Every phase so far has followed this and it's caught real bugs before they shipped — keep doing it.
- **Independent code review before every PR**: at minimum `python-reviewer`; add `security-reviewer` in parallel for anything touching file I/O, uploads, or user input (Phase 3's file-upload surface got both, and the security pass caught a real memory-exhaustion DoS). Fix CRITICAL/HIGH findings before opening the PR, not after.
- **Verify against real data/real running services, not just mocks.** Several real bugs (SMOTE fabricating data from 2 real examples, a missing-MLII crash, a caching gap that only showed up at real latency) were only caught by actually running things against real MIT-BIH records and a real `uvicorn` server — unit tests with synthetic fixtures alone missed all three.
- **Never fabricate data to reach a number** (user's standing hard rule) — this directly killed the original SMOTE-based class-imbalance approach; see the Phase 2 entry in `docs/progress.md` for the full story if this comes up again.

## What's next

**v2.1 Phase 1 — research & evaluation protocol.** Nothing implemented yet; the plan is written and the `v2.1` branch exists. Start here:

1. Pin the QRS-detection evaluation standard to its **primary source** — the matching tolerance and the exact TP/FP/FN definitions. Do not write these from memory; this project has a standing rule about verifying at the source, and Phase 0 of v2.0 caught real errors by following it.
2. Confirm Pan-Tompkins' stages against the original paper, not a blog reimplementation.
3. Decide the learned detector's label formulation and write down the reasoning.
4. Produce `docs/beat-detection-architecture.md`, then move to Phase 2.

Phases 2 (windowed endpoint) and 5 (frontend) are an independent thread from Phases 3-4 (the model work) and can run alongside them — same two-thread shape v2.0 had with Phases 4 and 5.

**Watch for this trap in Phase 4:** Pan-Tompkins is a genuinely strong baseline. If the learned detector doesn't beat it, report that — do not tune until the number looks good. An honest negative is a real result and the standing rule against fabricating numbers applies to this release exactly as it did to the SMOTE decision in v2.0 Phase 2.

**Already done in v2.0, don't redo:** the original pre-rebuild PyQt app is retired — `UI/`, `ModelCreation/`, `Images/`, the legacy `Models/*.h5`/`.pk1` binaries, `Data/RawData/`, and `Data/ecg_features3.csv` are all gone (permanently recoverable from the frozen `v1.0` tag). The current model artifact `models/beat-classifier-*` is a different, gitignored path and was untouched.

## Facts worth not re-deriving

- **AAMI EC57 classes**: N/S/V/F/Q, confirmed against the canonical WFDB `ecgcodes.h` reference (`docs/data-pipeline-architecture.md` §6).
- **DS1/DS2 split**: the de Chazal et al. (2004) inter-patient protocol, exact record lists in `ecg_pipeline/splits.py` (single source of truth) and `docs/data-pipeline-architecture.md` §3.
- **Current model performance (DS2, honest inter-patient numbers)**: 63.22% accuracy. Per class: N Se=64.8%/PPV=95.1%, S Se=24.5%/PPV=13.1%, V Se=64.6%/PPV=21.8%, F Se=61.3%/PPV=3.6%, Q Se=0%/PPV=0% (only 2 training examples exist for Q — expected, not a bug). This is a first correct baseline, not a tuned final model — there's real headroom versus the published benchmarks researched in Phase 0, noted but not chased yet.
- **Class imbalance handling**: capped class weights, not SMOTE — SMOTE was tried first and found to fabricate ~38,000 synthetic rows from 2 real examples for the rarest class. Full story in `docs/progress.md`'s Phase 2 entry.
- Repo is public on GitHub already (confirmed via `gh repo view`).
- `.claude/` is excluded via `.git/info/exclude` (local-only), deliberately not via the shared `.gitignore` — this was an explicit user preference, don't "fix" it by adding `.claude` to `.gitignore`.

## Docs map

- [docs/plan.md](docs/plan.md) — the fixed roadmap, phase checklists
- [docs/progress.md](docs/progress.md) — dated log, the actual reasoning trail (read this for "why," not just "what")
- [docs/data-pipeline-architecture.md](docs/data-pipeline-architecture.md) — dataset/feature engineering
- [docs/neural-network-architecture.md](docs/neural-network-architecture.md) — model + training + real results
- [docs/system-design.md](docs/system-design.md) — overall architecture, API contract, repo split rationale
- [docs/portfolio-brief.md](docs/portfolio-brief.md) — canonical CV/portfolio source: verified numbers, the framing that makes this project worth showing, and explicit guardrails on what must never be claimed (e.g. never quote the old leaky ~98%). Written for another agent to consume; re-verify its numbers if the model is ever retrained.
