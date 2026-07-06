# Handoff

Live state for picking this project back up cold. For the fixed roadmap see [docs/plan.md](docs/plan.md); for the full dated history/reasoning behind every decision see [docs/progress.md](docs/progress.md) — this file is deliberately short, that one is the deep record.

## TL;DR

Rebuilding a solo degree-project ECG arrhythmia classifier (`Ivan-LB/Arrhythmia-Detector`, now renamed **`Ivan-LB/arrhythmia-detector-backend`**) into a methodologically-correct, portfolio + academic-grade project. Original repo tagged `v1.0` and frozen as a snapshot. Rebuild happens on branch `v2.0.0`, one phase per branch, PR'd and reviewed before merging into `v2.0.0`. `main` stays untouched until the whole backend rebuild is done — **`v2.0.0` does not get merged to `main` yet.**

**Phases 0-5 are all done and merged into `v2.0.0`.** Phases 0-3, the CORS + rate-limiting follow-up fixes, and Phase 5 (repo hygiene, PRs #4-#7) are all merged. A real trained model exists and a working FastAPI service sits in front of it, verified against a real running server. **Phase 4** (React/Next.js frontend) is done in its own separate repo (`arrhythmia-detector-web`, polyrepo decision, see below) — upload flow, ECG trace + per-beat overlay, and the class-distribution/confidence summary view, all browser-verified against this real API. The original pre-rebuild PyQt app (`UI/`, `ModelCreation/`, `Images/`, legacy `Models/*.h5`/`.pk1`, `Data/RawData/`, `Data/ecg_features3.csv`) has been retired now that the web frontend has real functional parity — see the `retire-legacy-ui` branch/PR.

**Next real decision point:** PR #8 (`v2.0.0` → `main`) is open — merging it retires `v1.0` as the active state. Timing/how is explicitly the user's call, don't just do it.

Phase 6 (SwiftUI macOS app) is future work, also its own repo, explicitly deferred.

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
- **Branch pattern**: `v2.0.0` is the long-lived dev branch. Each phase gets its own branch off `v2.0.0` (e.g. `phase-4-...`), pushed, PR'd against `v2.0.0`. The user reviews and merges on GitHub themselves — after they say "merged", pull `v2.0.0`, delete the now-merged branch locally and on origin, then start the next phase's branch.
- **TDD, every phase**: write the test, confirm it fails (RED) for the right reason, implement, confirm it passes (GREEN). Every phase so far has followed this and it's caught real bugs before they shipped — keep doing it.
- **Independent code review before every PR**: at minimum `python-reviewer`; add `security-reviewer` in parallel for anything touching file I/O, uploads, or user input (Phase 3's file-upload surface got both, and the security pass caught a real memory-exhaustion DoS). Fix CRITICAL/HIGH findings before opening the PR, not after.
- **Verify against real data/real running services, not just mocks.** Several real bugs (SMOTE fabricating data from 2 real examples, a missing-MLII crash, a caching gap that only showed up at real latency) were only caught by actually running things against real MIT-BIH records and a real `uvicorn` server — unit tests with synthetic fixtures alone missed all three.
- **Never fabricate data to reach a number** (user's standing hard rule) — this directly killed the original SMOTE-based class-imbalance approach; see the Phase 2 entry in `docs/progress.md` for the full story if this comes up again.

## What's next

All planned implementation work is done. What's left is procedural:

1. Review/merge the `retire-legacy-ui` PR (removes the now-retired original app and its residual artifacts, see below).
2. Review/merge PR #8 (`v2.0.0` → `main`) whenever ready — their call, don't just do it.
3. After each PR merges: pull the target branch, delete the merged branch locally and on origin.

**The original pre-rebuild PyQt app has been retired**, now that the web frontend has real functional parity (Phase 4 done). Removed: `UI/` (the PyQt app itself), `ModelCreation/` (old training scripts, superseded by `training/`), `Images/` (unreferenced legacy report plots), legacy `Models/*.h5`/`.pk1` binaries (the current model artifact, `models/beat-classifier-*`, is untouched — it's a different, gitignored path), `Data/RawData/` and `Data/ecg_features3.csv` (legacy derived data, unreferenced by the rebuilt pipeline). All of this remains permanently recoverable from the frozen `v1.0` tag if ever needed. Confirmed via repo-wide grep that nothing in `ecg_pipeline/`, `training/`, `api/`, or `tests/` referenced any of it before deleting.

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
