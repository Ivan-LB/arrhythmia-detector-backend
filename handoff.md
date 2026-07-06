# Handoff

Live state for picking this project back up cold. For the fixed roadmap see [docs/plan.md](docs/plan.md); for the full dated history/reasoning behind every decision see [docs/progress.md](docs/progress.md) — this file is deliberately short, that one is the deep record.

## TL;DR

Rebuilding a solo degree-project ECG arrhythmia classifier (`Ivan-LB/Arrhythmia-Detector`, now renamed **`Ivan-LB/arrhythmia-detector-backend`**) into a methodologically-correct, portfolio + academic-grade project. Original repo tagged `v1.0` and frozen as a snapshot. Rebuild happens on branch `v2.0.0`, one phase per branch, PR'd and reviewed before merging into `v2.0.0`. `main` stays untouched until the whole backend rebuild is done — **`v2.0.0` does not get merged to `main` yet.**

**Phases 0-5 are all done.** Phases 0-3 (plus CORS + rate-limiting follow-up fixes) are merged into `v2.0.0`. A real trained model exists and a working FastAPI service sits in front of it, verified against a real running server. **Phase 4** (React/Next.js frontend) is done in its own separate repo (`arrhythmia-detector-web`, polyrepo decision, see below) — upload flow, ECG trace + per-beat overlay, and the class-distribution/confidence summary view, all browser-verified against this real API. **Phase 5** (repo hygiene, this repo) is done on branch `phase-5-repo-hygiene`, PR not yet opened/merged as of this writing.

**Next real decision point:** once `phase-5-repo-hygiene` is reviewed and merged into `v2.0.0`, merging `v2.0.0` → `main` is next — but that's explicitly the user's call on timing/how, don't just do it.

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

Both remaining phases are done. What's left is procedural, not implementation work:

1. Open the PR for `phase-5-repo-hygiene` → `v2.0.0` (independent code review first, per the standing rule below).
2. Once the user merges it on GitHub: pull `v2.0.0`, delete the branch locally and on origin.
3. Ask the user whether/when to merge `v2.0.0` → `main` — their call, don't just do it.

Phase 5 left one thing deliberately unresolved rather than deciding unilaterally: `Data/RawData/`, `Data/ecg_features3.csv`, and the legacy `Models/*.h5`/`.pk1` binaries are still tracked in git even though nothing in the rebuilt pipeline (`ecg_pipeline/`, `training/`, `api/`) reads them — they're only used by the original pre-rebuild `UI/`/`ModelCreation/` PyQt app, which is being kept as-is until the new web frontend reaches parity. Untracking them would break a fresh clone's ability to run that old app. Worth a real decision once the old app is finally retired, not before.

The old PyQt `UI/` folder in this repo stays as-is until the new web app has visible functional parity — don't delete it preemptively.

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
