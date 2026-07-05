# Handoff

Live state for picking this project back up cold. For the fixed roadmap see [docs/plan.md](docs/plan.md); for the full dated history/reasoning behind every decision see [docs/progress.md](docs/progress.md) — this file is deliberately short, that one is the deep record.

## TL;DR

Rebuilding a solo degree-project ECG arrhythmia classifier (`Ivan-LB/Arrhythmia-Detector`, now renamed **`Ivan-LB/arrhythmia-detector-backend`**) into a methodologically-correct, portfolio + academic-grade project. Original repo tagged `v1.0` and frozen as a snapshot. Rebuild happens on branch `v2.0.0`, one phase per branch, PR'd and reviewed before merging into `v2.0.0`. `main` stays untouched until the whole backend rebuild is done — **`v2.0.0` does not get merged to `main` yet.**

**Phases 0-3 are done and merged into `v2.0.0`.** A real trained model exists and a working FastAPI service sits in front of it, verified against a real running server. Two independent threads of work remain, in no required order:
1. **Phase 5** (repo hygiene) — still in this repo, blocks merging `v2.0.0` → `main`.
2. **Phase 4** (React/Next.js frontend) — a **new, separate repo** (polyrepo decision, see below), doesn't block this repo at all.

Phase 6 (SwiftUI macOS app) is future work, also its own repo, explicitly deferred.

## Environment quickstart

```bash
cd /Users/ivanlorenzanabelli/Projects/Python/Arrhythmia-Detector
source .venv/bin/activate          # already set up, don't recreate
python -m pytest tests/ -q         # 151 tests, ~94% coverage, should be green
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

## What's next — two independent threads

### Thread A: Phase 5 (repo hygiene, this repo)
- [ ] `pyproject.toml` with pinned dependencies (currently loose `>=` bounds)
- [ ] Real `README.md` (setup, usage, dataset download step, links to `docs/`)
- [ ] `.gitignore` with proper globs (currently has some minimal entries added ad hoc per-phase, needs a real pass)
- [ ] Document the dataset/model-artifact download-or-regenerate step instead of relying on what's committed
- [ ] CI (test run on push — this is also a portfolio piece)
- [ ] Delete `ModelCreation/sineWave.py` (dead code from the original repo, superseded by `ecg_pipeline/features.py`'s Hann windowing)

Once this is done: merge `v2.0.0` → `main` in this repo (the user's call on timing/how — ask, don't just do it).

### Thread B: Phase 4 (frontend, new repo)
Not started. Needs, in order:
1. Decide/confirm the new repo's name (never discussed yet).
2. Scaffold a Next.js/TypeScript project there.
3. Build against the API contract in `docs/system-design.md` §4 (`/health`, `POST /records`, `GET /records/{id}/beats`, `GET /records/{id}/signal` — all implemented and stable).
4. Upload flow → ECG trace rendering + per-beat classification overlay → summary view (class distribution, confidence).

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
